"""Road graph metrics derived from a one-pixel raster skeleton.

These scores use pixel coordinates. They are reproducible across masks evaluated
with the same protocol, but are not the published vector-GT benchmark scores.
"""

from collections import defaultdict
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def harmonic(a, b):
    return 2.0 * a * b / (a + b) if a + b else 0.0


def control_nodes(graph, limit):
    """Deterministic pixel-graph control points, prioritizing endpoints/junctions."""
    nodes = np.arange(graph.shape[0], dtype=np.int32)
    if len(nodes) <= limit:
        return nodes
    special = nodes[np.diff(graph.indptr) != 2]
    regular = nodes[np.diff(graph.indptr) == 2]
    if len(special) >= limit:
        return special[np.linspace(0, len(special) - 1, limit).astype(int)]
    remainder = limit - len(special)
    sampled = regular[np.linspace(0, len(regular) - 1, remainder).astype(int)]
    return np.sort(np.concatenate((special, sampled)))


def prepare_graph(points, graph, max_controls, controls=None, lengths=None):
    if controls is None:
        controls = control_nodes(graph, max_controls)
    if lengths is None:
        lengths = (np.atleast_2d(dijkstra(graph, directed=False, indices=controls))
                   if len(controls) else np.empty((0, 0)))
    return {"points": points, "graph": graph, "controls": controls, "lengths": lengths}


def _project_to_edges(source_points, target, radius):
    """Find the closest point on a pixel-graph edge, with deterministic ties."""
    points, graph = target["points"], target["graph"]
    if not len(points):
        return [None] * len(source_points)
    tree = cKDTree(points)
    results = []
    for location in source_points:
        nearest_distance, _ = tree.query(location)
        if nearest_distance > radius + math.sqrt(2):
            results.append(None)
            continue
        candidates = tree.query_ball_point(location, min(radius, nearest_distance) + math.sqrt(2) + 1e-9)
        best = None
        checked = set()
        for node in candidates:
            for neighbor in graph.indices[graph.indptr[node]:graph.indptr[node + 1]]:
                edge = (min(int(node), int(neighbor)), max(int(node), int(neighbor)))
                if edge in checked:
                    continue
                checked.add(edge)
                first, second = points[list(edge)].astype(np.float64)
                delta = second - first
                fraction = float(np.clip(np.dot(location - first, delta) / np.dot(delta, delta), 0, 1))
                projection = first + fraction * delta
                distance = float(np.linalg.norm(location - projection))
                choice = (distance, edge[0], edge[1], fraction)
                if best is None or choice < best:
                    best = choice
        if best is not None and best[0] <= radius + 1e-9:
            results.append((best[1], best[2], best[3]))
        elif nearest_distance <= radius + 1e-9:
            # A one-pixel component has no edge to project onto.
            nearest = min(tree.query_ball_point(location, nearest_distance + 1e-9))
            results.append((nearest, nearest, 0.0))
        else:
            results.append(None)
    return results


def _insert_control_points(graph, projections):
    """Split each touched edge at all matched controls before path search."""
    added = defaultdict(list)
    already_inserted = {}
    matched = np.full(len(projections), -1, dtype=np.int32)
    next_node = graph.shape[0]
    for index, projection in enumerate(projections):
        if projection is None:
            continue
        first, second, fraction = projection
        if first == second or fraction <= 1e-9:
            matched[index] = first
        elif fraction >= 1 - 1e-9:
            matched[index] = second
        else:
            key = (first, second, round(fraction, 9))
            if key not in already_inserted:
                already_inserted[key] = next_node
                added[(first, second)].append((fraction, next_node))
                next_node += 1
            matched[index] = already_inserted[key]
    if not added:
        return graph, matched
    upper = coo_matrix(graph)
    rows, cols, weights = [], [], []

    def connect(first, second, length):
        rows.extend((first, second))
        cols.extend((second, first))
        weights.extend((length, length))

    for first, second, length in zip(upper.row, upper.col, upper.data):
        if first >= second:
            continue
        controls = added.get((int(first), int(second)))
        if not controls:
            connect(int(first), int(second), float(length))
            continue
        chain = [(0.0, int(first)), *sorted(controls), (1.0, int(second))]
        for (left_t, left), (right_t, right) in zip(chain, chain[1:]):
            if right_t > left_t:
                connect(left, right, (right_t - left_t) * float(length))
    expanded = coo_matrix((weights, (rows, cols)), shape=(next_node, next_node)).tocsr()
    return expanded, matched


def directional_apls(reference, proposal, radius):
    """Sampled raster-graph APLS with controls projected and inserted on edges."""
    controls = reference["controls"]
    if len(controls) < 2:
        return {"score": 0.0, "pairs": 0, "unmatched": 0, "disconnected": 0}
    projections = _project_to_edges(reference["points"][controls], proposal, radius)
    augmented, matched = _insert_control_points(proposal["graph"], projections)
    unique = np.unique(matched[matched >= 0])
    paths = (np.atleast_2d(dijkstra(augmented, directed=False, indices=unique))
             if len(unique) else None)
    index = {int(node): i for i, node in enumerate(unique)}
    score_sum = pairs = unmatched = disconnected = 0
    for i, origin in enumerate(controls):
        for j in range(i + 1, len(controls)):
            distance = reference["lengths"][i, controls[j]]
            if not np.isfinite(distance) or distance <= 0:
                continue
            pairs += 1
            if matched[i] < 0 or matched[j] < 0:
                unmatched += 1
                continue
            proposed = paths[index[int(matched[i])], matched[j]]
            if not np.isfinite(proposed):
                disconnected += 1
                continue
            score_sum += max(0.0, 1.0 - abs(proposed - distance) / distance)
    return {"score": score_sum / pairs if pairs else 0.0, "pairs": pairs,
            "unmatched": unmatched, "disconnected": disconnected}


def _local_samples(prepared, seed, radius, interval):
    graph = prepared["graph"]
    distances = dijkstra(graph, directed=False, indices=int(seed), limit=radius)
    reached = np.flatnonzero(np.isfinite(distances))
    if not len(reached):
        return np.empty((0, 2))
    points = prepared["points"][reached]
    # One representative per spatial cell approximates regular marble/hole
    # placement without giving dense junction pixels extra weight.
    cells = np.floor_divide(points, interval)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(keep)]


def _local_match(reference, proposal, tolerance):
    if not len(reference):
        return 0.0
    if not len(proposal):
        return 0.0
    tree = cKDTree(proposal)
    candidates = []
    for left, location in enumerate(reference):
        for right in tree.query_ball_point(location, tolerance + 1e-9):
            distance = float(np.linalg.norm(location - proposal[right]))
            candidates.append((distance, left, right))
    used_left, used_right = set(), set()
    for _, left, right in sorted(candidates):
        if left not in used_left and right not in used_right:
            used_left.add(left)
            used_right.add(right)
    return len(used_left) / len(reference)


def _directional_topo(reference, proposal, radius, interval, tolerance, max_seeds):
    seeds = control_nodes(reference["graph"], max_seeds)
    if not len(seeds):
        return 0.0, 0
    if not len(proposal["points"]):
        return 0.0, len(seeds)
    tree = cKDTree(proposal["points"])
    scores = []
    for seed in seeds:
        point = reference["points"][seed]
        local_ref = _local_samples(reference, seed, radius, interval)
        distance, nearest = tree.query(point)
        if distance > tolerance:
            scores.append(0.0)
            continue
        local_prop = _local_samples(proposal, nearest, radius, interval)
        scores.append(_local_match(local_ref, local_prop, tolerance))
    return float(np.mean(scores)), len(seeds)


def graph_scores(gt, pred, protocol):
    radius = protocol["apls_snap_radius"]
    forward = directional_apls(gt, pred, radius)
    reverse = directional_apls(pred, gt, radius)
    topo_args = (protocol["topo_local_radius_px"], protocol["topo_sample_interval_px"],
                 protocol["topo_match_radius_px"], protocol["topo_max_seeds"])
    topo_recall, gt_seeds = _directional_topo(gt, pred, *topo_args)
    topo_precision, pred_seeds = _directional_topo(pred, gt, *topo_args)
    total = forward["pairs"]
    return {
        "topo_precision": topo_precision,
        "topo_recall": topo_recall,
        "topo_f1": harmonic(topo_precision, topo_recall),
        "topo_gt_seeds": gt_seeds, "topo_pred_seeds": pred_seeds,
        "apls_gt_to_pred": forward["score"],
        "apls_pred_to_gt": reverse["score"],
        "apls_bidirectional": harmonic(forward["score"], reverse["score"]),
        "apls_gt_pairs": total, "apls_pred_pairs": reverse["pairs"],
        "tlts_infeasible": ((forward["unmatched"] + forward["disconnected"]) / total
                            if total else 0.0),
        "tlts_unmatched_pairs": forward["unmatched"],
        "tlts_disconnected_pairs": forward["disconnected"],
        "tlts_infeasible_pairs": forward["unmatched"] + forward["disconnected"],
        "tlts_matched_pairs": total - forward["unmatched"],
        "tlts_disconnected_matched_rate": (
            forward["disconnected"] / (total - forward["unmatched"])
            if total > forward["unmatched"] else 0.0),
    }
