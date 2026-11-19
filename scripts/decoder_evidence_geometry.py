"""Native-pixel road and path evidence for a paired decoder replay."""

from collections import defaultdict
from contextlib import redirect_stdout
import math
from unittest.mock import patch

import cv2
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def components(mask):
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[1]


def distance_to(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def groups(labels):
    points = np.argwhere(labels)
    ids = labels[tuple(points.T)]
    order = np.argsort(ids, kind="stable")
    points, ids = points[order], ids[order]
    return (part for part in np.split(points, np.flatnonzero(np.diff(ids)) + 1) if len(part))


def bounds(points, shape, pad):
    top, left = np.maximum(points.min(axis=0) - pad, 0)
    bottom, right = np.minimum(points.max(axis=0) + pad + 1, shape)
    return slice(top, bottom), slice(left, right)


def road_geometry(gt, skeleton, width_cutoff, junction_radius, occluded):
    degree = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - skeleton
    # Diagonal neighbors can be redundant at a corner. Count incident runs.
    padded = np.pad(skeleton, 1)
    h, w = skeleton.shape
    neighbors = [padded[y:y+h, x:x+w] for y, x in
                 ((0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0))]
    runs = sum(left & ~right for left, right in zip(neighbors, neighbors[1:] + neighbors[:1]))
    junction = skeleton & (degree >= 3) & (runs >= 3)
    near_junction = distance_to(junction) <= junction_radius if junction.any() else junction
    width = 2 * distance_to(~gt)
    kind = np.zeros(skeleton.shape, np.uint8)
    kind[skeleton] = 1
    kind[skeleton & (width <= width_cutoff)] = 2
    kind[skeleton & near_junction] = 3
    kind[skeleton & occluded] = 4
    return kind, width


COHORTS = {1: "ordinary", 2: "narrow", 3: "junction", 4: "occluded_marked"}


def nearby_component_ids(labels, point, radius):
    y, x = map(int, point)
    ys, xs = bounds(np.array([[y, x]]), labels.shape, radius)
    yy, xx = np.ogrid[ys.start:ys.stop, xs.start:xs.stop]
    values = labels[ys, xs][(yy-y)**2 + (xx-x)**2 <= radius**2]
    return set(values[values > 0].tolist())


def road_rows(case, kind, width, masks, radius, minimum):
    rows = []
    for value, name in COHORTS.items():
        for index, pts in enumerate(groups(components(kind == value)), 1):
            if len(pts) < minimum:
                continue
            ys, xs = bounds(pts, kind.shape, max(16, 2 * radius))
            offset = np.array([ys.start, xs.start])
            region = np.zeros(kind[ys, xs].shape, bool)
            region[tuple((pts-offset).T)] = True
            degrees = cv2.filter2D(region.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - region
            ends = np.argwhere(region & (degrees == 1))
            on, off = masks["on"], masks["off"]
            restored = int((on[tuple(pts.T)] & ~off[tuple(pts.T)]).sum())
            lost = int((off[tuple(pts.T)] & ~on[tuple(pts.T)]).sum())
            row = dict(case_id=case, segment_id=f"{value}-{index}", cohort=name,
                       gt_skeleton_px=len(pts), median_width_px=float(np.median(width[tuple(pts.T)])),
                       restored_px=restored, lost_px=lost, endpoints=len(ends))
            for mode, pred in masks.items():
                row[f"{mode}_covered_px"] = int(pred[tuple(pts.T)].sum())
                connected = None
                if len(ends) == 2:
                    labels = components(pred[ys, xs])
                    connected = bool(nearby_component_ids(labels, ends[0], radius)
                                     & nearby_component_ids(labels, ends[1], radius))
                row[f"{mode}_local_connected"] = connected
            rows.append(row)
    return rows


def error_rows(case, gt, pred_skeletons, tolerance, minimum):
    """One common spatial inventory, then classify each model's support."""
    distance = distance_to(gt)
    outside = {name: skel & (distance > tolerance) for name, skel in pred_skeletons.items()}
    inventory = np.logical_or.reduce(list(outside.values()))
    rows = []
    for index, pts in enumerate(groups(components(inventory)), 1):
        if len(pts) < minimum:
            continue
        ys, xs = bounds(pts, gt.shape, max(24, tolerance * 4))
        local = pts - [ys.start, xs.start]
        truth = gt[ys, xs]
        gt_labels = components(truth)
        row = dict(case_id=case, event_id=f"fp-{index}", y=int(pts[0, 0]), x=int(pts[0, 1]),
                   top=ys.start, bottom=ys.stop, left=xs.start, right=xs.stop)
        for mode, pred in pred_skeletons.items():
            own = outside[mode][tuple(pts.T)]
            count = int(own.sum())
            row[f"{mode}_unmatched_px"] = count
            category = "absent"
            if count >= minimum:
                region = np.zeros(truth.shape, bool)
                region[tuple(local[own].T)] = True
                nearby = distance_to(region) <= max(12, tolerance * 3)
                skel_labels = components(pred[ys, xs] & nearby)
                touches = set()
                bridge = False
                for part in np.unique(skel_labels[region]):
                    if not part:
                        continue
                    contacts = set(np.unique(gt_labels[skel_labels == part])) - {0}
                    touches.update(contacts)
                    bridge |= len(contacts) >= 2
                if bridge:
                    category = "bridge_candidate"
                elif touches:
                    category = "spur_or_offset_candidate"
                elif np.median(distance[tuple(pts[own].T)]) <= 2 * tolerance:
                    category = "offset_candidate"
                else:
                    category = "isolated_candidate"
            row[f"{mode}_category"] = category
        rows.append(row)
    return rows


def reference_paths(points, graph, seed, controls, per_bin):
    """Sample connected GT paths without looking at either prediction."""
    rng = np.random.default_rng(seed)
    nodes = np.sort(rng.choice(len(points), min(len(points), controls), replace=False))
    if len(nodes) < 2:
        return nodes, np.empty((0, len(points))), []
    distances, previous = dijkstra(graph, indices=nodes, return_predecessors=True)
    bins = defaultdict(list)
    for i, source in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            length = float(distances[i, nodes[j]])
            if math.isfinite(length) and length >= 32:
                bins["32-127" if length < 128 else "128-255" if length < 256 else "256+"].append((i, j, length))
    paths = []
    for name, candidates in sorted(bins.items()):
        for pick in sorted(rng.choice(len(candidates), min(len(candidates), per_bin), replace=False)):
            i, j, length = candidates[pick]
            path = [int(nodes[j])]
            while path[-1] != nodes[i]:
                path.append(int(previous[i, path[-1]]))
            paths.append((name, i, j, length, np.array(path[::-1])))
    return nodes, distances, paths


def predicted_lengths(points, graph, endpoints, tolerance):
    if not len(points):
        return np.full(len(endpoints), -1), np.empty((0, 0)), {}
    distances, ids = cKDTree(points).query(endpoints)
    ids = np.where(distances <= tolerance, ids, -1)
    active = np.unique(ids[ids >= 0])
    lengths = np.atleast_2d(dijkstra(graph, indices=active)) if len(active) else np.empty((0, 0))
    return ids, lengths, {int(node): i for i, node in enumerate(active)}


def anchor_support(records, points, endpoints, distances, tolerance):
    if not records:
        return []
    tree = cKDTree(points)
    windows = []
    for record in records:
        top, left, grid_h, grid_w, candidates, selected = record
        origin = np.array([top, left])
        inside = np.all((endpoints >= origin) & (endpoints < origin + 512), axis=1)
        flags = []
        for coords in (candidates, selected):
            native = origin + (coords + .5) * (512 / np.array([grid_h, grid_w])) - .5
            nearest_distance, nearest = tree.query(native)
            good = (nearest_distance <= tolerance)[None, :] & (distances[:, nearest] <= tolerance)
            flags.append(good & inside[:, None])
        windows.append((inside, *flags))
    return windows


def path_rows(case, points, graph, pred_graphs, masks, delta, anchors, protocol, changed_routes):
    seed = protocol["seed"] ^ int.from_bytes(case.encode(), "little") % (2**32)
    nodes, dist, paths = reference_paths(points, graph, seed, protocol["controls"], protocol["paths_per_bin"])
    if not paths:
        return []
    endpoints = points[nodes]
    tolerance = protocol["endpoint_tolerance"]
    predictions = {mode: predicted_lengths(*geometry, endpoints, tolerance)
                   for mode, geometry in pred_graphs.items()}
    support = anchor_support(anchors, points, endpoints, dist, protocol["anchor_tolerance"])
    rows = []
    for index, (length_bin, i, j, gt_length, path) in enumerate(paths, 1):
        start, end = endpoints[[i, j]]
        yy, xx = points[path].T
        missing = ~masks["off"][yy, xx]
        joint, candidates, selected = 0, 0, 0
        for inside, cand, fps in support:
            joint += int(inside[i] and inside[j])
            a, b = np.flatnonzero(cand[i]), np.flatnonzero(cand[j])
            candidates += int(bool(len(a) and len(b) and (len(a) > 1 or len(b) > 1 or a[0] != b[0])))
            a, b = np.flatnonzero(fps[i]), np.flatnonzero(fps[j])
            selected += int(bool(len(a) and len(b) and (len(a) > 1 or len(b) > 1 or a[0] != b[0])))
        row = dict(case_id=case, path_id=f"path-{index}", length_bin=length_bin, gt_length_px=gt_length,
                   start_y=int(start[0]), start_x=int(start[1]), end_y=int(end[0]), end_x=int(end[1]),
                   joint_windows=joint, candidate_pair_windows=candidates, selected_pair_windows=selected,
                   gt_path_logit_delta=float(delta[yy, xx].mean()), off_missing_gt_path_px=int(missing.sum()),
                   missing_gt_path_logit_delta=float(delta[yy[missing], xx[missing]].mean()) if missing.any() else None)
        for mode, (ids, lengths, lookup) in predictions.items():
            a, b = ids[[i, j]]
            status, length = "endpoint_unmatched", None
            if a >= 0 and b >= 0:
                if a == b:
                    status = "endpoints_collapsed"
                else:
                    distance = lengths[lookup[a], b]
                    status = "reachable" if np.isfinite(distance) else "disconnected"
                    if status == "reachable":
                        length = float(distance)
            row[f"{mode}_status"] = status
            row[f"{mode}_length_px"] = length
            row[f"{mode}_length_relative_error"] = (length - gt_length) / gt_length if length is not None else None
        if changed_routes is not None and len(changed_routes) < 12 and row["on_status"] != row["off_status"]:
            changed_routes[row["path_id"]] = points[path]
        rows.append(row)
    return rows


def official_topo_rows(case, gt_graph, pred_graph, cohort, protocol, official, mode):
    """Record the existing evaluator's seed scores and precision matching."""
    topo = official.sat_topo
    make_seeds, compare = topo.TOPOGenerateStartingPoints, topo.TOPOWithPairs
    observed = {"seeds": [], "retained": [], "precision": {}}

    def record_seeds(*args, **kwargs):
        observed["seeds"] = make_seeds(*args, **kwargs)
        return observed["seeds"]

    def record_compare(proposal, ground, pairs, seeds, *args, **kwargs):
        walk = proposal.TOPOWalk
        matcher = topo.HopcroftKarp
        indices = iter(pairs)
        state = {}

        def record_walk(*walk_args, **walk_kwargs):
            state["seed"] = next(indices)
            state["marbles"] = walk(*walk_args, **walk_kwargs)
            return state["marbles"]

        class MatchingRecorder(matcher):
            def maximum_matching(self, *match_args, **match_kwargs):
                matching = super().maximum_matching(*match_args, **match_kwargs)
                index = state["seed"]
                if index not in observed["precision"]:
                    # In the pinned TOPOWithPairs the first matching after
                    # each predicted walk is the precision assignment.
                    observed["precision"][index] = [(point, point in matching) for point in state["marbles"]]
                return matching

        with patch.object(proposal, "TOPOWalk", record_walk), patch.object(topo, "HopcroftKarp", MatchingRecorder):
            observed["retained"] = compare(proposal, ground, pairs, seeds, *args, **kwargs)
        return observed["retained"]

    with patch.object(topo, "TOPOGenerateStartingPoints", record_seeds), patch.object(
            topo, "TOPOWithPairs", record_compare):
        p, r, count = official._sat_topo_score(gt_graph, pred_graph, protocol)
    retained = {tuple(row[:2]): row for row in observed["retained"]}
    gt_points = gt_graph["points"]
    tree = cKDTree(gt_points) if len(gt_points) else None
    rows, precision_points = [], []
    for index, seed in enumerate(observed["seeds"]):
        lat, lon = seed[:2]
        y, x = 111111 * (41 - lat), 111111 * math.cos(math.radians(41)) * (lon + 71)
        nearest_distance, node = tree.query([y, x])
        match = retained.get((lat, lon))
        rows.append(dict(case_id=case, mode=mode, seed_id=index, y=float(y), x=float(x),
                         cohort=COHORTS.get(cohort[tuple(gt_points[node])], "unclassified"),
                         gt_snap_px=float(nearest_distance), retained=match is not None,
                         local_precision=float(match[2]) if match is not None else None,
                         recall_contribution=float(match[3]) if match is not None else 0.0))
        points = observed["precision"].get(index, [])
        if match is not None and points and not math.isclose(sum(matched for _, matched in points)/len(points), match[2]):
            raise RuntimeError("TOPO precision trace does not match the official seed score")
        for point, matched in points:
            py = 111111 * (41 - point[0])
            px = 111111 * math.cos(math.radians(41)) * (point[1] + 71)
            precision_points.append(dict(case_id=case, mode=mode, seed_id=index,
                y=float(py), x=float(px), matched=matched, seed_retained=match is not None))
    return rows, precision_points, dict(case_id=case, mode=mode, topo_precision=p, topo_recall=r,
                      seeds=count, retained=len(observed["retained"]))
