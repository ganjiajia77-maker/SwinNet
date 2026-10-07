"""Shared binary-mask metrics. APLS is a raster proxy, not official APLS."""

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


METRIC_VERSION = "road-mask-v1"
DEFAULT_PROTOCOL = {
    "metric_version": METRIC_VERSION,
    "image_size": 1024,
    "short_area_threshold": 20,
    "apls_max_nodes": 64,
    "apls_snap_radius": 5.0,
    "connectivity": 8,
    "skeleton": "numpy_zhang_suen_zero_padded",
    "snap_rule": "distance <= radius; ties use lowest row-major node",
    "segmentation_aggregation": "global TP/FP/FN",
    "topology_aggregation": "per-image mean; bidirectional APLS harmonic per image",
    "postprocessing": "none; use supplied binary masks without resizing",
}


def ratio(a, b):
    return float(a) / float(b) if b else 0.0


def harmonic(a, b):
    return ratio(2 * a * b, a + b)


def skeletonize(mask):
    # Always use the same boundary convention, independent of opencv-contrib.
    image = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            center = image[1:-1, 1:-1]
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum((a == 0) & (b == 1) for a, b in
                              zip(neighbors, neighbors[1:] + neighbors[:1]))
            remove = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if phase == 0:
                remove &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if remove.any():
                changed = True
                center[remove] = 0
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


def component_stats(mask, area_threshold):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        "count": count - 1,
        "short": int((areas < area_threshold).sum()),
        "largest_ratio": ratio(areas.max(), areas.sum()) if areas.size else 0.0,
        "max_area": int(areas.max()) if areas.size else 0,
    }


def skeleton_graph(skeleton):
    points = np.argwhere(skeleton)
    height, width = skeleton.shape
    indices = np.full(skeleton.shape, -1, dtype=np.int32)
    indices[skeleton] = np.arange(len(points), dtype=np.int32)
    rows, columns, weights = [], [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        first = indices[y0:y1, x0:x1].ravel()
        second = indices[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        valid = (first >= 0) & (second >= 0)
        source, target = first[valid], second[valid]
        rows.extend((source, target))
        columns.extend((target, source))
        edge_weights = np.full(len(source), np.hypot(dy, dx))
        weights.extend((edge_weights, edge_weights))
    graph = csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(points), len(points)),
    )
    return points, graph


def sample_graph_nodes(graph, max_nodes):
    nodes = np.arange(graph.shape[0])
    if len(nodes) <= max_nodes:
        return nodes
    special = nodes[np.diff(graph.indptr) != 2]
    if len(special) >= max_nodes:
        return special[np.linspace(0, len(special) - 1, max_nodes).astype(int)]
    regular = nodes[np.diff(graph.indptr) == 2]
    return np.concatenate((special, regular[np.linspace(
        0, len(regular) - 1, max_nodes - len(special),
    ).astype(int)]))


def prepare_paths(skeleton, max_nodes):
    points, graph = skeleton_graph(skeleton)
    nodes = sample_graph_nodes(graph, max_nodes)
    paths = (np.atleast_2d(dijkstra(graph, directed=False, indices=nodes))
             if len(nodes) else np.empty((0, 0)))
    return {"points": points, "graph": graph, "nodes": nodes, "paths": paths}


def snap_nodes(source, target, radius):
    snapped = np.full(len(source), -1, dtype=np.int64)
    if not len(target):
        return snapped
    tree = cKDTree(target)
    distances, _ = tree.query(source)
    for i, distance in enumerate(distances):
        if distance > radius:
            continue
        candidates = np.asarray(tree.query_ball_point(source[i], distance + 1e-9))
        squared = ((target[candidates] - source[i]) ** 2).sum(axis=1)
        snapped[i] = candidates[squared == squared.min()].min()
    return snapped


def directional_apls(reference, proposal, radius):
    nodes = reference["nodes"]
    if len(nodes) < 2:
        return 0.0, 0
    snapped = snap_nodes(reference["points"][nodes], proposal["points"], radius)
    matched = np.unique(snapped[snapped >= 0])
    proposed_paths = (np.atleast_2d(dijkstra(
        proposal["graph"], directed=False, indices=matched,
    )) if len(matched) else None)
    source_rows = {int(node): i for i, node in enumerate(matched)}
    scores = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            length = reference["paths"][i, nodes[j]]
            if not np.isfinite(length) or length <= 0:
                continue
            score = 0.0
            if snapped[i] >= 0 and snapped[j] >= 0:
                proposed = proposed_paths[source_rows[int(snapped[i])], snapped[j]]
                if np.isfinite(proposed):
                    score = max(0.0, 1.0 - abs(proposed - length) / length)
            scores.append(score)
    return (float(np.mean(scores)) if scores else 0.0), len(scores)


def prepare_target(target, protocol):
    skeleton = skeletonize(target)
    return {
        "mask": target, "skeleton": skeleton,
        "components": component_stats(target, protocol["short_area_threshold"]),
        "paths": prepare_paths(skeleton, protocol["apls_max_nodes"]),
    }


def confusion(pred, target):
    return {"tp": int((pred & target).sum()), "fp": int((pred & ~target).sum()),
            "fn": int((~pred & target).sum())}


def segmentation(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {"iou": ratio(tp, tp + fp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}


def image_metrics(pred, prepared, protocol):
    target, gt_skeleton = prepared["mask"], prepared["skeleton"]
    pred_skeleton = skeletonize(pred)
    pred_stats = component_stats(pred, protocol["short_area_threshold"])
    gt_stats = prepared["components"]
    missing = gt_skeleton & ~pred
    gap_stats = component_stats(missing, 2)
    topo_precision = ratio((pred_skeleton & target).sum(), pred_skeleton.sum())
    topo_recall = ratio((gt_skeleton & pred).sum(), gt_skeleton.sum())
    pred_paths = prepare_paths(pred_skeleton, protocol["apls_max_nodes"])
    forward, forward_pairs = directional_apls(
        prepared["paths"], pred_paths, protocol["apls_snap_radius"],
    )
    reverse, reverse_pairs = directional_apls(
        pred_paths, prepared["paths"], protocol["apls_snap_radius"],
    )
    row = confusion(pred, target)
    row.update({
        "topo_precision": topo_precision, "topo_recall": topo_recall,
        "cldice": harmonic(topo_precision, topo_recall),
        "pred_components": pred_stats["count"], "gt_components": gt_stats["count"],
        "frag_idx": ratio(pred_stats["count"], max(gt_stats["count"], 1)),
        "fragment_density_per_1000_px": ratio(1000 * pred_stats["count"], pred.sum()),
        "short_pred_components": pred_stats["short"], "short_gt_components": gt_stats["short"],
        "extra_components": max(pred_stats["count"] - gt_stats["count"], 0),
        "largest_component_ratio": pred_stats["largest_ratio"],
        "gt_largest_component_ratio": gt_stats["largest_ratio"],
        "break_pixels": int(missing.sum()), "break_rate": ratio(missing.sum(), gt_skeleton.sum()),
        "gap_components": gap_stats["count"], "max_gap_pixels": gap_stats["max_area"],
        "apls_gt_to_pred_approx": forward, "apls_pred_to_gt_approx": reverse,
        "apls_bidirectional_approx": harmonic(forward, reverse),
        "apls_gt_valid_pairs": forward_pairs, "apls_pred_valid_pairs": reverse_pairs,
    })
    return row


def summarize(rows):
    counts = {k: sum(row[k] for row in rows) for k in ("tp", "fp", "fn")}
    result = {"images": len(rows), **counts, **segmentation(counts)}
    for key in rows[0]:
        if key not in {"case_id", "tp", "fp", "fn"}:
            result[key] = float(np.mean([row[key] for row in rows]))
    result["topo_f1"] = result["cldice"]
    return result
