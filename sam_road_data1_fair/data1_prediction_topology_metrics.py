"""Raster connectivity metrics matching the data1 Swin/CoANet comparison.

APLS here is sampled from pixel skeleton graphs; it is not the official
vector-road-network APLS computed from graph annotations.
"""

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def zhang_suen_skeletonize(mask):
    image = mask.astype(np.uint8).copy()
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
            if np.any(remove):
                changed = True
                center[remove] = 0
        if not changed:
            return image.astype(bool)


def skeletonize(mask):
    if (hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning")
            and hasattr(cv2.ximgproc, "THINNING_ZHANGSUEN")):
        return cv2.ximgproc.thinning(
            mask.astype(np.uint8) * 255,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        ) > 127
    return zhang_suen_skeletonize(mask)


def component_stats(mask, short_area_threshold):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int64)
    total = float(areas.sum())
    return {
        "components": float(max(count - 1, 0)),
        "short_components": float((areas < short_area_threshold).sum()) if areas.size else 0.0,
        "largest_ratio": float(areas.max()) / (total + 1e-8) if areas.size else 0.0,
        "max_area": float(areas.max()) if areas.size else 0.0,
    }


def skeleton_graph(skeleton):
    points = np.argwhere(skeleton)
    height, width = skeleton.shape
    indices = np.full((height, width), -1, dtype=np.int32)
    indices[skeleton] = np.arange(len(points), dtype=np.int32)
    rows, columns, weights = [], [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        first = indices[y0:y1, x0:x1].ravel()
        second = indices[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        valid = (first >= 0) & (second >= 0)
        source, target = first[valid], second[valid]
        weight = float(np.hypot(dy, dx))
        rows.extend((source, target))
        columns.extend((target, source))
        weights.extend((np.full(len(source), weight), np.full(len(source), weight)))
    if rows:
        graph = csr_matrix(
            (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
            shape=(len(points), len(points)),
        )
    else:
        graph = csr_matrix((len(points), len(points)))
    return points, graph


def sample_graph_nodes(graph, max_nodes):
    nodes = np.arange(graph.shape[0])
    if len(nodes) <= max_nodes:
        return nodes
    special = nodes[np.diff(graph.indptr) != 2]
    if len(special) >= max_nodes:
        indices = np.linspace(0, len(special) - 1, max_nodes).astype(int)
        return special[indices]
    remaining = max_nodes - len(special)
    regular = nodes[np.diff(graph.indptr) == 2]
    indices = np.linspace(0, len(regular) - 1, remaining).astype(int)
    return np.concatenate((special, regular[indices]))


def sampled_raster_apls(gt_skeleton, pred_skeleton, max_nodes, snap_radius):
    gt_points, gt_graph = skeleton_graph(gt_skeleton)
    pred_points, pred_graph = skeleton_graph(pred_skeleton)
    if len(gt_points) < 2 or len(pred_points) < 2:
        return 0.0
    gt_nodes = sample_graph_nodes(gt_graph, max_nodes)
    pred_points = pred_points.astype(np.float32)
    snapped = []
    for node in gt_nodes:
        delta = pred_points - gt_points[node].astype(np.float32)
        distance = np.sqrt((delta * delta).sum(axis=1))
        index = int(distance.argmin())
        snapped.append(index if distance[index] <= snap_radius else None)
    gt_paths = np.atleast_2d(dijkstra(gt_graph, directed=False, indices=gt_nodes))
    pred_sources = list(dict.fromkeys(node for node in snapped if node is not None))
    pred_paths = (np.atleast_2d(dijkstra(pred_graph, directed=False, indices=pred_sources))
                  if pred_sources else None)
    pred_source_rows = {node: row for row, node in enumerate(pred_sources)}
    scores = []
    for i in range(len(gt_nodes)):
        for j in range(i + 1, len(gt_nodes)):
            gt_distance = gt_paths[i, gt_nodes[j]]
            if not np.isfinite(gt_distance) or gt_distance <= 0:
                continue
            pred_source, pred_target = snapped[i], snapped[j]
            pred_distance = None
            if pred_source is not None and pred_target is not None:
                pred_distance = pred_paths[pred_source_rows[pred_source], pred_target]
            if pred_distance is None or not np.isfinite(pred_distance):
                scores.append(0.0)
            else:
                scores.append(max(0.0, 1.0 - abs(pred_distance - gt_distance) / gt_distance))
    return float(np.mean(scores)) if scores else 0.0


def image_metrics(prediction, target, short_area_threshold=20,
                  apls_max_nodes=64, apls_snap_radius=5.0):
    """Full-resolution binary-mask metrics using the common data1 definitions."""
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    pred_stats = component_stats(prediction, short_area_threshold)
    gt_stats = component_stats(target, short_area_threshold)
    pred_skeleton = skeletonize(prediction)
    gt_skeleton = skeletonize(target)
    pred_count = float(pred_skeleton.sum())
    gt_count = float(gt_skeleton.sum())
    topo_precision = float((pred_skeleton & target).sum()) / (pred_count + 1e-8)
    topo_recall = float((gt_skeleton & prediction).sum()) / (gt_count + 1e-8)
    cldice = 2.0 * topo_precision * topo_recall / (topo_precision + topo_recall + 1e-8)
    missing = gt_skeleton & ~prediction
    gap_stats = component_stats(missing, 2)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "iou": tp / max(tp + fp + fn, 1),
        "pred_components": pred_stats["components"],
        "gt_components": gt_stats["components"],
        "frag_idx": pred_stats["components"] / max(gt_stats["components"], 1.0),
        "extra_components": max(pred_stats["components"] - gt_stats["components"], 0.0),
        "short_pred_components": pred_stats["short_components"],
        "short_gt_components": gt_stats["short_components"],
        "largest_component_ratio": pred_stats["largest_ratio"],
        "gt_largest_component_ratio": gt_stats["largest_ratio"],
        "fragment_density_per_1000_px": 1000.0 * pred_stats["components"] /
                                        (float(prediction.sum()) + 1e-8),
        "topo_precision": topo_precision,
        "topo_recall": topo_recall,
        "cldice": cldice,
        "break_pixels": float(missing.sum()),
        "break_rate": float(missing.sum()) / (gt_count + 1e-8),
        "gap_components": gap_stats["components"],
        "max_gap_pixels": gap_stats["max_area"],
        "apls": sampled_raster_apls(
            gt_skeleton, pred_skeleton, apls_max_nodes, apls_snap_radius,
        ) if apls_max_nodes > 0 else None,
    }


def summarize(rows, short_area_threshold, apls_max_nodes, apls_snap_radius):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    summary = {
        "images": len(rows),
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "precision": precision,
        "recall": recall,
        "short_area_threshold": short_area_threshold,
        "apls_max_nodes": apls_max_nodes,
        "apls_snap_radius": apls_snap_radius,
        "apls_kind": "sampled raster-skeleton proxy; not official vector-graph APLS",
    }
    if rows:
        for key in rows[0]:
            if key in ("image_id", "tp", "fp", "fn", "iou"):
                continue
            values = [row[key] for row in rows if row[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
    return summary
