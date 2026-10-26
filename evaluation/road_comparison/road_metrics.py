"""Shared mask metrics; graph scores call pinned official TOPO and APLS code."""

import cv2
import numpy as np
import time
from scipy.sparse import csr_matrix

from official_graph_metrics import graph_scores, prepare_official_graph


METRIC_VERSION = "road-mask-v5.0-official-sparse"
DEFAULT_PROTOCOL = {
    "metric_version": METRIC_VERSION,
    "image_size": 1024,
    "short_area_threshold": 20,
    "legacy_apls_max_nodes_ignored": 64,
    "apls_snap_radius": 5.0,
    "apls_control_spacing_px": 100.0,
    "apls_min_path_px": 10.0,
    "tlts_n_paths": 200,
    "topo_seed_density_deg": 0.00100,
    "topo_step_deg": 0.00010,
    "tlts_max_node_dist_px": 25.0,
    "tlts_length_tolerance": 0.05,
    "tlts_seed": 1234,
    "connectivity": 8,
    "skeleton": "numpy_zhang_suen_zero_padded",
    "snap_rule": "official CosmiQ nearest-edge projection and control-point insertion",
    "segmentation_aggregation": "global TP/FP/FN",
    "topology_aggregation": "per-image mean of Sat2Graph TOPO, CosmiQ APLS and metrics_delin TLTS outputs",
    "graph_protocol": "raster skeleton to 8-neighbor graph; exact degree-two chain contraction for APLS",
    "graph_units": "mask pixels mapped to synthetic metres for Sat2Graph; not vector-GT benchmark scores",
    "topo_source": "Sat2Graph 7ca4a345a2ceae4c62c120661608a60e9e962be4",
    "apls_source": "CosmiQ/apls 4a78807cafde62b773e3a1190177526853c61207",
    "tlts_source": "lcit/metrics_delin 30f1ad9ccc901e63770f39a80b0e1ec6bbfb34d9",
    "topo_settings": "Sat2Graph functions: configurable seed density and walk step; threshold 0.00010, radius 0.00150 or 0.00300 degrees",
    "tlts_protocol": "authors' toolong_tooshort: independent random GT paths; 25-pixel endpoint match; 5-percent length tolerance",
    "tlts_seed_rule": "sha256(GT skeleton points + base seed), first four bytes; same GT paths for all models",
    "postprocessing": "none; use supplied binary masks without resizing",
    "gt_encoding": "single-channel or identical-channel RGB binary mask; predictions single-channel",
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


def prepare_target(target, protocol):
    skeleton = skeletonize(target)
    points, pixel_graph = skeleton_graph(skeleton)
    return {
        "mask": target, "skeleton": skeleton,
        "components": component_stats(target, protocol["short_area_threshold"]),
        "graph": prepare_official_graph(points, pixel_graph),
    }


def confusion(pred, target):
    return {"tp": int((pred & target).sum()), "fp": int((pred & ~target).sum()),
            "fn": int((~pred & target).sum())}


def segmentation(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {"iou": ratio(tp, tp + fp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}


def image_metrics(pred, prepared, protocol, timing_label=None):
    started = time.perf_counter()
    target, gt_skeleton = prepared["mask"], prepared["skeleton"]
    pred_skeleton = skeletonize(pred)
    pred_stats = component_stats(pred, protocol["short_area_threshold"])
    gt_stats = prepared["components"]
    missing = gt_skeleton & ~pred
    gap_stats = component_stats(missing, 2)
    topo_precision = ratio((pred_skeleton & target).sum(), pred_skeleton.sum())
    topo_recall = ratio((gt_skeleton & pred).sum(), gt_skeleton.sum())
    cldice_seconds = time.perf_counter() - started
    if timing_label:
        print(f"[TIMING] {timing_label} clDice_mask={cldice_seconds:.1f}s", flush=True)
    graph_started = time.perf_counter()
    pred_points, pred_pixel_graph = skeleton_graph(pred_skeleton)
    pred_graph = prepare_official_graph(pred_points, pred_pixel_graph, id_offset=1_000_000_000)
    preparation_seconds = time.perf_counter() - graph_started
    if timing_label:
        print(f"[TIMING] {timing_label} pred_graph={preparation_seconds:.1f}s", flush=True)
    graph_result = graph_scores(prepared["graph"], pred_graph, protocol,
                                timing_label=timing_label)
    row = confusion(pred, target)
    row.update({
        "cldice_precision": topo_precision, "cldice_recall": topo_recall,
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
    })
    row.update(graph_result)
    row["timing_cldice_mask_seconds"] = cldice_seconds
    row["timing_pred_graph_seconds"] = preparation_seconds
    row["timing_total_seconds"] = time.perf_counter() - started
    return row


def summarize(rows):
    counts = {k: sum(row[k] for row in rows) for k in ("tp", "fp", "fn")}
    result = {"images": len(rows), **counts, **segmentation(counts)}
    for key in rows[0]:
        if key not in {"case_id", "tp", "fp", "fn"}:
            result[key] = float(np.mean([row[key] for row in rows]))
    return result
