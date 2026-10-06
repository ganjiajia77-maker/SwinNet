"""Evaluate topology of saved full-resolution road prediction masks."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from datasets.dataset_road_skeleton import RoadSkeletonDataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_files(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prediction_directory(path):
    if (path / "surface").is_dir():
        return path / "surface"
    if image_files(path):
        return path
    candidates = sorted(item for item in path.glob("*/surface") if item.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Expected one surface prediction directory under {path}; found {len(candidates)}. "
        "Pass the exact surface directory when several test runs exist."
    )


def split_image_directory(root, split):
    nested = root / split / "image"
    return nested if nested.is_dir() else root / split


def split_label_directory(root, split):
    for path in (root / split / "mask", root / split / "label", root / f"{split}_labels"):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Cannot find ground-truth masks for {split} under {root}")


def prediction_case(path):
    stem = path.stem
    for suffix in ("_surface_pred", "_mask_pred", "_pred"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    raise ValueError(f"Prediction filename must end with _pred: {path.name}")


def label_path(label_dir, case):
    bases = (case, case.replace("_sat", "_mask"), case.replace("_image", "_mask"),
             case.replace("_img", "_mask"))
    for base in dict.fromkeys(bases):
        for extension in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
            path = label_dir / f"{base}{extension}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No ground-truth mask found for {case} in {label_dir}")


def component_stats(mask, short_area_threshold):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    foreground = int(areas.sum())
    return {
        "components": int(count - 1),
        "short_components": int((areas < short_area_threshold).sum()),
        "largest_ratio": float(areas.max() / foreground) if foreground else 0.0,
        "largest_area": int(areas.max()) if areas.size else 0,
    }


def skeletonize(mask):
    return RoadSkeletonDataset._skeletonize_binary(mask.astype(np.uint8) * 255) > 127


def skeleton_graph(skeleton):
    """Create an undirected, distance-weighted 8-neighbor pixel graph."""
    points = np.argwhere(skeleton)
    node_count = len(points)
    if node_count == 0:
        return csr_matrix((0, 0), dtype=np.float64), points
    height, width = skeleton.shape
    node_ids = np.full((height, width), -1, dtype=np.int32)
    node_ids[points[:, 0], points[:, 1]] = np.arange(node_count, dtype=np.int32)
    row_parts = []
    col_parts = []
    weight_parts = []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y_start, y_stop = max(0, -dy), height - max(0, dy)
        x_start, x_stop = max(0, -dx), width - max(0, dx)
        source = node_ids[y_start:y_stop, x_start:x_stop]
        target = node_ids[y_start + dy:y_stop + dy, x_start + dx:x_stop + dx]
        connected = (source >= 0) & (target >= 0)
        rows, cols = source[connected], target[connected]
        row_parts.extend((rows, cols))
        col_parts.extend((cols, rows))
        weight_parts.extend((np.full(len(rows), np.hypot(dy, dx)),) * 2)
    if row_parts:
        graph = csr_matrix(
            (np.concatenate(weight_parts),
             (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(node_count, node_count),
        )
    else:
        graph = csr_matrix((node_count, node_count), dtype=np.float64)
    return graph, points


def sample_graph_nodes(graph, max_nodes):
    """Prefer endpoints and junctions, then sample along ordinary paths."""
    node_count = graph.shape[0]
    if node_count <= max_nodes:
        return np.arange(node_count, dtype=np.int32)
    special = np.flatnonzero(np.diff(graph.indptr) != 2)
    if len(special) >= max_nodes:
        return special[np.linspace(0, len(special) - 1, max_nodes, dtype=int)]
    ordinary = np.flatnonzero(np.diff(graph.indptr) == 2)
    remaining = max_nodes - len(special)
    selected = ordinary[np.linspace(0, len(ordinary) - 1, remaining, dtype=int)]
    return np.concatenate((special, selected))


def directional_path_similarity(source_graph, source_points, target_graph, target_points,
                                max_nodes, snap_radius):
    if source_graph.shape[0] < 2 or target_graph.shape[0] < 2:
        return 0.0
    control_nodes = sample_graph_nodes(source_graph, max_nodes)
    distances, nearest = cKDTree(target_points).query(
        source_points[control_nodes], distance_upper_bound=snap_radius
    )
    snapped = np.where(np.isfinite(distances), nearest, -1).astype(np.int32)
    source_paths = dijkstra(source_graph, directed=False, indices=control_nodes)
    source_paths = source_paths[:, control_nodes]
    matched = np.unique(snapped[snapped >= 0])
    target_paths = None
    matched_rows = {}
    if matched.size:
        target_paths = dijkstra(target_graph, directed=False, indices=matched)
        matched_rows = {int(node): row for row, node in enumerate(matched)}

    scores = []
    for left in range(len(control_nodes)):
        for right in range(left + 1, len(control_nodes)):
            reference_length = source_paths[left, right]
            if not np.isfinite(reference_length) or reference_length <= 0:
                continue
            route_score = 0.0
            if snapped[left] >= 0 and snapped[right] >= 0:
                proposed_length = target_paths[matched_rows[int(snapped[left])], snapped[right]]
                if np.isfinite(proposed_length):
                    route_score = max(
                        0.0,
                        1.0 - abs(proposed_length - reference_length) / reference_length,
                    )
            scores.append(route_score)
    return float(np.mean(scores)) if scores else 0.0


def approximate_apls(gt_skeleton, pred_skeleton, max_nodes, snap_radius):
    gt_graph, gt_points = skeleton_graph(gt_skeleton)
    pred_graph, pred_points = skeleton_graph(pred_skeleton)
    gt_to_pred = directional_path_similarity(
        gt_graph, gt_points, pred_graph, pred_points, max_nodes, snap_radius
    )
    pred_to_gt = directional_path_similarity(
        pred_graph, pred_points, gt_graph, gt_points, max_nodes, snap_radius
    )
    symmetric = (
        2 * gt_to_pred * pred_to_gt / (gt_to_pred + pred_to_gt)
        if gt_to_pred + pred_to_gt else 0.0
    )
    return gt_to_pred, pred_to_gt, symmetric


def case_metrics(case, prediction, ground_truth, short_area_threshold,
                 apls_max_nodes, apls_snap_radius):
    pred = prediction > 127
    gt = ground_truth > 127
    if pred.shape != gt.shape:
        raise ValueError(f"{case}: prediction {pred.shape} and label {gt.shape} differ")

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pred_stats = component_stats(pred, short_area_threshold)
    gt_stats = component_stats(gt, short_area_threshold)
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)
    pred_skel_count = int(pred_skel.sum())
    gt_skel_count = int(gt_skel.sum())
    topo_precision = (
        int((pred_skel & gt).sum()) / pred_skel_count if pred_skel_count else 0.0
    )
    topo_recall = int((gt_skel & pred).sum()) / gt_skel_count if gt_skel_count else 0.0
    if not pred_skel_count and not gt_skel_count:
        cldice = 1.0
    else:
        cldice = (
            2 * topo_precision * topo_recall / (topo_precision + topo_recall)
            if topo_precision + topo_recall else 0.0
        )
    missing = gt_skel & ~pred
    gap_stats = component_stats(missing, 1)
    pred_skel_components = component_stats(pred_skel, 1)["components"]
    gt_skel_components = component_stats(gt_skel, 1)["components"]
    apls_gt_to_pred, apls_pred_to_gt, apls_approx = approximate_apls(
        gt_skel, pred_skel, apls_max_nodes, apls_snap_radius
    )

    return {
        "case": case,
        "height": int(pred.shape[0]),
        "width": int(pred.shape[1]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall": recall,
        "cldice": cldice,
        "topo_precision": topo_precision,
        "topo_recall": topo_recall,
        "pred_components": pred_stats["components"],
        "gt_components": gt_stats["components"],
        "fragment_index": pred_stats["components"] / max(gt_stats["components"], 1),
        "extra_pred_components": max(pred_stats["components"] - gt_stats["components"], 0),
        "short_pred_components": pred_stats["short_components"],
        "short_gt_components": gt_stats["short_components"],
        "pred_largest_ratio": pred_stats["largest_ratio"],
        "gt_largest_ratio": gt_stats["largest_ratio"],
        "pred_skeleton_components": pred_skel_components,
        "gt_skeleton_components": gt_skel_components,
        "extra_skeleton_components": max(pred_skel_components - gt_skel_components, 0),
        "missing_gt_skeleton_pixels": int(missing.sum()),
        "missing_gt_skeleton_rate": float(missing.sum() / gt_skel_count) if gt_skel_count else 0.0,
        "gap_components": gap_stats["components"],
        "max_gap_pixels": gap_stats["largest_area"],
        "apls_gt_to_pred_approx": apls_gt_to_pred,
        "apls_pred_to_gt_approx": apls_pred_to_gt,
        "apls_approx": apls_approx,
    }


def summarize(rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary = {
        "images": len(rows),
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall": recall,
    }
    for key in rows[0]:
        if key not in {"case", "height", "width", "tp", "fp", "fn", "iou", "f1", "precision", "recall"}:
            summary[key] = float(np.mean([row[key] for row in rows]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--short_area_threshold", type=int, default=20)
    parser.add_argument("--apls_max_nodes", type=int, default=64)
    parser.add_argument("--apls_snap_radius", type=float, default=5.0)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.short_area_threshold < 1:
        parser.error("--short_area_threshold must be positive")
    if args.apls_max_nodes < 2 or args.apls_snap_radius <= 0:
        parser.error("--apls_max_nodes must be at least 2 and --apls_snap_radius must be positive")

    pred_dir = prediction_directory(args.pred_dir)
    source_dir = split_image_directory(args.root_path, args.split)
    label_dir = split_label_directory(args.root_path, args.split)
    expected_cases = {path.stem for path in image_files(source_dir)}
    pred_files = image_files(pred_dir)
    if not expected_cases or not pred_files:
        raise RuntimeError("Source images or prediction masks are empty")
    prediction_cases = [prediction_case(path) for path in pred_files]
    if len(prediction_cases) != len(set(prediction_cases)):
        raise ValueError("Duplicate prediction case IDs found")
    if set(prediction_cases) != expected_cases:
        missing = sorted(expected_cases - set(prediction_cases))
        extra = sorted(set(prediction_cases) - expected_cases)
        raise ValueError(f"Prediction set does not match {args.split}: missing={missing[:8]}, extra={extra[:8]}")

    rows = []
    for index, (path, case) in enumerate(zip(pred_files, prediction_cases), 1):
        pred = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt_file = label_path(label_dir, case)
        gt = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            raise OSError(f"Cannot read prediction or label for {case}")
        rows.append(case_metrics(
            case, pred, gt, args.short_area_threshold,
            args.apls_max_nodes, args.apls_snap_radius,
        ))
        print(f"[{index}/{len(pred_files)}] {case}: IoU={rows[-1]['iou']:.4f} "
              f"clDice={rows[-1]['cldice']:.4f} "
              f"APLS~={rows[-1]['apls_approx']:.4f} "
              f"pred_comp={rows[-1]['pred_components']}", flush=True)

    summary = summarize(rows)
    summary.update({
        "apls_method": "bidirectional sampled 8-neighbor pixel-skeleton paths; harmonic mean",
        "apls_max_nodes": args.apls_max_nodes,
        "apls_snap_radius_pixels": args.apls_snap_radius,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_dir / 'summary.json'} and {args.output_dir / 'per_image.csv'}")


if __name__ == "__main__":
    main()
