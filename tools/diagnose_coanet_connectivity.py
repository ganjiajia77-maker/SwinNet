#!/usr/bin/env python3
"""Topology diagnostics for binary road masks exported by a CoANet evaluator.

This deliberately consumes prediction masks instead of importing a particular
CoANet fork, whose model/test APIs and checkpoint formats vary between repos.
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute clDice, topology P/R/F1, fragmentation, gap and raster-APLS metrics."
    )
    parser.add_argument("--pred_dir", required=True, help="Directory containing CoANet binary predictions")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground-truth road masks")
    parser.add_argument("--output_dir", required=True, help="Directory for CSV summaries")
    parser.add_argument("--threshold", type=int, default=127, help="Foreground threshold for 8-bit masks")
    parser.add_argument("--centerline_tolerance", type=float, default=1.5,
                        help="Uncovered centerline distance tolerance in pixels")
    parser.add_argument("--apls_snap_tolerance", type=int, default=3, help="Maximum GT-to-prediction graph snap distance")
    parser.add_argument("--max_apls_nodes", type=int, default=12, help="Maximum sampled GT graph nodes per image")
    parser.add_argument("--min_component_length", type=int, default=5,
                        help="Minimum skeleton component pixels counted in components/img")
    parser.add_argument("--min_gap_length", type=int, default=5,
                        help="Minimum uncovered centerline pixels counted as a gap")
    return parser.parse_args()


def normalized_stem(path):
    stem = Path(path).stem
    suffixes = (
        "_surface_pred", "_mask_pred", "_prediction", "_pred", "_surface", "_mask",
        "_sat", "_image", "_img",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break
    return stem


def index_images(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    indexed = {}
    for path in sorted(files):
        key = normalized_stem(path)
        if key in indexed:
            raise RuntimeError(f"Ambiguous image id {key!r}: {indexed[key]} and {path}")
        indexed[key] = path
    return indexed


def read_mask(path, threshold):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return image > threshold


def skeletonize(mask):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 127
    # Zhang-Suen thinning fallback, implemented with vectorized neighborhood operations.
    image = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p = np.pad(image, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum(((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1)) for i in range(8))
            keep = (image == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if phase == 0:
                keep &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                keep &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = keep
            if remove.any():
                image[remove] = 0
                changed = True
    return image.astype(bool)


def topology_pr(pred_skel, gt_skel, pred_mask, gt_mask):
    precision = float((pred_skel & gt_mask).sum()) / float(pred_skel.sum() + 1e-12)
    recall = float((gt_skel & pred_mask).sum()) / float(gt_skel.sum() + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def skeleton_component_metrics(skeleton, min_length):
    count, _, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64) if count > 1 else np.empty((0,), dtype=np.int64)
    qualified = areas[areas >= min_length]
    qualified_pixels = int(qualified.sum())
    return {
        "components": int(qualified.size),
        "short_components": int((areas < min_length).sum()),
        "short_component_density_per_mpix": float((areas < min_length).sum()) * 1e6 / skeleton.size,
        "largest_share": float(qualified.max()) / qualified_pixels if qualified_pixels else 0.0,
        "qualified_pixels": qualified_pixels,
    }


def raster_graph(skeleton):
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return coords, None
    h, w = skeleton.shape
    index = np.full((h, w), -1, dtype=np.int32)
    index[coords[:, 0], coords[:, 1]] = np.arange(len(coords), dtype=np.int32)
    rows, cols, weights = [], [], []
    for dy, dx, weight in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5)):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        a = index[y0:y1, x0:x1]
        b = index[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        valid = (a >= 0) & (b >= 0)
        ia, ib = a[valid], b[valid]
        rows.extend((ia, ib))
        cols.extend((ib, ia))
        weights.extend((np.full(len(ia), weight), np.full(len(ia), weight)))
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    data = np.concatenate(weights) if weights else np.empty(0, dtype=np.float32)
    graph = sparse.csr_matrix((data, (row, col)), shape=(len(coords), len(coords)))
    return coords, graph


def raster_apls(gt_skel, pred_skel, snap_tolerance, max_nodes):
    """Approximate APLS via sampled shortest-path length preservation on pixel graphs."""
    gt_coords, gt_graph = raster_graph(gt_skel)
    pred_coords, pred_graph = raster_graph(pred_skel)
    if gt_graph is None or pred_graph is None:
        return (1.0 if gt_graph is None and pred_graph is None else 0.0), 0
    if len(gt_coords) < 2 or len(pred_coords) < 2:
        return 0.0, 0

    # Sample across the GT skeleton's raster order, then snap each sample to prediction.
    sample_ids = np.unique(np.linspace(0, len(gt_coords) - 1, min(max_nodes, len(gt_coords)), dtype=int))
    samples = gt_coords[sample_ids]
    pred_distance, pred_nearest = ndimage.distance_transform_edt(~pred_skel, return_indices=True)
    mapped = []
    kept_gt_ids = []
    pred_index = np.full(pred_skel.shape, -1, dtype=np.int32)
    pred_index[pred_coords[:, 0], pred_coords[:, 1]] = np.arange(len(pred_coords), dtype=np.int32)
    gt_index = np.full(gt_skel.shape, -1, dtype=np.int32)
    gt_index[gt_coords[:, 0], gt_coords[:, 1]] = np.arange(len(gt_coords), dtype=np.int32)
    for gt_id, (y, x) in zip(sample_ids, samples):
        py, px = int(pred_nearest[0, y, x]), int(pred_nearest[1, y, x])
        if pred_distance[y, x] <= snap_tolerance:
            mapped.append(int(pred_index[py, px]))
            kept_gt_ids.append(int(gt_id))
    if len(mapped) < 2:
        return 0.0, 0

    scores = []
    for source_idx in range(len(mapped) - 1):
        gt_dist = dijkstra(gt_graph, directed=False, indices=kept_gt_ids[source_idx])
        pred_dist = dijkstra(pred_graph, directed=False, indices=mapped[source_idx])
        for target_idx in range(source_idx + 1, len(mapped)):
            base = gt_dist[kept_gt_ids[target_idx]]
            test = pred_dist[mapped[target_idx]]
            if np.isfinite(base) and base > 0:
                score = 0.0 if not np.isfinite(test) else max(0.0, 1.0 - abs(float(test) - float(base)) / float(base))
                scores.append(score)
    return (float(np.mean(scores)) if scores else 0.0), len(scores)


def evaluate_one(pred_path, gt_path, args):
    pred = read_mask(pred_path, args.threshold)
    gt = read_mask(gt_path, args.threshold)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    pred_skel, gt_skel = skeletonize(pred), skeletonize(gt)
    tprec, trecall, tf1 = topology_pr(pred_skel, gt_skel, pred, gt)
    pred_comp = skeleton_component_metrics(pred_skel, args.min_component_length)
    gt_comp = skeleton_component_metrics(gt_skel, args.min_component_length)
    if pred.any():
        distance_to_pred = ndimage.distance_transform_edt(~pred)
    else:
        distance_to_pred = np.full(pred.shape, np.inf, dtype=np.float32)
    gap_mask = gt_skel & (distance_to_pred > args.centerline_tolerance)
    gap_count, _, gap_stats, _ = cv2.connectedComponentsWithStats(gap_mask.astype(np.uint8), connectivity=8)
    gap_areas = gap_stats[1:, cv2.CC_STAT_AREA] if gap_count > 1 else np.empty((0,), dtype=np.int32)
    valid_gaps = gap_areas[gap_areas >= args.min_gap_length]
    apls, apls_pairs = raster_apls(gt_skel, pred_skel, args.apls_snap_tolerance, args.max_apls_nodes)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p_skel_gt = float((pred_skel & gt).sum()) / (float(pred_skel.sum()) + 1e-12)
    r_skel_pred = float((gt_skel & pred).sum()) / (float(gt_skel.sum()) + 1e-12)
    cldice = 2 * p_skel_gt * r_skel_pred / (p_skel_gt + r_skel_pred + 1e-12)
    return {
        "image": Path(pred_path).name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pixel_precision": tp / (tp + fp + 1e-12),
        "pixel_recall": tp / (tp + fn + 1e-12),
        "pixel_f1": 2 * tp / (2 * tp + fp + fn + 1e-12),
        "cldice": cldice,
        "topo_precision": tprec,
        "topo_recall": trecall,
        "topo_f1": tf1,
        "components_img": pred_comp["components"],
        "delta_beta0": abs(pred_comp["components"] - gt_comp["components"]),
        "uncovered_centerline": float(gap_mask.sum()) / float(gt_skel.sum() + 1e-12),
        "gaps_img": int(valid_gaps.size),
        "mean_gap_px": float(valid_gaps.mean()) if valid_gaps.size else 0.0,
        "short_components_img": pred_comp["short_components"],
        "fragment_density_per_mpix": pred_comp["short_component_density_per_mpix"],
        "largest_share": pred_comp["largest_share"],
        "raster_apls": apls,
        "apls_path_pairs": apls_pairs,
    }


def main():
    args = parse_args()
    pred_files, gt_files = index_images(args.pred_dir), index_images(args.gt_dir)
    missing = sorted(set(pred_files) - set(gt_files))
    extra = sorted(set(gt_files) - set(pred_files))
    if missing:
        raise RuntimeError(f"Missing GT masks for {len(missing)} prediction(s), e.g. {missing[:5]}")
    if not pred_files:
        root = Path(args.pred_dir)
        sample_entries = [str(p.relative_to(root)) for p in list(root.iterdir())[:12]]
        raise RuntimeError(
            f"No supported image predictions found under {args.pred_dir} (searched recursively). "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTS))}. "
            f"Top-level entries: {sample_entries or '[empty directory]'}"
        )
    if extra:
        raise RuntimeError(
            f"Only {len(pred_files)}/{len(gt_files)} GT images have predictions; "
            f"missing predictions for {len(extra)} image(s), e.g. {extra[:5]}. "
            "The prediction run likely stopped early; finish inference before reporting metrics."
        )
    rows = [evaluate_one(pred_files[key], gt_files[key], args) for key in sorted(pred_files)]
    os.makedirs(args.output_dir, exist_ok=True)
    per_image = os.path.join(args.output_dir, "coanet_connectivity_per_image.csv")
    summary = os.path.join(args.output_dir, "coanet_connectivity_summary.csv")
    fields = list(rows[0])
    with open(per_image, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_fields = [k for k in fields if k not in ("image", "tp", "fp", "fn")]
    avg = {"images": len(rows)}
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    avg["pixel_precision_micro"] = tp / (tp + fp + 1e-12)
    avg["pixel_recall_micro"] = tp / (tp + fn + 1e-12)
    avg["pixelF1"] = 2 * tp / (2 * tp + fp + fn + 1e-12)
    avg.update({key: float(np.mean([float(row[key]) for row in rows])) for key in metric_fields
                if key not in ("pixel_precision", "pixel_recall", "pixel_f1")})
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg))
        writer.writeheader()
        writer.writerow(avg)
    print(f"Evaluated all {len(rows)} matched images")
    print(f"pixelF1={avg['pixelF1']:.4f}")
    print(f"clDice={avg['cldice']:.4f}")
    print(f"topoP={avg['topo_precision']:.4f}")
    print(f"topoR={avg['topo_recall']:.4f}")
    print(f"topoF1={2 * avg['topo_precision'] * avg['topo_recall'] / (avg['topo_precision'] + avg['topo_recall'] + 1e-12):.4f}")
    print(f"components/img={avg['components_img']:.2f}")
    print(f"|Δbeta0|={avg['delta_beta0']:.2f}")
    print(f"uncovered_centerline={avg['uncovered_centerline']:.4f}")
    print(f"gaps/img={avg['gaps_img']:.2f}")
    print(f"mean_gap_px={avg['mean_gap_px']:.2f}")
    print(f"largest_share={avg['largest_share']:.4f}")
    print(f"fragment_density_per_mpix={avg['fragment_density_per_mpix']:.2f}")
    print(f"raster_apls_approx={avg['raster_apls']:.4f}")
    print(f"Per-image CSV: {per_image}")
    print(f"Summary CSV: {summary}")
    print("Note: raster_apls is a pixel-graph approximation; report the definition with results.")


if __name__ == "__main__":
    main()
