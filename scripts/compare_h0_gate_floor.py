"""Paired frozen-checkpoint H0 gate diagnostic on native validation images."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "road_comparison"))
from road_metrics import DEFAULT_PROTOCOL, image_metrics, prepare_target, skeletonize, summarize


def index_logits(parent):
    files = list(Path(parent).glob("*/logits/*_logit.npy"))
    if not files:
        raise FileNotFoundError(f"No stitched logits below {parent}")
    result = {}
    for path in files:
        key = path.stem.removesuffix("_logit")
        if key in result:
            raise ValueError(f"Duplicate logit for {key}: {result[key]}, {path}")
        result[key] = path
    return result


def index_ground_truth(root, split, keys):
    root = Path(root)
    directories = [root / split / "mask", root / split / "label", root / f"{split}_labels"]
    directory = next((d for d in directories if d.is_dir()), None)
    if directory is None:
        raise FileNotFoundError(f"No {split} labels under {root}")
    result = {}
    for key in keys:
        candidates = [key, key.replace("_sat", "_mask")]
        paths = [directory / f"{stem}{ext}" for stem in candidates
                 for ext in (".png", ".tif", ".tiff", ".bmp", ".jpg")]
        found = [p for p in paths if p.is_file()]
        if len(found) != 1:
            raise ValueError(f"Expected one GT mask for {key}, got {found}")
        result[key] = found[0]
    return result


def index_images(root, split):
    directory = Path(root) / split / "image"
    if not directory.is_dir():
        directory = Path(root) / split
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return {path.stem for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes}


def read_gt(path, size):
    raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.shape[:2] != (size, size):
        raise ValueError(f"GT missing or wrong shape: {path}")
    if raw.ndim == 3:
        if raw.shape[2] != 3 or not (np.array_equal(raw[:, :, 0], raw[:, :, 1])
                                     and np.array_equal(raw[:, :, 0], raw[:, :, 2])):
            raise ValueError(f"Non-binary color GT: {path}")
        raw = raw[:, :, 0]
    values = np.unique(raw)
    if not np.isin(values, (0, 1, 255)).all() or (1 in values and 255 in values):
        raise ValueError(f"Unexpected GT values {values[:10]}: {path}")
    return raw > 0


def read_prediction(logit_path, key, size):
    path = logit_path.parent.parent / "surface" / f"{key}_pred.png"
    raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.shape != (size, size) or not np.isin(np.unique(raw), (0, 255)).all():
        raise ValueError(f"Missing, wrong shape, or nonbinary predicted mask: {path}")
    return raw > 0


def nearby_ids(anchor, pred, labels, kernel):
    nearby = cv2.dilate(anchor.astype(np.uint8), kernel).astype(bool) & pred
    return set(int(i) for i in np.unique(labels[nearby]) if i > 0)


def verified_gaps(gt_skeleton, baseline, tolerance, margin, min_pixels):
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    covered = cv2.dilate(baseline.astype(np.uint8), kernel).astype(bool)
    missing = gt_skeleton & ~covered
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8)
    gaps = []
    for gap_id in range(1, count):
        left, top, width, height, area = map(int, stats[gap_id])
        if area < min_pixels:
            continue
        y0, y1 = max(0, top - margin), min(missing.shape[0], top + height + margin)
        x0, x1 = max(0, left - margin), min(missing.shape[1], left + width + margin)
        region = components[y0:y1, x0:x1] == gap_id
        boundary = (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    & gt_skeleton[y0:y1, x0:x1] & covered[y0:y1, x0:x1] & ~region)
        anchor_count, anchor_labels, _, centers = cv2.connectedComponentsWithStats(
            boundary.astype(np.uint8), connectivity=8)
        local_baseline = baseline[y0:y1, x0:x1]
        _, baseline_labels = cv2.connectedComponents(local_baseline.astype(np.uint8), connectivity=8)
        anchors = []
        for anchor_id in range(1, anchor_count):
            anchor = anchor_labels == anchor_id
            ids = nearby_ids(anchor, local_baseline, baseline_labels, kernel)
            if ids:
                anchors.append((anchor, centers[anchor_id], ids))
        pairs = [(i, j) for i, j in combinations(range(len(anchors)), 2)
                 if np.linalg.norm(anchors[i][1] - anchors[j][1]) >= 4
                 and anchors[i][2].isdisjoint(anchors[j][2])]
        if pairs:
            gaps.append((gap_id, (y0, y1, x0, x1), region, anchors, pairs, area))
    return gaps, missing


def gap_metrics(gaps, floor, baseline_logit, floor_logit, tolerance):
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    covered_floor = cv2.dilate(floor.astype(np.uint8), kernel).astype(bool)
    records = []
    for gap_id, (y0, y1, x0, x1), region, anchors, pairs, area in gaps:
        local_floor = floor[y0:y1, x0:x1]
        _, floor_labels = cv2.connectedComponents(local_floor.astype(np.uint8), connectivity=8)
        floor_anchor_ids = [nearby_ids(anchor, local_floor, floor_labels, kernel)
                            for anchor, _, _ in anchors]
        connected = any(not floor_anchor_ids[i].isdisjoint(floor_anchor_ids[j])
                        and floor_anchor_ids[i] for i, j in pairs)
        center_covered = covered_floor[y0:y1, x0:x1] & region
        # Require the whole missed GT centerline and both flanks, not just edge pixels.
        full = bool(connected and np.all(center_covered[region]))
        boundary = np.zeros_like(region)
        for anchor, _, _ in anchors:
            boundary |= anchor
        edge = region & (cv2.distanceTransform((~boundary).astype(np.uint8),
                                                cv2.DIST_L2, 3) <= 4)
        interior = region & ~edge
        difference = floor_logit[y0:y1, x0:x1] - baseline_logit[y0:y1, x0:x1]
        records.append({"gap_id": gap_id, "x": x0, "y": y0, "pixels": area,
                        "connected_flanks": int(bool(connected)), "complete_closure": int(full),
                        "centerline_recovered_pixels": int(center_covered.sum()),
                        "centerline_recovered_fraction": float(center_covered.sum() / area),
                        "edge_recovered_pixels": int((center_covered & edge).sum()),
                        "interior_recovered_pixels": int((center_covered & interior).sum()),
                        "interior_pixels": int(interior.sum()),
                        "mean_gap_logit_delta": float(difference[region].mean()),
                        "mean_interior_logit_delta": float(difference[interior].mean())
                        if interior.any() else None})
    return records


def bbox_near(first, second, distance):
    x1, y1, w1, h1 = map(int, first)
    x2, y2, w2, h2 = map(int, second)
    dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
    dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
    return np.hypot(dx, dy) <= distance


def adjacent_gt_joins(pred, gt_labels, gt_stats, near_pixels):
    """Conservative screen: predicted components linking distinct nearby GT components."""
    _, labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    joined = set()
    overlapping = (pred & (gt_labels > 0))
    component_pairs = np.unique(np.stack((labels[overlapping], gt_labels[overlapping]), axis=1), axis=0)
    by_pred = {}
    for pred_id, gt_id in component_pairs:
        by_pred.setdefault(int(pred_id), set()).add(int(gt_id))
    for ids in by_pred.values():
        ids = sorted(ids)
        for first, second in combinations(ids, 2):
            if bbox_near(gt_stats[first, :4], gt_stats[second, :4], near_pixels):
                joined.add((first, second))
    return joined


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--floor_dir", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--source_patch_size", type=int, required=True)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--gap_margin", type=int, default=20)
    parser.add_argument("--min_gap_pixels", type=int, default=3)
    parser.add_argument("--near_road_pixels", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=0,
                        help="0 evaluates all paired validation images")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.tolerance < 0 or args.source_patch_size < 1:
        parser.error("Invalid threshold, tolerance or image size")
    baseline = index_logits(args.baseline_dir)
    floor = index_logits(args.floor_dir)
    if set(baseline) != set(floor):
        raise ValueError(f"Prediction pairs incomplete: baseline_only={sorted(set(baseline)-set(floor))[:10]}, "
                         f"floor_only={sorted(set(floor)-set(baseline))[:10]}")
    keys = sorted(baseline)
    images = index_images(args.root_path, args.split)
    if set(keys) != images:
        raise ValueError(f"Incomplete validation image pairing: missing={sorted(images-set(keys))[:10]}, "
                         f"extra={sorted(set(keys)-images)[:10]}")
    gt_paths = index_ground_truth(args.root_path, args.split, keys)
    if args.max_images:
        keys = keys[:args.max_images]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(DEFAULT_PROTOCOL, image_size=args.source_patch_size)
    metric_baseline, metric_floor, cases, all_gaps = [], [], [], []
    total_new_joins = total_resolved_joins = 0
    for index, key in enumerate(keys, 1):
        base_logit = np.load(baseline[key], allow_pickle=False).astype(np.float32, copy=False)
        floor_logit = np.load(floor[key], allow_pickle=False).astype(np.float32, copy=False)
        size = args.source_patch_size
        if base_logit.shape != (size, size) or floor_logit.shape != (size, size):
            raise ValueError(f"Mismatched logit shape for {key}")
        if not np.isfinite(base_logit).all() or not np.isfinite(floor_logit).all():
            raise ValueError(f"Nonfinite logit for {key}")
        gt = read_gt(gt_paths[key], size)
        baseline_mask = read_prediction(baseline[key], key, size)
        floor_mask = read_prediction(floor[key], key, size)
        prepared = prepare_target(gt, protocol)
        base_row = image_metrics(baseline_mask, prepared, protocol)
        floor_row = image_metrics(floor_mask, prepared, protocol)
        metric_baseline.append(base_row)
        metric_floor.append(floor_row)
        gaps, missing = verified_gaps(prepared["skeleton"], baseline_mask,
                                      args.tolerance, args.gap_margin, args.min_gap_pixels)
        gap_rows = gap_metrics(gaps, floor_mask, base_logit, floor_logit, args.tolerance)
        for gap in gap_rows:
            gap["case"] = key
        all_gaps.extend(gap_rows)
        _, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(
            gt.astype(np.uint8), connectivity=8)
        base_joins = adjacent_gt_joins(baseline_mask, gt_labels, gt_stats, args.near_road_pixels)
        floor_joins = adjacent_gt_joins(floor_mask, gt_labels, gt_stats, args.near_road_pixels)
        new_joins, resolved_joins = len(floor_joins - base_joins), len(base_joins - floor_joins)
        total_new_joins += new_joins
        total_resolved_joins += resolved_joins
        delta = floor_logit - base_logit
        added = floor_mask & ~baseline_mask
        removed = baseline_mask & ~floor_mask
        cases.append({"case": key, "verified_baseline_gaps": len(gaps),
                      "completely_closed_gaps": sum(g["complete_closure"] for g in gap_rows),
                      "connected_flank_gaps": sum(g["connected_flanks"] for g in gap_rows),
                      "baseline_missing_centerline_pixels": int(missing.sum()),
                      "new_true_positive_pixels": int((added & gt).sum()),
                      "new_false_positive_pixels": int((added & ~gt).sum()),
                      "lost_true_positive_pixels": int((removed & gt).sum()),
                      "removed_false_positive_pixels": int((removed & ~gt).sum()),
                      "mean_logit_delta_verified_gap": float(np.mean([
                          g["mean_gap_logit_delta"] for g in gap_rows])) if gap_rows else None,
                      "mean_logit_delta_gt_road": float(delta[gt].mean()) if gt.any() else None,
                      "mean_logit_delta_background": float(delta[~gt].mean()) if (~gt).any() else None,
                      "new_nearby_gt_component_joins_proxy": new_joins,
                      "resolved_nearby_gt_component_joins_proxy": resolved_joins,
                      "baseline_apls_bidirectional_approx": base_row["apls_bidirectional_approx"],
                      "floor_apls_bidirectional_approx": floor_row["apls_bidirectional_approx"]})
        print(f"[{index}/{len(keys)}] {key}: closed={cases[-1]['completely_closed_gaps']}/"
              f"{len(gaps)} new_FP={cases[-1]['new_false_positive_pixels']} "
              f"APLS={base_row['apls_bidirectional_approx']:.4f}->"
              f"{floor_row['apls_bidirectional_approx']:.4f}", flush=True)
    baseline_summary, floor_summary = summarize(metric_baseline), summarize(metric_floor)
    interiors = [g["mean_interior_logit_delta"] for g in all_gaps
                 if g["mean_interior_logit_delta"] is not None]
    summary = {"protocol": protocol, "split": args.split, "threshold": args.threshold,
               "images": len(keys), "baseline": baseline_summary, "floor_0_2": floor_summary,
               "diagnostic": {
                   "verified_baseline_gaps": len(all_gaps),
                   "completely_closed_gaps": sum(g["complete_closure"] for g in all_gaps),
                   "connected_flank_gaps": sum(g["connected_flanks"] for g in all_gaps),
                   "interior_recovered_pixels": sum(g["interior_recovered_pixels"] for g in all_gaps),
                   "edge_recovered_pixels": sum(g["edge_recovered_pixels"] for g in all_gaps),
                   "mean_interior_logit_delta_per_gap": float(np.mean(interiors)) if interiors else None,
                   "interior_gaps_with_positive_logit_delta": sum(value > 0 for value in interiors),
                   "interior_gaps_measured": len(interiors),
                   "new_true_positive_pixels": sum(c["new_true_positive_pixels"] for c in cases),
                   "new_false_positive_pixels": sum(c["new_false_positive_pixels"] for c in cases),
                   "lost_true_positive_pixels": sum(c["lost_true_positive_pixels"] for c in cases),
                   "removed_false_positive_pixels": sum(c["removed_false_positive_pixels"] for c in cases),
                   "new_nearby_gt_component_joins_proxy": total_new_joins,
                   "resolved_nearby_gt_component_joins_proxy": total_resolved_joins}}
    save_csv(out / "per_image.csv", cases)
    save_csv(out / "verified_gaps.csv", all_gaps)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["diagnostic"], ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
