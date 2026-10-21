"""Measure the output-logit effect of global writeback at verified baseline gaps."""

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np

from compare_h0_gate_floor import (
    index_ground_truth, index_images, index_logits, read_gt, read_prediction,
    save_csv, skeletonize, verified_gaps,
)
from road_metrics import confusion, segmentation


def logged_alpha(path, name):
    matches = re.findall(rf"{re.escape(name)}=([-+0-9.eE]+)",
                         path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"Missing {name} in {path}")
    return float(matches[-1])


def measure(values, totals):
    if not values.size:
        return None
    mean = float(values.mean(dtype=np.float64))
    totals["means"].append(mean)
    totals["sum"] += float(values.sum(dtype=np.float64))
    totals["pixels"] += int(values.size)
    return mean


def summarize_effect(totals, tolerance):
    means = np.asarray(totals["means"], dtype=np.float64)
    return {
        "gaps_measured": int(means.size),
        "pixels_measured": totals["pixels"],
        "mean_logit_delta_per_gap": float(means.mean()) if means.size else None,
        "mean_logit_delta_per_pixel": totals["sum"] / totals["pixels"]
        if totals["pixels"] else None,
        "gaps_positive": int((means > tolerance).sum()),
        "gaps_negative": int((means < -tolerance).sum()),
        "gaps_near_zero": int((np.abs(means) <= tolerance).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline_dir", required=True)
    p.add_argument("--floor_dir", required=True)
    p.add_argument("--zero_dir", required=True)
    p.add_argument("--root_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--source_patch_size", type=int, required=True)
    p.add_argument("--tolerance", type=int, default=2)
    p.add_argument("--gap_margin", type=int, default=20)
    p.add_argument("--min_gap_pixels", type=int, default=3)
    p.add_argument("--sign_tolerance", type=float, default=0.001,
                   help="near-zero interval for gap-mean logit effects; saved logits are float16")
    args = p.parse_args()
    if not 0 < args.threshold < 1 or args.sign_tolerance < 0:
        p.error("Invalid threshold or sign tolerance")
    if args.source_patch_size < 1 or args.tolerance < 0 or args.gap_margin < 0 or args.min_gap_pixels < 1:
        p.error("Invalid image size or gap parameters")

    paths = {"original": index_logits(args.baseline_dir),
             "floor_0p2": index_logits(args.floor_dir),
             "zero_writeback": index_logits(args.zero_dir)}
    keys = set(paths["original"])
    if any(set(index) != keys for index in paths.values()):
        raise ValueError("Original, floor and zero-writeback predictions are not fully paired")
    if keys != index_images(args.root_path, args.split):
        raise ValueError("Prediction images do not match the complete requested split")
    labels = index_ground_truth(args.root_path, args.split, keys)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    original_log = Path(args.baseline_dir).parent / "original_inference.log"
    floor_log = Path(args.floor_dir).parent / "floor_0p2_inference.log"
    zero_log = Path(args.zero_dir).parent / "zero_inference.log"
    alpha = logged_alpha(original_log, "trained_alpha_global")
    saved_alphas = (logged_alpha(floor_log, "trained_alpha_global"),
                    logged_alpha(zero_log, "trained_alpha_global"))
    if not all(np.isclose(alpha, value, rtol=0, atol=1e-8) for value in saved_alphas):
        raise ValueError("Trained global alpha differs between the three inference logs")
    if logged_alpha(zero_log, "active_alpha_global") != 0:
        raise ValueError("Zero-writeback inference did not use alpha=0")

    effect_names = ("original_minus_zero", "floor_minus_original", "floor_minus_zero")
    totals = {name: {"means": [], "sum": 0.0, "pixels": 0} for name in effect_names}
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in paths}
    per_gap, per_image = [], []
    for n, key in enumerate(sorted(keys), 1):
        logits, masks = {}, {}
        size = args.source_patch_size
        for name, index in paths.items():
            value = np.load(index[key], allow_pickle=False).astype(np.float32)
            if value.shape != (size, size) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} logits for {key}")
            logits[name] = value
            masks[name] = read_prediction(index[key], key, size)
        gt = read_gt(labels[key], size)
        for name, mask in masks.items():
            for metric, value in confusion(mask, gt).items():
                counts[name][metric] += value
        gaps, _ = verified_gaps(skeletonize(gt), masks["original"],
                                args.tolerance, args.gap_margin, args.min_gap_pixels)
        differences = {
            "original_minus_zero": logits["original"] - logits["zero_writeback"],
            "floor_minus_original": logits["floor_0p2"] - logits["original"],
            "floor_minus_zero": logits["floor_0p2"] - logits["zero_writeback"],
        }
        for gap_id, (y0, y1, x0, x1), region, anchors, _, area in gaps:
            boundary = np.zeros_like(region)
            for anchor, _, _ in anchors:
                boundary |= anchor
            edge = region & (cv2.distanceTransform((~boundary).astype(np.uint8),
                                                   cv2.DIST_L2, 3) <= 4)
            interior = region & ~edge
            row = {"case": key, "gap_id": gap_id, "x": x0, "y": y0,
                   "pixels": area, "interior_pixels": int(interior.sum())}
            for name, difference in differences.items():
                local = difference[y0:y1, x0:x1]
                row[f"{name}_gap_mean"] = float(local[region].mean(dtype=np.float64))
                row[f"{name}_interior_mean"] = measure(local[interior], totals[name])
            per_gap.append(row)
        original, zero = masks["original"], masks["zero_writeback"]
        added, removed = original & ~zero, zero & ~original
        per_image.append({
            "case": key, "verified_baseline_gaps": len(gaps),
            "original_vs_zero_added_tp": int((added & gt).sum()),
            "original_vs_zero_added_fp": int((added & ~gt).sum()),
            "original_vs_zero_removed_tp": int((removed & gt).sum()),
            "original_vs_zero_removed_fp": int((removed & ~gt).sum()),
            "original_minus_zero_gt_road_mean": float(differences["original_minus_zero"][gt].mean())
            if gt.any() else None,
            "original_minus_zero_background_mean": float(differences["original_minus_zero"][~gt].mean())
            if (~gt).any() else None,
        })
        print(f"[{n}/{len(keys)}] {key}: verified_gaps={len(gaps)}", flush=True)

    summary = {
        "images": len(keys), "split": args.split, "threshold": args.threshold,
        "trained_alpha_global": alpha, "verified_baseline_gaps": len(per_gap),
        "gap_definition": {"centerline_tolerance": args.tolerance,
                           "gap_margin": args.gap_margin,
                           "min_gap_pixels": args.min_gap_pixels,
                           "edge_distance": 4},
        "sign_tolerance": args.sign_tolerance,
        "effect_definition": "Positive means higher downstream road logits; channel-wise feature signs are not interpreted.",
        "original_minus_zero": summarize_effect(totals["original_minus_zero"], args.sign_tolerance),
        "floor_minus_original": summarize_effect(totals["floor_minus_original"], args.sign_tolerance),
        "floor_minus_zero": summarize_effect(totals["floor_minus_zero"], args.sign_tolerance),
        "segmentation": {name: dict(counts[name], **segmentation(counts[name])) for name in paths},
        "prediction_dirs": {"original": args.baseline_dir, "floor_0p2": args.floor_dir,
                            "zero_writeback": args.zero_dir},
    }
    save_csv(out / "per_gap.csv", per_gap)
    save_csv(out / "per_image.csv", per_image)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
