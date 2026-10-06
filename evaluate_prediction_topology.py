"""Evaluate topology of saved full-resolution road prediction masks."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

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


def case_metrics(case, prediction, ground_truth, short_area_threshold):
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
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.short_area_threshold < 1:
        parser.error("--short_area_threshold must be positive")

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
        rows.append(case_metrics(case, pred, gt, args.short_area_threshold))
        print(f"[{index}/{len(pred_files)}] {case}: IoU={rows[-1]['iou']:.4f} "
              f"clDice={rows[-1]['cldice']:.4f} pred_comp={rows[-1]['pred_components']}", flush=True)

    summary = summarize(rows)
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
