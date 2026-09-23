"""Diagnose raster road connectivity for the SAM-Road DeepGlobe data1 split.

This reports centerline/raster proxies only. It does not calculate graph TOPO
or APLS, which require vector road-network ground truth.
"""

import argparse
import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data1_road_dataset import Data1RoadDataset
from eval_data1_road import collect, skeletonize
from model import SAMRoad
from utils import load_config


def component_sizes(binary, connectivity=8):
    """Return foreground component pixel counts under explicit connectivity."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=connectivity
    )
    return stats[1:, cv2.CC_STAT_AREA]


def diagnose_one(prob, target, name, threshold, min_component_pixels,
                 min_gap_pixels, match_tolerance):
    gt = target[0, 0].numpy() > 0.5
    pred = prob[0, 0].numpy() >= threshold
    gt_skel = skeletonize(gt)
    pred_skel = skeletonize(pred)

    gt_skel_n = int(gt_skel.sum())
    pred_skel_n = int(pred_skel.sum())
    topo_p = float((pred_skel & gt).sum()) / max(pred_skel_n, 1)
    topo_r = float((gt_skel & pred).sum()) / max(gt_skel_n, 1)
    cldice = 2.0 * topo_p * topo_r / max(topo_p + topo_r, 1e-12)

    pred_sizes_all = component_sizes(pred_skel)
    gt_sizes_all = component_sizes(gt_skel)
    pred_sizes = pred_sizes_all[pred_sizes_all >= min_component_pixels]
    gt_sizes = gt_sizes_all[gt_sizes_all >= min_component_pixels]
    pred_components = int(pred_sizes.size)
    gt_components = int(gt_sizes.size)
    pred_kept_pixels = int(pred_sizes.sum()) if pred_components else 0
    largest_share = (float(pred_sizes.max() / pred_kept_pixels)
                     if pred_components else 0.0)

    # Count GT centerline pixels not covered by the predicted road mask,
    # allowing a small pixel-distance tolerance for raster alignment.
    distance_to_pred = cv2.distanceTransform(
        (~pred).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    missing_centerline = gt_skel & (distance_to_pred > match_tolerance)
    gap_sizes_all = component_sizes(missing_centerline)
    gap_sizes = gap_sizes_all[gap_sizes_all >= min_gap_pixels]
    gap_count = int(gap_sizes.size)
    gap_pixels = int(gap_sizes.sum()) if gap_count else 0

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "image": name,
        "threshold": float(threshold),
        "pixel_iou": tp / max(tp + fp + fn, 1),
        "pixel_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "pixel_precision": precision,
        "pixel_recall": recall,
        "topology_precision": topo_p,
        "topology_recall": topo_r,
        "clDice": cldice,
        "gt_skeleton_pixels": gt_skel_n,
        "pred_skeleton_pixels": pred_skel_n,
        "gt_components": gt_components,
        "pred_components": pred_components,
        "beta0_abs_error": abs(pred_components - gt_components),
        "components_per_1000_gt_skeleton_pixels":
            pred_components * 1000.0 / max(gt_skel_n, 1),
        "largest_component_share": largest_share,
        "uncovered_gt_centerline_pixels": int(missing_centerline.sum()),
        "uncovered_centerline_fraction":
            float(missing_centerline.sum()) / max(gt_skel_n, 1),
        "gap_count": gap_count,
        "gap_pixels": gap_pixels,
        "mean_gap_pixels": float(gap_sizes.mean()) if gap_count else 0.0,
        "max_gap_pixels": int(gap_sizes.max()) if gap_count else 0,
    }


def summarize(rows, threshold):
    keys = [
        "pixel_iou", "pixel_f1", "pixel_precision", "pixel_recall",
        "topology_precision", "topology_recall", "clDice",
        "gt_components", "pred_components", "beta0_abs_error",
        "components_per_1000_gt_skeleton_pixels", "largest_component_share",
        "uncovered_centerline_fraction", "gap_count", "gap_pixels",
        "mean_gap_pixels", "max_gap_pixels",
    ]
    # Macro means give each image equal weight; totals/counts are also retained
    # for fragmentation and missing-centerline diagnostics.
    result = {"threshold": float(threshold), "images": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"macro_{key}"] = float(values.mean()) if len(values) else 0.0
        result[f"median_{key}"] = float(np.median(values)) if len(values) else 0.0
    result["total_gap_count"] = int(sum(row["gap_count"] for row in rows))
    result["total_gap_pixels"] = int(sum(row["gap_pixels"] for row in rows))
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Raster-based connectivity diagnostics for SAM-Road data1"
    )
    parser.add_argument("--config", default="config/data1_road_vitb_512.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam_ckpt", default="")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds", default="",
                        help="Comma-separated validation thresholds; mutually exclusive with --threshold")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_component_pixels", type=int, default=5)
    parser.add_argument("--min_gap_pixels", type=int, default=2)
    parser.add_argument("--match_tolerance", type=float, default=1.5,
                        help="Allowed distance in pixels when finding uncovered GT centerline")
    args = parser.parse_args()

    if (args.threshold is None) == (not args.thresholds.strip()):
        parser.error("Provide exactly one of --threshold or --thresholds")
    if args.thresholds.strip() and args.split != "val":
        parser.error("Threshold sweeps are allowed on val only; use a fixed --threshold on test")
    if args.min_component_pixels < 1 or args.min_gap_pixels < 1:
        parser.error("Component and gap minimum lengths must be positive")

    thresholds = ([float(x.strip()) for x in args.thresholds.split(",")]
                  if args.thresholds.strip() else [args.threshold])
    if any(t < 0 or t > 1 for t in thresholds):
        parser.error("Thresholds must be between 0 and 1")

    config = load_config(args.config)
    config.SAM_CKPT_PATH = args.sam_ckpt or config.SAM_CKPT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMRoad(config).to(device)
    # Training checkpoints contain NumPy metadata; only load trusted checkpoints.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(f"No model_state_dict/state_dict in checkpoint; keys={list(checkpoint)[:20]}")
    incompat = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}; missing={len(incompat.missing_keys)}, "
          f"unexpected={len(incompat.unexpected_keys)}; device={device}")

    dataset = Data1RoadDataset(args.data_root, args.split,
                               random_crop=False, augment=False)
    values = collect(model, DataLoader(dataset, batch_size=1, shuffle=False), device)

    all_image_rows = []
    summary_rows = []
    for threshold in thresholds:
        image_rows = [diagnose_one(prob, target, name, threshold,
                                   args.min_component_pixels, args.min_gap_pixels,
                                   args.match_tolerance)
                      for prob, target, name in values]
        summary = summarize(image_rows, threshold)
        summary.update({
            "min_component_pixels": args.min_component_pixels,
            "min_gap_pixels": args.min_gap_pixels,
            "match_tolerance_pixels": args.match_tolerance,
        })
        all_image_rows.extend(image_rows)
        summary_rows.append(summary)
        print(
            f"threshold={threshold:.3f} pixelF1={summary['macro_pixel_f1']:.4f} "
            f"clDice={summary['macro_clDice']:.4f} "
            f"topoP={summary['macro_topology_precision']:.4f} "
            f"topoR={summary['macro_topology_recall']:.4f} "
            f"components/img={summary['macro_pred_components']:.2f} "
            f"|Δbeta0|={summary['macro_beta0_abs_error']:.2f} "
            f"uncovered_centerline={summary['macro_uncovered_centerline_fraction']:.4f} "
            f"gaps/img={summary['macro_gap_count']:.2f} "
            f"mean_gap_px={summary['macro_mean_gap_pixels']:.2f} "
            f"largest_share={summary['macro_largest_component_share']:.4f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    per_image_path = os.path.join(args.output_dir, "connectivity_per_image.csv")
    summary_path = os.path.join(args.output_dir, "connectivity_summary.csv")
    write_csv(per_image_path, all_image_rows)
    write_csv(summary_path, summary_rows)
    print(f"Per-image diagnostics: {per_image_path}")
    print(f"Summary diagnostics: {summary_path}")
    print("Note: these are raster skeleton proxies, not graph TOPO/APLS scores.")


if __name__ == "__main__":
    main()
