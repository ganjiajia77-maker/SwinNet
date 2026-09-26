"""Sweep coarse road ROIs from cached 256x256 surface probability maps."""

import argparse
import csv
import os

import cv2
import numpy as np
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_road_skeleton import RoadSkeletonDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability_dir", required=True, help="Folder of per-image .npy probability maps.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--original_tile_size", type=int, default=256)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.55, 0.65])
    parser.add_argument("--dilation_radii", type=int, nargs="+", default=[0, 2, 4, 8, 12, 16])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0, help="0 processes every image with a matching map.")
    return parser.parse_args()


def load_probability(path, target_shape):
    probability = np.load(path).astype(np.float32)
    probability = np.squeeze(probability)
    if probability.ndim != 2:
        raise ValueError(f"Expected a 2D probability map at {path}, got shape {probability.shape}")
    if probability.shape != target_shape:
        probability = cv2.resize(
            probability,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"Non-finite values in probability map: {path}")
    return np.clip(probability, 0.0, 1.0)


def make_roi(probability, threshold, radius):
    selected = (probability >= threshold).astype(np.uint8)
    if radius <= 0 or not selected.any():
        return selected.astype(bool)
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(selected, kernel, iterations=1).astype(bool)


def count_selected_tiles(roi, source_patch_size, image_size, tile_size):
    # Map the 256-resolution ROI back to the native source patch, then count
    # source tiles touched by at least one selected pixel.
    native = cv2.resize(
        roi.astype(np.uint8),
        (source_patch_size, source_patch_size),
        interpolation=cv2.INTER_NEAREST,
    )
    selected = 0
    total = 0
    for top in range(0, source_patch_size, tile_size):
        for left in range(0, source_patch_size, tile_size):
            tile = native[top:min(top + tile_size, source_patch_size), left:min(left + tile_size, source_patch_size)]
            total += 1
            selected += int(tile.any())
    return selected, total


def road_connectivity_breaks(roi, gt_skeleton):
    gt_count, gt_labels = cv2.connectedComponents(gt_skeleton.astype(np.uint8), connectivity=8)
    clipped = np.logical_and(roi, gt_skeleton).astype(np.uint8)
    clipped_count, clipped_labels = cv2.connectedComponents(clipped, connectivity=8)
    split_components = 0
    broken_connections = 0
    missed_components = 0
    for component_id in range(1, gt_count):
        positions = gt_labels == component_id
        observed = np.unique(clipped_labels[positions])
        observed = observed[observed > 0]
        if observed.size == 0:
            missed_components += 1
        elif observed.size > 1:
            split_components += 1
            broken_connections += int(observed.size - 1)
    return split_components, broken_connections, missed_components


def evaluate_one(probability, target, thresholds, radii, args):
    target = target.astype(bool)
    gt_skeleton = skeletonize(target)
    road_pixels = int(target.sum())
    skeleton_pixels = int(gt_skeleton.sum())
    rows = []
    for threshold in thresholds:
        for radius in radii:
            roi = make_roi(probability, threshold, radius)
            covered_road = int(np.logical_and(roi, target).sum())
            covered_skeleton = int(np.logical_and(roi, gt_skeleton).sum())
            split_components, broken_connections, missed_components = road_connectivity_breaks(
                roi, gt_skeleton
            )
            selected_tiles, total_tiles = count_selected_tiles(
                roi,
                args.source_patch_size,
                args.img_size,
                args.original_tile_size,
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "dilation_radius_256px": int(radius),
                    "dilation_radius_source_px": int(round(radius * args.source_patch_size / args.img_size)),
                    "road_coverage": covered_road / max(road_pixels, 1),
                    "skeleton_coverage": covered_skeleton / max(skeleton_pixels, 1),
                    "broken_true_connections": broken_connections,
                    "split_gt_skeleton_components": split_components,
                    "missed_gt_skeleton_components": missed_components,
                    "selected_tiles": selected_tiles,
                    "total_tiles": total_tiles,
                    "selected_tile_fraction": selected_tiles / max(total_tiles, 1),
                    "roi_pixel_fraction": float(roi.mean()),
                    "gt_road_pixels": road_pixels,
                    "gt_skeleton_pixels": skeleton_pixels,
                }
            )
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        tile_size=None,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    thresholds = sorted(set(float(value) for value in args.thresholds))
    radii = sorted(set(int(value) for value in args.dilation_radii))
    if not thresholds or any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("thresholds must be within [0, 1]")
    if not radii or any(value < 0 for value in radii):
        raise ValueError("dilation radii must be non-negative")
    if args.original_tile_size <= 0:
        raise ValueError("original_tile_size must be positive")

    per_image_rows = []
    missing_maps = []
    for batch in tqdm(loader, desc="Coarse road ROI sweep"):
        masks = batch["mask"].squeeze(1).numpy() > 0.5
        names = batch["image_name"]
        for index, image_name in enumerate(names):
            stem = os.path.splitext(os.path.basename(str(image_name)))[0]
            map_path = os.path.join(args.probability_dir, stem + ".npy")
            if not os.path.isfile(map_path):
                missing_maps.append(stem)
                continue
            probability = load_probability(map_path, masks[index].shape)
            rows = evaluate_one(probability, masks[index], thresholds, radii, args)
            for row in rows:
                per_image_rows.append({"image": str(image_name), **row})
            if args.max_images > 0 and len({row["image"] for row in per_image_rows}) >= args.max_images:
                break
        if args.max_images > 0 and len({row["image"] for row in per_image_rows}) >= args.max_images:
            break

    if not per_image_rows:
        raise RuntimeError(f"No probability maps matched images in {args.probability_dir}")
    if missing_maps:
        print(f"[WARN] Probability maps missing for {len(missing_maps)} dataset images; skipped them.", flush=True)

    per_image_path = os.path.join(args.output_dir, "coarse_roi_per_image.csv")
    with open(per_image_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_rows = []
    keys = (
        "road_coverage",
        "skeleton_coverage",
        "broken_true_connections",
        "split_gt_skeleton_components",
        "missed_gt_skeleton_components",
        "selected_tile_fraction",
        "roi_pixel_fraction",
    )
    for threshold in thresholds:
        for radius in radii:
            selected_rows = [
                row for row in per_image_rows
                if row["threshold"] == threshold and row["dilation_radius_256px"] == radius
            ]
            summary = {
                "images": len(selected_rows),
                "threshold": threshold,
                "dilation_radius_256px": radius,
                "dilation_radius_source_px": int(round(radius * args.source_patch_size / args.img_size)),
            }
            for key in keys:
                values = np.asarray([row[key] for row in selected_rows], dtype=np.float64)
                summary[f"mean_{key}"] = float(values.mean()) if values.size else float("nan")
                if key in {"broken_true_connections", "split_gt_skeleton_components", "missed_gt_skeleton_components"}:
                    summary[f"total_{key}"] = int(values.sum()) if values.size else 0
            summary_rows.append(summary)

    summary_path = os.path.join(args.output_dir, "coarse_roi_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Images evaluated: {len(set(row['image'] for row in per_image_rows))}")
    print(f"Per-image metrics: {per_image_path}")
    print(f"Threshold/radius summary: {summary_path}")
    print("\nTop candidates by road coverage with fewer selected tiles:")
    ranked = sorted(summary_rows, key=lambda row: (-row["mean_road_coverage"], row["mean_selected_tile_fraction"]))
    for row in ranked[:10]:
        print(
            f"thr={row['threshold']:.2f} radius={row['dilation_radius_256px']:>2d}px "
            f"road={row['mean_road_coverage']:.4f} skeleton={row['mean_skeleton_coverage']:.4f} "
            f"broken={row['total_broken_true_connections']} "
            f"tiles={row['mean_selected_tile_fraction']:.3f} roi={row['mean_roi_pixel_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
