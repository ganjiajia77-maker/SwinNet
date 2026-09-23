import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from eval_data1_common import image_names, load_model, mask_path, predict_full


EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def component_stats(mask):
    labels, count = ndimage.label(mask, structure=EIGHT_CONNECTED)
    if count == 0:
        return 0, 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    foreground = int(mask.sum())
    return int(count), int(sizes.max()), float(sizes.max() / max(foreground, 1))


def image_metrics(probability, target, threshold):
    prediction = probability >= threshold
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    pred_components, pred_largest, pred_largest_fraction = component_stats(prediction)
    gt_components, gt_largest, gt_largest_fraction = component_stats(target)
    pred_skeleton = skeletonize(prediction)
    gt_skeleton = skeletonize(target)
    pred_skel_precision = int(np.logical_and(pred_skeleton, target).sum()) / max(int(pred_skeleton.sum()), 1)
    gt_skel_recall = int(np.logical_and(gt_skeleton, prediction).sum()) / max(int(gt_skeleton.sum()), 1)
    cldice = 2 * pred_skel_precision * gt_skel_recall / max(pred_skel_precision + gt_skel_recall, 1e-12)
    pred_skel_components, _, _ = component_stats(pred_skeleton)
    gt_skel_components, _, _ = component_stats(gt_skeleton)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "pred_components_8": pred_components,
        "gt_components_8": gt_components,
        "fragmentation_excess": max(pred_components - gt_components, 0) / max(gt_components, 1),
        "pred_largest_component_pixels": pred_largest,
        "gt_largest_component_pixels": gt_largest,
        "pred_largest_component_fraction": pred_largest_fraction,
        "gt_largest_component_fraction": gt_largest_fraction,
        "pred_skeleton_components_8": pred_skel_components,
        "gt_skeleton_components_8": gt_skel_components,
        "skeleton_precision": pred_skel_precision,
        "skeleton_recall": gt_skel_recall,
        "cldice": cldice,
    }


def aggregate(rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "num_images": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
    }
    for key in rows[0]:
        if key not in ("image_id", "tp", "fp", "fn", "iou", "f1", "precision", "recall"):
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            result[key + "_mean_per_image"] = float(values.mean())
    return result


def main():
    parser = argparse.ArgumentParser(description="CoANet full-image connectivity diagnostics for data1")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap_stride", type=int, default=512)
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--out_stride", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    names = image_names(args.root_path, args.split)
    rows = []

    for index, name in enumerate(names, 1):
        image = Image.open(os.path.join(args.root_path, args.split, "image", name)).convert("RGB")
        target = np.asarray(Image.open(mask_path(args.root_path, args.split, name)).convert("L")) >= 128
        probability = predict_full(model, image, args.tile_size, args.overlap_stride, device)
        metrics = image_metrics(probability, target, args.threshold)
        metrics["image_id"] = name
        rows.append(metrics)
        print("[{}/{}] {} IoU={:.4f} clDice={:.4f} pred_components={} frag_excess={:.4f}".format(
            index, len(names), name, metrics["iou"], metrics["cldice"],
            metrics["pred_components_8"], metrics["fragmentation_excess"]), flush=True)

    if not rows:
        raise RuntimeError("No images found under split: " + args.split)

    per_image_path = os.path.join(args.output_dir, "connectivity_metrics_per_image.csv")
    with open(per_image_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id"] + [key for key in rows[0] if key != "image_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {"split": args.split, "threshold": args.threshold, "tile_size": args.tile_size,
               "overlap_stride": args.overlap_stride, **aggregate(rows)}
    summary_path = os.path.join(args.output_dir, "connectivity_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Per-image CSV: " + per_image_path)
    print("Summary: " + summary_path)


if __name__ == "__main__":
    main()
