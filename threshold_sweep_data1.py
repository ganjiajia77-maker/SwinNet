import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from eval_data1_common import image_names, load_model, mask_path, predict_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap_stride", type=int, default=512)
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--out_stride", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    names = image_names(args.root_path, args.split)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    probabilities = []
    targets = []
    for index, name in enumerate(names, 1):
        image = Image.open(os.path.join(args.root_path, args.split, "image", name)).convert("RGB")
        target = np.asarray(Image.open(mask_path(args.root_path, args.split, name)).convert("L")) >= 128
        probabilities.append(predict_full(model, image, args.tile_size, args.overlap_stride, device))
        targets.append(target)
        print("[{}/{}] {}".format(index, len(names), name), flush=True)
    rows = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        for probability, target in zip(probabilities, targets):
            prediction = probability >= threshold
            tp += int(np.logical_and(prediction, target).sum())
            fp += int(np.logical_and(prediction, ~target).sum())
            fn += int(np.logical_and(~prediction, target).sum())
            tn += int(np.logical_and(~prediction, ~target).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({"threshold": threshold, "iou": tp / max(tp + fp + fn, 1),
                     "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                     "precision": precision, "recall": recall,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    with open(os.path.join(args.output_dir, "threshold_sweep_val.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["iou"])
    with open(os.path.join(args.output_dir, "best_threshold.txt"), "w") as handle:
        handle.write("{:.6f}\n".format(best["threshold"]))
    print("Best threshold (IoU): {:.2f} -> IoU: {:.6f}".format(best["threshold"], best["iou"]))


if __name__ == "__main__":
    main()
