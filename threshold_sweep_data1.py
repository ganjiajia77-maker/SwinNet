import argparse
import csv
import os

import numpy as np
import torch
from eval_data1_common import (binary_prediction, image_names, load_eval_case,
                               load_model, predict_components_full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap_stride", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--prediction_mode", choices=["paper_fusion", "surface"], default="paper_fusion")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--out_stride", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    names = image_names(args.root_path, args.split)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    totals = {threshold: [0, 0, 0, 0] for threshold in thresholds}
    for index, name in enumerate(names, 1):
        image, target = load_eval_case(args.root_path, args.split, name, args.source_patch_size)
        components = predict_components_full(model, image, args.tile_size,
                                             args.overlap_stride, device, tta=not args.no_tta)
        for threshold in thresholds:
            prediction = binary_prediction(components, threshold, args.prediction_mode)
            counts = totals[threshold]
            counts[0] += int(np.logical_and(prediction, target).sum())
            counts[1] += int(np.logical_and(prediction, ~target).sum())
            counts[2] += int(np.logical_and(~prediction, target).sum())
            counts[3] += int(np.logical_and(~prediction, ~target).sum())
        print("[{}/{}] {}".format(index, len(names), name), flush=True)
    rows = []
    for threshold in thresholds:
        tp, fp, fn, tn = totals[threshold]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({"threshold": threshold, "mode": args.prediction_mode,
                     "tta": not args.no_tta, "iou": tp / max(tp + fp + fn, 1),
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
