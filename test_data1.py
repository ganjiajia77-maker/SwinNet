import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from eval_data1_common import (binary_prediction, image_names, load_eval_case,
                               load_model, predict_components_full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap_stride", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--prediction_mode", choices=["paper_fusion", "surface"], default="paper_fusion")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--out_stride", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    rows = []
    names = image_names(args.root_path, args.split)
    for index, name in enumerate(names, 1):
        image, target = load_eval_case(args.root_path, args.split, name, args.source_patch_size)
        components = predict_components_full(model, image, args.tile_size,
                                             args.overlap_stride, device, tta=not args.no_tta)
        prediction = binary_prediction(components, args.threshold, args.prediction_mode)
        mask = prediction.astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(args.output_dir, "masks", os.path.splitext(name)[0] + ".png"))
        tp = int(np.logical_and(prediction, target).sum())
        fp = int(np.logical_and(prediction, ~target).sum())
        fn = int(np.logical_and(~prediction, target).sum())
        rows.append({"image_id": name, "tp": tp, "fp": fp, "fn": fn})
        print("[{}/{}] {}".format(index, len(names), name), flush=True)
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    summary = {"threshold": args.threshold, "mode": args.prediction_mode,
               "tta": not args.no_tta, "tp": tp, "fp": fp, "fn": fn,
               "iou": tp / max(tp + fp + fn, 1),
               "f1": 2 * precision * recall / max(precision + recall, 1e-12),
               "precision": precision, "recall": recall}
    with open(os.path.join(args.output_dir, "test_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with open(os.path.join(args.output_dir, "test_metrics_per_image.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(summary)


if __name__ == "__main__":
    main()
