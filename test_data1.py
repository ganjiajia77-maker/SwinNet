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
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap_stride", type=int, default=512)
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--out_stride", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    rows = []
    names = image_names(args.root_path, args.split)
    for index, name in enumerate(names, 1):
        image = Image.open(os.path.join(args.root_path, args.split, "image", name)).convert("RGB")
        probability = predict_full(model, image, args.tile_size, args.overlap_stride, device)
        mask = (probability >= args.threshold).astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(args.output_dir, "masks", os.path.splitext(name)[0] + ".png"))
        target = np.asarray(Image.open(mask_path(args.root_path, args.split, name)).convert("L")) >= 128
        prediction = probability >= args.threshold
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
    summary = {"threshold": args.threshold, "tp": tp, "fp": fp, "fn": fn,
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
