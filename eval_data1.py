"""Write full-resolution binary masks for the shared validation/test metric tool."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data1 import FullImageData1
from inference import counts, metrics, predict_full
from networks.WeavingUnet import WeavingUnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.split == "test" and args.threshold is None:
        parser.error("Test requires the validation-selected --threshold")
    if args.split == "val" and args.threshold is not None:
        parser.error("Val uses --thresholds, not --threshold")
    thresholds = ([args.threshold] if args.split == "test" else
                  [float(part) for part in args.thresholds.split(",")])
    if len(thresholds) != len(set(thresholds)) or any(not 0 <= t <= 1 for t in thresholds):
        parser.error("Thresholds must be unique and in [0, 1]")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeavingUnet(pretrained_encoder=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = FullImageData1(args.data_root, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    totals = {t: [0, 0, 0] for t in thresholds}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, (case, image, target) in enumerate(loader, 1):
            probability = predict_full(model, image.to(device), args.tile_size, args.stride)[0, 0].cpu().numpy()
            label = target[0].numpy()
            for threshold in thresholds:
                directory = (args.output_dir / f"threshold_{threshold:.12g}" / "surface"
                             if args.split == "val" else args.output_dir / "surface")
                directory.mkdir(parents=True, exist_ok=True)
                mask = np.uint8(probability >= threshold) * 255
                if not cv2.imwrite(str(directory / f"{case[0]}_pred.png"), mask):
                    raise OSError(f"Could not save prediction for {case[0]}")
                values = counts(probability, label, threshold)
                totals[threshold] = [a + b for a, b in zip(totals[threshold], values)]
            if i == 1 or i % 25 == 0 or i == len(dataset):
                print(f"{args.split}: {i}/{len(dataset)}", flush=True)
    report = {"split": args.split, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_epoch": checkpoint["epoch"], "images": len(dataset),
              "pretrained_source": str(checkpoint["args"].get("encoder_ckpt") or
                                       "torchvision EfficientNet_V2_S_Weights.IMAGENET1K_V1"),
              "tile_size": args.tile_size, "overlap_stride": args.stride,
              "fusion": "taper-weighted probabilities", "tta": False, "postprocessing": "none",
              "threshold_scores": {str(t): metrics(*totals[t]) for t in thresholds}}
    (args.output_dir / "inference_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
