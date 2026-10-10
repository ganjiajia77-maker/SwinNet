"""Export complete native 1024 binary masks from 512-window DARENet inference."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data1_common import Data1RoadDataset, build_model, sliding_logits


def threshold_dir(value):
    return f"thr{round(value * 1000):03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_repo", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--progress_every", type=int, default=10)
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    if (len(thresholds) != len(args.thresholds) or
            any(not 0 < value < 1 for value in thresholds) or
            len({threshold_dir(value) for value in thresholds}) != len(thresholds)):
        parser.error("Thresholds must be unique, in (0,1), and differ by at least 0.001")
    if args.num_workers < 0 or args.progress_every < 1:
        parser.error("Invalid workers/progress interval")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for value in thresholds:
        (output / threshold_dir(value) / "surface").mkdir(parents=True)
    dataset = Data1RoadDataset(args.root_path, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    if not torch.cuda.is_available():
        parser.error("CUDA required for server inference")
    device = torch.device("cuda:0")
    model, model_file = build_model(args.original_repo)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "upstream_darenet_lmz_512_binary":
        raise ValueError("Checkpoint is not the Data1 DARENet baseline")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for image_index, batch in enumerate(loader, 1):
        probability = torch.sigmoid(sliding_logits(model, batch["image"][0], device))
        probability = probability.cpu().numpy()
        case = batch["case_id"][0]
        for threshold in thresholds:
            binary = (probability >= threshold).astype(np.uint8) * 255
            path = output / threshold_dir(threshold) / "surface" / f"{case}_pred.png"
            if not cv2.imwrite(str(path), binary):
                raise OSError(f"Failed to write {path}")
        if image_index == 1 or image_index % args.progress_every == 0 or image_index == len(dataset):
            print(f"[{args.split}] exported {image_index}/{len(dataset)} 1024 masks", flush=True)
    metadata = {"split": args.split, "checkpoint": str(Path(args.model_path).resolve()),
                "model_file": model_file, "source_size": 1024, "tile_size": 512,
                "stride": 256, "thresholds": thresholds, "cases": len(dataset),
                "merge": "taper-weighted logits -> sigmoid -> binary threshold",
                "tta": "none", "precision": "FP32"}
    (output / "inference.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
