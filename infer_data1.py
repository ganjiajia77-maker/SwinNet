"""Export native 1024x1024 binary Data1 masks from 512-window Swin-Unet inference."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data1_common import Data1RoadDataset, build_model, sliding_road_logits


def threshold_dir(value):
    return f"thr{round(value * 1000):03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cfg", default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--progress_every", type=int, default=10)
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    if len(thresholds) != len(args.thresholds) or any(not 0 < value < 1 for value in thresholds):
        parser.error("Provide unique thresholds strictly between zero and one")
    names = [threshold_dir(value) for value in thresholds]
    if len(set(names)) != len(names):
        parser.error("Thresholds must differ by at least 0.001")
    if args.num_workers < 0 or args.progress_every < 1:
        parser.error("num_workers must be nonnegative and progress_every positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        (output / name / "surface").mkdir(parents=True)
    dataset = Data1RoadDataset(args.root_path, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.cfg)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if checkpoint.get("architecture") != "original_swin_unet_swin_t_window8_512_two_class":
        raise ValueError("Checkpoint is not from the Data1 original Swin-Unet baseline")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for image_index, batch in enumerate(loader, 1):
        with torch.no_grad():
            logits = sliding_road_logits(model, batch["image"][0], device)
            probability = torch.sigmoid(logits).cpu().numpy()
        case = batch["case_id"][0]
        for threshold, name in zip(thresholds, names):
            binary = (probability >= threshold).astype(np.uint8) * 255
            path = output / name / "surface" / f"{case}_pred.png"
            if not cv2.imwrite(str(path), binary):
                raise OSError(f"Failed to write {path}")
        if image_index == 1 or image_index % args.progress_every == 0 or image_index == len(dataset):
            print(f"[{args.split}] exported {image_index}/{len(dataset)} native masks", flush=True)
    metadata = {"split": args.split, "checkpoint": str(Path(args.model_path).resolve()),
                "source_size": 1024, "tile_size": 512, "stride": 256,
                "merge": "taper-weighted road logit -> sigmoid -> threshold",
                "tta": "none", "postprocessing": "none", "precision": "FP32",
                "thresholds": thresholds, "cases": len(dataset)}
    (output / "inference.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
