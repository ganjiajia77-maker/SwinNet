"""Export full-resolution binary masks for the shared road comparison tool.

This uses the same 512/256 weighted-logit stitching as the official-512
evaluator. It runs inference once per image and saves every requested
validation threshold without recomputing the model forward pass.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data1_road_512_official_common import sliding_probability
from data1_road_dataset import Data1RoadDataset
from model import SAMRoad
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/data1_road_vitb_512_official.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sam_ckpt", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--precision", choices=("16", "32"), default="16")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.split == "test" and args.threshold is None:
        parser.error("Test export requires the threshold selected on val")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    try:
        thresholds = (
            [args.threshold] if args.threshold is not None
            else [float(item.strip()) for item in args.thresholds.split(",")]
        )
    except ValueError as exc:
        parser.error(f"Invalid threshold list: {exc}")
    if not thresholds or not all(np.isfinite(t) and 0 < t < 1 for t in thresholds):
        parser.error("Every threshold must be strictly between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        parser.error("Thresholds must be unique")
    folders = {t: f"threshold_{t:.8g}" for t in thresholds}
    if len(set(folders.values())) != len(folders):
        parser.error("Threshold directory names collide; use more separated thresholds")
    if args.output_root.exists() and (
        not args.output_root.is_dir() or any(args.output_root.iterdir())
    ):
        parser.error("--output_root must be new or empty; use a fresh path for a rerun")

    config = load_config(args.config)
    if args.sam_ckpt:
        config.SAM_CKPT_PATH = args.sam_ckpt
    if (int(config.PATCH_SIZE), int(config.SOURCE_SIZE),
            int(config.EVAL_TILE), int(config.EVAL_STRIDE)) != (512, 1024, 512, 256):
        parser.error("Expected PATCH_SIZE=512, SOURCE_SIZE=1024, EVAL_TILE=512, EVAL_STRIDE=256")

    dataset = Data1RoadDataset(args.data_root, args.split, source_size=1024)
    names = [f"{Path(name).stem}_pred.png" for name in dataset.image_files]
    if len(set(names)) != len(names):
        parser.error("Duplicate prediction filenames after converting image names to PNG")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMRoad(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    weights_kind = "ema" if checkpoint.get("use_ema", False) else "raw"
    print(f"Loaded {weights_kind} weights from {args.checkpoint}", flush=True)
    model.eval()

    args.output_root.mkdir(parents=True, exist_ok=True)
    directories = {}
    for threshold, folder in folders.items():
        path = args.output_root / folder
        path.mkdir(exist_ok=False)
        directories[threshold] = path

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    for batch in tqdm(loader, total=len(loader), desc=f"Export {args.split}"):
        probability = sliding_probability(
            model, batch["image"], device,
            tile=int(config.EVAL_TILE), stride=int(config.EVAL_STRIDE),
            use_amp=device.type == "cuda" and args.precision == "16",
        )
        if tuple(probability.shape) != (1, 1, 1024, 1024):
            raise ValueError(f"Expected a 1024x1024 probability map, got {tuple(probability.shape)}")
        image = probability[0, 0].cpu().numpy()
        filename = f"{Path(batch['name'][0]).stem}_pred.png"
        for threshold, directory in directories.items():
            mask = (image >= threshold).astype(np.uint8) * 255
            destination = directory / filename
            if not cv2.imwrite(str(destination), mask):
                raise OSError(f"Could not write {destination}")

    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": weights_kind,
        "ema_decay": checkpoint.get("ema_decay") if weights_kind == "ema" else None,
        "split": args.split,
        "image_count": len(dataset),
        "source_size": 1024,
        "tile": 512,
        "stride": 256,
        "aggregation": "weighted logits / weights, then sigmoid; no TTA or postprocessing",
        "predictions": [
            {"threshold": threshold, "path": str(directory.resolve())}
            for threshold, directory in directories.items()
        ],
    }
    with (args.output_root / "export_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset)} masks per threshold to {args.output_root}", flush=True)
    print(f"Directory list: {args.output_root / 'export_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
