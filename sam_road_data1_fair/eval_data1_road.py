import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data1_road_dataset import Data1RoadDataset
from model import SAMRoad
from utils import load_config
from train_data1_road import road_logits


@torch.no_grad()
def collect(model, loader, device, tile=512, stride=256):
    model.eval()
    weight_1d = torch.linspace(-1.0, 1.0, steps=tile, device=device).abs()
    weight_1d = (1.0 - weight_1d).clamp_min(0.1)
    tile_weight = (weight_1d[:, None] * weight_1d[None, :]).view(1, 1, tile, tile)
    values = []
    for batch in tqdm(loader, total=len(loader), desc="Evaluation", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        _, h, w, _ = image.shape
        canvas = torch.zeros((1, 1, h, w), device=device)
        weights = torch.zeros_like(canvas)
        for top in range(0, h - tile + 1, stride):
            for left in range(0, w - tile + 1, stride):
                patch = image[:, top:top + tile, left:left + tile]
                canvas[:, :, top:top + tile, left:left + tile] += road_logits(model, patch) * tile_weight
                weights[:, :, top:top + tile, left:left + tile] += tile_weight
        values.append((torch.sigmoid(canvas / weights.clamp_min(1.0)).cpu(), target.cpu()))
    return values


def metrics(values, threshold):
    tp = fp = fn = 0
    for prob, target in values:
        pred = prob >= threshold
        gt = target > 0.5
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {"iou": tp / (tp + fp + fn + 1e-8),
            "f1": 2 * precision * recall / (precision + recall + 1e-8),
            "precision": precision, "recall": recall}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/data1_road_vitb_512.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--sam_ckpt", default="")
    args = p.parse_args()
    config = load_config(args.config)
    config.SAM_CKPT_PATH = args.sam_ckpt or config.SAM_CKPT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMRoad(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt["state_dict"]), strict=False)
    ds = Data1RoadDataset(args.data_root, args.split, random_crop=False, augment=False)
    values = collect(model, DataLoader(ds, batch_size=1, shuffle=False), device)
    thresholds = [args.threshold] if args.threshold is not None else [float(x) for x in args.thresholds.split(",")]
    result = {t: metrics(values, t) for t in thresholds}
    for t, m in result.items():
        print(f"threshold={t:.4f} IoU={m['iou']:.6f} F1={m['f1']:.6f} "
              f"P={m['precision']:.6f} R={m['recall']:.6f}")
    if args.threshold is None:
        best = max(result, key=lambda t: result[t]["f1"])
        print(f"BEST_F1_THRESHOLD={best:.4f} F1={result[best]['f1']:.6f}")
        best = max(result, key=lambda t: result[t]["iou"])
        print(f"BEST_IOU_THRESHOLD={best:.4f} IoU={result[best]['iou']:.6f}")


if __name__ == "__main__":
    main()
