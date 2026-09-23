import argparse
import csv
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
        values.append((torch.sigmoid(canvas / weights.clamp_min(1.0)).cpu(), target.cpu(),
                       batch["name"][0]))
    return values


def metrics(values, threshold):
    tp = fp = fn = 0
    for prob, target, _ in values:
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


def skeletonize(mask):
    """Morphological skeletonization, matching the Swin-Unet data1 pipeline."""
    import cv2
    binary = (mask.astype(np.uint8) * 255)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(binary) > 0:
        eroded = cv2.erode(binary, element)
        residue = cv2.subtract(binary, cv2.dilate(eroded, element))
        skeleton = cv2.bitwise_or(skeleton, residue)
        binary = eroded
    return skeleton > 0


def connectivity_metrics(values, threshold, min_component_pixels):
    """Raster topology proxies; these are not graph-based APLS/Topo scores."""
    import cv2
    rows = []
    for prob, target, name in values:
        gt = target[0, 0].numpy() > 0.5
        pred = prob[0, 0].numpy() >= threshold
        gt_skel = skeletonize(gt)
        pred_skel = skeletonize(pred)

        pred_skel_n = int(pred_skel.sum())
        gt_skel_n = int(gt_skel.sum())
        topo_p = float((pred_skel & gt).sum()) / max(pred_skel_n, 1)
        topo_r = float((gt_skel & pred).sum()) / max(gt_skel_n, 1)
        cldice = 2 * topo_p * topo_r / max(topo_p + topo_r, 1e-12)

        _, _, stats, _ = cv2.connectedComponentsWithStats(
            pred_skel.astype(np.uint8), connectivity=8
        )
        sizes = stats[1:, cv2.CC_STAT_AREA]
        sizes = sizes[sizes >= min_component_pixels]
        components = int(len(sizes))
        accepted_pixels = int(sizes.sum()) if components else 0
        largest_share = float(sizes.max() / accepted_pixels) if accepted_pixels else 0.0
        rows.append({
            "image": name, "threshold": threshold,
            "topology_precision": topo_p, "topology_recall": topo_r,
            "clDice": cldice, "gt_skeleton_pixels": gt_skel_n,
            "pred_skeleton_pixels": pred_skel_n,
            "pred_components": components,
            "components_per_1000_gt_skeleton_pixels": components * 1000.0 / max(gt_skel_n, 1),
            "largest_component_share": largest_share,
        })
    keys = ["topology_precision", "topology_recall", "clDice",
            "pred_components", "components_per_1000_gt_skeleton_pixels",
            "largest_component_share"]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/data1_road_vitb_512.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--sam_ckpt", default="")
    p.add_argument("--connectivity_metrics", action="store_true",
                   help="Also report clDice and skeleton fragmentation proxies")
    p.add_argument("--metrics_csv", default="",
                   help="Optional per-image CSV output for connectivity metrics")
    p.add_argument("--min_component_pixels", type=int, default=5,
                   help="Ignore predicted skeleton components shorter than this pixel count")
    args = p.parse_args()
    config = load_config(args.config)
    config.SAM_CKPT_PATH = args.sam_ckpt or config.SAM_CKPT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMRoad(config).to(device)
    # Checkpoints produced by this training script include NumPy metadata,
    # so PyTorch 2.6+ must load them with weights_only=False. Only use trusted files.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError(
            "Checkpoint has neither 'model_state_dict' nor 'state_dict'. "
            f"Available keys: {list(ckpt)[:20]}"
        )
    model.load_state_dict(state_dict, strict=False)
    ds = Data1RoadDataset(args.data_root, args.split, random_crop=False, augment=False)
    values = collect(model, DataLoader(ds, batch_size=1, shuffle=False), device)
    thresholds = [args.threshold] if args.threshold is not None else [float(x) for x in args.thresholds.split(",")]
    result = {t: metrics(values, t) for t in thresholds}
    for t, m in result.items():
        print(f"threshold={t:.4f} IoU={m['iou']:.6f} F1={m['f1']:.6f} "
              f"P={m['precision']:.6f} R={m['recall']:.6f}")
    if args.connectivity_metrics:
        if args.threshold is None:
            raise ValueError("Use --threshold with --connectivity_metrics; select it on val first")
        topo, rows = connectivity_metrics(values, args.threshold, args.min_component_pixels)
        print("Raster connectivity proxies (macro image mean; not graph APLS/Topo): "
              f"clDice={topo['clDice']:.6f} topoP={topo['topology_precision']:.6f} "
              f"topoR={topo['topology_recall']:.6f} "
              f"fragments/image={topo['pred_components']:.3f} "
              f"fragments/1000-GT-skel-px={topo['components_per_1000_gt_skeleton_pixels']:.3f} "
              f"largest-component-share={topo['largest_component_share']:.6f}")
        if args.metrics_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_csv)), exist_ok=True)
            with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Per-image connectivity metrics saved: {args.metrics_csv}")
    if args.threshold is None:
        best = max(result, key=lambda t: result[t]["f1"])
        print(f"BEST_F1_THRESHOLD={best:.4f} F1={result[best]['f1']:.6f}")
        best = max(result, key=lambda t: result[t]["iou"])
        print(f"BEST_IOU_THRESHOLD={best:.4f} IoU={result[best]['iou']:.6f}")


if __name__ == "__main__":
    main()
