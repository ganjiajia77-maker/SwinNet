import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data1_road_dataset import Data1RoadDataset
from eval_data1_road import connectivity_metrics
from model import SAMRoad
from train_data1_road import road_logits
from utils import load_config


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    values = []
    for batch in tqdm(loader, total=len(loader), desc="Evaluation"):
        images = batch["image"].to(device, non_blocking=True)
        probs = torch.sigmoid(road_logits(model, images)).cpu()
        targets = batch["mask"].cpu()
        for index, name in enumerate(batch["name"]):
            values.append((probs[index:index + 1], targets[index:index + 1], name))
    return values


def pixel_metrics(values, threshold):
    tp = fp = fn = 0
    for prob, target, _ in values:
        pred = prob >= threshold
        gt = target > 0.5
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "iou": tp / (tp + fp + fn + 1e-8),
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
        "precision": precision,
        "recall": recall,
    }


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get(
                "training_model_state_dict",
                checkpoint.get("state_dict"),
            ),
        )
    else:
        state = checkpoint
    if state is None:
        raise KeyError(
            f"Checkpoint has no model state. Available keys: "
            f"{list(checkpoint)[:20]}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"checkpoint={path} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data1_road_vitb_256_official.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam_ckpt", default="")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
    )
    parser.add_argument("--connectivity_metrics", action="store_true")
    parser.add_argument("--metrics_csv", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--min_component_pixels", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    config.SAM_CKPT_PATH = args.sam_ckpt or config.SAM_CKPT_PATH
    config.PATCH_SIZE = int(config.RESIZE_INPUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMRoad(config).to(device)
    load_checkpoint(model, args.checkpoint)

    dataset = Data1RoadDataset(
        args.data_root, args.split,
        source_size=int(config.SOURCE_SIZE),
        resize_size=int(config.RESIZE_INPUT),
        seed=1234,
    )
    loader = DataLoader(
        dataset, batch_size=int(config.INFER_BATCH_SIZE), shuffle=False,
        num_workers=int(config.DATA_WORKER_NUM), pin_memory=True,
    )
    values = collect(model, loader, device)
    if args.threshold is None:
        thresholds = [float(item) for item in args.thresholds.split(",")]
    else:
        thresholds = [float(args.threshold)]
    results = {str(t): pixel_metrics(values, t) for t in thresholds}
    for threshold, result in results.items():
        print(
            f"threshold={float(threshold):.4f} "
            f"IoU={result['iou']:.6f} F1={result['f1']:.6f} "
            f"P={result['precision']:.6f} R={result['recall']:.6f}"
        )

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "input_transform": "1024x1024 source -> 256x256",
        "thresholds": results,
    }
    if args.threshold is None:
        best_f1 = max(results, key=lambda key: results[key]["f1"])
        best_iou = max(results, key=lambda key: results[key]["iou"])
        print(
            f"BEST_F1_THRESHOLD={float(best_f1):.4f} "
            f"F1={results[best_f1]['f1']:.6f}"
        )
        print(
            f"BEST_IOU_THRESHOLD={float(best_iou):.4f} "
            f"IoU={results[best_iou]['iou']:.6f}"
        )
        summary["best_f1_threshold"] = float(best_f1)
        summary["best_iou_threshold"] = float(best_iou)

    if args.connectivity_metrics:
        if args.threshold is None:
            raise ValueError(
                "--connectivity_metrics requires --threshold; select it on val first"
            )
        connectivity, rows = connectivity_metrics(
            values, float(args.threshold), args.min_component_pixels
        )
        summary["connectivity"] = connectivity
        print(
            "Raster connectivity proxies (macro image mean; not graph APLS/Topo): "
            f"clDice={connectivity['clDice']:.6f} "
            f"topoP={connectivity['topology_precision']:.6f} "
            f"topoR={connectivity['topology_recall']:.6f} "
            f"fragments/image={connectivity['pred_components']:.3f} "
            f"fragments/1000-GT-skel-px="
            f"{connectivity['components_per_1000_gt_skeleton_pixels']:.3f} "
            f"largest-component-share="
            f"{connectivity['largest_component_share']:.6f}"
        )
        if args.metrics_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_csv)), exist_ok=True)
            with open(args.metrics_csv, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Per-image connectivity metrics saved: {args.metrics_csv}")

    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"Summary saved: {args.summary_json}")


if __name__ == "__main__":
    main()
