"""Threshold sweep and full-resolution test for SAM-Road data1 512 training."""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data1_road_512_official_common import (
    confusion_counts, counts_to_metrics, sliding_probability,
)
from data1_road_dataset import Data1RoadDataset
from eval_data1_road import connectivity_metrics
from model import SAMRoad
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data1_road_vitb_512_official.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sam_ckpt", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--precision", choices=["16", "32"], default="16")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--connectivity_metrics", action="store_true")
    parser.add_argument("--min_component_pixels", type=int, default=5)
    parser.add_argument("--metrics_csv", default="")
    parser.add_argument("--topology_metrics", action="store_true",
                        help="Data1 full-mask connectivity and sampled raster APLS proxy")
    parser.add_argument("--topology_csv", default="")
    parser.add_argument("--short_area_threshold", type=int, default=20)
    parser.add_argument("--apls_max_nodes", type=int, default=64)
    parser.add_argument("--apls_snap_radius", type=float, default=5.0)
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    if args.split == "test" and args.threshold is None:
        parser.error("Select a threshold on val, then pass --threshold for test")
    if args.connectivity_metrics and args.threshold is None:
        parser.error("--connectivity_metrics requires a single --threshold")
    if args.metrics_csv and not args.connectivity_metrics:
        parser.error("--metrics_csv requires --connectivity_metrics")
    if args.topology_metrics and args.threshold is None:
        parser.error("--topology_metrics requires a single --threshold")
    if args.topology_csv and not args.topology_metrics:
        parser.error("--topology_csv requires --topology_metrics")
    if args.short_area_threshold < 1 or args.apls_max_nodes < 0 or args.apls_snap_radius < 0:
        parser.error("Topology thresholds must be nonnegative; short area must be positive")
    if args.topology_metrics:
        from data1_prediction_topology_metrics import image_metrics, summarize
    thresholds = (
        [args.threshold] if args.threshold is not None
        else [float(value) for value in args.thresholds.split(",")]
    )
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        parser.error("Thresholds must lie strictly between 0 and 1")

    config = load_config(args.config)
    if args.sam_ckpt:
        config.SAM_CKPT_PATH = args.sam_ckpt
    if args.workers is not None:
        config.DATA_WORKER_NUM = args.workers
    if int(config.PATCH_SIZE) != 512 or int(config.SOURCE_SIZE) != 1024:
        parser.error("This pipeline requires PATCH_SIZE=512 and SOURCE_SIZE=1024")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.precision == "16"
    model = SAMRoad(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    weights_kind = "ema" if checkpoint.get("use_ema", False) else "raw"
    print(f"Loaded {weights_kind} weights from {args.checkpoint}", flush=True)
    model.eval()

    dataset = Data1RoadDataset(args.data_root, args.split, source_size=1024)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=int(config.DATA_WORKER_NUM), pin_memory=True,
    )
    counts = {threshold: [0, 0, 0] for threshold in thresholds}
    topology_rows = []
    extended_topology_rows = []
    for batch in tqdm(loader, total=len(loader), desc=f"Evaluation {args.split}"):
        probability = sliding_probability(
            model, batch["image"], device,
            tile=int(config.EVAL_TILE), stride=int(config.EVAL_STRIDE),
            use_amp=use_amp,
        ).cpu()
        target = batch["mask"]
        for threshold in thresholds:
            for i, value in enumerate(confusion_counts(probability, target, threshold)):
                counts[threshold][i] += value
        if args.connectivity_metrics:
            _, rows = connectivity_metrics(
                [(probability, target, batch["name"][0])],
                args.threshold, args.min_component_pixels,
            )
            topology_rows.extend(rows)
        if args.topology_metrics:
            pred_mask = probability[0, 0].numpy() >= args.threshold
            gt_mask = target[0, 0].numpy() > 0.5
            extended_topology_rows.append({
                "image_id": batch["name"][0],
                **image_metrics(
                    pred_mask, gt_mask,
                    short_area_threshold=args.short_area_threshold,
                    apls_max_nodes=args.apls_max_nodes,
                    apls_snap_radius=args.apls_snap_radius,
                ),
            })

    results = {str(t): counts_to_metrics(counts[t]) for t in thresholds}
    for threshold in thresholds:
        result = results[str(threshold)]
        print(
            f"threshold={threshold:.4f} IoU={result['iou']:.6f} "
            f"F1={result['f1']:.6f} P={result['precision']:.6f} "
            f"R={result['recall']:.6f}", flush=True,
        )
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "weights": weights_kind,
        "ema_decay": checkpoint.get("ema_decay") if weights_kind == "ema" else None,
        "split": args.split,
        "source_size": 1024,
        "tile": int(config.EVAL_TILE),
        "stride": int(config.EVAL_STRIDE),
        "aggregation": "weighted logit stitching, then sigmoid and threshold",
        "thresholds": results,
    }
    if args.threshold is None:
        best_f1 = max(thresholds, key=lambda t: results[str(t)]["f1"])
        best_iou = max(thresholds, key=lambda t: results[str(t)]["iou"])
        summary["best_f1_threshold"] = best_f1
        summary["best_iou_threshold"] = best_iou
        print(f"BEST_F1_THRESHOLD={best_f1:.4f} F1={results[str(best_f1)]['f1']:.6f}")
        print(f"BEST_IOU_THRESHOLD={best_iou:.4f} IoU={results[str(best_iou)]['iou']:.6f}")

    if topology_rows:
        keys = [
            "topology_precision", "topology_recall", "clDice", "pred_components",
            "components_per_1000_gt_skeleton_pixels", "largest_component_share",
        ]
        summary["connectivity"] = {
            key: float(np.mean([row[key] for row in topology_rows])) for key in keys
        }
        print("Raster connectivity proxies:", summary["connectivity"], flush=True)
        if args.metrics_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_csv)), exist_ok=True)
            with open(args.metrics_csv, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(topology_rows[0]))
                writer.writeheader()
                writer.writerows(topology_rows)

    if extended_topology_rows:
        extended_summary = summarize(
            extended_topology_rows, args.short_area_threshold,
            args.apls_max_nodes, args.apls_snap_radius,
        )
        summary["topology"] = extended_summary
        print("Data1 topology metrics (macro image mean):", extended_summary, flush=True)
        if args.topology_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.topology_csv)), exist_ok=True)
            with open(args.topology_csv, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(extended_topology_rows[0]))
                writer.writeheader()
                writer.writerows(extended_topology_rows)

    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"Summary saved: {args.summary_json}")


if __name__ == "__main__":
    main()
