"""Threshold sweep for 0626 stage23 boundary profile (global pixel aggregation)."""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to best.pth or latest.pth",
    )
    parser.add_argument("--root_path", default="./data1")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument(
        "--bottleneck_type",
        type=str,
        default="global_local",
        choices=("global_local", "legacy_global_local", "g2l2"),
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.10,0.15,0.20,0.22,0.24,0.25,0.26,0.28,0.30,0.32,0.35,0.40",
        help="comma-separated surface thresholds",
    )
    parser.add_argument(
        "--cfg",
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    return parser.parse_args()


def build_model(args):
    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=0.0,
        final_gap_rho_init=0.0,
        stage_topology_stages="none",
        structure_profile=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
        enable_final_graph_prop=False,
    )
    return model.cuda()


@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    logits_batches = []
    target_batches = []
    for batch in tqdm(loader, desc="inference"):
        images = batch["image"].cuda(non_blocking=True)
        labels = batch["mask"].cuda(non_blocking=True)
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        logits_batches.append(logits.detach().cpu())
        target_batches.append(labels.detach().cpu())
    return logits_batches, target_batches


def global_metrics(logits_batches, target_batches, threshold):
    tp = fp = fn = 0
    for logits, labels in zip(logits_batches, target_batches):
        pred = (torch.sigmoid(logits) >= threshold).float()
        labels = (labels > 0.5).float()
        tp += int((pred * labels).sum().item())
        fp += int((pred * (1.0 - labels)).sum().item())
        fn += int(((1.0 - pred) * labels).sum().item())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    model = build_model(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    print("[LOAD]", args.checkpoint)
    print("[LOAD] epoch:", checkpoint.get("epoch"))
    print("[LOAD] saved val_iou:", checkpoint.get("val_iou"))
    print_topology_coefficients(model, prefix="[SWEEP]")

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logits_batches, target_batches = collect_logits(model, loader)

    print("=" * 80)
    print(f"SURFACE THRESHOLD SWEEP | split={args.split} | samples={len(dataset)}")
    print("=" * 80)
    print(f"{'Threshold':<12}{'IoU':<12}{'F1':<12}{'Precision':<12}{'Recall':<12}")
    print("-" * 60)

    results = {}
    for threshold in thresholds:
        metrics = global_metrics(logits_batches, target_batches, threshold)
        results[threshold] = metrics
        print(
            f"{threshold:<12.2f}"
            f"{metrics['iou']:<12.4f}"
            f"{metrics['f1']:<12.4f}"
            f"{metrics['precision']:<12.4f}"
            f"{metrics['recall']:<12.4f}"
        )

    best_iou_t = max(results, key=lambda t: results[t]["iou"])
    best_f1_t = max(results, key=lambda t: results[t]["f1"])
    print("-" * 60)
    print(
        f"Best IoU: threshold={best_iou_t:.2f} -> "
        f"IoU={results[best_iou_t]['iou']:.4f}, F1={results[best_iou_t]['f1']:.4f}"
    )
    print(
        f"Best F1:  threshold={best_f1_t:.2f} -> "
        f"IoU={results[best_f1_t]['iou']:.4f}, F1={results[best_f1_t]['f1']:.4f}"
    )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
