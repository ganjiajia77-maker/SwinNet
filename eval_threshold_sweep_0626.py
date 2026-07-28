"""Threshold sweep for 0626 stage23 boundary profile (global pixel aggregation)."""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--overlap_infer", action="store_true")
    parser.add_argument("--overlap_tile_size", type=int, default=0)
    parser.add_argument("--overlap_stride", type=int, default=0)
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


def sliding_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def find_label_path(label_dir, image_name):
    base = os.path.splitext(image_name)[0]
    candidates = [
        image_name,
        base + ".png",
        base.replace("_sat", "_mask") + ".png",
        base.replace("_image", "_mask") + ".png",
        base.replace("_img", "_mask") + ".png",
    ]
    for name in candidates:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None


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


@torch.no_grad()
def collect_overlap_probs(model, args):
    model.eval()
    tile_size = args.overlap_tile_size if args.overlap_tile_size > 0 else args.source_patch_size
    stride = args.overlap_stride if args.overlap_stride > 0 else tile_size // 2
    image_dir = os.path.join(args.root_path, args.split, "image")
    label_dir = os.path.join(args.root_path, args.split, "mask")
    if not os.path.exists(label_dir):
        label_dir = os.path.join(args.root_path, args.split, "label")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(image_dir)
    if not os.path.exists(label_dir):
        raise FileNotFoundError(label_dir)

    image_list = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    prob_batches = []
    target_batches = []

    for image_name in tqdm(image_list, desc="overlap inference"):
        image_path = os.path.join(image_dir, image_name)
        label_path = find_label_path(label_dir, image_name)
        if label_path is None:
            raise FileNotFoundError(f"Cannot find label for {image_name} in {label_dir}")

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        if label is None:
            raise FileNotFoundError(label_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if label.shape[:2] != (height, width):
            label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)

        prob_canvas = np.zeros((height, width), dtype=np.float32)
        count_canvas = np.zeros((height, width), dtype=np.float32)
        for y in sliding_positions(height, tile_size, stride):
            for x in sliding_positions(width, tile_size, stride):
                y2 = min(y + tile_size, height)
                x2 = min(x + tile_size, width)
                crop = image[y:y2, x:x2]
                pad_h = tile_size - crop.shape[0]
                pad_w = tile_size - crop.shape[1]
                if pad_h > 0 or pad_w > 0:
                    crop = np.pad(
                        crop,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                if crop.shape[0] != args.img_size or crop.shape[1] != args.img_size:
                    crop = cv2.resize(
                        crop,
                        (args.img_size, args.img_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                crop = (crop.astype(np.float32) / 255.0 - mean) / std
                crop = crop.transpose(2, 0, 1)
                inputs = torch.from_numpy(crop).unsqueeze(0).cuda()
                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
                if prob.shape[0] != tile_size or prob.shape[1] != tile_size:
                    prob = cv2.resize(
                        prob,
                        (tile_size, tile_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                prob = prob[:(y2 - y), :(x2 - x)]
                prob_canvas[y:y2, x:x2] += prob
                count_canvas[y:y2, x:x2] += 1.0

        avg_prob = prob_canvas / (count_canvas + 1e-7)
        prob_batches.append(torch.from_numpy(avg_prob).unsqueeze(0).unsqueeze(0))
        target_batches.append(
            torch.from_numpy((label > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        )

    return prob_batches, target_batches


def global_metrics_from_probs(prob_batches, target_batches, threshold):
    tp = fp = fn = 0
    for probs, labels in zip(prob_batches, target_batches):
        pred = (probs >= threshold).float()
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

    if args.overlap_infer:
        logits_batches = None
        target_batches = None
        prob_batches, overlap_target_batches = collect_overlap_probs(model, args)
        sample_count = len(prob_batches)
        mode = (
            f"overlap tile={args.overlap_tile_size or args.source_patch_size}, "
            f"stride={args.overlap_stride or (args.overlap_tile_size or args.source_patch_size) // 2}, "
            f"model_img={args.img_size}"
        )
    else:
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
        prob_batches = None
        overlap_target_batches = None
        sample_count = len(dataset)
        mode = f"resize {args.source_patch_size}->{args.img_size}"

    print("=" * 80)
    print(f"SURFACE THRESHOLD SWEEP | split={args.split} | samples={sample_count} | {mode}")
    print("=" * 80)
    print(f"{'Threshold':<12}{'IoU':<12}{'F1':<12}{'Precision':<12}{'Recall':<12}")
    print("-" * 60)

    results = {}
    for threshold in thresholds:
        if args.overlap_infer:
            metrics = global_metrics_from_probs(
                prob_batches,
                overlap_target_batches,
                threshold,
            )
        else:
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
