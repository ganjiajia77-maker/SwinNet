import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


def parse_thresholds(value):
    if value:
        return [float(item) for item in value.split(",")]
    return [0.10, 0.15, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50]


def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return iou, f1, precision, recall


def overlap_logits(model, image, tile_size, stride, device):
    _, _, height, width = image.shape
    positions = RoadSkeletonDataset.sliding_positions
    logit_canvas = torch.zeros((1, 1, height, width), device=device)
    weight_canvas = torch.zeros_like(logit_canvas)

    weight_1d = torch.linspace(-1.0, 1.0, steps=tile_size, device=device).abs()
    weight_1d = (1.0 - weight_1d).clamp_min(0.1)
    tile_weight = (weight_1d[:, None] * weight_1d[None, :]).view(1, 1, tile_size, tile_size)

    for top in positions(height, tile_size, stride):
        for left in positions(width, tile_size, stride):
            tile = image[:, :, top:top + tile_size, left:left + tile_size]
            outputs = model(tile, topology_alpha_scale=0.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            tile_height, tile_width = surface_logits.shape[-2:]
            weight = tile_weight[:, :, :tile_height, :tile_width]
            logit_canvas[:, :, top:top + tile_height, left:left + tile_width] += surface_logits * weight
            weight_canvas[:, :, top:top + tile_height, left:left + tile_width] += weight

    if weight_canvas.min().item() <= 0:
        raise RuntimeError("Overlap threshold sweep left uncovered pixels in the full-image canvas.")
    return logit_canvas / weight_canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--overlap_stride", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--thresholds", type=str, default="")
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none", choices=["none", "stage3", "stage23"])
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_FULL, choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626])
    parser.add_argument(
        "--disable_msfe_skip",
        action="store_true",
        help="ablate MSFE blocks on decoder skip stages inx=2,3; auto-read from checkpoint when present",
    )
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument(
        "--highres_structure_fuse_stages",
        type=str,
        default="stage23",
        choices=["stage2", "stage3", "stage23"],
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
        args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
        args.enable_highres_structure_stream = bool(
            saved_args.get(
                "enable_highres_structure_stream",
                args.enable_highres_structure_stream,
            )
        )
        args.highres_structure_channels = int(
            saved_args.get(
                "highres_structure_channels",
                args.highres_structure_channels,
            )
        )
        args.highres_structure_fuse_stages = saved_args.get(
            "highres_structure_fuse_stages",
            args.highres_structure_fuse_stages,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model.to(device).eval()
    print_topology_coefficients(model)

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=None,
        source_patch_size=args.source_patch_size,
        return_full_image=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(
        f"Running overlap global threshold sweep on {args.split}: "
        f"cases={len(dataset)}, tile={args.img_size}, stride={args.overlap_stride}",
        flush=True,
    )
    counts = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Overlap global threshold sweep [{args.split}]"):
            image = batch["image"].to(device)
            mask = (batch["mask"].to(device) > 0.5)
            logits = overlap_logits(model, image, args.img_size, args.overlap_stride, device)
            prob = torch.sigmoid(logits)
            for threshold in thresholds:
                pred = prob >= threshold
                counts[threshold]["tp"] += int((pred & mask).sum().item())
                counts[threshold]["fp"] += int((pred & ~mask).sum().item())
                counts[threshold]["fn"] += int((~pred & mask).sum().item())

    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("Threshold", "IoU", "F1", "Precision", "Recall"))
    print("-" * 56)
    best_iou_threshold = None
    best_iou = -1.0
    for threshold in thresholds:
        item = counts[threshold]
        iou, f1, precision, recall = metrics_from_counts(item["tp"], item["fp"], item["fn"])
        if iou > best_iou:
            best_iou = iou
            best_iou_threshold = threshold
        print("{:<10.2f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(threshold, iou, f1, precision, recall))
    print("Best IoU threshold: {:.2f} -> {:.4f}".format(best_iou_threshold, best_iou))


if __name__ == "__main__":
    main()
