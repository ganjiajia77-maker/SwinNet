import argparse
import os
import sys

import torch
import torch.nn.functional as F
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


def apply_checkpoint_args(args, checkpoint):
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved_args, dict):
        return
    if "structure_profile" in saved_args:
        args.structure_profile = saved_args["structure_profile"]
    for name in (
        "disable_msfe_skip",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "highres_structure_fusion_mode",
        "enable_post_refine_structure_interaction",
    ):
        if name in saved_args:
            setattr(args, name, saved_args[name])


def build_model(args, checkpoint, device):
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
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    model.to(device).eval()
    return model


def resize_target(target, logits):
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    return target


def metrics_from_counts(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return iou, f1, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--crop_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
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
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none", choices=["none", "stage3", "stage23"])
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument(
        "--structure_profile",
        type=str,
        default=STRUCTURE_PROFILE_FULL,
        choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626],
    )
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23", choices=["stage2", "stage3", "stage23"])
    parser.add_argument(
        "--highres_structure_fusion_mode",
        type=str,
        default="stage23",
        choices=["stage23", "final_correction", "stage23_final_correction", "post_refine_interaction", "none"],
    )
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    apply_checkpoint_args(args, checkpoint)
    model = build_model(args, checkpoint, device)
    print(f"Using device: {device}")
    print(f"Model: {args.model_path}")
    print_topology_coefficients(model)

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.crop_list,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"{args.split.capitalize()} set size: {len(dataset)}")
    print("Metric protocol: training-style global TP/FP/FN over all pixels")

    thresholds = [0.10, 0.15, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    counts = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in thresholds}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            if not isinstance(outputs, tuple):
                raise RuntimeError("Expected tuple outputs from structure-guided model.")
            logits = outputs[0]
            masks = resize_target(masks, logits)
            masks_bin = (masks > 0.5).float()
            prob = torch.sigmoid(logits)
            for thr in thresholds:
                pred = (prob >= thr).float()
                counts[thr]["tp"] += int((pred * masks_bin).sum().item())
                counts[thr]["fp"] += int((pred * (1.0 - masks_bin)).sum().item())
                counts[thr]["fn"] += int(((1.0 - pred) * masks_bin).sum().item())

    print("\nSURFACE SEGMENTATION - TRAINING-METRIC THRESHOLD SWEEP")
    print(f"{'Threshold':<12} {'IoU':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    results = {}
    for thr in thresholds:
        row = counts[thr]
        iou, f1, precision, recall = metrics_from_counts(row["tp"], row["fp"], row["fn"])
        results[thr] = {"iou": iou, "f1": f1, "precision": precision, "recall": recall}
        print(f"{thr:<12.2f} {iou:<12.4f} {f1:<12.4f} {precision:<12.4f} {recall:<12.4f}")
    best_iou_thr = max(results, key=lambda t: results[t]["iou"])
    best_f1_thr = max(results, key=lambda t: results[t]["f1"])
    print(f"\nBest threshold (IoU): {best_iou_thr:.2f} -> IoU: {results[best_iou_thr]['iou']:.4f}")
    print(f"Best threshold (F1):  {best_f1_thr:.2f} -> F1: {results[best_f1_thr]['f1']:.4f}")


if __name__ == "__main__":
    main()
