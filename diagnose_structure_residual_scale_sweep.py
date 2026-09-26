import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.cldice_loss import soft_skeletonize
from networks.vision_transformer import STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626


K_VALUES = (0.0, 0.5, 1.0, 2.0, 4.0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inference-only sweep for decoder structure residual strength."
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target", type=str, default="both", choices=["stage2", "stage3", "both"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./analysis_out/structure_residual_scale_sweep")

    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=["auto", "standard", "selective"])
    return parser


def decoder_blocks(model):
    module = model.module if hasattr(model, "module") else model
    swin = module.swin_unet
    return {
        "stage2": swin.decoder_structure_blocks[2],
        "stage3": swin.decoder_structure_blocks[3],
    }


def set_structure_residual_scale(model, target, k):
    """Scale current gamma1 for the active decoder structure blocks."""
    targets = {"stage2", "stage3"} if target == "both" else {target}
    active = []
    for name, block in decoder_blocks(model).items():
        if hasattr(block, "raw_gamma1") and hasattr(block, "gamma_limit"):
            if not hasattr(block, "_diagnostic_raw_gamma1"):
                block._diagnostic_raw_gamma1 = block.raw_gamma1.detach().clone()
                block._diagnostic_base_gamma = float(block.gamma1.detach().cpu())
            base_raw = block._diagnostic_raw_gamma1
            base_gamma = float(block._diagnostic_base_gamma)
            gamma_limit = block.gamma_limit
            desired = base_gamma * float(k) if name in targets else base_gamma
            with torch.no_grad():
                if gamma_limit is None:
                    block.raw_gamma1.copy_(
                        torch.as_tensor(desired, device=base_raw.device, dtype=base_raw.dtype)
                    )
                else:
                    gamma_limit = float(gamma_limit)
                    desired = max(min(desired, gamma_limit * 0.999999), -gamma_limit * 0.999999)
                    raw_value = math.atanh(desired / gamma_limit) if gamma_limit > 0 else 0.0
                    block.raw_gamma1.copy_(
                        torch.as_tensor(raw_value, device=base_raw.device, dtype=base_raw.dtype)
                    )
            if name in targets:
                active.append(name)
            continue
        if hasattr(block, "structure_residual_scale"):
            block.structure_residual_scale = float(k) if name in targets else 1.0
            if name in targets:
                active.append(name)
            continue
        raise RuntimeError(
            f"{name} has neither raw_gamma1/gamma_limit nor structure_residual_scale."
        )
    return active


def enable_residual_diagnostics(model):
    for block in decoder_blocks(model).values():
        if hasattr(block, "capture_diagnostics"):
            block.capture_diagnostics = True


def add_runtime_residual_stats(stats, model):
    for name, block in decoder_blocks(model).items():
        gamma = float(block.gamma1.detach().cpu()) if hasattr(block, "gamma1") else 0.0
        diagnostics = getattr(block, "last_diagnostics", None) or {}
        residual_norm = float(diagnostics.get("gate_residual_relative_norm", 0.0))
        stats[f"gamma1_sum_{name}"] += gamma
        stats[f"residual_norm_sum_{name}"] += residual_norm
    stats["runtime_count"] += 1


def update_confusion(stats, logits, target, threshold):
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    pred = torch.sigmoid(logits) >= threshold
    truth = target > 0.5
    stats["tp"] += int((pred & truth).sum().item())
    stats["fp"] += int((pred & ~truth).sum().item())
    stats["fn"] += int((~pred & truth).sum().item())
    stats["tn"] += int((~pred & ~truth).sum().item())


def update_topology_stats(stats, logits, mask, skeleton, threshold):
    """Measure final surface topology against GT mask and skeleton."""
    if mask.shape[-2:] != logits.shape[-2:]:
        mask = F.interpolate(mask.float(), size=logits.shape[-2:], mode="nearest")
    if skeleton.shape[-2:] != logits.shape[-2:]:
        skeleton = F.interpolate(skeleton.float(), size=logits.shape[-2:], mode="nearest")

    pred_prob = torch.sigmoid(logits)
    pred = (pred_prob >= threshold).float()
    truth = (mask > 0.5).float()
    gt_skeleton = skeleton > 0.5
    pred_skeleton = pred > 0.5

    pred_skel_soft = soft_skeletonize(pred)
    gt_skel_soft = soft_skeletonize(truth)
    overlap = (pred_skel_soft * gt_skel_soft).sum(dim=(1, 2, 3))
    pred_precision = overlap / (pred_skel_soft.sum(dim=(1, 2, 3)) + 1e-6)
    gt_recall = overlap / (gt_skel_soft.sum(dim=(1, 2, 3)) + 1e-6)
    cldice = (2.0 * pred_precision * gt_recall) / (
        pred_precision + gt_recall + 1e-6
    )

    stats["cldice_sum"] += float(cldice.sum().item())
    stats["image_count"] += int(logits.shape[0])
    stats["break_pixels"] += int((gt_skeleton & ~pred_skeleton).sum().item())

    for pred_item, gt_item in zip(
        pred[:, 0].detach().cpu().numpy(),
        truth[:, 0].detach().cpu().numpy(),
    ):
        pred_bin = (pred_item > 0.5).astype(np.uint8)
        gt_bin = (gt_item > 0.5).astype(np.uint8)
        pred_count, _, _, _ = cv2.connectedComponentsWithStats(
            pred_bin, connectivity=8
        )
        gt_count, _, _, _ = cv2.connectedComponentsWithStats(
            gt_bin, connectivity=8
        )
        fp_only = (pred_bin & (1 - gt_bin)).astype(np.uint8)
        fp_count, _, _, _ = cv2.connectedComponentsWithStats(
            fp_only, connectivity=8
        )
        stats["pred_components"] += max(pred_count - 1, 0)
        stats["gt_components"] += max(gt_count - 1, 0)
        stats["false_positive_components"] += max(fp_count - 1, 0)
        stats["extra_components"] += max(pred_count - gt_count, 0)


def metrics(stats):
    eps = 1e-7
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return iou, f1, precision, recall


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    enable_residual_diagnostics(model)

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
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    baseline_logits = []
    print(f"\nStructure residual scale sweep: target={args.target}")
    print(
        "k        IoU      F1       Precision  Recall   clDice   break_px  fp_comp  "
        "extra_comp  gamma2    gamma3    resnorm2  resnorm3  dlogit    dmax     changed"
    )
    for k in K_VALUES:
        active = set_structure_residual_scale(model, args.target, k)
        if k == K_VALUES[0]:
            print("[INFO] scaled structure residual blocks: " + ", ".join(active))
        stats = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "cldice_sum": 0.0,
            "image_count": 0,
            "break_pixels": 0,
            "pred_components": 0,
            "gt_components": 0,
            "false_positive_components": 0,
            "extra_components": 0,
            "gamma1_sum_stage2": 0.0,
            "gamma1_sum_stage3": 0.0,
            "residual_norm_sum_stage2": 0.0,
            "residual_norm_sum_stage3": 0.0,
            "runtime_count": 0,
            "delta_abs_sum": 0.0,
            "delta_count": 0,
            "delta_max": 0.0,
            "changed_pixels": 0,
        }
        with torch.no_grad():
            for index, batch in enumerate(tqdm(loader, desc=f"k={k:g}")):
                if args.max_batches and index >= args.max_batches:
                    break
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                skeletons = batch["skeleton"].to(device, non_blocking=True)
                outputs = model(images)
                surface_logits = outputs[0]
                if k == K_VALUES[0]:
                    baseline_logits.append(surface_logits.detach().cpu())
                else:
                    baseline = baseline_logits[index].to(device=device, dtype=surface_logits.dtype)
                    delta = surface_logits - baseline
                    stats["delta_abs_sum"] += float(delta.abs().sum().item())
                    stats["delta_count"] += int(delta.numel())
                    stats["delta_max"] = max(
                        stats["delta_max"],
                        float(delta.abs().max().item()),
                    )
                    changed = (
                        (torch.sigmoid(surface_logits) >= args.threshold)
                        != (torch.sigmoid(baseline) >= args.threshold)
                    )
                    stats["changed_pixels"] += int(changed.sum().item())
                add_runtime_residual_stats(stats, model)
                update_confusion(stats, surface_logits, masks, args.threshold)
                update_topology_stats(stats, surface_logits, masks, skeletons, args.threshold)
        iou, f1, precision, recall = metrics(stats)
        cldice = stats["cldice_sum"] / max(stats["image_count"], 1)
        images_seen = max(stats["image_count"], 1)
        break_per_image = stats["break_pixels"] / images_seen
        fp_components_per_image = stats["false_positive_components"] / images_seen
        extra_components_per_image = stats["extra_components"] / images_seen
        runtime_count = max(stats["runtime_count"], 1)
        gamma2 = stats["gamma1_sum_stage2"] / runtime_count
        gamma3 = stats["gamma1_sum_stage3"] / runtime_count
        residual_norm2 = stats["residual_norm_sum_stage2"] / runtime_count
        residual_norm3 = stats["residual_norm_sum_stage3"] / runtime_count
        mean_abs_delta = stats["delta_abs_sum"] / max(stats["delta_count"], 1)
        rows.append(
            {
                "target": args.target,
                "k": k,
                "iou": iou,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "cldice": cldice,
                "break_pixels_per_image": break_per_image,
                "false_positive_components_per_image": fp_components_per_image,
                "extra_components_per_image": extra_components_per_image,
                "gamma1_stage2": gamma2,
                "gamma1_stage3": gamma3,
                "gate_residual_relative_norm_stage2": residual_norm2,
                "gate_residual_relative_norm_stage3": residual_norm3,
                "mean_abs_logit_delta_vs_k0": mean_abs_delta,
                "max_abs_logit_delta_vs_k0": stats["delta_max"],
                "changed_pixel_count_vs_k0": stats["changed_pixels"],
                **stats,
            }
        )
        print(
            f"{k:<8.2f} {iou:<8.4f} {f1:<8.4f} {precision:<10.4f} {recall:<8.4f} "
            f"{cldice:<8.4f} {break_per_image:<9.2f} {fp_components_per_image:<8.2f} "
            f"{extra_components_per_image:<10.2f} {gamma2:<9.5f} {gamma3:<9.5f} "
            f"{residual_norm2:<9.5f} {residual_norm3:<9.5f} {mean_abs_delta:<9.6f} "
            f"{stats['delta_max']:<8.6f} {stats['changed_pixels']:<8d}"
        )

    set_structure_residual_scale(model, args.target, 1.0)
    output_path = os.path.join(args.output_dir, f"structure_residual_scale_{args.target}.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target", "k", "iou", "f1", "precision", "recall", "cldice",
                "break_pixels_per_image", "false_positive_components_per_image",
                "extra_components_per_image", "tp", "fp", "fn", "tn", "cldice_sum",
                "image_count", "break_pixels", "pred_components", "gt_components",
                "false_positive_components", "extra_components",
                "gamma1_stage2", "gamma1_stage3",
                "gate_residual_relative_norm_stage2",
                "gate_residual_relative_norm_stage3",
                "mean_abs_logit_delta_vs_k0", "max_abs_logit_delta_vs_k0",
                "changed_pixel_count_vs_k0",
                "gamma1_sum_stage2", "gamma1_sum_stage3",
                "residual_norm_sum_stage2", "residual_norm_sum_stage3", "runtime_count",
                "delta_abs_sum", "delta_count", "delta_max", "changed_pixels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
