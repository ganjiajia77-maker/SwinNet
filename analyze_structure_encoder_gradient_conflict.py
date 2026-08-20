import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=r"D:\Code\Swin-Unet-main\data1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./analysis_out/structure_encoder_gradients")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
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
    parser.add_argument(
        "--highres_structure_fuse_stages",
        type=str,
        default="stage23",
        choices=["stage2", "stage3", "stage23"],
    )
    parser.add_argument("--highres_structure_detach_surface", action="store_true")
    parser.add_argument("--highres_structure_skeleton_weight", type=float, default=0.0)
    return parser.parse_args()


def inherit_checkpoint_args(args, checkpoint):
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        for name in (
            "structure_profile",
            "disable_msfe_skip",
            "stage_topology_stages",
            "stage_topology_alpha_max",
            "stage_topology_alpha_init",
            "stage_topology_bias_mode",
            "stage_topology_ratio",
            "stage_topology_topo_clip",
            "stage2_skeleton_gradient_ratio",
            "stage3_skeleton_gradient_ratio",
            "final_skeleton_gradient_ratio",
            "bottleneck_type",
            "enable_highres_structure_stream",
            "highres_structure_channels",
            "highres_structure_fuse_stages",
            "highres_structure_detach_surface",
            "highres_structure_skeleton_weight",
            "img_size",
            "source_patch_size",
        ):
            if name in saved_args:
                setattr(args, name, saved_args[name])
    if checkpoint.get("structure_profile"):
        args.structure_profile = checkpoint["structure_profile"]


def load_model(args, device):
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    inherit_checkpoint_args(args, checkpoint)
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        stage_topology_bias_mode=args.stage_topology_bias_mode,
        stage_topology_ratio=args.stage_topology_ratio,
        stage_topology_topo_clip=args.stage_topology_topo_clip,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_detach_surface=args.highres_structure_detach_surface,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=(args.bottleneck_type == "global_local"),
    )
    model.to(device).train()
    print_topology_coefficients(model)
    return model


def build_criterion(args, device):
    return SurfaceStructureLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        highres_structure_skeleton_weight=float(args.highres_structure_skeleton_weight),
    ).to(device)


def select_highres_skeleton_logits(stage_outputs):
    logits = []
    for item in stage_outputs:
        value = item.get("highres_structure_skeleton")
        if value is not None:
            logits.append(value)
    return logits


def highres_skeleton_raw_loss(criterion, stage_outputs, skeleton_gt, skeleton_dilate_gt):
    total = skeleton_gt.sum() * 0.0
    found = 0
    for skeleton_logits in select_highres_skeleton_logits(stage_outputs):
        skeleton_logits_full = F.interpolate(
            skeleton_logits,
            size=skeleton_gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss_skeleton, _, _ = criterion.skeleton_pixel_loss(
            skeleton_logits_full,
            skeleton_gt,
            skeleton_dilate_gt,
        )
        total = total + loss_skeleton
        found += 1
    if found == 0:
        raise RuntimeError("No highres_structure_skeleton output found.")
    return total


def structure_encoder_params(model):
    params = []
    for name, param in model.named_parameters():
        if (
            (
                "prepatch_structure_encoder" in name
                or "highres_structure_encoder" in name
            )
            and param.requires_grad
        ):
            params.append((name, param))
    if not params:
        raise RuntimeError("No trainable structure encoder parameters found.")
    return params


def zero_model_grads(model):
    for param in model.parameters():
        param.grad = None


def add_grads(accum, params):
    for name, param in params:
        if param.grad is None:
            continue
        if name not in accum:
            accum[name] = torch.zeros_like(param.detach(), device="cpu")
        accum[name].add_(param.grad.detach().cpu())


def flatten(accum, params):
    chunks = []
    for name, param in params:
        value = accum.get(name)
        if value is None:
            value = torch.zeros_like(param.detach(), device="cpu")
        chunks.append(value.reshape(-1).double())
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    criterion = build_criterion(args, device)
    params = structure_encoder_params(model)
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
    total_batches = len(loader)
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)
    surface_accum = {}
    highres_raw_accum = {}
    surface_loss_sum = 0.0
    highres_raw_loss_sum = 0.0

    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        skeleton = batch["skeleton"].to(device)
        skeleton_dilate = batch["skeleton_dilate"].to(device)

        outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
        surface_logits = outputs[0]
        stage_outputs = outputs[4] if len(outputs) > 4 else []
        loss_surface, _, _ = criterion.surface_loss(surface_logits, masks)
        loss_highres_raw = highres_skeleton_raw_loss(
            criterion,
            stage_outputs,
            skeleton,
            skeleton_dilate,
        )
        surface_loss_sum += float(loss_surface.detach().cpu())
        highres_raw_loss_sum += float(loss_highres_raw.detach().cpu())

        zero_model_grads(model)
        (loss_surface / total_batches).backward(retain_graph=True)
        add_grads(surface_accum, params)

        zero_model_grads(model)
        (loss_highres_raw / total_batches).backward()
        add_grads(highres_raw_accum, params)
        zero_model_grads(model)

        if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == total_batches:
            print(f"{batch_idx + 1}/{total_batches}", flush=True)

    surface_grad = flatten(surface_accum, params)
    highres_raw_grad = flatten(highres_raw_accum, params)
    highres_weight = float(args.highres_structure_skeleton_weight)
    highres_weighted_grad = highres_raw_grad * highres_weight

    surface_norm = float(torch.linalg.vector_norm(surface_grad).item())
    highres_raw_norm = float(torch.linalg.vector_norm(highres_raw_grad).item())
    highres_weighted_norm = float(torch.linalg.vector_norm(highres_weighted_grad).item())
    dot = float(torch.dot(surface_grad, highres_raw_grad).item())
    denom = surface_norm * highres_raw_norm
    cosine = dot / denom if denom > 0 else float("nan")
    ratio_raw = highres_raw_norm / surface_norm if surface_norm > 0 else float("nan")
    ratio_weighted = highres_weighted_norm / surface_norm if surface_norm > 0 else float("nan")

    results = {
        "split": args.split,
        "batches": total_batches,
        "samples": min(len(dataset), total_batches * args.batch_size),
        "structure_params": len(params),
        "surface_loss_mean": surface_loss_sum / max(total_batches, 1),
        "highres_skeleton_raw_loss_mean": highres_raw_loss_sum / max(total_batches, 1),
        "highres_structure_skeleton_weight": highres_weight,
        "surface_gradient_norm": surface_norm,
        "highres_skeleton_raw_gradient_norm": highres_raw_norm,
        "highres_skeleton_weighted_gradient_norm": highres_weighted_norm,
        "raw_ratio_highres_over_surface": ratio_raw,
        "weighted_ratio_highres_over_surface": ratio_weighted,
        "cosine_surface_vs_highres": cosine,
    }

    path = os.path.join(args.output_dir, "structure_encoder_gradient_conflict.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)

    print("\nStructure encoder gradient conflict")
    print(f"surface gradient norm          {surface_norm:.8e}")
    print(f"highres skeleton gradient norm {highres_raw_norm:.8e}  (raw)")
    print(f"highres skeleton gradient norm {highres_weighted_norm:.8e}  (weighted by {highres_weight:g})")
    print(f"ratio raw                      {ratio_raw:.8e}")
    print(f"ratio weighted                 {ratio_weighted:.8e}")
    print(f"cosine                         {cosine:.8f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
