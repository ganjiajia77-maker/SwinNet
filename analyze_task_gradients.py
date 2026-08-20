import argparse
import math
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
)


def checkpoint_args(checkpoint, cli_args):
    saved = dict(checkpoint.get("args") or {})
    defaults = {
        "root_path": "./data1",
        "dataset": "ImageData",
        "cfg": "./configs/swin_tiny_patch4_window7_224_lite.yaml",
        "zip": False,
        "cache_mode": "",
        "resume": "",
        "accumulation_steps": 0,
        "use_checkpoint": False,
        "amp_opt_level": "",
        "tag": "",
        "eval": False,
        "throughput": False,
        "opts": None,
        "n_class": 2,
        "num_classes": 1,
        "img_size": 256,
        "source_patch_size": 1024,
        "overlap_stride": 512,
        "direct_resize_train": True,
        "disable_msfe_skip": True,
        "bottleneck_type": "global_local",
        "structure_profile": STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
        "final_topology_eta_init": 0.0,
        "final_gap_rho_init": 0.0,
        "stage_topology_stages": "none",
        "stage_topology_alpha_max": 1.0,
        "stage_topology_alpha_init": 0.1,
        "stage_topology_bias_mode": "pairwise_skeleton",
        "stage_topology_ratio": 0.08,
        "stage_topology_topo_clip": 4.0,
        "stage2_skeleton_gradient_ratio": 0.5,
        "stage3_skeleton_gradient_ratio": 0.5,
        "final_skeleton_gradient_ratio": 0.0,
        "stage2_skeleton_weight": 0.008,
        "stage3_skeleton_weight": 0.012,
        "stage_direction_factor": 0.2,
        "stage_sc_s2c_weight": 1.0,
        "stage_sc_c2s_weight": 0.2,
        "road_attention_weight": 0.003,
        "pretrain_ckpt": "",
    }
    defaults.update(saved)
    defaults["root_path"] = cli_args.root_path or defaults["root_path"]
    defaults["batch_size"] = cli_args.batch_size
    defaults["num_workers"] = cli_args.num_workers
    return SimpleNamespace(**defaults)


def build_criterion(args):
    return SurfaceStructureLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=0.10,
        connectivity_weight=0.0,
        connectivity_erode_kernel_size=1,
        boundary_weight=0.01,
        boundary_radius=1,
        stage_structure_weights=(
            0.0,
            0.0,
            float(args.stage2_skeleton_weight),
            float(args.stage3_skeleton_weight),
        ),
        stage_connectivity_factor=0.5,
        stage_direction_factor=float(args.stage_direction_factor),
        stage_skeleton_connectivity_s2c_weight=float(args.stage_sc_s2c_weight),
        stage_skeleton_connectivity_c2s_weight=float(args.stage_sc_c2s_weight),
        road_attention_weight=float(args.road_attention_weight),
        use_legacy_stage_connectivity_loss=True,
    )


def crop_to_shape(tensor, shape):
    if tensor is None:
        return None
    return tensor[..., : shape[-2], : shape[-1]]


def shared_decoder_parameters(model):
    prefixes = (
        "swin_unet.layers_up.",
        "swin_unet.up.",
        "swin_unet.concat_back_dim.",
        "swin_unet.msfe_blocks.",
        "swin_unet.dca_fpn_blocks.",
        "swin_unet.decoder_structure_blocks.",
    )
    excluded = (
        ".skeleton_head.",
        ".connectivity_head.",
        ".direction_head.",
        ".structure_gate.",
        ".gate_branch.",
        ".feature_residual.",
        ".raw_gamma1",
    )
    params = []
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith(prefixes):
            continue
        if any(part in name for part in excluded):
            continue
        params.append(param)
        names.append(name)
    return names, params


def flatten_grads(params):
    pieces = []
    for param in params:
        if param.grad is None:
            pieces.append(torch.zeros(param.numel(), device=param.device))
        else:
            pieces.append(param.grad.detach().reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0)


def stage_component_losses(criterion, stage_outputs, batch, component):
    total = batch["mask"].sum() * 0.0
    for stage_output in stage_outputs:
        try:
            stage = int(stage_output.get("stage", -1))
        except (TypeError, ValueError):
            continue
        if stage >= len(criterion.stage_structure_weights):
            continue
        stage_weight = (
            criterion.stage_structure_weights[stage]
            * float(stage_output.get("stage_loss_scale", 1.0))
        )
        if stage_weight <= 0:
            continue

        skeleton_logits = stage_output["skeleton"]
        connectivity_logits = stage_output["connectivity"]
        target_size = skeleton_logits.shape[-2:]
        skeleton_gt = F.interpolate(batch["skeleton"], size=target_size, mode="nearest")
        skeleton_dilate = F.interpolate(
            batch["skeleton_dilate"],
            size=target_size,
            mode="nearest",
        )

        if component == "skeleton":
            loss, _, _ = criterion.skeleton_pixel_loss(
                skeleton_logits,
                skeleton_gt,
                skeleton_dilate,
            )
            total = total + stage_weight * loss
        elif component == "connectivity":
            conn_gt = F.interpolate(
                batch["connectivity_gt"],
                size=target_size,
                mode="nearest",
            )
            valid_mask = F.interpolate(
                batch["valid_mask"],
                size=target_size,
                mode="nearest",
            )
            loss = criterion.stage_connectivity_loss(
                connectivity_logits,
                conn_gt,
                skeleton_dilate,
                valid_mask,
            )
            total = total + stage_weight * criterion.stage_connectivity_factor * loss
        elif component == "direction":
            direction_logits = stage_output.get("direction")
            if direction_logits is None:
                continue
            direction_gt = F.interpolate(
                batch["direction_gt"],
                size=target_size,
                mode="nearest",
            )
            direction_gt = F.normalize(direction_gt, dim=1, eps=1e-6)
            valid_mask = F.interpolate(
                batch["valid_mask"],
                size=target_size,
                mode="nearest",
            )
            stage_valid = skeleton_gt * criterion._spatial_boundary_mask(
                direction_gt,
                valid_mask,
            )
            direction_pred = F.normalize(direction_logits, dim=1, eps=1e-6)
            cosine = (direction_pred * direction_gt).sum(dim=1, keepdim=True)
            loss = ((1.0 - cosine) * stage_valid).sum() / stage_valid.sum().clamp_min(1.0)
            total = total + stage_weight * criterion.stage_direction_factor * loss
        elif component == "consistency":
            loss = criterion.skeleton_connectivity_consistency_loss(
                skeleton_logits,
                connectivity_logits,
            )
            total = total + stage_weight * loss
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./model_out/20260806_002825__112__train_ablate_no_msfe_swinv2_direct256/best.pth",
    )
    parser.add_argument("--root_path", default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_index", type=int, default=0)
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args_cli.model_path, map_location="cpu", weights_only=False)
    args = checkpoint_args(checkpoint, args_cli)
    config = get_config(args)

    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
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
    ).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"[INFO] missing_keys={len(load_result.missing_keys)} unexpected_keys={len(load_result.unexpected_keys)}")
    model.train()

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="train",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        tile_size=None,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    batch = None
    for index, item in enumerate(loader):
        if index == args_cli.batch_index:
            batch = item
            break
    if batch is None:
        raise RuntimeError("batch_index is out of range")
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)

    outputs = model(
        batch["image"],
        gt_skeleton=batch["skeleton"],
        topology_alpha_scale=1.0,
        teacher_forcing_ratio=0.0,
    )
    (
        surface_logits,
        boundary_logits,
        skeleton_logits,
        connectivity_logits,
        stage_outputs,
    ) = outputs[:5]
    shape = batch["mask"].shape
    surface_logits = crop_to_shape(surface_logits, shape)
    boundary_logits = crop_to_shape(boundary_logits, shape)
    skeleton_logits = crop_to_shape(skeleton_logits, shape)
    connectivity_logits = crop_to_shape(connectivity_logits, shape)

    criterion = build_criterion(args).to(device)
    task_losses = {}
    task_losses["surface"] = criterion.surface_loss(surface_logits, batch["mask"])[0]
    task_losses["skeleton"] = stage_component_losses(criterion, stage_outputs, batch, "skeleton")
    task_losses["connectivity"] = stage_component_losses(criterion, stage_outputs, batch, "connectivity")
    task_losses["direction"] = stage_component_losses(criterion, stage_outputs, batch, "direction")
    task_losses["consistency"] = stage_component_losses(criterion, stage_outputs, batch, "consistency")
    boundary_gt = criterion._match_spatial_size(batch["boundary_gt"], boundary_logits)
    task_losses["boundary"] = criterion.boundary_weight * criterion.boundary_loss(
        boundary_logits,
        boundary_gt,
    )[0]

    names, params = shared_decoder_parameters(model)
    print(f"[INFO] shared_decoder_param_tensors={len(params)}")
    print("[INFO] shared decoder prefix examples:")
    for name in names[:8]:
        print(f"  {name}")

    grad_vectors = {}
    for task, loss in task_losses.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        vec = flatten_grads(params).float()
        grad_vectors[task] = vec.detach().cpu()
        norm = torch.linalg.vector_norm(vec).item()
        print(f"{task:12s} loss={loss.detach().item():.6f} grad_norm={norm:.6e}")

    tasks = list(task_losses)
    print("\ncosine_similarity")
    print(" " * 14 + " ".join(f"{task[:11]:>11s}" for task in tasks))
    for left in tasks:
        row = []
        for right in tasks:
            a = grad_vectors[left]
            b = grad_vectors[right]
            denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
            value = float(torch.dot(a, b) / denom) if denom.item() > 0 else math.nan
            row.append(f"{value:11.4f}")
        print(f"{left[:12]:12s}  " + " ".join(row))


if __name__ == "__main__":
    main()
