from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import (
    SurfaceStructureLoss,
    build_boundary_target,
    build_connectivity_target,
    build_stage_skeleton_target,
)
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet,
    load_topology_checkpoint_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pairwise gradient cosine similarity between loss terms."
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", default=2, type=int)
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

    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
                        choices=(STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626))
    parser.add_argument("--bottleneck_type", type=str, default="global_local",
                        choices=("global_local", "legacy_global_local", "g2l2"))
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")

    parser.add_argument("--stage2_skeleton_weight", type=float, default=0.008)
    parser.add_argument("--stage3_skeleton_weight", type=float, default=0.012)
    parser.add_argument("--highres_structure_skeleton_weight", type=float, default=0.008)
    parser.add_argument("--stage_connectivity_factor", type=float, default=1.0)
    parser.add_argument("--stage_direction_factor", type=float, default=0.2)
    parser.add_argument("--stage_sc_s2c_weight", type=float, default=1.0)
    parser.add_argument("--stage_sc_c2s_weight", type=float, default=0.2)
    parser.add_argument("--final_skeleton_weight", type=float, default=0.10)
    parser.add_argument("--final_connectivity_weight", type=float, default=0.03)
    parser.add_argument("--boundary_weight", type=float, default=0.01)
    parser.add_argument("--road_attention_weight", type=float, default=0.003)
    parser.add_argument("--masked_connectivity_center_experiment", action="store_true")
    parser.add_argument("--connectivity_pos_weight", type=float, default=5.0)
    parser.add_argument("--connectivity_cardinal_pos_weight", type=float, default=None)
    parser.add_argument("--connectivity_diagonal_pos_weight", type=float, default=None)
    parser.add_argument("--connectivity_focal_gamma", type=float, default=1.5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.0)
    parser.add_argument("--topology_alpha_scale", type=float, default=1.0)
    parser.add_argument(
        "--shared_param_filter",
        type=str,
        default="backbone",
        choices=("backbone", "all_trainable"),
        help="backbone excludes obvious task-specific heads before flattening gradients.",
    )
    return parser.parse_args()


def inherit_checkpoint_args(args, checkpoint):
    if not isinstance(checkpoint, dict):
        return
    saved_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
    saved_profile = checkpoint.get("structure_profile")
    if saved_profile:
        args.structure_profile = saved_profile
    for name in (
        "structure_profile",
        "bottleneck_type",
        "stage_topology_stages",
        "stage_topology_alpha_max",
        "stage_topology_alpha_init",
        "stage_topology_bias_mode",
        "stage_topology_ratio",
        "stage_topology_topo_clip",
        "stage2_skeleton_gradient_ratio",
        "stage3_skeleton_gradient_ratio",
        "final_skeleton_gradient_ratio",
        "disable_msfe_skip",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "highres_structure_fusion_mode",
        "enable_post_refine_structure_interaction",
        "stage2_skeleton_weight",
        "stage3_skeleton_weight",
        "highres_structure_skeleton_weight",
        "stage_connectivity_factor",
        "stage_direction_factor",
        "stage_sc_s2c_weight",
        "stage_sc_c2s_weight",
        "masked_connectivity_center_experiment",
        "connectivity_pos_weight",
        "connectivity_cardinal_pos_weight",
        "connectivity_diagonal_pos_weight",
        "connectivity_focal_gamma",
    ):
        if name in saved_args:
            setattr(args, name, saved_args[name])


def build_model(args):
    config = get_config(args)
    return SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=0.0,
        final_gap_rho_init=0.0,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        stage_topology_bias_mode=args.stage_topology_bias_mode,
        stage_topology_ratio=args.stage_topology_ratio,
        stage_topology_topo_clip=args.stage_topology_topo_clip,
        structure_profile=args.structure_profile,
        enable_final_graph_prop=False,
        use_msfe_skip=not args.disable_msfe_skip,
        stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
    ).cuda()


def build_criterion(args):
    return SurfaceStructureLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=args.final_skeleton_weight,
        connectivity_weight=args.final_connectivity_weight,
        connectivity_erode_kernel_size=1,
        boundary_weight=args.boundary_weight,
        boundary_radius=1,
        stage_structure_weights=(0.0, 0.0, args.stage2_skeleton_weight, args.stage3_skeleton_weight),
        road_attention_weight=args.road_attention_weight,
        stage_connectivity_factor=args.stage_connectivity_factor,
        stage_direction_factor=args.stage_direction_factor,
        stage_skeleton_connectivity_s2c_weight=args.stage_sc_s2c_weight,
        stage_skeleton_connectivity_c2s_weight=args.stage_sc_c2s_weight,
        highres_structure_skeleton_weight=args.highres_structure_skeleton_weight,
        use_legacy_stage_connectivity_loss=(args.structure_profile == STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626),
        use_masked_connectivity_center_experiment=args.masked_connectivity_center_experiment,
        connectivity_pos_weight=args.connectivity_pos_weight,
        connectivity_cardinal_pos_weight=args.connectivity_cardinal_pos_weight,
        connectivity_diagonal_pos_weight=args.connectivity_diagonal_pos_weight,
        connectivity_focal_gamma=args.connectivity_focal_gamma,
    ).cuda()


def shared_parameters(model, mode):
    excluded = (
        "boundary_head",
        "skeleton_head",
        "connectivity_head",
        "direction_head",
        "highres_structure_skeleton_head",
        "detached_skeleton_head",
        "detached_skeleton_refine",
    )
    selected = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if mode == "backbone" and any(token in name for token in excluded):
            continue
        selected.append((name, param))
    return selected


def flatten_grad(loss, params, model):
    model.zero_grad(set_to_none=True)
    if not loss.requires_grad:
        if not params:
            return torch.empty(0, device=loss.device)
        return torch.cat(
            [
                torch.zeros(param.numel(), device=param.device, dtype=param.dtype)
                for _, param in params
            ]
        )
    loss.backward(retain_graph=True)
    chunks = []
    for _, param in params:
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            chunks.append(param.grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.empty(0, device=loss.device)


def cosine(a, b):
    denom = a.norm() * b.norm()
    if denom.item() <= 0:
        return float("nan")
    return float(torch.dot(a, b).div(denom).item())


def stage_loss_components(criterion, stage_outputs, skeleton_gt, skeleton_dilate_gt, direction_gt, valid_mask):
    zero = skeleton_gt.sum() * 0.0
    stage_ske = zero
    stage_con = zero
    stage_dir = zero
    debug_rows = []
    if not stage_outputs:
        return stage_ske, stage_con, stage_dir, debug_rows

    for idx, stage_output in enumerate(stage_outputs):
        row = {
            "idx": idx,
            "stage": stage_output.get("stage", idx),
            "scale": stage_output.get("stage_loss_scale", 1.0),
            "keys": sorted(str(key) for key in stage_output.keys()),
            "used": False,
            "reason": "",
            "weight": 0.0,
            "raw_ske": float("nan"),
            "raw_con": float("nan"),
            "raw_dir": float("nan"),
            "weighted_ske": float("nan"),
            "weighted_con": float("nan"),
            "weighted_dir": float("nan"),
        }
        if (
            "skeleton" not in stage_output
            and "connectivity" not in stage_output
            and "direction" not in stage_output
        ):
            row["reason"] = "missing skeleton/connectivity/direction"
            debug_rows.append(row)
            continue
        try:
            stage_index = int(stage_output.get("stage", idx))
        except (TypeError, ValueError):
            row["reason"] = "non-integer stage"
            debug_rows.append(row)
            continue
        if stage_index >= len(criterion.stage_structure_weights):
            row["reason"] = "stage index out of weight range"
            debug_rows.append(row)
            continue
        stage_weight = (
            criterion.stage_structure_weights[stage_index]
            * float(stage_output.get("stage_loss_scale", 1.0))
        )
        row["weight"] = float(stage_weight)
        if stage_weight <= 0:
            row["reason"] = "stage weight <= 0"
            debug_rows.append(row)
            continue

        skel_logits = stage_output.get("skeleton")
        con_logits = stage_output.get("connectivity")
        dir_logits = stage_output.get("direction")
        reference_logits = (
            skel_logits
            if skel_logits is not None
            else con_logits
            if con_logits is not None
            else dir_logits
        )
        target_size = reference_logits.shape[-2:]
        stage_skel = build_stage_skeleton_target(skeleton_gt, target_size)
        stage_skel_dilate = build_stage_skeleton_target(skeleton_dilate_gt, target_size)

        if skel_logits is not None:
            raw_ske, _, _ = criterion.skeleton_pixel_loss(skel_logits, stage_skel, stage_skel_dilate)
            stage_ske = stage_ske + stage_weight * raw_ske
            row["raw_ske"] = float(raw_ske.detach().item())
            row["weighted_ske"] = float((stage_weight * raw_ske).detach().item())
        else:
            raw_ske = reference_logits.sum() * 0.0
            row["raw_ske"] = 0.0
            row["weighted_ske"] = 0.0

        if con_logits is not None:
            con_gt = build_connectivity_target(stage_skel).to(device=con_logits.device, dtype=con_logits.dtype)
            raw_con = criterion.stage_connectivity_loss(
                con_logits,
                con_gt,
                stage_skel_dilate,
                valid_mask=stage_skel,
                use_skeleton_center_mask=criterion.use_masked_connectivity_center_experiment,
                symmetry_weight=0.05 if criterion.use_masked_connectivity_center_experiment else 0.20,
            )
            stage_con = stage_con + stage_weight * criterion.stage_connectivity_factor * raw_con
            row["raw_con"] = float(raw_con.detach().item())
            row["weighted_con"] = float(
                (stage_weight * criterion.stage_connectivity_factor * raw_con).detach().item()
            )
        else:
            row["raw_con"] = 0.0
            row["weighted_con"] = 0.0

        if dir_logits is None or criterion.stage_direction_factor <= 0:
            row["reason"] = "direction missing or factor <= 0"
            row["used"] = True
            debug_rows.append(row)
            continue
        stage_dir_gt, stage_valid = criterion.build_direction_target(stage_skel)
        stage_dir_gt = stage_dir_gt.to(device=dir_logits.device, dtype=dir_logits.dtype)
        stage_valid = stage_valid.to(device=dir_logits.device, dtype=dir_logits.dtype)
        stage_valid = stage_valid * criterion._spatial_boundary_mask(dir_logits, valid_mask)
        dir_pred = F.normalize(dir_logits, dim=1, eps=1e-6)
        dir_cos = (dir_pred * stage_dir_gt).sum(dim=1, keepdim=True)
        raw_dir = ((1.0 - dir_cos) * stage_valid).sum() / stage_valid.sum().clamp_min(1.0)
        stage_dir = stage_dir + stage_weight * criterion.stage_direction_factor * raw_dir
        row["raw_dir"] = float(raw_dir.detach().item())
        row["weighted_dir"] = float(
            (stage_weight * criterion.stage_direction_factor * raw_dir).detach().item()
        )
        row["used"] = True
        row["reason"] = "ok"
        debug_rows.append(row)

    return stage_ske, stage_con, stage_dir, debug_rows


def main():
    args = parse_args()
    checkpoint = None
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        inherit_checkpoint_args(args, checkpoint)

    model = build_model(args)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        adapt_connectivity_modules_for_checkpoint(model, checkpoint["model_state_dict"], "standard")
        load_topology_checkpoint_state(
            model,
            checkpoint["model_state_dict"],
            checkpoint.get("topology_attention_version", "legacy-unrecorded"),
            strict=False,
        )
    model.train()
    criterion = build_criterion(args)

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
    batch = None
    for idx, item in enumerate(loader):
        if idx == args.batch_index:
            batch = item
            break
    if batch is None:
        raise RuntimeError(f"batch_index {args.batch_index} is out of range")

    images = batch["image"].cuda(non_blocking=True)
    masks = batch["mask"].cuda(non_blocking=True)
    skeletons = batch["skeleton"].cuda(non_blocking=True)
    skeletons_dilate = batch["skeleton_dilate"].cuda(non_blocking=True)
    direction_gt = batch.get("direction_gt")
    valid_mask = batch.get("valid_mask")
    boundary_gt = batch.get("boundary_gt")
    direction_gt = direction_gt.cuda(non_blocking=True) if direction_gt is not None else None
    valid_mask = valid_mask.cuda(non_blocking=True) if valid_mask is not None else None
    boundary_gt = boundary_gt.cuda(non_blocking=True) if boundary_gt is not None else None

    outputs = model(
        images,
        gt_skeleton=skeletons,
        topology_alpha_scale=args.topology_alpha_scale,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
    )
    if not isinstance(outputs, tuple) or len(outputs) < 5:
        raise RuntimeError("Expected model outputs: surface, boundary, skeleton, connectivity, stage_outputs")
    surface_logits, boundary_logits, skeleton_logits, connectivity_logits, stage_outputs = outputs[:5]

    masks = criterion._match_spatial_size(masks, surface_logits)
    skeletons = criterion._match_spatial_size(skeletons, skeleton_logits)
    skeletons_dilate = criterion._match_spatial_size(skeletons_dilate, skeleton_logits)

    seg, _, _ = criterion.surface_loss(surface_logits, masks)
    final_ske_raw, _, _ = criterion.skeleton_pixel_loss(skeleton_logits, skeletons, skeletons_dilate)
    stage_ske, con, direction, stage_debug_rows = stage_loss_components(
        criterion,
        stage_outputs,
        skeletons,
        skeletons_dilate,
        direction_gt,
        valid_mask,
    )
    ske = criterion.skeleton_weight * final_ske_raw + stage_ske

    if boundary_gt is None:
        boundary_gt = build_boundary_target(masks, radius=criterion.boundary_radius)
    boundary_gt = criterion._match_spatial_size(boundary_gt, boundary_logits)
    boundary_raw, _, _ = criterion.boundary_loss(
        boundary_logits,
        boundary_gt.to(device=boundary_logits.device, dtype=boundary_logits.dtype),
    )
    boundary = criterion.boundary_weight * boundary_raw

    high, _ = criterion.highres_structure_skeleton_loss(stage_outputs, skeletons, skeletons_dilate)
    losses = {
        "seg": seg,
        "ske": ske,
        "high": high,
        "dir": direction,
        "con": con,
        "boundary": boundary,
    }

    params = shared_parameters(model, args.shared_param_filter)
    print(
        "[INFO] architecture: "
        f"profile={args.structure_profile}, disable_msfe_skip={args.disable_msfe_skip}, "
        f"highres={args.enable_highres_structure_stream}, fuse={args.highres_structure_fuse_stages}, "
        f"fusion={args.highres_structure_fusion_mode}"
    )
    print(f"[INFO] split={args.split}, batch_index={args.batch_index}, batch_size={args.batch_size}")
    print(f"[INFO] shared_param_filter={args.shared_param_filter}, tensors={len(params)}")
    print(
        "[INFO] loss weights: "
        f"stage_weights={criterion.stage_structure_weights}, "
        f"con_factor={criterion.stage_connectivity_factor}, "
        f"dir_factor={criterion.stage_direction_factor}, "
        f"masked_center={criterion.use_masked_connectivity_center_experiment}, "
        f"high_weight={criterion.highres_structure_skeleton_weight}"
    )
    print("[INFO] stage outputs:")
    if not stage_outputs:
        print("  none")
    else:
        for idx, stage_output in enumerate(stage_outputs):
            keys = sorted(str(key) for key in stage_output.keys())
            stage = stage_output.get("stage", idx)
            scale = stage_output.get("stage_loss_scale", 1.0)
            print(f"  idx={idx}, stage={stage}, scale={scale}, keys={keys}")
    print("[INFO] stage loss contributions:")
    if not stage_debug_rows:
        print("  none")
    else:
        for row in stage_debug_rows:
            print(
                "  "
                f"idx={row['idx']} stage={row['stage']} scale={row['scale']} "
                f"weight={row['weight']:.6g} used={row['used']} reason={row['reason']} "
                f"raw_ske={row['raw_ske']:.6g} raw_con={row['raw_con']:.6g} raw_dir={row['raw_dir']:.6g} "
                f"w_ske={row['weighted_ske']:.6g} w_con={row['weighted_con']:.6g} w_dir={row['weighted_dir']:.6g}"
            )
    print("\nLoss values")
    for name, value in losses.items():
        print(f"  {name:<8} {float(value.detach().item()):.8f}")

    vectors = {name: flatten_grad(loss, params, model) for name, loss in losses.items()}
    print("\nGradient norms")
    for name, grad in vectors.items():
        print(f"  {name:<8} {float(grad.norm().item()):.8e}")

    names = list(losses.keys())
    print("\nGradient cosine matrix")
    print(" " * 12 + "".join(f"{name:>12}" for name in names))
    for left in names:
        row = [left.ljust(12)]
        for right in names:
            value = cosine(vectors[left], vectors[right])
            cell = "nan" if math.isnan(value) else f"{value:.4f}"
            row.append(f"{cell:>12}")
        print("".join(row))

    print("\nNegative pairs")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            value = cosine(vectors[left], vectors[right])
            if not math.isnan(value) and value < 0:
                print(f"  {left} <-> {right}: {value:.4f}")


if __name__ == "__main__":
    main()
