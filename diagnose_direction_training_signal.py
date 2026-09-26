from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import (
    SurfaceStructureLoss,
    build_boundary_target,
    build_connectivity_target,
    build_stage_skeleton_target,
)
from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", "--checkpoint", dest="model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stage", type=str, default="stage3_refine", choices=["stage2_refine", "stage3_refine"])
    parser.add_argument("--shared_scope", type=str, default="structure_branch", choices=["structure_branch", "stage_block"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
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
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--pretrained_lr", type=float, default=5e-5)
    parser.add_argument("--new_lr", type=float, default=2e-4)
    parser.add_argument("--pretrained_min_lr", type=float, default=5e-6)
    parser.add_argument("--new_min_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str, default="")
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626_final_ske")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
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
    parser.add_argument("--enable_global_topology", action="store_true")
    parser.add_argument("--global_topology_max_nodes", type=int, default=32)
    parser.add_argument("--global_topology_heads", type=int, default=4)
    parser.add_argument("--global_topology_alpha_max", type=float, default=0.05)
    parser.add_argument("--stage2_skeleton_weight", type=float, default=0.008)
    parser.add_argument("--stage3_skeleton_weight", type=float, default=0.012)
    parser.add_argument("--highres_structure_skeleton_weight", type=float, default=0.01)
    parser.add_argument("--skeleton_pos_weight", type=float, default=None)
    parser.add_argument("--stage_connectivity_factor", type=float, default=2.0)
    parser.add_argument("--stage_direction_factor", type=float, default=0.1)
    parser.add_argument("--stage_sc_s2c_weight", type=float, default=1.0)
    parser.add_argument("--stage_sc_c2s_weight", type=float, default=0.2)
    parser.add_argument("--final_skeleton_weight", type=float, default=0.10)
    parser.add_argument("--final_connectivity_weight", type=float, default=0.0)
    parser.add_argument("--boundary_weight", type=float, default=0.0)
    parser.add_argument("--road_attention_weight", type=float, default=0.0)
    parser.add_argument("--masked_connectivity_center_experiment", action="store_true")
    parser.add_argument("--connectivity_pos_weight", type=float, default=2.0)
    parser.add_argument("--directional_pos_weight_cardinal", type=float, default=1.0)
    parser.add_argument("--directional_pos_weight_diagonal", type=float, default=2.5)
    parser.add_argument("--connectivity_focal_gamma", type=float, default=1.5)
    parser.add_argument("--edge_contrastive_margin", type=float, default=0.1)
    parser.add_argument("--direction_axial_weight_ns", type=float, default=1.0)
    parser.add_argument("--direction_axial_weight_nesw", type=float, default=1.1)
    parser.add_argument("--direction_axial_weight_ew", type=float, default=1.0)
    parser.add_argument("--direction_axial_weight_senw", type=float, default=1.2)
    parser.add_argument("--topology_alpha_scale", type=float, default=1.0)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.0)
    return parser.parse_args()


def inherit_checkpoint_args(args, checkpoint):
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved_args, dict):
        return set()
    inherited = set()
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
        "enable_global_topology",
        "global_topology_max_nodes",
        "global_topology_heads",
        "global_topology_alpha_max",
        "stage2_skeleton_weight",
        "stage3_skeleton_weight",
        "highres_structure_skeleton_weight",
        "stage_connectivity_factor",
        "stage_direction_factor",
        "stage_sc_s2c_weight",
        "stage_sc_c2s_weight",
        "final_skeleton_weight",
        "final_connectivity_weight",
        "boundary_weight",
        "road_attention_weight",
        "masked_connectivity_center_experiment",
        "connectivity_pos_weight",
        "directional_pos_weight_cardinal",
        "directional_pos_weight_diagonal",
        "connectivity_focal_gamma",
        "edge_contrastive_margin",
        "direction_axial_weight_ns",
        "direction_axial_weight_nesw",
        "direction_axial_weight_ew",
        "direction_axial_weight_senw",
        "pretrained_lr",
        "new_lr",
    ):
        if name in saved_args:
            setattr(args, name, saved_args[name])
            inherited.add(name)
    if checkpoint.get("structure_profile"):
        args.structure_profile = checkpoint["structure_profile"]
        inherited.add("structure_profile")
    return inherited


def build_model(args, state_dict):
    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes,
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
        use_msfe_skip=not args.disable_msfe_skip,
        stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
        enable_global_topology=args.enable_global_topology,
        global_topology_max_nodes=args.global_topology_max_nodes,
        global_topology_heads=args.global_topology_heads,
        global_topology_alpha_max=args.global_topology_alpha_max,
    )
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if (
            key.startswith("swin_unet.global_topology.")
            and key in model_state
            and hasattr(value, "shape")
            and value.shape != model_state[key].shape
        ):
            skipped.append(key)
            continue
        filtered[key] = value
    if skipped:
        print("[WARN] Reinitialized shape-incompatible global_topology keys: " + ", ".join(skipped))
    adapt_connectivity_modules_for_checkpoint(model, filtered, "standard")
    load_topology_checkpoint_state(model, filtered, "diagnostic", strict=False)
    return model.cuda()


def build_criterion(args):
    kwargs = dict(
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
        skeleton_pos_weight=args.skeleton_pos_weight,
        use_legacy_stage_connectivity_loss=(args.structure_profile == "stage23_boundary_0626"),
        use_masked_connectivity_center_experiment=args.masked_connectivity_center_experiment,
        connectivity_pos_weight=args.connectivity_pos_weight,
        directional_pos_weight_cardinal=args.directional_pos_weight_cardinal,
        directional_pos_weight_diagonal=args.directional_pos_weight_diagonal,
        connectivity_focal_gamma=args.connectivity_focal_gamma,
        edge_contrastive_margin=args.edge_contrastive_margin,
        direction_axial_weights=(
            args.direction_axial_weight_ns,
            args.direction_axial_weight_nesw,
            args.direction_axial_weight_ew,
            args.direction_axial_weight_senw,
        ),
    )
    try:
        return SurfaceStructureLoss(**kwargs).cuda()
    except TypeError as exc:
        if "direction_axial_weights" not in str(exc):
            raise
        kwargs.pop("direction_axial_weights", None)
        return SurfaceStructureLoss(**kwargs).cuda()


def stage_key_to_index(stage):
    return 3 if stage == "stage3_refine" else 2


def selected_stage_output(stage_outputs, stage):
    target = stage_key_to_index(stage)
    for item in stage_outputs:
        if item.get("stage") == target and item.get("refinement_step") == 1:
            return item
    for item in stage_outputs:
        if item.get("stage") == target:
            return item
    raise RuntimeError(f"{stage} not found in stage outputs.")


def named_params_matching(model, substrings):
    rows = []
    for name, param in model.named_parameters():
        if any(token in name for token in substrings):
            rows.append((name, param))
    return rows


def optimizer_groups(model, checkpoint):
    loaded_names = set()
    restored = checkpoint.get("loaded_pretrained_names") if isinstance(checkpoint, dict) else None
    if restored is not None:
        loaded_names = set(restored)
    groups = {"imagenet_loaded": [], "random_or_custom": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "imagenet_loaded" if name in loaded_names else "random_or_custom"
        groups[group].append((name, parameter))
    return groups


def flatten_grad(loss, params, model):
    model.zero_grad(set_to_none=True)
    if not params:
        return torch.empty(0, device=loss.device)
    if not loss.requires_grad:
        return torch.cat([torch.zeros(param.numel(), device=param.device) for _, param in params])
    loss.backward(retain_graph=True)
    chunks = []
    for _, param in params:
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            chunks.append(param.grad.detach().reshape(-1))
    return torch.cat(chunks)


def norm(x):
    return float(x.norm().item())


def cosine(a, b):
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 0:
        return float("nan")
    return float((a @ b / denom).item())


def connectivity_valid_mask(criterion, logits, valid_mask):
    return criterion._connectivity_boundary_mask(logits, valid_mask)


def connectivity_loss_no_rank(criterion, logits, gt, skel_dilate, stage_skel):
    full = criterion.stage_connectivity_loss(
        logits,
        gt,
        skel_dilate,
        valid_mask=stage_skel,
        use_skeleton_center_mask=criterion.use_masked_connectivity_center_experiment,
        symmetry_weight=0.05 if criterion.use_masked_connectivity_center_experiment else 0.20,
    )
    valid = connectivity_valid_mask(criterion, logits, stage_skel)
    if criterion.use_masked_connectivity_center_experiment:
        valid = valid * (stage_skel > 0.5).to(dtype=valid.dtype).expand_as(valid)
    if hasattr(criterion, "_same_pixel_edge_rank_loss"):
        rank = criterion._same_pixel_edge_rank_loss(
            logits,
            gt,
            valid,
            criterion.edge_contrastive_margin,
        )
    else:
        rank = logits.sum() * 0.0
    return full - rank, rank


def direction_loss_raw(criterion, dir_logits, stage_skel, valid_mask):
    target, valid = criterion.build_direction_target(stage_skel)
    target = target.to(device=dir_logits.device, dtype=dir_logits.dtype)
    valid = valid.to(device=dir_logits.device, dtype=dir_logits.dtype)
    valid = valid * criterion._spatial_boundary_mask(dir_logits, valid_mask)
    pred = F.normalize(dir_logits, dim=1, eps=1e-6)
    cosine_value = (pred * target).sum(dim=1, keepdim=True)
    if hasattr(criterion, "_direction_weight_map"):
        weights = criterion._direction_weight_map(target)
    else:
        weights = torch.ones_like(valid)
    weighted_valid = valid * weights
    return ((1.0 - cosine_value) * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)


def main():
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    inherit_checkpoint_args(args, checkpoint)
    model = build_model(args, checkpoint["model_state_dict"])
    model.train()
    criterion = build_criterion(args)

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    batch = None
    for index, item in enumerate(loader):
        if index == args.batch_index:
            batch = item
            break
    if batch is None:
        raise RuntimeError(f"batch_index {args.batch_index} is out of range")

    images = batch["image"].cuda(non_blocking=True)
    masks = batch["mask"].cuda(non_blocking=True)
    skeletons = batch["skeleton"].cuda(non_blocking=True)
    skeletons_dilate = batch["skeleton_dilate"].cuda(non_blocking=True)
    valid_mask = batch.get("valid_mask")
    boundary_gt = batch.get("boundary_gt")
    valid_mask = valid_mask.cuda(non_blocking=True) if valid_mask is not None else None
    boundary_gt = boundary_gt.cuda(non_blocking=True) if boundary_gt is not None else None

    outputs = model(
        images,
        gt_skeleton=skeletons,
        topology_alpha_scale=args.topology_alpha_scale,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
    )
    surface_logits, boundary_logits, skeleton_logits, _connectivity_logits, stage_outputs = outputs[:5]
    masks = criterion._match_spatial_size(masks, surface_logits)
    skeleton_reference = skeleton_logits if skeleton_logits is not None else surface_logits
    skeletons = criterion._match_spatial_size(skeletons, skeleton_reference)
    skeletons_dilate = criterion._match_spatial_size(skeletons_dilate, skeleton_reference)
    stage_output = selected_stage_output(stage_outputs, args.stage)
    stage_index = int(stage_output.get("stage", stage_key_to_index(args.stage)))
    stage_weight = criterion.stage_structure_weights[stage_index] * float(stage_output.get("stage_loss_scale", 1.0))
    stage_skel = build_stage_skeleton_target(skeletons, stage_output["direction"].shape[-2:])
    stage_skel_dilate = build_stage_skeleton_target(skeletons_dilate, stage_output["direction"].shape[-2:])

    raw_seg, _, _ = criterion.surface_loss(surface_logits, masks)
    weighted_seg = raw_seg
    if skeleton_logits is not None:
        raw_final_skeleton, _, _ = criterion.skeleton_pixel_loss(
            skeleton_logits, skeletons, skeletons_dilate
        )
    else:
        raw_final_skeleton = surface_logits.sum() * 0.0
    weighted_final_skeleton = float(criterion.skeleton_weight or 0.0) * raw_final_skeleton
    if stage_output.get("skeleton") is not None:
        raw_stage_skeleton, _, _ = criterion.skeleton_pixel_loss(
            stage_output["skeleton"],
            stage_skel,
            stage_skel_dilate,
        )
        weighted_stage_skeleton = stage_weight * raw_stage_skeleton
    else:
        raw_stage_skeleton = raw_final_skeleton * 0.0
        weighted_stage_skeleton = raw_final_skeleton * 0.0
    raw_skeleton = raw_final_skeleton + raw_stage_skeleton
    weighted_skeleton = weighted_final_skeleton + weighted_stage_skeleton

    con_gt = build_connectivity_target(stage_skel).to(
        device=stage_output["connectivity"].device,
        dtype=stage_output["connectivity"].dtype,
    )
    raw_connectivity, raw_rank = connectivity_loss_no_rank(
        criterion,
        stage_output["connectivity"],
        con_gt,
        stage_skel_dilate,
        stage_skel,
    )
    weighted_connectivity = stage_weight * criterion.stage_connectivity_factor * raw_connectivity
    weighted_rank = stage_weight * criterion.stage_connectivity_factor * raw_rank
    raw_direction = direction_loss_raw(criterion, stage_output["direction"], stage_skel, valid_mask)
    weighted_direction = stage_weight * criterion.stage_direction_factor * raw_direction

    weighted_high, high_stats = criterion.highres_structure_skeleton_loss(
        stage_outputs,
        skeletons,
        skeletons_dilate,
    )
    raw_high = high_stats.get(
        "highres_structure_skeleton_raw",
        weighted_high.detach() / max(float(criterion.highres_structure_skeleton_weight), 1e-12),
    )
    raw_road_attention = criterion.road_attention_loss(stage_outputs, masks)
    weighted_road_attention = raw_road_attention
    if criterion.boundary_weight > 0 and criterion.boundary_loss is not None:
        if boundary_gt is None:
            boundary_gt = build_boundary_target(masks, radius=criterion.boundary_radius)
        boundary_gt = criterion._match_spatial_size(boundary_gt, boundary_logits)
        raw_boundary, _, _ = criterion.boundary_loss(boundary_logits, boundary_gt.to(boundary_logits.device, boundary_logits.dtype))
        weighted_boundary = criterion.boundary_weight * raw_boundary
    else:
        raw_boundary = masks.sum() * 0.0
        weighted_boundary = raw_boundary

    other_loss = (
        weighted_seg
        + weighted_skeleton
        + weighted_connectivity
        + weighted_rank
        + weighted_high
        + weighted_road_attention
        + weighted_boundary
    )
    dir_loss = weighted_direction

    stage_prefix = f"swin_unet.decoder_structure_blocks.{stage_index}."
    direction_head_params = named_params_matching(model, [stage_prefix + "direction_head."])
    if args.shared_scope == "structure_branch":
        shared_params = named_params_matching(model, [stage_prefix + "structure_branch."])
    else:
        excluded = ("direction_head.", "skeleton_head.", "connectivity_head.")
        shared_params = [
            (name, param)
            for name, param in model.named_parameters()
            if name.startswith(stage_prefix)
            and not any(token in name for token in excluded)
            and param.requires_grad
        ]

    groups = optimizer_groups(model, checkpoint)
    param_to_group = {}
    for group_name, rows in groups.items():
        for name, param in rows:
            param_to_group[id(param)] = group_name

    dir_head_in_optimizer = all(id(param) in param_to_group for _, param in direction_head_params)
    dir_head_requires_grad = all(param.requires_grad for _, param in direction_head_params) and bool(direction_head_params)
    dir_head_lrs = sorted(
        {
            args.pretrained_lr if param_to_group.get(id(param)) == "imagenet_loaded" else args.new_lr
            for _, param in direction_head_params
            if id(param) in param_to_group
        }
    )

    g_dir_shared = flatten_grad(dir_loss, shared_params, model)
    g_other_shared = flatten_grad(other_loss, shared_params, model)
    g_dir_head = flatten_grad(dir_loss, direction_head_params, model)

    dir_norm = norm(g_dir_shared)
    other_norm = norm(g_other_shared)
    ratio = dir_norm / other_norm if other_norm > 0 else float("nan")
    cos_value = cosine(g_dir_shared, g_other_shared)

    print("\nDirection Training Signal Diagnostic")
    print("====================================")
    print(f"checkpoint: {args.model_path}")
    print(f"split={args.split} batch_index={args.batch_index} batch_size={args.batch_size}")
    print(f"stage={args.stage} stage_weight={stage_weight:.8f} shared_scope={args.shared_scope}")
    print("")
    print("Direction head")
    print(f"  requires_grad: {dir_head_requires_grad}")
    print(f"  params: {len(direction_head_params)} tensors")
    print(f"  included_in_optimizer: {dir_head_in_optimizer}")
    print(f"  lr: {dir_head_lrs}")
    print(f"  gradient_norm_from_weighted_L_direction: {norm(g_dir_head):.8e}")
    print("")
    print("Loss contributions")
    print(f"  raw L_seg                 {float(raw_seg.detach().item()):.8f}")
    print(f"  weighted L_seg            {float(weighted_seg.detach().item()):.8f}")
    print(f"  raw L_skeleton            {float(raw_skeleton.detach().item()):.8f}")
    print(f"  weighted L_skeleton       {float(weighted_skeleton.detach().item()):.8f}")
    print(f"  raw L_connectivity        {float(raw_connectivity.detach().item()):.8f}")
    print(f"  weighted L_connectivity   {float(weighted_connectivity.detach().item()):.8f}")
    print(f"  raw L_rank                {float(raw_rank.detach().item()):.8f}")
    print(f"  weighted L_rank           {float(weighted_rank.detach().item()):.8f}")
    print(f"  raw L_direction           {float(raw_direction.detach().item()):.8f}")
    print(f"  lambda_dir * L_direction  {float(weighted_direction.detach().item()):.8f}")
    print(f"  raw L_highres_skeleton    {float(raw_high.detach().item()):.8f}")
    print(f"  weighted L_highres_skel   {float(weighted_high.detach().item()):.8f}")
    print(f"  weighted L_road_attention {float(weighted_road_attention.detach().item()):.8f}")
    print(f"  weighted L_boundary       {float(weighted_boundary.detach().item()):.8f}")
    print("")
    print("Shared stage3_refine gradient")
    print(f"  shared tensors: {len(shared_params)}")
    print(f"  ||g_dir||                 {dir_norm:.8e}")
    print(f"  ||g_other||               {other_norm:.8e}")
    print(f"  ratio dir/other           {ratio:.8e}")
    print(f"  cosine(g_dir,g_other)     {'nan' if math.isnan(cos_value) else f'{cos_value:.8f}'}")


if __name__ == "__main__":
    main()
