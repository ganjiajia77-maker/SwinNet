from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_gradient_conflict import (
    build_criterion,
    cosine,
    flatten_grad,
    inherit_checkpoint_args,
    shared_parameters,
    stage_loss_components,
)
from direction_target_utils import build_continuous_direction_target
from losses.road_losses import build_connectivity_target, build_stage_skeleton_target
from topology_direction_constants import AXIAL_DIR_NAMES, axial_double_angle_basis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit direction/highres targets, stage loss dispatch, structure influence, and gradient conflict."
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
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

    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--final_topology_eta_init", type=float, default=0.0)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.0)
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
    parser.add_argument("--model_impl", type=str, default="auto", choices=("auto", "standard", "selective"))

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
    parser.add_argument("--connectivity_focal_gamma", type=float, default=1.5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.0)
    parser.add_argument("--topology_alpha_scale", type=float, default=1.0)
    parser.add_argument("--shared_param_filter", type=str, default="backbone", choices=("backbone", "all_trainable"))
    return parser.parse_args()


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def axial_class_target(direction, valid):
    basis = axial_double_angle_basis().to(device=direction.device, dtype=direction.dtype)
    vec = F.normalize(direction.float(), dim=1, eps=1e-6)
    score = torch.einsum("bchw,kc->bkhw", vec, basis)
    cls = torch.full(direction.shape[:1] + direction.shape[2:], -1, device=direction.device, dtype=torch.long)
    mask = valid.squeeze(1).bool()
    cls[mask] = score.argmax(dim=1)[mask]
    return cls


def audit_direction_targets(criterion, stage_outputs, skeleton, valid_mask):
    print("\n[1] Direction Target Audit")
    if not stage_outputs:
        print("  no stage_outputs")
        return
    for idx, item in enumerate(stage_outputs):
        direction_logits = item.get("direction")
        if direction_logits is None:
            continue
        stage = item.get("stage", idx)
        if stage not in (2, 3):
            continue
        target_size = direction_logits.shape[-2:]
        stage_skel = build_stage_skeleton_target(skeleton, target_size)
        direction_target, direction_valid = criterion.build_direction_target(stage_skel)
        direction_target = direction_target.to(device=direction_logits.device, dtype=direction_logits.dtype)
        direction_valid = direction_valid.to(device=direction_logits.device, dtype=direction_logits.dtype)
        if valid_mask is not None:
            direction_valid = direction_valid * criterion._spatial_boundary_mask(direction_logits, valid_mask)
        cls = axial_class_target(direction_target, direction_valid)
        valid = cls >= 0
        hist = torch.bincount(cls[valid].detach().cpu(), minlength=4).numpy() if valid.any() else np.zeros(4, dtype=np.int64)
        invalid_count = int((~valid).sum().item())
        zero_count = int((direction_target.norm(dim=1) <= 1e-6).sum().item())
        invalid_class0_leak = int(((cls == 0) & (~valid)).sum().item())
        shape_ok = tuple(direction_target.shape) == tuple(direction_logits.shape)
        print(
            f"  idx={idx} stage={stage} step={item.get('refinement_step', 'na')} "
            f"pred={tuple(direction_logits.shape)} target={tuple(direction_target.shape)} "
            f"shape_ok={shape_ok} built_once_from_stage_skel={tuple(stage_skel.shape[-2:]) == target_size}"
        )
        print(
            f"    valid_count={int(valid.sum().item())} invalid_count={invalid_count} "
            f"zero_direction_count={zero_count} invalid_class0_leak={invalid_class0_leak}"
        )
        print(f"    axial_hist={dict(zip(AXIAL_DIR_NAMES, hist.tolist()))}")
        if not shape_ok or invalid_class0_leak != 0:
            raise AssertionError("Direction target audit failed: shape mismatch or invalid class leak")


def audit_highres_targets(stage_outputs, skeleton):
    print("\n[2] HighRes Target Audit")
    found = False
    full_ratio = skeleton.float().mean(dim=(1, 2, 3))
    for idx, item in enumerate(stage_outputs or []):
        logits = item.get("highres_structure_skeleton")
        if logits is None:
            continue
        found = True
        target_size = logits.shape[-2:]
        target = build_stage_skeleton_target(skeleton, target_size)
        nearest = F.interpolate((skeleton > 0.5).float(), size=target_size, mode="nearest")
        ratio = target.float().mean(dim=(1, 2, 3))
        nearest_ratio = nearest.float().mean(dim=(1, 2, 3)).clamp_min(1e-8)
        inflation = ratio / nearest_ratio
        print(f"  idx={idx} pred_shape={tuple(logits.shape)} target_shape={tuple(target.shape)}")
        print("  batch positive ratios:")
        for b in range(target.shape[0]):
            print(
                f"    b={b} full256={float(full_ratio[b].item()):.6f} "
                f"target{target_size}={float(ratio[b].item()):.6f} "
                f"nearest{target_size}={float(nearest_ratio[b].item()):.6f} "
                f"maxpool/nearest={float(inflation[b].item()):.3f}"
            )
    if not found:
        print("  no highres_structure_skeleton output")


def audit_stage_dispatch(criterion, stage_outputs, skeleton, skeleton_dilate, direction_gt, valid_mask):
    print("\n[3] Stage2/3 Loss Dispatch Audit")
    _, _, _, rows = stage_loss_components(
        criterion,
        stage_outputs,
        skeleton,
        skeleton_dilate,
        direction_gt,
        valid_mask,
    )
    if not rows:
        print("  no used stage rows")
        return
    for row in rows:
        stage = row["stage"]
        if stage not in (2, 3):
            continue
        print(
            "  "
            f"idx={row['idx']} stage={stage} step_scale={row['scale']} "
            f"stage_weight={row['weight']:.6g} used={row['used']} reason={row['reason']} "
            f"raw_dir={row['raw_dir']:.6g} weighted_dir={row['weighted_dir']:.6g} "
            f"raw_con={row['raw_con']:.6g} weighted_con={row['weighted_con']:.6g} "
            f"raw_ske={row['raw_ske']:.6g} weighted_ske={row['weighted_ske']:.6g}"
        )


@contextlib.contextmanager
def structure_off(model):
    module = model.module if hasattr(model, "module") else model
    swin = getattr(module, "swin_unet", module)
    originals = {}

    def remember(name, value):
        originals[name] = value

    if hasattr(swin, "_run_decoder_structure_block"):
        remember("_run_decoder_structure_block", swin._run_decoder_structure_block)

        def zero_run(self, feature_map, stage, bottleneck_tokens, block_stage=None, **kwargs):
            target = getattr(self, "decoder_structure_blocks", None)
            block = None
            if target is not None:
                try:
                    block = target[stage if block_stage is None else block_stage]
                except Exception:
                    block = None
            if block is not None and hasattr(block, "forward"):
                try:
                    return block(
                        feature_map,
                        apply_feature_refinement=False,
                        disable_skeleton_prediction=True,
                        skeleton_prior=kwargs.get("skeleton_prior"),
                    )
                except TypeError:
                    pass
            batch, _, height, width = feature_map.shape
            dtype = feature_map.dtype
            device = feature_map.device
            return (
                feature_map,
                torch.zeros(batch, 1, height, width, device=device, dtype=dtype),
                torch.zeros(batch, 8, height, width, device=device, dtype=dtype),
                torch.zeros(batch, 2, height, width, device=device, dtype=dtype),
                torch.zeros(batch, 1, height, width, device=device, dtype=dtype),
                None,
            )

        swin._run_decoder_structure_block = types.MethodType(zero_run, swin)

    if hasattr(swin, "_apply_highres_structure_fusion"):
        remember("_apply_highres_structure_fusion", swin._apply_highres_structure_fusion)

        def no_highres_fusion(self, x, z_struct, stage, target_hw):
            return x

        swin._apply_highres_structure_fusion = types.MethodType(no_highres_fusion, swin)

    if hasattr(swin, "_apply_structure_surface_correction"):
        remember("_apply_structure_surface_correction", swin._apply_structure_surface_correction)

        def no_surface_correction(self, outputs, z_struct, structure_outputs):
            return outputs

        swin._apply_structure_surface_correction = types.MethodType(no_surface_correction, swin)

    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(swin, name, value)


def audit_structure_surface_delta(model, images, skeletons, args):
    print("\n[4] Structure -> Surface Effect Audit")
    model.eval()
    with torch.no_grad():
        on = model(
            images,
            gt_skeleton=skeletons,
            topology_alpha_scale=args.topology_alpha_scale,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
        )[0]
        with structure_off(model):
            off = model(
                images,
                gt_skeleton=skeletons,
                topology_alpha_scale=0.0,
                teacher_forcing_ratio=0.0,
            )[0]
    delta = (on - off).abs()
    base_abs_mean = off.abs().mean()
    base_std = off.std(unbiased=False)
    delta_mean = delta.mean()
    delta_p95 = torch.quantile(delta.flatten(), 0.95)
    r_delta = delta_mean / base_std.clamp_min(1e-8)
    print(
        f"  surface_on={tuple(on.shape)} surface_off={tuple(off.shape)} "
        f"surface_logit_abs_mean={float(base_abs_mean.item()):.8f} "
        f"surface_logit_std={float(base_std.item()):.8f} "
        f"delta_mean={float(delta_mean.item()):.8f} "
        f"delta_max={float(delta.max().item()):.8f} "
        f"delta_p95={float(delta_p95.item()):.8f} "
        f"R_delta={float(r_delta.item()):.8f}"
    )
    if float(delta.mean().item()) < 1e-6:
        print("  verdict=near_zero: structure branch is not measurably changing surface logits on this batch")
    else:
        print("  verdict=non_zero: structure branch changes surface logits; compare topology metrics to judge direction")


def audit_gradients(model, criterion, batch, outputs, args):
    print("\n[5] Gradient Conflict Audit")
    model.train()
    surface_logits, boundary_logits, skeleton_logits, connectivity_logits, stage_outputs = outputs[:5]
    masks = criterion._match_spatial_size(batch["mask"], surface_logits)
    skeletons = criterion._match_spatial_size(batch["skeleton"], skeleton_logits)
    skeletons_dilate = criterion._match_spatial_size(batch["skeleton_dilate"], skeleton_logits)
    direction_gt = batch.get("direction_gt")
    valid_mask = batch.get("valid_mask")
    boundary_gt = batch.get("boundary_gt")

    seg, _, _ = criterion.surface_loss(surface_logits, masks)
    final_ske_raw, _, _ = criterion.skeleton_pixel_loss(skeleton_logits, skeletons, skeletons_dilate)
    stage_ske, con, direction, _ = stage_loss_components(
        criterion,
        stage_outputs,
        skeletons,
        skeletons_dilate,
        direction_gt,
        valid_mask,
    )
    ske = criterion.skeleton_weight * final_ske_raw + stage_ske
    if boundary_gt is None:
        from losses.road_losses import build_boundary_target

        boundary_gt = build_boundary_target(masks, radius=criterion.boundary_radius)
    boundary_gt = criterion._match_spatial_size(boundary_gt, boundary_logits)
    boundary_raw, _, _ = criterion.boundary_loss(boundary_logits, boundary_gt.to(boundary_logits))
    boundary = criterion.boundary_weight * boundary_raw
    high, _ = criterion.highres_structure_skeleton_loss(stage_outputs, skeletons, skeletons_dilate)

    losses = {
        "Seg": seg,
        "Ske": ske,
        "Con": con,
        "Dir": direction,
        "High": high,
        "Boundary": boundary,
    }
    params = shared_parameters(model, args.shared_param_filter)
    print(f"  shared_param_filter={args.shared_param_filter}, tensors={len(params)}")
    for name, value in losses.items():
        print(f"  loss[{name}]={float(value.detach().item()):.8f}")
    vectors = {name: flatten_grad(loss, params, model) for name, loss in losses.items()}
    names = list(losses.keys())
    print("  cosine matrix:")
    print(" " * 12 + "".join(f"{name:>12}" for name in names))
    for left in names:
        row = [left.ljust(12)]
        for right in names:
            value = cosine(vectors[left], vectors[right])
            cell = "nan" if math.isnan(value) else f"{value:.4f}"
            row.append(f"{cell:>12}")
        print("".join(row))
    print("  focus:")
    for left, right in (("Seg", "Con"), ("Seg", "High"), ("Con", "High"), ("Dir", "Con")):
        print(f"    cos({left},{right})={cosine(vectors[left], vectors[right]):.4f}")

    highres_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "highres_structure" in name
    ]
    print("\n[5b] HighRes-Only Gradient Audit")
    print(f"  highres_param_tensors={len(highres_params)}")
    if not highres_params:
        print("  verdict=no_highres_params_found")
        return
    high_seg = flatten_grad(seg, highres_params, model)
    high_high = flatten_grad(high, highres_params, model)
    high_seg_norm = float(torch.linalg.vector_norm(high_seg).detach().item())
    high_high_norm = float(torch.linalg.vector_norm(high_high).detach().item())
    high_cos = cosine(high_seg, high_high)
    high_cos_text = "nan" if math.isnan(high_cos) else f"{high_cos:.4f}"
    print(f"  grad_norm_highres_params[Seg]={high_seg_norm:.8e}")
    print(f"  grad_norm_highres_params[High]={high_high_norm:.8e}")
    print(f"  cos_highres_params(Seg,High)={high_cos_text}")
    for loss_name, loss_value in (("Seg", seg), ("High", high)):
        model.zero_grad(set_to_none=True)
        if loss_value.requires_grad:
            loss_value.backward(retain_graph=True)
        norms = []
        for name, param in highres_params:
            if param.grad is None:
                value = 0.0
            else:
                value = float(torch.linalg.vector_norm(param.grad.detach()).item())
            if value > 0.0:
                norms.append((value, name))
        norms.sort(reverse=True)
        print(f"  top_highres_param_grads[{loss_name}]:")
        for value, name in norms[:8]:
            print(f"    {value:.8e}  {name}")
        if not norms:
            print("    none")


def audit_teacher_forcing(args, model):
    module = model.module if hasattr(model, "module") else model
    swin = getattr(module, "swin_unet", module)
    stages = getattr(swin, "stage_topology_stages", args.stage_topology_stages)
    enabled = str(stages).lower() != "none"
    print("\n[6] Stage Topology / Teacher Forcing Audit")
    print(
        f"  stage_topology_stages={stages} topology_enabled={enabled} "
        f"requested_teacher_forcing_ratio={args.teacher_forcing_ratio}"
    )
    if not enabled:
        print("  verdict=inert_schedule: topology teacher forcing schedule/logging should be treated as historical noise")


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    inherit_checkpoint_args(args, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    criterion = build_criterion(args).to(device)

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
    batch = None
    for idx, item in enumerate(loader):
        if idx == args.batch_index:
            batch = item
            break
    if batch is None:
        raise RuntimeError(f"batch_index {args.batch_index} is out of range")
    batch = to_device(batch, device)
    images = batch["image"]
    skeletons = batch["skeleton"]

    model.train()
    outputs = model(
        images,
        gt_skeleton=skeletons,
        topology_alpha_scale=args.topology_alpha_scale,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
    )
    if not isinstance(outputs, tuple) or len(outputs) < 5:
        raise RuntimeError("Expected model outputs: surface, boundary, skeleton, connectivity, stage_outputs")
    stage_outputs = outputs[4]

    print(
        f"[INFO] checkpoint={args.model_path}\n"
        f"[INFO] split={args.split}, batch_index={args.batch_index}, batch_size={args.batch_size}\n"
        f"[INFO] profile={args.structure_profile}, highres={args.enable_highres_structure_stream}, "
        f"fuse={args.highres_structure_fuse_stages}, fusion={args.highres_structure_fusion_mode}\n"
        f"[INFO] weights: stage2={args.stage2_skeleton_weight}, stage3={args.stage3_skeleton_weight}, "
        f"stage_con_factor={args.stage_connectivity_factor}, stage_dir_factor={args.stage_direction_factor}, "
        f"highres_skel={args.highres_structure_skeleton_weight}, boundary={args.boundary_weight}"
    )
    audit_direction_targets(criterion, stage_outputs, batch["skeleton"], batch.get("valid_mask"))
    audit_highres_targets(stage_outputs, batch["skeleton"])
    audit_stage_dispatch(
        criterion,
        stage_outputs,
        batch["skeleton"],
        batch["skeleton_dilate"],
        batch.get("direction_gt"),
        batch.get("valid_mask"),
    )
    audit_structure_surface_delta(model, images, skeletons, args)
    outputs_for_grad = model(
        images,
        gt_skeleton=skeletons,
        topology_alpha_scale=args.topology_alpha_scale,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
    )
    audit_gradients(model, criterion, batch, outputs_for_grad, args)
    audit_teacher_forcing(args, model)


if __name__ == "__main__":
    main()
