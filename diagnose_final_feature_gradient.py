from __future__ import annotations

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_gradient_conflict import (
    build_criterion,
    build_model,
    inherit_checkpoint_args,
)
from networks.vision_transformer import load_topology_checkpoint_state
from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure gradient agreement between surface loss and final skeleton "
            "loss on the shared final feature before SkeletonGuidedHead."
        )
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--checkpoint", "--model_path", dest="checkpoint", type=str, default="")
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
    parser.add_argument("--final_connectivity_weight", type=float, default=0.0)
    parser.add_argument("--boundary_weight", type=float, default=0.0)
    parser.add_argument("--road_attention_weight", type=float, default=0.003)
    parser.add_argument("--masked_connectivity_center_experiment", action="store_true")
    parser.add_argument("--connectivity_pos_weight", type=float, default=5.0)
    parser.add_argument("--connectivity_focal_gamma", type=float, default=1.5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.0)
    parser.add_argument("--topology_alpha_scale", type=float, default=1.0)
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="Use eval mode instead of train mode for deterministic dropout/drop-path behavior.",
    )
    return parser.parse_args()


def cosine(a, b):
    denom = a.norm() * b.norm()
    if denom.item() <= 0:
        return float("nan")
    return float(torch.dot(a, b).div(denom).item())


def tensor_stats(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    flat = tensor.detach().float().reshape(-1)
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"norm={flat.norm().item():.8e} "
        f"mean_abs={flat.abs().mean().item():.8e} "
        f"max_abs={flat.abs().max().item():.8e}"
    )


def main():
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("--checkpoint/--model_path is required")
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
    if args.eval_mode:
        model.eval()
    else:
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
        pin_memory=(args.num_workers > 0),
    )

    batch = None
    for idx, item in enumerate(loader):
        if idx == args.batch_index:
            batch = item
            break
    if batch is None:
        raise RuntimeError(f"batch_index {args.batch_index} is out of range")

    swin_unet = getattr(model, "swin_unet", model)
    guided_head = getattr(swin_unet, "guided_head", None)
    if guided_head is None:
        raise RuntimeError("Could not find model.swin_unet.guided_head")

    captured = {}

    def capture_guided_head_input(_module, inputs):
        final_feature = inputs[0]
        final_feature.retain_grad()
        captured["final_feature"] = final_feature

    hook = guided_head.register_forward_pre_hook(capture_guided_head_input)
    try:
        images = batch["image"].cuda(non_blocking=True)
        masks = batch["mask"].cuda(non_blocking=True)
        skeletons = batch["skeleton"].cuda(non_blocking=True)
        skeletons_dilate = batch["skeleton_dilate"].cuda(non_blocking=True)

        outputs = model(
            images,
            gt_skeleton=skeletons,
            topology_alpha_scale=args.topology_alpha_scale,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
        )
    finally:
        hook.remove()

    if "final_feature" not in captured:
        raise RuntimeError("guided_head pre-hook did not capture a final feature")
    final_feature = captured["final_feature"]

    if not isinstance(outputs, tuple) or len(outputs) < 3:
        raise RuntimeError("Expected model outputs: surface, boundary, skeleton, ...")
    surface_logits, _boundary_logits, skeleton_logits = outputs[:3]
    if skeleton_logits is None:
        raise RuntimeError("Final skeleton logits are None; cannot measure final skeleton gradient")

    masks = criterion._match_spatial_size(masks, surface_logits)
    skeletons = criterion._match_spatial_size(skeletons, skeleton_logits)
    skeletons_dilate = criterion._match_spatial_size(skeletons_dilate, skeleton_logits)

    surface_raw, surface_bce, surface_dice = criterion.surface_loss(surface_logits, masks)
    final_ske_raw, final_ske_bce, final_ske_dice = criterion.skeleton_pixel_loss(
        skeleton_logits,
        skeletons,
        skeletons_dilate,
    )
    surface_weight = 1.0
    final_ske_weight = float(criterion.skeleton_weight)
    surface_weighted = surface_weight * surface_raw
    final_ske_weighted = final_ske_weight * final_ske_raw

    grad_surface = torch.autograd.grad(
        surface_weighted,
        final_feature,
        retain_graph=True,
        allow_unused=False,
    )[0].detach().reshape(-1)
    grad_final_ske = torch.autograd.grad(
        final_ske_weighted,
        final_feature,
        retain_graph=True,
        allow_unused=False,
    )[0].detach().reshape(-1)

    cos_value = cosine(grad_surface, grad_final_ske)
    norm_surface = grad_surface.norm().item()
    norm_final_ske = grad_final_ske.norm().item()
    norm_ratio = norm_final_ske / norm_surface if norm_surface > 0 else math.nan

    print("\nFinal Feature Gradient Diagnostic")
    print("=================================")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split={args.split} batch_index={args.batch_index} batch_size={args.batch_size}")
    print(f"model_mode={'eval' if args.eval_mode else 'train'}")
    print(f"final feature capture: FinalPatchExpand_X4 -> guided_head input")
    print(f"final_feature_shape: {tuple(final_feature.shape)}")
    print("")
    print("SkeletonGuidedHead structure")
    print(f"  class: {guided_head.__class__.__name__}")
    print(f"  enable_final_structure: {getattr(guided_head, 'enable_final_structure', None)}")
    print(
        "  final skeleton relation: parallel auxiliary head in stage23 profile; "
        "final skeleton prediction is not fed back into surface refinement."
    )
    print("")
    print("Loss values")
    print(f"  surface raw:        {surface_raw.detach().item():.8f}")
    print(f"  surface BCE/Dice:   {surface_bce.detach().item():.8f} / {surface_dice.detach().item():.8f}")
    print(f"  final_ske raw:      {final_ske_raw.detach().item():.8f}")
    print(f"  final_ske BCE/Dice: {final_ske_bce.detach().item():.8f} / {final_ske_dice.detach().item():.8f}")
    print(f"  final_ske weight:   {final_ske_weight:.8f}")
    print(f"  final_ske weighted: {final_ske_weighted.detach().item():.8f}")
    print("")
    print("Weighted gradients on final feature")
    print(f"  surface grad norm:        {norm_surface:.8e}")
    print(f"  final_ske grad norm:      {norm_final_ske:.8e}")
    print(f"  final_ske/surface norm:   {norm_ratio:.8e}")
    print(f"  cosine(surface, final_ske): {cos_value:.6f}")
    print("")
    tensor_stats("surface_grad", grad_surface)
    tensor_stats("final_ske_grad", grad_final_ske)


if __name__ == "__main__":
    main()
