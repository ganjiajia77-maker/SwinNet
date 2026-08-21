import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss
from networks.vision_transformer_selective_fusion import (
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
)


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collect_batches(loader, max_batches):
    batches = []
    for idx, batch in enumerate(loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batches.append(batch)
    return batches


def resize_like(target, reference, mode="nearest"):
    if target.shape[-2:] != reference.shape[-2:]:
        target = F.interpolate(target.float(), size=reference.shape[-2:], mode=mode)
    return target


def apply_checkpoint_args(args, checkpoint):
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved_args, dict):
        return
    for name in (
        "structure_profile",
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
    return model.to(device).train()


def select_stage_outputs(outputs, stage_name):
    if not isinstance(outputs, tuple) or not isinstance(outputs[-1], list):
        raise RuntimeError("Expected model outputs to end with structure_outputs list.")
    for item in outputs[-1]:
        if isinstance(item, dict) and item.get("stage_name") == stage_name:
            return item
    stage_index = 3 if stage_name == "stage3_refine" else 2
    fallback = None
    for item in outputs[-1]:
        if isinstance(item, dict) and item.get("stage") == stage_index:
            fallback = item
    if fallback is None:
        raise RuntimeError(f"Could not find stage output: {stage_name}")
    return fallback


def attach_con_hidden_capture(model, stage_index):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    block = swin.decoder_structure_blocks[stage_index]
    captured = {"hidden": None}

    def hook(_module, _inp, output):
        output.retain_grad()
        captured["hidden"] = output

    handle = block.structure_branch.register_forward_hook(hook)
    return captured, handle


def summarize(rows):
    keys = rows[0].keys()
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
        if isinstance(rows[0][key], (int, float))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--stage", type=str, default="stage3_refine", choices=["stage2_refine", "stage3_refine"])
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
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
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    parser.add_argument("--connectivity_weight", type=float, default=0.03)
    args = parser.parse_args()

    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    apply_checkpoint_args(args, checkpoint)

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batches = collect_batches(loader, args.max_batches)

    model = build_model(args, checkpoint, device)
    stage_index = 3 if args.stage == "stage3_refine" else 2
    captured, handle = attach_con_hidden_capture(model, stage_index)
    criterion = SurfaceStructureLoss(
        connectivity_weight=args.connectivity_weight,
        use_masked_connectivity_center_experiment=True,
        use_legacy_stage_connectivity_loss=True,
    ).to(device)

    rows = []
    for batch_idx, batch in enumerate(tqdm(batches, desc="Con hidden gradient")):
        model.zero_grad(set_to_none=True)
        captured["hidden"] = None
        images = batch["image"].to(device)
        skeleton = batch["skeleton"].to(device).float()
        skeleton_dilate = batch["skeleton_dilate"].to(device).float()

        outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
        stage_output = select_stage_outputs(outputs, args.stage)
        connectivity_logits = stage_output["connectivity"]
        connectivity_gt = resize_like(batch["connectivity_gt"].to(device).float(), connectivity_logits)
        skeleton_match = resize_like(skeleton, connectivity_logits[:, :1])
        skeleton_dilate_match = resize_like(skeleton_dilate, connectivity_logits[:, :1])
        hidden = captured["hidden"]
        if hidden is None:
            raise RuntimeError("Failed to capture con_hidden from structure_branch.")

        raw_loss = criterion.stage_connectivity_loss(
            connectivity_logits,
            connectivity_gt,
            skeleton_dilate_match,
            valid_mask=skeleton_match,
            use_skeleton_center_mask=True,
            symmetry_weight=0.05,
        )
        weighted_loss = raw_loss * float(args.connectivity_weight)
        weighted_loss.backward()
        grad = hidden.grad
        if grad is None:
            raise RuntimeError("con_hidden.grad is None after backward.")

        valid = (skeleton_match > 0.5).expand(hidden.shape[0], hidden.shape[1], *hidden.shape[-2:])
        grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu())
        grad_abs_mean = float(grad.detach().abs().mean().cpu())
        grad_valid_abs_mean = float(grad.detach()[valid].abs().mean().cpu()) if valid.any() else float("nan")
        hidden_norm = float(torch.linalg.vector_norm(hidden.detach()).cpu())
        hidden_abs_mean = float(hidden.detach().abs().mean().cpu())
        rows.append(
            {
                "raw_loss": float(raw_loss.detach().cpu()),
                "weighted_loss": float(weighted_loss.detach().cpu()),
                "grad_norm": grad_norm,
                "grad_abs_mean": grad_abs_mean,
                "grad_valid_abs_mean": grad_valid_abs_mean,
                "hidden_norm": hidden_norm,
                "hidden_abs_mean": hidden_abs_mean,
                "grad_over_hidden": grad_norm / (hidden_norm + 1e-12),
                "center_pixels": int((skeleton_match > 0.5).sum().item()),
            }
        )

    handle.remove()
    summary = summarize(rows)
    print("\nConnectivity loss gradient into con_hidden")
    print(f"Checkpoint: {args.model_path}")
    print(f"stage={args.stage}, split={args.split}, batches={len(rows)}, images={sum(b['image'].shape[0] for b in batches)}")
    print("loss definition: masked center BCE + 0.05 reciprocal symmetry")
    print(f"reported grad is for weighted_loss = raw_loss * connectivity_weight({args.connectivity_weight})")
    for key in (
        "raw_loss",
        "weighted_loss",
        "grad_norm",
        "grad_abs_mean",
        "grad_valid_abs_mean",
        "hidden_norm",
        "hidden_abs_mean",
        "grad_over_hidden",
        "center_pixels",
    ):
        print(f"{key}: {summary[key]:.8e}")


if __name__ == "__main__":
    main()
