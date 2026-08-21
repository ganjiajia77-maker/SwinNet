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
from networks.vision_transformer_selective_fusion import (
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


REGIONS = ("WeakFN", "SkeletonTP", "HardBG")
LAYERS = (
    "surface_proj",
    "surface_branch",
    "surface_refine",
    "guided_surface_feat",
    "surface_logits",
)
EXPECTED_COUNTS = {"WeakFN": 7783, "SkeletonTP": 32204, "HardBG": 103760}
EXPECTED_FINAL = {
    "WeakFN": -0.050060,
    "SkeletonTP": 0.423948,
    "HardBG": 0.371769,
}


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_forward(model, swin, images, mode):
    swin.guided_head._diagnostic_surface_refine_alpha = 1.0
    swin._diagnostic_mode = mode
    swin.guided_head._diagnostic_trace = {}
    model(images)
    return dict(swin.guided_head._diagnostic_trace)


def collect_batches(loader, max_batches):
    batches = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batches.append(batch)
    return batches


def resize_mask(mask, spatial_size):
    if mask.shape[-2:] != spatial_size:
        mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
    return mask


def build_model(args, checkpoint):
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
        enable_post_refine_structure_interaction=(
            args.enable_post_refine_structure_interaction
        ),
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def build_fixed_masks(args, batches, model, swin):
    fixed_masks = []
    counts = {name: 0 for name in REGIONS}

    with torch.no_grad():
        for batch in tqdm(batches, desc="Build fixed masks"):
            images = batch["image"].to(args.device)
            trace = run_forward(model, swin, images, "normal")
            logits_normal = trace["surface_logits"].cpu()
            prob_normal = torch.sigmoid(logits_normal).squeeze(1)

            surface_gt = resize_mask(batch["mask"].float(), logits_normal.shape[-2:]).squeeze(1) > 0.5
            gt_skeleton = resize_mask(batch["skeleton"].float(), logits_normal.shape[-2:]).squeeze(1) > 0.5
            masks = {
                "WeakFN": gt_skeleton & (prob_normal < args.threshold),
                "SkeletonTP": gt_skeleton & (prob_normal >= args.threshold),
                "HardBG": (~surface_gt) & (prob_normal >= args.threshold),
            }
            fixed_masks.append(masks)
            for name in REGIONS:
                counts[name] += int(masks[name].sum().item())

    return fixed_masks, counts


def append_layer_sums(sums, counts, layer, region, values):
    values = values.detach().cpu().double()
    if values.numel() == 0:
        return
    sums[layer][region] += float(values.sum().item())
    counts[layer][region] += int(values.numel())


def layer_score(layer, normal_trace, zero_trace, surface_head_weight):
    delta = normal_trace[layer] - zero_trace[layer]
    if layer == "surface_logits":
        return delta.squeeze(1)
    return (delta * surface_head_weight).sum(dim=1)


def compute_directional(args, batches, model, swin, fixed_masks, surface_head_weight):
    sums = {layer: {region: 0.0 for region in REGIONS} for layer in LAYERS}
    counts = {layer: {region: 0 for region in REGIONS} for layer in LAYERS}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Directional normal-zero")):
            images = batch["image"].to(args.device)
            normal = run_forward(model, swin, images, "normal")
            zero = run_forward(model, swin, images, "zero_both")
            masks = fixed_masks[batch_idx]
            for layer in LAYERS:
                score = layer_score(layer, normal, zero, surface_head_weight)
                for region in REGIONS:
                    append_layer_sums(sums, counts, layer, region, score[masks[region]])

    means = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        for region in REGIONS:
            means[layer][region] = sums[layer][region] / counts[layer][region]
    return means, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none", choices=["none", "stage3", "stage23"])
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
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--structure_profile", type=str, default="full")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23", choices=["stage2", "stage3", "stage23"])
    parser.add_argument(
        "--highres_structure_fusion_mode",
        type=str,
        default="stage23",
        choices=[
            "stage23",
            "final_correction",
            "stage23_final_correction",
            "post_refine_interaction",
            "none",
        ],
    )
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    args = parser.parse_args()

    set_deterministic(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
        args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
        args.enable_highres_structure_stream = bool(saved_args.get("enable_highres_structure_stream", args.enable_highres_structure_stream))
        args.highres_structure_channels = int(saved_args.get("highres_structure_channels", args.highres_structure_channels))
        args.highres_structure_fuse_stages = str(saved_args.get("highres_structure_fuse_stages", args.highres_structure_fuse_stages))
        args.highres_structure_fusion_mode = str(saved_args.get("highres_structure_fusion_mode", args.highres_structure_fusion_mode))
        args.enable_post_refine_structure_interaction = bool(
            saved_args.get(
                "enable_post_refine_structure_interaction",
                args.enable_post_refine_structure_interaction,
            )
            or args.enable_post_refine_structure_interaction
        )

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batches = collect_batches(loader, args.max_batches)

    model = build_model(args, checkpoint)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    surface_head_weight = swin.guided_head.surface_head.weight.detach().cpu().view(1, -1, 1, 1)

    print(f"Checkpoint: {args.model_path}")
    print(f"split={args.split}, batches={len(batches)}, images={sum(batch['image'].shape[0] for batch in batches)}")
    print(f"threshold={args.threshold}, seed={args.seed}")
    print("delta_logit definition: logits_normal - logits_zero_both")
    print("WeakFN mask: GT skeleton & sigmoid(logits_normal_alpha1) < 0.45")
    print("SkeletonTP mask: GT skeleton & sigmoid(logits_normal_alpha1) >= 0.45")
    print("HardBG mask: surface_gt == 0 & sigmoid(logits_normal_alpha1) >= 0.45")

    fixed_masks, fixed_counts = build_fixed_masks(args, batches, model, swin)
    print("\nFixed mask pixel counts:")
    for region in REGIONS:
        ok = fixed_counts[region] == EXPECTED_COUNTS[region]
        print(f"{region}: {fixed_counts[region]} expected={EXPECTED_COUNTS[region]} match={ok}")

    means, counts = compute_directional(args, batches, model, swin, fixed_masks, surface_head_weight)

    print("\nLayer mask count consistency:")
    for layer in LAYERS:
        parts = []
        for region in REGIONS:
            parts.append(f"{region}={counts[layer][region]}")
        print(f"{layer}: " + ", ".join(parts))

    print("\n========== Directional contribution: normal - zero_both ==========")
    print(f"{'layer':<22} {'WeakFN':>12} {'SkeletonTP':>12} {'HardBG':>12}")
    for layer in LAYERS:
        print(
            f"{layer:<22} {means[layer]['WeakFN']:>12.6f} "
            f"{means[layer]['SkeletonTP']:>12.6f} {means[layer]['HardBG']:>12.6f}"
        )

    final_ok = True
    print("\nFinal surface_logits check:")
    for region in REGIONS:
        value = means["surface_logits"][region]
        expected = EXPECTED_FINAL[region]
        diff = abs(value - expected)
        ok = diff < 5e-7
        final_ok = final_ok and ok
        print(f"{region}: value={value:+.6f}, expected={expected:+.6f}, diff={diff:.12f}, match={ok}")
    if not final_ok:
        raise RuntimeError("surface_logits final check failed; directional script is not using the verified protocol.")


if __name__ == "__main__":
    main()
