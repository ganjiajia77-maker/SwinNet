import argparse
import csv
import os
import random
import sys
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_surface_refine_identity_sweep import PrePatchLiteStructureEncoder
from losses.road_losses import binary_metrics_from_logits
from networks.swin_transformer_unet_skip_expand_decoder_sys import map_to_token, token_to_map
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


REGIONS = ("WeakFN", "SkeletonTP", "HardBG")
EXPECTED_COUNTS = {"WeakFN": 7783, "SkeletonTP": 32204, "HardBG": 103760}
EXPECTED_ALPHA1 = {
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


def patch_prepatch_lite_proj_refined(model):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    swin.prepatch_structure_encoder = PrePatchLiteStructureEncoder(swin.highres_structure_channels)
    swin.highres_structure_source = "prepatch-lite-proj-refined-identity-sweep"
    swin._diagnostic_mode = "normal"
    swin.guided_head._diagnostic_trace = {}
    swin.guided_head._diagnostic_proj_refined_alpha = 1.0

    def apply_fusion(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        z_for_surface = z_struct.detach()
        if getattr(self, "_diagnostic_mode", "normal") == "zero_both":
            z_for_surface = torch.zeros_like(z_for_surface)
        feature_map = self.highres_structure_fusion[str(stage)](feature_map, z_for_surface)
        return map_to_token(feature_map)

    def mix_proj_refined(self, surface_proj, refined_feat):
        alpha = float(getattr(self, "_diagnostic_proj_refined_alpha", 1.0))
        if alpha == 1.0:
            return refined_feat
        if alpha == 0.0:
            return surface_proj
        return (1.0 - alpha) * surface_proj + alpha * refined_feat

    def guided_forward(self, x, z_struct=None):
        surface_proj = self.surface_proj(x)
        surface_branch_feat = self.surface_branch(surface_proj)
        skeleton_feat = self.skeleton_proj(x)
        z_for_surface = z_struct
        if (
            z_for_surface is not None
            and getattr(swin, "_diagnostic_mode", "normal") == "zero_both"
        ):
            z_for_surface = torch.zeros_like(z_for_surface)

        if not self.enable_final_structure:
            final_structure_feat = self.structure_branch(skeleton_feat)
            final_skeleton_logits = self.skeleton_head(final_structure_feat)
            refined_feat = self.surface_refine(surface_branch_feat)
            guided_surface_feat = mix_proj_refined(self, surface_proj, refined_feat)
            guided_surface_feat = self._apply_post_refine_structure_interaction(
                guided_surface_feat,
                z_for_surface,
            )
            boundary_feat = self.boundary_branch(guided_surface_feat)
            boundary_logits = self.boundary_head(boundary_feat)
            boundary_attn = torch.sigmoid(boundary_logits)
            boundary_correction = self.boundary_residual(guided_surface_feat)
            guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
            surface_logits = self.surface_head(guided_surface_feat)
            self._diagnostic_trace["surface_logits"] = surface_logits.detach().cpu()
            return surface_logits, boundary_logits, final_skeleton_logits, None

        structure_feat = self.structure_branch(skeleton_feat)
        seed_skeleton_logits = self.skeleton_head(structure_feat)
        seed_connectivity_logits = self.connectivity_head(structure_feat)
        structure_feat = self.final_topology_attention(
            structure_feat,
            torch.sigmoid(seed_skeleton_logits).detach(),
            torch.sigmoid(seed_connectivity_logits).detach(),
        )
        skeleton_logits = self.skeleton_head(structure_feat)
        connectivity_logits = self.connectivity_head(structure_feat)
        refined_feat = self.surface_refine(surface_branch_feat)
        guided_surface_feat = mix_proj_refined(self, surface_proj, refined_feat)
        guided_surface_feat = self._apply_post_refine_structure_interaction(
            guided_surface_feat,
            z_for_surface,
        )
        boundary_feat = self.boundary_branch(guided_surface_feat)
        boundary_logits = self.boundary_head(boundary_feat)
        boundary_attn = torch.sigmoid(boundary_logits)
        boundary_correction = self.boundary_residual(guided_surface_feat)
        guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
        surface_logits = self.surface_head(guided_surface_feat)
        self._diagnostic_trace["surface_logits"] = surface_logits.detach().cpu()
        return surface_logits, boundary_logits, skeleton_logits, connectivity_logits

    swin._apply_highres_structure_fusion = MethodType(apply_fusion, swin)
    swin.guided_head.forward = MethodType(guided_forward, swin.guided_head)


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
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
    )
    patch_prepatch_lite_proj_refined(model)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def run_forward(model, swin, images, alpha, mode):
    swin.guided_head._diagnostic_proj_refined_alpha = float(alpha)
    swin._diagnostic_mode = mode
    swin.guided_head._diagnostic_trace = {}
    model(images)
    return dict(swin.guided_head._diagnostic_trace)


def resize_mask(mask, spatial_size):
    if mask.shape[-2:] != spatial_size:
        mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
    return mask


def collect_batches(loader, max_batches):
    batches = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batches.append(batch)
    return batches


def build_fixed_masks(args, batches, model, swin):
    fixed_masks = []
    counts = {name: 0 for name in REGIONS}
    with torch.no_grad():
        for batch in tqdm(batches, desc="Build alpha=1 fixed masks"):
            images = batch["image"].to(args.device)
            trace = run_forward(model, swin, images, 1.0, "normal")
            logits = trace["surface_logits"].cpu()
            prob = torch.sigmoid(logits).squeeze(1)
            surface_gt = resize_mask(batch["mask"].float(), logits.shape[-2:]).squeeze(1) > 0.5
            skeleton = resize_mask(batch["skeleton"].float(), logits.shape[-2:]).squeeze(1) > 0.5
            masks = {
                "WeakFN": skeleton & (prob < args.threshold),
                "SkeletonTP": skeleton & (prob >= args.threshold),
                "HardBG": (~surface_gt) & (prob >= args.threshold),
            }
            fixed_masks.append(masks)
            for name in REGIONS:
                counts[name] += int(masks[name].sum().item())
    return fixed_masks, counts


def compute_metrics(logits_list, targets_list, threshold):
    metrics = {"iou": [], "f1": [], "precision": [], "recall": []}
    for logits, targets in zip(logits_list, targets_list):
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(targets.float(), size=logits.shape[-2:], mode="nearest")
        item = binary_metrics_from_logits(logits, targets, threshold=threshold)
        for key in metrics:
            metrics[key].append(item[key])
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def evaluate_alpha(args, batches, model, swin, alpha, fixed_masks):
    logits_for_metrics = []
    targets = []
    sums = {name: 0.0 for name in REGIONS}
    counts = {name: 0 for name in REGIONS}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=f"alpha={alpha:g}")):
            images = batch["image"].to(args.device)
            normal = run_forward(model, swin, images, alpha, "normal")
            zero = run_forward(model, swin, images, alpha, "zero_both")
            logits_normal = normal["surface_logits"]
            logits_zero = zero["surface_logits"]
            logits_for_metrics.append(logits_normal)
            targets.append(batch["mask"].float())
            delta = (logits_normal - logits_zero).squeeze(1)
            masks = fixed_masks[batch_idx]
            for name in REGIONS:
                values = delta[masks[name]]
                sums[name] += float(values.double().sum().item())
                counts[name] += int(values.numel())

    metrics = compute_metrics(logits_for_metrics, targets, args.threshold)
    return {
        "alpha": float(alpha),
        "iou": metrics["iou"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "weak_fn_dlogit": sums["WeakFN"] / counts["WeakFN"],
        "skeleton_tp_dlogit": sums["SkeletonTP"] / counts["SkeletonTP"],
        "hard_bg_dlogit": sums["HardBG"] / counts["HardBG"],
    }, counts


def parse_alphas(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def apply_checkpoint_args(args, checkpoint):
    if not isinstance(checkpoint.get("args"), dict):
        return
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--alphas", type=str, default="0.80,0.90,0.95,0.975,0.99,1.00")
    parser.add_argument("--output_csv", type=str, default="")
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
    apply_checkpoint_args(args, checkpoint)

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

    fixed_masks, fixed_counts = build_fixed_masks(args, batches, model, swin)
    rows = []
    count_checks = []
    for alpha in parse_alphas(args.alphas):
        row, counts = evaluate_alpha(args, batches, model, swin, alpha, fixed_masks)
        rows.append(row)
        count_checks.append((alpha, counts))

    print(f"\nCheckpoint: {args.model_path}")
    print(f"split={args.split}, batches={len(batches)}, images={sum(batch['image'].shape[0] for batch in batches)}")
    print(f"threshold={args.threshold}, seed={args.seed}")
    print("F_alpha = (1-alpha) * surface_proj + alpha * surface_refine(surface_branch(surface_proj))")
    print("delta_logit = logits_normal - logits_zero_both")
    print("fixed masks from alpha=1 normal:")
    for name in REGIONS:
        print(f"  {name}: {fixed_counts[name]} expected={EXPECTED_COUNTS[name]} match={fixed_counts[name] == EXPECTED_COUNTS[name]}")

    print("\nMask counts reused by alpha:")
    for alpha, counts in count_checks:
        print(
            f"  alpha={alpha:g}: "
            f"WeakFN={counts['WeakFN']}, SkeletonTP={counts['SkeletonTP']}, HardBG={counts['HardBG']}"
        )

    print("\n========== surface_proj <-> refined interpolation sweep ==========")
    print(
        f"{'alpha':>7} | {'IoU':>8} | {'F1':>8} | {'P':>8} | {'R':>8} | "
        f"{'WeakFN dlogit':>14} | {'SkeletonTP dlogit':>18} | {'HardBG dlogit':>14}"
    )
    for row in rows:
        print(
            f"{row['alpha']:>7.3f} | {row['iou']:>8.4f} | {row['f1']:>8.4f} | "
            f"{row['precision']:>8.4f} | {row['recall']:>8.4f} | "
            f"{row['weak_fn_dlogit']:>14.6f} | {row['skeleton_tp_dlogit']:>18.6f} | "
            f"{row['hard_bg_dlogit']:>14.6f}"
        )

    alpha_one = next((row for row in rows if abs(row["alpha"] - 1.0) < 1e-9), None)
    if alpha_one is not None:
        print("\nalpha=1 final-logit consistency check:")
        for region, key in (
            ("WeakFN", "weak_fn_dlogit"),
            ("SkeletonTP", "skeleton_tp_dlogit"),
            ("HardBG", "hard_bg_dlogit"),
        ):
            value = alpha_one[key]
            expected = EXPECTED_ALPHA1[region]
            diff = abs(value - expected)
            print(f"  {region}: value={value:+.6f}, expected={expected:+.6f}, diff={diff:.12f}, match={diff < 5e-7}")
            if diff >= 5e-7:
                raise RuntimeError("alpha=1 final-logit consistency check failed.")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
