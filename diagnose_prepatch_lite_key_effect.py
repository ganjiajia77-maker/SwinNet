import argparse
import os
import sys
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import binary_metrics_from_logits
from networks.swin_transformer_unet_skip_expand_decoder_sys import map_to_token, token_to_map
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


def largest_group_divisor(channels, candidates=(8, 4, 2, 1)):
    for groups in candidates:
        if channels % groups == 0:
            return groups
    return 1


class PrePatchLiteStructureEncoder(nn.Module):
    def __init__(self, struct_channels):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(largest_group_divisor(16), 16),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.GroupNorm(largest_group_divisor(32), 32),
            nn.GELU(),
        )
        self.project = nn.Conv2d(32, struct_channels, kernel_size=1, bias=False)
        bottleneck_channels = max(struct_channels // 4, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(struct_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(largest_group_divisor(bottleneck_channels), bottleneck_channels),
            nn.GELU(),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=1,
                groups=bottleneck_channels,
                bias=False,
            ),
            nn.GroupNorm(largest_group_divisor(bottleneck_channels), bottleneck_channels),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, struct_channels, kernel_size=1, bias=False),
            nn.GroupNorm(largest_group_divisor(struct_channels), struct_channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.project(x)
        return self.act(self.refine(x) + x)


def patch_prepatch_lite(model):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    swin.prepatch_structure_encoder = PrePatchLiteStructureEncoder(swin.highres_structure_channels)
    swin.highres_structure_source = "prepatch-lite-diagnostic"
    swin._diagnostic_mode = "normal"
    swin.guided_head._diagnostic_trace = {}

    def apply_fusion(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        z_for_surface = z_struct.detach()
        if getattr(self, "_diagnostic_mode", "normal") == "zero_both":
            z_for_surface = torch.zeros_like(z_for_surface)
        feature_map = self.highres_structure_fusion[str(stage)](feature_map, z_for_surface)
        return map_to_token(feature_map)

    def guided_forward(self, x):
        surface_feat = self.surface_branch(self.surface_proj(x))
        skeleton_feat = self.skeleton_proj(x)

        if not self.enable_final_structure:
            final_structure_feat = self.structure_branch(skeleton_feat)
            final_skeleton_logits = self.skeleton_head(final_structure_feat)
            guided_surface_feat = self.surface_refine(surface_feat)
            boundary_feat = self.boundary_branch(guided_surface_feat)
            boundary_logits = self.boundary_head(boundary_feat)
            boundary_attn = torch.sigmoid(boundary_logits)
            boundary_correction = self.boundary_residual(guided_surface_feat)
            guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
            self._diagnostic_trace["surface_classifier_feature"] = guided_surface_feat.detach().cpu()
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
        guided_surface_feat = self.surface_refine(surface_feat)
        boundary_feat = self.boundary_branch(guided_surface_feat)
        boundary_logits = self.boundary_head(boundary_feat)
        boundary_attn = torch.sigmoid(boundary_logits)
        boundary_correction = self.boundary_residual(guided_surface_feat)
        guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
        self._diagnostic_trace["surface_classifier_feature"] = guided_surface_feat.detach().cpu()
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
    )
    patch_prepatch_lite(model)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def compute_metrics(logits_list, targets_list, threshold):
    metrics = {"iou": [], "f1": [], "precision": [], "recall": []}
    for logits, targets in zip(logits_list, targets_list):
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(targets.float(), size=logits.shape[-2:], mode="nearest")
        item = binary_metrics_from_logits(logits, targets, threshold=threshold)
        for key in metrics:
            metrics[key].append(item[key])
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def append_values(store, name, values):
    values = values.detach().float().cpu()
    if values.numel() > 0:
        store.setdefault(name, []).append(values.flatten())


def summarize(store):
    return {
        key: float(torch.cat(values).mean().item()) if values else float("nan")
        for key, values in store.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
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
        choices=["stage23", "final_correction", "stage23_final_correction", "none"],
    )
    args = parser.parse_args()

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

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args, checkpoint)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    weight = swin.guided_head.surface_head.weight.detach().cpu().view(-1)
    weight_norm = torch.linalg.vector_norm(weight)

    normal_logits = []
    zero_logits = []
    targets = []
    dlogit_store = {}
    cos_store = {}
    counts = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="PrePatch-Lite key effect")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(args.device)

            swin._diagnostic_mode = "normal"
            model(images)
            normal_trace = dict(swin.guided_head._diagnostic_trace)

            swin._diagnostic_mode = "zero_both"
            model(images)
            zero_trace = dict(swin.guided_head._diagnostic_trace)

            normal_logit = normal_trace["surface_logits"]
            zero_logit = zero_trace["surface_logits"]
            normal_logits.append(normal_logit)
            zero_logits.append(zero_logit)
            targets.append(batch["mask"].float())

            delta_logit = (normal_logit - zero_logit).squeeze(1)
            delta_feat = normal_trace["surface_classifier_feature"] - zero_trace["surface_classifier_feature"]
            numerator = (delta_feat * weight.view(1, -1, 1, 1)).sum(dim=1)
            cosine = numerator / (torch.linalg.vector_norm(delta_feat, dim=1) * weight_norm + 1e-8)

            road = batch["mask"].float()
            skeleton = batch["skeleton"].float()
            if road.shape[-2:] != delta_logit.shape[-2:]:
                road = F.interpolate(road, size=delta_logit.shape[-2:], mode="nearest")
            if skeleton.shape[-2:] != delta_logit.shape[-2:]:
                skeleton = F.interpolate(skeleton, size=delta_logit.shape[-2:], mode="nearest")
            road = road.squeeze(1) > 0.5
            skeleton = skeleton.squeeze(1) > 0.5
            pred = torch.sigmoid(normal_logit).squeeze(1) >= args.threshold
            masks = {
                "Weak skeleton FN": skeleton & (~pred),
                "Skeleton TP": skeleton & pred,
                "Background": ~road,
            }
            for name, mask in masks.items():
                counts[name] = counts.get(name, 0) + int(mask.sum().item())
                append_values(dlogit_store, name, delta_logit[mask])
                append_values(cos_store, name, cosine[mask])

    normal_metrics = compute_metrics(normal_logits, targets, args.threshold)
    zero_metrics = compute_metrics(zero_logits, targets, args.threshold)
    dlogit_summary = summarize(dlogit_store)
    cos_summary = summarize(cos_store)

    print(f"\nCheckpoint: {args.model_path}")
    print(f"split={args.split}, batches={args.max_batches}, threshold={args.threshold}")
    print("\n========== normal vs zero_both ==========")
    print(f"{'mode':<12} {'IoU':>8} {'F1':>8} {'P':>8} {'R':>8}")
    for mode, metrics in (("normal", normal_metrics), ("zero_both", zero_metrics)):
        print(f"{mode:<12} {metrics['iou']:>8.4f} {metrics['f1']:>8.4f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f}")

    print("\n========== Structure-induced signed delta logit ==========")
    print(f"{'region':<18} {'mean dlogit':>12} {'cos(dF,W)':>12} {'pixels':>10}")
    for name in ("Weak skeleton FN", "Skeleton TP", "Background"):
        print(f"{name:<18} {dlogit_summary.get(name, float('nan')):>12.6f} {cos_summary.get(name, float('nan')):>12.6f} {counts.get(name, 0):>10d}")


if __name__ == "__main__":
    main()
