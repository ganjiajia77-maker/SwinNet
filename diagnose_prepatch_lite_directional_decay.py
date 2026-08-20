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


def patch_model(model):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    swin.prepatch_structure_encoder = PrePatchLiteStructureEncoder(swin.highres_structure_channels)
    swin.highres_structure_source = "prepatch-lite-directional-decay"
    swin._diagnostic_mode = "normal"
    swin._directional_trace = {}

    def trace(self, name, value):
        self._directional_trace[name] = value.detach().cpu()

    def apply_fusion(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        z_for_surface = z_struct.detach()
        if getattr(self, "_diagnostic_mode", "normal") == "zero_both":
            z_for_surface = torch.zeros_like(z_for_surface)
        feature_map = self.highres_structure_fusion[str(stage)](feature_map, z_for_surface)
        return map_to_token(feature_map)

    def up_x4_trace(self, x, structure_outputs=None, z_struct=None):
        h, w = self.patches_resolution
        batch, length, channels = x.shape
        assert length == h * w, "input features has wrong size"
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(batch, 4 * h, 4 * w, -1)
            x = x.permute(0, 3, 1, 2)
            trace(self, "final_patch_expand", x)
            if self.return_skeleton:
                x = self.guided_head(x, z_struct=z_struct)
            else:
                x = self.output(x)
        return x

    def guided_forward(self, x, z_struct=None):
        surface_proj = self.surface_proj(x)
        swin._directional_trace["surface_proj"] = surface_proj.detach().cpu()
        surface_feat = self.surface_branch(surface_proj)
        swin._directional_trace["surface_branch"] = surface_feat.detach().cpu()
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
            guided_surface_feat = self.surface_refine(surface_feat)
            guided_surface_feat = self._apply_post_refine_structure_interaction(
                guided_surface_feat,
                z_for_surface,
            )
            swin._directional_trace["surface_refine"] = guided_surface_feat.detach().cpu()
            boundary_feat = self.boundary_branch(guided_surface_feat)
            boundary_logits = self.boundary_head(boundary_feat)
            boundary_attn = torch.sigmoid(boundary_logits)
            boundary_correction = self.boundary_residual(guided_surface_feat)
            guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
            swin._directional_trace["guided_surface_feat"] = guided_surface_feat.detach().cpu()
            surface_logits = self.surface_head(guided_surface_feat)
            swin._directional_trace["surface_logits"] = surface_logits.detach().cpu()
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
        guided_surface_feat = self._apply_post_refine_structure_interaction(
            guided_surface_feat,
            z_for_surface,
        )
        swin._directional_trace["surface_refine"] = guided_surface_feat.detach().cpu()
        boundary_feat = self.boundary_branch(guided_surface_feat)
        boundary_logits = self.boundary_head(boundary_feat)
        boundary_attn = torch.sigmoid(boundary_logits)
        boundary_correction = self.boundary_residual(guided_surface_feat)
        guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
        swin._directional_trace["guided_surface_feat"] = guided_surface_feat.detach().cpu()
        surface_logits = self.surface_head(guided_surface_feat)
        swin._directional_trace["surface_logits"] = surface_logits.detach().cpu()
        return surface_logits, boundary_logits, skeleton_logits, connectivity_logits

    def norm_up_hook(_module, _inputs, output):
        trace(swin, "final_decoder_feature", token_to_map(output, swin.patches_resolution[0], swin.patches_resolution[1]))

    def stage3_hook(_module, _inputs, output):
        if isinstance(output, tuple) and output:
            trace(swin, "stage3_structure_block", output[0])

    swin._apply_highres_structure_fusion = MethodType(apply_fusion, swin)
    swin.up_x4 = MethodType(up_x4_trace, swin)
    swin.guided_head.forward = MethodType(guided_forward, swin.guided_head)
    swin.norm_up.register_forward_hook(norm_up_hook)
    swin.decoder_structure_blocks[3].register_forward_hook(stage3_hook)


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
    patch_model(model)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def append_values(store, name, values):
    values = values.detach().float().cpu()
    if values.numel() > 0:
        store.setdefault(name, []).append(values.flatten())


def summarize(store):
    return {
        key: float(torch.cat(values).mean().item()) if values else float("nan")
        for key, values in store.items()
    }


def rel_diff(normal, zero, mask):
    if normal.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask[:, None].float(), size=normal.shape[-2:], mode="nearest").squeeze(1) > 0.5
    diff = normal - zero
    dims = tuple(range(1, normal.dim()))
    masked_diff = diff.permute(0, *range(2, diff.dim()), 1)[mask]
    masked_norm = normal.permute(0, *range(2, normal.dim()), 1)[mask]
    if masked_diff.numel() == 0:
        return float("nan")
    return float(torch.linalg.vector_norm(masked_diff) / (torch.linalg.vector_norm(masked_norm) + 1e-8))


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
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_FULL, choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626])
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

    dataset = RoadSkeletonDataset(root_dir=args.root_path, split=args.split, image_size=args.img_size, source_patch_size=args.source_patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args, checkpoint)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    weight = swin.guided_head.surface_head.weight.detach().cpu().view(1, -1, 1, 1)

    decision_layers = ["surface_proj", "surface_branch", "surface_refine", "guided_surface_feat", "surface_logits"]
    survival_layers = ["stage3_structure_block", "final_decoder_feature", "final_patch_expand"]
    regions = ["Weak FN", "Skeleton TP", "Hard BG"]
    decision_store = {layer: {} for layer in decision_layers}
    survival_store = {layer: {} for layer in survival_layers}
    counts = {region: 0 for region in regions}

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Directional decay")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(args.device)

            swin._diagnostic_mode = "normal"
            swin._directional_trace = {}
            model(images)
            normal = dict(swin._directional_trace)

            swin._diagnostic_mode = "zero_both"
            swin._directional_trace = {}
            model(images)
            zero = dict(swin._directional_trace)

            logits = normal["surface_logits"]
            road = batch["mask"].float()
            skeleton = batch["skeleton"].float()
            if road.shape[-2:] != logits.shape[-2:]:
                road = F.interpolate(road, size=logits.shape[-2:], mode="nearest")
            if skeleton.shape[-2:] != logits.shape[-2:]:
                skeleton = F.interpolate(skeleton, size=logits.shape[-2:], mode="nearest")
            road = road.squeeze(1) > 0.5
            skeleton = skeleton.squeeze(1) > 0.5
            pred = torch.sigmoid(logits).squeeze(1) >= args.threshold
            masks = {
                "Weak FN": skeleton & (~pred),
                "Skeleton TP": skeleton & pred,
                "Hard BG": (~road) & pred,
            }
            for region, mask in masks.items():
                counts[region] += int(mask.sum().item())
                for layer in decision_layers:
                    delta = normal[layer] - zero[layer]
                    if layer == "surface_logits":
                        score = delta.squeeze(1)
                    else:
                        score = (delta * weight).sum(dim=1)
                    append_values(decision_store[layer], region, score[mask])
                for layer in survival_layers:
                    value = rel_diff(normal[layer], zero[layer], mask)
                    if np.isfinite(value):
                        survival_store[layer].setdefault(region, []).append(torch.tensor([value]))

    print(f"\nCheckpoint: {args.model_path}")
    print(f"split={args.split}, batches={args.max_batches}, threshold={args.threshold}")
    print("\n========== Directional Decision Contribution ==========")
    print(f"{'position':<28} {'Weak FN':>12} {'Skeleton TP':>12} {'Hard BG':>12}")
    for layer in decision_layers:
        summary = summarize(decision_store[layer])
        print(f"{layer:<28} {summary.get('Weak FN', float('nan')):>12.6f} {summary.get('Skeleton TP', float('nan')):>12.6f} {summary.get('Hard BG', float('nan')):>12.6f}")

    print("\n========== Signal Survival R=||normal-zero||/||normal|| ==========")
    print(f"{'position':<28} {'Weak FN':>12} {'Skeleton TP':>12} {'Hard BG':>12}")
    for layer in survival_layers:
        summary = summarize(survival_store[layer])
        print(f"{layer:<28} {summary.get('Weak FN', float('nan')):>12.6f} {summary.get('Skeleton TP', float('nan')):>12.6f} {summary.get('Hard BG', float('nan')):>12.6f}")

    print("\nPixels:", counts)


if __name__ == "__main__":
    main()
