import argparse
import os
import sys
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.skeleton_guided_head import scale_gradient
from networks.swin_transformer_unet_skip_expand_decoder_sys import (
    HighResStructureEncoder,
    map_to_token,
    token_to_map,
)
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


def patch_stage1_highres(model, mode):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    if hasattr(swin, "prepatch_structure_encoder"):
        delattr(swin, "prepatch_structure_encoder")
    swin.highres_structure_encoder = HighResStructureEncoder(
        in_channels=swin.embed_dim,
        struct_channels=swin.highres_structure_channels,
    )
    swin.highres_structure_source = "stage1-alignment"
    swin._alignment_mode = mode

    def build_highres_from_stage1(self, stage1_tokens):
        if not self.enable_highres_structure_stream or stage1_tokens is None:
            return None, None
        stage1_map = token_to_map(stage1_tokens, self.patches_resolution[0], self.patches_resolution[1])
        z_struct = self.highres_structure_encoder(stage1_map)
        skeleton_logits = self.highres_structure_skeleton_head(z_struct)
        return z_struct, skeleton_logits

    def apply_fusion(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        z_for_surface = z_struct.detach()
        if getattr(self, "_alignment_mode", "normal") == "zero_both":
            z_for_surface = torch.zeros_like(z_for_surface)
        feature_map = self.highres_structure_fusion[str(stage)](feature_map, z_for_surface)
        return map_to_token(feature_map)

    def forward_stage1_highres(self, x, gt_skeleton=None, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0):
        x, x_downsample, road_attentions, stage1_tokens = self.forward_features(x)
        z_struct, highres_structure_skeleton = self._build_highres_structure_outputs(stage1_tokens)
        x, structure_outputs = self.forward_up_features(
            x,
            x_downsample,
            bottleneck_tokens=x,
            gt_skeleton=gt_skeleton,
            topology_alpha_scale=topology_alpha_scale,
            teacher_forcing_ratio=teacher_forcing_ratio,
            z_struct=z_struct,
        )
        if self.return_skeleton and highres_structure_skeleton is not None:
            structure_outputs.append(
                {
                    "stage": "highres_structure",
                    "highres_structure_skeleton": highres_structure_skeleton,
                }
            )
        if self.return_skeleton and road_attentions:
            structure_outputs.extend(road_attentions)
        x = self.up_x4(x, structure_outputs=structure_outputs if self.return_skeleton else None)
        if self.return_skeleton and isinstance(x, tuple):
            x = (*x, structure_outputs)
        return x

    swin._build_highres_structure_outputs = MethodType(build_highres_from_stage1, swin)
    swin._apply_highres_structure_fusion = MethodType(apply_fusion, swin)
    swin.forward = MethodType(forward_stage1_highres, swin)


def patch_guided_head_trace(model):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    head = swin.guided_head
    head._alignment_trace = {}

    def traced_forward(self, x):
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
            boundary_residual = self.beta * boundary_attn * boundary_correction
            guided_surface_feat = guided_surface_feat + boundary_residual
            self._alignment_trace["surface_classifier_feature"] = guided_surface_feat.detach().cpu()
            surface_logits = self.surface_head(guided_surface_feat)
            self._alignment_trace["surface_logits"] = surface_logits.detach().cpu()
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
        skeleton_prob = torch.sigmoid(skeleton_logits)
        connectivity_prob = torch.sigmoid(connectivity_logits)

        guided_surface_feat = self.surface_refine(surface_feat)
        boundary_feat = self.boundary_branch(guided_surface_feat)
        boundary_logits = self.boundary_head(boundary_feat)
        boundary_attn = torch.sigmoid(boundary_logits)
        boundary_correction = self.boundary_residual(guided_surface_feat)
        boundary_residual = self.beta * boundary_attn * boundary_correction
        guided_surface_feat = guided_surface_feat + boundary_residual
        self._alignment_trace["surface_classifier_feature"] = guided_surface_feat.detach().cpu()
        surface_logits = self.surface_head(guided_surface_feat)
        self._alignment_trace["surface_logits"] = surface_logits.detach().cpu()
        return surface_logits, boundary_logits, skeleton_logits, connectivity_logits

    head.forward = MethodType(traced_forward, head)


def build_model(args, checkpoint, mode):
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
    )
    patch_stage1_highres(model, mode)
    patch_guided_head_trace(model)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def run_model(args, checkpoint, images, mode):
    model = build_model(args, checkpoint, mode)
    with torch.no_grad():
        model(images)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    head = swin.guided_head
    trace = dict(head._alignment_trace)
    weight = head.surface_head.weight.detach().cpu().view(-1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trace, weight


def append_values(store, name, values):
    values = values.detach().float().cpu()
    if values.numel() > 0:
        store.setdefault(name, []).append(values)


def summarize(store):
    out = {}
    for key, chunks in store.items():
        if not chunks:
            out[key] = float("nan")
        else:
            out[key] = float(torch.cat([x.flatten() for x in chunks]).mean().item())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=8)
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

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cos_store = {}
    dlogit_store = {}
    counts = {}
    weight_ref = None

    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        images = batch["image"].to(args.device)
        normal, weight = run_model(args, checkpoint, images, "normal")
        zero, _ = run_model(args, checkpoint, images, "zero_both")
        weight_ref = weight

        f_normal = normal["surface_classifier_feature"]
        f_zero = zero["surface_classifier_feature"]
        delta_f = f_normal - f_zero
        logits_normal = normal["surface_logits"]
        logits_zero = zero["surface_logits"]
        delta_logit = (logits_normal - logits_zero).squeeze(1)

        weight_vec = weight.view(1, -1, 1, 1)
        numerator = (delta_f * weight_vec).sum(dim=1)
        delta_norm = torch.linalg.vector_norm(delta_f, dim=1)
        weight_norm = torch.linalg.vector_norm(weight)
        cosine = numerator / (delta_norm * weight_norm + 1e-8)

        road = batch["mask"].float()
        skeleton = batch["skeleton"].float()
        if road.shape[-2:] != delta_logit.shape[-2:]:
            road = torch.nn.functional.interpolate(road, size=delta_logit.shape[-2:], mode="nearest")
        if skeleton.shape[-2:] != delta_logit.shape[-2:]:
            skeleton = torch.nn.functional.interpolate(skeleton, size=delta_logit.shape[-2:], mode="nearest")
        road = road.squeeze(1) > 0.5
        skeleton = skeleton.squeeze(1) > 0.5
        pred = torch.sigmoid(logits_normal).squeeze(1) >= args.threshold

        masks = {
            "GT skeleton": skeleton,
            "Weak skeleton FN": skeleton & (~pred),
            "Skeleton TP": skeleton & pred,
            "Road non-skeleton": road & (~skeleton),
            "Background": ~road,
        }
        for name, mask in masks.items():
            counts[name] = counts.get(name, 0) + int(mask.sum().item())
            append_values(cos_store, name, cosine[mask])
            append_values(dlogit_store, name, delta_logit[mask])

    cos_summary = summarize(cos_store)
    dlogit_summary = summarize(dlogit_store)

    print(f"Using checkpoint: {args.model_path}")
    print(f"split={args.split}, batches={args.max_batches}, threshold={args.threshold}")
    if weight_ref is not None:
        print(f"surface_head ||W||={torch.linalg.vector_norm(weight_ref).item():.6f}")

    print("\n========== cos(delta_F, W_surface) ==========")
    print(f"{'region':<22} {'cosine':>12} {'pixels':>12}")
    for name in ("GT skeleton", "Weak skeleton FN", "Skeleton TP", "Road non-skeleton", "Background"):
        print(f"{name:<22} {cos_summary.get(name, float('nan')):>12.6f} {counts.get(name, 0):>12d}")

    print("\n========== signed delta logit: normal - zero ==========")
    print(f"{'region':<22} {'mean dlogit':>12} {'pixels':>12}")
    for name in ("GT skeleton", "Weak skeleton FN", "Skeleton TP", "Road non-skeleton", "Background"):
        print(f"{name:<22} {dlogit_summary.get(name, float('nan')):>12.6f} {counts.get(name, 0):>12d}")


if __name__ == "__main__":
    main()
