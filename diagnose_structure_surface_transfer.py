import argparse
import csv
import os
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
from losses.road_losses import binary_metrics_from_logits
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
    print_topology_coefficients,
)


THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.22,
    0.24,
    0.25,
    0.26,
    0.28,
    0.30,
    0.32,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]


def compute_metrics_all_samples(logits_list, targets_list, threshold):
    values = {"iou": [], "f1": [], "precision": [], "recall": []}
    for logits, targets in zip(logits_list, targets_list):
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(targets.float(), size=logits.shape[-2:], mode="nearest")
        metrics = binary_metrics_from_logits(logits, targets, threshold=threshold)
        for key in values:
            values[key].append(metrics[key])
    return {key: float(np.mean(item_values)) for key, item_values in values.items()}


def patch_stage1_highres_and_fusion_mode(model, mode, seed):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    if hasattr(swin, "prepatch_structure_encoder"):
        delattr(swin, "prepatch_structure_encoder")
    swin.highres_structure_encoder = HighResStructureEncoder(
        in_channels=swin.embed_dim,
        struct_channels=swin.highres_structure_channels,
    )
    swin.highres_structure_source = "stage1-diagnostic"
    swin._diagnostic_z_struct_mode = mode
    swin._diagnostic_shuffle_seed = int(seed)
    swin._diagnostic_shuffle_cache = {}
    swin._diagnostic_fusion_stats = []

    def build_highres_from_stage1(self, stage1_tokens):
        if not self.enable_highres_structure_stream or stage1_tokens is None:
            return None, None
        stage1_map = token_to_map(
            stage1_tokens,
            self.patches_resolution[0],
            self.patches_resolution[1],
        )
        z_struct = self.highres_structure_encoder(stage1_map)
        skeleton_logits = self.highres_structure_skeleton_head(z_struct)
        return z_struct, skeleton_logits

    def apply_fusion_mode(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        z_for_surface = z_struct.detach()
        mode_name = getattr(self, "_diagnostic_z_struct_mode", "normal")
        stage_name = f"stage{stage}"
        zero_stage = (
            mode_name == "zero"
            or mode_name == "zero_both"
            or (mode_name == "zero_stage2" and int(stage) == 2)
            or (mode_name == "zero_stage3" and int(stage) == 3)
        )
        if zero_stage:
            z_for_surface = torch.zeros_like(z_for_surface)
        elif mode_name == "shuffle":
            bsz, channels, height, width = z_for_surface.shape
            cache_key = (height, width, z_for_surface.device.type, z_for_surface.device.index)
            perm = self._diagnostic_shuffle_cache.get(cache_key)
            if perm is None or perm.device != z_for_surface.device:
                generator = torch.Generator(device=z_for_surface.device)
                generator.manual_seed(getattr(self, "_diagnostic_shuffle_seed", 1234))
                perm = torch.randperm(height * width, generator=generator, device=z_for_surface.device)
                self._diagnostic_shuffle_cache[cache_key] = perm
            z_for_surface = z_for_surface.flatten(2)[:, :, perm].view(bsz, channels, height, width)
        elif mode_name not in {"normal", "zero_stage2", "zero_stage3", "zero_both"}:
            raise ValueError(f"Unknown z_struct mode: {mode_name}")

        fusion = self.highres_structure_fusion[str(stage)]
        z = F.interpolate(
            z_for_surface,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z = fusion.project(z)
        delta = fusion.delta(torch.cat([feature_map, z], dim=1))
        self._diagnostic_fusion_stats.append(
            {
                "stage": stage_name,
                "semantic_abs": feature_map.detach().abs().mean().item(),
                "delta_abs": delta.detach().abs().mean().item(),
            }
        )
        feature_map = feature_map + delta
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
        x = self.up_x4(
            x,
            structure_outputs=structure_outputs if self.return_skeleton else None,
        )
        if self.return_skeleton and isinstance(x, tuple):
            x = (*x, structure_outputs)
        return x

    swin._build_highres_structure_outputs = MethodType(build_highres_from_stage1, swin)
    swin._apply_highres_structure_fusion = MethodType(apply_fusion_mode, swin)
    swin.forward = MethodType(forward_stage1_highres, swin)


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
    patch_stage1_highres_and_fusion_mode(model, mode=mode, seed=args.shuffle_seed)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    model = model.to(args.device)
    model.eval()
    return model


def infer_mode(args, checkpoint, loader, mode):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(args, checkpoint, mode)
    logits_list = []
    targets_list = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"Inference {mode}")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            outputs = model(images)
            if not isinstance(outputs, tuple):
                raise RuntimeError("Expected tuple outputs with surface logits.")
            logits_list.append(outputs[0].detach().cpu())
            targets_list.append(masks.detach().cpu())
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    fusion_stats = list(getattr(swin, "_diagnostic_fusion_stats", []))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits_list, targets_list, fusion_stats


def summarize_fusion_stats(fusion_stats):
    summary = {}
    for stage in ("stage2", "stage3"):
        items = [item for item in fusion_stats if item["stage"] == stage]
        if not items:
            continue
        semantic_abs = float(np.mean([item["semantic_abs"] for item in items]))
        delta_abs = float(np.mean([item["delta_abs"] for item in items]))
        summary[stage] = {
            "semantic_abs": semantic_abs,
            "delta_abs": delta_abs,
            "ratio": delta_abs / (semantic_abs + 1e-8),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--crop_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument(
        "--modes",
        type=str,
        default="normal,zero,shuffle",
        help="comma-separated modes: normal,zero,shuffle,zero_stage2,zero_stage3,zero_both",
    )
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

    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
        args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
        args.enable_highres_structure_stream = bool(
            saved_args.get("enable_highres_structure_stream", args.enable_highres_structure_stream)
        )
        args.highres_structure_channels = int(saved_args.get("highres_structure_channels", args.highres_structure_channels))
        args.highres_structure_fuse_stages = str(
            saved_args.get("highres_structure_fuse_stages", args.highres_structure_fuse_stages)
        )

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    print(f"Checkpoint: {args.model_path}")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    print(
        "Diagnostic: source=stage1_highres, modes={}, split={}, img_size={}, source_patch_size={}".format(
            "/".join(modes),
            args.split,
            args.img_size,
            args.source_patch_size,
        )
    )

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.crop_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"{args.split} set size: {len(dataset)}")
    if args.max_batches > 0:
        print(f"Limiting to first {args.max_batches} batches.")

    all_rows = []
    summary = {}
    fusion_summary = {}
    for mode in modes:
        logits, targets, fusion_stats = infer_mode(args, checkpoint, loader, mode)
        fusion_summary[mode] = summarize_fusion_stats(fusion_stats)
        mode_results = {}
        for threshold in THRESHOLDS:
            metrics = compute_metrics_all_samples(logits, targets, threshold)
            mode_results[threshold] = metrics
            all_rows.append({"mode": mode, "threshold": threshold, **metrics})
        best_iou_threshold = max(mode_results, key=lambda item: mode_results[item]["iou"])
        best_f1_threshold = max(mode_results, key=lambda item: mode_results[item]["f1"])
        summary[mode] = {
            "best_iou_threshold": best_iou_threshold,
            "best_iou": mode_results[best_iou_threshold]["iou"],
            "best_f1_threshold": best_f1_threshold,
            "best_f1": mode_results[best_f1_threshold]["f1"],
            "metrics_at_0.40": mode_results[0.40],
        }

    print("\nSUMMARY")
    print("mode      best_iou_t  best_iou  best_f1_t   best_f1   iou@0.40  f1@0.40  precision@0.40  recall@0.40")
    for mode in modes:
        item = summary[mode]
        fixed = item["metrics_at_0.40"]
        print(
            f"{mode:<9} {item['best_iou_threshold']:<11.2f} {item['best_iou']:<9.4f} "
            f"{item['best_f1_threshold']:<11.2f} {item['best_f1']:<9.4f} "
            f"{fixed['iou']:<8.4f} {fixed['f1']:<7.4f} {fixed['precision']:<15.4f} {fixed['recall']:<10.4f}"
        )

    print("\nFUSION EFFECTIVE DELTA STATS")
    for mode in modes:
        print(f"{mode}:")
        for stage in ("stage2", "stage3"):
            stats = fusion_summary.get(mode, {}).get(stage)
            if not stats:
                print(f"  {stage}: no samples")
                continue
            print(
                f"  {stage}: mean |F_sem| = {stats['semantic_abs']:.6f}, "
                f"mean |delta_effective| = {stats['delta_abs']:.6f}, "
                f"ratio = {stats['ratio']:.6f}"
            )

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["mode", "threshold", "iou", "f1", "precision", "recall"],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
