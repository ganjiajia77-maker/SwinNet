import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_surface_refine_identity_sweep import build_model


REGIONS = ("WeakFN", "SkeletonTP", "HardBG")


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


def build_fixed_masks(args, batches, model, swin):
    fixed_masks = []
    counts = {name: 0 for name in REGIONS}
    normal_logits = []

    with torch.no_grad():
        for batch in tqdm(batches, desc="Build alpha=1 normal fixed masks"):
            images = batch["image"].to(args.device)
            trace = run_forward(model, swin, images, "normal")
            logits_normal = trace["surface_logits"].cpu()
            normal_logits.append(logits_normal)

            prob_normal = torch.sigmoid(logits_normal).squeeze(1)
            surface_gt = resize_mask(batch["mask"].float(), logits_normal.shape[-2:]).squeeze(1) > 0.5
            gt_skeleton = resize_mask(batch["skeleton"].float(), logits_normal.shape[-2:]).squeeze(1) > 0.5

            masks = {
                "WeakFN": gt_skeleton & (prob_normal < args.threshold),
                "SkeletonTP": gt_skeleton & (prob_normal >= args.threshold),
                "HardBG": (~surface_gt) & (prob_normal >= args.hard_bg_threshold),
            }
            fixed_masks.append(masks)
            for name in REGIONS:
                counts[name] += int(masks[name].sum().item())

    return fixed_masks, counts, normal_logits


def summarize_with_fixed_masks(args, batches, model, swin, fixed_masks, pass_name):
    sums = {name: 0.0 for name in REGIONS}
    counts = {name: 0 for name in REGIONS}
    normal_sums = {name: 0.0 for name in REGIONS}
    zero_sums = {name: 0.0 for name in REGIONS}

    max_identity_error = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=pass_name)):
            images = batch["image"].to(args.device)
            normal = run_forward(model, swin, images, "normal")
            zero = run_forward(model, swin, images, "zero_both")
            logits_normal = normal["surface_logits"].cpu().squeeze(1)
            logits_zero = zero["surface_logits"].cpu().squeeze(1)
            delta_logit = logits_normal - logits_zero

            for name in REGIONS:
                mask = fixed_masks[batch_idx][name]
                count = int(mask.sum().item())
                counts[name] += count
                if count == 0:
                    continue
                normal_sum = float(logits_normal[mask].double().sum().item())
                zero_sum = float(logits_zero[mask].double().sum().item())
                delta_sum = float(delta_logit[mask].double().sum().item())
                normal_sums[name] += normal_sum
                zero_sums[name] += zero_sum
                sums[name] += delta_sum
                identity_error = abs((normal_sum - zero_sum) - delta_sum) / max(count, 1)
                max_identity_error = max(max_identity_error, identity_error)

    rows = {}
    for name in REGIONS:
        count = counts[name]
        if count == 0:
            rows[name] = {
                "count": 0,
                "normal_mean": float("nan"),
                "zero_mean": float("nan"),
                "delta_mean": float("nan"),
                "normal_minus_zero_mean": float("nan"),
            }
            continue
        rows[name] = {
            "count": count,
            "normal_mean": normal_sums[name] / count,
            "zero_mean": zero_sums[name] / count,
            "delta_mean": sums[name] / count,
            "normal_minus_zero_mean": (normal_sums[name] - zero_sums[name]) / count,
        }

    return rows, counts, max_identity_error


def print_rows(title, rows):
    print(f"\n========== {title} ==========")
    print(
        f"{'region':<12} {'pixels':>10} {'mean normal':>14} "
        f"{'mean zero':>14} {'mean(n-z)':>14} {'mean delta':>14}"
    )
    for name in REGIONS:
        row = rows[name]
        print(
            f"{name:<12} {row['count']:>10d} "
            f"{row['normal_mean']:>14.6f} {row['zero_mean']:>14.6f} "
            f"{row['normal_minus_zero_mean']:>14.6f} {row['delta_mean']:>14.6f}"
        )


def compare_rows(first, second):
    max_diff = 0.0
    for name in REGIONS:
        for key in ("normal_mean", "zero_mean", "delta_mean", "normal_minus_zero_mean"):
            a = first[name][key]
            b = second[name][key]
            if math.isnan(a) and math.isnan(b):
                continue
            max_diff = max(max_diff, abs(a - b))
    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--hard_bg_threshold", type=float, default=0.45)
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batches = collect_batches(loader, args.max_batches)

    model = build_model(args, checkpoint)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet

    print(f"Checkpoint: {args.model_path}")
    print(f"split={args.split}, batches={len(batches)}, images={sum(batch['image'].shape[0] for batch in batches)}")
    print(f"threshold={args.threshold}, hard_bg_threshold={args.hard_bg_threshold}, seed={args.seed}")
    print("delta_logit definition: logits_normal - logits_zero_both")
    print("WeakFN mask: gt_skeleton == 1 and sigmoid(logits_normal_alpha1) < threshold")
    print("SkeletonTP mask: gt_skeleton == 1 and sigmoid(logits_normal_alpha1) >= threshold")
    print("HardBG mask: surface_gt == 0 and sigmoid(logits_normal_alpha1) >= hard_bg_threshold")

    fixed_masks, fixed_counts, _ = build_fixed_masks(args, batches, model, swin)
    print("\nFixed mask pixel counts:")
    for name in REGIONS:
        print(f"{name}: {fixed_counts[name]}")

    rows_a, counts_a, identity_error_a = summarize_with_fixed_masks(
        args,
        batches,
        model,
        swin,
        fixed_masks,
        "alpha=1 normal/zero pass A",
    )
    rows_b, counts_b, identity_error_b = summarize_with_fixed_masks(
        args,
        batches,
        model,
        swin,
        fixed_masks,
        "alpha=1 normal/zero pass B",
    )

    print_rows("alpha=1 pass A", rows_a)
    print_rows("alpha=1 pass B", rows_b)

    print("\nMask count consistency across alpha=1 repeated passes:")
    for name in REGIONS:
        ok = fixed_counts[name] == counts_a[name] == counts_b[name]
        print(f"{name}: fixed={fixed_counts[name]}, passA={counts_a[name]}, passB={counts_b[name]}, consistent={ok}")

    repeat_diff = compare_rows(rows_a, rows_b)
    print(f"\nmax repeat mean abs diff: {repeat_diff:.12f}")
    print(f"max identity check abs error per pixel pass A: {identity_error_a:.12e}")
    print(f"max identity check abs error per pixel pass B: {identity_error_b:.12e}")

    print("\n========== Verified alpha=1 Structure-induced delta_logit ==========")
    for name in REGIONS:
        print(f"{name} Δlogit: {rows_a[name]['delta_mean']:+.6f}")


if __name__ == "__main__":
    main()
