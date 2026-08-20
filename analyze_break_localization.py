import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


DIR_OFFSETS = torch.tensor(
    [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=torch.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether missed GT skeleton pixels still have S/C/D structure signals."
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./analysis_out/break_localization")
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--stage", type=int, default=3, choices=[2, 3])
    parser.add_argument("--refinement_step", type=int, default=1)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.add_argument(
        "--radii",
        type=str,
        default="1,2,4,8,12",
        help="comma-separated pixel radii for GT-tangent two-side diagnostics",
    )
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--num_classes", default=1, type=int)
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
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument(
        "--stage_topology_stages",
        type=str,
        default="none",
        choices=["none", "stage3", "stage23"],
    )
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument(
        "--stage_topology_bias_mode",
        type=str,
        default="pairwise_skeleton",
        choices=["pairwise_skeleton", "gap_query"],
    )
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument(
        "--structure_profile",
        type=str,
        default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
        choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626],
    )
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    return parser.parse_args()


def load_model(args, device):
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
        args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
        for name in (
            "stage_topology_stages",
            "stage_topology_alpha_max",
            "stage_topology_alpha_init",
            "stage_topology_bias_mode",
            "stage_topology_ratio",
            "stage_topology_topo_clip",
            "stage2_skeleton_gradient_ratio",
            "stage3_skeleton_gradient_ratio",
            "final_skeleton_gradient_ratio",
            "bottleneck_type",
        ):
            if name in saved_args:
                setattr(args, name, saved_args[name])
    saved_profile = checkpoint.get("structure_profile") if isinstance(checkpoint, dict) else None
    if saved_profile:
        args.structure_profile = saved_profile

    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        stage_topology_bias_mode=args.stage_topology_bias_mode,
        stage_topology_ratio=args.stage_topology_ratio,
        stage_topology_topo_clip=args.stage_topology_topo_clip,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=(args.bottleneck_type == "global_local"),
    )
    model.to(device).eval()
    print_topology_coefficients(model)
    return model


def select_stage_output(structure_outputs, stage, refinement_step):
    candidates = [
        item
        for item in structure_outputs
        if item.get("stage") == stage
        and item.get("refinement_step", refinement_step) == refinement_step
        and "skeleton" in item
    ]
    if candidates:
        return candidates[-1]
    fallback = [
        item for item in structure_outputs if item.get("stage") == stage and "skeleton" in item
    ]
    if not fallback:
        raise RuntimeError(f"No structure output found for stage {stage}.")
    return fallback[-1]


def resize_like(x, reference, mode="bilinear"):
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(x, **kwargs)


def direction_alignment(direction_logits):
    direction = F.normalize(direction_logits, dim=1, eps=1e-6)
    offsets = DIR_OFFSETS.to(device=direction.device, dtype=direction.dtype)
    offsets = F.normalize(offsets, dim=1, eps=1e-6)
    return torch.einsum("bchw,kc->bkhw", direction, offsets).clamp_min(0.0)


def shifted_prob(prob, dy, dx):
    radius = max(abs(int(dy)), abs(int(dx)))
    batch, channels, height, width = prob.shape
    padded = F.pad(prob, (radius, radius, radius, radius), mode="replicate")
    y0 = radius + int(dy)
    x0 = radius + int(dx)
    return padded[:, :, y0 : y0 + height, x0 : x0 + width]


def directional_road_stats(surface_prob, connectivity_prob, direction_align):
    offsets = DIR_OFFSETS.to(device=surface_prob.device)
    samples = []
    for idx, (dy, dx) in enumerate(offsets.tolist()):
        far = shifted_prob(surface_prob, int(dy), int(dx))
        opposite = shifted_prob(surface_prob, -int(dy), -int(dx))
        pair_mean = 0.5 * (far + opposite)
        pair_min = torch.minimum(far, opposite)
        score = connectivity_prob[:, idx : idx + 1] * direction_align[:, idx : idx + 1]
        samples.append((score, pair_mean, pair_min))
    score_stack = torch.cat([x[0] for x in samples], dim=1)
    mean_stack = torch.cat([x[1] for x in samples], dim=1)
    min_stack = torch.cat([x[2] for x in samples], dim=1)
    best_idx = score_stack.argmax(dim=1, keepdim=True)
    best_pair_mean = torch.gather(mean_stack, 1, best_idx)
    best_pair_min = torch.gather(min_stack, 1, best_idx)
    best_score = torch.gather(score_stack, 1, best_idx)
    return best_pair_mean, best_pair_min, best_score


def gt_tangent_direction_index(gt_skeleton):
    skel = gt_skeleton.float()
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=skel.device,
        dtype=skel.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=skel.device,
        dtype=skel.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(skel, kernel_x, padding=1)
    gy = F.conv2d(skel, kernel_y, padding=1)
    jxx = F.avg_pool2d(gx * gx, kernel_size=7, stride=1, padding=3)
    jxy = F.avg_pool2d(gx * gy, kernel_size=7, stride=1, padding=3)
    jyy = F.avg_pool2d(gy * gy, kernel_size=7, stride=1, padding=3)

    # Dominant local edge normal angle; tangent is normal + pi/2.
    normal_angle = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy + 1e-6)
    tangent_y = torch.cos(normal_angle)
    tangent_x = -torch.sin(normal_angle)
    tangent = torch.cat([tangent_y, tangent_x], dim=1)
    tangent = F.normalize(tangent, dim=1, eps=1e-6)
    offsets = DIR_OFFSETS.to(device=skel.device, dtype=skel.dtype)
    offsets = F.normalize(offsets, dim=1, eps=1e-6)
    score = torch.einsum("bchw,kc->bkhw", tangent, offsets).abs()
    return score.argmax(dim=1, keepdim=True)


def gather_direction_channel(x, index):
    return torch.gather(x, 1, index)


def tangent_connectivity_stats(connectivity_prob, tangent_idx, radius):
    offsets = DIR_OFFSETS.to(device=connectivity_prob.device)
    idx = tangent_idx.long()
    opposite_idx = torch.tensor(
        [1, 0, 3, 2, 7, 6, 5, 4],
        device=connectivity_prob.device,
        dtype=torch.long,
    )
    perp_pairs = torch.tensor(
        [
            [2, 3],  # N/S -> W/E
            [2, 3],
            [0, 1],  # W/E -> N/S
            [0, 1],
            [5, 6],  # NW/SE diagonal perpendiculars
            [4, 7],
            [4, 7],
            [5, 6],
        ],
        device=connectivity_prob.device,
        dtype=torch.long,
    )

    c_forward = gather_direction_channel(connectivity_prob, idx)
    c_backward = gather_direction_channel(connectivity_prob, opposite_idx[idx])
    parallel = c_forward + c_backward
    perp_a_idx = perp_pairs[idx.squeeze(1), 0].unsqueeze(1)
    perp_b_idx = perp_pairs[idx.squeeze(1), 1].unsqueeze(1)
    perpendicular = (
        gather_direction_channel(connectivity_prob, perp_a_idx)
        + gather_direction_channel(connectivity_prob, perp_b_idx)
    )

    sym_terms = []
    for dir_idx, (dy, dx) in enumerate(offsets.tolist()):
        channel_here = connectivity_prob[:, dir_idx : dir_idx + 1]
        channel_there = shifted_prob(
            connectivity_prob[:, opposite_idx[dir_idx] : opposite_idx[dir_idx] + 1],
            int(dy) * radius,
            int(dx) * radius,
        )
        sym_terms.append(torch.sqrt((channel_here * channel_there).clamp_min(0.0) + 1e-8))
    sym_stack = torch.cat(sym_terms, dim=1)
    c_sym = gather_direction_channel(sym_stack, idx)

    return {
        f"gt_c_parallel_r{radius}": parallel,
        f"gt_c_perpendicular_r{radius}": perpendicular,
        f"gt_c_par_minus_perp_r{radius}": parallel - perpendicular,
        f"gt_c_par_over_perp_r{radius}": parallel / (perpendicular + 1e-6),
        f"gt_c_sym_r{radius}": c_sym,
    }


def tangent_two_side_surface(surface_prob, tangent_idx, radius):
    offsets = DIR_OFFSETS.to(device=surface_prob.device)
    means = []
    mins = []
    for dir_idx, (dy, dx) in enumerate(offsets.tolist()):
        left = shifted_prob(surface_prob, int(dy) * radius, int(dx) * radius)
        right = shifted_prob(surface_prob, -int(dy) * radius, -int(dx) * radius)
        means.append(0.5 * (left + right))
        mins.append(torch.minimum(left, right))
    mean_stack = torch.cat(means, dim=1)
    min_stack = torch.cat(mins, dim=1)
    return {
        f"gt_two_side_mean_r{radius}": gather_direction_channel(mean_stack, tangent_idx),
        f"gt_two_side_min_r{radius}": gather_direction_channel(min_stack, tangent_idx),
    }


def direction_distribution_stats(direction_logits):
    direction = F.normalize(direction_logits, dim=1, eps=1e-6)
    offsets = DIR_OFFSETS.to(device=direction.device, dtype=direction.dtype)
    offsets = F.normalize(offsets, dim=1, eps=1e-6)
    logits = torch.einsum("bchw,kc->bkhw", direction, offsets) * 8.0
    probs = torch.softmax(logits, dim=1)
    top2 = probs.topk(k=2, dim=1).values
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)
    return {
        "dir_entropy": entropy,
        "dir_top1_minus_top2": top2[:, 0:1] - top2[:, 1:2],
    }


def direction_continuity_stats(direction_logits, tangent_idx, radii):
    direction = F.normalize(direction_logits, dim=1, eps=1e-6)
    offsets = DIR_OFFSETS.to(device=direction.device)
    stats = {}
    for radius in radii:
        cos_terms = []
        for _, (dy, dx) in enumerate(offsets.tolist()):
            left = shifted_prob(direction, int(dy) * radius, int(dx) * radius)
            right = shifted_prob(direction, -int(dy) * radius, -int(dx) * radius)
            cos_terms.append((left * right).sum(dim=1, keepdim=True).abs())
        cos_stack = torch.cat(cos_terms, dim=1)
        stats[f"dir_consistency_r{radius}"] = gather_direction_channel(
            cos_stack,
            tangent_idx,
        )
    return stats


class StatBucket:
    def __init__(self):
        self.count = 0
        self.sums = {}

    def add(self, mask, values):
        mask = mask.bool()
        count = int(mask.sum().item())
        if count == 0:
            return
        self.count += count
        for name, value in values.items():
            self.sums[name] = self.sums.get(name, 0.0) + float(value[mask].sum().item())

    def means(self):
        if self.count == 0:
            return {}
        return {name: value / self.count for name, value in self.sums.items()}


def sample_mask(mask, max_pixels):
    indices = mask.flatten().nonzero(as_tuple=False).flatten()
    if indices.numel() <= max_pixels:
        return mask
    perm = torch.randperm(indices.numel(), device=indices.device)[:max_pixels]
    sampled = torch.zeros_like(mask.flatten(), dtype=torch.bool)
    sampled[indices[perm]] = True
    return sampled.view_as(mask)


def main():
    args = parse_args()
    radii = [int(part) for part in args.radii.split(",") if part.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
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
        pin_memory=True,
    )

    buckets = {
        "break_skeleton_fn": StatBucket(),
        "surface_tp": StatBucket(),
        "surface_fp": StatBucket(),
        "surface_tn_sampled": StatBucket(),
    }
    case_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            mask = batch["mask"].to(device).float()
            gt_skeleton = batch["skeleton"].to(device).float()
            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0]
            surface_prob = torch.sigmoid(surface_logits)
            structure_outputs = outputs[-1]
            stage_output = select_stage_output(
                structure_outputs,
                args.stage,
                args.refinement_step,
            )

            skeleton_prob = torch.sigmoid(
                resize_like(stage_output["skeleton"], surface_prob, mode="bilinear")
            )
            connectivity_prob = torch.sigmoid(
                resize_like(stage_output["connectivity"], surface_prob, mode="bilinear")
            )
            direction_logits = resize_like(
                stage_output["direction"],
                surface_prob,
                mode="bilinear",
            )
            direction_align = direction_alignment(direction_logits)
            c_dir_score = connectivity_prob * direction_align
            c_conf = connectivity_prob.max(dim=1, keepdim=True).values
            d_conf = direction_align.max(dim=1, keepdim=True).values
            cd_conf = c_dir_score.max(dim=1, keepdim=True).values
            two_side_mean, two_side_min, two_side_cd_score = directional_road_stats(
                surface_prob,
                connectivity_prob,
                direction_align,
            )

            if mask.shape[-2:] != surface_prob.shape[-2:]:
                mask = resize_like(mask, surface_prob, mode="nearest")
            if gt_skeleton.shape[-2:] != surface_prob.shape[-2:]:
                gt_skeleton = resize_like(gt_skeleton, surface_prob, mode="nearest")

            tangent_idx = gt_tangent_direction_index(gt_skeleton)
            dir_stats = direction_distribution_stats(direction_logits)
            dir_stats.update(direction_continuity_stats(direction_logits, tangent_idx, radii))
            tangent_values = {}
            for radius in radii:
                tangent_values.update(
                    tangent_two_side_surface(surface_prob, tangent_idx, radius)
                )
                tangent_values.update(
                    tangent_connectivity_stats(connectivity_prob, tangent_idx, radius)
                )

            pred = surface_prob >= args.threshold
            gt = mask >= 0.5
            gt_skel = gt_skeleton >= 0.5
            fn = gt & (~pred)
            tp = gt & pred
            fp = (~gt) & pred
            tn = (~gt) & (~pred)
            break_fn = fn & gt_skel

            values = {
                "surface_p": surface_prob,
                "stage_s": skeleton_prob,
                "stage_c_max": c_conf,
                "stage_dir_max": d_conf,
                "stage_cdir_max": cd_conf,
                "two_side_road_mean": two_side_mean,
                "two_side_road_min": two_side_min,
                "two_side_cdir_score": two_side_cd_score,
            }
            values.update(dir_stats)
            values.update(tangent_values)
            bucket_masks = {
                "break_skeleton_fn": break_fn,
                "surface_tp": tp,
                "surface_fp": fp,
                "surface_tn_sampled": sample_mask(tn, max_pixels=break_fn.numel() // 20),
            }
            for name, bucket_mask in bucket_masks.items():
                buckets[name].add(bucket_mask, values)

            case_rows.append(
                {
                    "batch": batch_idx,
                    "break_skeleton_fn_pixels": int(break_fn.sum().item()),
                    "tp_pixels": int(tp.sum().item()),
                    "fp_pixels": int(fp.sum().item()),
                    "tn_pixels": int(tn.sum().item()),
                    "surface_iou": float(
                        (tp.sum() / (tp.sum() + fp.sum() + fn.sum()).clamp_min(1)).item()
                    ),
                }
            )
            print(
                f"[{batch_idx + 1}/{len(loader)}] break_fn={case_rows[-1]['break_skeleton_fn_pixels']} "
                f"tp={case_rows[-1]['tp_pixels']} fp={case_rows[-1]['fp_pixels']} "
                f"iou={case_rows[-1]['surface_iou']:.4f}",
                flush=True,
            )

    summary_path = os.path.join(args.output_dir, "break_localization_summary.csv")
    case_path = os.path.join(args.output_dir, "break_localization_cases.csv")
    metric_names = [
        "surface_p",
        "stage_s",
        "stage_c_max",
        "stage_dir_max",
        "stage_cdir_max",
        "two_side_road_mean",
        "two_side_road_min",
        "two_side_cdir_score",
        "dir_entropy",
        "dir_top1_minus_top2",
    ]
    for radius in radii:
        metric_names.extend(
            [
                f"gt_two_side_mean_r{radius}",
                f"gt_two_side_min_r{radius}",
                f"gt_c_parallel_r{radius}",
                f"gt_c_perpendicular_r{radius}",
                f"gt_c_par_minus_perp_r{radius}",
                f"gt_c_par_over_perp_r{radius}",
                f"gt_c_sym_r{radius}",
                f"dir_consistency_r{radius}",
            ]
        )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "pixels"] + metric_names)
        writer.writeheader()
        for region, bucket in buckets.items():
            row = {"region": region, "pixels": bucket.count}
            row.update(bucket.means())
            writer.writerow(row)
    with open(case_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch",
                "break_skeleton_fn_pixels",
                "tp_pixels",
                "fp_pixels",
                "tn_pixels",
                "surface_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(case_rows)

    print("\nSummary")
    print("region,pixels," + ",".join(metric_names))
    for region, bucket in buckets.items():
        means = bucket.means()
        values = [f"{means.get(name, float('nan')):.6f}" for name in metric_names]
        print(f"{region},{bucket.count}," + ",".join(values))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {case_path}")


if __name__ == "__main__":
    main()
