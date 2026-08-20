import argparse
import csv
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


NEIGHBORS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=r"D:\Code\Swin-Unet-main\data1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./analysis_out/weak_fn_signal_table")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--high_threshold", type=float, default=None)
    parser.add_argument("--flank_steps", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", type=int, default=2)
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
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument(
        "--highres_structure_fuse_stages",
        type=str,
        default="stage23",
        choices=["stage2", "stage3", "stage23"],
    )
    parser.add_argument("--highres_structure_detach_surface", action="store_true")
    return parser.parse_args()


def resize_like(x, reference, mode="bilinear"):
    if x is None:
        return None
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(x.float(), **kwargs)


def load_model(args, device):
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        for name in (
            "structure_profile",
            "disable_msfe_skip",
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
            "enable_highres_structure_stream",
            "highres_structure_channels",
            "highres_structure_fuse_stages",
            "highres_structure_detach_surface",
        ):
            if name in saved_args:
                setattr(args, name, saved_args[name])
    if checkpoint.get("structure_profile"):
        args.structure_profile = checkpoint["structure_profile"]

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
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
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


def neighbor_coords(y, x, height, width):
    for dy, dx in NEIGHBORS:
        yy = y + dy
        xx = x + dx
        if 0 <= yy < height and 0 <= xx < width:
            yield yy, xx


def collect_flank_side(start, skeleton, component, max_steps):
    height, width = skeleton.shape
    queue = deque([(start[0], start[1], 1)])
    visited = {start}
    coords = []
    while queue:
        y, x, dist = queue.popleft()
        coords.append((y, x))
        if dist >= max_steps:
            continue
        for yy, xx in neighbor_coords(y, x, height, width):
            if (yy, xx) in visited:
                continue
            if not skeleton[yy, xx] or component[yy, xx]:
                continue
            visited.add((yy, xx))
            queue.append((yy, xx, dist + 1))
    return coords


def get_component_flanks(component, skeleton, max_steps):
    height, width = skeleton.shape
    seeds = set()
    ys, xs = np.nonzero(component)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for yy, xx in neighbor_coords(y, x, height, width):
            if skeleton[yy, xx] and not component[yy, xx]:
                seeds.add((yy, xx))
    sides = []
    used = set()
    for seed in sorted(seeds):
        if seed in used:
            continue
        coords = collect_flank_side(seed, skeleton, component, max_steps)
        side_set = set(coords)
        used.update(side_set)
        sides.append((coords, side_set))
    return sides


def classify_component(row, high_threshold):
    has_two_flanks = row["flank_count"] >= 2
    both_high = has_two_flanks and row["min_flank_p"] >= high_threshold
    same_surface = row["surface_component_relation"] == "same"
    different_surface = row["surface_component_relation"] == "different"
    if both_high and different_surface:
        return "A_true_short_break"
    if both_high and same_surface:
        return "B_surface_still_connected"
    if not has_two_flanks or row["min_flank_p"] < high_threshold:
        return "C_weak_road_segment"
    return "D_ambiguous"


def weak_c_mask_for_sample(surface_prob, surface_pred, gt_mask, skeleton, flank_steps, high_threshold):
    break_mask = (skeleton & gt_mask & (~surface_pred)).astype(np.uint8)
    surface_labels = cv2.connectedComponents(surface_pred.astype(np.uint8), connectivity=8)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(break_mask, connectivity=8)
    weak_mask = np.zeros_like(break_mask, dtype=bool)
    components = 0
    pixels = 0
    for label in range(1, num_labels):
        component = labels == label
        sides = get_component_flanks(component, skeleton, flank_steps)
        side_infos = []
        for coords, _ in sides:
            values = np.array([surface_prob[y, x] for y, x in coords], dtype=np.float32)
            pred_labels = [int(surface_labels[y, x]) for y, x in coords if surface_pred[y, x]]
            positive_labels = [value for value in pred_labels if value > 0]
            side_infos.append(
                {
                    "mean_p": float(values.mean()) if values.size else 0.0,
                    "max_p": float(values.max()) if values.size else 0.0,
                    "surface_label": max(set(positive_labels), key=positive_labels.count)
                    if positive_labels
                    else 0,
                }
            )
        side_infos.sort(key=lambda item: item["max_p"], reverse=True)
        chosen = side_infos[:2]
        if len(chosen) >= 2:
            min_flank_p = min(chosen[0]["mean_p"], chosen[1]["mean_p"])
            labels_pair = (chosen[0]["surface_label"], chosen[1]["surface_label"])
            if labels_pair[0] == 0 or labels_pair[1] == 0:
                relation = "missing_side_prediction"
            elif labels_pair[0] == labels_pair[1]:
                relation = "same"
            else:
                relation = "different"
        elif len(chosen) == 1:
            min_flank_p = 0.0
            relation = "one_flank"
        else:
            min_flank_p = 0.0
            relation = "no_flank"
        row = {
            "flank_count": len(side_infos),
            "min_flank_p": min_flank_p,
            "surface_component_relation": relation,
        }
        if classify_component(row, high_threshold) == "C_weak_road_segment":
            weak_mask |= component
            components += 1
            pixels += int(stats[label, cv2.CC_STAT_AREA])
    return weak_mask, components, pixels


def select_stage_skeleton(structure_outputs, stage):
    candidates = [
        item
        for item in structure_outputs
        if item.get("stage") == stage and item.get("refinement_step", 1) == 1 and "skeleton" in item
    ]
    if not candidates:
        candidates = [
            item for item in structure_outputs if item.get("stage") == stage and "skeleton" in item
        ]
    return candidates[-1]["skeleton"] if candidates else None


def select_highres_skeleton(structure_outputs):
    for item in structure_outputs:
        if item.get("stage") == "highres_structure":
            return item.get("highres_structure_skeleton")
    return None


class RegionStats:
    def __init__(self):
        self.regions = ("weak_fn", "surface_tp", "surface_fp", "surface_tn")
        self.signals = {
            "surface_p": {"available": True, "sum": {r: 0.0 for r in self.regions}},
            "old_stage2_s": {"available": False, "sum": {r: 0.0 for r in self.regions}},
            "old_stage3_s": {"available": False, "sum": {r: 0.0 for r in self.regions}},
            "highres_structure_s": {"available": False, "sum": {r: 0.0 for r in self.regions}},
        }
        self.count = {r: 0 for r in self.regions}

    def add_signal(self, name, prob, masks):
        if prob is None:
            return
        self.signals[name]["available"] = True
        prob_np = prob.detach().cpu().float().numpy()
        for region, mask in masks.items():
            if mask.any():
                self.signals[name]["sum"][region] += float(prob_np[mask].sum())

    def add_counts(self, masks):
        for region, mask in masks.items():
            self.count[region] += int(mask.sum())

    def rows(self):
        rows = []
        for name, item in self.signals.items():
            row = {"signal": name, "available": item["available"]}
            for region in self.regions:
                if item["available"] and self.count[region] > 0:
                    row[region] = item["sum"][region] / self.count[region]
                else:
                    row[region] = ""
            rows.append(row)
        return rows


def main():
    args = parse_args()
    high_threshold = args.high_threshold if args.high_threshold is not None else args.threshold
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
    stats = RegionStats()
    weak_components = 0
    weak_pixels = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            masks = (batch["mask"].to(device) > 0.5)
            skeletons = resize_like(batch["skeleton"].to(device), masks, mode="nearest") > 0.5
            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0]
            surface_prob = torch.sigmoid(surface_logits)
            surface_pred = surface_prob >= args.threshold
            structure_outputs = outputs[4] if len(outputs) > 4 else []

            stage2_prob = resize_like(select_stage_skeleton(structure_outputs, 2), surface_prob)
            stage3_prob = resize_like(select_stage_skeleton(structure_outputs, 3), surface_prob)
            highres_prob = resize_like(select_highres_skeleton(structure_outputs), surface_prob)
            if stage2_prob is not None:
                stage2_prob = torch.sigmoid(stage2_prob)
            if stage3_prob is not None:
                stage3_prob = torch.sigmoid(stage3_prob)
            if highres_prob is not None:
                highres_prob = torch.sigmoid(highres_prob)

            for item_idx in range(images.shape[0]):
                surface_np = surface_prob[item_idx, 0].detach().cpu().numpy()
                pred_np = surface_pred[item_idx, 0].detach().cpu().numpy().astype(bool)
                gt_np = masks[item_idx, 0].detach().cpu().numpy().astype(bool)
                sk_np = skeletons[item_idx, 0].detach().cpu().numpy().astype(bool)
                weak_np, comp_count, pix_count = weak_c_mask_for_sample(
                    surface_np,
                    pred_np,
                    gt_np,
                    sk_np,
                    args.flank_steps,
                    high_threshold,
                )
                weak_components += comp_count
                weak_pixels += pix_count
                region_masks = {
                    "weak_fn": weak_np,
                    "surface_tp": pred_np & gt_np,
                    "surface_fp": pred_np & (~gt_np),
                    "surface_tn": (~pred_np) & (~gt_np),
                }
                stats.add_counts(region_masks)
                stats.add_signal("surface_p", surface_prob[item_idx, 0], region_masks)
                if stage2_prob is not None:
                    stats.add_signal("old_stage2_s", stage2_prob[item_idx, 0], region_masks)
                if stage3_prob is not None:
                    stats.add_signal("old_stage3_s", stage3_prob[item_idx, 0], region_masks)
                if highres_prob is not None:
                    stats.add_signal("highres_structure_s", highres_prob[item_idx, 0], region_masks)

            if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == len(loader):
                print(
                    f"{batch_idx + 1}/{len(loader)} weak_components={weak_components} weak_pixels={weak_pixels}",
                    flush=True,
                )

    table_path = os.path.join(args.output_dir, "weak_fn_signal_table.csv")
    counts_path = os.path.join(args.output_dir, "weak_fn_region_counts.csv")
    rows = stats.rows()
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["signal", "available", "weak_fn", "surface_tp", "surface_fp", "surface_tn"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(counts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "pixels"])
        writer.writeheader()
        for region, pixels in stats.count.items():
            writer.writerow({"region": region, "pixels": pixels})
        writer.writerow({"region": "weak_fn_components", "pixels": weak_components})

    print("\nRegion pixel counts")
    for region, pixels in stats.count.items():
        print(f"{region}: {pixels}")
    print(f"weak_fn_components: {weak_components}")
    print("\nMean probability table")
    print("{:<24} {:<10} {:>12} {:>12} {:>12} {:>12}".format("signal", "available", "weak FN", "TP", "FP", "TN"))
    print("-" * 88)
    for row in rows:
        values = []
        for region in ("weak_fn", "surface_tp", "surface_fp", "surface_tn"):
            value = row[region]
            values.append("" if value == "" else f"{float(value):.6f}")
        print(
            "{:<24} {:<10} {:>12} {:>12} {:>12} {:>12}".format(
                row["signal"],
                str(row["available"]),
                *values,
            )
        )
    print(f"\nSaved: {table_path}")
    print(f"Saved: {counts_path}")


if __name__ == "__main__":
    main()
