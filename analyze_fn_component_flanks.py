import argparse
import csv
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
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
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            r"D:\Code\Swin-Unet-main\model_out"
            r"\data1_stage2_gate_a1a2_w008012_gr05_aux1024_dir_noconcons_swinv2_direct256_20260810"
            r"\best.pth"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./analysis_out/fn_component_flanks_data1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--flank_steps", type=int, default=3)
    parser.add_argument("--high_threshold", type=float, default=None)
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
    return parser.parse_args()


def resize_like(x, reference):
    if x.shape[-2:] == reference.shape[-2:]:
        return x.float()
    return torch.nn.functional.interpolate(
        x.float(),
        size=reference.shape[-2:],
        mode="nearest",
    )


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
        probs = [coords, side_set]
        sides.append(probs)
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


def bucket_name(size):
    if size <= 2:
        return "<=2px"
    if size <= 4:
        return "2-4px"
    if size <= 8:
        return "4-8px"
    if size <= 16:
        return "8-16px"
    return ">16px"


def analyze_sample(prob, pred, gt_mask, skeleton, case_name, threshold, flank_steps, high_threshold):
    break_mask = (skeleton & gt_mask & (~pred)).astype(np.uint8)
    pred_uint = pred.astype(np.uint8)
    num_surface, surface_labels = cv2.connectedComponents(pred_uint, connectivity=8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(break_mask, connectivity=8)
    rows = []
    for label in range(1, num_labels):
        component = labels == label
        size = int(stats[label, cv2.CC_STAT_AREA])
        gap_probs = prob[component]
        sides = get_component_flanks(component, skeleton, flank_steps)
        side_infos = []
        for coords, _ in sides:
            values = np.array([prob[y, x] for y, x in coords], dtype=np.float32)
            pred_labels = [int(surface_labels[y, x]) for y, x in coords if pred[y, x]]
            positive_labels = [value for value in pred_labels if value > 0]
            side_infos.append(
                {
                    "coords": coords,
                    "mean_p": float(values.mean()) if values.size else 0.0,
                    "max_p": float(values.max()) if values.size else 0.0,
                    "min_p": float(values.min()) if values.size else 0.0,
                    "surface_label": max(set(positive_labels), key=positive_labels.count)
                    if positive_labels
                    else 0,
                }
            )
        side_infos.sort(key=lambda item: item["max_p"], reverse=True)
        chosen = side_infos[:2]
        if len(chosen) >= 2:
            left_p = chosen[0]["mean_p"]
            right_p = chosen[1]["mean_p"]
            min_flank_p = min(left_p, right_p)
            labels_pair = (chosen[0]["surface_label"], chosen[1]["surface_label"])
            if labels_pair[0] == 0 or labels_pair[1] == 0:
                relation = "missing_side_prediction"
            elif labels_pair[0] == labels_pair[1]:
                relation = "same"
            else:
                relation = "different"
        elif len(chosen) == 1:
            left_p = chosen[0]["mean_p"]
            right_p = 0.0
            min_flank_p = 0.0
            relation = "one_flank"
            labels_pair = (chosen[0]["surface_label"], 0)
        else:
            left_p = right_p = min_flank_p = 0.0
            relation = "no_flank"
            labels_pair = (0, 0)
        row = {
            "case_name": case_name,
            "component_id": label,
            "bucket": bucket_name(size),
            "size_pixels": size,
            "gap_mean_p": float(gap_probs.mean()) if gap_probs.size else 0.0,
            "gap_min_p": float(gap_probs.min()) if gap_probs.size else 0.0,
            "gap_max_p": float(gap_probs.max()) if gap_probs.size else 0.0,
            "flank_count": len(side_infos),
            "p_left_flank": left_p,
            "p_right_flank": right_p,
            "min_flank_p": min_flank_p,
            "surface_component_relation": relation,
            "left_surface_label": labels_pair[0],
            "right_surface_label": labels_pair[1],
        }
        row["type"] = classify_component(row, high_threshold)
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
    rows = []
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            masks = (batch["mask"].to(device) > 0.5)
            skeleton = resize_like(batch["skeleton"].to(device), masks) > 0.5
            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            logits = outputs[0]
            prob = torch.sigmoid(logits)
            pred = prob >= args.threshold
            tp += int((pred & masks).sum().item())
            fp += int((pred & (~masks)).sum().item())
            fn += int(((~pred) & masks).sum().item())
            tn += int(((~pred) & (~masks)).sum().item())
            for item_idx in range(images.shape[0]):
                case_name = batch.get("case_name", [f"batch{batch_idx}_item{item_idx}"])[item_idx]
                rows.extend(
                    analyze_sample(
                        prob[item_idx, 0].detach().cpu().numpy(),
                        pred[item_idx, 0].detach().cpu().numpy().astype(bool),
                        masks[item_idx, 0].detach().cpu().numpy().astype(bool),
                        skeleton[item_idx, 0].detach().cpu().numpy().astype(bool),
                        case_name,
                        args.threshold,
                        args.flank_steps,
                        high_threshold,
                    )
                )
            if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == len(loader):
                print(
                    f"{batch_idx + 1}/{len(loader)} components={len(rows)} TP={tp} FP={fp} FN={fn} TN={tn}",
                    flush=True,
                )

    details_path = os.path.join(args.output_dir, "fn_component_flank_details.csv")
    summary_path = os.path.join(args.output_dir, "fn_component_flank_summary.csv")
    fields = [
        "case_name",
        "component_id",
        "bucket",
        "type",
        "size_pixels",
        "gap_mean_p",
        "gap_min_p",
        "gap_max_p",
        "flank_count",
        "p_left_flank",
        "p_right_flank",
        "min_flank_p",
        "surface_component_relation",
        "left_surface_label",
        "right_surface_label",
    ]
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        key = (row["bucket"], row["type"])
        if key not in summary:
            summary[key] = {
                "components": 0,
                "pixels": 0,
                "gap_mean_sum": 0.0,
                "min_flank_sum": 0.0,
            }
        summary[key]["components"] += 1
        summary[key]["pixels"] += int(row["size_pixels"])
        summary[key]["gap_mean_sum"] += float(row["gap_mean_p"])
        summary[key]["min_flank_sum"] += float(row["min_flank_p"])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "type",
                "components",
                "pixels",
                "component_ratio",
                "pixel_ratio",
                "gap_mean_p",
                "min_flank_p",
            ],
        )
        writer.writeheader()
        total_components = len(rows)
        total_pixels = sum(int(row["size_pixels"]) for row in rows)
        for (bucket, type_name), values in sorted(summary.items()):
            components = values["components"]
            pixels = values["pixels"]
            writer.writerow(
                {
                    "bucket": bucket,
                    "type": type_name,
                    "components": components,
                    "pixels": pixels,
                    "component_ratio": components / max(total_components, 1),
                    "pixel_ratio": pixels / max(total_pixels, 1),
                    "gap_mean_p": values["gap_mean_sum"] / max(components, 1),
                    "min_flank_p": values["min_flank_sum"] / max(components, 1),
                }
            )

    print("\nSurface confusion")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"components={len(rows)}")
    print(f"Saved: {details_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
