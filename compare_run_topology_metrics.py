import argparse
import csv
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_structure_supervision import load_model, resize_like
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.cldice_loss import soft_skeletonize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--current_dir", type=str, required=True)
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--current_name", type=str, default="current")
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--current_threshold", type=float, default=None)
    parser.add_argument("--baseline_threshold", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--short_area_threshold", type=int, default=20)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
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
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=["auto", "standard", "selective"])
    return parser.parse_args()


def component_stats(mask_bool, short_area_threshold):
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.empty((0,), dtype=np.int32)
    total = float(areas.sum()) if areas.size else 0.0
    return {
        "components": float(num - 1),
        "short_components": float((areas < short_area_threshold).sum()) if areas.size else 0.0,
        "largest_ratio": float(areas.max() / total) if total > 0 else 0.0,
    }


def cldice_scores(surface_prob, surface_gt):
    pred_skel = soft_skeletonize(surface_prob.float().clamp(0.0, 1.0), iter_num=10)
    gt_skel = soft_skeletonize(surface_gt.float().clamp(0.0, 1.0), iter_num=10)
    tprec = (pred_skel * surface_gt).sum(dim=(1, 2, 3)) / (
        pred_skel.sum(dim=(1, 2, 3)) + 1e-8
    )
    tsens = (gt_skel * surface_prob).sum(dim=(1, 2, 3)) / (
        gt_skel.sum(dim=(1, 2, 3)) + 1e-8
    )
    return ((2.0 * tprec * tsens) / (tprec + tsens + 1e-8)).detach().cpu().numpy()


def make_model_args(base_args, model_path):
    values = vars(base_args).copy()
    values["model_path"] = model_path
    return SimpleNamespace(**values)


def evaluate_run(name, run_dir, threshold, args, loader, device):
    model_path = os.path.join(run_dir, args.checkpoint)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    model = load_model(make_model_args(args, model_path), device)
    tp = fp = fn = 0.0
    cldice_values = []
    pred_components = []
    gt_components = []
    frag_indices = []
    extra_components = []
    pred_short = []
    gt_short = []
    pred_largest = []
    gt_largest = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=name)):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).float()
            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            masks = resize_like(masks, surface_logits, mode="nearest")
            surface_prob = torch.sigmoid(surface_logits)
            pred = surface_prob >= threshold
            gt = masks > 0.5
            tp += float((pred & gt).sum().item())
            fp += float((pred & (~gt)).sum().item())
            fn += float(((~pred) & gt).sum().item())
            cldice_values.extend(cldice_scores(surface_prob, masks).tolist())

            pred_np = pred.detach().cpu().numpy()[:, 0]
            gt_np = gt.detach().cpu().numpy()[:, 0]
            for pred_mask, gt_mask in zip(pred_np, gt_np):
                ps = component_stats(pred_mask, args.short_area_threshold)
                gs = component_stats(gt_mask, args.short_area_threshold)
                pred_components.append(ps["components"])
                gt_components.append(gs["components"])
                frag_indices.append(ps["components"] / max(gs["components"], 1.0))
                extra_components.append(max(ps["components"] - gs["components"], 0.0))
                pred_short.append(ps["short_components"])
                gt_short.append(gs["short_components"])
                pred_largest.append(ps["largest_ratio"])
                gt_largest.append(gs["largest_ratio"])

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "checkpoint": model_path,
        "threshold": threshold,
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "cldice": float(np.mean(cldice_values)) if cldice_values else float("nan"),
        "pred_comp": float(np.mean(pred_components)) if pred_components else float("nan"),
        "gt_comp": float(np.mean(gt_components)) if gt_components else float("nan"),
        "frag_idx": float(np.mean(frag_indices)) if frag_indices else float("nan"),
        "extra_comp": float(np.mean(extra_components)) if extra_components else float("nan"),
        "pred_short": float(np.mean(pred_short)) if pred_short else float("nan"),
        "gt_short": float(np.mean(gt_short)) if gt_short else float("nan"),
        "pred_largest_ratio": float(np.mean(pred_largest)) if pred_largest else float("nan"),
        "gt_largest_ratio": float(np.mean(gt_largest)) if gt_largest else float("nan"),
    }


def print_rows(rows):
    print("\nMetric comparison")
    header = (
        "name",
        "thr",
        "IoU",
        "F1",
        "P",
        "R",
        "clDice",
        "frag_idx↓",
        "extra_comp↓",
        "pred_comp",
        "gt_comp",
        "pred_short↓",
        "largest_ratio",
    )
    print(" | ".join(header))
    for row in rows:
        print(
            f"{row['name']} | {row['threshold']:.3f} | {row['iou']:.4f} | "
            f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['cldice']:.4f} | {row['frag_idx']:.3f} | "
            f"{row['extra_comp']:.2f} | {row['pred_comp']:.2f} | "
            f"{row['gt_comp']:.2f} | {row['pred_short']:.2f} | "
            f"{row['pred_largest_ratio']:.3f}"
        )
    if len(rows) == 2:
        base, cur = rows[0], rows[1]
        print("\nDelta current - baseline")
        for key in ("iou", "f1", "cldice", "frag_idx", "extra_comp", "pred_short", "pred_largest_ratio"):
            print(f"  {key}: {cur[key] - base[key]:+.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        pin_memory=device.type == "cuda",
    )
    baseline_threshold = args.baseline_threshold if args.baseline_threshold is not None else args.threshold
    current_threshold = args.current_threshold if args.current_threshold is not None else args.threshold
    rows = [
        evaluate_run(args.baseline_name, args.baseline_dir, baseline_threshold, args, loader, device),
        evaluate_run(args.current_name, args.current_dir, current_threshold, args, loader, device),
    ]
    print_rows(rows)
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
