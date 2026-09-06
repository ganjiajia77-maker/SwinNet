import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_transformer import load_topology_checkpoint_state


class ConfigArgs:
    root_path = "./data1"
    dataset = "ImageData"
    list_dir = "./lists/lists_Synapse"
    num_classes = 2
    cfg = "./configs/swin_tiny_patch4_window7_224_lite.yaml"
    img_size = 256
    batch_size = 1
    num_workers = 0
    zip = False
    cache_mode = ""
    resume = ""
    accumulation_steps = 0
    use_checkpoint = False
    amp_opt_level = ""
    tag = ""
    eval = True
    throughput = False
    n_class = 2
    opts = None


def update_args_from_checkpoint(args, checkpoint):
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        for name, value in saved_args.items():
            setattr(args, name, value)
    return args


def build_model(args, checkpoint, device):
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=getattr(args, "bottleneck_type", "global_local"),
        structure_profile=getattr(args, "structure_profile", "full"),
        use_msfe_skip=not getattr(args, "disable_msfe_skip", False),
        enable_highres_structure_stream=getattr(
            args,
            "enable_highres_structure_stream",
            False,
        ),
        highres_structure_channels=getattr(args, "highres_structure_channels", 64),
        highres_structure_fuse_stages=getattr(
            args,
            "highres_structure_fuse_stages",
            "stage23",
        ),
        highres_structure_fusion_mode=getattr(
            args,
            "highres_structure_fusion_mode",
            "stage23",
        ),
        enable_post_refine_structure_interaction=getattr(
            args,
            "enable_post_refine_structure_interaction",
            False,
        ),
        enable_global_topology=True,
        global_topology_max_nodes=getattr(args, "global_topology_max_nodes", 32),
        global_topology_heads=getattr(args, "global_topology_heads", 4),
        global_topology_reach_hops=getattr(args, "global_topology_reach_hops", 12),
        global_topology_nms_radius=getattr(args, "global_topology_nms_radius", 2),
        global_topology_skeleton_threshold=getattr(
            args,
            "global_topology_skeleton_threshold",
            0.5,
        ),
        global_topology_connectivity_threshold=getattr(
            args,
            "global_topology_connectivity_threshold",
            0.25,
        ),
        global_topology_bend_angle_threshold=getattr(
            args,
            "global_topology_bend_angle_threshold",
            45.0,
        ),
        global_topology_alpha_max=getattr(args, "global_topology_alpha_max", 0.05),
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=True,
    )
    return model.to(device).eval()


def core_swin(model):
    module = model.module if hasattr(model, "module") else model
    return module.swin_unet


def set_global_topology_enabled(model, enabled):
    swin = core_swin(model)
    previous = (
        bool(getattr(swin, "enable_global_topology", False)),
        bool(getattr(swin.global_topology, "enable_global_topology", False)),
    )
    swin.enable_global_topology = bool(enabled)
    swin.global_topology.enable_global_topology = bool(enabled)
    return previous


def restore_global_topology_enabled(model, previous):
    swin = core_swin(model)
    swin.enable_global_topology = previous[0]
    swin.global_topology.enable_global_topology = previous[1]


def scalar(value):
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float("nan")
        return float(value.float().mean().detach().cpu().item())
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=np.float32)
        return float(array.mean()) if array.size else float("nan")
    return float(value)


def forward_surface(model, images, enabled, capture=False):
    previous = set_global_topology_enabled(model, enabled)
    swin = core_swin(model)
    old_capture = bool(getattr(swin.global_topology, "capture_diagnostics", False))
    swin.global_topology.capture_diagnostics = bool(capture)
    swin.global_topology.last_diagnostics = None
    try:
        outputs = model(images)
        if isinstance(outputs, tuple):
            surface_logits = outputs[0]
        else:
            surface_logits = outputs
        diagnostics = swin.global_topology.last_diagnostics or {}
    finally:
        swin.global_topology.capture_diagnostics = old_capture
        restore_global_topology_enabled(model, previous)
    return surface_logits, diagnostics


def count_transitions(base_pred, global_pred, gt):
    base_pos = base_pred > 0
    global_pos = global_pred > 0
    gt_pos = gt > 0
    return {
        "baseline_fn_to_tp": int((~base_pos & gt_pos & global_pos).sum().item()),
        "baseline_tp_to_fn": int((base_pos & gt_pos & ~global_pos).sum().item()),
        "baseline_fp_to_tn": int((base_pos & ~gt_pos & ~global_pos).sum().item()),
        "baseline_tn_to_fp": int((~base_pos & ~gt_pos & global_pos).sum().item()),
        "baseline_tp_stay_tp": int((base_pos & gt_pos & global_pos).sum().item()),
        "baseline_fn_stay_fn": int((~base_pos & gt_pos & ~global_pos).sum().item()),
        "baseline_fp_stay_fp": int((base_pos & ~gt_pos & global_pos).sum().item()),
        "baseline_tn_stay_tn": int((~base_pos & ~gt_pos & ~global_pos).sum().item()),
    }


def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou


def positive_counts(pred, gt):
    pred_pos = pred > 0
    gt_pos = gt > 0
    tp = int((pred_pos & gt_pos).sum().item())
    fp = int((pred_pos & ~gt_pos).sum().item())
    fn = int((~pred_pos & gt_pos).sum().item())
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether the global topology residual is active and which "
            "confusion regions it changes."
        )
    )
    parser.add_argument("--checkpoint", "--model_path", dest="checkpoint", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[global-effect] device={device}", flush=True)
    print(f"[global-effect] checkpoint={args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    config_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    start = time.perf_counter()
    model = build_model(config_args, checkpoint, device)
    print(f"[global-effect] model ready in {time.perf_counter() - start:.2f}s", flush=True)

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[global-effect] dataset size={len(dataset)} threshold={args.threshold}", flush=True)

    rows = []
    diag_rows = []
    totals = {
        "baseline_tp": 0,
        "baseline_fp": 0,
        "baseline_fn": 0,
        "global_tp": 0,
        "global_fp": 0,
        "global_fn": 0,
        "baseline_fn_to_tp": 0,
        "baseline_tp_to_fn": 0,
        "baseline_fp_to_tn": 0,
        "baseline_tn_to_fp": 0,
        "baseline_tp_stay_tp": 0,
        "baseline_fn_stay_fn": 0,
        "baseline_fp_stay_fp": 0,
        "baseline_tn_stay_tn": 0,
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Global effect")):
            if args.max_batches and batch_index >= args.max_batches:
                break
            should_print = args.print_freq > 0 and batch_index % args.print_freq == 0
            images = batch["image"].to(device)
            masks = (batch["mask"].to(device) > 0.5)

            base_logits, _ = forward_surface(model, images, enabled=False, capture=False)
            global_logits, diagnostics = forward_surface(
                model,
                images,
                enabled=True,
                capture=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            base_prob = torch.sigmoid(base_logits)
            global_prob = torch.sigmoid(global_logits)
            base_pred = base_prob >= args.threshold
            global_pred = global_prob >= args.threshold
            logit_delta = global_logits - base_logits

            image_names = batch.get("image_name", [""] * images.shape[0])
            for sample_index in range(images.shape[0]):
                image_id = os.path.splitext(str(image_names[sample_index]))[0]
                gt = masks[sample_index : sample_index + 1]
                base_sample = base_pred[sample_index : sample_index + 1]
                global_sample = global_pred[sample_index : sample_index + 1]
                delta_sample = logit_delta[sample_index : sample_index + 1]
                transitions = count_transitions(base_sample, global_sample, gt)
                base_tp, base_fp, base_fn = positive_counts(base_sample, gt)
                global_tp, global_fp, global_fn = positive_counts(global_sample, gt)
                base_precision, base_recall, base_f1, base_iou = metrics_from_counts(
                    base_tp,
                    base_fp,
                    base_fn,
                )
                global_precision, global_recall, global_f1, global_iou = metrics_from_counts(
                    global_tp,
                    global_fp,
                    global_fn,
                )

                row = {
                    "image_id": image_id,
                    "threshold": args.threshold,
                    "baseline_iou": base_iou,
                    "baseline_f1": base_f1,
                    "baseline_precision": base_precision,
                    "baseline_recall": base_recall,
                    "global_iou": global_iou,
                    "global_f1": global_f1,
                    "global_precision": global_precision,
                    "global_recall": global_recall,
                    "delta_iou": global_iou - base_iou,
                    "delta_f1": global_f1 - base_f1,
                    "baseline_tp": base_tp,
                    "baseline_fp": base_fp,
                    "baseline_fn": base_fn,
                    "global_tp": global_tp,
                    "global_fp": global_fp,
                    "global_fn": global_fn,
                    **transitions,
                    "mean_logit_delta": float(delta_sample.mean().detach().cpu().item()),
                    "mean_prob_delta": float(
                        (global_prob[sample_index : sample_index + 1]
                         - base_prob[sample_index : sample_index + 1])
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                }
                rows.append(row)
                for name in totals:
                    if name in row:
                        totals[name] += int(row[name])

            diag_row = {
                "batch_index": batch_index,
                "alpha_global": scalar(diagnostics.get("alpha_global")),
                "attention_output_norm": scalar(diagnostics.get("attention_output_norm")),
                "attention_output_relative_norm": scalar(
                    diagnostics.get("attention_output_relative_norm")
                ),
                "delta_feature_norm": scalar(diagnostics.get("delta_feature_norm")),
                "delta_feature_relative_norm": scalar(
                    diagnostics.get("delta_feature_relative_norm")
                ),
                "global_residual_relative_norm": scalar(
                    diagnostics.get("global_residual_relative_norm")
                ),
                "anchor_count": scalar(diagnostics.get("anchor_count")),
                "candidate_count": scalar(diagnostics.get("candidate_count")),
                "anchor_score_mean": scalar(diagnostics.get("anchor_score_mean")),
                "anchor_score_max": scalar(diagnostics.get("anchor_score_max")),
            }
            diag_rows.append(diag_row)

            if should_print:
                print(
                    "[global-effect] batch {} alpha={:.6g} delta_rel={:.6g} "
                    "residual_rel={:.6g} anchors={:.1f}".format(
                        batch_index + 1,
                        diag_row["alpha_global"],
                        diag_row["delta_feature_relative_norm"],
                        diag_row["global_residual_relative_norm"],
                        diag_row["anchor_count"],
                    ),
                    flush=True,
                )

    per_image_path = os.path.join(args.output_dir, "global_topology_effect_per_image.csv")
    with open(per_image_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    diag_path = os.path.join(args.output_dir, "global_topology_alpha_delta.csv")
    with open(diag_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(diag_rows[0].keys()) if diag_rows else [],
        )
        if diag_rows:
            writer.writeheader()
            writer.writerows(diag_rows)

    base_precision, base_recall, base_f1, base_iou = metrics_from_counts(
        totals["baseline_tp"],
        totals["baseline_fp"],
        totals["baseline_fn"],
    )
    global_precision, global_recall, global_f1, global_iou = metrics_from_counts(
        totals["global_tp"],
        totals["global_fp"],
        totals["global_fn"],
    )
    diag_means = {
        key: float(np.nanmean([row[key] for row in diag_rows])) if diag_rows else float("nan")
        for key in (
            "alpha_global",
            "attention_output_norm",
            "attention_output_relative_norm",
            "delta_feature_norm",
            "delta_feature_relative_norm",
            "global_residual_relative_norm",
            "anchor_count",
            "candidate_count",
            "anchor_score_mean",
            "anchor_score_max",
        )
    }
    summary = {
        "threshold": args.threshold,
        "num_images": len(rows),
        "baseline_iou": base_iou,
        "baseline_f1": base_f1,
        "baseline_precision": base_precision,
        "baseline_recall": base_recall,
        "global_iou": global_iou,
        "global_f1": global_f1,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "delta_iou": global_iou - base_iou,
        "delta_f1": global_f1 - base_f1,
        **totals,
        **{f"mean_{key}": value for key, value in diag_means.items()},
    }
    summary_path = os.path.join(args.output_dir, "global_topology_effect_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n[global-effect] summary")
    for key in (
        "baseline_iou",
        "global_iou",
        "delta_iou",
        "baseline_fn_to_tp",
        "baseline_tp_to_fn",
        "baseline_fp_to_tn",
        "baseline_tn_to_fp",
        "mean_alpha_global",
        "mean_delta_feature_relative_norm",
        "mean_global_residual_relative_norm",
    ):
        print(f"{key}={summary[key]}", flush=True)
    print(f"saved: {summary_path}", flush=True)
    print(f"saved: {per_image_path}", flush=True)
    print(f"saved: {diag_path}", flush=True)


if __name__ == "__main__":
    main()
