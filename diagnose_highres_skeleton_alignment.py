import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import soft_skeletonize
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
    return args, saved_args if isinstance(saved_args, dict) else {}


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
        enable_highres_structure_stream=getattr(args, "enable_highres_structure_stream", False),
        highres_structure_channels=getattr(args, "highres_structure_channels", 64),
        highres_structure_fuse_stages=getattr(args, "highres_structure_fuse_stages", "stage23"),
        highres_structure_fusion_mode=getattr(args, "highres_structure_fusion_mode", "stage23"),
        enable_post_refine_structure_interaction=getattr(
            args,
            "enable_post_refine_structure_interaction",
            False,
        ),
        enable_global_topology=getattr(args, "enable_global_topology", True),
        global_topology_max_nodes=getattr(args, "global_topology_max_nodes", 32),
        global_topology_heads=getattr(args, "global_topology_heads", 4),
        global_topology_reach_hops=getattr(args, "global_topology_reach_hops", 12),
        global_topology_nms_radius=getattr(args, "global_topology_nms_radius", 2),
        global_topology_skeleton_threshold=getattr(args, "global_topology_skeleton_threshold", 0.5),
        global_topology_connectivity_threshold=getattr(args, "global_topology_connectivity_threshold", 0.25),
        global_topology_bend_angle_threshold=getattr(args, "global_topology_bend_angle_threshold", 45.0),
        global_topology_alpha_max=getattr(args, "global_topology_alpha_max", 0.05),
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=True,
    )
    return model.to(device).eval()


def find_highres_structure_skeleton(stage_outputs):
    for item in reversed(stage_outputs):
        if item.get("stage") == "highres_structure":
            skeleton = item.get("highres_structure_skeleton")
            if skeleton is not None:
                return skeleton
    return None


def find_final_skeleton(outputs):
    if len(outputs) > 2 and outputs[2] is not None:
        return outputs[2]
    return None


def resolve_skeleton_logits(outputs, source):
    stage_outputs = outputs[4] if len(outputs) > 4 else []
    if source == "highres":
        logits = find_highres_structure_skeleton(stage_outputs)
        if logits is None:
            raise RuntimeError("No highres_structure_skeleton found in stage outputs.")
        return logits, "highres_structure_skeleton"
    if source == "final":
        logits = find_final_skeleton(outputs)
        if logits is None:
            raise RuntimeError("No final skeleton logits found in model outputs.")
        return logits, "final_skeleton_logits"
    raise ValueError(f"Unknown skeleton source: {source}")


def parse_thresholds(raw):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("--thresholds must contain at least one value.")
    return values


def binary_counts(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    return intersection, pred_sum, target_sum


def binary_metrics(pred, target, eps=1e-6):
    intersection, pred_sum, target_sum = binary_counts(pred, target)
    precision = intersection / pred_sum.clamp_min(eps)
    recall = intersection / target_sum.clamp_min(eps)
    dice = (2.0 * intersection) / (pred_sum + target_sum).clamp_min(eps)
    return precision, recall, dice, intersection, pred_sum, target_sum


def soft_cldice_score(prob, target, iterations=10, smooth=1.0):
    prob = prob.float().clamp(0.0, 1.0)
    target = (target > 0.5).float()
    pred_skel = soft_skeletonize(prob, iterations=iterations)
    target_skel = soft_skeletonize(target, iterations=iterations)
    dims = (1, 2, 3)
    tprec = (pred_skel * target).sum(dims) / (pred_skel.sum(dims) + smooth)
    tsens = (target_skel * prob).sum(dims) / (target_skel.sum(dims) + smooth)
    return (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)


def main():
    parser = argparse.ArgumentParser(
        description="Measure skeleton-head alignment against GT skeleton and dilated skeleton."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--source", choices=("highres", "final"), default="highres")
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--cldice_iterations", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[skel-align] device={device}", flush=True)
    print(f"[skel-align] loading checkpoint: {args.checkpoint}", flush=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[skel-align] checkpoint loaded in {time.perf_counter() - start_time:.2f}s", flush=True)

    config_args, saved_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    print("[skel-align] building model", flush=True)
    start_time = time.perf_counter()
    model = build_model(config_args, checkpoint, device)
    print(f"[skel-align] model ready in {time.perf_counter() - start_time:.2f}s", flush=True)

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
    print(f"[skel-align] dataset size={len(dataset)}", flush=True)

    aggregate = {
        threshold: {
            "hard_intersection": 0.0,
            "hard_pred": 0.0,
            "hard_target": 0.0,
            "dilate_intersection": 0.0,
            "dilate_pred": 0.0,
            "dilate_target": 0.0,
            "cldice_hard_sum": 0.0,
            "cldice_dilate_sum": 0.0,
            "samples": 0,
        }
        for threshold in thresholds
    }
    rows = []
    source_counts = {}

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            should_print = args.print_freq > 0 and batch_index % args.print_freq == 0
            if should_print:
                print(f"[skel-align] batch {batch_index + 1} forward start", flush=True)

            images = batch["image"].to(device)
            gt_hard = batch["skeleton"].to(device).float()
            gt_dilate = batch["skeleton_dilate"].to(device).float()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logits, source_name = resolve_skeleton_logits(outputs, args.source)
            prob = torch.sigmoid(logits)
            if prob.shape[-2:] != gt_hard.shape[-2:]:
                prob = F.interpolate(
                    prob,
                    size=gt_hard.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            source_counts[source_name] = source_counts.get(source_name, 0) + int(prob.shape[0])

            cldice_hard = soft_cldice_score(
                prob,
                gt_hard,
                iterations=args.cldice_iterations,
            )
            cldice_dilate = soft_cldice_score(
                prob,
                gt_dilate,
                iterations=args.cldice_iterations,
            )

            image_names = batch.get("image_name", [""] * prob.shape[0])
            for threshold in thresholds:
                pred = (prob >= threshold).float()
                (
                    hard_precision,
                    hard_recall,
                    hard_dice,
                    hard_intersection,
                    hard_pred,
                    hard_target,
                ) = binary_metrics(pred, gt_hard)
                (
                    dilate_precision,
                    dilate_recall,
                    dilate_dice,
                    dilate_intersection,
                    dilate_pred,
                    dilate_target,
                ) = binary_metrics(pred, gt_dilate)

                stats = aggregate[threshold]
                stats["hard_intersection"] += float(hard_intersection.sum().item())
                stats["hard_pred"] += float(hard_pred.sum().item())
                stats["hard_target"] += float(hard_target.sum().item())
                stats["dilate_intersection"] += float(dilate_intersection.sum().item())
                stats["dilate_pred"] += float(dilate_pred.sum().item())
                stats["dilate_target"] += float(dilate_target.sum().item())
                stats["cldice_hard_sum"] += float(cldice_hard.sum().item())
                stats["cldice_dilate_sum"] += float(cldice_dilate.sum().item())
                stats["samples"] += int(prob.shape[0])

                for sample_index in range(prob.shape[0]):
                    rows.append(
                        {
                            "batch_index": batch_index,
                            "sample_index": sample_index,
                            "image_name": image_names[sample_index],
                            "source": source_name,
                            "threshold": threshold,
                            "pred_pixels": int(hard_pred[sample_index].item()),
                            "gt_skeleton_pixels": int(hard_target[sample_index].item()),
                            "gt_skeleton_dilate_pixels": int(dilate_target[sample_index].item()),
                            "precision_hard": float(hard_precision[sample_index].item()),
                            "recall_hard": float(hard_recall[sample_index].item()),
                            "dice_hard": float(hard_dice[sample_index].item()),
                            "precision_dilate": float(dilate_precision[sample_index].item()),
                            "recall_dilate": float(dilate_recall[sample_index].item()),
                            "dice_dilate": float(dilate_dice[sample_index].item()),
                            "soft_cldice_hard": float(cldice_hard[sample_index].item()),
                            "soft_cldice_dilate": float(cldice_dilate[sample_index].item()),
                            "prob_mean": float(prob[sample_index].mean().item()),
                            "prob_max": float(prob[sample_index].max().item()),
                        }
                    )

            if should_print:
                pred_counts = [int((prob >= threshold).sum().item()) for threshold in thresholds]
                print(
                    "[skel-align] batch {} done: source={} shape={}x{} mean={:.6f} max={:.6f} pred_pixels_by_thr={}".format(
                        batch_index + 1,
                        source_name,
                        int(prob.shape[-2]),
                        int(prob.shape[-1]),
                        float(prob.mean().item()),
                        float(prob.max().item()),
                        pred_counts,
                    ),
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No batches were processed.")

    summary_rows = []
    for threshold in thresholds:
        stats = aggregate[threshold]
        eps = 1e-6
        hard_precision = stats["hard_intersection"] / max(stats["hard_pred"], eps)
        hard_recall = stats["hard_intersection"] / max(stats["hard_target"], eps)
        hard_dice = (
            2.0 * stats["hard_intersection"]
            / max(stats["hard_pred"] + stats["hard_target"], eps)
        )
        dilate_precision = stats["dilate_intersection"] / max(stats["dilate_pred"], eps)
        dilate_recall = stats["dilate_intersection"] / max(stats["dilate_target"], eps)
        dilate_dice = (
            2.0 * stats["dilate_intersection"]
            / max(stats["dilate_pred"] + stats["dilate_target"], eps)
        )
        samples = max(stats["samples"], 1)
        summary_rows.append(
            {
                "source": args.source,
                "threshold": threshold,
                "samples": stats["samples"],
                "precision_hard": hard_precision,
                "recall_hard": hard_recall,
                "dice_hard": hard_dice,
                "precision_dilate": dilate_precision,
                "recall_dilate": dilate_recall,
                "dice_dilate": dilate_dice,
                "soft_cldice_hard_mean": stats["cldice_hard_sum"] / samples,
                "soft_cldice_dilate_mean": stats["cldice_dilate_sum"] / samples,
                "pred_pixels_total": int(stats["hard_pred"]),
                "gt_skeleton_pixels_total": int(stats["hard_target"]),
                "gt_skeleton_dilate_pixels_total": int(stats["dilate_target"]),
                "highres_structure_skeleton_weight": saved_args.get(
                    "highres_structure_skeleton_weight",
                    "not_saved_default_0.0",
                ),
                "final_skeleton_weight": saved_args.get("final_skeleton_weight", "not_saved"),
                "target_note": "BCE uses skeleton_dilate_gt; Dice uses skeleton_gt in skeleton_pixel_loss",
                "sources": ";".join(
                    f"{key}:{value}" for key, value in sorted(source_counts.items())
                ),
            }
        )

    cases_path = os.path.join(args.output_dir, "highres_skeleton_alignment_cases.csv")
    with open(cases_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(args.output_dir, "highres_skeleton_alignment_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nHigh-res skeleton alignment summary", flush=True)
    for row in summary_rows:
        print(
            "thr={threshold}: hard P/R/Dice={precision_hard:.4f}/{recall_hard:.4f}/{dice_hard:.4f}, "
            "dilate P/R/Dice={precision_dilate:.4f}/{recall_dilate:.4f}/{dice_dilate:.4f}, "
            "clDice hard/dilate={soft_cldice_hard_mean:.4f}/{soft_cldice_dilate_mean:.4f}".format(
                **row
            ),
            flush=True,
        )
    print(
        "highres_structure_skeleton_weight={}".format(
            summary_rows[0]["highres_structure_skeleton_weight"]
        ),
        flush=True,
    )
    print("target_note={}".format(summary_rows[0]["target_note"]), flush=True)
    print("saved:", summary_path, flush=True)
    print("saved:", cases_path, flush=True)


if __name__ == "__main__":
    main()
