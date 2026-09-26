"""Paired validation ablation for the decoder Stage 2 topology attention bias."""

import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_skeleton_signal_probe import load_model


HISTOGRAM_BINS = 4096


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--output_dir",
        default="",
        help="Artifact directory; defaults to <output_csv stem>_artifacts.",
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        help="Shared thresholds for per-image clDice and road connectivity.",
    )
    parser.add_argument("--connectivity_max_anchors", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=0, help="0 evaluates the full validation set.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_model_args(args):
    # load_model uses these standard Swin configuration fields in addition to
    # the model/data paths; architecture options are recovered from checkpoint.
    values = vars(args).copy()
    values.update(
        {
            "output_dir": os.path.dirname(os.path.abspath(args.output_csv)),
            "fixed_iou_threshold": None,
            "probe_epochs": 10,
            "probe_batch_size": 8,
            "probe_lr": 1e-2,
            "probe_weight_decay": 1e-4,
            "probe_pos_weight_cap": 100.0,
            "num_classes": 1,
            "n_class": 2,
            "dataset": "ImageData",
            "opts": None,
            "zip": False,
            "cache_mode": "",
            "resume": "",
            "accumulation_steps": 0,
            "use_checkpoint": False,
            "amp_opt_level": "",
            "tag": "",
            "eval": False,
            "throughput": False,
        }
    )
    return SimpleNamespace(**values)


def set_stage2_bias_enabled(model, enabled):
    swin = model.swin_unet
    stage2_layer = swin.layers_up[2]
    stage2_layer.use_decoder_structure_bias = bool(enabled)
    for block in stage2_layer.blocks:
        block.use_decoder_structure_bias = bool(enabled)


def new_stats():
    return {
        "fixed_tp": 0,
        "fixed_fp": 0,
        "fixed_fn": 0,
        "positive_hist": np.zeros(HISTOGRAM_BINS, dtype=np.int64),
        "negative_hist": np.zeros(HISTOGRAM_BINS, dtype=np.int64),
        "pixels": 0,
        "positive_pixels": 0,
    }


def accumulate(stats, logits, target, fixed_threshold):
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy().reshape(-1)
    positive = target.detach().cpu().numpy().reshape(-1) > 0.5
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite surface probabilities found during validation.")

    prediction = probabilities >= fixed_threshold
    stats["fixed_tp"] += int(np.logical_and(prediction, positive).sum())
    stats["fixed_fp"] += int(np.logical_and(prediction, ~positive).sum())
    stats["fixed_fn"] += int(np.logical_and(~prediction, positive).sum())
    bins = np.minimum((probabilities * HISTOGRAM_BINS).astype(np.int32), HISTOGRAM_BINS - 1)
    stats["positive_hist"] += np.bincount(bins[positive], minlength=HISTOGRAM_BINS)
    stats["negative_hist"] += np.bincount(bins[~positive], minlength=HISTOGRAM_BINS)
    stats["pixels"] += int(positive.size)
    stats["positive_pixels"] += int(positive.sum())


def summarize(stats, fixed_threshold):
    tp = stats["fixed_tp"]
    fp = stats["fixed_fp"]
    fn = stats["fixed_fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fixed_iou = tp / max(tp + fp + fn, 1)
    fixed_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    positive_hist = stats["positive_hist"][::-1]
    negative_hist = stats["negative_hist"][::-1]
    tp_by_bin = np.cumsum(positive_hist, dtype=np.int64)
    fp_by_bin = np.cumsum(negative_hist, dtype=np.int64)
    total_positive = max(stats["positive_pixels"], 1)
    recall_by_bin = tp_by_bin / total_positive
    precision_by_bin = tp_by_bin / np.maximum(tp_by_bin + fp_by_bin, 1)
    recall_steps = np.diff(np.concatenate(([0.0], recall_by_bin)))
    auprc_binned = float(np.sum(recall_steps * precision_by_bin))
    iou_by_bin = tp_by_bin / np.maximum(total_positive + fp_by_bin, 1)
    best_index = int(np.argmax(iou_by_bin))
    best_iou = float(iou_by_bin[best_index])
    best_threshold = float(1.0 - (best_index + 0.5) / HISTOGRAM_BINS)

    return {
        "threshold": fixed_threshold,
        "iou": float(fixed_iou),
        "f1": float(fixed_f1),
        "precision": float(precision),
        "recall": float(recall),
        "auprc_binned_4096": auprc_binned,
        "best_iou": best_iou,
        "best_iou_threshold": best_threshold,
        "pixels": int(stats["pixels"]),
    }


def cldice_score(prediction, target):
    pred_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    pred_count = int(pred_skeleton.sum())
    target_count = int(target_skeleton.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0
    topology_precision = np.logical_and(pred_skeleton, target).sum() / pred_count
    topology_sensitivity = np.logical_and(target_skeleton, prediction).sum() / target_count
    denom = topology_precision + topology_sensitivity
    return float(2.0 * topology_precision * topology_sensitivity / denom) if denom else 0.0


def connected_pair_recall(prediction, target_skeleton, max_anchors=32, tolerance=2.0):
    """Fraction of sampled GT skeleton point pairs still connected by predicted road."""
    gt_count, gt_labels = cv2.connectedComponents(
        target_skeleton.astype(np.uint8), connectivity=8
    )
    pred_count, pred_labels = cv2.connectedComponents(
        prediction.astype(np.uint8), connectivity=8
    )
    if gt_count <= 1:
        return float("nan"), 0

    if pred_count <= 1:
        nearest_labels = np.zeros_like(gt_labels)
        distances = np.full(gt_labels.shape, np.inf, dtype=np.float32)
    else:
        distances, nearest = distance_transform_edt(
            ~prediction, return_indices=True
        )
        nearest_labels = pred_labels[nearest[0], nearest[1]]

    connected_pairs = 0
    total_pairs = 0
    for component_id in range(1, gt_count):
        coords = np.argwhere(gt_labels == component_id)
        if len(coords) < 2:
            continue
        if len(coords) > max_anchors:
            indices = np.linspace(0, len(coords) - 1, max_anchors, dtype=np.int64)
            coords = coords[indices]
        ys, xs = coords[:, 0], coords[:, 1]
        mapped_labels = nearest_labels[ys, xs]
        mapped_valid = (distances[ys, xs] <= tolerance) & (mapped_labels > 0)
        # Evaluate all anchor pairs within each GT-connected component.
        for left in range(len(coords) - 1):
            right_valid = mapped_valid[left + 1:]
            pair_count = len(right_valid)
            total_pairs += pair_count
            if not mapped_valid[left]:
                continue
            connected_pairs += int(
                np.logical_and(right_valid, mapped_labels[left + 1:] == mapped_labels[left]).sum()
            )
    if total_pairs == 0:
        return float("nan"), 0
    return connected_pairs / total_pairs, total_pairs


def save_prediction_artifacts(output_dir, image_name, full_prob, off_prob, target, threshold):
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    on_dir = os.path.join(output_dir, "probability_maps", "bias_on")
    off_dir = os.path.join(output_dir, "probability_maps", "bias_off")
    preview_dir = os.path.join(output_dir, "comparison_previews")
    os.makedirs(on_dir, exist_ok=True)
    os.makedirs(off_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    np.save(os.path.join(on_dir, f"{stem}.npy"), full_prob.astype(np.float32))
    np.save(os.path.join(off_dir, f"{stem}.npy"), off_prob.astype(np.float32))

    gt = target.astype(bool)
    on = full_prob >= threshold
    off = off_prob >= threshold
    # RGB panels: GT road is green; predicted road is red (on) or blue (off).
    gt_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    gt_rgb[gt, 1] = 255
    on_rgb = gt_rgb.copy()
    on_rgb[on, 0] = 255
    off_rgb = gt_rgb.copy()
    off_rgb[off, 2] = 255
    diff_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    diff_rgb[on & off] = (150, 150, 150)
    diff_rgb[on & ~off] = (255, 0, 0)
    diff_rgb[~on & off] = (0, 0, 255)
    panels = [gt_rgb, on_rgb, off_rgb, diff_rgb]
    panel = np.concatenate(panels, axis=1)
    panel_width = gt.shape[1]
    for index, label in enumerate(("GT", "Bias ON", "Bias OFF", "Changed pixels")):
        cv2.putText(
            panel,
            label,
            (index * panel_width + 5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(
        os.path.join(preview_dir, f"{stem}_thr{threshold:.2f}.png"),
        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
    )


def metric_row(probability, target, threshold, max_anchors):
    prediction = probability >= threshold
    target_skeleton = skeletonize(target)
    conn_recall, pair_count = connected_pair_recall(
        prediction,
        target_skeleton,
        max_anchors=max_anchors,
    )
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return {
        "threshold": float(threshold),
        "iou": intersection / max(union, 1),
        "cldice": cldice_score(prediction, target),
        "connected_pair_recall": conn_recall,
        "gt_connected_pairs": pair_count,
        "predicted_pixels": int(prediction.sum()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(make_model_args(args), device)
    model.eval()

    stage2_blocks = model.swin_unet.layers_up[2].blocks
    if not stage2_blocks or not all(block.use_decoder_structure_bias for block in stage2_blocks):
        raise RuntimeError(
            "Stage 2 decoder attention bias is not enabled in the loaded checkpoint architecture."
        )
    print(
        f"Stage 2 decoder bias blocks: {len(stage2_blocks)}; "
        "Stage 3 and all other paths remain enabled.",
        flush=True,
    )

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="val",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        tile_size=None,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    stats = {"full": new_stats(), "stage2_attention_bias_off": new_stats()}
    thresholds = sorted(set(float(t) for t in args.thresholds) | {float(args.threshold)})
    per_image_rows = []
    pixel_change = {float(t): {"on_only": 0, "off_only": 0, "changed": 0} for t in thresholds}
    image_names_seen = 0
    output_dir = args.output_dir or (
        os.path.splitext(os.path.abspath(args.output_csv))[0] + "_artifacts"
    )
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Paired Stage 2 bias ablation")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

            set_stage2_bias_enabled(model, True)
            full_outputs = model(images)
            if not isinstance(full_outputs, tuple):
                raise RuntimeError("Expected surface logits and auxiliary outputs from this checkpoint.")
            full_logits = full_outputs[0]

            try:
                set_stage2_bias_enabled(model, False)
                ablated_outputs = model(images)
            finally:
                set_stage2_bias_enabled(model, True)
            ablated_logits = ablated_outputs[0]

            accumulate(stats["full"], full_logits, targets, args.threshold)
            accumulate(stats["stage2_attention_bias_off"], ablated_logits, targets, args.threshold)

            full_probs = torch.sigmoid(full_logits.float()).squeeze(1).cpu().numpy()
            off_probs = torch.sigmoid(ablated_logits.float()).squeeze(1).cpu().numpy()
            target_arrays = targets.squeeze(1).cpu().numpy() > 0.5
            names = batch["image_name"]
            for index, image_name in enumerate(names):
                image_names_seen += 1
                full_prob = full_probs[index]
                off_prob = off_probs[index]
                target = target_arrays[index]
                save_prediction_artifacts(
                    output_dir, image_name, full_prob, off_prob, target, args.threshold
                )
                target_skeleton = skeletonize(target)
                for threshold in thresholds:
                    on_row = metric_row(
                        full_prob, target, threshold, args.connectivity_max_anchors
                    )
                    off_row = metric_row(
                        off_prob, target, threshold, args.connectivity_max_anchors
                    )
                    on_prediction = full_prob >= threshold
                    off_prediction = off_prob >= threshold
                    on_only = int(np.logical_and(on_prediction, ~off_prediction).sum())
                    off_only = int(np.logical_and(~on_prediction, off_prediction).sum())
                    pixel_change[threshold]["on_only"] += on_only
                    pixel_change[threshold]["off_only"] += off_only
                    pixel_change[threshold]["changed"] += on_only + off_only
                    common = {
                        "image": str(image_name),
                        "threshold": threshold,
                        "gt_skeleton_components": int(
                            cv2.connectedComponents(
                                target_skeleton.astype(np.uint8), connectivity=8
                            )[0] - 1
                        ),
                    }
                    per_image_rows.append({"mode": "bias_on", **common, **on_row})
                    per_image_rows.append({"mode": "bias_off", **common, **off_row})
                    delta_row = {
                        f"{key}_delta": on_row[key] - off_row[key]
                        for key in on_row
                        if key not in {"threshold", "gt_connected_pairs"}
                    }
                    per_image_rows.append(
                        {
                            "mode": "bias_on_minus_off",
                            **common,
                            **delta_row,
                            "changed_pixels": on_only + off_only,
                            "bias_on_only_pixels": on_only,
                            "bias_off_only_pixels": off_only,
                        }
                    )

    summaries = {
        mode: summarize(mode_stats, args.threshold)
        for mode, mode_stats in stats.items()
    }
    metric_keys = list(summaries["full"].keys())
    rows = [{"mode": mode, **summary} for mode, summary in summaries.items()]
    rows.append(
        {
            "mode": "bias_off_minus_full",
            **{
                key: (
                    summaries["stage2_attention_bias_off"][key] - summaries["full"][key]
                    if key != "pixels"
                    else 0
                )
                for key in metric_keys
            },
        }
    )

    output_path = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", *metric_keys])
        writer.writeheader()
        writer.writerows(rows)

    per_image_path = os.path.splitext(output_path)[0] + "_per_image.csv"
    with open(per_image_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(
            key for row in per_image_rows for key in row
        ))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    topology_summary = []
    for threshold in thresholds:
        on_rows = [
            row for row in per_image_rows
            if row["mode"] == "bias_on" and row["threshold"] == threshold
        ]
        off_rows = [
            row for row in per_image_rows
            if row["mode"] == "bias_off" and row["threshold"] == threshold
        ]
        deltas = [
            row for row in per_image_rows
            if row["mode"] == "bias_on_minus_off" and row["threshold"] == threshold
        ]
        summary_row = {
            "threshold": threshold,
            "images": len(deltas),
            "changed_pixels": pixel_change[threshold]["changed"],
            "bias_on_only_pixels": pixel_change[threshold]["on_only"],
            "bias_off_only_pixels": pixel_change[threshold]["off_only"],
        }
        for metric in ("iou", "cldice", "connected_pair_recall"):
            on_values = np.asarray([row[metric] for row in on_rows], dtype=np.float64)
            off_values = np.asarray([row[metric] for row in off_rows], dtype=np.float64)
            delta_values = np.asarray(
                [row[f"{metric}_delta"] for row in deltas], dtype=np.float64
            )
            valid_delta = delta_values[np.isfinite(delta_values)]
            summary_row[f"bias_on_mean_{metric}"] = float(np.nanmean(on_values)) if on_values.size else float("nan")
            summary_row[f"bias_off_mean_{metric}"] = float(np.nanmean(off_values)) if off_values.size else float("nan")
            summary_row[f"mean_delta_{metric}"] = float(np.nanmean(delta_values)) if delta_values.size else float("nan")
            summary_row[f"improved_images_{metric}"] = int((valid_delta > 1e-12).sum())
            summary_row[f"degraded_images_{metric}"] = int((valid_delta < -1e-12).sum())
            summary_row[f"unchanged_images_{metric}"] = int((np.abs(valid_delta) <= 1e-12).sum())
        topology_summary.append(summary_row)

    topology_summary_path = os.path.splitext(output_path)[0] + "_topology_summary.csv"
    with open(topology_summary_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(topology_summary[0].keys()) if topology_summary else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topology_summary)

    for mode, summary in summaries.items():
        print(
            f"{mode:26s} IoU={summary['iou']:.5f} F1={summary['f1']:.5f} "
            f"P={summary['precision']:.5f} R={summary['recall']:.5f} "
            f"AUPRC~={summary['auprc_binned_4096']:.5f} "
            f"bestIoU={summary['best_iou']:.5f}@{summary['best_iou_threshold']:.3f}",
            flush=True,
        )
    print(f"Saved paired validation comparison: {output_path}", flush=True)
    print(f"Saved per-image topology comparison: {per_image_path}", flush=True)
    print(f"Saved topology threshold summary: {topology_summary_path}", flush=True)
    print(f"Saved probability maps and threshold-{args.threshold:.2f} previews under: {output_dir}", flush=True)
    print(f"Images evaluated: {image_names_seen}", flush=True)
    for threshold in thresholds:
        changes = pixel_change[threshold]
        print(
            f"thr={threshold:.2f}: changed_pixels={changes['changed']} "
            f"bias_on_only={changes['on_only']} bias_off_only={changes['off_only']}",
            flush=True,
        )
    for summary in topology_summary:
        print(
            f"topology thr={summary['threshold']:.2f}: "
            f"clDice delta={summary['mean_delta_cldice']:+.5f} "
            f"(improved/degraded={summary['improved_images_cldice']}/"
            f"{summary['degraded_images_cldice']} images), "
            f"connected-pair recall delta={summary['mean_delta_connected_pair_recall']:+.5f} "
            f"(improved/degraded={summary['improved_images_connected_pair_recall']}/"
            f"{summary['degraded_images_connected_pair_recall']} images)",
            flush=True,
        )


if __name__ == "__main__":
    main()
