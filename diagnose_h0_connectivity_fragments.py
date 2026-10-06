"""Paired H0 connectivity ablations and GT-based short-component audit."""

import argparse
import csv
import json
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from compare_connectivity_topology_metrics import (
    load_metric_model,
    parse_args as metric_parse_args,
)
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_h0_interactions import prediction_metrics
from losses.road_losses import build_connectivity_target, build_stage_skeleton_target


MODES = ("baseline", "stage3_gate_off", "stage3_gate_shift", "stage3_gate_gt", "global_no_c3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cfg", default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--max_images", type=int, default=200, help="0 evaluates the full split")
    parser.add_argument("--apls_images", type=int, default=30,
                        help="paired first N images; 0 disables APLS")
    parser.add_argument("--short_area_threshold", type=int, default=20)
    parser.add_argument("--apls_max_nodes", type=int, default=64)
    parser.add_argument("--apls_snap_radius", type=float, default=5.0)
    parser.add_argument("--gt_tolerance_px", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be in (0, 1)")
    if args.max_images < 0 or args.apls_images < 0 or args.gt_tolerance_px < 0:
        parser.error("image counts and GT tolerance must be nonnegative")
    if args.short_area_threshold < 1:
        parser.error("--short_area_threshold must be positive")
    return args


def load_model(args, device):
    import sys
    original = sys.argv
    sys.argv = [original[0], "--root_path", args.root_path, "--model_path", args.model_path,
                "--cfg", args.cfg, "--img_size", str(args.img_size),
                "--source_patch_size", str(args.source_patch_size), "--num_workers", "0"]
    try:
        loader_args = metric_parse_args()
    finally:
        sys.argv = original
    model = load_metric_model(args.model_path, loader_args, device).eval()
    net = model.swin_unet
    if not net.enable_global_topology or net.global_topology is None:
        raise RuntimeError("Checkpoint does not have enabled H0 global topology")
    if "3" not in net.decoder_structure_blocks:
        raise RuntimeError("Checkpoint does not have a Stage 3 structure block")
    return model


class ConnectivityProbe:
    def __init__(self, model):
        self.net = model.swin_unet
        self.mode = "baseline"
        self.skeleton_gt = None
        self.gate_calls = 0
        self.global_calls = 0
        self.original_local_features = self.net._latest_local_topology_features
        self.gate_hook = self.net.decoder_structure_blocks["3"].structure_gate.register_forward_pre_hook(
            self._modify_gate
        )
        self.net._latest_local_topology_features = types.MethodType(self._global_input, self.net)

    def _modify_gate(self, module, inputs):
        if self.mode not in ("stage3_gate_off", "stage3_gate_shift", "stage3_gate_gt"):
            return None
        gate_input = inputs[0]
        self.gate_calls += 1
        conn = gate_input[:, -1:]
        if self.mode == "stage3_gate_off":
            replacement = torch.zeros_like(conn)
        elif self.mode == "stage3_gate_shift":
            replacement = torch.roll(conn, shifts=(conn.shape[-2] // 2, conn.shape[-1] // 2), dims=(-2, -1))
        else:
            target = build_stage_skeleton_target(self.skeleton_gt, conn.shape[-2:])
            connectivity = build_connectivity_target(target).to(device=conn.device, dtype=conn.dtype)
            replacement = connectivity.topk(k=2, dim=1).values.mean(dim=1, keepdim=True)
        return (torch.cat((gate_input[:, :-1], replacement), dim=1),)

    def _global_input(self, net, outputs):
        if self.mode == "global_no_c3":
            self.global_calls += 1
            return None
        return self.original_local_features(outputs)

    def run(self, model, image, skeleton_gt, mode):
        self.mode = mode
        self.skeleton_gt = skeleton_gt
        self.gate_calls = 0
        self.global_calls = 0
        logits = model(image)[0]
        if mode.startswith("stage3_gate_") and self.gate_calls != 1:
            raise RuntimeError(f"Stage 3 gate hook ran {self.gate_calls} times")
        if mode == "global_no_c3" and self.global_calls != 1:
            raise RuntimeError(f"Global C3 hook ran {self.global_calls} times")
        return logits

    def close(self):
        self.gate_hook.remove()
        del self.net._latest_local_topology_features


def audit_short_components(case, pred, gt, gt_skeleton, area_limit, tolerance):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    _, gt_labels = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    if tolerance:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
        near_gt = cv2.dilate(gt.astype(np.uint8), kernel) > 0
    else:
        near_gt = gt
    pieces = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        pixels = labels == component
        supported = gt_labels[pixels]
        supported = supported[supported > 0]
        if supported.size:
            gt_component, dominant_count = np.unique(supported, return_counts=True)
            dominant = int(gt_component[dominant_count.argmax()])
        else:
            dominant = 0
        pieces.append({
            "case": case, "component": component, "area": area,
            "gt_overlap_fraction": float(np.logical_and(pixels, gt).sum() / area),
            "near_gt_fraction": float(np.logical_and(pixels, near_gt).sum() / area),
            "gt_skeleton_pixels": int(np.logical_and(pixels, gt_skeleton).sum()),
            "dominant_gt_component": dominant,
        })
    gt_to_pred = defaultdict(set)
    for piece in pieces:
        if piece["dominant_gt_component"] and piece["gt_overlap_fraction"] >= 0.5:
            gt_to_pred[piece["dominant_gt_component"]].add(piece["component"])
    short_rows = []
    for piece in pieces:
        if piece["area"] >= area_limit:
            continue
        gt_id = piece["dominant_gt_component"]
        if piece["near_gt_fraction"] < 0.1 and piece["gt_skeleton_pixels"] == 0:
            category = "likely_background_fp"
        elif piece["gt_overlap_fraction"] >= 0.5 and gt_id:
            category = ("supported_fragment" if len(gt_to_pred[gt_id]) > 1
                        else "supported_independent")
        else:
            category = "uncertain"
        short_rows.append({**piece, "category": category})
    return short_rows


def summarize(mode, rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "images": len(rows), "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "precision": precision, "recall": recall,
    }
    for key in ("cldice", "break_rate", "gap_components", "pred_components",
                "fragment_density_per_1000_px", "largest_component_ratio", "apls_approx"):
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        if values:
            result[key] = float(np.mean(values))
    if mode != "baseline":
        for key in ("added_gt_road_pixels", "added_background_pixels",
                    "removed_gt_road_pixels", "removed_background_pixels"):
            result[key] = sum(row.get(key, 0) for row in rows)
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    dataset = RoadSkeletonDataset(root_dir=args.root_path, split=args.split,
                                  image_size=args.img_size, source_patch_size=args.source_patch_size)
    if args.max_images and args.max_images < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, args.max_images, dtype=int).tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    probe = ConnectivityProbe(model)
    all_rows = []
    component_rows = []
    try:
        with torch.inference_mode():
            for index, batch in enumerate(tqdm(loader, desc="H0 connectivity and fragments")):
                if args.max_images and index >= args.max_images:
                    break
                image = batch["image"].to(device)
                skeleton = batch["skeleton"].to(device)
                gt = batch["mask"][0, 0].numpy() > 0.5
                case = batch["case_name"][0]
                paired = {}
                for mode in MODES:
                    logits = probe.run(model, image, skeleton, mode)
                    args.with_apls = index < args.apls_images
                    metrics, pred, gt_skeleton = prediction_metrics(logits, gt, args)
                    row = {"case": case, "mode": mode, **metrics}
                    row.pop("pred_skel", None)
                    row.pop("gt_skel", None)
                    if mode == "baseline":
                        paired["pred"] = pred
                        component_rows.extend(audit_short_components(
                            case, pred, gt, gt_skeleton,
                            args.short_area_threshold, args.gt_tolerance_px,
                        ))
                    else:
                        original = paired["pred"]
                        row.update({
                            "added_gt_road_pixels": int(np.logical_and(pred & ~original, gt).sum()),
                            "added_background_pixels": int(np.logical_and(pred & ~original, ~gt).sum()),
                            "removed_gt_road_pixels": int(np.logical_and(~pred & original, gt).sum()),
                            "removed_background_pixels": int(np.logical_and(~pred & original, ~gt).sum()),
                        })
                    all_rows.append(row)
                if (index + 1) % 25 == 0:
                    print(f"[PROGRESS] {index + 1} images", flush=True)
    finally:
        probe.close()
    if not all_rows:
        raise RuntimeError("No images evaluated")
    grouped = {mode: [row for row in all_rows if row["mode"] == mode] for mode in MODES}
    summary = {mode: summarize(mode, rows) for mode, rows in grouped.items()}
    counts = {category: sum(row["category"] == category for row in component_rows)
              for category in ("likely_background_fp", "supported_fragment",
                               "supported_independent", "uncertain")}
    summary["short_components"] = {"total": len(component_rows), **counts}
    report = {
        "checkpoint": args.model_path, "split": args.split,
        "threshold": args.threshold, "img_size": args.img_size,
        "short_area_threshold": args.short_area_threshold,
        "gt_tolerance_px": args.gt_tolerance_px,
        "apls_images": min(args.apls_images, len(grouped["baseline"])),
        "modes": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    fields = sorted({key for row in all_rows for key in row})
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    fields = ("case", "component", "area", "gt_overlap_fraction", "near_gt_fraction",
              "gt_skeleton_pixels", "dominant_gt_component", "category")
    with (output_dir / "short_components.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(component_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
