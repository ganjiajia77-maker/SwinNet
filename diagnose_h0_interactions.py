"""Paired H0 path and global-topology ablations on one unchanged checkpoint."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from compare_connectivity_topology_metrics import (
    apls_score,
    component_stats,
    load_metric_model,
    parse_args as metric_parse_args,
    topology_scores,
)
from datasets.dataset_road_skeleton import RoadSkeletonDataset


MODES = (
    "baseline",
    "global_off",
    "stage2_injection_off",
    "stage2_exchange_off",
    "stage2_both_off",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--max_images", type=int, default=250,
                        help="0 evaluates the complete split")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--short_area_threshold", type=int, default=20)
    parser.add_argument("--apls_max_nodes", type=int, default=64)
    parser.add_argument("--apls_snap_radius", type=float, default=5.0)
    parser.add_argument("--with_apls", action="store_true",
                        help="Slow sampled-graph APLS; use on a smaller subset first")
    parser.add_argument("--anchor_radius", type=int, default=3)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.max_images < 0 or args.anchor_radius < 0:
        parser.error("--max_images and --anchor_radius must be nonnegative")
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes contains duplicates")
    return args


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)


def relative_norm(new, old):
    return scalar(torch.linalg.vector_norm((new - old).float()) /
                  (torch.linalg.vector_norm(old.float()) + 1e-6))


class Probes:
    def __init__(self, model):
        net = model.swin_unet
        self.direct = net.highres_structure_fusion["2"]
        self.exchange = net.decoder_structure_blocks["2"].highres_skeleton_fusion
        self.global_module = net.global_topology
        if self.exchange is None or self.global_module is None:
            raise RuntimeError("Checkpoint model lacks the H0 exchange or global topology module")
        self.stage_blocks = [net.decoder_structure_blocks[key] for key in ("2", "3")]
        self.observation = {}
        self.mode = "baseline"
        self.handles = [
            self.direct.register_forward_hook(self.direct_hook),
            self.exchange.register_forward_hook(self.exchange_hook),
        ]
        self.original_anchor_method = self.global_module._extract_fps_anchors
        self.global_module._extract_fps_anchors = MethodType(self.anchor_method, self.global_module)
        for block in self.stage_blocks:
            block.capture_diagnostics = True
        self.global_module.capture_diagnostics = True

    def direct_hook(self, module, inputs, output):
        self.observation["stage2_injection_relative_norm"] = relative_norm(output, inputs[0])
        if self.mode in ("stage2_injection_off", "stage2_both_off"):
            return inputs[0]
        return None

    def exchange_hook(self, module, inputs, output):
        self.observation["stage2_exchange_G_relative_norm"] = relative_norm(output[0], inputs[1])
        self.observation["stage2_exchange_H_relative_norm"] = relative_norm(output[1], inputs[0])
        if self.mode in ("stage2_exchange_off", "stage2_both_off"):
            h0 = inputs[0]
            if h0.shape[-2:] != inputs[1].shape[-2:]:
                h0 = F.interpolate(h0, size=inputs[1].shape[-2:], mode="bilinear", align_corners=False)
            return inputs[1], h0
        return None

    def anchor_method(self, module, anchor_score):
        result = self.original_anchor_method(anchor_score)
        self.observation["anchor_coords"] = result[0][0, result[1][0]].detach().cpu().numpy()
        return result

    def run(self, model, image, mode):
        self.mode = mode
        self.observation = {}
        self.global_module.enable_global_topology = mode != "global_off"
        self.global_module.last_diagnostics = None
        for block in self.stage_blocks:
            block.last_diagnostics = None
        output = model(image)
        logits = output[0] if isinstance(output, tuple) else output
        obs = {key: value for key, value in self.observation.items() if key != "anchor_coords"}
        for stage, block in zip((2, 3), self.stage_blocks):
            if block.last_diagnostics:
                for key in ("gamma1", "gate_mean", "gate_residual_relative_norm"):
                    if key in block.last_diagnostics:
                        obs[f"stage{stage}_{key}"] = scalar(block.last_diagnostics[key])
        if self.global_module.last_diagnostics:
            for key in ("anchor_count", "candidate_count", "alpha_global",
                        "surface_gate_mean", "global_residual_relative_norm"):
                obs[key] = scalar(self.global_module.last_diagnostics[key])
        return logits, obs, self.observation.get("anchor_coords")

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.global_module._extract_fps_anchors = self.original_anchor_method
        self.global_module.enable_global_topology = True
        self.global_module.capture_diagnostics = False
        for block in self.stage_blocks:
            block.capture_diagnostics = False


def anchor_coverage(coords, gt, gt_skel, radius):
    if coords is None or len(coords) == 0:
        return {}
    h, w = gt.shape
    points = np.zeros((h, w), dtype=np.uint8)
    coords = np.asarray(coords)
    points[coords[:, 0].clip(0, h - 1), coords[:, 1].clip(0, w - 1)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    near = cv2.dilate(points, kernel) > 0
    neighbors = cv2.filter2D(gt_skel.astype(np.uint8), cv2.CV_16S,
                             np.ones((3, 3), dtype=np.uint8)) - gt_skel.astype(np.int16)
    endpoints = gt_skel & (neighbors == 1)
    junctions = gt_skel & (neighbors >= 3)
    return {
        "anchor_near_road_fraction": float((cv2.dilate(gt.astype(np.uint8), kernel) > 0)[points > 0].mean()),
        "anchor_near_skeleton_fraction": float((cv2.dilate(gt_skel.astype(np.uint8), kernel) > 0)[points > 0].mean()),
        "gt_skeleton_near_anchor_fraction": float(near[gt_skel].mean()) if gt_skel.any() else float("nan"),
        "gt_endpoint_near_anchor_fraction": float(near[endpoints].mean()) if endpoints.any() else float("nan"),
        "gt_junction_near_anchor_fraction": float(near[junctions].mean()) if junctions.any() else float("nan"),
    }


def prediction_metrics(logits, gt, args):
    if logits.shape[-2:] != gt.shape:
        logits = F.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)
    pred = (torch.sigmoid(logits)[0, 0] >= args.threshold).cpu().numpy()
    topo = topology_scores(pred, gt)
    pred_skel = topo.pop("pred_skel")
    gt_skel = topo.pop("gt_skel")
    stats = component_stats(pred, args.short_area_threshold)
    topo.update({
        "pred_components": stats["components"],
        "fragment_density_per_1000_px": 1000.0 * stats["components"] / (float(pred.sum()) + 1e-8),
        "largest_component_ratio": stats["largest_ratio"],
    })
    if args.with_apls:
        topo["apls_approx"] = apls_score(gt_skel, pred_skel, args.apls_max_nodes, args.apls_snap_radius)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    topo.update(tp=tp, fp=fp, fn=fn)
    return topo, pred, gt_skel


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reuse the audited checkpoint loader and direct256 preprocessing from this code revision.
    import sys
    argv = sys.argv
    try:
        sys.argv = [argv[0], "--root_path", args.root_path, "--model_path", args.model_path,
                    "--cfg", args.cfg, "--img_size", str(args.img_size),
                    "--source_patch_size", str(args.source_patch_size)]
        model_args = metric_parse_args()
    finally:
        sys.argv = argv
    model = load_metric_model(args.model_path, model_args, device).eval()
    if not model.swin_unet.enable_global_topology:
        raise RuntimeError("Checkpoint has no enabled global topology; on/off test would be meaningless")
    dataset = RoadSkeletonDataset(root_dir=args.root_path, split=args.split,
                                  image_size=args.img_size, source_patch_size=args.source_patch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    probes = Probes(model)
    rows = []
    try:
        with torch.inference_mode():
            for index, batch in enumerate(tqdm(loader, desc="H0 paired ablations")):
                if args.max_images and index >= args.max_images:
                    break
                image = batch["image"].to(device, non_blocking=True)
                gt = batch["mask"][0, 0].cpu().numpy() > 0.5
                case = batch["case_name"][0]
                paired = {}
                for mode in args.modes:
                    logits, obs, coords = probes.run(model, image, mode)
                    metrics, pred, gt_skel = prediction_metrics(logits, gt, args)
                    row = {"case": case, "mode": mode, **metrics, **obs}
                    if mode == "baseline":
                        row.update(anchor_coverage(coords, gt, gt_skel, args.anchor_radius))
                    rows.append(row)
                    if mode in ("baseline", "global_off"):
                        probability = torch.sigmoid(logits)
                        if probability.shape[-2:] != gt.shape:
                            probability = F.interpolate(probability, size=gt.shape,
                                                        mode="bilinear", align_corners=False)
                        paired[mode] = (row, pred, probability[0, 0].cpu().numpy())
                if "baseline" in paired and "global_off" in paired:
                    base_row, base_pred, base_prob = paired["baseline"]
                    off_row, off_pred, off_prob = paired["global_off"]
                    delta = np.abs(base_prob - off_prob)
                    base_row["global_on_off_probability_abs_mean"] = float(delta.mean())
                    base_row["global_on_off_changed_pixel_fraction"] = float((base_pred != off_pred).mean())
                    misses = gt_skel & ~off_pred
                    base_row["global_recovered_skeleton_pixels"] = float((misses & base_pred).sum())
                    base_row["global_lost_skeleton_pixels"] = float((gt_skel & off_pred & ~base_pred).sum())
                if (index + 1) % 25 == 0:
                    print(f"[PROGRESS] {index + 1} images", flush=True)
    finally:
        probes.close()
    if not rows:
        raise RuntimeError("No images evaluated")
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in ("case", "mode"), key))
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    summary = {}
    for mode, values in grouped.items():
        tp, fp, fn = (sum(row[key] for row in values) for key in ("tp", "fp", "fn"))
        result = {
            "images": len(values),
            "iou": tp / (tp + fp + fn + 1e-8),
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
        }
        result["f1"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"] + 1e-8)
        for key in fields:
            if key in {"case", "mode", "tp", "fp", "fn"}:
                continue
            numbers = [row[key] for row in values if key in row and np.isfinite(row[key])]
            if numbers:
                result[key] = float(np.mean(numbers))
        summary[mode] = result
    report = {"checkpoint": args.model_path, "split": args.split, "threshold": args.threshold,
              "img_size": args.img_size, "source_patch_size": args.source_patch_size,
              "anchor_radius": args.anchor_radius, "apls_is_sampled_approximation": args.with_apls,
              "modes": summary}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved {output_dir / 'per_image.csv'} and {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
