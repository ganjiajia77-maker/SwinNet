"""Soft skeleton graph propagation ON/OFF ablation + G / delta-logit diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import binary_metrics_from_logits
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet,
    configure_graph_diagnostics,
    get_graph_propagation_state,
    load_topology_checkpoint_state,
    print_topology_coefficients,
    set_soft_graph_eval_mode,
)


@dataclass
class RegionStats:
    count: int = 0
    mean: float = 0.0
    gt_ratio: float = 0.0


@dataclass
class AblationResult:
    name: str
    use_soft_graph: bool
    lambda_scale: float
    lambda_eff: float
    iou: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    lambda_eff_mean: float = 0.0
    message_over_feature_norm_mean: float = 0.0
    residual_over_feature_norm_mean: float = 0.0
    surface_logit_delta_relative_mean: float = 0.0
    G_TP_mean: float = 0.0
    G_FN_mean: float = 0.0
    G_FP_mean: float = 0.0
    G_TN_mean: float = 0.0
    G_gt_0_1_TP: float = 0.0
    G_gt_0_1_FN: float = 0.0
    G_gt_0_1_FP: float = 0.0
    G_gt_0_1_TN: float = 0.0
    delta_logit_TP: float = 0.0
    delta_logit_FN: float = 0.0
    delta_logit_FP: float = 0.0
    delta_logit_TN: float = 0.0
    extra: dict = field(default_factory=dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./model_out/train_stage23_graph_prop_nw0_2/best.pth",
    )
    parser.add_argument("--root_path", default="./data1")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--cfg",
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    parser.add_argument(
        "--structure_profile",
        default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    )
    parser.add_argument("--enable_graph_prop", action="store_true")
    parser.add_argument(
        "--diag_samples",
        type=int,
        default=0,
        help="0 means all samples; otherwise limit for quick runs",
    )
    parser.add_argument(
        "--output_dir",
        default="./topology_diagnostics/soft_graph_ablation",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
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
    return parser.parse_args()


def load_model(args, device):
    enable_graph_prop = args.enable_graph_prop
    structure_profile = args.structure_profile
    stage_topology_stages = "none"
    final_topology_eta_init = 0.0
    final_gap_rho_init = 0.0
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        structure_profile = checkpoint.get("structure_profile", structure_profile)
        if isinstance(checkpoint.get("args"), dict):
            ckpt_args = checkpoint["args"]
            structure_profile = ckpt_args.get(
                "structure_profile",
                structure_profile,
            )
            stage_topology_stages = ckpt_args.get(
                "stage_topology_stages",
                stage_topology_stages,
            )
            final_topology_eta_init = ckpt_args.get(
                "final_topology_eta_init",
                final_topology_eta_init,
            )
            final_gap_rho_init = ckpt_args.get(
                "final_gap_rho_init",
                final_gap_rho_init,
            )
            if not enable_graph_prop:
                enable_graph_prop = bool(
                    ckpt_args.get("enable_graph_prop", False)
                )

    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        structure_profile=structure_profile,
        enable_final_graph_prop=enable_graph_prop,
        stage_topology_stages=stage_topology_stages,
        final_topology_eta_init=final_topology_eta_init,
        final_gap_rho_init=final_gap_rho_init,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model = model.to(device)
    model.eval()
    return model, structure_profile, enable_graph_prop


def confusion_regions(gt, pred):
    gt_b = gt > 0.5
    pred_b = pred > 0.5
    tp = gt_b & pred_b
    fn = gt_b & ~pred_b
    fp = ~gt_b & pred_b
    tn = ~gt_b & ~pred_b
    return tp, fn, fp, tn


def masked_mean(tensor, mask):
    mask = mask.to(dtype=tensor.dtype, device=tensor.device)
    count = int(mask.sum().item())
    if count <= 0:
        return 0.0, 0
    value = float((tensor * mask).sum().item() / count)
    return value, count


def masked_ratio_gt(tensor, mask, threshold=0.1):
    mask = mask.to(dtype=torch.bool, device=tensor.device)
    count = int(mask.sum().item())
    if count <= 0:
        return 0.0
    active = (tensor > threshold) & mask
    return float(active.sum().item() / count)


def aggregate_region_stats(accumulator, gt, pred, G, delta_logits):
    tp, fn, fp, tn = confusion_regions(gt, pred)
    for key, region in (
        ("TP", tp),
        ("FN", fn),
        ("FP", fp),
        ("TN", tn),
    ):
        g_mean, count = masked_mean(G, region)
        accumulator.setdefault(f"G_{key}_mean", 0.0)
        accumulator.setdefault(f"G_{key}_count", 0)
        accumulator.setdefault(f"G_gt_0_1_{key}", 0.0)
        accumulator.setdefault(f"delta_logit_{key}", 0.0)
        accumulator.setdefault(f"delta_{key}_count", 0)
        accumulator[f"G_{key}_mean"] += g_mean * count
        accumulator[f"G_{key}_count"] += count
        accumulator[f"G_gt_0_1_{key}"] += masked_ratio_gt(G, region) * count
        d_mean, d_count = masked_mean(delta_logits, region)
        accumulator[f"delta_logit_{key}"] += d_mean * d_count
        accumulator[f"delta_{key}_count"] += d_count


def finalize_region_stats(accumulator):
    out = {}
    for key in ("TP", "FN", "FP", "TN"):
        g_count = accumulator.get(f"G_{key}_count", 0)
        d_count = accumulator.get(f"delta_{key}_count", 0)
        out[f"G_{key}_mean"] = (
            accumulator.get(f"G_{key}_mean", 0.0) / g_count if g_count else 0.0
        )
        out[f"G_gt_0_1_{key}"] = (
            accumulator.get(f"G_gt_0_1_{key}", 0.0) / g_count if g_count else 0.0
        )
        out[f"delta_logit_{key}"] = (
            accumulator.get(f"delta_logit_{key}", 0.0) / d_count if d_count else 0.0
        )
    return out


def run_ablation(
    model,
    loader,
    device,
    threshold,
    name,
    use_soft_graph,
    lambda_scale,
    collect_diagnostics,
):
    mode = set_soft_graph_eval_mode(
        model,
        use_soft_graph=use_soft_graph,
        lambda_scale=lambda_scale,
    )
    configure_graph_diagnostics(model, enabled=collect_diagnostics and use_soft_graph)
    head = model.swin_unet.guided_head
    head.last_graph_diagnostics = None

    tp = fp = fn = 0
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    diag_acc = {
        "lambda_eff": 0.0,
        "message_over_feature_norm": 0.0,
        "residual_over_feature_norm": 0.0,
        "surface_logit_delta_relative": 0.0,
        "diag_batches": 0,
    }
    region_acc = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc=name, leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            surface_logits = outputs[0]
            pred = (torch.sigmoid(surface_logits) >= threshold).float()
            gt = masks
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            gt_bin = (gt > 0.5).float()

            tp += int((pred * gt_bin).sum().item())
            fp += int((pred * (1.0 - gt_bin)).sum().item())
            fn += int(((1.0 - pred) * gt_bin).sum().item())

            metrics = binary_metrics_from_logits(
                surface_logits,
                masks,
                threshold=threshold,
            )
            iou_list.append(metrics["iou"])
            f1_list.append(metrics["f1"])
            precision_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])

            if not collect_diagnostics or not use_soft_graph:
                continue

            diag = head.last_graph_diagnostics
            if diag is None:
                continue

            diag_acc["diag_batches"] += 1
            for key in (
                "lambda_eff",
                "message_over_feature_norm",
                "residual_over_feature_norm",
                "surface_logit_delta_relative",
            ):
                if key in diag:
                    diag_acc[key] += float(diag[key])

            export = diag.get("export")
            if export is None:
                continue

            G = export["G"]
            delta = diag.get("graph_logit_delta")
            if delta is None:
                delta = diag["surface_post_graph_logits"] - diag["surface_pre_logits"]
            aggregate_region_stats(region_acc, gt_bin, pred, G, delta)

    batches = max(len(iou_list), 1)
    diag_batches = max(diag_acc["diag_batches"], 1)
    region_stats = finalize_region_stats(region_acc)
    eps = 1e-8
    global_precision = tp / (tp + fp + eps)
    global_recall = tp / (tp + fn + eps)
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall + eps)
    )
    global_iou = tp / (tp + fp + fn + eps)

    return AblationResult(
        name=name,
        use_soft_graph=bool(mode["use_soft_graph"]),
        lambda_scale=float(mode["lambda_scale"]),
        lambda_eff=float(mode["lambda_eff"]),
        iou=float(global_iou),
        f1=float(global_f1),
        precision=float(global_precision),
        recall=float(global_recall),
        lambda_eff_mean=diag_acc["lambda_eff"] / diag_batches,
        message_over_feature_norm_mean=diag_acc["message_over_feature_norm"]
        / diag_batches,
        residual_over_feature_norm_mean=diag_acc["residual_over_feature_norm"]
        / diag_batches,
        surface_logit_delta_relative_mean=diag_acc["surface_logit_delta_relative"]
        / diag_batches,
        G_TP_mean=region_stats.get("G_TP_mean", 0.0),
        G_FN_mean=region_stats.get("G_FN_mean", 0.0),
        G_FP_mean=region_stats.get("G_FP_mean", 0.0),
        G_TN_mean=region_stats.get("G_TN_mean", 0.0),
        G_gt_0_1_TP=region_stats.get("G_gt_0_1_TP", 0.0),
        G_gt_0_1_FN=region_stats.get("G_gt_0_1_FN", 0.0),
        G_gt_0_1_FP=region_stats.get("G_gt_0_1_FP", 0.0),
        G_gt_0_1_TN=region_stats.get("G_gt_0_1_TN", 0.0),
        delta_logit_TP=region_stats.get("delta_logit_TP", 0.0),
        delta_logit_FN=region_stats.get("delta_logit_FN", 0.0),
        delta_logit_FP=region_stats.get("delta_logit_FP", 0.0),
        delta_logit_TN=region_stats.get("delta_logit_TN", 0.0),
        extra={"evaluated_batches": batches},
    )


def format_result_line(result: AblationResult) -> str:
    return (
        f"{result.name:<18} "
        f"IoU={result.iou:.4f} F1={result.f1:.4f} "
        f"P={result.precision:.4f} R={result.recall:.4f} "
        f"lambda_eff={result.lambda_eff_mean:.4f} "
        f"msg/F={result.message_over_feature_norm_mean:.4f} "
        f"res/F={result.residual_over_feature_norm_mean:.4f} "
        f"dlogit_rel={result.surface_logit_delta_relative_mean:.4f}"
    )


def write_report(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, structure_profile, enable_graph_prop = load_model(args, device)
    print(f"Loaded: {args.model_path}")
    print(f"structure_profile={structure_profile}, enable_graph_prop={enable_graph_prop}")
    print_topology_coefficients(model)
    print("Graph state:", get_graph_propagation_state(model))

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    if args.diag_samples > 0:
        dataset = torch.utils.data.Subset(
            dataset,
            list(range(min(args.diag_samples, len(dataset)))),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"{args.split} size: {len(dataset)}, threshold={args.threshold}")

    ablations = [
        ("graph_on", True, 1.0, True),
        ("lambda_off", True, 0.0, True),
        ("graph_off", False, 1.0, True),
    ]

    results = []
    for name, use_graph, lambda_scale, collect_diag in ablations:
        result = run_ablation(
            model,
            loader,
            device,
            args.threshold,
            name=name,
            use_soft_graph=use_graph,
            lambda_scale=lambda_scale,
            collect_diagnostics=collect_diag,
        )
        results.append(result)
        print(format_result_line(result))

    on = results[0]
    lambda_off = results[1]
    graph_off = results[2]
    print("\n" + "=" * 80)
    print("ON/OFF delta (same checkpoint, same split, same threshold)")
    print("=" * 80)
    print(
        f"graph_on vs lambda_off: "
        f"dIoU={on.iou - lambda_off.iou:+.4f}, dF1={on.f1 - lambda_off.f1:+.4f}"
    )
    print(
        f"graph_on vs graph_off:  "
        f"dIoU={on.iou - graph_off.iou:+.4f}, dF1={on.f1 - graph_off.f1:+.4f}"
    )
    print(
        f"graph_off baseline check (target test@0.25: IoU=0.5372, F1=0.6989): "
        f"IoU={graph_off.iou:.4f}, F1={graph_off.f1:.4f}"
    )
    if abs(lambda_off.iou - graph_off.iou) < 0.002:
        print("lambda_off ~= graph_off -> OFF modes are consistent.")
    if abs(on.iou - lambda_off.iou) < 0.002 and abs(on.f1 - lambda_off.f1) < 0.002:
        print(
            "WARNING: graph ON ~= OFF. Soft graph likely has negligible effect "
            "at this threshold, or the backbone is not the expected 0.5372 baseline."
        )

    print("\nG region diagnostics (graph_on):")
    print(
        f"  G mean  TP={on.G_TP_mean:.4f} FN={on.G_FN_mean:.4f} "
        f"FP={on.G_FP_mean:.4f} TN={on.G_TN_mean:.4f}"
    )
    print(
        f"  G>0.1  TP={on.G_gt_0_1_TP:.4f} FN={on.G_gt_0_1_FN:.4f} "
        f"FP={on.G_gt_0_1_FP:.4f} TN={on.G_gt_0_1_TN:.4f}"
    )
    print("Delta logit by region (graph_on, post_graph - pre_graph):")
    print(
        f"  TP={on.delta_logit_TP:+.4f} FN={on.delta_logit_FN:+.4f} "
        f"FP={on.delta_logit_FP:+.4f} TN={on.delta_logit_TN:+.4f}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if not os.path.basename(output_dir).endswith(timestamp[:8]):
        output_dir = os.path.join(output_dir, timestamp)
    report = {
        "model_path": args.model_path,
        "split": args.split,
        "threshold": args.threshold,
        "structure_profile": structure_profile,
        "enable_graph_prop": enable_graph_prop,
        "graph_state": get_graph_propagation_state(model),
        "results": [asdict(item) for item in results],
        "comparison": {
            "graph_on_vs_lambda_off": {
                "delta_iou": on.iou - lambda_off.iou,
                "delta_f1": on.f1 - lambda_off.f1,
            },
            "graph_on_vs_graph_off": {
                "delta_iou": on.iou - graph_off.iou,
                "delta_f1": on.f1 - graph_off.f1,
            },
        },
    }
    json_path = os.path.join(output_dir, "soft_graph_ablation_report.json")
    txt_path = os.path.join(output_dir, "soft_graph_ablation_report.txt")
    write_report(json_path, report)
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("Soft skeleton graph propagation ablation\n")
        handle.write(f"model={args.model_path}\n")
        handle.write(f"split={args.split}, threshold={args.threshold}\n\n")
        for item in results:
            handle.write(format_result_line(item) + "\n")
        handle.write("\nG / delta diagnostics (graph_on):\n")
        handle.write(
            f"G mean TP/FN/FP/TN = "
            f"{on.G_TP_mean:.4f}/{on.G_FN_mean:.4f}/"
            f"{on.G_FP_mean:.4f}/{on.G_TN_mean:.4f}\n"
        )
        handle.write(
            f"G>0.1 TP/FN/FP/TN = "
            f"{on.G_gt_0_1_TP:.4f}/{on.G_gt_0_1_FN:.4f}/"
            f"{on.G_gt_0_1_FP:.4f}/{on.G_gt_0_1_TN:.4f}\n"
        )
        handle.write(
            f"delta_logit TP/FN/FP/TN = "
            f"{on.delta_logit_TP:+.4f}/{on.delta_logit_FN:+.4f}/"
            f"{on.delta_logit_FP:+.4f}/{on.delta_logit_TN:+.4f}\n"
        )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
