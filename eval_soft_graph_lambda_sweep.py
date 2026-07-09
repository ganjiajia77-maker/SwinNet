"""Eval-only Soft Graph lambda sweep on a frozen 0626 (or any) checkpoint.

Inserts untrained graph with identity dir_convs so topology weights W_d/G/H
drive the message without trained 1x1 convs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.cldice_loss import SoftCLDiceLoss
from losses.road_losses import binary_metrics_from_logits
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet,
    apply_structure_profile_runtime,
    configure_graph_diagnostics,
    load_topology_checkpoint_state,
    print_topology_coefficients,
    set_soft_graph_eval_mode,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=(
            "./model_out/train_stage23_structure_final_boundary_nw0_20260626/best.pth"
        ),
    )
    parser.add_argument("--root_path", default="./data1")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 = full split",
    )
    parser.add_argument(
        "--lambda_values",
        type=float,
        nargs="+",
        default=[0.0, 0.03, 0.05, 0.10],
    )
    parser.add_argument(
        "--cfg",
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    parser.add_argument(
        "--structure_profile",
        default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    )
    parser.add_argument(
        "--output_dir",
        default="./topology_diagnostics/soft_graph_lambda_sweep",
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


@dataclass
class SweepResult:
    name: str
    use_soft_graph: bool
    lambda_eff: float
    iou: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    skeleton_cldice: float = 0.0
    num_components_mean: float = 0.0
    gt_skeleton_coverage_mean: float = 0.0
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


def load_frozen_backbone(args, device):
    structure_profile = args.structure_profile
    stage_topology_stages = "none"
    final_topology_eta_init = 0.0
    final_gap_rho_init = 0.0

    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        ckpt_args = checkpoint["args"]
        structure_profile = ckpt_args.get("structure_profile", structure_profile)
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

    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        structure_profile=structure_profile,
        enable_final_graph_prop=True,
        stage_topology_stages=stage_topology_stages,
        final_topology_eta_init=final_topology_eta_init,
        final_gap_rho_init=final_gap_rho_init,
    )
    state_dict = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} mismatched/missing keys when loading backbone")
        for key in skipped[:8]:
            print(f"  skip: {key}")
    apply_structure_profile_runtime(model)
    model = model.to(device)
    model.eval()
    return model, checkpoint, structure_profile


def count_components(mask_uint8: np.ndarray) -> int:
    binary = (mask_uint8 > 127).astype(np.uint8)
    if binary.sum() == 0:
        return 0
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    return int(labels.max())


def skeleton_coverage(pred_bin: np.ndarray, skeleton_bin: np.ndarray) -> float:
    skel = skeleton_bin > 0.5
    if skel.sum() == 0:
        return 0.0
    covered = skel & (pred_bin > 0.5)
    return float(covered.sum() / skel.sum())


def confusion_regions(gt, pred):
    gt_b = gt > 0.5
    pred_b = pred > 0.5
    return (
        gt_b & pred_b,
        gt_b & ~pred_b,
        ~gt_b & pred_b,
        ~gt_b & ~pred_b,
    )


def masked_mean(tensor, mask):
    mask = mask.to(dtype=tensor.dtype, device=tensor.device)
    count = int(mask.sum().item())
    if count <= 0:
        return 0.0, 0
    return float((tensor * mask).sum().item() / count), count


def masked_ratio_gt(tensor, mask, threshold=0.1):
    mask = mask.to(dtype=torch.bool, device=tensor.device)
    count = int(mask.sum().item())
    if count <= 0:
        return 0.0
    return float(((tensor > threshold) & mask).sum().item() / count)


def aggregate_region_stats(accumulator, gt, pred, G, delta_logits):
    tp, fn, fp, tn = confusion_regions(gt, pred)
    for key, region in (("TP", tp), ("FN", fn), ("FP", fp), ("TN", tn)):
        g_mean, count = masked_mean(G, region)
        for prefix in (f"G_{key}_mean", f"G_{key}_count", f"G_gt_0_1_{key}",
                       f"delta_logit_{key}", f"delta_{key}_count"):
            accumulator.setdefault(prefix, 0.0 if "count" not in prefix else 0)
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


def run_eval_mode(
    model,
    loader,
    device,
    threshold,
    name,
    use_soft_graph,
    lambda_override=None,
    identity_dir_convs=True,
):
    mode = set_soft_graph_eval_mode(
        model,
        use_soft_graph=use_soft_graph,
        lambda_override=lambda_override,
        identity_dir_convs=identity_dir_convs and use_soft_graph,
    )
    configure_graph_diagnostics(model, enabled=use_soft_graph)

    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    cldice_list = []
    comp_list = []
    cov_list = []

    diag_acc = {
        "lambda_eff": 0.0,
        "message_over_feature_norm": 0.0,
        "residual_over_feature_norm": 0.0,
        "surface_logit_delta_relative": 0.0,
        "diag_batches": 0,
    }
    region_acc = {}
    cldice_fn = SoftCLDiceLoss(iter_num=10).to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc=name, leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            skeletons = batch["skeleton"].to(device)

            outputs = model(images)
            surface_logits = outputs[0]
            metrics = binary_metrics_from_logits(
                surface_logits,
                masks,
                threshold=threshold,
            )
            iou_list.append(metrics["iou"])
            f1_list.append(metrics["f1"])
            precision_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])

            probs = torch.sigmoid(surface_logits)
            cldice_val = 1.0 - cldice_fn(probs, skeletons)
            cldice_list.append(float(cldice_val.item()))

            pred_np = (probs >= threshold).cpu().numpy()
            skel_np = (skeletons > 0.5).cpu().numpy()
            for b in range(pred_np.shape[0]):
                pred_u8 = (pred_np[b, 0] * 255).astype(np.uint8)
                comp_list.append(count_components(pred_u8))
                cov_list.append(skeleton_coverage(pred_np[b, 0], skel_np[b, 0]))

            if not use_soft_graph:
                continue

            head = model.swin_unet.guided_head
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
            gt = masks.unsqueeze(1) if masks.dim() == 3 else masks
            pred = torch.sigmoid(surface_logits) >= threshold
            delta = diag.get("graph_logit_delta")
            if delta is None:
                delta = diag["surface_post_graph_logits"] - diag["surface_pre_logits"]
            aggregate_region_stats(region_acc, gt, pred, export["G"], delta)

    diag_batches = max(diag_acc["diag_batches"], 1)
    region_stats = finalize_region_stats(region_acc)
    batches = max(len(iou_list), 1)

    return SweepResult(
        name=name,
        use_soft_graph=bool(mode["use_soft_graph"]),
        lambda_eff=float(mode["lambda_eff"]),
        iou=float(np.mean(iou_list)),
        f1=float(np.mean(f1_list)),
        precision=float(np.mean(precision_list)),
        recall=float(np.mean(recall_list)),
        skeleton_cldice=float(np.mean(cldice_list)),
        num_components_mean=float(np.mean(comp_list)),
        gt_skeleton_coverage_mean=float(np.mean(cov_list)),
        message_over_feature_norm_mean=diag_acc["message_over_feature_norm"] / diag_batches,
        residual_over_feature_norm_mean=diag_acc["residual_over_feature_norm"] / diag_batches,
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
        extra={
            "identity_dir_convs": bool(mode.get("identity_dir_convs", False)),
            "evaluated_batches": batches,
        },
    )


def format_row(result: SweepResult) -> str:
    return (
        f"{result.name:<14} lam={result.lambda_eff:.3f} "
        f"IoU={result.iou:.4f} F1={result.f1:.4f} "
        f"P={result.precision:.4f} R={result.recall:.4f} "
        f"clDice={result.skeleton_cldice:.4f} "
        f"#comp={result.num_components_mean:.2f} "
        f"GTcov={result.gt_skeleton_coverage_mean:.4f} | "
        f"dFN={result.delta_logit_FN:+.4f} dFP={result.delta_logit_FP:+.4f} "
        f"G_FN={result.G_FN_mean:.4f} G_FP={result.G_FP_mean:.4f}"
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Frozen checkpoint: {args.model_path}")

    model, checkpoint, structure_profile = load_frozen_backbone(args, device)
    print(f"structure_profile={structure_profile}")
    print_topology_coefficients(model)
    print(
        "[INFO] Graph module: untrained, dir_convs reset to identity for eval-only sweep"
    )

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"{args.split} samples={len(dataset)}, threshold={args.threshold}")

    results = []
    off = run_eval_mode(
        model,
        loader,
        device,
        args.threshold,
        name="graph_off",
        use_soft_graph=False,
    )
    results.append(off)
    print(format_row(off))

    for lam in args.lambda_values:
        name = f"lambda_{lam:.2f}"
        row = run_eval_mode(
            model,
            loader,
            device,
            args.threshold,
            name=name,
            use_soft_graph=True,
            lambda_override=lam,
            identity_dir_convs=True,
        )
        results.append(row)
        print(format_row(row))

    print("\n" + "=" * 80)
    print("Region diagnostics (delta_logit = post_graph - pre_graph)")
    print("=" * 80)
    header = (
        f"{'name':<14} {'lam':>5} | "
        f"{'dTP':>7} {'dFN':>7} {'dFP':>7} {'dTN':>7} | "
        f"{'G_TP':>6} {'G_FN':>6} {'G_FP':>6} {'G_TN':>6} | "
        f"{'G>0.1 FN':>8} {'G>0.1 FP':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        if not row.use_soft_graph:
            print(
                f"{row.name:<14} {'off':>5} | "
                f"{'—':>7} {'—':>7} {'—':>7} {'—':>7} | "
                f"{'—':>6} {'—':>6} {'—':>6} {'—':>6} | "
                f"{'—':>8} {'—':>8}"
            )
            continue
        print(
            f"{row.name:<14} {row.lambda_eff:5.3f} | "
            f"{row.delta_logit_TP:+7.4f} {row.delta_logit_FN:+7.4f} "
            f"{row.delta_logit_FP:+7.4f} {row.delta_logit_TN:+7.4f} | "
            f"{row.G_TP_mean:6.4f} {row.G_FN_mean:6.4f} "
            f"{row.G_FP_mean:6.4f} {row.G_TN_mean:6.4f} | "
            f"{row.G_gt_0_1_FN:8.4f} {row.G_gt_0_1_FP:8.4f}"
        )

    best_lam = max(
        [r for r in results if r.use_soft_graph],
        key=lambda r: r.f1,
        default=None,
    )
    if best_lam is not None:
        print(
            f"\nvs graph_off: dIoU={best_lam.iou - off.iou:+.4f} "
            f"dF1={best_lam.f1 - off.f1:+.4f} "
            f"dclDice={best_lam.skeleton_cldice - off.skeleton_cldice:+.4f} "
            f"d#comp={best_lam.num_components_mean - off.num_components_mean:+.2f} "
            f"dGTcov={best_lam.gt_skeleton_coverage_mean - off.gt_skeleton_coverage_mean:+.4f}"
        )
        hopeful = (
            best_lam.delta_logit_FN > 0
            and best_lam.delta_logit_FP <= 0.05
            and best_lam.delta_logit_TN >= -0.05
            and best_lam.skeleton_cldice >= off.skeleton_cldice - 0.005
        )
        print(
            "Direction check (FN up, FP/TN stable, clDice not drop): "
            + ("PASS" if hopeful else "FAIL")
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "soft_graph_lambda_sweep.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": args.model_path,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "split": args.split,
                "threshold": args.threshold,
                "structure_profile": structure_profile,
                "lambda_values": args.lambda_values,
                "identity_dir_convs": True,
                "results": [asdict(item) for item in results],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    txt_path = os.path.join(output_dir, "soft_graph_lambda_sweep.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("Soft graph eval-only lambda sweep (frozen backbone)\n")
        handle.write(f"model={args.model_path}\n")
        handle.write(f"split={args.split}, threshold={args.threshold}\n\n")
        for row in results:
            handle.write(format_row(row) + "\n")
    print(f"\nSaved: {report_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
