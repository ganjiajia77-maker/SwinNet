import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model, select_stage_outputs
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_connectivity_target, build_stage_skeleton_target
from topology_direction_constants import CONNECTIVITY_DIR_NAMES


CARDINAL = [0, 2, 4, 6]
DIAGONAL = [1, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--name", type=str, default="single")
    parser.add_argument("--baseline_model_path", type=str, default="")
    parser.add_argument("--current_model_path", type=str, default="")
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--current_name", type=str, default="current")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--stage", type=str, default="stage3_refine", choices=["stage2_refine", "stage3_refine"])
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--connectivity_threshold", type=float, default=0.5)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
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


def safe_auc(gt, prob):
    gt = np.asarray(gt).astype(np.int32)
    prob = np.asarray(prob).astype(np.float32)
    if gt.size == 0 or np.unique(gt).size < 2:
        return float("nan")
    return float(roc_auc_score(gt, prob))


def metric_counts(prob, gt, threshold):
    pred = prob >= float(threshold)
    gt = gt.bool()
    tp = torch.logical_and(pred, gt).sum().item()
    fp = torch.logical_and(pred, ~gt).sum().item()
    fn = torch.logical_and(~pred, gt).sum().item()
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return iou, f1, precision, recall, pred


def set_capture_diagnostics(model, enabled):
    for module in model.modules():
        if hasattr(module, "capture_diagnostics"):
            module.capture_diagnostics = bool(enabled)


def collect_run(name, model_path, base_args, loader, device):
    args = argparse.Namespace(**vars(base_args))
    args.model_path = model_path
    model = load_model(args, device)
    set_capture_diagnostics(model, True)

    surface_probs = []
    masks_all = []
    conn_strength_maps = []
    conn_probs = []
    conn_logits = []
    conn_gts = []
    conn_valids = []
    diag_rows = defaultdict(list)
    steps = []
    images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if base_args.max_batches > 0 and batch_idx >= base_args.max_batches:
                break
            image = batch["image"].to(device)
            mask = (batch["mask"].to(device) > 0.5)
            skeleton = batch["skeleton"].to(device).float()
            output = model(image, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = output[0]
            surface_prob = torch.sigmoid(surface_logits)
            if mask.shape[-2:] != surface_prob.shape[-2:]:
                mask = F.interpolate(mask.float(), size=surface_prob.shape[-2:], mode="nearest") > 0.5

            selected = select_stage_outputs(output[-1])
            if base_args.stage not in selected:
                raise RuntimeError(f"{name}: missing {base_args.stage} in structure outputs.")
            stage_output = selected[base_args.stage]
            c_logit = stage_output["connectivity"]
            c_prob = torch.sigmoid(c_logit)
            c_gt_skeleton = build_stage_skeleton_target(skeleton, c_prob.shape[-2:]).to(device)
            c_gt = build_connectivity_target(c_gt_skeleton).to(device)
            valid = c_gt_skeleton > 0.5
            valid8 = valid.expand_as(c_gt)
            conn_strength = c_prob.topk(k=min(2, c_prob.shape[1]), dim=1).values.mean(dim=1, keepdim=True)
            conn_strength_up = F.interpolate(
                conn_strength,
                size=surface_prob.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            for item in output[-1]:
                if item.get("stage") in (2, 3):
                    steps.append((int(item.get("stage")), item.get("refinement_step", None)))
            for module in model.modules():
                diag = getattr(module, "last_diagnostics", None)
                if not isinstance(diag, dict):
                    continue
                for key, value in diag.items():
                    diag_rows[key].append(float(value))

            surface_probs.append(surface_prob.detach().cpu())
            masks_all.append(mask.detach().cpu())
            conn_strength_maps.append(conn_strength_up.detach().cpu())
            conn_probs.append(c_prob.detach().cpu())
            conn_logits.append(c_logit.detach().cpu())
            conn_gts.append((c_gt.detach().cpu() > 0.5))
            conn_valids.append(valid8.detach().cpu())
            images += int(image.shape[0])

    surface_prob = torch.cat(surface_probs, dim=0)
    mask = torch.cat(masks_all, dim=0)
    iou, f1, precision, recall, pred = metric_counts(surface_prob, mask, base_args.threshold)
    c_prob_tensor = torch.cat(conn_probs, dim=0)
    c_logit_tensor = torch.cat(conn_logits, dim=0)
    c_gt_tensor = torch.cat(conn_gts, dim=0)
    c_valid_tensor = torch.cat(conn_valids, dim=0)
    c_prob_flat = c_prob_tensor[c_valid_tensor].reshape(-1).numpy()
    c_logit_flat = c_logit_tensor[c_valid_tensor].reshape(-1).numpy()
    c_gt_flat = c_gt_tensor[c_valid_tensor].reshape(-1).numpy().astype(np.int32)
    delta_c = float(c_prob_flat[c_gt_flat == 1].mean() - c_prob_flat[c_gt_flat == 0].mean())
    pooled_auc = safe_auc(c_gt_flat, c_prob_flat)

    per_dir_auc = []
    per_dir_pos = []
    per_dir_neg = []
    per_dir_logit_mean = []
    per_dir_logit_std = []
    for direction in range(8):
        valid_d = c_valid_tensor[:, direction]
        prob_d = c_prob_tensor[:, direction][valid_d].reshape(-1).numpy()
        logit_d = c_logit_tensor[:, direction][valid_d].reshape(-1).numpy()
        gt_d = c_gt_tensor[:, direction][valid_d].reshape(-1).numpy().astype(np.int32)
        per_dir_auc.append(safe_auc(gt_d, prob_d))
        per_dir_pos.append(float(prob_d[gt_d == 1].mean()) if np.any(gt_d == 1) else float("nan"))
        per_dir_neg.append(float(prob_d[gt_d == 0].mean()) if np.any(gt_d == 0) else float("nan"))
        per_dir_logit_mean.append(float(logit_d.mean()) if logit_d.size else float("nan"))
        per_dir_logit_std.append(float(logit_d.std()) if logit_d.size else float("nan"))

    def group_logit_stats(indices):
        group_logits = []
        for direction in indices:
            valid_d = c_valid_tensor[:, direction]
            group_logits.append(c_logit_tensor[:, direction][valid_d].reshape(-1).numpy())
        values = np.concatenate(group_logits, axis=0) if group_logits else np.array([], dtype=np.float32)
        return (
            float(values.mean()) if values.size else float("nan"),
            float(values.std()) if values.size else float("nan"),
        )

    cardinal_logit_mean, cardinal_logit_std = group_logit_stats(CARDINAL)
    diagonal_logit_mean, diagonal_logit_std = group_logit_stats(DIAGONAL)

    return {
        "name": name,
        "model_path": model_path,
        "images": images,
        "surface_prob": surface_prob,
        "mask": mask,
        "pred": pred,
        "conn_strength": torch.cat(conn_strength_maps, dim=0),
        "surface": (iou, f1, precision, recall),
        "c_pos": float(c_prob_flat[c_gt_flat == 1].mean()),
        "c_neg": float(c_prob_flat[c_gt_flat == 0].mean()),
        "delta_c": delta_c,
        "pooled_auc": pooled_auc,
        "per_dir_auc": per_dir_auc,
        "per_dir_pos": per_dir_pos,
        "per_dir_neg": per_dir_neg,
        "per_dir_logit_mean": per_dir_logit_mean,
        "per_dir_logit_std": per_dir_logit_std,
        "macro_per_dir_auc": float(np.nanmean(np.array(per_dir_auc, dtype=np.float32))),
        "cardinal_logit_mean": cardinal_logit_mean,
        "cardinal_logit_std": cardinal_logit_std,
        "diagonal_logit_mean": diagonal_logit_mean,
        "diagonal_logit_std": diagonal_logit_std,
        "diag_rows": {key: float(np.mean(value)) for key, value in diag_rows.items() if value},
        "steps": sorted(set(steps), key=str),
    }


def region_mean(tensor, mask):
    if tensor.numel() == 0 or not mask.any():
        return float("nan")
    return float(tensor[mask].mean().item())


def print_run(row):
    iou, f1, precision, recall = row["surface"]
    print(f"\n[{row['name']}]")
    print(f"  model={row['model_path']}")
    print(f"  Surface IoU={iou:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
    print(f"  C_pos={row['c_pos']:.4f}, C_neg={row['c_neg']:.4f}, DeltaC={row['delta_c']:.4f}, pooled_AUROC={row['pooled_auc']:.4f}")
    print(f"  macro_per_dir_AUROC={row['macro_per_dir_auc']:.4f}")
    print(
        "  cardinal logits: "
        f"mean={row['cardinal_logit_mean']:.4f}, std={row['cardinal_logit_std']:.4f}"
    )
    print(
        "  diagonal logits: "
        f"mean={row['diagonal_logit_mean']:.4f}, std={row['diagonal_logit_std']:.4f}"
    )
    print(f"  conn_strength mean={row['conn_strength'].mean().item():.4f}, std={row['conn_strength'].std(unbiased=False).item():.4f}")
    print("  per-direction AUROC / C_pos / C_neg / logit_mean / logit_std:")
    for idx, direction_name in enumerate(CONNECTIVITY_DIR_NAMES):
        print(
            f"    {direction_name}: "
            f"auc={row['per_dir_auc'][idx]:.4f}, "
            f"pos={row['per_dir_pos'][idx]:.4f}, "
            f"neg={row['per_dir_neg'][idx]:.4f}, "
            f"logit_mean={row['per_dir_logit_mean'][idx]:.4f}, "
            f"logit_std={row['per_dir_logit_std'][idx]:.4f}"
        )
    print(f"  structure steps present={row['steps']}")
    for key in (
        "gate_mean",
        "gate_max",
        "conn_strength_mean",
        "gamma1",
        "gate_residual_relative_norm",
        "total_residual_relative_norm",
        "reliability_beta",
        "reliability_context_mean",
    ):
        if key in row["diag_rows"]:
            print(f"  diag {key}={row['diag_rows'][key]:.6f}")


def main():
    args = parse_args()
    single_mode = bool(args.model_path)
    compare_mode = bool(args.baseline_model_path or args.current_model_path)
    if single_mode and compare_mode:
        raise ValueError("Use either --model_path for single-checkpoint diagnostics or --baseline_model_path/--current_model_path for comparison.")
    if not single_mode and not (args.baseline_model_path and args.current_model_path):
        raise ValueError("Pass --model_path, or pass both --baseline_model_path and --current_model_path.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    if single_mode:
        row = collect_run(args.name, args.model_path, args, loader, device)
        print(f"split={args.split}, stage={args.stage}, surface_threshold={args.threshold}, images={row['images']}")
        print("\nConnectivity / Gate Single Checkpoint")
        print_run(row)
        return

    baseline = collect_run(args.baseline_name, args.baseline_model_path, args, loader, device)
    current = collect_run(args.current_name, args.current_model_path, args, loader, device)
    print(f"split={args.split}, stage={args.stage}, surface_threshold={args.threshold}, images={current['images']}")
    print("\nConnectivity / Gate Shift")
    for row in (baseline, current):
        print_run(row)

    gt = current["mask"]
    b_pred = baseline["pred"]
    c_pred = current["pred"]
    fn_to_tp = (~b_pred) & c_pred & gt
    tn_to_fp = (~b_pred) & c_pred & (~gt)
    fp_to_tn = b_pred & (~c_pred) & (~gt)
    tp_to_fn = b_pred & (~c_pred) & gt
    print("\nSurface Changed Regions: conn_strength mean")
    print("  region      baseline    current     count")
    for label, mask in (
        ("FN->TP", fn_to_tp),
        ("TN->FP", tn_to_fp),
        ("FP->TN", fp_to_tn),
        ("TP->FN", tp_to_fn),
    ):
        print(
            f"  {label:<9} "
            f"{region_mean(baseline['conn_strength'], mask):>9.4f} "
            f"{region_mean(current['conn_strength'], mask):>9.4f} "
            f"{int(mask.sum().item()):>9d}"
        )


if __name__ == "__main__":
    main()
