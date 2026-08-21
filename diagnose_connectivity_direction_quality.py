import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import (
    DIR_NAMES,
    DIR_OFFSETS,
    load_model,
    resize_like,
    select_stage_outputs,
)
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_connectivity_target


CONNECTIVITY_DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
CONNECTIVITY_DIR_OFFSETS = torch.tensor(
    [
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
    ],
    dtype=torch.float32,
)


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collect_batches(loader, max_batches):
    batches = []
    for idx, batch in enumerate(loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batches.append(batch)
    return batches


def reciprocal_symmetry_error(prob):
    offsets = CONNECTIVITY_DIR_OFFSETS.to(prob.device, prob.dtype)
    opposite = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], device=prob.device)
    total_err = 0.0
    total_count = 0
    for direction, (dy, dx) in enumerate(offsets.tolist()):
        dy = int(dy)
        dx = int(dx)
        opp = int(opposite[direction].item())
        lhs = prob[:, direction]
        rhs = prob[:, opp]
        rhs_shifted = torch.roll(rhs, shifts=(-dy, -dx), dims=(1, 2))
        valid = torch.ones_like(lhs, dtype=torch.bool)
        if dy > 0:
            valid[:, -dy:, :] = False
        elif dy < 0:
            valid[:, : -dy, :] = False
        if dx > 0:
            valid[:, :, -dx:] = False
        elif dx < 0:
            valid[:, :, : -dx] = False
        diff = (lhs - rhs_shifted).abs()[valid]
        total_err += float(diff.sum().item())
        total_count += int(diff.numel())
    return total_err / max(total_count, 1)


def direction_stats(direction_logits, direction_gt, valid_mask):
    pred_vec = F.normalize(direction_logits.float(), dim=1, eps=1e-6)
    gt_vec = F.normalize(direction_gt.float(), dim=1, eps=1e-6)
    offsets = DIR_OFFSETS.to(direction_logits.device, direction_logits.dtype)
    offsets = F.normalize(offsets, dim=1, eps=1e-6)
    pred_score = torch.einsum("bchw,kc->bkhw", pred_vec, offsets)
    gt_score = torch.einsum("bchw,kc->bkhw", gt_vec, offsets)
    pred_idx = pred_score.argmax(dim=1)
    gt_idx = gt_score.argmax(dim=1)
    cosine = (pred_vec * gt_vec).sum(dim=1).clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(cosine))
    mask = valid_mask.squeeze(1).bool()
    pred_flat = pred_idx[mask].detach().cpu().numpy()
    gt_flat = gt_idx[mask].detach().cpu().numpy()
    angle_flat = angle_deg[mask].detach().cpu().numpy()
    pred_hist = np.bincount(pred_flat, minlength=8)
    gt_hist = np.bincount(gt_flat, minlength=8)
    return {
        "count": int(mask.sum().item()),
        "pred_hist": pred_hist,
        "gt_hist": gt_hist,
        "angle_mean": float(angle_flat.mean()) if angle_flat.size else float("nan"),
        "angle_median": float(np.median(angle_flat)) if angle_flat.size else float("nan"),
        "accuracy_8dir": float((pred_flat == gt_flat).mean()) if pred_flat.size else float("nan"),
    }


def connectivity_stats(connectivity_prob, connectivity_gt, valid_mask):
    prob = connectivity_prob.detach().cpu()
    gt = (connectivity_gt.detach().cpu() > 0.5)
    valid = valid_mask.detach().cpu().bool().expand_as(gt)
    pred = (prob >= 0.5) & valid

    per_dir = []
    pos_probs = []
    neg_probs = []
    flat_prob = []
    flat_gt = []
    for direction in range(8):
        p = prob[:, direction][valid[:, direction]]
        g = gt[:, direction][valid[:, direction]]
        if p.numel() == 0:
            per_dir.append({
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "gt_pos": 0.0,
                "gt_neg": 0.0,
                "pos_sum": 0.0,
                "neg_sum": 0.0,
            })
            continue
        pred_d = pred[:, direction][valid[:, direction]]
        tp = float((pred_d & g).sum().item())
        fp = float((pred_d & (~g)).sum().item())
        fn = float(((~pred_d) & g).sum().item())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        per_dir.append({"precision": precision, "recall": recall, "f1": f1})
        pos = p[g]
        neg = p[~g]
        pos_probs.append(pos.numpy())
        neg_probs.append(neg.numpy())
        flat_prob.append(p.numpy())
        flat_gt.append(g.numpy().astype(np.int32))
        per_dir[-1].update(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_pos": float(g.sum().item()),
                "gt_neg": float((~g).sum().item()),
                "pos_sum": float(pos.sum().item()),
                "neg_sum": float(neg.sum().item()),
            }
        )

    flat_prob = np.concatenate(flat_prob, axis=0) if flat_prob else np.empty((0,), dtype=np.float32)
    flat_gt = np.concatenate(flat_gt, axis=0) if flat_gt else np.empty((0,), dtype=np.int32)
    pos_prob = np.concatenate(pos_probs, axis=0) if pos_probs else np.empty((0,), dtype=np.float32)
    neg_prob = np.concatenate(neg_probs, axis=0) if neg_probs else np.empty((0,), dtype=np.float32)
    auc = float(roc_auc_score(flat_gt, flat_prob)) if np.unique(flat_gt).size > 1 else float("nan")
    auprc = float(average_precision_score(flat_gt, flat_prob)) if np.unique(flat_gt).size > 1 else float("nan")
    macro_precision = float(np.mean([row["precision"] for row in per_dir]))
    macro_recall = float(np.mean([row["recall"] for row in per_dir]))
    macro_f1 = float(np.mean([row["f1"] for row in per_dir]))
    reciprocity = reciprocal_symmetry_error(prob)
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_dir": per_dir,
        "positive_mean_prob": float(pos_prob.mean()) if pos_prob.size else float("nan"),
        "negative_mean_prob": float(neg_prob.mean()) if neg_prob.size else float("nan"),
        "auroc": auc,
        "auprc": auprc,
        "reciprocal_symmetry_error": reciprocity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--stage", type=str, default="stage3_refine", choices=["stage2_refine", "stage3_refine"])
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
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
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
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
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    args = parser.parse_args()

    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    batches = collect_batches(loader, args.max_batches)

    stage_key = args.stage
    conn_gt_rows = []
    conn_prob_rows = []
    dir_stats_acc = {
        "pred_hist": np.zeros(8, dtype=np.int64),
        "gt_hist": np.zeros(8, dtype=np.int64),
        "count": 0,
        "angle_sum": 0.0,
        "angle_median_values": [],
        "accuracy_count": 0,
    }
    conn_tp = np.zeros(8, dtype=np.float64)
    conn_fp = np.zeros(8, dtype=np.float64)
    conn_fn = np.zeros(8, dtype=np.float64)
    conn_pos_count = np.zeros(8, dtype=np.float64)
    conn_neg_count = np.zeros(8, dtype=np.float64)
    conn_pos_sum = np.zeros(8, dtype=np.float64)
    conn_neg_sum = np.zeros(8, dtype=np.float64)
    pos_probs = []
    neg_probs = []
    flat_conn_prob = []
    flat_conn_gt = []
    symmetry_errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=f"Evaluate {stage_key}")):
            images = batch["image"].to(device)
            masks = (batch["mask"].to(device) > 0.5)
            skeleton_raw = batch["skeleton"].to(device).float()
            direction_gt = batch["direction_gt"].to(device).float()

            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0]
            skeleton = resize_like(skeleton_raw, surface_logits, mode="nearest") > 0.5
            selected = select_stage_outputs(outputs[-1])
            if stage_key not in selected:
                raise RuntimeError(f"Stage {stage_key} not present in structure outputs.")
            stage_output = selected[stage_key]
            c_prob = torch.sigmoid(stage_output["connectivity"])
            d_logits = stage_output["direction"]
            c_gt = resize_like(
                build_connectivity_target(skeleton.float()),
                c_prob,
                mode="nearest",
            )
            d_gt = resize_like(direction_gt, d_logits[:, :1], mode="nearest")
            skeleton_dir = resize_like(skeleton.float(), d_logits[:, :1], mode="nearest") > 0.5

            conn_valid = resize_like(skeleton.float(), c_prob[:, :1], mode="nearest") > 0.5
            conn = connectivity_stats(c_prob, c_gt, conn_valid)
            for i, item in enumerate(conn["per_dir"]):
                conn_tp[i] += item["tp"]
                conn_fp[i] += item["fp"]
                conn_fn[i] += item["fn"]
                conn_pos_count[i] += item["gt_pos"]
                conn_neg_count[i] += item["gt_neg"]
                conn_pos_sum[i] += item["pos_sum"]
                conn_neg_sum[i] += item["neg_sum"]
            pos_probs.append(np.array([conn["positive_mean_prob"]], dtype=np.float32))
            neg_probs.append(np.array([conn["negative_mean_prob"]], dtype=np.float32))
            flat_conn_prob.append(c_prob.detach().cpu().reshape(-1).numpy())
            flat_conn_gt.append((c_gt.detach().cpu().reshape(-1) > 0.5).numpy().astype(np.int32))
            symmetry_errors.append(reciprocal_symmetry_error(c_prob.detach().cpu()))

            d_stats = direction_stats(d_logits, d_gt, skeleton_dir.float())
            dir_stats_acc["pred_hist"] += d_stats["pred_hist"]
            dir_stats_acc["gt_hist"] += d_stats["gt_hist"]
            dir_stats_acc["count"] += d_stats["count"]
            dir_stats_acc["angle_sum"] += d_stats["angle_mean"] * d_stats["count"]
            dir_stats_acc["angle_median_values"].append(d_stats["angle_median"])
            dir_stats_acc["accuracy_count"] += d_stats["accuracy_8dir"] * d_stats["count"]

    flat_conn_prob = np.concatenate(flat_conn_prob, axis=0)
    flat_conn_gt = np.concatenate(flat_conn_gt, axis=0)
    overall_auc = float(roc_auc_score(flat_conn_gt, flat_conn_prob))
    overall_auprc = float(average_precision_score(flat_conn_gt, flat_conn_prob))
    per_dir_precision = conn_tp / (conn_tp + conn_fp + 1e-8)
    per_dir_recall = conn_tp / (conn_tp + conn_fn + 1e-8)
    per_dir_f1 = 2.0 * per_dir_precision * per_dir_recall / (per_dir_precision + per_dir_recall + 1e-8)
    macro_precision = float(per_dir_precision.mean())
    macro_recall = float(per_dir_recall.mean())
    macro_f1 = float(per_dir_f1.mean())
    positive_mean = float((conn_pos_sum / np.maximum(conn_pos_count, 1.0)).mean())
    negative_mean = float((conn_neg_sum / np.maximum(conn_neg_count, 1.0)).mean())
    reciprocal_error = float(np.mean(symmetry_errors))

    print(f"\nCheckpoint: {args.model_path}")
    print(f"Stage: {stage_key}")
    print(f"split={args.split}, batches={len(batches)}, images={sum(batch['image'].shape[0] for batch in batches)}")
    print(f"threshold={args.threshold}, seed={args.seed}")
    print("\nConnectivity")
    print(f"  8-dir macro Precision: {macro_precision:.4f}")
    print(f"  8-dir macro Recall:    {macro_recall:.4f}")
    print(f"  8-dir macro F1:        {macro_f1:.4f}")
    print(f"  GT connection=1 mean C_prob: {positive_mean:.4f}")
    print(f"  GT connection=0 mean C_prob: {negative_mean:.4f}")
    print(f"  AUROC: {overall_auc:.4f}")
    print(f"  AUPRC: {overall_auprc:.4f}")
    print(f"  Reciprocal symmetry error: {reciprocal_error:.4f}")
    print("  Per-direction F1:")
    for name, value in zip(CONNECTIVITY_DIR_NAMES, per_dir_f1):
        print(f"    {name}: {value:.4f}")

    print("\nDirection")
    accuracy_8dir = dir_stats_acc["accuracy_count"] / max(dir_stats_acc["count"], 1)
    angle_mean = dir_stats_acc["angle_sum"] / max(dir_stats_acc["count"], 1)
    angle_median = float(np.median(np.array(dir_stats_acc["angle_median_values"], dtype=np.float32)))
    print(f"  8-dir accuracy:        {accuracy_8dir:.4f}")
    print(f"  Mean angular error:    {angle_mean:.4f}")
    print(f"  Median angular error:   {angle_median:.4f}")
    print(f"  Pred histogram: {dict(zip(DIR_NAMES, dir_stats_acc['pred_hist'].tolist()))}")
    print(f"  GT histogram:   {dict(zip(DIR_NAMES, dir_stats_acc['gt_hist'].tolist()))}")


if __name__ == "__main__":
    main()
