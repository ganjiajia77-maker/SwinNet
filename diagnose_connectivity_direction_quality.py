import argparse
import os
import random
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import (
    load_model,
    resize_like,
    select_stage_outputs,
)
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.cldice_loss import soft_skeletonize
from losses.road_losses import build_connectivity_target, build_stage_skeleton_target
from direction_target_utils import build_continuous_direction_target
from topology_direction_constants import (
    AXIAL_DIR_NAMES,
    AXIAL_DIRECTIONS,
    CONNECTIVITY_DIR_NAMES,
    CONNECTIVITY_DIRECTIONS,
    CONNECTIVITY_OPPOSITE,
    axial_double_angle_basis,
)


CONNECTIVITY_DIR_OFFSETS = torch.tensor(CONNECTIVITY_DIRECTIONS, dtype=torch.float32)
AXIAL_DIR_OFFSETS = torch.tensor(AXIAL_DIRECTIONS, dtype=torch.float32)


def install_connectivity_ablation(model, mode):
    if mode == "full":
        return 0
    replaced = 0
    for module in model.modules():
        head = getattr(module, "connectivity_head", None)
        if head is None or not hasattr(head, "edge_mlp"):
            continue
        original_forward = head.forward

        def ablated_forward(self, feature, direction_alignment=None, skeleton_prob=None, _original=original_forward):
            if mode in ("no_dir", "no_ske_dir"):
                direction_alignment = None
            if mode in ("no_ske", "no_ske_dir"):
                skeleton_prob = None
            return _original(feature, direction_alignment, skeleton_prob=skeleton_prob)

        head.forward = types.MethodType(ablated_forward, head)
        replaced += 1
    return replaced


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
    direction_norm = direction_gt.float().norm(dim=1, keepdim=True)
    direction_valid = direction_norm > 1e-6
    gt_vec = F.normalize(direction_gt.float(), dim=1, eps=1e-6)
    offsets = AXIAL_DIR_OFFSETS.to(direction_logits.device, direction_logits.dtype)
    theta = torch.atan2(offsets[:, 0], offsets[:, 1])
    axis_basis = torch.stack(
        [torch.cos(2.0 * theta), torch.sin(2.0 * theta)],
        dim=1,
    )
    pred_score = torch.einsum("bchw,kc->bkhw", pred_vec, axis_basis)
    gt_score = torch.einsum("bchw,kc->bkhw", gt_vec, axis_basis)
    pred_idx = pred_score.argmax(dim=1)
    gt_idx = gt_score.argmax(dim=1)
    cosine = (pred_vec * gt_vec).sum(dim=1).clamp(-1.0, 1.0)
    angle_deg = 0.5 * torch.rad2deg(torch.acos(cosine))
    mask = (valid_mask.bool() & direction_valid).squeeze(1)
    pred_flat = pred_idx[mask].detach().cpu().numpy()
    gt_flat = gt_idx[mask].detach().cpu().numpy()
    angle_flat = angle_deg[mask].detach().cpu().numpy()
    pred_hist = np.bincount(pred_flat, minlength=4)
    gt_hist = np.bincount(gt_flat, minlength=4)
    return {
        "count": int(mask.sum().item()),
        "pred_hist": pred_hist,
        "gt_hist": gt_hist,
        "angle_mean": float(angle_flat.mean()) if angle_flat.size else float("nan"),
        "angle_median": float(np.median(angle_flat)) if angle_flat.size else float("nan"),
        "accuracy_axial4": float((pred_flat == gt_flat).mean()) if pred_flat.size else float("nan"),
    }


def connectivity_stats(connectivity_prob, connectivity_gt, valid_mask, threshold=0.5):
    prob = connectivity_prob.detach().cpu()
    gt = (connectivity_gt.detach().cpu() > 0.5)
    valid = valid_mask.detach().cpu().bool().expand_as(gt)
    pred = (prob >= float(threshold)) & valid

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


def surface_cldice_score(surface_prob, surface_gt, iter_num=10):
    prob = surface_prob.float().clamp(0.0, 1.0)
    target = surface_gt.float().clamp(0.0, 1.0)
    pred_skel = soft_skeletonize(prob, iter_num=iter_num)
    target_skel = soft_skeletonize(target, iter_num=iter_num)
    tprec = (pred_skel * target).sum(dim=(1, 2, 3)) / (
        pred_skel.sum(dim=(1, 2, 3)) + 1e-8
    )
    tsens = (target_skel * prob).sum(dim=(1, 2, 3)) / (
        target_skel.sum(dim=(1, 2, 3)) + 1e-8
    )
    cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)
    return cldice.detach().cpu().numpy()


def _component_stats(mask_bool, short_area_threshold):
    u8 = mask_bool.astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.empty((0,), dtype=np.int32)
    total = float(areas.sum()) if areas.size else 0.0
    return {
        "components": float(num - 1),
        "short_components": float((areas < short_area_threshold).sum()) if areas.size else 0.0,
        "largest_ratio": float(areas.max() / total) if total > 0 else 0.0,
    }


def surface_fragmentation_stats(surface_prob, surface_gt, threshold=0.2, short_area_threshold=20):
    pred = (surface_prob.detach().cpu().numpy()[:, 0] >= float(threshold))
    gt = (surface_gt.detach().cpu().numpy()[:, 0] > 0.5)
    rows = []
    for pred_mask, gt_mask in zip(pred, gt):
        pred_stats = _component_stats(pred_mask, short_area_threshold)
        gt_stats = _component_stats(gt_mask, short_area_threshold)
        gt_components = max(gt_stats["components"], 1.0)
        rows.append(
            {
                "pred_components": pred_stats["components"],
                "gt_components": gt_stats["components"],
                "fragmentation_index": pred_stats["components"] / gt_components,
                "extra_components": max(pred_stats["components"] - gt_stats["components"], 0.0),
                "pred_short_components": pred_stats["short_components"],
                "gt_short_components": gt_stats["short_components"],
                "pred_largest_ratio": pred_stats["largest_ratio"],
                "gt_largest_ratio": gt_stats["largest_ratio"],
            }
        )
    return rows


def _edge_slices(height, width, dy, dx):
    src_y0 = max(-dy, 0)
    src_y1 = height - max(dy, 0)
    src_x0 = max(-dx, 0)
    src_x1 = width - max(dx, 0)
    dst_y0 = src_y0 + dy
    dst_y1 = src_y1 + dy
    dst_x0 = src_x0 + dx
    dst_x1 = src_x1 + dx
    return (
        slice(src_y0, src_y1),
        slice(src_x0, src_x1),
        slice(dst_y0, dst_y1),
        slice(dst_x0, dst_x1),
    )


def _active_nodes_from_gt_edges(valid, gt_edges):
    height, width = valid.shape
    active = np.zeros_like(valid, dtype=bool)
    for direction, (dy, dx) in enumerate(CONNECTIVITY_DIR_OFFSETS.int().tolist()):
        sy, sx, dy_slice, dx_slice = _edge_slices(height, width, dy, dx)
        edge = gt_edges[direction, sy, sx] & valid[sy, sx] & valid[dy_slice, dx_slice]
        active[sy, sx] |= edge
        active[dy_slice, dx_slice] |= edge
    return active


def _count_graph_components(active, edges):
    height, width = active.shape
    node_ids = -np.ones((height, width), dtype=np.int64)
    ys, xs = np.nonzero(active)
    node_ids[ys, xs] = np.arange(len(ys), dtype=np.int64)
    if len(ys) == 0:
        return 0.0
    parent = np.arange(len(ys), dtype=np.int64)

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a, b):
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    for direction, (dy, dx) in enumerate(CONNECTIVITY_DIR_OFFSETS.int().tolist()):
        sy, sx, dy_slice, dx_slice = _edge_slices(height, width, dy, dx)
        edge = edges[direction, sy, sx] & active[sy, sx] & active[dy_slice, dx_slice]
        src_ids = node_ids[sy, sx][edge]
        dst_ids = node_ids[dy_slice, dx_slice][edge]
        for src_id, dst_id in zip(src_ids, dst_ids):
            union(src_id, dst_id)
    roots = {find(idx) for idx in range(len(ys))}
    return float(len(roots))


def connectivity_graph_fragmentation_stats(connectivity_prob, connectivity_gt, valid_mask, threshold=0.5):
    prob = connectivity_prob.detach().cpu().numpy()
    gt = connectivity_gt.detach().cpu().numpy() > 0.5
    valid = valid_mask.detach().cpu().numpy()[:, 0].astype(bool)
    pred = prob >= float(threshold)
    rows = []
    for batch_idx in range(prob.shape[0]):
        active = _active_nodes_from_gt_edges(valid[batch_idx], gt[batch_idx])
        if not active.any():
            continue
        gt_components = _count_graph_components(active, gt[batch_idx])
        pred_components = _count_graph_components(active, pred[batch_idx])
        rows.append(
            {
                "pred_components": pred_components,
                "gt_components": gt_components,
                "fragmentation_index": pred_components / max(gt_components, 1.0),
                "extra_components": max(pred_components - gt_components, 0.0),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--surface_threshold", type=float, default=0.2)
    parser.add_argument("--threshold_sweep_start", type=float, default=0.01)
    parser.add_argument("--threshold_sweep_end", type=float, default=0.20)
    parser.add_argument("--threshold_sweep_step", type=float, default=0.01)
    parser.add_argument("--fragment_short_area", type=int, default=20)
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
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=["auto", "standard", "selective"])
    parser.add_argument(
        "--connectivity_ablation",
        type=str,
        default="full",
        choices=["full", "no_dir", "no_ske", "no_ske_dir"],
        help="diagnostic-only ablation of priors passed into the connectivity head",
    )
    args = parser.parse_args()

    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    ablated_heads = install_connectivity_ablation(model, args.connectivity_ablation)
    if args.connectivity_ablation != "full":
        print(
            f"[DIAG] connectivity_ablation={args.connectivity_ablation}; "
            f"patched_heads={ablated_heads}",
            flush=True,
        )
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    stage_key = args.stage
    conn_gt_rows = []
    conn_prob_rows = []
    dir_stats_acc = {
        "pred_hist": np.zeros(4, dtype=np.int64),
        "gt_hist": np.zeros(4, dtype=np.int64),
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
    conn_pos_logit_sum = np.zeros(8, dtype=np.float64)
    conn_neg_logit_sum = np.zeros(8, dtype=np.float64)
    per_dir_prob_rows = [[] for _ in range(8)]
    per_dir_gt_rows = [[] for _ in range(8)]
    pos_probs = []
    neg_probs = []
    flat_conn_prob = []
    flat_conn_gt = []
    symmetry_errors = []
    surface_cldice_values = []
    surface_fragment_rows = []
    graph_fragment_rows = []
    batches_processed = 0
    images_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluate {stage_key}")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batches_processed += 1
            images_processed += int(batch["image"].shape[0])
            images = batch["image"].to(device)
            masks = (batch["mask"].to(device) > 0.5)
            skeleton_raw = batch["skeleton"].to(device).float()

            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0]
            surface_prob = torch.sigmoid(surface_logits)
            masks_resized = resize_like(masks.float(), surface_logits, mode="nearest")
            surface_cldice_values.extend(
                surface_cldice_score(surface_prob, masks_resized).tolist()
            )
            surface_fragment_rows.extend(
                surface_fragmentation_stats(
                    surface_prob,
                    masks_resized,
                    threshold=args.surface_threshold,
                    short_area_threshold=args.fragment_short_area,
                )
            )
            skeleton = resize_like(skeleton_raw, surface_logits, mode="nearest") > 0.5
            selected = select_stage_outputs(outputs[-1])
            if stage_key not in selected:
                raise RuntimeError(f"Stage {stage_key} not present in structure outputs.")
            stage_output = selected[stage_key]
            c_logits = stage_output["connectivity"]
            c_prob = torch.sigmoid(c_logits)
            d_logits = stage_output["direction"]
            stage_skeleton_gt = build_stage_skeleton_target(skeleton_raw, c_prob.shape[-2:]).to(device)
            c_gt = build_connectivity_target(stage_skeleton_gt)
            skeleton_dir = resize_like(stage_skeleton_gt, d_logits[:, :1], mode="nearest") > 0.5
            d_gt = build_continuous_direction_target(skeleton_dir.float(), radius=3).to(
                device=d_logits.device,
                dtype=d_logits.dtype,
            )

            conn_valid = stage_skeleton_gt > 0.5
            conn = connectivity_stats(c_prob, c_gt, conn_valid, threshold=args.threshold)
            graph_fragment_rows.extend(
                connectivity_graph_fragmentation_stats(
                    c_prob,
                    c_gt,
                    conn_valid,
                    threshold=args.threshold,
                )
            )
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
            valid_flat = conn_valid.detach().cpu().bool().expand_as(c_gt)
            flat_conn_prob.append(c_prob.detach().cpu()[valid_flat].reshape(-1).numpy())
            flat_conn_gt.append((c_gt.detach().cpu()[valid_flat].reshape(-1) > 0.5).numpy().astype(np.int32))
            c_logits_cpu = c_logits.detach().cpu()
            c_prob_cpu = c_prob.detach().cpu()
            c_gt_cpu = c_gt.detach().cpu() > 0.5
            valid_cpu = conn_valid.detach().cpu().bool()
            for dir_idx in range(8):
                dir_valid = valid_cpu[:, 0]
                dir_gt = c_gt_cpu[:, dir_idx][dir_valid]
                dir_prob = c_prob_cpu[:, dir_idx][dir_valid]
                dir_logit = c_logits_cpu[:, dir_idx][dir_valid]
                if dir_gt.numel() == 0:
                    continue
                dir_gt_np = dir_gt.reshape(-1).numpy().astype(np.int32)
                per_dir_gt_rows[dir_idx].append(dir_gt_np)
                per_dir_prob_rows[dir_idx].append(dir_prob.reshape(-1).numpy())
                pos_mask = dir_gt
                neg_mask = ~dir_gt
                if pos_mask.any():
                    conn_pos_logit_sum[dir_idx] += float(dir_logit[pos_mask].double().sum().item())
                if neg_mask.any():
                    conn_neg_logit_sum[dir_idx] += float(dir_logit[neg_mask].double().sum().item())
            symmetry_errors.append(reciprocal_symmetry_error(c_prob.detach().cpu()))

            d_stats = direction_stats(d_logits, d_gt, skeleton_dir.float())
            dir_stats_acc["pred_hist"] += d_stats["pred_hist"]
            dir_stats_acc["gt_hist"] += d_stats["gt_hist"]
            dir_stats_acc["count"] += d_stats["count"]
            dir_stats_acc["angle_sum"] += d_stats["angle_mean"] * d_stats["count"]
            dir_stats_acc["angle_median_values"].append(d_stats["angle_median"])
            dir_stats_acc["accuracy_count"] += d_stats["accuracy_axial4"] * d_stats["count"]

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
    per_dir_positive_ratio = conn_pos_count / np.maximum(conn_pos_count + conn_neg_count, 1.0)
    per_dir_pos_logit_mean = conn_pos_logit_sum / np.maximum(conn_pos_count, 1.0)
    per_dir_neg_logit_mean = conn_neg_logit_sum / np.maximum(conn_neg_count, 1.0)
    per_dir_auc = np.full(8, np.nan, dtype=np.float64)
    per_dir_auprc = np.full(8, np.nan, dtype=np.float64)
    for dir_idx in range(8):
        if not per_dir_gt_rows[dir_idx]:
            continue
        dir_gt = np.concatenate(per_dir_gt_rows[dir_idx], axis=0)
        dir_prob = np.concatenate(per_dir_prob_rows[dir_idx], axis=0)
        if np.unique(dir_gt).size > 1:
            per_dir_auc[dir_idx] = float(roc_auc_score(dir_gt, dir_prob))
            per_dir_auprc[dir_idx] = float(average_precision_score(dir_gt, dir_prob))
    reciprocal_error = float(np.mean(symmetry_errors))
    surface_cldice = float(np.mean(surface_cldice_values)) if surface_cldice_values else float("nan")

    def mean_row(rows, key):
        return float(np.mean([row[key] for row in rows])) if rows else float("nan")

    surface_frag = {
        "pred_components": mean_row(surface_fragment_rows, "pred_components"),
        "gt_components": mean_row(surface_fragment_rows, "gt_components"),
        "fragmentation_index": mean_row(surface_fragment_rows, "fragmentation_index"),
        "extra_components": mean_row(surface_fragment_rows, "extra_components"),
        "pred_short_components": mean_row(surface_fragment_rows, "pred_short_components"),
        "gt_short_components": mean_row(surface_fragment_rows, "gt_short_components"),
        "pred_largest_ratio": mean_row(surface_fragment_rows, "pred_largest_ratio"),
        "gt_largest_ratio": mean_row(surface_fragment_rows, "gt_largest_ratio"),
    }
    graph_frag = {
        "pred_components": mean_row(graph_fragment_rows, "pred_components"),
        "gt_components": mean_row(graph_fragment_rows, "gt_components"),
        "fragmentation_index": mean_row(graph_fragment_rows, "fragmentation_index"),
        "extra_components": mean_row(graph_fragment_rows, "extra_components"),
    }
    sweep_rows = []
    start = float(args.threshold_sweep_start)
    end = float(args.threshold_sweep_end)
    step = float(args.threshold_sweep_step)
    if step > 0 and end >= start:
        for threshold in np.arange(start, end + 0.5 * step, step):
            pred = flat_conn_prob >= threshold
            gt = flat_conn_gt.astype(bool)
            tp = float(np.logical_and(pred, gt).sum())
            fp = float(np.logical_and(pred, ~gt).sum())
            fn = float(np.logical_and(~pred, gt).sum())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
            sweep_rows.append((float(threshold), precision, recall, f1, tp, fp, fn))
    best_sweep = max(sweep_rows, key=lambda row: row[3]) if sweep_rows else None

    print(f"\nCheckpoint: {args.model_path}")
    print(f"Stage: {stage_key}")
    print(f"connectivity_ablation={args.connectivity_ablation}")
    print(f"split={args.split}, batches={batches_processed}, images={images_processed}")
    print(f"threshold={args.threshold}, surface_threshold={args.surface_threshold}, seed={args.seed}")
    print("\nConnectivity")
    print(f"  8-dir macro Precision: {macro_precision:.4f}")
    print(f"  8-dir macro Recall:    {macro_recall:.4f}")
    print(f"  8-dir macro F1:        {macro_f1:.4f}")
    print(f"  GT connection=1 mean C_prob: {positive_mean:.4f}")
    print(f"  GT connection=0 mean C_prob: {negative_mean:.4f}")
    print(f"  AUROC: {overall_auc:.4f}")
    print(f"  AUPRC: {overall_auprc:.4f}")
    print(f"  Reciprocal symmetry error: {reciprocal_error:.4f}")
    if best_sweep is not None:
        threshold, precision, recall, f1, tp, fp, fn = best_sweep
        print(
            "  Best threshold sweep: "
            f"thr={threshold:.3f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
        )
    print("  Per-direction F1:")
    for name, value in zip(CONNECTIVITY_DIR_NAMES, per_dir_f1):
        print(f"    {name}: {value:.4f}")
    print("  Per-direction train/valid connectivity stats:")
    print(
        "    dir positive_count negative_count positive_ratio "
        "AUROC AUPRC mean_positive_logit mean_negative_logit"
    )
    for idx, name in enumerate(CONNECTIVITY_DIR_NAMES):
        print(
            f"    {name} "
            f"{int(conn_pos_count[idx])} "
            f"{int(conn_neg_count[idx])} "
            f"{per_dir_positive_ratio[idx]:.6f} "
            f"{per_dir_auc[idx]:.4f} "
            f"{per_dir_auprc[idx]:.4f} "
            f"{per_dir_pos_logit_mean[idx]:.4f} "
            f"{per_dir_neg_logit_mean[idx]:.4f}"
        )

    print("\nTopology")
    print(f"  Surface clDice: {surface_cldice:.4f}")
    print(
        "  Surface fragmentation: "
        f"pred_comp={surface_frag['pred_components']:.2f}, "
        f"gt_comp={surface_frag['gt_components']:.2f}, "
        f"frag_idx={surface_frag['fragmentation_index']:.3f}, "
        f"extra_comp={surface_frag['extra_components']:.2f}"
    )
    print(
        "  Surface short components: "
        f"pred={surface_frag['pred_short_components']:.2f}, "
        f"gt={surface_frag['gt_short_components']:.2f}, "
        f"short_area<{args.fragment_short_area}"
    )
    print(
        "  Surface largest-component ratio: "
        f"pred={surface_frag['pred_largest_ratio']:.3f}, "
        f"gt={surface_frag['gt_largest_ratio']:.3f}"
    )
    print(
        "  Connectivity graph fragmentation: "
        f"pred_comp={graph_frag['pred_components']:.2f}, "
        f"gt_comp={graph_frag['gt_components']:.2f}, "
        f"frag_idx={graph_frag['fragmentation_index']:.3f}, "
        f"extra_comp={graph_frag['extra_components']:.2f}"
    )

    print("\nDirection")
    accuracy_axial4 = dir_stats_acc["accuracy_count"] / max(dir_stats_acc["count"], 1)
    angle_mean = dir_stats_acc["angle_sum"] / max(dir_stats_acc["count"], 1)
    angle_median = float(np.median(np.array(dir_stats_acc["angle_median_values"], dtype=np.float32)))
    print(f"  Axial-4 accuracy:      {accuracy_axial4:.4f}")
    print(f"  Mean axial error:      {angle_mean:.4f}")
    print(f"  Median axial error:    {angle_median:.4f}")
    print(f"  Pred axis histogram: {dict(zip(AXIAL_DIR_NAMES, dir_stats_acc['pred_hist'].tolist()))}")
    print(f"  GT axis histogram:   {dict(zip(AXIAL_DIR_NAMES, dir_stats_acc['gt_hist'].tolist()))}")


if __name__ == "__main__":
    main()
