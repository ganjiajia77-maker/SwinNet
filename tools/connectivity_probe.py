import argparse
import csv
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE
from networks.vision_transformer_selective_fusion import SwinUnet as ViT_seg


DIR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def make_config_args(saved_args, overrides):
    values = dict(saved_args or {})
    defaults = {
        "cfg": "./configs/swin_tiny_patch4_window7_224_lite.yaml",
        "batch_size": 1,
        "img_size": 256,
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
    defaults.update(values)
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**defaults)


def build_model(config, args, device):
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=getattr(args, "num_classes", 2),
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=getattr(args, "bottleneck_type", "global_local"),
        final_topology_eta_init=getattr(args, "final_topology_eta_init", 0.0),
        final_gap_rho_init=getattr(args, "final_gap_rho_init", 0.0),
        stage_topology_stages=getattr(args, "stage_topology_stages", "none"),
        stage_topology_alpha_max=getattr(args, "stage_topology_alpha_max", 1.0),
        stage_topology_alpha_init=getattr(args, "stage_topology_alpha_init", 0.1),
        stage_topology_bias_mode=getattr(args, "stage_topology_bias_mode", "pairwise_skeleton"),
        stage_topology_ratio=getattr(args, "stage_topology_ratio", 0.08),
        stage_topology_topo_clip=getattr(args, "stage_topology_topo_clip", 4.0),
        structure_profile=getattr(args, "structure_profile", "full"),
        enable_final_graph_prop=getattr(args, "enable_graph_prop", False),
        use_msfe_skip=not getattr(args, "disable_msfe_skip", False),
        stage2_skeleton_gradient_ratio=getattr(args, "stage2_skeleton_gradient_ratio", 0.5),
        stage3_skeleton_gradient_ratio=getattr(args, "stage3_skeleton_gradient_ratio", 0.5),
        final_skeleton_gradient_ratio=getattr(args, "final_skeleton_gradient_ratio", 0.0),
        enable_highres_structure_stream=getattr(args, "enable_highres_structure_stream", False),
        highres_structure_channels=getattr(args, "highres_structure_channels", 64),
        highres_structure_fuse_stages=getattr(args, "highres_structure_fuse_stages", "stage23"),
        highres_structure_fusion_mode=getattr(args, "highres_structure_fusion_mode", "stage23"),
        enable_post_refine_structure_interaction=getattr(
            args,
            "enable_post_refine_structure_interaction",
            False,
        ),
    ).to(device)
    return model


def shift_map(x, dy, dx):
    _, _, height, width = x.shape
    pad_left = max(-dx, 0)
    pad_right = max(dx, 0)
    pad_top = max(-dy, 0)
    pad_bottom = max(dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    y0 = max(dy, 0)
    x0 = max(dx, 0)
    return padded[:, :, y0:y0 + height, x0:x0 + width]


def boundary_valid_like(connectivity):
    batch, _, height, width = connectivity.shape
    valid = torch.ones((batch, 8, height, width), device=connectivity.device, dtype=torch.bool)
    for idx, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        if dy < 0:
            valid[:, idx, 0, :] = False
        if dy > 0:
            valid[:, idx, -1, :] = False
        if dx < 0:
            valid[:, idx, :, 0] = False
        if dx > 0:
            valid[:, idx, :, -1] = False
    return valid


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summarize_probe(scores, labels, symmetry_errors):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    if scores.size == 0:
        return None
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    pred = scores >= 0.5
    tp = float(np.logical_and(pred, labels == 1).sum())
    fp = float(np.logical_and(pred, labels == 0).sum())
    fn = float(np.logical_and(~pred, labels == 1).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n_edges": int(scores.size),
        "n_pos": int(labels.sum()),
        "n_neg": int(scores.size - labels.sum()),
        "positive_ratio": float(labels.mean()),
        "auroc": rank_auc(scores, labels),
        "pos_prob_mean": float(pos_scores.mean()) if pos_scores.size else float("nan"),
        "neg_prob_mean": float(neg_scores.mean()) if neg_scores.size else float("nan"),
        "prob_gap": (
            float(pos_scores.mean() - neg_scores.mean())
            if pos_scores.size and neg_scores.size
            else float("nan")
        ),
        "pos_prob_p10": float(np.quantile(pos_scores, 0.10)) if pos_scores.size else float("nan"),
        "pos_prob_p50": float(np.quantile(pos_scores, 0.50)) if pos_scores.size else float("nan"),
        "pos_prob_p90": float(np.quantile(pos_scores, 0.90)) if pos_scores.size else float("nan"),
        "neg_prob_p10": float(np.quantile(neg_scores, 0.10)) if neg_scores.size else float("nan"),
        "neg_prob_p50": float(np.quantile(neg_scores, 0.50)) if neg_scores.size else float("nan"),
        "neg_prob_p90": float(np.quantile(neg_scores, 0.90)) if neg_scores.size else float("nan"),
        "brier": float(np.mean((scores - labels) ** 2)),
        "probe_precision_at_0_5": precision,
        "probe_recall_at_0_5": recall,
        "probe_f1_at_0_5": f1,
        "symmetry_error": float(np.mean(symmetry_errors)) if symmetry_errors else float("nan"),
    }


def collect_head(store, name, logits, gt, skeleton_mask):
    if logits is None:
        return
    if logits.shape[-2:] != gt.shape[-2:]:
        gt = F.interpolate(gt.float(), size=logits.shape[-2:], mode="nearest")
        skeleton_mask = F.interpolate(skeleton_mask.float(), size=logits.shape[-2:], mode="nearest") > 0.5
    prob = torch.sigmoid(logits)
    valid = boundary_valid_like(prob) & skeleton_mask.expand_as(prob)
    selected_scores = prob[valid].detach().cpu().numpy()
    selected_labels = (gt[valid] > 0.5).detach().cpu().numpy().astype(np.uint8)
    store.setdefault(name, {"scores": [], "labels": [], "symmetry": []})
    store[name]["scores"].append(selected_scores)
    store[name]["labels"].append(selected_labels)

    for idx, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        forward = prob[:, idx:idx + 1]
        backward = shift_map(prob[:, CONNECTIVITY_OPPOSITE[idx]:CONNECTIVITY_OPPOSITE[idx] + 1], dy, dx)
        sym_valid = valid[:, idx:idx + 1]
        if sym_valid.any():
            err = torch.abs(forward - backward)[sym_valid].mean().detach().cpu().item()
            store[name]["symmetry"].append(float(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    config_args = make_config_args(
        saved_args,
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "resume": "",
        },
    )
    config = get_config(config_args)
    device = torch.device(args.device)
    model = build_model(config, config_args, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    dataset = RoadSkeletonDataset(
        root_dir=getattr(config_args, "root_path", "./data1"),
        split=args.split,
        image_size=getattr(config_args, "img_size", 256),
        source_patch_size=getattr(config_args, "source_patch_size", 1024),
        tile_size=None,
        tile_stride=getattr(config_args, "overlap_stride", 256),
        augment=False,
        return_full_image=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or getattr(config_args, "batch_size", 1),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    store = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            skeleton = batch["skeleton"].to(device) > 0.5
            connectivity_gt = batch["connectivity_gt"].to(device).float()
            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            final_connectivity = outputs[3] if len(outputs) > 3 else None
            stage_outputs = outputs[4] if len(outputs) > 4 else []
            collect_head(store, "final", final_connectivity, connectivity_gt, skeleton)
            for stage_output in stage_outputs:
                conn = stage_output.get("connectivity") if isinstance(stage_output, dict) else None
                if conn is None:
                    continue
                stage_name = "stage{}".format(stage_output.get("stage", "unknown"))
                collect_head(store, stage_name, conn, connectivity_gt, skeleton)

    rows = []
    for name, values in sorted(store.items()):
        scores = np.concatenate(values["scores"]) if values["scores"] else np.array([])
        labels = np.concatenate(values["labels"]) if values["labels"] else np.array([])
        summary = summarize_probe(scores, labels, values["symmetry"])
        if summary is None:
            continue
        summary = {"head": name, **summary}
        rows.append(summary)

    if not rows:
        print("No connectivity logits were found.")
        return 1

    fieldnames = list(rows[0].keys())
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(
            "{head}: AUROC={auroc:.4f}, pos_mean={pos_prob_mean:.4f}, "
            "neg_mean={neg_prob_mean:.4f}, gap={prob_gap:.4f}, "
            "pos_ratio={positive_ratio:.4f}, sym_err={symmetry_error:.4f}, "
            "probe_f1@0.5={probe_f1_at_0_5:.4f}, n={n_edges}".format(**row)
        )
    if args.output_csv:
        print("Wrote {}".format(args.output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
