"""Measure whether the new connectivity pair features affect candidate scores."""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_connectivity_target, build_stage_skeleton_target, SurfaceStructureLoss


def parse_args():
    p = argparse.ArgumentParser(description="Connectivity pair-feature ablation diagnostic")
    p.add_argument("--model_path", required=True)
    p.add_argument("--root_path", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--source_patch_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=20)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--structure_profile", default="stage23_boundary_0626")
    p.add_argument("--bottleneck_type", default="global_local")
    p.add_argument("--disable_msfe_skip", action="store_true")
    p.add_argument("--enable_highres_structure_stream", action="store_true")
    p.add_argument("--highres_structure_channels", type=int, default=64)
    p.add_argument("--highres_structure_fuse_stages", default="stage23")
    p.add_argument("--highres_structure_fusion_mode", default="stage23")
    p.add_argument("--enable_global_topology", action="store_true")
    p.add_argument("--global_topology_max_nodes", type=int, default=32)
    p.add_argument("--global_topology_heads", type=int, default=4)
    p.add_argument("--global_topology_alpha_max", type=float, default=0.05)
    p.add_argument("--final_topology_eta_init", type=float, default=0.0)
    p.add_argument("--final_gap_rho_init", type=float, default=0.0)
    p.add_argument("--stage_topology_stages", default="none")
    p.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    p.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    p.add_argument("--stage_topology_bias_mode", default="pairwise_skeleton")
    p.add_argument("--stage_topology_ratio", type=float, default=0.08)
    p.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    p.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    p.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    p.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    p.add_argument("--model_impl", default="auto", choices=["auto", "standard", "selective"])
    p.add_argument("--dataset", default="ImageData")
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--n_class", type=int, default=2)
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    p.add_argument("--zip", action="store_true")
    p.add_argument("--cache_mode", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--accumulation_steps", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--amp_opt_level", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--throughput", action="store_true")
    return p.parse_args()


def score_with_parts(head, feature, neighbor, prior, use_delta=True, use_product=True):
    b, k, c, h, w = neighbor.shape
    anchor = feature.unsqueeze(1).expand(-1, k, -1, -1, -1)
    delta = neighbor - anchor if use_delta else torch.zeros_like(anchor)
    product = neighbor * anchor if use_product else torch.zeros_like(anchor)
    edge = torch.cat([anchor, neighbor, delta, product, prior], dim=2)
    edge = edge.reshape(b * k, edge.shape[2], h, w)
    logits = F.conv2d(edge, head.edge_linear.weight, head.edge_linear.bias)
    return logits.reshape(b, k, h, w)


def summarize(values, labels):
    values = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    labels = np.concatenate(labels) if labels else np.empty(0, dtype=np.int32)
    pos = values[labels == 1]
    neg = values[labels == 0]
    if values.size and np.unique(labels).size == 2:
        auroc = float(roc_auc_score(labels, values))
        auprc = float(average_precision_score(labels, values))
    else:
        auroc = float("nan")
        auprc = float("nan")
    if pos.size and neg.size:
        # Compute the Mann-Whitney probability without materializing
        # the full positive-by-negative comparison matrix.
        neg_sorted = np.sort(neg)
        p_positive_gt_negative = float(
            np.searchsorted(neg_sorted, pos, side="left").mean() / neg.size
        )
    else:
        p_positive_gt_negative = float("nan")
    return {
        "positive_count": int(pos.size),
        "negative_count": int(neg.size),
        "positive_mean": float(pos.mean()) if pos.size else float("nan"),
        "negative_mean": float(neg.mean()) if neg.size else float("nan"),
        "gap": float(pos.mean() - neg.mean()) if pos.size and neg.size else float("nan"),
        "p_positive_gt_negative": p_positive_gt_negative,
        "auroc": auroc,
        "auprc": auprc,
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    head = model.swin_unet.decoder_structure_blocks[3].connectivity_head
    if not hasattr(head, "edge_linear"):
        raise RuntimeError("Checkpoint/model does not use the new edge_linear pair head.")
    head.collect_pair_diagnostics = True
    helper = SurfaceStructureLoss()
    loader = DataLoader(
        RoadSkeletonDataset(
            root_dir=args.root_path,
            split=args.split,
            image_size=args.img_size,
            source_patch_size=args.source_patch_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    modes = ["full", "no_delta", "no_product", "no_pair", "shuffled_neighbor"]
    score_chunks = {name: [] for name in modes}
    label_chunks = {name: [] for name in modes}
    candidate_stds = {name: [] for name in modes}
    delta_norms = []
    product_norms = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="pair feature diagnostic")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            skeleton = batch["skeleton"].to(device).float()
            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            diag = head.last_pair_diagnostics
            if diag is None:
                raise RuntimeError("Stage3 connectivity head was not called in this forward.")
            feature = diag["feature"]
            neighbor = diag["neighbor"]
            prior = diag["prior"]
            full = diag["logits"].reshape_as(diag["logits"])
            target_skeleton = build_stage_skeleton_target(skeleton, full.shape[-2:]).to(device)
            target = build_connectivity_target(target_skeleton).float()
            valid = helper._connectivity_boundary_mask(full, None)
            valid = valid * target_skeleton.expand_as(full)
            labels = (target > 0.5).float()
            delta_norms.append((neighbor - feature.unsqueeze(1)).flatten(2).norm(dim=2).mean().item())
            product_norms.append((neighbor * feature.unsqueeze(1)).flatten(2).norm(dim=2).mean().item())

            scores = {
                "full": full,
                "no_delta": score_with_parts(head, feature, neighbor, prior, False, True),
                "no_product": score_with_parts(head, feature, neighbor, prior, True, False),
                "no_pair": score_with_parts(head, feature, neighbor, prior, False, False),
                "shuffled_neighbor": score_with_parts(
                    head, feature, neighbor.roll(1, dims=1), prior, True, True
                ),
            }
            for name, score in scores.items():
                mask = valid > 0.5
                score_chunks[name].append(score[mask].float().cpu().numpy())
                label_chunks[name].append(labels[mask].cpu().numpy().astype(np.int32))
                count = mask.sum(dim=1)
                mean = (score * valid).sum(dim=1) / count.clamp_min(1.0)
                var = ((score - mean.unsqueeze(1)).square() * valid).sum(dim=1) / count.clamp_min(1.0)
                candidate_stds[name].append(var.sqrt()[count > 1].cpu().numpy())

    rows = []
    for name in modes:
        row = {"mode": name, **summarize(score_chunks[name], label_chunks[name])}
        std = np.concatenate(candidate_stds[name]) if candidate_stds[name] else np.empty(0)
        row.update({
            "candidate_std_mean": float(std.mean()) if std.size else float("nan"),
            "candidate_std_median": float(np.median(std)) if std.size else float("nan"),
            "candidate_std_lt_005": float((std < 0.05).mean()) if std.size else float("nan"),
        })
        rows.append(row)
        print(
            f"{name:20s} gap={row['gap']:.6f} P+>P-={row['p_positive_gt_negative']:.4f} "
            f"AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} "
            f"candidate_std={row['candidate_std_mean']:.6f}"
        )
    print(f"pair delta norm={np.mean(delta_norms):.6f}; product norm={np.mean(product_norms):.6f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
