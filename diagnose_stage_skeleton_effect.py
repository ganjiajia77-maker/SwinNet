"""Evaluate high-res skeleton, stage residual skeleton, and fused Gate input."""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_soft_stage_skeleton_target


def parse_args():
    p = argparse.ArgumentParser(description="Stage skeleton residual/soft-target diagnostic")
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


def metrics(logits, target, thresholds):
    probability = torch.sigmoid(logits).flatten().cpu().numpy()
    hard = (target > 0.5).flatten().cpu().numpy().astype(np.int32)
    rows = []
    for threshold in thresholds:
        pred = probability >= threshold
        tp = np.logical_and(pred, hard == 1).sum()
        fp = np.logical_and(pred, hard == 0).sum()
        fn = np.logical_and(~pred, hard == 1).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
    if np.unique(hard).size == 2:
        auroc = roc_auc_score(hard, probability)
        auprc = average_precision_score(hard, probability)
    else:
        auroc = float("nan")
        auprc = float("nan")
    return rows, auroc, auprc


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
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
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    by_source = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="stage skeleton diagnostic")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            skeleton = batch["skeleton"].to(device).float()
            # The current standard SwinUnet forward accepts only the image tensor.
            # These topology controls belonged to an older forward API.
            outputs = model(images)
            structure_outputs = outputs[-1]
            highres = model.swin_unet.last_highres_structure_skeleton
            if highres is None:
                raise RuntimeError("Checkpoint/model did not produce high-res skeleton logits.")
            for item in structure_outputs:
                stage = item.get("stage")
                if stage not in (2, 3) or item.get("refinement_step") != 1:
                    continue
                fused = item.get("skeleton")
                if fused is None:
                    continue
                prior = F.interpolate(highres, size=fused.shape[-2:], mode="bilinear", align_corners=False)
                hard_target, soft_target = build_soft_stage_skeleton_target(skeleton, fused.shape[-2:])
                sources = {
                    f"stage{stage}_highres_prior": prior,
                    f"stage{stage}_fused": fused,
                }
                for source, logits in sources.items():
                    delta = logits - prior if source.endswith("fused") else torch.zeros_like(logits)
                    key = source
                    state = by_source.setdefault(key, {"logits": [], "target": [], "soft": [], "delta": []})
                    state["logits"].append(logits.detach().cpu())
                    state["target"].append(hard_target.detach().cpu())
                    state["soft"].append(soft_target.detach().cpu())
                    state["delta"].append(delta.detach().abs().mean().item())

    records = []
    for source, state in by_source.items():
        logits = torch.cat(state["logits"])
        target = torch.cat(state["target"])
        soft = torch.cat(state["soft"])
        threshold_rows, auroc, auprc = metrics(logits, target, thresholds)
        best = max(threshold_rows, key=lambda row: row["f1"])
        mean_abs_delta = float(np.mean(state["delta"]))
        for row in threshold_rows:
            records.append({
                "source": source,
                "threshold": row["threshold"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "auroc": auroc,
                "auprc": auprc,
                "mean_abs_delta": mean_abs_delta,
            })
        probability = torch.sigmoid(logits)
        positive = probability[target > 0.5]
        negative = probability[target <= 0.5]
        print(
            f"{source:24s} best_thr={best['threshold']:.3f} F1={best['f1']:.4f} "
            f"AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"pos_mean={positive.mean().item():.6f} neg_mean={negative.mean().item():.6f} "
            f"mean_abs_delta={mean_abs_delta:.6f} soft_mean={soft.mean().item():.6f}"
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
