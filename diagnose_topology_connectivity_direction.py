import argparse
import csv
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss, build_connectivity_target
from networks.vision_transformer import SwinUnet as ViT_seg
from topology_direction_constants import CONNECTIVITY_DIR_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose edge discrimination, global topology attention bias, and direction quality."
    )
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--angle_threshold", type=float, default=22.5)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument(
        "--structure_profile",
        default="stage23_boundary_0626_final_ske",
        choices=["full", "stage23_boundary_0626", "stage23_boundary_0626_final_ske"],
    )
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument(
        "--highres_structure_fuse_stages",
        default="stage23",
        choices=["stage2", "stage3", "stage23"],
    )
    parser.add_argument(
        "--highres_structure_fusion_mode",
        default="stage23",
        choices=[
            "stage23",
            "final_correction",
            "stage23_final_correction",
            "post_refine_interaction",
            "none",
        ],
    )
    parser.add_argument("--enable_global_topology", action="store_true")
    parser.add_argument("--global_topology_max_nodes", type=int, default=32)
    parser.add_argument("--global_topology_heads", type=int, default=4)
    parser.add_argument("--global_topology_alpha_max", type=float, default=0.05)

    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--dataset", default="ImageData")
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def match_size(tensor, spatial_size, mode="nearest"):
    if tensor is None or tensor.shape[-2:] == spatial_size:
        return tensor
    return F.interpolate(tensor.float(), size=spatial_size, mode=mode)


def get_last_stage_item(outputs):
    if not isinstance(outputs, tuple) or len(outputs) < 5:
        raise RuntimeError("Expected structure-guided tuple output with stage_outputs.")
    stage_outputs = outputs[4]
    if stage_outputs:
        for item in reversed(stage_outputs):
            if (
                isinstance(item, dict)
                and item.get("connectivity") is not None
                and item.get("direction") is not None
            ):
                return item, "stage3"
    return {"connectivity": outputs[3], "direction": None}, "final_fallback"


def cat_or_empty(values):
    return torch.cat(values) if values else torch.empty(0)


def mean_or_zero(values):
    return float(values.mean().item()) if values.numel() else 0.0


def auroc_from_scores(pos_scores, neg_scores):
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    scores = torch.cat([pos_scores, neg_scores]).float()
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).float()
    order = torch.argsort(scores, descending=False)
    sorted_labels = labels[order]
    ranks = torch.arange(1, labels.numel() + 1, device=labels.device, dtype=torch.float32)
    pos_ranks = ranks[sorted_labels > 0.5].sum()
    n_pos = float(pos_scores.numel())
    n_neg = float(neg_scores.numel())
    return float(((pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0)).item())


def auprc_from_scores(pos_scores, neg_scores):
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    scores = torch.cat([pos_scores, neg_scores]).float()
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).float()
    order = torch.argsort(scores, descending=True)
    labels = labels[order]
    tp = torch.cumsum(labels, dim=0)
    fp = torch.cumsum(1.0 - labels, dim=0)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / labels.sum().clamp_min(1.0)
    recall_prev = torch.cat([recall.new_zeros(1), recall[:-1]])
    return float((precision * (recall - recall_prev)).sum().item())


def q(values, percentile):
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), percentile).item())


def build_model(args, device):
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_global_topology=args.enable_global_topology,
        global_topology_max_nodes=args.global_topology_max_nodes,
        global_topology_heads=args.global_topology_heads,
        global_topology_alpha_max=args.global_topology_alpha_max,
    ).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.model_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def enable_global_diagnostics(model):
    module = model.module if hasattr(model, "module") else model
    global_topology = getattr(module.swin_unet, "global_topology", None)
    if global_topology is not None:
        global_topology.capture_diagnostics = True
    return global_topology


def update_direction_metrics(
    metrics,
    direction_logits,
    skeleton,
    helper,
    angle_threshold,
):
    if direction_logits is None:
        return
    spatial_size = direction_logits.shape[-2:]
    skeleton_stage = match_size(skeleton, spatial_size, mode="nearest")
    direction_target, direction_valid = helper.build_direction_target(skeleton_stage)
    direction_target = direction_target.to(
        device=direction_logits.device,
        dtype=direction_logits.dtype,
    )
    direction_valid = direction_valid.to(
        device=direction_logits.device,
        dtype=direction_logits.dtype,
    )
    connectivity_gt = build_connectivity_target(skeleton_stage).to(
        device=direction_logits.device,
        dtype=direction_logits.dtype,
    )
    neighbor_count = connectivity_gt.sum(dim=1, keepdim=True)
    direction_pred = F.normalize(direction_logits, dim=1, eps=1e-6)
    cosine = (direction_pred * direction_target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    angle_error = torch.rad2deg(torch.acos(cosine)) * 0.5
    valid = direction_valid > 0.5
    groups = {
        "n1": neighbor_count == 1,
        "n2": neighbor_count == 2,
        "n3": neighbor_count == 3,
        "n4plus": neighbor_count >= 4,
    }
    for name, group_mask in groups.items():
        mask = valid & group_mask
        count = int(mask.sum().item())
        if count == 0:
            continue
        err = angle_error[mask]
        metrics[f"direction_{name}_count"] += count
        metrics[f"direction_{name}_error_sum"] += float(err.sum().item())
        metrics[f"direction_{name}_correct"] += int((err <= angle_threshold).sum().item())


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(args, device)
    global_topology = enable_global_diagnostics(model)
    helper = SurfaceStructureLoss()

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
    )

    pos_logits_all = []
    hard_neg_logits_all = []
    pos_probs_all = []
    hard_neg_probs_all = []
    pair_logit_gaps_all = []
    straight_pos_logits = []
    straight_hard_neg_logits = []
    junction_pos_logits = []
    junction_hard_neg_logits = []
    pos_direction_hist = torch.zeros(8, dtype=torch.long)
    hard_neg_direction_hist = torch.zeros(8, dtype=torch.long)

    topology_stats = {
        "token_qk_std": [],
        "token_btopo_std": [],
        "token_btopo_qk_std_ratio": [],
        "attention_conn_corr": [],
        "attention_dir_corr": [],
    }
    direction_metrics = {}
    for group in ("n1", "n2", "n3", "n4plus"):
        direction_metrics[f"direction_{group}_count"] = 0
        direction_metrics[f"direction_{group}_error_sum"] = 0.0
        direction_metrics[f"direction_{group}_correct"] = 0

    source_name = "unknown"
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="diagnostics")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device)
            skeleton = batch["skeleton"].to(device)

            outputs = model(images, teacher_forcing_ratio=0.0)
            stage_item, source_name = get_last_stage_item(outputs)
            connectivity_logits = stage_item["connectivity"]
            direction_logits = stage_item.get("direction")

            spatial_size = connectivity_logits.shape[-2:]
            skeleton_stage = match_size(skeleton, spatial_size, mode="nearest")
            connectivity_gt = build_connectivity_target(skeleton_stage).to(
                device=device,
                dtype=connectivity_logits.dtype,
            )
            center_corridor = (skeleton_stage > 0.5).expand_as(connectivity_gt)
            valid = helper._connectivity_boundary_mask(connectivity_logits, None)
            valid = valid * center_corridor.to(dtype=valid.dtype)
            connectivity_prob = torch.sigmoid(connectivity_logits)
            neighbor_count = connectivity_gt.sum(dim=1, keepdim=True).expand_as(connectivity_gt)

            for sample_index in range(connectivity_logits.shape[0]):
                pos_mask = (
                    (connectivity_gt[sample_index] > 0.5)
                    & (valid[sample_index] > 0.5)
                )
                neg_mask = (
                    (connectivity_gt[sample_index] <= 0.5)
                    & (valid[sample_index] > 0.5)
                )
                pos_logits = connectivity_logits[sample_index][pos_mask]
                neg_logits = connectivity_logits[sample_index][neg_mask]
                pos_probs = connectivity_prob[sample_index][pos_mask]
                neg_probs = connectivity_prob[sample_index][neg_mask]
                if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                    continue
                hard_count = min(int(pos_logits.numel()), int(neg_logits.numel()))
                pos_topk = pos_logits.topk(k=hard_count, largest=False)
                hard_neg_topk = neg_logits.topk(k=hard_count, largest=True)
                pos_pair = pos_topk.values
                hard_neg_logits = hard_neg_topk.values
                pos_prob_pair = pos_probs[pos_topk.indices]
                hard_neg_probs = neg_probs[hard_neg_topk.indices]

                pos_logits_all.append(pos_pair.detach().cpu())
                hard_neg_logits_all.append(hard_neg_logits.detach().cpu())
                pos_probs_all.append(pos_prob_pair.detach().cpu())
                hard_neg_probs_all.append(hard_neg_probs.detach().cpu())
                pair_logit_gaps_all.append((pos_pair - hard_neg_logits).detach().cpu())

                pos_channel = (
                    pos_mask.nonzero(as_tuple=False)[pos_topk.indices, 0]
                    .detach()
                    .cpu()
                )
                neg_channel = (
                    neg_mask.nonzero(as_tuple=False)[hard_neg_topk.indices, 0]
                    .detach()
                    .cpu()
                )
                for channel in range(8):
                    pos_direction_hist[channel] += int((pos_channel == channel).sum().item())
                    hard_neg_direction_hist[channel] += int((neg_channel == channel).sum().item())

                neighbor_i = neighbor_count[sample_index]
                straight_mask = neighbor_i <= 2
                junction_mask = neighbor_i >= 3
                sp = connectivity_logits[sample_index][pos_mask & straight_mask]
                sn = connectivity_logits[sample_index][neg_mask & straight_mask]
                jp = connectivity_logits[sample_index][pos_mask & junction_mask]
                jn = connectivity_logits[sample_index][neg_mask & junction_mask]
                if sp.numel() and sn.numel():
                    k = min(int(sp.numel()), int(sn.numel()))
                    straight_pos_logits.append(sp.topk(k=k, largest=False).values.detach().cpu())
                    straight_hard_neg_logits.append(sn.topk(k=k, largest=True).values.detach().cpu())
                if jp.numel() and jn.numel():
                    k = min(int(jp.numel()), int(jn.numel()))
                    junction_pos_logits.append(jp.topk(k=k, largest=False).values.detach().cpu())
                    junction_hard_neg_logits.append(jn.topk(k=k, largest=True).values.detach().cpu())

            update_direction_metrics(
                direction_metrics,
                direction_logits,
                skeleton,
                helper,
                args.angle_threshold,
            )

            if global_topology is not None and global_topology.last_diagnostics:
                diagnostics = global_topology.last_diagnostics
                for key in topology_stats:
                    value = diagnostics.get(key)
                    if value is not None:
                        topology_stats[key].append(float(value.detach().float().mean().cpu().item()))

    pos_logits = cat_or_empty(pos_logits_all)
    hard_neg_logits = cat_or_empty(hard_neg_logits_all)
    pos_probs = cat_or_empty(pos_probs_all)
    hard_neg_probs = cat_or_empty(hard_neg_probs_all)
    pair_logit_gaps = cat_or_empty(pair_logit_gaps_all)
    straight_pos = cat_or_empty(straight_pos_logits)
    straight_neg = cat_or_empty(straight_hard_neg_logits)
    junction_pos = cat_or_empty(junction_pos_logits)
    junction_neg = cat_or_empty(junction_hard_neg_logits)

    row = {
        "split": args.split,
        "source": source_name,
        "margin": args.margin,
        "C_pos_mean": mean_or_zero(pos_probs),
        "C_hard_neg_mean": mean_or_zero(hard_neg_probs),
        "AUROC": auroc_from_scores(pos_probs, hard_neg_probs),
        "AUPRC": auprc_from_scores(pos_probs, hard_neg_probs),
        "positive_logit_mean": mean_or_zero(pos_logits),
        "hard_negative_logit_mean": mean_or_zero(hard_neg_logits),
        "logit_gap_mean": mean_or_zero(pair_logit_gaps),
        "P_zpos_gt_zneg": (
            float((pair_logit_gaps > 0).float().mean().item())
            if pair_logit_gaps.numel()
            else 0.0
        ),
        "margin_violation_rate": (
            float((pair_logit_gaps < args.margin).float().mean().item())
            if pair_logit_gaps.numel()
            else 0.0
        ),
        "gap_p10": q(pair_logit_gaps, 0.10),
        "gap_p25": q(pair_logit_gaps, 0.25),
        "gap_p50": q(pair_logit_gaps, 0.50),
        "gap_p75": q(pair_logit_gaps, 0.75),
        "gap_p90": q(pair_logit_gaps, 0.90),
        "straight_logit_gap": (
            mean_or_zero(straight_pos[: min(straight_pos.numel(), straight_neg.numel())]
            - straight_neg[: min(straight_pos.numel(), straight_neg.numel())])
            if straight_pos.numel() and straight_neg.numel()
            else 0.0
        ),
        "junction_logit_gap": (
            mean_or_zero(junction_pos[: min(junction_pos.numel(), junction_neg.numel())]
            - junction_neg[: min(junction_pos.numel(), junction_neg.numel())])
            if junction_pos.numel() and junction_neg.numel()
            else 0.0
        ),
    }
    for key, values in topology_stats.items():
        row[key] = float(sum(values) / len(values)) if values else 0.0
    for group in ("n1", "n2", "n3", "n4plus"):
        count = direction_metrics[f"direction_{group}_count"]
        row[f"direction_{group}_count"] = count
        row[f"direction_{group}_accuracy"] = (
            direction_metrics[f"direction_{group}_correct"] / max(count, 1)
        )
        row[f"direction_{group}_angle_error"] = (
            direction_metrics[f"direction_{group}_error_sum"] / max(count, 1)
        )
    for index, name in enumerate(CONNECTIVITY_DIR_NAMES):
        row[f"pos_dir_{name}"] = int(pos_direction_hist[index].item())
        row[f"hard_neg_dir_{name}"] = int(hard_neg_direction_hist[index].item())

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print("\nA. Connectivity diagnostic")
    for key in (
        "C_pos_mean",
        "C_hard_neg_mean",
        "AUROC",
        "AUPRC",
        "positive_logit_mean",
        "hard_negative_logit_mean",
        "logit_gap_mean",
        "P_zpos_gt_zneg",
        "margin_violation_rate",
        "straight_logit_gap",
        "junction_logit_gap",
    ):
        print(f"{key}: {row[key]}")
    print("\nB. Global topology diagnostic")
    for key in (
        "token_qk_std",
        "token_btopo_std",
        "token_btopo_qk_std_ratio",
        "attention_conn_corr",
        "attention_dir_corr",
    ):
        print(f"{key}: {row[key]}")
    print("\nC. Direction diagnostic")
    for group in ("n1", "n2", "n3", "n4plus"):
        print(
            f"{group}: count={row[f'direction_{group}_count']} "
            f"acc={row[f'direction_{group}_accuracy']:.4f} "
            f"angle_error={row[f'direction_{group}_angle_error']:.4f}"
        )
    print("\nDirection histogram")
    for name in CONNECTIVITY_DIR_NAMES:
        print(f"{name}: pos={row[f'pos_dir_{name}']} hard_neg={row[f'hard_neg_dir_{name}']}")
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
