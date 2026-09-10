import argparse
import csv
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


def match_size(tensor, spatial_size, mode="nearest"):
    if tensor.shape[-2:] == spatial_size:
        return tensor
    return F.interpolate(tensor, size=spatial_size, mode=mode)


def get_stage3_connectivity(outputs):
    if not isinstance(outputs, tuple) or len(outputs) < 5:
        raise RuntimeError("Expected structure-guided tuple output with stage_outputs.")

    connectivity_logits = outputs[3]
    stage_outputs = outputs[4]
    if stage_outputs:
        for item in reversed(stage_outputs):
            if isinstance(item, dict) and item.get("connectivity") is not None:
                return item["connectivity"], "stage3"
    return connectivity_logits, "final_fallback"


def summarize_quantiles(values):
    if values.numel() == 0:
        return {
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    quantiles = torch.quantile(
        values.float(),
        torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=values.device),
    )
    return {
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
    }


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


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether connectivity logits separate GT positive edges from hard negative edges."
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
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(args, device)

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
    helper = SurfaceStructureLoss()

    positive_logits = []
    negative_logits = []
    positive_probs = []
    negative_probs = []
    hard_negative_logits = []
    image_logit_gaps = []
    image_prob_gaps = []
    source_name = "unknown"

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="edge discrimination")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break

            images = batch["image"].to(device)
            skeleton = batch["skeleton"].to(device)

            outputs = model(images, teacher_forcing_ratio=0.0)
            connectivity_logits, source_name = get_stage3_connectivity(outputs)
            spatial_size = connectivity_logits.shape[-2:]

            skeleton_stage = match_size(skeleton, spatial_size, mode="nearest")
            connectivity_gt = build_connectivity_target(skeleton_stage).to(
                device=device,
                dtype=connectivity_logits.dtype,
            )
            valid = helper._connectivity_boundary_mask(connectivity_logits, None)
            center_mask = (skeleton_stage > 0.5).to(dtype=connectivity_logits.dtype)
            valid = valid * center_mask.expand_as(connectivity_logits)
            connectivity_prob = torch.sigmoid(connectivity_logits)

            for sample_index in range(connectivity_logits.shape[0]):
                pos_mask = (
                    (connectivity_gt[sample_index] > 0.5)
                    & (valid[sample_index] > 0.5)
                )
                neg_mask = (
                    (connectivity_gt[sample_index] <= 0.5)
                    & (valid[sample_index] > 0.5)
                )

                pos_logit = connectivity_logits[sample_index][pos_mask]
                neg_logit = connectivity_logits[sample_index][neg_mask]
                pos_prob = connectivity_prob[sample_index][pos_mask]
                neg_prob = connectivity_prob[sample_index][neg_mask]
                if pos_logit.numel() == 0 or neg_logit.numel() == 0:
                    continue

                hard_count = min(pos_logit.numel(), neg_logit.numel())
                hard_neg_logit = neg_logit.topk(k=hard_count, largest=True).values
                hard_neg_prob = neg_prob.topk(k=hard_count, largest=True).values

                positive_logits.append(pos_logit.detach().cpu())
                negative_logits.append(hard_neg_logit.detach().cpu())
                positive_probs.append(pos_prob.detach().cpu())
                negative_probs.append(hard_neg_prob.detach().cpu())
                hard_negative_logits.append(hard_neg_logit.detach().cpu())
                image_logit_gaps.append(
                    (pos_logit.mean() - hard_neg_logit.mean()).detach().reshape(1).cpu()
                )
                image_prob_gaps.append(
                    (pos_prob.mean() - hard_neg_prob.mean()).detach().reshape(1).cpu()
                )

    pos_logits = torch.cat(positive_logits) if positive_logits else torch.empty(0)
    neg_logits = torch.cat(negative_logits) if negative_logits else torch.empty(0)
    pos_probs = torch.cat(positive_probs) if positive_probs else torch.empty(0)
    neg_probs = torch.cat(negative_probs) if negative_probs else torch.empty(0)
    hard_neg_logits = (
        torch.cat(hard_negative_logits) if hard_negative_logits else torch.empty(0)
    )
    image_logit_gaps = (
        torch.cat(image_logit_gaps) if image_logit_gaps else torch.empty(0)
    )
    image_prob_gaps = torch.cat(image_prob_gaps) if image_prob_gaps else torch.empty(0)

    pair_count = min(pos_logits.numel(), neg_logits.numel())
    if pair_count > 0:
        logit_gap = pos_logits[:pair_count] - neg_logits[:pair_count]
        prob_gap = pos_probs[:pair_count] - neg_probs[:pair_count]
        p_positive_greater = float((logit_gap > 0).float().mean().item())
        margin_satisfaction = float((logit_gap >= args.margin).float().mean().item())
        margin_violation = float((logit_gap < args.margin).float().mean().item())
        quantiles = summarize_quantiles(logit_gap)
    else:
        logit_gap = torch.empty(0)
        prob_gap = torch.empty(0)
        p_positive_greater = 0.0
        margin_satisfaction = 0.0
        margin_violation = 0.0
        quantiles = summarize_quantiles(logit_gap)

    row = {
        "source": source_name,
        "split": args.split,
        "margin": args.margin,
        "num_positive": int(pos_logits.numel()),
        "num_hard_negative": int(neg_logits.numel()),
        "positive_logit_mean": float(pos_logits.mean().item()) if pos_logits.numel() else 0.0,
        "negative_logit_mean": float(neg_logits.mean().item()) if neg_logits.numel() else 0.0,
        "logit_gap_mean": float(logit_gap.mean().item()) if logit_gap.numel() else 0.0,
        "positive_prob_mean": float(pos_probs.mean().item()) if pos_probs.numel() else 0.0,
        "negative_prob_mean": float(neg_probs.mean().item()) if neg_probs.numel() else 0.0,
        "prob_gap_mean": float(prob_gap.mean().item()) if prob_gap.numel() else 0.0,
        "P_zpos_gt_zneg": p_positive_greater,
        "P_zpos_gt_zneg_plus_margin": margin_satisfaction,
        "margin_satisfaction_rate": margin_satisfaction,
        "margin_violation_rate": margin_violation,
        "gap_p10": quantiles["p10"],
        "gap_p25": quantiles["p25"],
        "gap_p50": quantiles["p50"],
        "gap_p75": quantiles["p75"],
        "gap_p90": quantiles["p90"],
        "hard_negative_logit_mean": (
            float(hard_neg_logits.mean().item()) if hard_neg_logits.numel() else 0.0
        ),
        "image_level_logit_gap_mean": (
            float(image_logit_gaps.mean().item()) if image_logit_gaps.numel() else 0.0
        ),
        "image_level_prob_gap_mean": (
            float(image_prob_gaps.mean().item()) if image_prob_gaps.numel() else 0.0
        ),
    }

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print("\nEdge discrimination diagnostic")
    for key, value in row.items():
        print(f"{key}: {value}")
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
