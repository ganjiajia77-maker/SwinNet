import argparse
import contextlib
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_connectivity_direction_quality import (
    surface_cldice_score,
    surface_fragmentation_stats,
)
from diagnose_structure_delta_quadrants import (
    FEATURE_NAMES,
    bucket_stats,
    empty_bucket,
    extract_structure_features,
    update_bucket,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare reliability correction ON versus OFF while keeping "
            "structure features, skeleton, connectivity, old gate, and feature "
            "refinement active."
        )
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fragment_short_area", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
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
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--final_topology_eta_init", type=float, default=0.0)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.0)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=("auto", "standard", "selective"))
    return parser.parse_args()


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_swin_unet(model):
    module = model.module if hasattr(model, "module") else model
    return getattr(module, "swin_unet", module)


def reliability_modules(model):
    swin = get_swin_unet(model)
    modules = []
    stage2_source = getattr(swin, "stage2_topology_source", None)
    if stage2_source is not None and hasattr(stage2_source, "reliability_beta"):
        modules.append(("stage2_source", stage2_source))
    for idx, block in enumerate(getattr(swin, "decoder_structure_blocks", [])):
        if hasattr(block, "reliability_beta"):
            modules.append((f"decoder_stage{idx}", block))
    return modules


@contextlib.contextmanager
def reliability_off(model):
    originals = []
    for name, module in reliability_modules(model):
        originals.append((name, module, module.reliability_beta.detach().clone()))
        with torch.no_grad():
            module.reliability_beta.zero_()
    try:
        yield
    finally:
        for _name, module, value in originals:
            with torch.no_grad():
                module.reliability_beta.copy_(value)


def resize_like(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def empty_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def classification_counts(pred, gt):
    return {
        "tp": int((pred & gt).sum().item()),
        "fp": int((pred & (~gt)).sum().item()),
        "fn": int(((~pred) & gt).sum().item()),
        "tn": int(((~pred) & (~gt)).sum().item()),
    }


def add_counts(total, counts):
    for key in total:
        total[key] += counts[key]


def summarize_counts(total):
    precision = total["tp"] / max(total["tp"] + total["fp"], 1)
    recall = total["tp"] / max(total["tp"] + total["fn"], 1)
    iou = total["tp"] / max(total["tp"] + total["fp"] + total["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, iou, f1


def empty_feature_buckets():
    return {name: empty_bucket() for name in FEATURE_NAMES}


def mean_row(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def topology_summary(cldice_values, fragment_rows):
    return {
        "cldice": float(np.mean(cldice_values)) if cldice_values else float("nan"),
        "frag_idx": mean_row(fragment_rows, "fragmentation_index"),
        "extra_comp": mean_row(fragment_rows, "extra_components"),
        "short_components": mean_row(fragment_rows, "pred_short_components"),
        "largest_ratio": mean_row(fragment_rows, "pred_largest_ratio"),
    }


def add_topology(cldice_values, fragment_rows, surface_logits, mask, threshold, short_area):
    surface_prob = torch.sigmoid(surface_logits)
    resized_mask = resize_like(mask.float(), surface_logits, mode="nearest")
    cldice_values.extend(surface_cldice_score(surface_prob, resized_mask).tolist())
    fragment_rows.extend(
        surface_fragmentation_stats(
            surface_prob,
            resized_mask,
            threshold=threshold,
            short_area_threshold=short_area,
        )
    )


def format_surface_line(name, total):
    precision, recall, iou, f1 = summarize_counts(total)
    return (
        f"  {name:<16} IoU={iou:.6f} F1={f1:.6f} "
        f"Precision={precision:.6f} Recall={recall:.6f} "
        f"TP={total['tp']} FP={total['fp']} FN={total['fn']} TN={total['tn']}"
    )


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    model.eval()

    beta_snapshot = [
        (name, float(module.reliability_beta.detach().cpu()))
        for name, module in reliability_modules(model)
    ]

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
        pin_memory=torch.cuda.is_available() and args.num_workers > 0,
    )

    off_total = empty_counts()
    on_total = empty_counts()
    off_cldice = []
    on_cldice = []
    off_frag_rows = []
    on_frag_rows = []
    transitions = {
        "FN_to_TP": 0,
        "FP_to_TN": 0,
        "TN_to_FP": 0,
        "TP_to_FN": 0,
        "changed": 0,
    }
    transition_features = {
        "FN_to_TP": empty_feature_buckets(),
        "TN_to_FP": empty_feature_buckets(),
    }
    batches = 0
    images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Reliability delta quadrants")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batches += 1
            images += int(batch["image"].shape[0])
            image = batch["image"].to(device)
            mask = (batch["mask"].to(device) > 0.5).float()
            skeleton = batch["skeleton"].to(device).float()

            with reliability_off(model):
                off_outputs = model(
                    image,
                    gt_skeleton=skeleton,
                    topology_alpha_scale=1.0,
                    teacher_forcing_ratio=0.0,
                )
            on_outputs = model(
                image,
                gt_skeleton=skeleton,
                topology_alpha_scale=1.0,
                teacher_forcing_ratio=0.0,
            )

            off_logits = off_outputs[0]
            on_logits = resize_like(on_outputs[0], off_logits, mode="bilinear")
            stage_outputs = on_outputs[4] if isinstance(on_outputs, tuple) and len(on_outputs) > 4 else []
            feature_maps, _ = extract_structure_features(stage_outputs, off_logits)

            gt = resize_like(mask, off_logits, mode="nearest") > 0.5
            off_pred = torch.sigmoid(off_logits) >= args.threshold
            on_pred = torch.sigmoid(on_logits) >= args.threshold

            add_counts(off_total, classification_counts(off_pred, gt))
            add_counts(on_total, classification_counts(on_pred, gt))
            add_topology(off_cldice, off_frag_rows, off_logits, mask, args.threshold, args.fragment_short_area)
            add_topology(on_cldice, on_frag_rows, on_logits, mask, args.threshold, args.fragment_short_area)

            fn_off = gt & (~off_pred)
            fp_off = (~gt) & off_pred
            tp_off = gt & off_pred
            tn_off = (~gt) & (~off_pred)
            changed = off_pred != on_pred

            transitions["changed"] += int(changed.sum().item())
            transitions["FN_to_TP"] += int((fn_off & on_pred).sum().item())
            transitions["FP_to_TN"] += int((fp_off & (~on_pred)).sum().item())
            transitions["TN_to_FP"] += int((tn_off & on_pred).sum().item())
            transitions["TP_to_FN"] += int((tp_off & (~on_pred)).sum().item())

            masks_by_transition = {
                "FN_to_TP": fn_off & on_pred,
                "TN_to_FP": tn_off & on_pred,
            }
            for transition_name, transition_mask in masks_by_transition.items():
                for feature_name, feature_map in feature_maps.items():
                    update_bucket(
                        transition_features[transition_name][feature_name],
                        feature_map[transition_mask],
                    )

    off_topology = topology_summary(off_cldice, off_frag_rows)
    on_topology = topology_summary(on_cldice, on_frag_rows)
    good = transitions["FN_to_TP"] + transitions["FP_to_TN"]
    bad = transitions["TN_to_FP"] + transitions["TP_to_FN"]
    quality = good / max(transitions["changed"], 1)

    print("\nRELIABILITY CORRECTION DELTA DIAGNOSTIC")
    print(f"split={args.split} batches={batches} images={images} threshold={args.threshold:.3f}")
    print(f"model_path={args.model_path}")
    print("")
    print("Reliability beta checkpoint values")
    if beta_snapshot:
        for name, beta in beta_snapshot:
            print(f"  {name}: {beta:.8f}")
    else:
        print("  no reliability_beta parameters found")
    print("")
    print("Surface metrics")
    print(format_surface_line("Reliability OFF", off_total))
    print(format_surface_line("Reliability ON", on_total))
    print("")
    print("Topology metrics")
    print(
        "  Reliability OFF clDice={cldice:.6f} frag_idx={frag_idx:.6f} "
        "extra_comp={extra_comp:.6f} short_components={short_components:.6f} "
        "largest_ratio={largest_ratio:.6f}".format(**off_topology)
    )
    print(
        "  Reliability ON  clDice={cldice:.6f} frag_idx={frag_idx:.6f} "
        "extra_comp={extra_comp:.6f} short_components={short_components:.6f} "
        "largest_ratio={largest_ratio:.6f}".format(**on_topology)
    )
    print("")
    print("Binary transitions: Reliability ON relative to OFF")
    print(f"  FN -> TP: {transitions['FN_to_TP']}")
    print(f"  FP -> TN: {transitions['FP_to_TN']}")
    print(f"  TN -> FP: {transitions['TN_to_FP']}")
    print(f"  TP -> FN: {transitions['TP_to_FN']}")
    print(f"  changed:  {transitions['changed']}")
    print(f"  good:     {good}")
    print(f"  bad:      {bad}")
    print(f"  Q_rel:    {quality:.8f}")
    print("")
    print("Transition feature statistics from Reliability ON pass")
    for transition_name in ("FN_to_TP", "TN_to_FP"):
        print(f"  {transition_name}")
        print(
            "    {:<18} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
                "feature",
                "mean",
                "p05",
                "p25",
                "p50",
                "p75",
                "p95",
                "pixels",
            )
        )
        print("    " + "-" * 102)
        for feature_name in FEATURE_NAMES:
            stats = bucket_stats(transition_features[transition_name][feature_name])
            print(
                "    {:<18} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12}".format(
                    feature_name,
                    stats["mean"],
                    stats["p05"],
                    np.nan if stats["count"] <= 0 else float(
                        np.quantile(
                            np.concatenate(
                                transition_features[transition_name][feature_name]["values"],
                                axis=0,
                            ),
                            0.25,
                        )
                    ),
                    stats["p50"],
                    np.nan if stats["count"] <= 0 else float(
                        np.quantile(
                            np.concatenate(
                                transition_features[transition_name][feature_name]["values"],
                                axis=0,
                            ),
                            0.75,
                        )
                    ),
                    stats["p95"],
                    stats["count"],
                )
            )


if __name__ == "__main__":
    main()
