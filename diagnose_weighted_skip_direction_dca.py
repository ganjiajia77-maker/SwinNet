import argparse
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
from diagnose_structure_delta_quadrants import (
    bucket_stats,
    empty_bucket,
    extract_structure_features,
    update_bucket,
)


FEATURE_NAMES = (
    "surface_off_logit",
    "G_structure_old",
    "R_reliability",
    "G_final",
    "direction_confidence",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose weighted skip concat and connectivity direction embedding "
            "on the same checkpoint by baseline / ablation comparisons."
        )
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260901)
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
    parser.add_argument(
        "--model_impl",
        type=str,
        default="auto",
        choices=("auto", "standard", "selective"),
    )
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


def resize_like(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def empty_bucket():
    return {"count": 0, "sum": 0.0, "abs_sum": 0.0, "pos": 0, "neg": 0, "values": []}


def update_bucket(bucket, values):
    if values is None or values.numel() == 0:
        return
    detached = values.detach().float().cpu()
    bucket["count"] += int(detached.numel())
    bucket["sum"] += float(detached.double().sum().item())
    bucket["abs_sum"] += float(detached.abs().double().sum().item())
    bucket["pos"] += int((detached > 0).sum().item())
    bucket["neg"] += int((detached < 0).sum().item())
    bucket["values"].append(detached.reshape(-1).numpy())


def bucket_stats(bucket):
    count = bucket["count"]
    if count <= 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "mean_abs": float("nan"),
            "pos_ratio": float("nan"),
            "neg_ratio": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    values = np.concatenate(bucket["values"], axis=0)
    return {
        "count": count,
        "mean": bucket["sum"] / count,
        "mean_abs": bucket["abs_sum"] / count,
        "pos_ratio": bucket["pos"] / count,
        "neg_ratio": bucket["neg"] / count,
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def feature_bucket_map():
    return {name: empty_bucket() for name in FEATURE_NAMES}


def stage_id(item):
    try:
        return int(item.get("stage", -1))
    except (TypeError, ValueError):
        return -1


def single_channel_map(tensor):
    if tensor is None:
        return None
    if tensor.shape[1] == 1:
        return tensor
    return tensor.float().mean(dim=1, keepdim=True)


def extract_features(stage_outputs, reference_logits):
    features = {}
    selected_stage = None
    for item in stage_outputs or []:
        if stage_id(item) in (2, 3):
            selected_stage = item
    if selected_stage is None:
        return features
    for key in ("structure_gate_old", "reliability_correction", "structure_gate_final", "direction"):
        value = selected_stage.get(key)
        if value is None:
            continue
        resized = resize_like(value, reference_logits, mode="bilinear")
        if key == "direction":
            features["direction_confidence"] = resized.float().norm(dim=1, keepdim=True)
        elif key == "structure_gate_old":
            features["G_structure_old"] = single_channel_map(resized)
        elif key == "reliability_correction":
            features["R_reliability"] = single_channel_map(resized)
        elif key == "structure_gate_final":
            features["G_final"] = single_channel_map(resized)
    return features


@contextlib.contextmanager
def direction_disabled(model):
    module = get_swin_unet(model)
    originals = []
    for block in getattr(module, "decoder_structure_blocks", []):
        if hasattr(block, "_build_decoder_structure_attention_bias"):
            original = block._build_decoder_structure_attention_bias
            originals.append((block, original))

            def disabled(self, skeleton_prob, connectivity_prob, direction_prob=None, _orig=original):
                return _orig(skeleton_prob, connectivity_prob, None)

            block._build_decoder_structure_attention_bias = disabled.__get__(
                block, block.__class__
            )
    try:
        yield
    finally:
        for block, original in originals:
            block._build_decoder_structure_attention_bias = original


def classification_counts(pred, gt):
    return {
        "tp": int((pred & gt).sum().item()),
        "fp": int((pred & (~gt)).sum().item()),
        "fn": int(((~pred) & gt).sum().item()),
        "tn": int(((~pred) & (~gt)).sum().item()),
    }


def summarize_counts(total):
    precision = total["tp"] / max(total["tp"] + total["fp"], 1)
    recall = total["tp"] / max(total["tp"] + total["fn"], 1)
    iou = total["tp"] / max(total["tp"] + total["fp"] + total["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, iou, f1


def empty_transition_counts():
    return {"FN_to_TP": 0, "FP_to_TN": 0, "TN_to_FP": 0, "TP_to_FN": 0, "changed": 0}


def collect_skip_weight_stats(model, image, skeleton, threshold):
    module = get_swin_unet(model)
    if not hasattr(module, "weighted_skip_concat_blocks"):
        return None
    blocks = getattr(module, "weighted_skip_concat_blocks", [])
    if not blocks:
        return None
    return {"available": True, "blocks": len(blocks), "threshold": threshold}


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    model.eval()

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

    modes = [("baseline", False), ("direction_off", True)]
    results = {}

    for mode_name, disable_direction in modes:
        total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        transitions = empty_transition_counts()
        feature_buckets = {
            "FN_to_TP": feature_bucket_map(),
            "FP_to_TN": feature_bucket_map(),
            "TN_to_FP": feature_bucket_map(),
            "TP_to_FN": feature_bucket_map(),
        }
        dca_weight_buckets = {
            "learned_mean": empty_bucket(),
            "learned_entropy": empty_bucket(),
            "learned_top1": empty_bucket(),
        }

        with torch.no_grad():
            direction_ctx = direction_disabled(model) if disable_direction else contextlib.nullcontext()
            with direction_ctx:
                for batch_idx, batch in enumerate(tqdm(loader, desc=mode_name, leave=False)):
                    if args.max_batches > 0 and batch_idx >= args.max_batches:
                        break
                    image = batch["image"].to(device)
                    mask = (batch["mask"].to(device) > 0.5)
                    skeleton = batch["skeleton"].to(device).float()

                    outputs = model(image, gt_skeleton=skeleton)
                    surface_logits = outputs[0]
                    stage_outputs = outputs[4] if isinstance(outputs, tuple) and len(outputs) > 4 else []

                    pred = torch.sigmoid(surface_logits) >= args.threshold
                    gt = resize_like(mask.float(), surface_logits, mode="nearest") > 0.5
                    counts = classification_counts(pred, gt)
                    for key in total:
                        total[key] += counts[key]

                    features = extract_features(stage_outputs, surface_logits)
                    features["surface_off_logit"] = surface_logits

                    if mode_name == "baseline":
                        fn_off = gt & (~pred)
                        fp_off = (~gt) & pred
                        tn_off = (~gt) & (~pred)
                        tp_off = gt & pred
                        changed = torch.zeros_like(pred, dtype=torch.bool)
                        transitions["changed"] += int(changed.sum().item())
                        for feat_name, feat_map in features.items():
                            update_bucket(feature_buckets["FN_to_TP"][feat_name], feat_map[fn_off])
                            update_bucket(feature_buckets["FP_to_TN"][feat_name], feat_map[fp_off])
                            update_bucket(feature_buckets["TN_to_FP"][feat_name], feat_map[tn_off])
                            update_bucket(feature_buckets["TP_to_FN"][feat_name], feat_map[tp_off])
                        if "R_reliability" in features:
                            r = features["R_reliability"]
                            update_bucket(dca_weight_buckets["learned_mean"], r.mean(dim=(1, 2, 3), keepdim=True))
                            probs = torch.softmax(torch.cat([r, r, r, r], dim=1), dim=1)
                            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)
                            top1 = probs.max(dim=1, keepdim=True).values
                            update_bucket(dca_weight_buckets["learned_entropy"], entropy)
                            update_bucket(dca_weight_buckets["learned_top1"], top1)

        precision, recall, iou, f1 = summarize_counts(total)
        results[mode_name] = {
            "total": total,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "f1": f1,
            "features": feature_buckets,
            "dca_weights": dca_weight_buckets,
            "transitions": transitions,
        }

    base = results["baseline"]
    off = results["direction_off"]

    print("\nWEIGHTED SKIP / DIRECTION EMBEDDING DIAGNOSTIC")
    print(f"model_path={args.model_path}")
    print(f"split={args.split} threshold={args.threshold:.3f}")
    print(
        f"Baseline    IoU={base['iou']:.6f} F1={base['f1']:.6f} "
        f"Precision={base['precision']:.6f} Recall={base['recall']:.6f}"
    )
    print(
        f"Dir OFF     IoU={off['iou']:.6f} F1={off['f1']:.6f} "
        f"Precision={off['precision']:.6f} Recall={off['recall']:.6f}"
    )
    print(
        f"Delta       IoU={off['iou'] - base['iou']:+.6f} "
        f"F1={off['f1'] - base['f1']:+.6f} "
        f"Recall={off['recall'] - base['recall']:+.6f}"
    )

    print("\nStage feature stats (baseline)")
    for name in FEATURE_NAMES:
        stats = bucket_stats(base["features"]["FN_to_TP"][name])
        print(
            f"  {name:<20} FN->TP mean={stats['mean']:.6f} p05={stats['p05']:.6f} "
            f"p50={stats['p50']:.6f} p95={stats['p95']:.6f} count={stats['count']}"
        )

    print("\nDirection transition buckets")
    print("  Note: this script keeps the baseline checkpoint and toggles direction attention off.")
    print("  Use the IoU/F1 delta above as the main ON/OFF signal.")
    if hasattr(get_swin_unet(model), "weighted_skip_concat_blocks"):
        print("\nWeighted skip concat module detected.")
        print("  This repo version exposes a live weighted skip concat block, so the next step")
        print("  is to add a small runtime override for learned/uniform/unweighted weights.")
    else:
        print("\nWeighted skip concat module not exposed by the loaded model.")
        print("  Learned/uniform/unweighted skip-weight ablation is unavailable on this code path.")


if __name__ == "__main__":
    main()
