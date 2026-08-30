import argparse
import contextlib
import os
import random
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare structure-on and structure-off surface logits by "
            "structure-off TP/FP/FN/TN quadrants."
        )
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260830)
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


@contextlib.contextmanager
def structure_off(model):
    module = model.module if hasattr(model, "module") else model
    swin = getattr(module, "swin_unet", module)
    originals = {}

    def remember(name, value):
        originals[name] = value

    if hasattr(swin, "_run_decoder_structure_block"):
        remember("_run_decoder_structure_block", swin._run_decoder_structure_block)

        def zero_run(self, feature_map, stage, bottleneck_tokens, block_stage=None, **kwargs):
            target = getattr(self, "decoder_structure_blocks", None)
            block = None
            if target is not None:
                try:
                    block = target[stage if block_stage is None else block_stage]
                except Exception:
                    block = None
            if block is not None and hasattr(block, "forward"):
                try:
                    return block(
                        feature_map,
                        apply_feature_refinement=False,
                        disable_skeleton_prediction=True,
                    )
                except TypeError:
                    pass
            batch, _, height, width = feature_map.shape
            dtype = feature_map.dtype
            device = feature_map.device
            return (
                feature_map,
                torch.zeros(batch, 1, height, width, device=device, dtype=dtype),
                torch.zeros(batch, 8, height, width, device=device, dtype=dtype),
                torch.zeros(batch, 2, height, width, device=device, dtype=dtype),
                torch.zeros(batch, 1, height, width, device=device, dtype=dtype),
                None,
            )

        swin._run_decoder_structure_block = types.MethodType(zero_run, swin)

    if hasattr(swin, "_apply_highres_structure_fusion"):
        remember("_apply_highres_structure_fusion", swin._apply_highres_structure_fusion)

        def no_highres_fusion(self, x, z_struct, stage, target_hw):
            return x

        swin._apply_highres_structure_fusion = types.MethodType(no_highres_fusion, swin)

    if hasattr(swin, "_apply_structure_surface_correction"):
        remember("_apply_structure_surface_correction", swin._apply_structure_surface_correction)

        def no_surface_correction(self, outputs, z_struct, structure_outputs):
            return outputs

        swin._apply_structure_surface_correction = types.MethodType(no_surface_correction, swin)

    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(swin, name, value)


def resize_like(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def empty_bucket():
    return {
        "count": 0,
        "sum": 0.0,
        "abs_sum": 0.0,
        "pos": 0,
        "neg": 0,
        "values": [],
    }


def update_bucket(bucket, values):
    if values.numel() == 0:
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


def classification_metrics(pred, gt):
    tp = int((pred & gt).sum().item())
    fp = int((pred & (~gt)).sum().item())
    fn = int(((~pred) & gt).sum().item())
    tn = int(((~pred) & (~gt)).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def add_metrics(total, metrics):
    for key in ("tp", "fp", "fn", "tn"):
        total[key] += metrics[key]


def summarize_metrics(total):
    precision = total["tp"] / max(total["tp"] + total["fp"], 1)
    recall = total["tp"] / max(total["tp"] + total["fn"], 1)
    iou = total["tp"] / max(total["tp"] + total["fp"] + total["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, iou, f1


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
        pin_memory=torch.cuda.is_available(),
    )

    buckets = {
        "FN_off_GTroad_predbg": empty_bucket(),
        "FP_off_GTbg_predroad": empty_bucket(),
        "TP_off_GTroad_predroad": empty_bucket(),
        "TN_off_GTbg_predbg": empty_bucket(),
    }
    off_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    on_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    transitions = {
        "FN_to_TP": 0,
        "FP_to_TN": 0,
        "TN_to_FP": 0,
        "TP_to_FN": 0,
        "changed": 0,
    }
    delta_abs_values = []
    base_std_values = []
    batches = 0
    images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Structure delta quadrants")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batches += 1
            images += int(batch["image"].shape[0])
            image = batch["image"].to(device)
            mask = (batch["mask"].to(device) > 0.5).float()
            skeleton = batch["skeleton"].to(device).float()

            on_logits = model(image, gt_skeleton=skeleton)[0]
            with structure_off(model):
                off_logits = model(
                    image,
                    gt_skeleton=skeleton,
                    topology_alpha_scale=0.0,
                    teacher_forcing_ratio=0.0,
                )[0]

            gt = resize_like(mask, off_logits, mode="nearest") > 0.5
            on_logits = resize_like(on_logits, off_logits, mode="bilinear")
            delta = on_logits - off_logits
            off_pred = torch.sigmoid(off_logits) >= args.threshold
            on_pred = torch.sigmoid(on_logits) >= args.threshold

            fn_off = gt & (~off_pred)
            fp_off = (~gt) & off_pred
            tp_off = gt & off_pred
            tn_off = (~gt) & (~off_pred)

            update_bucket(buckets["FN_off_GTroad_predbg"], delta[fn_off])
            update_bucket(buckets["FP_off_GTbg_predroad"], delta[fp_off])
            update_bucket(buckets["TP_off_GTroad_predroad"], delta[tp_off])
            update_bucket(buckets["TN_off_GTbg_predbg"], delta[tn_off])

            add_metrics(off_total, classification_metrics(off_pred, gt))
            add_metrics(on_total, classification_metrics(on_pred, gt))

            changed = off_pred != on_pred
            transitions["changed"] += int(changed.sum().item())
            transitions["FN_to_TP"] += int((fn_off & on_pred).sum().item())
            transitions["FP_to_TN"] += int((fp_off & (~on_pred)).sum().item())
            transitions["TN_to_FP"] += int((tn_off & on_pred).sum().item())
            transitions["TP_to_FN"] += int((tp_off & (~on_pred)).sum().item())

            delta_abs_values.append(delta.detach().abs().reshape(-1).cpu().numpy())
            base_std_values.append(float(off_logits.detach().float().std(unbiased=False).cpu().item()))

    off_precision, off_recall, off_iou, off_f1 = summarize_metrics(off_total)
    on_precision, on_recall, on_iou, on_f1 = summarize_metrics(on_total)
    delta_abs = np.concatenate(delta_abs_values, axis=0) if delta_abs_values else np.empty((0,), dtype=np.float32)
    delta_abs_mean = float(delta_abs.mean()) if delta_abs.size else float("nan")
    base_std_mean = float(np.mean(base_std_values)) if base_std_values else float("nan")
    r_delta = delta_abs_mean / max(base_std_mean, 1e-12)
    good = transitions["FN_to_TP"] + transitions["FP_to_TN"]
    bad = transitions["TN_to_FP"] + transitions["TP_to_FN"]
    quality = good / max(transitions["changed"], 1)

    print("\nSTRUCTURE DELTA QUADRANT DIAGNOSTIC")
    print(f"split={args.split} batches={batches} images={images} threshold={args.threshold:.3f}")
    print(f"model_path={args.model_path}")
    print("\nSurface metrics")
    print(
        "  off: IoU={:.6f} F1={:.6f} Precision={:.6f} Recall={:.6f} "
        "TP={} FP={} FN={} TN={}".format(
            off_iou,
            off_f1,
            off_precision,
            off_recall,
            off_total["tp"],
            off_total["fp"],
            off_total["fn"],
            off_total["tn"],
        )
    )
    print(
        "  on:  IoU={:.6f} F1={:.6f} Precision={:.6f} Recall={:.6f} "
        "TP={} FP={} FN={} TN={}".format(
            on_iou,
            on_f1,
            on_precision,
            on_recall,
            on_total["tp"],
            on_total["fp"],
            on_total["fn"],
            on_total["tn"],
        )
    )
    print(
        "  delta_abs_mean={:.8f} surface_off_logit_std_mean={:.8f} R_delta={:.8f}".format(
            delta_abs_mean,
            base_std_mean,
            r_delta,
        )
    )

    print("\nSigned delta z by structure-off quadrant")
    print(
        "{:<24} {:>12} {:>12} {:>12} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
            "Quadrant",
            "pixels",
            "mean_dz",
            "mean_abs",
            "pos_ratio",
            "neg_ratio",
            "p05",
            "p50",
            "p95",
        )
    )
    print("-" * 120)
    for name, bucket in buckets.items():
        stats = bucket_stats(bucket)
        print(
            "{:<24} {:>12} {:>12.6f} {:>12.6f} {:>10.4f} {:>10.4f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
                name,
                stats["count"],
                stats["mean"],
                stats["mean_abs"],
                stats["pos_ratio"],
                stats["neg_ratio"],
                stats["p05"],
                stats["p50"],
                stats["p95"],
            )
        )

    print("\nBinary transition counts")
    print(f"  FN->TP: {transitions['FN_to_TP']}")
    print(f"  FP->TN: {transitions['FP_to_TN']}")
    print(f"  TN->FP: {transitions['TN_to_FP']}")
    print(f"  TP->FN: {transitions['TP_to_FN']}")
    print(f"  changed: {transitions['changed']}")
    print(f"  good_changed: {good}")
    print(f"  bad_changed: {bad}")
    print(f"  Q=(FN->TP + FP->TN) / changed = {quality:.6f}")


if __name__ == "__main__":
    main()
