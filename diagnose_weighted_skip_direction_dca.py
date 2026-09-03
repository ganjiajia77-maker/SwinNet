import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


MODE_SETTINGS = {
    "current_full": {
        "dir_off": False,
        "skip_mode": "learned",
        "dca_mode": "current_unweighted",
        "skip_decoder_off": False,
        "skip_raw_off": False,
        "skip_dca_off": False,
    },
    "connectivity_embedding_off": {
        "dir_off": True,
        "skip_mode": "learned",
        "dca_mode": "current_unweighted",
        "skip_decoder_off": False,
        "skip_raw_off": False,
        "skip_dca_off": False,
    },
    "weighted_skip_uniform": {
        "dir_off": False,
        "skip_mode": "uniform",
        "dca_mode": "current_unweighted",
        "skip_decoder_off": False,
        "skip_raw_off": False,
        "skip_dca_off": False,
    },
    "weighted_skip_spatial_mean": {
        "dir_off": False,
        "skip_mode": "spatial_mean",
        "dca_mode": "current_unweighted",
        "skip_decoder_off": False,
        "skip_raw_off": False,
        "skip_dca_off": False,
    },
    "weighted_skip_no_decoder": {
        "dir_off": False,
        "skip_mode": "learned",
        "dca_mode": "current_unweighted",
        "skip_decoder_off": True,
        "skip_raw_off": False,
        "skip_dca_off": False,
    },
    "weighted_skip_no_raw": {
        "dir_off": False,
        "skip_mode": "learned",
        "dca_mode": "current_unweighted",
        "skip_decoder_off": False,
        "skip_raw_off": True,
        "skip_dca_off": False,
    },
    "weighted_skip_no_dca": {
        "dir_off": False,
        "skip_mode": "learned",
        "dca_mode": "current_unweighted",
        "skip_decoder_off": False,
        "skip_raw_off": False,
        "skip_dca_off": True,
    },
}


class WeightStats:
    def __init__(self):
        self.n = 0
        self.sum = None
        self.sumsq = None
        self.min = None
        self.max = None
        self.entropy_sum = 0.0
        self.top1_conf_sum = 0.0
        self.top1_counts = None
        self.samples = []
        self.sample_limit = 200000

    def update(self, weights):
        w = weights.detach().float().cpu()
        channels = w.shape[1]
        flat = w.permute(1, 0, 2, 3).reshape(channels, -1)
        count = flat.shape[1]
        if self.sum is None:
            self.sum = torch.zeros(channels, dtype=torch.float64)
            self.sumsq = torch.zeros(channels, dtype=torch.float64)
            self.min = torch.full((channels,), float("inf"), dtype=torch.float64)
            self.max = torch.full((channels,), float("-inf"), dtype=torch.float64)
            self.top1_counts = torch.zeros(channels, dtype=torch.float64)
        flat64 = flat.double()
        self.sum += flat64.sum(dim=1)
        self.sumsq += (flat64 * flat64).sum(dim=1)
        self.min = torch.minimum(self.min, flat64.min(dim=1).values)
        self.max = torch.maximum(self.max, flat64.max(dim=1).values)
        entropy = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=1)
        self.entropy_sum += float(entropy.sum().item())
        top1 = w.argmax(dim=1).reshape(-1)
        self.top1_counts += torch.bincount(top1, minlength=channels).double()
        self.top1_conf_sum += float(w.max(dim=1).values.sum().item())
        remaining = self.sample_limit - sum(sample.numel() for sample in self.samples)
        sample_count = min(count, remaining // channels) if remaining > 0 else 0
        if sample_count > 0:
            self.samples.append(flat[:, :sample_count].reshape(-1).clone())
        self.n += count

    def summary(self):
        if self.n == 0:
            return None
        mean = self.sum / self.n
        var = torch.clamp(self.sumsq / self.n - mean * mean, min=0.0)
        samples = torch.cat(self.samples).numpy()
        return {
            "mean": mean.numpy(),
            "std": torch.sqrt(var).numpy(),
            "min": self.min.numpy(),
            "max": self.max.numpy(),
            "p05": np.quantile(samples, 0.05),
            "p50": np.quantile(samples, 0.50),
            "p95": np.quantile(samples, 0.95),
            "entropy": self.entropy_sum / self.n,
            "top1_ratio": (self.top1_counts / self.n).numpy(),
            "top1_conf": self.top1_conf_sum / self.n,
        }


class RegionStats:
    def __init__(self):
        self.count = 0
        self.ed_sum = 0.0
        self.ed_sumsq = 0.0
        self.dg_sum = 0.0
        self.dg_sumsq = 0.0

    def update(self, edir, delta_gate, mask):
        if mask.sum().item() == 0:
            return
        edir_vals = torch.linalg.vector_norm(edir.float(), dim=1)[mask].reshape(-1)
        dg_vals = delta_gate[:, 0][mask].float().reshape(-1)
        if edir_vals.numel() == 0 or dg_vals.numel() == 0:
            return
        self.count += int(dg_vals.numel())
        self.ed_sum += float(edir_vals.sum().item())
        self.ed_sumsq += float((edir_vals * edir_vals).sum().item())
        self.dg_sum += float(dg_vals.sum().item())
        self.dg_sumsq += float((dg_vals * dg_vals).sum().item())

    def summary(self):
        if self.count == 0:
            return None
        ed_mean = self.ed_sum / max(self.count, 1)
        ed_var = self.ed_sumsq / max(self.count, 1) - ed_mean * ed_mean
        dg_mean = self.dg_sum / self.count
        dg_var = self.dg_sumsq / self.count - dg_mean * dg_mean
        return {
            "n": self.count,
            "edir_mean": ed_mean,
            "edir_std": max(ed_var, 0.0) ** 0.5,
            "delta_gate_mean": dg_mean,
            "delta_gate_std": max(dg_var, 0.0) ** 0.5,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--crop_list", type=str, default="")
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none", choices=["none", "stage3", "stage23"])
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton", choices=["pairwise_skeleton", "gap_query"])
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--bottleneck_type", type=str, default="global_local", choices=["global_local", "legacy_global_local", "g2l2"])
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_FULL, choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626])
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23", choices=["stage2", "stage3", "stage23"])
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23", choices=["stage23", "final_correction", "stage23_final_correction", "post_refine_interaction", "none"])
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    parser.add_argument("--modes", nargs="+", default=list(MODE_SETTINGS))
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    return parser.parse_args()


def apply_saved_args(args, checkpoint):
    saved_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
    saved_profile = checkpoint.get("structure_profile")
    if saved_profile:
        args.structure_profile = saved_profile
    elif saved_args:
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
    for name in (
        "stage_topology_stages",
        "stage_topology_alpha_max",
        "stage_topology_alpha_init",
        "stage_topology_bias_mode",
        "stage_topology_ratio",
        "stage_topology_topo_clip",
        "stage2_skeleton_gradient_ratio",
        "stage3_skeleton_gradient_ratio",
        "final_skeleton_gradient_ratio",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "highres_structure_fusion_mode",
        "enable_post_refine_structure_interaction",
        "bottleneck_type",
    ):
        if name in saved_args:
            setattr(args, name, saved_args[name])
    if "disable_msfe_skip" in saved_args:
        args.disable_msfe_skip = bool(saved_args["disable_msfe_skip"])


def build_model(args, checkpoint, device):
    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        stage_topology_bias_mode=args.stage_topology_bias_mode,
        stage_topology_ratio=args.stage_topology_ratio,
        stage_topology_topo_clip=args.stage_topology_topo_clip,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
    )
    adapt_connectivity_modules_for_checkpoint(model, checkpoint["model_state_dict"], "standard")
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=(args.bottleneck_type == "global_local"),
    )
    model.to(device)
    model.eval()
    print_topology_coefficients(model)
    return model


def set_runtime_mode(model, settings, capture):
    for module in model.modules():
        if hasattr(module, "disable_direction_embedding"):
            module.disable_direction_embedding = bool(settings["dir_off"])
            module.capture_diagnostics = capture
            module.last_diagnostics = None
        if module.__class__.__name__ == "SwinTransformerSys":
            module.disable_weighted_skip_decoder = bool(settings["skip_decoder_off"])
            module.disable_weighted_skip_raw = bool(settings["skip_raw_off"])
            module.disable_weighted_skip_dca = bool(settings["skip_dca_off"])
        if module.__class__.__name__ == "WeightedSkipConcat":
            module.mode = settings["skip_mode"]
            module.capture_diagnostics = capture
            module.last_weights = None
        if module.__class__.__name__ == "DCAFPNLite":
            module.weight_mode = settings["dca_mode"]
            module.capture_diagnostics = capture
            module.last_weights = None


def update_weight_stats(model, dca_stats, skip_stats):
    for name, module in model.named_modules():
        weights = getattr(module, "last_weights", None)
        if weights is None:
            continue
        if module.__class__.__name__ == "DCAFPNLite":
            dca_stats.setdefault(name, WeightStats()).update(weights)
        elif module.__class__.__name__ == "WeightedSkipConcat":
            skip_stats.setdefault(name, WeightStats()).update(weights)


def update_region_stats(model, pred, target, region_stats):
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    regions = {
        "TP": pred & target,
        "TN": (~pred) & (~target),
        "FP": pred & (~target),
        "FN": (~pred) & target,
    }
    for name, module in model.named_modules():
        diag = getattr(module, "last_diagnostics", None)
        if not diag or "direction_embedding" not in diag:
            continue
        edir = diag["direction_embedding"]
        delta_gate = diag["structure_gate_delta_direction_embedding"]
        h, w = edir.shape[-2:]
        module_stats = region_stats.setdefault(
            name,
            {key: RegionStats() for key in regions},
        )
        for region_name, mask in regions.items():
            resized = F.interpolate(
                mask.float().unsqueeze(1),
                size=(h, w),
                mode="nearest",
            )[:, 0].bool().cpu()
            module_stats[region_name].update(edir.cpu(), delta_gate.cpu(), resized)


def run_mode(model, loader, device, args, mode_name):
    settings = MODE_SETTINGS[mode_name]
    set_runtime_mode(model, settings, capture=True)
    tp = fp = fn = 0
    preds = []
    targets = []
    dca_stats = {}
    skip_stats = {}
    region_stats = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=mode_name):
            images = batch["image"].to(device)
            target = (batch["mask"].to(device) > 0.5)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if target.shape[-2:] != logits.shape[-2:]:
                target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest").bool()
            pred = torch.sigmoid(logits) >= args.threshold
            tp += int((pred & target).sum().item())
            fp += int((pred & (~target)).sum().item())
            fn += int(((~pred) & target).sum().item())
            preds.append(pred.cpu())
            targets.append(target.cpu())
            update_weight_stats(model, dca_stats, skip_stats)
            update_region_stats(model, pred.detach().cpu(), target.detach().cpu(), region_stats)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "metrics": {"iou": iou, "f1": f1, "precision": precision, "recall": recall},
        "preds": preds,
        "targets": targets,
        "dca_stats": dca_stats,
        "skip_stats": skip_stats,
        "region_stats": region_stats,
    }


def changed_pixel_summary(base, variant):
    counts = {"FN->TP": 0, "FP->TN": 0, "TN->FP": 0, "TP->FN": 0, "changed": 0}
    for base_pred, var_pred, target in zip(base["preds"], variant["preds"], base["targets"]):
        changed = base_pred != var_pred
        counts["changed"] += int(changed.sum().item())
        counts["FN->TP"] += int(((~base_pred) & var_pred & target).sum().item())
        counts["FP->TN"] += int((base_pred & (~var_pred) & (~target)).sum().item())
        counts["TN->FP"] += int(((~base_pred) & var_pred & (~target)).sum().item())
        counts["TP->FN"] += int((base_pred & (~var_pred) & target).sum().item())
    good = counts["FN->TP"] + counts["FP->TN"]
    counts["Q"] = good / max(counts["changed"], 1)
    return counts


def format_array(values):
    return " ".join(f"{float(v):.4f}" for v in values)


def print_weight_stats(title, stats):
    print(f"\n{title}")
    if not stats:
        print("  no captured weights")
        return
    for name, stat in stats.items():
        summary = stat.summary()
        if summary is None:
            continue
        stage = "Stage2" if "stage2" in name or name.endswith(".0") else "Stage3"
        print(f"  {stage} ({name})")
        print(f"    mean: {format_array(summary['mean'])}")
        print(f"    std : {format_array(summary['std'])}")
        print(f"    min : {format_array(summary['min'])}")
        print(f"    max : {format_array(summary['max'])}")
        print(
            f"    p05/p50/p95: {summary['p05']:.4f}/"
            f"{summary['p50']:.4f}/{summary['p95']:.4f}"
        )
        print(f"    entropy: {summary['entropy']:.4f}")
        print(f"    top1 ratio: {format_array(summary['top1_ratio'])}")
        print(f"    top1 confidence: {summary['top1_conf']:.4f}")


def print_region_stats(stats):
    print("\nDirectional embedding / gate delta by region")
    if not stats:
        print("  no captured direction embedding diagnostics")
        return
    for module_name, module_stats in stats.items():
        print(f"  {module_name}")
        for region in ("TP", "TN", "FP", "FN"):
            summary = module_stats[region].summary()
            if summary is None:
                print(f"    {region}: n=0")
                continue
            print(
                "    {}: n={} ||E_conn||_mean={:.6f} ||E_conn||_std={:.6f} "
                "DeltaG_mean={:.6f} DeltaG_std={:.6f}".format(
                    region,
                    summary["n"],
                    summary["edir_mean"],
                    summary["edir_std"],
                    summary["delta_gate_mean"],
                    summary["delta_gate_std"],
                )
            )


def main():
    args = parse_args()
    for mode in args.modes:
        if mode not in MODE_SETTINGS:
            raise ValueError(f"Unknown mode: {mode}")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    apply_saved_args(args, checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold:.4f}")
    model = build_model(args, checkpoint, device)
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.crop_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    results = {}
    for mode in args.modes:
        results[mode] = run_mode(model, loader, device, args, mode)

    print("\nMode metrics")
    print("{:<22} {:<10} {:<10} {:<10} {:<10}".format("mode", "IoU", "F1", "Prec", "Recall"))
    for mode in args.modes:
        metrics = results[mode]["metrics"]
        print(
            "{:<22} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                mode,
                metrics["iou"],
                metrics["f1"],
                metrics["precision"],
                metrics["recall"],
            )
        )

    base = results.get("current_full")
    if base is not None:
        print("\nChanged-pixel attribution vs both_on")
        for mode in args.modes:
            if mode == "both_on":
                continue
            counts = changed_pixel_summary(base, results[mode])
            print(
                "{}: changed={} FN->TP={} FP->TN={} TN->FP={} TP->FN={} Q={:.4f}".format(
                    mode,
                    counts["changed"],
                    counts["FN->TP"],
                    counts["FP->TN"],
                    counts["TN->FP"],
                    counts["TP->FN"],
                    counts["Q"],
                )
            )

    for mode in args.modes:
        print_weight_stats(f"DCA learned weight stats ({mode})", results[mode]["dca_stats"])
        print_weight_stats(f"Skip learned weight stats ({mode})", results[mode]["skip_stats"])
        if mode in ("current_full", "connectivity_embedding_off"):
            print_region_stats(results[mode]["region_stats"])


if __name__ == "__main__":
    main()
