import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


class StructureConditionedResidualCorrectionHead(nn.Module):
    def __init__(self, struct_channels):
        super().__init__()
        self.struct_project = nn.Sequential(
            nn.Conv2d(struct_channels, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(17, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.out = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self._init_weights()
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z_struct, base_surface_prob):
        z = self.struct_project(z_struct.detach())
        z = F.interpolate(
            z,
            size=base_surface_prob.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([z, base_surface_prob.detach()], dim=1)
        return self.out(self.fuse(x))


def cli_has(flag):
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def make_unique_dir(base_dir, run_name):
    run_dir = os.path.join(base_dir, run_name)
    if not os.path.exists(run_dir):
        return run_dir
    for index in range(2, 1000):
        candidate = f"{run_dir}_{index}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"Could not create unique output directory for {run_dir}")


def match_spatial_size(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def normalize_residual_weight(surface_gt, base_prob):
    with torch.no_grad():
        raw_weight = torch.abs(surface_gt - base_prob)
        normalized = torch.ones_like(raw_weight)
        pos_mask = surface_gt > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any():
            pos_weight = raw_weight[pos_mask]
            normalized[pos_mask] = pos_weight / pos_weight.mean().clamp_min(1e-6)
        if neg_mask.any():
            neg_weight = raw_weight[neg_mask]
            normalized[neg_mask] = neg_weight / neg_weight.mean().clamp_min(1e-6)
    return normalized


def empty_bucket():
    return {"sum": 0.0, "abs_sum": 0.0, "pos": 0, "count": 0}


def update_bucket(bucket, values):
    if values.numel() == 0:
        return
    values = values.detach()
    bucket["sum"] += float(values.sum().cpu())
    bucket["abs_sum"] += float(values.abs().sum().cpu())
    bucket["pos"] += int((values > 0).sum().cpu())
    bucket["count"] += int(values.numel())


def bucket_stats(bucket):
    count = bucket["count"]
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        bucket["sum"] / count,
        bucket["abs_sum"] / count,
        bucket["pos"] / count,
        count,
    )


def compute_binary_counts(pred, target):
    tp = int((pred & target).sum().cpu())
    fp = int((pred & (~target)).sum().cpu())
    fn = int(((~pred) & target).sum().cpu())
    tn = int(((~pred) & (~target)).sum().cpu())
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return iou, f1, precision, recall


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train only a tiny structure-conditioned residual correction head."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "./model_out/"
            "data1_highres_struct_detach_surface_w008_direct256_20260815/"
            "best.pth"
        ),
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--output_dir", type=str, default="./model_out")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--surface_pos_weight", type=float, default=0.0)
    parser.add_argument(
        "--delta_sign",
        type=str,
        default="add",
        choices=["add", "subtract"],
        help="add keeps positive delta as road-logit boost; subtract follows base - delta.",
    )
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
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument(
        "--structure_profile",
        type=str,
        default=STRUCTURE_PROFILE_FULL,
        choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626],
    )
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument(
        "--highres_structure_fuse_stages",
        type=str,
        default="stage23",
        choices=["stage2", "stage3", "stage23"],
    )
    parser.add_argument(
        "--highres_structure_fusion_mode",
        type=str,
        default="stage23",
        choices=["stage23", "final_correction", "stage23_final_correction", "none"],
    )
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument(
        "--stage_topology_stages",
        type=str,
        default="none",
        choices=["none", "stage3", "stage23"],
    )
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    return parser


def inherit_checkpoint_args(args, checkpoint):
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint, dict):
        saved_profile = checkpoint.get("structure_profile")
        if saved_profile:
            args.structure_profile = saved_profile
        elif isinstance(saved_args, dict):
            args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
    if not isinstance(saved_args, dict):
        return
    for name in (
        "disable_msfe_skip",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "img_size",
        "source_patch_size",
        "stage_topology_stages",
    ):
        if name in saved_args and not cli_has("--" + name):
            setattr(args, name, saved_args[name])
    args.highres_structure_fusion_mode = "stage23"


def corrected_logits(base_logits, delta_logits, delta_sign):
    if delta_sign == "subtract":
        return base_logits.detach() - delta_logits
    return base_logits.detach() + delta_logits


def forward_base_and_delta(model, correction_head, images, delta_sign):
    with torch.no_grad():
        outputs = model(images)
        if not isinstance(outputs, tuple):
            raise RuntimeError("Structure residual diagnostic requires auxiliary outputs.")
        base_surface_logits = outputs[0].detach()
        z_struct = getattr(model.swin_unet, "last_highres_z_struct", None)
        if z_struct is None:
            raise RuntimeError("High-res structure stream is not enabled.")
        z_struct = z_struct.detach()
        base_prob = torch.sigmoid(base_surface_logits).detach()
    delta_logits = correction_head(z_struct, base_prob)
    delta_logits = match_spatial_size(delta_logits, base_surface_logits, mode="bilinear")
    final_logits = corrected_logits(base_surface_logits, delta_logits, delta_sign)
    return base_surface_logits, base_prob, delta_logits, final_logits


def evaluate(model, correction_head, loader, args, device):
    model.eval()
    correction_head.eval()
    base_tp = base_fp = base_fn = 0
    corr_tp = corr_fp = corr_fn = 0
    rescued_fn = damaged_tp = removed_fp = added_fp = 0
    base_fn_total = base_tp_total = base_fp_total = base_tn_total = 0
    buckets = {
        "All": empty_bucket(),
        "Weak Skeleton FN": empty_bucket(),
        "Skeleton TP": empty_bucket(),
        "Background": empty_bucket(),
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            surface_gt = batch["mask"].to(device)
            skeleton_gt = batch["skeleton"].to(device)
            base_logits, _, delta_logits, final_logits = forward_base_and_delta(
                model,
                correction_head,
                images,
                args.delta_sign,
            )
            surface_gt = match_spatial_size(surface_gt, base_logits, mode="nearest") > 0.5
            skeleton_gt = match_spatial_size(skeleton_gt, base_logits, mode="nearest") > 0.5
            base_pred = torch.sigmoid(base_logits) >= args.threshold
            corr_pred = torch.sigmoid(final_logits) >= args.threshold

            batch_base_tp, batch_base_fp, batch_base_fn, batch_base_tn = compute_binary_counts(
                base_pred,
                surface_gt,
            )
            batch_corr_tp, batch_corr_fp, batch_corr_fn, _ = compute_binary_counts(
                corr_pred,
                surface_gt,
            )
            base_tp += batch_base_tp
            base_fp += batch_base_fp
            base_fn += batch_base_fn
            corr_tp += batch_corr_tp
            corr_fp += batch_corr_fp
            corr_fn += batch_corr_fn

            base_fn_mask = surface_gt & (~base_pred)
            base_tp_mask = surface_gt & base_pred
            base_fp_mask = (~surface_gt) & base_pred
            base_tn_mask = (~surface_gt) & (~base_pred)

            rescued_fn += int((base_fn_mask & corr_pred).sum().cpu())
            damaged_tp += int((base_tp_mask & (~corr_pred)).sum().cpu())
            removed_fp += int((base_fp_mask & (~corr_pred)).sum().cpu())
            added_fp += int((base_tn_mask & corr_pred).sum().cpu())
            base_fn_total += int(base_fn_mask.sum().cpu())
            base_tp_total += int(base_tp_mask.sum().cpu())
            base_fp_total += int(base_fp_mask.sum().cpu())
            base_tn_total += int(base_tn_mask.sum().cpu())

            weak_skeleton_fn = skeleton_gt & surface_gt & (~base_pred)
            skeleton_tp = skeleton_gt & surface_gt & base_pred
            background = ~surface_gt
            update_bucket(buckets["All"], delta_logits.reshape(-1))
            update_bucket(buckets["Weak Skeleton FN"], delta_logits[weak_skeleton_fn])
            update_bucket(buckets["Skeleton TP"], delta_logits[skeleton_tp])
            update_bucket(buckets["Background"], delta_logits[background])

    base_metrics = metrics_from_counts(base_tp, base_fp, base_fn)
    corr_metrics = metrics_from_counts(corr_tp, corr_fp, corr_fn)
    stats = {
        "base_iou": base_metrics[0],
        "base_f1": base_metrics[1],
        "base_precision": base_metrics[2],
        "base_recall": base_metrics[3],
        "iou": corr_metrics[0],
        "f1": corr_metrics[1],
        "precision": corr_metrics[2],
        "recall": corr_metrics[3],
        "rescued_fn": rescued_fn,
        "damaged_tp": damaged_tp,
        "removed_fp": removed_fp,
        "added_fp": added_fp,
        "rescue_rate": rescued_fn / base_fn_total if base_fn_total else float("nan"),
        "damage_rate": damaged_tp / base_tp_total if base_tp_total else float("nan"),
        "removed_fp_rate": removed_fp / base_fp_total if base_fp_total else float("nan"),
        "added_fp_rate": added_fp / base_tn_total if base_tn_total else float("nan"),
    }
    for name, bucket in buckets.items():
        prefix = name.lower().replace(" ", "_")
        mean_delta, mean_abs, pos_ratio, count = bucket_stats(bucket)
        stats[f"{prefix}_mean_delta"] = mean_delta
        stats[f"{prefix}_mean_abs"] = mean_abs
        stats[f"{prefix}_pos_ratio"] = pos_ratio
        stats[f"{prefix}_pixels"] = count
    return stats


def print_eval(epoch, stats):
    print(
        (
            f"[VAL][Epoch {epoch}] "
            f"Base IoU={stats['base_iou']:.6f} F1={stats['base_f1']:.6f} | "
            f"Corrected IoU={stats['iou']:.6f} F1={stats['f1']:.6f} "
            f"P={stats['precision']:.6f} R={stats['recall']:.6f}"
        ),
        flush=True,
    )
    print("Delta surface logits:", flush=True)
    for name in ("all", "weak_skeleton_fn", "skeleton_tp", "background"):
        print(
            (
                f"  {name:<18} "
                f"mean={stats[name + '_mean_delta']:.6f} "
                f"abs={stats[name + '_mean_abs']:.6f} "
                f"pos_ratio={stats[name + '_pos_ratio']:.4f} "
                f"pixels={int(stats[name + '_pixels'])}"
            ),
            flush=True,
        )
    print(
        (
            "Correction effectiveness: "
            f"rescued_fn={stats['rescued_fn']} ({stats['rescue_rate']:.4f}), "
            f"damaged_tp={stats['damaged_tp']} ({stats['damage_rate']:.4f}), "
            f"removed_fp={stats['removed_fp']} ({stats['removed_fp_rate']:.4f}), "
            f"added_fp={stats['added_fp']} ({stats['added_fp_rate']:.6f})"
        ),
        flush=True,
    )


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    inherit_checkpoint_args(args, checkpoint)
    args.highres_structure_fusion_mode = "stage23"
    if not args.enable_highres_structure_stream:
        raise RuntimeError("The diagnostic checkpoint must use high-res structure stream.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"structure_residual_diag_{timestamp}"
    output_dir = make_unique_dir(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "correction_diag_metrics.csv")
    best_path = os.path.join(output_dir, "best_correction_head.pth")
    last_path = os.path.join(output_dir, "last_correction_head.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)
    print(f"[INFO] Base checkpoint: {args.model_path}", flush=True)
    print(f"[INFO] Output dir: {output_dir}", flush=True)
    print(
        (
            "[INFO] Diagnostic setup: base model frozen, stage2/3 fusion kept, "
            f"delta_sign={args.delta_sign}, threshold={args.threshold}"
        ),
        flush=True,
    )

    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode="stage23",
    )
    uses_legacy_highres_encoder = any(
        key.startswith("swin_unet.highres_structure_encoder.")
        for key in checkpoint["model_state_dict"]
    )
    if uses_legacy_highres_encoder:
        model.swin_unet.highres_structure_source = "stage1_tokens"
        print(
            "[INFO] Detected legacy highres_structure_encoder checkpoint; "
            "using stage1_tokens as Z_struct source.",
            flush=True,
        )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print_topology_coefficients(model)

    correction_head = StructureConditionedResidualCorrectionHead(
        struct_channels=args.highres_structure_channels,
    ).to(device)
    trainable_params = sum(p.numel() for p in correction_head.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Frozen base params: {frozen_params:,}", flush=True)
    print(f"[INFO] Trainable correction params: {trainable_params:,}", flush=True)

    train_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="train",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        augment=False,
    )
    val_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="val",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        augment=False,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f"[INFO] Dataset: train={len(train_dataset)}, val={len(val_dataset)}",
        flush=True,
    )

    pos_weight = None
    if args.surface_pos_weight and args.surface_pos_weight > 0:
        pos_weight = torch.tensor([args.surface_pos_weight], dtype=torch.float32, device=device)
        print(f"[INFO] Surface BCE pos_weight={args.surface_pos_weight}", flush=True)

    optimizer = torch.optim.AdamW(
        correction_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "base_iou",
        "base_f1",
        "iou",
        "f1",
        "precision",
        "recall",
        "all_mean_delta",
        "all_mean_abs",
        "all_pos_ratio",
        "weak_skeleton_fn_mean_delta",
        "weak_skeleton_fn_mean_abs",
        "weak_skeleton_fn_pos_ratio",
        "skeleton_tp_mean_delta",
        "skeleton_tp_mean_abs",
        "skeleton_tp_pos_ratio",
        "background_mean_delta",
        "background_mean_abs",
        "background_pos_ratio",
        "rescued_fn",
        "damage_rate",
        "removed_fp",
        "added_fp",
        "rescue_rate",
        "removed_fp_rate",
        "added_fp_rate",
    ]
    best_iou = -1.0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.max_epochs + 1):
            correction_head.train()
            train_loss = 0.0
            train_batches = 0
            for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
                images = batch["image"].to(device)
                surface_gt = batch["mask"].to(device)
                base_logits, base_prob, _, final_logits = forward_base_and_delta(
                    model,
                    correction_head,
                    images,
                    args.delta_sign,
                )
                surface_gt = match_spatial_size(surface_gt, base_logits, mode="nearest")
                residual_weight = normalize_residual_weight(surface_gt, base_prob)
                corr_loss_map = F.binary_cross_entropy_with_logits(
                    final_logits,
                    surface_gt,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                loss = (corr_loss_map * residual_weight).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(correction_head.parameters(), 1.0)
                optimizer.step()

                train_loss += float(loss.detach().cpu())
                train_batches += 1

            train_avg = train_loss / max(train_batches, 1)
            stats = evaluate(model, correction_head, val_loader, args, device)
            stats["epoch"] = epoch
            stats["train_loss"] = train_avg
            print_eval(epoch, stats)

            row = {key: stats.get(key, "") for key in fieldnames}
            writer.writerow(row)
            csv_file.flush()

            state = {
                "epoch": epoch,
                "correction_head_state_dict": correction_head.state_dict(),
                "args": vars(args),
                "base_model_path": args.model_path,
                "val_stats": stats,
            }
            torch.save(state, last_path)
            if stats["iou"] > best_iou:
                best_iou = stats["iou"]
                torch.save(state, best_path)
                print(
                    f"[BEST] saved best_correction_head.pth with IoU={best_iou:.6f}",
                    flush=True,
                )

    print(f"[DONE] Best IoU={best_iou:.6f}", flush=True)
    print(f"[DONE] CSV: {csv_path}", flush=True)
    print(f"[DONE] Best head: {best_path}", flush=True)


if __name__ == "__main__":
    main()
