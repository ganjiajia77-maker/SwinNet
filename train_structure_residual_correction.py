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


def largest_group_divisor(channels, candidates=(8, 4, 2, 1)):
    for groups in candidates:
        if channels % groups == 0:
            return groups
    return 1


class StructureConditionedCorrectionHead(nn.Module):
    def __init__(self, struct_channels):
        super().__init__()
        self.z_project = nn.Sequential(
            nn.Conv2d(struct_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(17, 16, kernel_size=3, padding=1, bias=False),
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

    def forward(self, z_struct, base_prob):
        z = self.z_project(z_struct)
        z = F.interpolate(z, size=base_prob.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([z, base_prob], dim=1)
        x = self.fuse(x)
        return self.out(x)


def cli_has(flag):
    return flag in sys.argv[1:]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def match_spatial_size(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def binary_counts(logits, target, threshold):
    pred = torch.sigmoid(logits) >= threshold
    gt = target > 0.5
    tp = int((pred & gt).sum().detach().cpu())
    fp = int((pred & (~gt)).sum().detach().cpu())
    fn = int(((~pred) & gt).sum().detach().cpu())
    tn = int(((~pred) & (~gt)).sum().detach().cpu())
    return tp, fp, fn, tn, pred, gt


def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return iou, f1, precision, recall


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


def summarize_bucket(bucket):
    count = bucket["count"]
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        bucket["sum"] / count,
        bucket["abs_sum"] / count,
        bucket["pos"] / count,
        count,
    )


def residual_weight(surface_gt, base_prob):
    with torch.no_grad():
        raw_weight = torch.abs(surface_gt - base_prob)
        normalized = torch.ones_like(raw_weight)
        pos_mask = surface_gt > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any():
            normalized[pos_mask] = (
                raw_weight[pos_mask]
                / raw_weight[pos_mask].mean().clamp_min(1e-6)
            )
        if neg_mask.any():
            normalized[neg_mask] = (
                raw_weight[neg_mask]
                / raw_weight[neg_mask].mean().clamp_min(1e-6)
            )
    return normalized


def prepare_model_input(images):
    if images.size(1) == 1:
        return images.repeat(1, 3, 1, 1)
    return images


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train only a tiny structure-conditioned surface residual correction head."
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./model_out")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--train_crop_list", type=str, default="")
    parser.add_argument("--val_crop_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--surface_pos_weight", type=float, default=0.0)
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
    if "structure_profile" in saved_args and not cli_has("--structure_profile"):
        args.structure_profile = saved_args["structure_profile"]


def make_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() if args.run_name.strip() else f"structure_residual_corr_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    if not os.path.exists(run_dir):
        return run_dir
    suffix = 2
    while True:
        candidate = f"{run_dir}_{suffix}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def build_loaders(args):
    train_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="train",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.train_crop_list,
    )
    val_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="val",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.val_crop_list,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_frozen_model(args, checkpoint, device):
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
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def forward_base_and_z(model, images):
    model_input = prepare_model_input(images)
    with torch.no_grad():
        outputs = model(images)
        if not isinstance(outputs, tuple):
            raise RuntimeError("Structure residual correction requires auxiliary outputs.")
        base_surface_logits = outputs[0].detach()
        z_struct = model.swin_unet.prepatch_structure_encoder(model_input).detach()
    return base_surface_logits, z_struct


def run_validation(model, correction_head, loader, device, args):
    correction_head.eval()
    total_loss = 0.0
    total_batches = 0
    base_tp = base_fp = base_fn = 0
    corr_tp = corr_fp = corr_fn = 0
    rescue_num = rescue_den = 0
    damage_num = damage_den = 0
    removed_fp_num = removed_fp_den = 0
    added_fp_num = added_fp_den = 0
    buckets = {
        "All": empty_bucket(),
        "Weak Skeleton FN": empty_bucket(),
        "Skeleton TP": empty_bucket(),
        "Background": empty_bucket(),
    }
    pos_weight = None
    if args.surface_pos_weight > 0:
        pos_weight = torch.tensor(args.surface_pos_weight, device=device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            skeletons = batch["skeleton"].to(device)
            base_logits, z_struct = forward_base_and_z(model, images)
            masks = match_spatial_size(masks, base_logits, mode="nearest")
            skeletons = match_spatial_size(skeletons, base_logits, mode="nearest")
            base_prob = torch.sigmoid(base_logits)
            raw_delta = correction_head(z_struct, base_prob.detach())
            corrected_logits = base_logits.detach() - raw_delta
            effective_delta = corrected_logits - base_logits.detach()

            weight = residual_weight(masks, base_prob.detach())
            loss_map = F.binary_cross_entropy_with_logits(
                corrected_logits,
                masks,
                reduction="none",
                pos_weight=pos_weight,
            )
            loss = (loss_map * weight).mean()
            total_loss += float(loss.detach().cpu())
            total_batches += 1

            btp, bfp, bfn, _, base_pred, gt = binary_counts(
                base_logits,
                masks,
                args.threshold,
            )
            ctp, cfp, cfn, _, corr_pred, _ = binary_counts(
                corrected_logits,
                masks,
                args.threshold,
            )
            base_tp += btp
            base_fp += bfp
            base_fn += bfn
            corr_tp += ctp
            corr_fp += cfp
            corr_fn += cfn

            skeleton_bin = skeletons > 0.5
            weak_skeleton_fn = skeleton_bin & gt & (~base_pred)
            skeleton_tp = skeleton_bin & gt & base_pred
            background = ~gt
            update_bucket(buckets["All"], effective_delta.reshape(-1))
            update_bucket(buckets["Weak Skeleton FN"], effective_delta[weak_skeleton_fn])
            update_bucket(buckets["Skeleton TP"], effective_delta[skeleton_tp])
            update_bucket(buckets["Background"], effective_delta[background])

            base_fn_mask = gt & (~base_pred)
            base_tp_mask = gt & base_pred
            base_fp_mask = (~gt) & base_pred
            base_tn_mask = (~gt) & (~base_pred)
            rescue_num += int((base_fn_mask & corr_pred).sum().cpu())
            rescue_den += int(base_fn_mask.sum().cpu())
            damage_num += int((base_tp_mask & (~corr_pred)).sum().cpu())
            damage_den += int(base_tp_mask.sum().cpu())
            removed_fp_num += int((base_fp_mask & (~corr_pred)).sum().cpu())
            removed_fp_den += int(base_fp_mask.sum().cpu())
            added_fp_num += int((base_tn_mask & corr_pred).sum().cpu())
            added_fp_den += int(base_tn_mask.sum().cpu())

    base_metrics = metrics_from_counts(base_tp, base_fp, base_fn)
    corr_metrics = metrics_from_counts(corr_tp, corr_fp, corr_fn)
    effect = {
        "rescue_rate": rescue_num / rescue_den if rescue_den else 0.0,
        "damage_rate": damage_num / damage_den if damage_den else 0.0,
        "removed_fp_rate": removed_fp_num / removed_fp_den if removed_fp_den else 0.0,
        "added_fp_rate": added_fp_num / added_fp_den if added_fp_den else 0.0,
        "rescued_fn": rescue_num,
        "base_fn": rescue_den,
        "damaged_tp": damage_num,
        "base_tp": damage_den,
        "removed_fp": removed_fp_num,
        "base_fp": removed_fp_den,
        "added_fp": added_fp_num,
        "base_tn": added_fp_den,
    }
    delta_stats = {name: summarize_bucket(bucket) for name, bucket in buckets.items()}
    return {
        "loss": total_loss / max(total_batches, 1),
        "base_metrics": base_metrics,
        "corr_metrics": corr_metrics,
        "effect": effect,
        "delta_stats": delta_stats,
    }


def print_validation(epoch, stats):
    base_iou, base_f1, base_precision, base_recall = stats["base_metrics"]
    corr_iou, corr_f1, corr_precision, corr_recall = stats["corr_metrics"]
    print(
        f"[Epoch {epoch}] val_loss={stats['loss']:.6f} | "
        f"Base IoU/F1/P/R={base_iou:.6f}/{base_f1:.6f}/{base_precision:.6f}/{base_recall:.6f} | "
        f"Corrected IoU/F1/P/R={corr_iou:.6f}/{corr_f1:.6f}/{corr_precision:.6f}/{corr_recall:.6f}",
        flush=True,
    )
    print("Delta surface logits, effective sign = corrected - base", flush=True)
    print("{:<18} {:>12} {:>12} {:>12} {:>12}".format(
        "Region",
        "mean(delta)",
        "mean(abs)",
        "pos_ratio",
        "pixels",
    ))
    for name in ("All", "Weak Skeleton FN", "Skeleton TP", "Background"):
        mean_delta, mean_abs, pos_ratio, count = stats["delta_stats"][name]
        print("{:<18} {:>12.6f} {:>12.6f} {:>12.4f} {:>12}".format(
            name,
            mean_delta,
            mean_abs,
            pos_ratio,
            count,
        ))
    effect = stats["effect"]
    print(
        "Effectiveness: "
        f"Rescue Rate={effect['rescue_rate']:.6f} ({effect['rescued_fn']}/{effect['base_fn']}), "
        f"Damage Rate={effect['damage_rate']:.6f} ({effect['damaged_tp']}/{effect['base_tp']}), "
        f"Removed FP={effect['removed_fp_rate']:.6f} ({effect['removed_fp']}/{effect['base_fp']}), "
        f"Added FP={effect['added_fp_rate']:.6f} ({effect['added_fp']}/{effect['base_tn']})",
        flush=True,
    )


def train_one_epoch(model, correction_head, loader, optimizer, device, args, epoch):
    correction_head.train()
    total_loss = 0.0
    total_batches = 0
    pos_weight = None
    if args.surface_pos_weight > 0:
        pos_weight = torch.tensor(args.surface_pos_weight, device=device)

    progress = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        base_logits, z_struct = forward_base_and_z(model, images)
        masks = match_spatial_size(masks, base_logits, mode="nearest")
        base_prob = torch.sigmoid(base_logits.detach())
        raw_delta = correction_head(z_struct, base_prob)
        corrected_logits = base_logits.detach() - raw_delta
        weight = residual_weight(masks, base_prob)
        loss_map = F.binary_cross_entropy_with_logits(
            corrected_logits,
            masks,
            reduction="none",
            pos_weight=pos_weight,
        )
        loss = (loss_map * weight).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1
        progress.set_postfix(loss=f"{total_loss / total_batches:.5f}")
    return total_loss / max(total_batches, 1)


def save_checkpoint(path, correction_head, optimizer, args, epoch, stats):
    torch.save(
        {
            "epoch": epoch,
            "correction_head_state_dict": correction_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "stats": stats,
            "correction_formula": "corrected_surface_logits = base_surface_logits.detach() - raw_delta",
            "reported_delta_formula": "reported_delta = corrected_surface_logits - base_surface_logits",
        },
        path,
    )


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    inherit_checkpoint_args(args, checkpoint)
    args.enable_highres_structure_stream = True
    args.highres_structure_fuse_stages = "stage23"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Base checkpoint: {args.model_path}")
    print(f"Output dir: {run_dir}")
    print("Base model is frozen; optimizer updates only StructureConditionedCorrectionHead.")

    model = build_frozen_model(args, checkpoint, device)
    print_topology_coefficients(model)
    correction_head = StructureConditionedCorrectionHead(
        struct_channels=args.highres_structure_channels,
    ).to(device)
    trainable = sum(p.numel() for p in correction_head.parameters() if p.requires_grad)
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Correction head trainable params: {trainable}")
    print(f"Frozen base trainable params: {frozen_trainable}")

    train_loader, val_loader = build_loaders(args)
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Val set size: {len(val_loader.dataset)}")
    optimizer = torch.optim.AdamW(
        correction_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    csv_path = os.path.join(run_dir, "correction_diagnostics.csv")
    best_iou = -1.0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "base_iou",
            "base_f1",
            "corr_iou",
            "corr_f1",
            "weak_fn_delta",
            "skeleton_tp_delta",
            "background_delta",
            "rescue_rate",
            "damage_rate",
            "removed_fp_rate",
            "added_fp_rate",
        ])
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(
                model,
                correction_head,
                train_loader,
                optimizer,
                device,
                args,
                epoch,
            )
            stats = run_validation(model, correction_head, val_loader, device, args)
            print_validation(epoch, stats)

            base_iou, base_f1, _, _ = stats["base_metrics"]
            corr_iou, corr_f1, _, _ = stats["corr_metrics"]
            weak_delta = stats["delta_stats"]["Weak Skeleton FN"][0]
            skel_tp_delta = stats["delta_stats"]["Skeleton TP"][0]
            bg_delta = stats["delta_stats"]["Background"][0]
            effect = stats["effect"]
            writer.writerow([
                epoch,
                f"{train_loss:.8f}",
                f"{stats['loss']:.8f}",
                f"{base_iou:.8f}",
                f"{base_f1:.8f}",
                f"{corr_iou:.8f}",
                f"{corr_f1:.8f}",
                f"{weak_delta:.8f}",
                f"{skel_tp_delta:.8f}",
                f"{bg_delta:.8f}",
                f"{effect['rescue_rate']:.8f}",
                f"{effect['damage_rate']:.8f}",
                f"{effect['removed_fp_rate']:.8f}",
                f"{effect['added_fp_rate']:.8f}",
            ])
            csv_file.flush()

            save_checkpoint(
                os.path.join(run_dir, "last_correction_head.pth"),
                correction_head,
                optimizer,
                args,
                epoch,
                stats,
            )
            if corr_iou > best_iou:
                best_iou = corr_iou
                save_checkpoint(
                    os.path.join(run_dir, "best_correction_head.pth"),
                    correction_head,
                    optimizer,
                    args,
                    epoch,
                    stats,
                )
                print(f"[BEST] correction head saved, corrected IoU={corr_iou:.6f}", flush=True)

    print(f"Done. Diagnostics CSV: {csv_path}")


if __name__ == "__main__":
    main()
