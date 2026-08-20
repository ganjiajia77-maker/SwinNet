import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import BCEDiceLoss
from networks.vision_transformer_selective_fusion import (
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


REGIONS = ("WeakFN", "SkeletonTP", "HardBG")


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ReZeroSelectiveFusion(nn.Module):
    def __init__(self, channels=48):
        super().__init__()
        self.gate_body = nn.Sequential(
            nn.Conv2d(channels * 2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate_body[-1].weight)
        nn.init.constant_(self.gate_body[-1].bias, -2.0)
        self.rho = nn.Parameter(torch.tensor(0.0))
        self.last_gate = None
        self.last_effective_gate = None

    def forward(self, f_proj, f_ref):
        gate = torch.sigmoid(self.gate_body(torch.cat([f_proj, f_ref], dim=1)))
        effective_gate = self.rho * gate
        self.last_gate = gate
        self.last_effective_gate = effective_gate
        return f_ref + effective_gate * (f_proj - f_ref)


def parse_outputs(outputs):
    if not isinstance(outputs, tuple):
        return outputs
    return outputs[0]


def resize_target(target, logits):
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    return target


def apply_checkpoint_args(args, checkpoint):
    saved = checkpoint.get("args", {})
    if not isinstance(saved, dict):
        return
    for name in (
        "structure_profile",
        "disable_msfe_skip",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "highres_structure_fusion_mode",
        "enable_post_refine_structure_interaction",
        "img_size",
        "source_patch_size",
        "direct_resize_train",
        "overlap_stride",
        "batch_size",
        "seed",
    ):
        if name in saved and getattr(args, name, None) == parser_defaults()[name]:
            setattr(args, name, saved[name])


def parser_defaults():
    return {
        "structure_profile": "full",
        "disable_msfe_skip": True,
        "enable_highres_structure_stream": False,
        "highres_structure_channels": 64,
        "highres_structure_fuse_stages": "stage23",
        "highres_structure_fusion_mode": "stage23",
        "enable_post_refine_structure_interaction": False,
        "img_size": 256,
        "source_patch_size": 1024,
        "direct_resize_train": True,
        "overlap_stride": 256,
        "batch_size": 4,
        "seed": 1234,
    }


def get_guided_head(model):
    module = model.module if hasattr(model, "module") else model
    return module.swin_unet.guided_head


def get_selector(model):
    return get_guided_head(model).surface_selective_fusion


def build_model(args, checkpoint):
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
    state_dict = dict(checkpoint.get("model_state_dict", checkpoint))
    model_state = model.state_dict()
    selector_prefix = "swin_unet.guided_head.surface_selective_fusion."
    for key, value in model_state.items():
        if key.startswith(selector_prefix) and key not in state_dict:
            state_dict[key] = value
    load_topology_checkpoint_state(
        model,
        state_dict,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    get_guided_head(model).surface_selective_fusion = ReZeroSelectiveFusion(channels=48)
    return model.to(args.device).eval()


def freeze_except_selector(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selector = get_selector(model)
    for parameter in selector.parameters():
        parameter.requires_grad_(True)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    print("Trainable parameters:", flush=True)
    for name, count in trainable:
        print(f"  {name}: {count}", flush=True)
    total = sum(count for _name, count in trainable)
    print(f"Trainable parameter count: {total}", flush=True)
    invalid = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "surface_selective_fusion" not in name
    ]
    if invalid:
        raise RuntimeError(f"Unexpected trainable params: {invalid}")
    return selector


def make_dataloaders(args):
    train_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="train",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        tile_size=None if args.direct_resize_train else args.img_size,
        tile_stride=args.overlap_stride,
        augment=args.augment,
        random_crop_train=False,
    )
    test_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="test",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, test_loader


def collect_batches(loader, max_batches):
    batches = []
    for idx, batch in enumerate(loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batches.append(batch)
    return batches


def run_forward(model, images):
    return parse_outputs(model(images))


def run_baseline_refine_forward(model, images):
    selector = get_selector(model)
    original_forward = selector.forward

    def refine_only(_f_proj, f_ref):
        return f_ref

    selector.forward = refine_only
    try:
        logits = run_forward(model, images)
    finally:
        selector.forward = original_forward
    return logits


def verify_rezero_equivalence(args, model, batches):
    model.eval()
    max_abs = 0.0
    with torch.no_grad():
        for batch in batches:
            images = batch["image"].to(args.device)
            logits_new = run_forward(model, images)
            logits_base = run_baseline_refine_forward(model, images)
            diff = (logits_new - logits_base).abs().max().item()
            max_abs = max(max_abs, diff)
    print(f"ReZero equivalence max_abs(logits_new - logits_refine_only): {max_abs:.10e}")
    if max_abs >= args.reproduce_tol:
        raise RuntimeError(
            f"ReZero equivalence failed: {max_abs:.10e} >= {args.reproduce_tol:.10e}"
        )


def metric_counts_from_logits(logits, target, threshold):
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()
    target = (target > 0.5).float()
    tp = (pred * target).sum().double()
    fp = (pred * (1.0 - target)).sum().double()
    fn = ((1.0 - pred) * target).sum().double()
    return tp, fp, fn


def evaluate_metrics(args, model, batches):
    model.eval()
    tp = torch.tensor(0.0, dtype=torch.float64)
    fp = torch.tensor(0.0, dtype=torch.float64)
    fn = torch.tensor(0.0, dtype=torch.float64)
    with torch.no_grad():
        for batch in batches:
            images = batch["image"].to(args.device)
            target = batch["mask"].float().to(args.device)
            logits = run_forward(model, images)
            target = resize_target(target, logits)
            btp, bfp, bfn = metric_counts_from_logits(logits.detach().cpu(), target.detach().cpu(), args.threshold)
            tp += btp
            fp += bfp
            fn += bfn
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return {
        "iou": float(iou.item()),
        "f1": float(f1.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }


def build_fixed_masks(args, model, batches):
    fixed = []
    counts = {name: 0 for name in REGIONS}
    model.eval()
    with torch.no_grad():
        for batch in batches:
            images = batch["image"].to(args.device)
            surface_gt = batch["mask"].float()
            skeleton_gt = batch["skeleton"].float()
            logits = run_baseline_refine_forward(model, images).detach().cpu()
            surface_gt = resize_target(surface_gt, logits)
            skeleton_gt = resize_target(skeleton_gt, logits)
            prob = torch.sigmoid(logits)
            masks = {
                "WeakFN": ((skeleton_gt == 1) & (prob < args.diagnostic_threshold)).squeeze(1).cpu(),
                "SkeletonTP": ((skeleton_gt == 1) & (prob >= args.diagnostic_threshold)).squeeze(1).cpu(),
                "HardBG": ((surface_gt == 0) & (prob >= args.diagnostic_threshold)).squeeze(1).cpu(),
            }
            for name in REGIONS:
                counts[name] += int(masks[name].sum().item())
            fixed.append(masks)
    print("Fixed diagnostic masks from epoch0 refine-only baseline:", flush=True)
    print(f"  WeakFN     = gt_skeleton==1 and sigmoid(baseline_logits)<{args.diagnostic_threshold}", flush=True)
    print(f"  SkeletonTP = gt_skeleton==1 and sigmoid(baseline_logits)>={args.diagnostic_threshold}", flush=True)
    print(f"  HardBG     = surface_gt==0 and sigmoid(baseline_logits)>={args.diagnostic_threshold}", flush=True)
    for name in REGIONS:
        print(f"  {name}: {counts[name]}", flush=True)
    return fixed, counts


def gate_stats(args, model, batches, fixed_masks):
    selector = get_selector(model)
    model.eval()
    gate_sum = 0.0
    gate_sq_sum = 0.0
    eff_sum = 0.0
    eff_sq_sum = 0.0
    n = 0
    region_sums = {name: 0.0 for name in REGIONS}
    region_counts = {name: 0 for name in REGIONS}
    with torch.no_grad():
        for idx, batch in enumerate(batches):
            images = batch["image"].to(args.device)
            _ = run_forward(model, images)
            gate = selector.last_gate.detach().cpu().squeeze(1)
            effective_gate = selector.last_effective_gate.detach().cpu().squeeze(1)
            gate_double = gate.double()
            eff_double = effective_gate.double()
            gate_sum += float(gate_double.sum().item())
            gate_sq_sum += float((gate_double * gate_double).sum().item())
            eff_sum += float(eff_double.sum().item())
            eff_sq_sum += float((eff_double * eff_double).sum().item())
            n += gate.numel()
            for name in REGIONS:
                mask = fixed_masks[idx][name]
                if mask.any():
                    region_sums[name] += float(eff_double[mask].sum().item())
                    region_counts[name] += int(mask.sum().item())
    gate_mean = gate_sum / max(n, 1)
    eff_mean = eff_sum / max(n, 1)
    gate_std = max(gate_sq_sum / max(n, 1) - gate_mean * gate_mean, 0.0) ** 0.5
    eff_std = max(eff_sq_sum / max(n, 1) - eff_mean * eff_mean, 0.0) ** 0.5
    stats = {
        "rho": float(selector.rho.detach().cpu().item()),
        "A_all_mean": gate_mean,
        "A_all_std": gate_std,
        "effective_gate_all_mean": eff_mean,
        "effective_gate_all_std": eff_std,
    }
    for name in REGIONS:
        stats[f"effective_gate_{name}_mean"] = (
            region_sums[name] / region_counts[name]
            if region_counts[name] > 0
            else float("nan")
        )
    return stats


def train_one_epoch(args, model, train_loader, criterion, optimizer, epoch):
    model.eval()
    selector = get_selector(model)
    selector.train()
    total_loss = 0.0
    batches = 0
    for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
        if args.max_train_batches > 0 and batches >= args.max_train_batches:
            break
        images = batch["image"].to(args.device)
        target = batch["mask"].float().to(args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = run_forward(model, images)
        target = resize_target(target, logits)
        loss, _bce, _dice = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1), batches


def save_selector_checkpoint(args, model, epoch, row):
    if not args.output_dir:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"rezero_selector_epoch{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "selector_state_dict": get_selector(model).state_dict(),
            "metrics": row,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_out/selective_fusion_rezero_probe")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=100)
    parser.add_argument("--max_eval_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--diagnostic_threshold", type=float, default=0.45)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--reproduce_tol", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct_resize_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap_stride", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
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
    parser.add_argument("--structure_profile", type=str, default="full")
    parser.add_argument("--disable_msfe_skip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    print(
        "Checkpoint summary: "
        f"epoch={checkpoint.get('epoch')} "
        f"val_iou={checkpoint.get('val_iou')} "
        f"val_f1={checkpoint.get('val_f1')} "
        f"val_precision={checkpoint.get('val_precision')} "
        f"val_recall={checkpoint.get('val_recall')}",
        flush=True,
    )
    apply_checkpoint_args(args, checkpoint)
    set_deterministic(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        "Probe config: "
        f"img_size={args.img_size}, source_patch_size={args.source_patch_size}, "
        f"direct_resize_train={args.direct_resize_train}, threshold={args.threshold}, "
        f"max_train_batches={args.max_train_batches}, max_eval_batches={args.max_eval_batches}, "
        f"lr={args.lr}",
        flush=True,
    )

    train_loader, test_loader = make_dataloaders(args)
    eval_batches = collect_batches(test_loader, args.max_eval_batches)
    model = build_model(args, checkpoint)
    selector = freeze_except_selector(model)
    verify_rezero_equivalence(args, model, eval_batches)
    fixed_masks, _counts = build_fixed_masks(args, model, eval_batches)

    criterion = BCEDiceLoss(dice_weight=0.5, bce_weight=1.0).to(args.device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    fieldnames = [
        "epoch",
        "train_loss",
        "train_batches",
        "iou",
        "f1",
        "precision",
        "recall",
        "rho",
        "A_all_mean",
        "A_all_std",
        "effective_gate_all_mean",
        "effective_gate_all_std",
        "effective_gate_WeakFN_mean",
        "effective_gate_SkeletonTP_mean",
        "effective_gate_HardBG_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        base_metrics = evaluate_metrics(args, model, eval_batches)
        base_stats = gate_stats(args, model, eval_batches, fixed_masks)
        row = {
            "epoch": 0,
            "train_loss": 0.0,
            "train_batches": 0,
            **base_metrics,
            **base_stats,
        }
        writer.writerow(row)
        handle.flush()
        print(
            "epoch=0 "
            f"IoU={base_metrics['iou']:.4f} F1={base_metrics['f1']:.4f} "
            f"P={base_metrics['precision']:.4f} R={base_metrics['recall']:.4f} "
            f"rho={base_stats['rho']:+.8f} "
            f"eff_gate_std={base_stats['effective_gate_all_std']:.8f}",
            flush=True,
        )

        for epoch in range(1, args.epochs + 1):
            loss, train_batches = train_one_epoch(args, model, train_loader, criterion, optimizer, epoch)
            metrics = evaluate_metrics(args, model, eval_batches)
            stats = gate_stats(args, model, eval_batches, fixed_masks)
            row = {
                "epoch": epoch,
                "train_loss": loss,
                "train_batches": train_batches,
                **metrics,
                **stats,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch} loss={loss:.5f} batches={train_batches} "
                f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"rho={stats['rho']:+.8f} "
                f"A_mean={stats['A_all_mean']:.6f} A_std={stats['A_all_std']:.6f} "
                f"eff_mean={stats['effective_gate_all_mean']:+.8f} "
                f"eff_std={stats['effective_gate_all_std']:.8f} "
                f"eff_WeakFN={stats['effective_gate_WeakFN_mean']:+.8f} "
                f"eff_SkeletonTP={stats['effective_gate_SkeletonTP_mean']:+.8f} "
                f"eff_HardBG={stats['effective_gate_HardBG_mean']:+.8f}",
                flush=True,
            )
            save_selector_checkpoint(args, model, epoch, row)
    print(f"Saved metrics: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
