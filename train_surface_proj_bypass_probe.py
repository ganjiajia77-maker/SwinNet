import argparse
import csv
import os
import random
import sys
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_surface_refine_identity_sweep import PrePatchLiteStructureEncoder
from losses.road_losses import BCEDiceLoss, binary_metrics_from_logits
from networks.swin_transformer_unet_skip_expand_decoder_sys import map_to_token, token_to_map
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


REGIONS = ("WeakFN", "SkeletonTP", "HardBG")
EXPECTED_COUNTS = {"WeakFN": 7783, "SkeletonTP": 32204, "HardBG": 103760}


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def largest_group_divisor(channels, candidates=(8, 4, 2, 1)):
    for groups in candidates:
        if channels % groups == 0:
            return groups
    return 1


class SurfaceProjBypass(nn.Module):
    def __init__(self, channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(largest_group_divisor(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def patch_model_with_bypass(model):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    swin.prepatch_structure_encoder = PrePatchLiteStructureEncoder(swin.highres_structure_channels)
    swin.highres_structure_source = "prepatch-lite-surface-proj-bypass-probe"
    swin._diagnostic_mode = "normal"
    swin.guided_head._diagnostic_trace = {}
    swin.guided_head.surface_proj_bypass = SurfaceProjBypass(channels=48)
    swin.guided_head._surface_proj_bypass_enabled = True

    def apply_fusion(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        z_for_surface = z_struct.detach()
        if getattr(self, "_diagnostic_mode", "normal") == "zero_both":
            z_for_surface = torch.zeros_like(z_for_surface)
        feature_map = self.highres_structure_fusion[str(stage)](feature_map, z_for_surface)
        return map_to_token(feature_map)

    def maybe_apply_bypass(self, surface_proj, guided_surface_feat):
        if bool(getattr(self, "_surface_proj_bypass_enabled", True)):
            guided_surface_feat = guided_surface_feat + self.surface_proj_bypass(surface_proj)
        return guided_surface_feat

    def guided_forward(self, x, z_struct=None):
        surface_proj = self.surface_proj(x)
        self._diagnostic_trace["surface_proj"] = surface_proj.detach().cpu()
        surface_feat = self.surface_branch(surface_proj)
        skeleton_feat = self.skeleton_proj(x)
        z_for_surface = z_struct
        if (
            z_for_surface is not None
            and getattr(swin, "_diagnostic_mode", "normal") == "zero_both"
        ):
            z_for_surface = torch.zeros_like(z_for_surface)

        if not self.enable_final_structure:
            final_structure_feat = self.structure_branch(skeleton_feat)
            final_skeleton_logits = self.skeleton_head(final_structure_feat)
            guided_surface_feat = self.surface_refine(surface_feat)
            guided_surface_feat = maybe_apply_bypass(self, surface_proj, guided_surface_feat)
            self._diagnostic_trace["surface_refine_bypass"] = guided_surface_feat.detach().cpu()
            guided_surface_feat = self._apply_post_refine_structure_interaction(guided_surface_feat, z_for_surface)
            boundary_feat = self.boundary_branch(guided_surface_feat)
            boundary_logits = self.boundary_head(boundary_feat)
            boundary_attn = torch.sigmoid(boundary_logits)
            boundary_correction = self.boundary_residual(guided_surface_feat)
            guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
            surface_logits = self.surface_head(guided_surface_feat)
            self._diagnostic_trace["surface_logits"] = surface_logits.detach().cpu()
            return surface_logits, boundary_logits, final_skeleton_logits, None

        structure_feat = self.structure_branch(skeleton_feat)
        seed_skeleton_logits = self.skeleton_head(structure_feat)
        seed_connectivity_logits = self.connectivity_head(structure_feat)
        structure_feat = self.final_topology_attention(
            structure_feat,
            torch.sigmoid(seed_skeleton_logits).detach(),
            torch.sigmoid(seed_connectivity_logits).detach(),
        )
        skeleton_logits = self.skeleton_head(structure_feat)
        connectivity_logits = self.connectivity_head(structure_feat)
        guided_surface_feat = self.surface_refine(surface_feat)
        guided_surface_feat = maybe_apply_bypass(self, surface_proj, guided_surface_feat)
        self._diagnostic_trace["surface_refine_bypass"] = guided_surface_feat.detach().cpu()
        guided_surface_feat = self._apply_post_refine_structure_interaction(guided_surface_feat, z_for_surface)
        boundary_feat = self.boundary_branch(guided_surface_feat)
        boundary_logits = self.boundary_head(boundary_feat)
        boundary_attn = torch.sigmoid(boundary_logits)
        boundary_correction = self.boundary_residual(guided_surface_feat)
        guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction
        surface_logits = self.surface_head(guided_surface_feat)
        self._diagnostic_trace["surface_logits"] = surface_logits.detach().cpu()
        return surface_logits, boundary_logits, skeleton_logits, connectivity_logits

    swin._apply_highres_structure_fusion = MethodType(apply_fusion, swin)
    swin.guided_head.forward = MethodType(guided_forward, swin.guided_head)


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
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
    )
    patch_model_with_bypass(model)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def apply_checkpoint_args(args, checkpoint):
    if not isinstance(checkpoint.get("args"), dict):
        return
    saved_args = checkpoint["args"]
    args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
    args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
    args.enable_highres_structure_stream = bool(saved_args.get("enable_highres_structure_stream", args.enable_highres_structure_stream))
    args.highres_structure_channels = int(saved_args.get("highres_structure_channels", args.highres_structure_channels))
    args.highres_structure_fuse_stages = str(saved_args.get("highres_structure_fuse_stages", args.highres_structure_fuse_stages))
    args.highres_structure_fusion_mode = str(saved_args.get("highres_structure_fusion_mode", args.highres_structure_fusion_mode))
    args.enable_post_refine_structure_interaction = bool(
        saved_args.get("enable_post_refine_structure_interaction", args.enable_post_refine_structure_interaction)
        or args.enable_post_refine_structure_interaction
    )
    args.img_size = int(saved_args.get("img_size", args.img_size))
    args.source_patch_size = int(saved_args.get("source_patch_size", args.source_patch_size))
    args.direct_resize_train = bool(saved_args.get("direct_resize_train", args.direct_resize_train))
    args.overlap_stride = int(saved_args.get("overlap_stride", args.overlap_stride))


def set_bypass_enabled(model, enabled):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    swin.guided_head._surface_proj_bypass_enabled = bool(enabled)


def get_bypass(model):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    return swin.guided_head.surface_proj_bypass


def freeze_except_bypass(model):
    for param in model.parameters():
        param.requires_grad_(False)
    bypass = get_bypass(model)
    for param in bypass.parameters():
        param.requires_grad_(True)
    return bypass


def parse_outputs(outputs):
    if not isinstance(outputs, tuple):
        raise RuntimeError("Expected structure model tuple output.")
    return outputs[0]


def resize_target(target, logits):
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    return target


def run_forward(model, images, mode="normal", bypass=True):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    swin._diagnostic_mode = mode
    swin.guided_head._diagnostic_trace = {}
    set_bypass_enabled(model, bypass)
    outputs = model(images)
    return parse_outputs(outputs)


def collect_batches(loader, max_batches):
    batches = []
    for idx, batch in enumerate(loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batches.append(batch)
    return batches


def verify_zero_init(args, model, batches):
    max_abs = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(batches, desc="zero-init reproduce check"):
            images = batch["image"].to(args.device)
            logits_old = run_forward(model, images, mode="normal", bypass=False)
            logits_new = run_forward(model, images, mode="normal", bypass=True)
            max_abs = max(max_abs, float((logits_new - logits_old).abs().max().item()))
    print(f"zero-init max_abs(logits_new - logits_old) = {max_abs:.12e}")
    if max_abs >= args.reproduce_tol:
        raise RuntimeError("Zero-init bypass does not reproduce the old checkpoint.")


def build_fixed_masks(args, model, batches):
    counts = {name: 0 for name in REGIONS}
    fixed_masks = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(batches, desc="build fixed masks"):
            images = batch["image"].to(args.device)
            logits = run_forward(model, images, mode="normal", bypass=False).detach().cpu()
            prob = torch.sigmoid(logits).squeeze(1)
            surface_gt = resize_target(batch["mask"].float(), logits).squeeze(1) > 0.5
            skeleton = resize_target(batch["skeleton"].float(), logits).squeeze(1) > 0.5
            masks = {
                "WeakFN": skeleton & (prob < args.threshold),
                "SkeletonTP": skeleton & (prob >= args.threshold),
                "HardBG": (~surface_gt) & (prob >= args.threshold),
            }
            fixed_masks.append(masks)
            for name in REGIONS:
                counts[name] += int(masks[name].sum().item())
    return fixed_masks, counts


def evaluate_metrics(args, model, batches):
    metrics = {"iou": [], "f1": [], "precision": [], "recall": []}
    model.eval()
    with torch.no_grad():
        for batch in batches:
            images = batch["image"].to(args.device)
            targets = batch["mask"].float()
            logits = run_forward(model, images, mode="normal", bypass=True).detach().cpu()
            targets = resize_target(targets, logits)
            item = binary_metrics_from_logits(logits, targets, threshold=args.threshold)
            for key in metrics:
                metrics[key].append(item[key])
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def diagnostic_delta(args, model, batches, fixed_masks):
    sums_total = {name: 0.0 for name in REGIONS}
    sums_bypass = {name: 0.0 for name in REGIONS}
    counts = {name: 0 for name in REGIONS}
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batches):
            images = batch["image"].to(args.device)
            logits_normal_bypass = run_forward(model, images, mode="normal", bypass=True).detach().cpu()
            logits_zero_bypass = run_forward(model, images, mode="zero_both", bypass=True).detach().cpu()
            logits_normal_old = run_forward(model, images, mode="normal", bypass=False).detach().cpu()
            delta_total = (logits_normal_bypass - logits_zero_bypass).squeeze(1)
            delta_bypass = (logits_normal_bypass - logits_normal_old).squeeze(1)
            for name in REGIONS:
                mask = fixed_masks[idx][name]
                sums_total[name] += float(delta_total[mask].double().sum().item())
                sums_bypass[name] += float(delta_bypass[mask].double().sum().item())
                counts[name] += int(mask.sum().item())
    return {
        f"{name}_total": sums_total[name] / counts[name]
        for name in REGIONS
    } | {
        f"{name}_bypass": sums_bypass[name] / counts[name]
        for name in REGIONS
    } | {f"{name}_count": counts[name] for name in REGIONS}


def train_one_epoch(args, model, train_loader, criterion, optimizer, epoch):
    model.eval()
    bypass = get_bypass(model)
    bypass.train()
    total_loss = 0.0
    batches = 0
    for i, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}")):
        if args.max_train_batches > 0 and batches >= args.max_train_batches:
            break
        images = batch["image"].to(args.device)
        targets = batch["mask"].float().to(args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = run_forward(model, images, mode="normal", bypass=True)
        targets = resize_target(targets, logits)
        loss, _bce, _dice = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bypass.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1), batches


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


def save_checkpoint(args, model, epoch, metrics):
    if not args.output_dir:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"surface_proj_bypass_epoch{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "surface_proj_bypass": get_bypass(model).state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_out/surface_proj_bypass_probe")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=100)
    parser.add_argument("--max_eval_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--reproduce_tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct_resize_train", action="store_true")
    parser.add_argument("--overlap_stride", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none", choices=["none", "stage3", "stage23"])
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
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
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23", choices=["stage2", "stage3", "stage23"])
    parser.add_argument(
        "--highres_structure_fusion_mode",
        type=str,
        default="stage23",
        choices=["stage23", "final_correction", "stage23_final_correction", "post_refine_interaction", "none"],
    )
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    args = parser.parse_args()

    set_deterministic(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    apply_checkpoint_args(args, checkpoint)
    train_loader, test_loader = make_dataloaders(args)
    eval_batches = collect_batches(test_loader, args.max_eval_batches)
    model = build_model(args, checkpoint)
    bypass = freeze_except_bypass(model)
    trainable = sum(p.numel() for p in bypass.parameters() if p.requires_grad)
    print(f"Trainable SurfaceProjBypass params: {trainable}")
    print(f"Frozen old model params: {sum(p.numel() for p in model.parameters()) - trainable}")

    verify_zero_init(args, model, eval_batches)
    fixed_masks, mask_counts = build_fixed_masks(args, model, eval_batches)
    print("Fixed mask counts:")
    for name in REGIONS:
        print(f"  {name}: {mask_counts[name]} expected={EXPECTED_COUNTS[name]} match={mask_counts[name] == EXPECTED_COUNTS[name]}")

    criterion = BCEDiceLoss(dice_weight=0.5, bce_weight=1.0).to(args.device)
    optimizer = torch.optim.AdamW(bypass.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_batches",
                "iou",
                "f1",
                "precision",
                "recall",
                "WeakFN_total",
                "SkeletonTP_total",
                "HardBG_total",
                "WeakFN_bypass",
                "SkeletonTP_bypass",
                "HardBG_bypass",
            ],
        )
        writer.writeheader()

        base_metrics = evaluate_metrics(args, model, eval_batches)
        base_diag = diagnostic_delta(args, model, eval_batches, fixed_masks)
        print(
            "epoch=0 "
            f"IoU={base_metrics['iou']:.4f} F1={base_metrics['f1']:.4f} "
            f"P={base_metrics['precision']:.4f} R={base_metrics['recall']:.4f} "
            f"WeakFN_total={base_diag['WeakFN_total']:+.6f} "
            f"WeakFN_bypass={base_diag['WeakFN_bypass']:+.6f}"
        )
        writer.writerow(
            {
                "epoch": 0,
                "train_loss": 0.0,
                "train_batches": 0,
                **base_metrics,
                "WeakFN_total": base_diag["WeakFN_total"],
                "SkeletonTP_total": base_diag["SkeletonTP_total"],
                "HardBG_total": base_diag["HardBG_total"],
                "WeakFN_bypass": base_diag["WeakFN_bypass"],
                "SkeletonTP_bypass": base_diag["SkeletonTP_bypass"],
                "HardBG_bypass": base_diag["HardBG_bypass"],
            }
        )
        handle.flush()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_batches = train_one_epoch(args, model, train_loader, criterion, optimizer, epoch)
            metrics = evaluate_metrics(args, model, eval_batches)
            diag = diagnostic_delta(args, model, eval_batches, fixed_masks)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_batches": train_batches,
                **metrics,
                "WeakFN_total": diag["WeakFN_total"],
                "SkeletonTP_total": diag["SkeletonTP_total"],
                "HardBG_total": diag["HardBG_total"],
                "WeakFN_bypass": diag["WeakFN_bypass"],
                "SkeletonTP_bypass": diag["SkeletonTP_bypass"],
                "HardBG_bypass": diag["HardBG_bypass"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch} loss={train_loss:.4f} batches={train_batches} "
                f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"WeakFN_total={diag['WeakFN_total']:+.6f} "
                f"SkeletonTP_total={diag['SkeletonTP_total']:+.6f} "
                f"HardBG_total={diag['HardBG_total']:+.6f} "
                f"WeakFN_bypass={diag['WeakFN_bypass']:+.6f} "
                f"SkeletonTP_bypass={diag['SkeletonTP_bypass']:+.6f} "
                f"HardBG_bypass={diag['HardBG_bypass']:+.6f}",
                flush=True,
            )
            save_checkpoint(args, model, epoch, row)
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
