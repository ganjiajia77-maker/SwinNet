import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--name", type=str, default="checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--force_reliability_beta_eff", type=float, default=0.05166)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
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
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=["auto", "standard", "selective"])
    return parser.parse_args()


def reliability_modules(model):
    modules = []
    for name, module in model.named_modules():
        if (
            hasattr(module, "reliability_correction")
            and hasattr(module, "context_out")
            and hasattr(module, "reliability_beta_eff")
        ):
            modules.append((name, module))
    return modules


def set_context_zero(modules):
    saved = []
    for _, module in modules:
        weight = module.context_out.weight
        bias = module.context_out.bias
        saved.append((weight, weight.detach().clone(), bias, None if bias is None else bias.detach().clone()))
        weight.data.zero_()
        if bias is not None:
            bias.data.zero_()
    return saved


def restore_context(saved):
    for weight, old_weight, bias, old_bias in saved:
        weight.data.copy_(old_weight)
        if bias is not None and old_bias is not None:
            bias.data.copy_(old_bias)


def set_beta_eff(modules, beta_eff):
    saved = []
    for _, module in modules:
        param = module.reliability_beta
        saved.append((param, param.detach().clone()))
        beta_max = float(module.reliability_beta_max)
        if beta_eff <= 0.0:
            raw = -30.0
        elif beta_eff >= beta_max:
            raw = 30.0
        else:
            ratio = float(beta_eff) / beta_max
            raw = float(np.log(ratio / (1.0 - ratio)))
        param.data.fill_(raw)
    return saved


def restore_beta(saved):
    for param, old_value in saved:
        param.data.copy_(old_value)


def metrics(prob, gt, threshold):
    pred = prob >= float(threshold)
    gt = gt.bool()
    tp = torch.logical_and(pred, gt).sum().item()
    fp = torch.logical_and(pred, ~gt).sum().item()
    fn = torch.logical_and(~pred, gt).sum().item()
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return iou, f1, precision, recall, pred


def capture_pass(model, loader, device, modules, beta_eff, threshold, max_batches, desc):
    captures = {name: {"r_old": [], "gate": []} for name, _ in modules}
    handles = []
    for name, module in modules:
        def make_r_hook(key):
            def hook(_module, _inputs, output):
                captures[key]["r_old"].append(output.detach().float().cpu())
            return hook

        def make_gate_hook(key):
            def hook(_module, _inputs, output):
                if isinstance(output, tuple) and len(output) >= 5:
                    captures[key]["gate"].append(output[4].detach().float().cpu())
            return hook

        handles.append(module.reliability_correction.register_forward_hook(make_r_hook(name)))
        handles.append(module.register_forward_hook(make_gate_hook(name)))

    beta_saved = set_beta_eff(modules, beta_eff)
    surface_probs = []
    masks = []
    images = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            image = batch["image"].to(device)
            mask = batch["mask"].to(device) > 0.5
            output = model(image, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            prob = torch.sigmoid(output[0])
            if mask.shape[-2:] != prob.shape[-2:]:
                mask = F.interpolate(mask.float(), size=prob.shape[-2:], mode="nearest") > 0.5
            surface_probs.append(prob.detach().float().cpu())
            masks.append(mask.detach().bool().cpu())
            images += int(image.shape[0])
    restore_beta(beta_saved)
    for handle in handles:
        handle.remove()

    surface_prob = torch.cat(surface_probs, dim=0)
    mask = torch.cat(masks, dim=0)
    iou, f1, precision, recall, pred = metrics(surface_prob, mask, threshold)
    packed = {}
    for name, values in captures.items():
        if values["gate"]:
            packed[name] = {
                "r_old": torch.cat(values["r_old"], dim=0),
                "gate": torch.cat(values["gate"], dim=0),
            }
    return {
        "surface_prob": surface_prob,
        "mask": mask,
        "pred": pred,
        "metrics": (iou, f1, precision, recall),
        "captures": packed,
        "images": images,
    }


def masked_mean_std(values, mask):
    if values.numel() == 0 or not mask.any():
        return float("nan"), float("nan")
    selected = values[mask]
    return float(selected.mean().item()), float(selected.std(unbiased=False).item())


def print_region_stats(label, tensor, regions):
    all_mean, all_std = masked_mean_std(tensor, torch.ones_like(tensor, dtype=torch.bool))
    print(f"    {label} mean={all_mean:+.6f}, std={all_std:.6f}")
    for region_name, region_mask in regions:
        mean, std = masked_mean_std(tensor, region_mask)
        print(f"    {label} {region_name:<2} mean={mean:+.6f}, std={std:.6f}, count={int(region_mask.sum().item())}")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = load_model(args, device)
    modules = reliability_modules(model)
    if not modules:
        raise RuntimeError("No decoder reliability modules were found.")
    context_saved = set_context_zero(modules)

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
        pin_memory=True,
    )

    gs = capture_pass(
        model,
        loader,
        device,
        modules,
        beta_eff=0.0,
        threshold=args.threshold,
        max_batches=args.max_batches,
        desc=f"{args.name} Gs beta=0 Cctx=0",
    )
    forced = capture_pass(
        model,
        loader,
        device,
        modules,
        beta_eff=args.force_reliability_beta_eff,
        threshold=args.threshold,
        max_batches=args.max_batches,
        desc=f"{args.name} Gfinal beta={args.force_reliability_beta_eff:g} Cctx=0",
    )
    restore_context(context_saved)

    gs_iou, gs_f1, gs_p, gs_r = gs["metrics"]
    forced_iou, forced_f1, forced_p, forced_r = forced["metrics"]
    print(f"\nCheckpoint: {args.model_path}")
    print(
        f"name={args.name}, split={args.split}, threshold={args.threshold}, "
        f"forced_beta_eff={args.force_reliability_beta_eff}, Cctx=0, images={forced['images']}"
    )
    print("\nSurface")
    print(f"  Gs beta=0:      IoU={gs_iou:.4f}, F1={gs_f1:.4f}, P={gs_p:.4f}, R={gs_r:.4f}")
    print(f"  Gfinal forced:  IoU={forced_iou:.4f}, F1={forced_f1:.4f}, P={forced_p:.4f}, R={forced_r:.4f}")
    print(
        "  Delta forced-Gs: "
        f"IoU={forced_iou - gs_iou:+.4f}, F1={forced_f1 - gs_f1:+.4f}, "
        f"P={forced_p - gs_p:+.4f}, R={forced_r - gs_r:+.4f}"
    )

    gt = forced["mask"]
    pred = forced["pred"]
    surface_regions = [
        ("TP", pred & gt),
        ("TN", (~pred) & (~gt)),
        ("FP", pred & (~gt)),
        ("FN", (~pred) & gt),
    ]

    print("\nR_old and DeltaG_R Regions")
    for name, forced_values in forced["captures"].items():
        if name not in gs["captures"]:
            continue
        r_old = forced_values["r_old"]
        g_final = forced_values["gate"]
        g_s = gs["captures"][name]["gate"]
        delta_g = g_final - g_s
        regions = []
        for region_name, surface_mask in surface_regions:
            mask = F.interpolate(
                surface_mask.float(),
                size=r_old.shape[-2:],
                mode="nearest",
            ).bool()
            regions.append((region_name, mask.expand_as(r_old)))
        print(f"  {name}")
        print_region_stats("R_old", r_old, regions)
        print_region_stats("DeltaG_R", delta_g, regions)


if __name__ == "__main__":
    main()
