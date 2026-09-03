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
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--single_pass", action="store_true")
    parser.add_argument("--context_mode", type=str, default="learned", choices=["learned", "zero"])
    parser.add_argument("--force_reliability_beta_eff", type=float, default=None)
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


def metric_counts(prob, gt, threshold):
    pred = prob >= float(threshold)
    gt = gt.bool()
    tp = torch.logical_and(pred, gt).sum().item()
    fp = torch.logical_and(pred, ~gt).sum().item()
    fn = torch.logical_and(~pred, gt).sum().item()
    return tp, fp, fn, pred


def metrics_from_counts(tp, fp, fn):
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return iou, f1, precision, recall


def context_modules(model):
    modules = []
    for name, module in model.named_modules():
        if hasattr(module, "context_out") and hasattr(module, "reliability_beta_eff"):
            modules.append((name, module))
    return modules


def set_context_enabled(modules, enabled):
    saved = []
    for _, module in modules:
        weight = module.context_out.weight
        bias = module.context_out.bias
        saved.append((weight.detach().clone(), None if bias is None else bias.detach().clone()))
        if not enabled:
            weight.data.zero_()
            if bias is not None:
                bias.data.zero_()
    return saved


def restore_context(modules, saved):
    for (_, module), (weight, bias) in zip(modules, saved):
        module.context_out.weight.data.copy_(weight)
        if bias is not None and module.context_out.bias is not None:
            module.context_out.bias.data.copy_(bias)


def force_reliability_beta_eff(modules, beta_eff):
    if beta_eff is None:
        return []
    saved = []
    for _, module in modules:
        if not hasattr(module, "reliability_beta") or not hasattr(module, "reliability_beta_max"):
            continue
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


def restore_reliability_beta(saved):
    for param, value in saved:
        param.data.copy_(value)


def run_pass(model, loader, device, threshold, modules, collect_context=False, max_batches=0, desc="Evaluate"):
    context_outputs = defaultdict(list)
    handles = []
    if collect_context:
        for name, module in modules:
            def make_hook(key):
                def hook(_module, _inputs, output):
                    context_outputs[key].append(output.detach().float().cpu())
                return hook
            handles.append(module.context_out.register_forward_hook(make_hook(name)))

    tp = fp = fn = 0.0
    probs = []
    gts = []
    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            image = batch["image"].to(device)
            gt = (batch["mask"].to(device) > 0.5)
            output = model(image, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = output[0]
            surface_prob = torch.sigmoid(surface_logits)
            if gt.shape[-2:] != surface_prob.shape[-2:]:
                gt = F.interpolate(gt.float(), size=surface_prob.shape[-2:], mode="nearest") > 0.5
            batch_tp, batch_fp, batch_fn, _ = metric_counts(surface_prob, gt, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            probs.append(surface_prob.detach().float().cpu())
            gts.append(gt.detach().bool().cpu())
            processed += int(image.shape[0])

    for handle in handles:
        handle.remove()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "metrics": metrics_from_counts(tp, fp, fn),
        "probs": torch.cat(probs, dim=0) if probs else torch.empty(0),
        "gts": torch.cat(gts, dim=0) if gts else torch.empty(0, dtype=torch.bool),
        "contexts": {key: torch.cat(value, dim=0) for key, value in context_outputs.items()},
        "images": processed,
    }


def masked_mean(values, mask):
    if values.numel() == 0 or not mask.any():
        return float("nan")
    return float(values[mask].mean().item())


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = load_model(args, device)
    modules = context_modules(model)
    if not modules:
        raise RuntimeError("No reliability context modules were found.")

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

    beta_saved = force_reliability_beta_eff(modules, args.force_reliability_beta_eff)
    if args.single_pass:
        context_saved = None
        if args.context_mode == "zero":
            context_saved = set_context_enabled(modules, enabled=False)
        result = run_pass(
            model,
            loader,
            device,
            args.threshold,
            modules,
            collect_context=True,
            max_batches=args.max_batches,
            desc=f"beta={args.force_reliability_beta_eff if args.force_reliability_beta_eff is not None else 'learned'} context={args.context_mode}",
        )
        if context_saved is not None:
            restore_context(modules, context_saved)
        iou, f1, precision, recall = result["metrics"]
        print(f"Checkpoint: {args.model_path}")
        print(
            f"split={args.split}, threshold={args.threshold}, images={result['images']}, "
            f"context_mode={args.context_mode}, forced_beta_eff={args.force_reliability_beta_eff}"
        )
        print("\nSurface Reliability Single Pass")
        print(f"  IoU={iou:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
        print("\nReliability Context Modules")
        gt = result["gts"]
        pred = result["probs"] >= float(args.threshold)
        fp = pred & (~gt)
        fn = (~pred) & gt
        tp = pred & gt
        tn = (~pred) & (~gt)
        for name, module in modules:
            beta = float(module.reliability_beta_eff().detach().cpu())
            ctx = result["contexts"].get(name)
            if ctx is None or ctx.numel() == 0:
                continue
            ctx_up = F.interpolate(ctx, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            print(f"  {name}")
            print(f"    beta_eff={beta:.8f}")
            print(f"    C_ctx mean={ctx.mean().item():+.6f}, std={ctx.std(unbiased=False).item():.6f}")
            print(f"    C_ctx FP mean={masked_mean(ctx_up, fp):+.6f}")
            print(f"    C_ctx FN mean={masked_mean(ctx_up, fn):+.6f}")
            print(f"    C_ctx TP mean={masked_mean(ctx_up, tp):+.6f}")
            print(f"    C_ctx TN mean={masked_mean(ctx_up, tn):+.6f}")
        restore_reliability_beta(beta_saved)
        return

    saved = set_context_enabled(modules, enabled=False)
    off = run_pass(
        model,
        loader,
        device,
        args.threshold,
        modules,
        collect_context=False,
        max_batches=args.max_batches,
        desc="Context OFF",
    )
    restore_context(modules, saved)
    on = run_pass(
        model,
        loader,
        device,
        args.threshold,
        modules,
        collect_context=True,
        max_batches=args.max_batches,
        desc="Context ON",
    )
    restore_reliability_beta(beta_saved)

    off_iou, off_f1, off_p, off_r = off["metrics"]
    on_iou, on_f1, on_p, on_r = on["metrics"]
    off_pred = off["probs"] >= float(args.threshold)
    on_pred = on["probs"] >= float(args.threshold)
    gt = on["gts"]
    fp_to_tn = off_pred & (~on_pred) & (~gt)
    fn_to_tp = (~off_pred) & on_pred & gt
    tn_to_fp = (~off_pred) & on_pred & (~gt)
    tp_to_fn = off_pred & (~on_pred) & gt

    print(f"Checkpoint: {args.model_path}")
    print(f"split={args.split}, threshold={args.threshold}, images={on['images']}")
    print("\nSurface Reliability Context Ablation")
    print(f"  OFF: IoU={off_iou:.4f}, F1={off_f1:.4f}, P={off_p:.4f}, R={off_r:.4f}")
    print(f"  ON:  IoU={on_iou:.4f}, F1={on_f1:.4f}, P={on_p:.4f}, R={on_r:.4f}")
    print(f"  DELTA: IoU={on_iou - off_iou:+.4f}, F1={on_f1 - off_f1:+.4f}, P={on_p - off_p:+.4f}, R={on_r - off_r:+.4f}")
    print("\nChanged Pixels")
    print(f"  FP->TN: {int(fp_to_tn.sum().item())}")
    print(f"  FN->TP: {int(fn_to_tp.sum().item())}")
    print(f"  TN->FP: {int(tn_to_fp.sum().item())}")
    print(f"  TP->FN: {int(tp_to_fn.sum().item())}")
    print("\nReliability Context Modules")
    for name, module in modules:
        beta = float(module.reliability_beta_eff().detach().cpu())
        ctx = on["contexts"].get(name)
        if ctx is None or ctx.numel() == 0:
            continue
        ctx_up = F.interpolate(ctx, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        print(f"  {name}")
        print(f"    beta_eff={beta:.8f}")
        print(f"    C_ctx mean={ctx.mean().item():+.6f}, std={ctx.std(unbiased=False).item():.6f}")
        print(f"    C_ctx FP->TN mean={masked_mean(ctx_up, fp_to_tn):+.6f}")
        print(f"    C_ctx FN->TP mean={masked_mean(ctx_up, fn_to_tp):+.6f}")
        print(f"    C_ctx TN->FP mean={masked_mean(ctx_up, tn_to_fp):+.6f}")
        print(f"    C_ctx TP->FN mean={masked_mean(ctx_up, tp_to_fn):+.6f}")


if __name__ == "__main__":
    main()
