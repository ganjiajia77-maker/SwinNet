"""Compare topology metrics vs baseline (IoU=0.5719 @ th=0.24).

Reports:
  - learnable residual / bias scales (gamma1/gamma2, decoder bias scales)
  - fixed high-order attention coeffs theta1/2/3 (A + 0.5 A^2 + 0.25 A^3)
  - whether soft graph_prop / graph_diffusion is enabled
  - clDice, component counts (pred vs GT), short components, skeleton endpoints,
    max-component area ratio
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.cldice_loss import SoftCLDiceLoss
from losses.road_losses import binary_metrics_from_logits
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet,
    format_topology_coefficients,
    get_topology_coefficients,
    load_topology_checkpoint_state,
)


# Fixed high-order terms in decoder structure attention bias:
#   bias ~ theta1*A + theta2*A^2 + theta3*A^3
FIXED_THETA = (1.0, 0.5, 0.25)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline_ckpt",
        default=(
            "./model_out/train_0718_a123_dirfield_softdir_highorder_w01_d02_20260722/"
            "best.pth"
        ),
    )
    p.add_argument(
        "--new_ckpt",
        default="./model_out/train_0626_dir_field_residual_scratch/best.pth",
    )
    p.add_argument("--root_path", default="./data1")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--threshold", type=float, default=0.24)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--source_patch_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--short_area_thresh", type=int, default=32)
    p.add_argument(
        "--cfg",
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    p.add_argument("--zip", action="store_true")
    p.add_argument("--cache_mode", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--accumulation_steps", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--amp_opt_level", type=str, default="")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--throughput", action="store_true")
    p.add_argument("--max_batches", type=int, default=0, help="0 = full split")
    return p.parse_args()


def load_model(ckpt_path, args, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = (
        checkpoint.get("args")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict)
        else {}
    )
    structure_profile = checkpoint.get(
        "structure_profile",
        saved_args.get("structure_profile", STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626),
    )
    enable_graph_prop = bool(saved_args.get("enable_graph_prop", False))
    enable_graph_diffusion = bool(saved_args.get("enable_graph_diffusion", False))
    enable_simple_c = bool(saved_args.get("enable_simple_c_diffusion", False))
    enable_sc = bool(saved_args.get("enable_sc_graph_diffusion", False))
    enable_structure_gate = bool(saved_args.get("enable_structure_gate", True))
    enable_decoder_attention_bias = bool(
        saved_args.get("enable_decoder_attention_bias", True)
    )

    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type="global_local",
        structure_profile=structure_profile,
        enable_final_graph_prop=enable_graph_prop,
        enable_graph_diffusion=enable_graph_diffusion,
        enable_simple_c_diffusion=enable_simple_c,
        enable_sc_graph_diffusion=enable_sc,
        enable_structure_gate=enable_structure_gate,
        enable_decoder_attention_bias=enable_decoder_attention_bias,
    ).to(device)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    load_topology_checkpoint_state(
        model,
        state,
        checkpoint.get("topology_attention_version", "legacy-unrecorded")
        if isinstance(checkpoint, dict)
        else "legacy-unrecorded",
        strict=True,
    )
    model.eval()
    return model, saved_args


def collect_bias_scales(model):
    scales = []
    for name, module in model.named_modules():
        if hasattr(module, "decoder_connectivity_bias_scale"):
            param = getattr(module, "decoder_connectivity_bias_scale")
            if param is not None:
                scales.append((name + ".conn", float(param.detach().cpu())))
        if hasattr(module, "decoder_skeleton_bias_scale"):
            param = getattr(module, "decoder_skeleton_bias_scale")
            if param is not None:
                scales.append((name + ".skel", float(param.detach().cpu())))
    return scales


def report_params(tag, model, saved_args):
    coef = get_topology_coefficients(model)
    print("=" * 88)
    print(f"[{tag}] parameter audit")
    print("=" * 88)
    print(format_topology_coefficients(model))
    print(
        "saved_args flags: "
        f"graph_prop={saved_args.get('enable_graph_prop', False)}, "
        f"graph_diffusion={saved_args.get('enable_graph_diffusion', False)}, "
        f"simple_c={saved_args.get('enable_simple_c_diffusion', False)}, "
        f"sc_graph={saved_args.get('enable_sc_graph_diffusion', False)}"
    )
    print(
        "fixed high-order attention coeffs "
        f"(theta1, theta2, theta3) = {FIXED_THETA}  "
        "# used as A + 0.5 A^2 + 0.25 A^3"
    )
    print(
        "note: these theta are hardcoded constants, not learnable; "
        "learnable scale is decoder_connectivity_bias_scale"
    )
    for stage in (0, 1, 2, 3):
        key = f"decoder_stage{stage}"
        if key in coef:
            s = coef[key]
            print(
                f"  stage{stage}: structure_enabled={s['structure_enabled']} "
                f"gamma1={s['gamma1']:.6f} gamma2={s['gamma2']:.6f} "
                f"graph_diff={s.get('graph_diffusion_enabled')} "
                f"sc_graph={s.get('sc_graph_diffusion_enabled')} "
                f"simple_c={s.get('simple_c_diffusion_enabled')}"
            )
    for name, value in collect_bias_scales(model):
        print(f"  bias_scale {name}: {value:.6f}")
    print(
        f"  graph_prop_enabled={coef.get('graph_prop_enabled')} "
        f"final_structure={coef.get('final_structure_enabled')}"
    )


def skeleton_endpoints(skel_bool: np.ndarray) -> int:
    skel = skel_bool.astype(np.uint8)
    if skel.max() == 0:
        return 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor = cv2.filter2D(skel, -1, kernel) - skel
    # endpoint: skeleton pixel with exactly 1 neighbor
    return int(((skel > 0) & (neighbor == 1)).sum())


def component_stats(mask_bool: np.ndarray, short_area_thresh: int):
    u8 = (mask_bool.astype(np.uint8)) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([], dtype=np.int32)
    n_comp = int(num - 1)
    short = int((areas < short_area_thresh).sum()) if areas.size else 0
    total = float(areas.sum()) if areas.size else 0.0
    max_ratio = float(areas.max() / total) if total > 0 else 0.0
    return n_comp, short, max_ratio


def eval_model(model, loader, device, threshold, short_area_thresh, max_batches=0):
    cldice_fn = SoftCLDiceLoss(iter_num=10).to(device)
    meters = OrderedDict(
        iou=0.0,
        f1=0.0,
        precision=0.0,
        recall=0.0,
        cldice=0.0,
        pred_comp=0.0,
        gt_comp=0.0,
        pred_short=0.0,
        gt_short=0.0,
        pred_endpoints=0.0,
        gt_endpoints=0.0,
        pred_max_ratio=0.0,
        gt_max_ratio=0.0,
        n=0,
    )
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
            if max_batches and bi >= max_batches:
                break
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            skeletons = batch["skeleton"].to(device)
            outputs = model(images)
            surface_logits = outputs[0]
            m = binary_metrics_from_logits(surface_logits, masks, threshold=threshold)
            probs = torch.sigmoid(surface_logits)
            cldice = 1.0 - cldice_fn(probs, skeletons)

            pred = (probs >= threshold).cpu().numpy()
            gt = (masks > 0.5).cpu().numpy()
            for b in range(pred.shape[0]):
                pred_bool = pred[b, 0] > 0
                gt_bool = gt[b, 0] > 0
                pred_skel = skeletonize(pred_bool)
                gt_skel = skeletonize(gt_bool)

                pc, ps, pmr = component_stats(pred_bool, short_area_thresh)
                gc, gs, gmr = component_stats(gt_bool, short_area_thresh)
                pe = skeleton_endpoints(pred_skel)
                ge = skeleton_endpoints(gt_skel)

                meters["pred_comp"] += pc
                meters["gt_comp"] += gc
                meters["pred_short"] += ps
                meters["gt_short"] += gs
                meters["pred_endpoints"] += pe
                meters["gt_endpoints"] += ge
                meters["pred_max_ratio"] += pmr
                meters["gt_max_ratio"] += gmr
                meters["n"] += 1

            meters["iou"] += m["iou"]
            meters["f1"] += m["f1"]
            meters["precision"] += m["precision"]
            meters["recall"] += m["recall"]
            meters["cldice"] += float(cldice.item())

    n_batch = max(meters["n"], 1)
    # iou/f1 averaged over batches in binary_metrics; renormalize by batch count used
    # Here we accumulated one value per batch for iou and one per-image for topology.
    # Re-run style: report topology / n_images; surface metrics / num batches.
    # Simpler: recompute surface as mean over batches via separate counters.
    return meters


def finalize(meters, num_batches):
    n_img = max(int(meters["n"]), 1)
    n_bat = max(int(num_batches), 1)
    return OrderedDict(
        iou=meters["iou"] / n_bat,
        f1=meters["f1"] / n_bat,
        precision=meters["precision"] / n_bat,
        recall=meters["recall"] / n_bat,
        cldice=meters["cldice"] / n_bat,
        pred_comp=meters["pred_comp"] / n_img,
        gt_comp=meters["gt_comp"] / n_img,
        delta_comp=(meters["pred_comp"] - meters["gt_comp"]) / n_img,
        pred_short=meters["pred_short"] / n_img,
        gt_short=meters["gt_short"] / n_img,
        pred_endpoints=meters["pred_endpoints"] / n_img,
        gt_endpoints=meters["gt_endpoints"] / n_img,
        delta_endpoints=(meters["pred_endpoints"] - meters["gt_endpoints"]) / n_img,
        pred_max_ratio=meters["pred_max_ratio"] / n_img,
        gt_max_ratio=meters["gt_max_ratio"] / n_img,
        n_images=n_img,
    )


def print_metrics(tag, metrics):
    print("-" * 88)
    print(f"[{tag}] threshold-matched topology comparison")
    print(
        f"  IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"clDice={metrics['cldice']:.4f}"
    )
    print(
        f"  components: pred={metrics['pred_comp']:.2f} gt={metrics['gt_comp']:.2f} "
        f"delta(pred-gt)={metrics['delta_comp']:+.2f}"
    )
    print(
        f"  short_components(<area): pred={metrics['pred_short']:.2f} "
        f"gt={metrics['gt_short']:.2f}"
    )
    print(
        f"  skeleton_endpoints: pred={metrics['pred_endpoints']:.2f} "
        f"gt={metrics['gt_endpoints']:.2f} "
        f"delta(pred-gt)={metrics['delta_endpoints']:+.2f}"
    )
    print(
        f"  max_component_ratio: pred={metrics['pred_max_ratio']:.4f} "
        f"gt={metrics['gt_max_ratio']:.4f}"
    )
    print(f"  n_images={metrics['n_images']}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        pin_memory=device.type == "cuda",
    )
    print(
        f"split={args.split} images={len(dataset)} threshold={args.threshold} "
        f"short_area_thresh={args.short_area_thresh}"
    )

    rows = []
    for tag, ckpt in (
        ("baseline_05719", args.baseline_ckpt),
        ("new_dir_field_scratch", args.new_ckpt),
    ):
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        print(f"\nLoading {tag}: {ckpt}")
        model, saved_args = load_model(ckpt, args, device)
        report_params(tag, model, saved_args)
        meters = eval_model(
            model,
            loader,
            device,
            args.threshold,
            args.short_area_thresh,
            max_batches=args.max_batches,
        )
        # num batches processed
        if args.max_batches:
            n_bat = min(args.max_batches, len(loader))
        else:
            n_bat = len(loader)
        metrics = finalize(meters, n_bat)
        print_metrics(tag, metrics)
        rows.append((tag, metrics))
        del model
        torch.cuda.empty_cache()

    if len(rows) == 2:
        a, b = rows[0][1], rows[1][1]
        print("=" * 88)
        print("Delta (new - baseline)")
        for key in (
            "iou",
            "f1",
            "cldice",
            "pred_comp",
            "delta_comp",
            "pred_short",
            "pred_endpoints",
            "delta_endpoints",
            "pred_max_ratio",
        ):
            print(f"  {key}: {b[key] - a[key]:+.4f}")


if __name__ == "__main__":
    main()
