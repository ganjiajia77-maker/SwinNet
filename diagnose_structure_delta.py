import argparse
import os
import sys

import numpy as np
import torch
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


def cli_has(flag):
    return flag in sys.argv[1:]


def match_spatial_size(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def find_delta_outputs(stage_outputs):
    if not stage_outputs:
        return None, None
    for stage_output in stage_outputs:
        delta_logits = stage_output.get("structure_surface_delta_logits")
        if delta_logits is not None:
            return delta_logits, stage_output.get("structure_surface_base_logits")
    return None, None


def empty_bucket():
    return {
        "sum": 0.0,
        "abs_sum": 0.0,
        "pos": 0,
        "neg": 0,
        "count": 0,
    }


def update_bucket(bucket, values):
    if values.numel() == 0:
        return
    detached = values.detach()
    bucket["sum"] += float(detached.sum().cpu())
    bucket["abs_sum"] += float(detached.abs().sum().cpu())
    bucket["pos"] += int((detached > 0).sum().cpu())
    bucket["neg"] += int((detached < 0).sum().cpu())
    bucket["count"] += int(detached.numel())


def bucket_row(name, bucket):
    count = bucket["count"]
    if count == 0:
        return name, float("nan"), float("nan"), float("nan"), 0
    return (
        name,
        bucket["sum"] / count,
        bucket["abs_sum"] / count,
        bucket["pos"] / count,
        count,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Diagnose final structure-to-surface delta logits by semantic regions."
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--crop_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--reference_logits",
        type=str,
        default="base",
        choices=["base", "final"],
        help="Use logits before or after delta correction to define Weak FN / TP masks.",
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
    if not isinstance(checkpoint, dict):
        return
    saved_profile = checkpoint.get("structure_profile")
    if saved_profile:
        args.structure_profile = saved_profile
    elif isinstance(checkpoint.get("args"), dict):
        args.structure_profile = checkpoint["args"].get(
            "structure_profile",
            args.structure_profile,
        )

    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        return
    for name in (
        "disable_msfe_skip",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "highres_structure_fusion_mode",
        "img_size",
        "source_patch_size",
        "stage_topology_stages",
    ):
        flag = "--" + name
        if name in saved_args and not cli_has(flag):
            setattr(args, name, saved_args[name])


def main():
    args = build_parser().parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    inherit_checkpoint_args(args, checkpoint)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(
        "Delta diagnostic: split={}, threshold={:.3f}, reference_logits={}".format(
            args.split,
            args.threshold,
            args.reference_logits,
        )
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
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model = model.to(device)
    model.eval()
    print_topology_coefficients(model)

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
    print(f"{args.split} set size: {len(dataset)}")

    buckets = {
        "Weak skeleton FN": empty_bucket(),
        "Skeleton TP": empty_bucket(),
        "Background": empty_bucket(),
    }
    global_delta = empty_bucket()
    surface_tp = surface_fp = surface_fn = 0
    missing_delta_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Delta diagnostic"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            skeletons = batch["skeleton"].to(device)
            outputs = model(images)
            if not isinstance(outputs, tuple) or len(outputs) < 5:
                raise RuntimeError("Delta diagnostic requires auxiliary model outputs.")

            final_surface_logits = outputs[0]
            stage_outputs = outputs[4]
            delta_logits, base_surface_logits = find_delta_outputs(stage_outputs)
            if delta_logits is None:
                missing_delta_batches += 1
                continue

            delta_logits = match_spatial_size(delta_logits, final_surface_logits, mode="bilinear")
            if args.reference_logits == "base" and base_surface_logits is not None:
                reference_logits = match_spatial_size(
                    base_surface_logits,
                    delta_logits,
                    mode="bilinear",
                )
            else:
                reference_logits = match_spatial_size(
                    final_surface_logits,
                    delta_logits,
                    mode="bilinear",
                )
            masks = match_spatial_size(masks, delta_logits, mode="nearest")
            skeletons = match_spatial_size(skeletons, delta_logits, mode="nearest")

            surface_pred = torch.sigmoid(reference_logits) >= args.threshold
            surface_bin = masks > 0.5
            skeleton_bin = skeletons > 0.5

            weak_skeleton_fn = skeleton_bin & surface_bin & (~surface_pred)
            skeleton_tp = skeleton_bin & surface_bin & surface_pred
            background = ~surface_bin

            update_bucket(global_delta, delta_logits.reshape(-1))
            update_bucket(buckets["Weak skeleton FN"], delta_logits[weak_skeleton_fn])
            update_bucket(buckets["Skeleton TP"], delta_logits[skeleton_tp])
            update_bucket(buckets["Background"], delta_logits[background])

            surface_tp += int((surface_pred & surface_bin).sum().cpu())
            surface_fp += int((surface_pred & (~surface_bin)).sum().cpu())
            surface_fn += int(((~surface_pred) & surface_bin).sum().cpu())

    if missing_delta_batches:
        print(
            f"[WARN] {missing_delta_batches} batches had no structure_surface_delta_logits. "
            "Use --highres_structure_fusion_mode final_correction or stage23_final_correction."
        )

    denom_iou = surface_tp + surface_fp + surface_fn
    surface_iou = surface_tp / denom_iou if denom_iou else float("nan")
    surface_f1 = (
        2 * surface_tp / (2 * surface_tp + surface_fp + surface_fn)
        if (2 * surface_tp + surface_fp + surface_fn)
        else float("nan")
    )

    print("\nSurface reference metrics")
    print(f"  IoU={surface_iou:.6f} F1={surface_f1:.6f} threshold={args.threshold:.3f}")
    print("\nDelta surface logits by region")
    print("{:<20} {:>12} {:>12} {:>12} {:>12}".format(
        "Region",
        "mean(delta)",
        "mean(abs)",
        "pos_ratio",
        "pixels",
    ))
    print("-" * 72)
    rows = [bucket_row("All pixels", global_delta)]
    rows.extend(bucket_row(name, bucket) for name, bucket in buckets.items())
    for name, mean_delta, mean_abs, pos_ratio, count in rows:
        print("{:<20} {:>12.6f} {:>12.6f} {:>12.4f} {:>12}".format(
            name,
            mean_delta,
            mean_abs,
            pos_ratio,
            count,
        ))

    weak_mean = bucket_row("Weak skeleton FN", buckets["Weak skeleton FN"])[1]
    tp_mean = bucket_row("Skeleton TP", buckets["Skeleton TP"])[1]
    bg_mean = bucket_row("Background", buckets["Background"])[1]
    print("\nMechanism check")
    if np.isfinite(weak_mean) and np.isfinite(tp_mean) and np.isfinite(bg_mean):
        if weak_mean > 0 and tp_mean > 0 and bg_mean < 0:
            print("  PASS direction: Weak FN > 0, Skeleton TP > 0, Background < 0")
        elif weak_mean < 0 and tp_mean > 0 and bg_mean < 0:
            print("  CONFIRMATION BIAS: Weak FN < 0, Skeleton TP > 0, Background < 0")
        else:
            print("  MIXED direction: inspect the table above before deciding.")
    else:
        print("  Not enough pixels in one or more regions.")


if __name__ == "__main__":
    main()
