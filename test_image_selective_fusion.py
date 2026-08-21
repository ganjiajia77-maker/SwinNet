import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import binary_metrics_from_logits
from networks.vision_transformer_selective_fusion import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


def _largest_group_divisor(n: int) -> int:
    for d in range(min(32, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def make_unique_dir(base_dir, name):
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        return path
    suffix = 1
    while True:
        candidate = f"{path}_{suffix}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def _ensure_4d(mask):
    if mask.dim() == 3:
        return mask.unsqueeze(0)
    return mask


def _resize_binary_to(logits_or_size, tensor):
    if isinstance(logits_or_size, tuple):
        target_size = logits_or_size
    else:
        target_size = logits_or_size.shape[-2:]
    tensor = _ensure_4d(tensor).float()
    if tensor.shape[-2:] != target_size:
        tensor = F.interpolate(tensor, size=target_size, mode="nearest")
    return tensor


def _to_prob_map(logits):
    if logits.dim() == 4 and logits.size(1) == 1:
        return torch.sigmoid(logits)
    if logits.dim() == 4 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1:2]
    raise RuntimeError(f"Unsupported logits shape: {tuple(logits.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--crop_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--skeleton_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./model_out")
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", type=int, default=2)
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
        choices=[
            "stage23",
            "final_correction",
            "stage23_final_correction",
            "post_refine_interaction",
            "none",
        ],
    )
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        saved_profile = checkpoint.get("structure_profile")
        if saved_profile:
            args.structure_profile = saved_profile
        elif isinstance(checkpoint.get("args"), dict):
            args.structure_profile = checkpoint["args"].get("structure_profile", args.structure_profile)
        if isinstance(checkpoint.get("args"), dict):
            saved_args = checkpoint["args"]
            for name in (
                "disable_msfe_skip",
                "enable_highres_structure_stream",
                "highres_structure_channels",
                "highres_structure_fuse_stages",
                "highres_structure_fusion_mode",
                "enable_post_refine_structure_interaction",
            ):
                if name in saved_args:
                    setattr(args, name, saved_args[name])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model = model.to(device)
    model.eval()
    print("Model loaded")
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
    print(f"{args.split.capitalize()} set size: {len(dataset)}")

    model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_dir = make_unique_dir(args.output_dir, f"{model_basename}_{args.split}_{timestamp}")
    surface_dir = os.path.join(pred_dir, "surface")
    skeleton_dir = os.path.join(pred_dir, "skeleton")
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)

    total_tp = total_fp = total_fn = total_tn = 0
    total_sk_tp = total_sk_fp = total_sk_fn = total_sk_tn = 0

    print(f"Saving predictions to: {pred_dir}")
    print(f"Threshold: {args.threshold:.4f}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            images_name = batch["image_name"]

            outputs = model(images)
            if not isinstance(outputs, tuple):
                raise RuntimeError("Selective fusion test expects tuple outputs.")
            surface_logits = outputs[0]
            skeleton_logits = outputs[2] if len(outputs) > 2 else None

            surface_prob = _to_prob_map(surface_logits).cpu()
            surface_pred = (surface_prob >= args.threshold).to(torch.uint8)
            masks_bin = (_resize_binary_to(surface_logits, masks).cpu() > 0.5).to(torch.uint8)

            batch_tp = int((surface_pred & masks_bin).sum().item())
            batch_fp = int((surface_pred & (1 - masks_bin)).sum().item())
            batch_fn = int(((1 - surface_pred) & masks_bin).sum().item())
            batch_tn = int(((1 - surface_pred) & (1 - masks_bin)).sum().item())
            total_tp += batch_tp
            total_fp += batch_fp
            total_fn += batch_fn
            total_tn += batch_tn

            sk_prob = None
            if skeleton_logits is not None:
                sk_prob = _to_prob_map(skeleton_logits).cpu()
                sk_pred = (sk_prob >= args.skeleton_threshold).to(torch.uint8)
                sk_gt = (_resize_binary_to(skeleton_logits, batch["skeleton"]).cpu() > 0.5).to(torch.uint8)
                total_sk_tp += int((sk_pred & sk_gt).sum().item())
                total_sk_fp += int((sk_pred & (1 - sk_gt)).sum().item())
                total_sk_fn += int(((1 - sk_pred) & sk_gt).sum().item())
                total_sk_tn += int(((1 - sk_pred) & (1 - sk_gt)).sum().item())

            for i in range(surface_prob.shape[0]):
                image_name = images_name[i]
                stem = os.path.splitext(image_name)[0]
                surf_np = (surface_prob[i, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(surf_np, mode="L").save(os.path.join(surface_dir, f"{stem}.png"))
                if sk_prob is not None:
                    sk_np = (sk_prob[i, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(sk_np, mode="L").save(os.path.join(skeleton_dir, f"{stem}.png"))

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    iou = total_tp / max(total_tp + total_fp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"IoU: {iou:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Predictions saved to: {pred_dir}")
    if total_sk_tp + total_sk_fp + total_sk_fn + total_sk_tn > 0:
        sk_precision = total_sk_tp / max(total_sk_tp + total_sk_fp, 1)
        sk_recall = total_sk_tp / max(total_sk_tp + total_sk_fn, 1)
        sk_iou = total_sk_tp / max(total_sk_tp + total_sk_fp + total_sk_fn, 1)
        sk_f1 = 2 * sk_precision * sk_recall / max(sk_precision + sk_recall, 1e-8)
        print(f"Skeleton IoU: {sk_iou:.6f}")
        print(f"Skeleton F1: {sk_f1:.6f}")
        print(f"Skeleton Precision: {sk_precision:.6f}")
        print(f"Skeleton Recall: {sk_recall:.6f}")

    with open(os.path.join(pred_dir, "test_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Threshold: {args.threshold:.6f}\n")
        f.write(f"IoU: {iou:.6f}\n")
        f.write(f"F1: {f1:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"Pred dir: {pred_dir}\n")


if __name__ == "__main__":
    main()
