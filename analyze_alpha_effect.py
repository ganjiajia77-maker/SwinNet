import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ablate the final structure-residual alpha on one checkpoint."
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--old_alpha", type=float, default=0.4666091203689575)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="./topology_diagnostics/alpha_ablation_0704.txt",
    )

    # Arguments consumed by config.update_config.
    parser.add_argument(
        "--cfg",
        type=str,
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
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
    return parser


def update_confusion(stats, logits, target, threshold):
    pred = torch.sigmoid(logits) >= threshold
    target = target > 0.5
    stats["tp"] += int((pred & target).sum().item())
    stats["fp"] += int((pred & ~target).sum().item())
    stats["fn"] += int((~pred & target).sum().item())


def metrics(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, iou, f1


def accumulate_region(stats, name, values, mask):
    mask = mask.bool()
    stats[name]["sum"] += float(values.masked_select(mask).sum().item())
    stats[name]["count"] += int(mask.sum().item())


def region_mean(stats, name):
    count = stats[name]["count"]
    return stats[name]["sum"] / count if count else float("nan")


def main():
    args = build_parser().parse_args()
    cudnn.benchmark = False
    cudnn.deterministic = True

    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=args.final_topology_eta_init,
    ).cuda()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    load_topology_checkpoint_state(
        model,
        state_dict,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=(args.bottleneck_type == "global_local"),
    )
    model.eval()

    head = model.swin_unet.guided_head
    checkpoint_alpha = float(head.alpha.detach().cpu())
    alpha_values = OrderedDict(
        [
            ("original", checkpoint_alpha),
            ("zero", 0.0),
            ("positive_abs", abs(checkpoint_alpha)),
            ("old_0621", float(args.old_alpha)),
            ("unit_residual", 1.0),
        ]
    )

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="test",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    final_head_input = {}

    def capture_head_input(module, inputs):
        final_head_input["feature"] = inputs[0]

    hook = head.register_forward_pre_hook(capture_head_input)
    confusion = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in alpha_values
        if name != "unit_residual"
    }
    region_names = (
        "tp",
        "fn",
        "fp",
        "skeleton_road",
        "non_skeleton_road",
        "dilated_skeleton_road",
    )
    delta_stats = {
        name: {"sum": 0.0, "count": 0}
        for name in region_names
    }
    residual_stats = {
        name: {"sum": 0.0, "count": 0}
        for name in region_names
    }

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Alpha diagnostics"):
                images = batch["image"].cuda(non_blocking=True)
                target = batch["mask"].cuda(non_blocking=True) > 0.5
                skeleton = batch["skeleton"].cuda(non_blocking=True) > 0.5
                skeleton_dilate = (
                    batch["skeleton_dilate"].cuda(non_blocking=True) > 0.5
                )

                # Run the expensive encoder/decoder once and reuse its final feature.
                model(images)
                feature = final_head_input["feature"]
                logits_by_name = {}
                for name, alpha in alpha_values.items():
                    head.alpha.fill_(alpha)
                    logits_by_name[name] = head(feature)[0]

                for name in confusion:
                    update_confusion(
                        confusion[name],
                        logits_by_name[name],
                        target,
                        args.threshold,
                    )

                original_logits = logits_by_name["original"]
                zero_logits = logits_by_name["zero"]
                delta_alpha = original_logits - zero_logits
                residual_effect = logits_by_name["unit_residual"] - zero_logits

                original_pred = torch.sigmoid(original_logits) >= args.threshold
                region_masks = {
                    "tp": original_pred & target,
                    "fn": ~original_pred & target,
                    "fp": original_pred & ~target,
                    "skeleton_road": target & skeleton,
                    "non_skeleton_road": target & ~skeleton,
                    "dilated_skeleton_road": target & skeleton_dilate,
                }
                for name, mask in region_masks.items():
                    accumulate_region(delta_stats, name, delta_alpha, mask)
                    accumulate_region(
                        residual_stats,
                        name,
                        residual_effect,
                        mask,
                    )
    finally:
        hook.remove()
        with torch.no_grad():
            head.alpha.fill_(checkpoint_alpha)

    lines = [
        "Final structure-residual alpha diagnostic",
        f"checkpoint: {args.model_path}",
        f"surface threshold: {args.threshold:.4f}",
        f"checkpoint alpha: {checkpoint_alpha:.8f}",
        "",
        "Test 1: alpha ablation",
        "name              alpha        Precision    Recall       IoU          F1",
    ]
    for name, alpha in alpha_values.items():
        if name == "unit_residual":
            continue
        precision, recall, iou, f1 = metrics(confusion[name])
        lines.append(
            f"{name:<17} {alpha:>10.6f}  "
            f"{precision:>10.6f}  {recall:>10.6f}  "
            f"{iou:>10.6f}  {f1:>10.6f}"
        )

    lines.extend(
        [
            "",
            "Test 2: delta_alpha = logits(checkpoint_alpha) - logits(alpha=0)",
            "region                    mean_delta       pixel_count",
        ]
    )
    for name in region_names:
        lines.append(
            f"{name:<25} {region_mean(delta_stats, name):>12.8f}  "
            f"{delta_stats[name]['count']:>12d}"
        )

    lines.extend(
        [
            "",
            "Test 3: R_effect = logits(alpha=+1) - logits(alpha=0)",
            "region                    mean_effect      pixel_count",
        ]
    )
    for name in region_names:
        lines.append(
            f"{name:<25} {region_mean(residual_stats, name):>12.8f}  "
            f"{residual_stats[name]['count']:>12d}"
        )

    output = "\n".join(lines)
    print(output)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(output + "\n")
    print(f"\nSaved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
