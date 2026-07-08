"""Verify 0626 checkpoint reload against the restored architecture."""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "./model_out/train_stage23_structure_final_boundary_nw0_20260626/best.pth"
        ),
    )
    parser.add_argument("--root_path", default="./data1")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--val_threshold", type=float, default=0.2)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--target_test_iou",
        type=float,
        default=0.5370,
    )
    parser.add_argument(
        "--target_test_f1",
        type=float,
        default=0.6985,
    )
    parser.add_argument(
        "--cfg",
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
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--img-size", dest="img_size", type=int)
    return parser.parse_args()


def build_model(args):
    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type="global_local",
        final_topology_eta_init=0.0,
        final_gap_rho_init=0.0,
        stage_topology_stages="none",
        structure_profile=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
        enable_final_graph_prop=False,
    )
    return model.cuda()


def summarize_load(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()

    missing = sorted(set(model_dict) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_dict))
    shape_mismatch = []
    for key in sorted(set(state_dict) & set(model_dict)):
        if tuple(state_dict[key].shape) != tuple(model_dict[key].shape):
            shape_mismatch.append(
                (key, tuple(state_dict[key].shape), tuple(model_dict[key].shape))
            )

    print("[LOAD] checkpoint:", checkpoint_path)
    print("[LOAD] epoch:", checkpoint.get("epoch"))
    print("[LOAD] saved val_iou:", checkpoint.get("val_iou"))
    print("[LOAD] saved val_f1:", checkpoint.get("val_f1"))
    print("[LOAD] missing keys:", len(missing))
    for key in missing:
        print("  - missing:", key)
    print("[LOAD] unexpected keys:", len(unexpected))
    for key in unexpected:
        print("  - unexpected:", key)
    print("[LOAD] shape mismatches:", len(shape_mismatch))
    for key, ckpt_shape, model_shape in shape_mismatch:
        print(f"  ! {key}: ckpt{ckpt_shape} vs model{model_shape}")

    load_topology_checkpoint_state(
        model,
        state_dict,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    print_topology_coefficients(model, prefix="[VERIFY]")
    return checkpoint, shape_mismatch


@torch.no_grad()
def evaluate_split(model, loader, threshold):
    model.eval()
    tp = fp = fn = 0
    count = 0

    for batch in tqdm(loader, desc=f"eval@{threshold:.2f}"):
        images = batch["image"].cuda(non_blocking=True)
        labels = batch["mask"].cuda(non_blocking=True)
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        pred = (torch.sigmoid(logits) >= threshold).float()
        labels = (labels > 0.5).float()
        tp += int((pred * labels).sum().item())
        fp += int((pred * (1.0 - labels)).sum().item())
        fn += int(((1.0 - pred) * labels).sum().item())
        count += images.size(0)

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "samples": count,
    }


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    model = build_model(args)
    checkpoint, shape_mismatch = summarize_load(args.checkpoint, model)

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

    threshold = args.val_threshold if args.split == "val" else args.threshold
    metrics = evaluate_split(model, loader, threshold)

    print("=" * 72)
    print(f"[RESULT] split={args.split} threshold={threshold:.2f}")
    print(
        "  IoU: {:.4f}  F1: {:.4f}  Precision: {:.4f}  Recall: {:.4f}".format(
            metrics["iou"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )
    )
    print("  samples:", metrics["samples"])
    print("=" * 72)

    load_ok = len(shape_mismatch) == 0
    if args.split == "test":
        metric_ok = (
            metrics["iou"] >= args.target_test_iou
            and metrics["f1"] >= args.target_test_f1
        )
    else:
        saved_iou = float(checkpoint.get("val_iou", 0.0))
        saved_f1 = float(checkpoint.get("val_f1", 0.0))
        metric_ok = (
            abs(metrics["iou"] - saved_iou) <= 0.003
            and abs(metrics["f1"] - saved_f1) <= 0.003
        )

    if load_ok and metric_ok:
        print("[PASS] 0626 reload verification succeeded.")
        return 0

    if not load_ok:
        print("[FAIL] shape mismatches remain after architecture restore.")
    if not metric_ok:
        print("[FAIL] metrics below acceptance threshold.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
