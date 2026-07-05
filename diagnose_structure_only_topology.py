import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import (
    BCEDiceLoss,
    build_boundary_target,
    build_connectivity_target,
)
from networks.vision_transformer import (
    SwinUnet,
    get_topology_coefficients,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--root_path", default="./data1")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--output_dir", default="./topology_diagnostics")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--surface_threshold", type=float, default=0.2)
    parser.add_argument("--skeleton_threshold", type=float, default=0.5)
    parser.add_argument(
        "--cfg",
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    return parser.parse_args()


def aggregate(records):
    records = [record for record in records if record]
    if not records:
        return {}
    result = {}
    for key in records[0]:
        values = [record[key] for record in records]
        result[key] = (
            max(values)
            if key.endswith("_max")
            else float(np.mean(values))
        )
    return result


def update_counts(counts, logits, target, threshold):
    prediction = torch.sigmoid(logits) >= threshold
    target = target > 0.5
    counts["tp"] += int((prediction & target).sum().item())
    counts["fp"] += int((prediction & ~target).sum().item())
    counts["fn"] += int((~prediction & target).sum().item())


def metrics(counts):
    eps = 1e-7
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": 2.0 * precision * recall / (precision + recall + eps),
    }


def module_grad_report(module):
    parameters = list(module.parameters())
    gradients = [parameter.grad for parameter in parameters]
    nonzero = [
        float(gradient.detach().abs().max().cpu())
        for gradient in gradients
        if gradient is not None
    ]
    return {
        "any_requires_grad": any(p.requires_grad for p in parameters),
        "grad_is_none_for_all": all(g is None for g in gradients),
        "grad_abs_max": max(nonzero) if nonzero else 0.0,
    }


def boundary_curve(model_path):
    path = os.path.join(os.path.dirname(model_path), "training_log.txt")
    pattern = re.compile(
        r"Epoch \[(\d+)/\d+\], Batch \[\d+/\d+\].*Boundary:\s*([-+0-9.eE]+)"
    )
    values = defaultdict(list)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    values[int(match.group(1))].append(float(match.group(2)))
    return {
        "log_path": path,
        "epochs": [
            {
                "epoch": epoch,
                "mean_logged_boundary_loss": float(np.mean(losses)),
                "logged_batches": len(losses),
            }
            for epoch, losses in sorted(values.items())
        ],
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = torch.load(args.model_path, map_location=device)
    checkpoint_state = checkpoint["model_state_dict"]
    is_legacy_0621 = not any(
        "final_topology_attention." in key for key in checkpoint_state
    )
    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        final_topology_eta_init=(
            0.0 if is_legacy_0621 else args.final_topology_eta_init
        ),
        final_gap_rho_init=args.final_gap_rho_init,
    ).to(device)
    load_topology_checkpoint_state(
        model,
        checkpoint_state,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model.eval()

    core = model.swin_unet
    head = core.guided_head
    head.use_legacy_surface_residual = is_legacy_0621
    topology = head.final_topology_attention
    topology.capture_diagnostics = True
    head.capture_surface_diagnostics = True
    for block in core.decoder_structure_blocks:
        block.capture_diagnostics = True

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
    )

    captures = {"skeleton": [], "connectivity": [], "head_input": None}
    hooks = [
        head.skeleton_head.register_forward_hook(
            lambda module, inputs, output: captures["skeleton"].append(output)
        ),
        head.connectivity_head.register_forward_hook(
            lambda module, inputs, output: captures["connectivity"].append(output)
        ),
        head.register_forward_pre_hook(
            lambda module, inputs: captures.update(head_input=inputs[0])
        ),
    ]

    seed_counts = {"tp": 0, "fp": 0, "fn": 0}
    final_counts = {"tp": 0, "fp": 0, "fn": 0}
    surface_counts = {"tp": 0, "fp": 0, "fn": 0}
    seed_conn_bce = final_conn_bce = boundary_loss_sum = 0.0
    topology_records = []
    boundary_records = []
    stage_records = [[] for _ in core.decoder_structure_blocks]
    first_batch = None
    batches = 0
    boundary_loss_fn = BCEDiceLoss(dice_weight=1.0, bce_weight=1.0)

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Structure-only diagnostics"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                skeleton_gt = batch["skeleton"].to(device)
                if first_batch is None:
                    first_batch = {
                        "image": images[:1].clone(),
                        "skeleton": skeleton_gt[:1].clone(),
                    }
                captures["skeleton"].clear()
                captures["connectivity"].clear()
                outputs = model(images)
                seed_skeleton = captures["skeleton"][0]
                seed_connectivity = captures["connectivity"][0]
                final_skeleton = outputs[2]
                final_connectivity = outputs[3]
                update_counts(
                    surface_counts,
                    outputs[0],
                    masks,
                    args.surface_threshold,
                )

                update_counts(
                    seed_counts,
                    seed_skeleton,
                    skeleton_gt,
                    args.skeleton_threshold,
                )
                update_counts(
                    final_counts,
                    final_skeleton,
                    skeleton_gt,
                    args.skeleton_threshold,
                )
                connectivity_gt = build_connectivity_target(
                    skeleton_gt,
                    erode_kernel_size=1,
                )
                seed_conn_bce += float(
                    F.binary_cross_entropy_with_logits(
                        seed_connectivity,
                        connectivity_gt,
                    ).cpu()
                )
                final_conn_bce += float(
                    F.binary_cross_entropy_with_logits(
                        final_connectivity,
                        connectivity_gt,
                    ).cpu()
                )
                boundary_gt = build_boundary_target(masks, radius=1)
                boundary_loss, _, _ = boundary_loss_fn(outputs[1], boundary_gt)
                boundary_loss_sum += float(boundary_loss.cpu())
                batches += 1

                topology_records.append(topology.last_diagnostics)
                boundary_records.append(head.last_surface_diagnostics)
                for index, block in enumerate(core.decoder_structure_blocks):
                    stage_records[index].append(block.last_diagnostics)
    finally:
        for hook in hooks:
            hook.remove()

    # Directly prove alpha cannot change surface output.
    final_feature = captures["head_input"][:1]
    original_alpha = float(head.alpha.detach().cpu())
    alpha_outputs = {}
    with torch.no_grad():
        for alpha in (-1.0, 0.0, 1.0):
            head.alpha.fill_(alpha)
            alpha_outputs[alpha] = head(final_feature)[0].clone()
        head.alpha.fill_(original_alpha)

    model.zero_grad(set_to_none=True)
    outputs = model(first_batch["image"])
    outputs[0].mean().backward()
    surface_backward = {
        "alpha_requires_grad": head.alpha.requires_grad,
        "alpha_grad": (
            None if head.alpha.grad is None else float(head.alpha.grad.cpu())
        ),
        "structure_fusion": module_grad_report(head.structure_fusion),
        "structure_residual": module_grad_report(head.structure_residual),
        "eta_grad": (
            None
            if topology.raw_eta.grad is None
            else float(topology.raw_eta.grad.cpu())
        ),
    }

    model.zero_grad(set_to_none=True)
    outputs = model(first_batch["image"])
    connectivity_gt = build_connectivity_target(
        first_batch["skeleton"],
        erode_kernel_size=1,
    )
    structure_loss = (
        F.binary_cross_entropy_with_logits(outputs[2], first_batch["skeleton"])
        + F.binary_cross_entropy_with_logits(outputs[3], connectivity_gt)
    )
    structure_loss.backward()
    structure_backward = {
        "eta_raw": float(topology.raw_eta.detach().cpu()),
        "eta_eff": float(topology.effective_eta().detach().cpu()),
        "eta_grad": (
            None
            if topology.raw_eta.grad is None
            else float(topology.raw_eta.grad.cpu())
        ),
        "qkv_grad_abs_mean": (
            None
            if topology.qkv.weight.grad is None
            else float(topology.qkv.weight.grad.abs().mean().cpu())
        ),
    }

    report = {
        "checkpoint": os.path.abspath(args.model_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_topology_version": checkpoint.get(
            "topology_attention_version",
            "legacy-unrecorded",
        ),
        "split": args.split,
        "samples": len(dataset),
        "legacy_0621_execution_mode": is_legacy_0621,
        "surface_threshold": args.surface_threshold,
        "surface_metrics": metrics(surface_counts),
        "group1_legacy_surface_residual": {
            "alpha_value": original_alpha,
            "surface_backward": surface_backward,
            "alpha_forward_invariance": {
                "minus1_vs_zero_max_abs_diff": float(
                    (alpha_outputs[-1.0] - alpha_outputs[0.0])
                    .abs()
                    .max()
                    .cpu()
                ),
                "plus1_vs_zero_max_abs_diff": float(
                    (alpha_outputs[1.0] - alpha_outputs[0.0])
                    .abs()
                    .max()
                    .cpu()
                ),
                "exactly_equal": bool(
                    torch.equal(alpha_outputs[-1.0], alpha_outputs[0.0])
                    and torch.equal(alpha_outputs[1.0], alpha_outputs[0.0])
                ),
            },
        },
        "group2_final_topology_refiner": {
            "coefficients": get_topology_coefficients(model)["final_topology"],
            "structure_backward": structure_backward,
            "runtime": aggregate(topology_records),
            "seed_skeleton": metrics(seed_counts),
            "final_skeleton": metrics(final_counts),
            "seed_connectivity_bce": seed_conn_bce / max(batches, 1),
            "final_connectivity_bce": final_conn_bce / max(batches, 1),
        },
        "group3_decoder_structure_gates": {
            f"stage{index}": aggregate(records)
            for index, records in enumerate(stage_records)
        },
        "group4_boundary": {
            "runtime": aggregate(boundary_records),
            "validation_boundary_loss": boundary_loss_sum / max(batches, 1),
            "configured_weight": 0.03,
            "training_curve": boundary_curve(args.model_path),
        },
    }

    report_path = os.path.join(
        args.output_dir,
        "structure_only_topology_diagnostics.json",
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
    print_topology_coefficients(model)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"Diagnostic report: {report_path}")


if __name__ == "__main__":
    main()
