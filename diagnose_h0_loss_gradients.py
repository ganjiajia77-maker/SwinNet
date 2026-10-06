"""Measure weighted H0 loss terms and their gradients on real training batches."""

import argparse
import copy
import json
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss
from compare_connectivity_topology_metrics import load_metric_model, parse_args as model_parse_args


@contextmanager
def without_cli_args():
    original = sys.argv
    sys.argv = [original[0]]
    try:
        yield
    finally:
        sys.argv = original


def training_criterion(saved, device):
    # Reuse the training entry's exact loss construction, including profile defaults.
    with without_cli_args():
        import train_image
    args = copy.copy(train_image.args)
    for key, value in saved.items():
        setattr(args, key, value)
    weights = train_image.get_final_loss_weights(args)
    criterion = train_image.build_criterion(args, weights, device)
    return args, weights, criterion


def model_from_checkpoint(path, root, cfg, size, source_size, device):
    original = sys.argv
    sys.argv = [original[0], "--root_path", root, "--model_path", path,
                "--cfg", cfg, "--img_size", str(size),
                "--source_patch_size", str(source_size), "--num_workers", "0"]
    try:
        model_args = model_parse_args()
    finally:
        sys.argv = original
    return load_metric_model(path, model_args, device)


def stage_term(criterion, outputs, skeleton, skeleton_dilate, kind):
    original = (criterion.stage_skeleton_only_loss_factor,
                criterion.stage_connectivity_factor, criterion.stage_direction_factor,
                criterion.stage_connectivity_factors, criterion.stage_direction_factors)
    criterion.stage_skeleton_only_loss_factor = float(kind == "skeleton") * original[0]
    criterion.stage_connectivity_factor = float(kind == "connectivity") * original[1]
    criterion.stage_direction_factor = float(kind == "direction") * original[2]
    criterion.stage_connectivity_factors = tuple(float(kind == "connectivity") * x for x in original[3])
    criterion.stage_direction_factors = tuple(float(kind == "direction") * x for x in original[4])
    try:
        return criterion.stage_structure_loss(
            outputs, skeleton, skeleton_dilate,
            stage_skeleton_gt=skeleton,
            stage_skeleton_dilate_gt=skeleton_dilate,
        )
    finally:
        (criterion.stage_skeleton_only_loss_factor,
         criterion.stage_connectivity_factor, criterion.stage_direction_factor,
         criterion.stage_connectivity_factors, criterion.stage_direction_factors) = original


def gradient_vector(loss, parameter):
    if not loss.requires_grad:
        return None
    grad, = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True)
    return None if grad is None else grad.detach().float().reshape(-1)


def gradient_cosine(left, right):
    if left is None or right is None:
        return None
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return float(torch.dot(left, right) / (left_norm * right_norm))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--cfg", default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_json", required=True)
    cli = parser.parse_args()
    if cli.batches < 1 or cli.batch_size < 1:
        parser.error("--batches and --batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint args must be a dictionary")
    train_args, weights, criterion = training_criterion(saved, device)
    model = model_from_checkpoint(cli.checkpoint, cli.root_path, cli.cfg,
                                  train_args.img_size, train_args.source_patch_size, device)
    model.train()
    criterion.train()
    dataset = RoadSkeletonDataset(
        root_dir=cli.root_path, split="train", image_size=train_args.img_size,
        source_patch_size=train_args.source_patch_size,
        tile_size=train_args.img_size if train_args.random_crop_train else
                  None if train_args.direct_resize_train else train_args.img_size,
        tile_stride=train_args.overlap_stride, augment=train_args.augment,
        random_crop_train=train_args.random_crop_train,
        random_crops_per_image=train_args.random_crops_per_image,
        random_crop_seed=train_args.seed,
    )
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(int(checkpoint.get("epoch", 0)))
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    generator = torch.Generator().manual_seed(cli.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    total = min(len(indices), cli.batches * cli.batch_size)
    named = dict(model.named_parameters())
    probes = {
        stage: named[f"swin_unet.decoder_structure_blocks.{stage}.structure_branch.0.block.0.weight"]
        for stage in (2, 3)
    }
    rows = []
    for batch_number, start in enumerate(range(0, total, cli.batch_size), 1):
        samples = [dataset[i] for i in indices[start:start + cli.batch_size]]
        images = torch.stack([item["image"] for item in samples]).to(device)
        mask = torch.stack([item["mask"] for item in samples]).to(device)
        skeleton = torch.stack([item["skeleton"] for item in samples]).to(device)
        dilate = torch.stack([item["skeleton_dilate"] for item in samples]).to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(images)
        surface, boundary, final_skeleton, connectivity, stage_outputs = outputs[:5]
        total_loss, loss_dict = criterion(
            surface.float(), surface_gt=mask, skeleton_gt=skeleton,
            skeleton_dilate_gt=dilate, stage_outputs=stage_outputs,
            boundary_logits=boundary, skeleton_logits=final_skeleton,
            connectivity_logits=connectivity,
        )
        terms = {"surface": criterion.surface_loss(surface.float(), mask)[0]}
        for kind in ("skeleton", "connectivity", "direction"):
            terms[kind] = stage_term(criterion, stage_outputs, skeleton, dilate, kind)
        terms["other"] = total_loss - sum(terms.values())
        row = {"batch": batch_number, "total": float(total_loss.detach()),
               "stage_structure": float(loss_dict["stage_structure_loss"])}
        gradients = {}
        for name, loss in terms.items():
            row[name] = {"weighted_loss": float(loss.detach())}
            for stage, parameter in probes.items():
                grad = gradient_vector(loss, parameter)
                gradients[name, stage] = grad
                row[name][f"stage{stage}_grad_norm"] = (
                    0.0 if grad is None else float(torch.linalg.vector_norm(grad))
                )
        row["cosines_vs_surface"] = {
            f"stage{stage}": {
                name: gradient_cosine(gradients[name, stage], gradients["surface", stage])
                for name in ("skeleton", "connectivity", "direction")
            }
            for stage in probes
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {name: {
        key: float(np.mean([row[name][key] for row in rows]))
        for key in rows[0][name]}
        for name in terms}
    for name in ("skeleton", "connectivity", "direction", "other"):
        summary[name]["loss_vs_surface"] = (
            summary[name]["weighted_loss"] /
            max(abs(summary["surface"]["weighted_loss"]), 1e-12)
        )
        for stage in probes:
            key = f"stage{stage}_grad_norm"
            summary[name][f"stage{stage}_grad_vs_surface"] = (
                summary[name][key] / max(summary["surface"][key], 1e-12)
            )
    cosine_summary = {}
    for stage in probes:
        stage_key = f"stage{stage}"
        cosine_summary[stage_key] = {}
        for name in ("skeleton", "connectivity", "direction"):
            values = [row["cosines_vs_surface"][stage_key][name] for row in rows]
            valid = [value for value in values if value is not None]
            negative = sum(value < 0 for value in valid)
            cosine_summary[stage_key][name] = {
                "mean": float(np.mean(valid)) if valid else None,
                "median": float(np.median(valid)) if valid else None,
                "negative_batches": negative,
                "valid_batches": len(valid),
                "negative_fraction": negative / len(valid) if valid else None,
            }
    result = {"checkpoint": cli.checkpoint,
              "checkpoint_epoch": (int(checkpoint["epoch"]) if "epoch" in checkpoint else None),
              "n_batches": len(rows), "batch_size": cli.batch_size,
              "weights": {"stage2": train_args.stage2_skeleton_weight,
                          "stage3": train_args.stage3_skeleton_weight,
                          "connectivity_factor": train_args.stage_connectivity_factor,
                          "direction_factor": train_args.stage_direction_factor,
                          "final": weights},
              "summary": summary, "cosine_summary": cosine_summary, "batches": rows}
    with open(cli.output_json, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)
    print("COSINE_SUMMARY " + json.dumps(cosine_summary), flush=True)


if __name__ == "__main__":
    main()
