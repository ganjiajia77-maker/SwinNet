"""Validation threshold sweep and test with the training overlap/global protocol."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet


ARCHITECTURE_ARGS = (
    "bottleneck_type", "final_topology_eta_init", "final_gap_rho_init", "structure_profile",
    "stage2_skeleton_gradient_ratio", "stage3_skeleton_gradient_ratio",
    "stage3_gate_topology_gradient_ratio", "final_skeleton_gradient_ratio",
    "enable_highres_structure_stream", "highres_structure_channels",
    "highres_structure_fuse_stages", "highres_structure_fusion_mode",
    "enable_post_refine_structure_interaction", "enable_h3_surface_fusion",
    "enable_global_topology", "global_topology_max_nodes", "global_topology_heads",
    "global_topology_alpha_max", "stage_skeleton_mode", "stage_skeleton_bias_init",
    "stage_skeleton_positive_prior", "remove_stage2_pre_topology_source",
)


def load_model(checkpoint_path, cfg, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    if saved.get("encoder_type") != "dinov2_l16" or saved.get("direct_resize_train", False):
        raise ValueError("Expected a DINOv2 H0 native-crop checkpoint")
    config_args = SimpleNamespace(
        cfg=cfg, opts=None, batch_size=1, img_size=saved["img_size"], zip=False,
        cache_mode=None, resume=None, accumulation_steps=None, use_checkpoint=False,
        amp_opt_level=None, tag=None, eval=True, throughput=False,
        encoder_type="dinov2_l16", freeze_pretrained_encoder=True,
    )
    config = get_config(config_args)
    kwargs = {name: saved[name] for name in ARCHITECTURE_ARGS if name in saved}
    model = SwinUnet(config, num_classes=1, return_skeleton=True, **kwargs)
    state = checkpoint["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, saved, checkpoint["epoch"]


@torch.inference_mode()
def overlap_logits(model, image, tile_size, stride):
    if not 0 < stride <= tile_size:
        raise ValueError("Overlap stride must be in (0, tile_size]")
    height, width = image.shape[-2:]
    if min(height, width) < tile_size:
        raise ValueError("Full image smaller than model tile")
    weight_1d = (1.0 - torch.linspace(-1.0, 1.0, tile_size, device=image.device).abs()).clamp_min(0.1)
    weight = (weight_1d[:, None] * weight_1d[None, :])[None, None]
    logits = image.new_zeros((1, 1, height, width))
    denominator = torch.zeros_like(logits)
    for top in RoadSkeletonDataset.sliding_positions(height, tile_size, stride):
        for left in RoadSkeletonDataset.sliding_positions(width, tile_size, stride):
            outputs = model(image[:, :, top:top + tile_size, left:left + tile_size])
            surface = outputs[0] if isinstance(outputs, tuple) else outputs
            if not torch.isfinite(surface).all():
                raise ValueError("Non-finite surface output during inference")
            logits[:, :, top:top + tile_size, left:left + tile_size] += surface * weight
            denominator[:, :, top:top + tile_size, left:left + tile_size] += weight
    if (denominator <= 0).any():
        raise ValueError("Uncovered pixels during overlap inference")
    return logits / denominator.clamp_min(1e-8)


def metrics(tp, fp, fn):
    return dict(iou=float(tp / max(tp + fp + fn, 1)),
                f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
                precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", default="configs/dinov2_l16_h0_512.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--threshold_file", help="best_threshold.json selected only on validation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_probabilities", action="store_true")
    args = parser.parse_args()
    if args.split == "test":
        if not args.threshold_file:
            parser.error("Test requires --threshold_file from validation sweep")
        selected = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
        if selected.get("split") != "val" or Path(selected["checkpoint"]).resolve() != Path(args.checkpoint).resolve():
            parser.error("Threshold must be selected on val using this same checkpoint")
        thresholds = [float(selected["threshold"])]
    else:
        thresholds = sorted(set(float(x) for x in args.thresholds.split(",")))
    if not thresholds or any(not 0 < t < 1 for t in thresholds):
        parser.error("Thresholds must be between zero and one")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved, epoch = load_model(args.checkpoint, args.cfg, device)
    tile_size, stride = int(saved["img_size"]), int(saved["overlap_stride"])
    dataset = RoadSkeletonDataset(args.root_path, split=args.split, image_size=None,
                                  source_patch_size=saved["source_patch_size"], return_full_image=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        (output_dir / "surface").mkdir(exist_ok=True)
    if args.save_probabilities:
        (output_dir / "probabilities").mkdir(exist_ok=True)
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for batch in tqdm(loader, desc=f"{args.split}: tile={tile_size}, stride={stride}"):
        probability = overlap_logits(model, batch["image"].to(device), tile_size, stride).sigmoid()[0, 0].cpu().numpy()
        gt = batch["mask"][0, 0].numpy() > 0.5
        case = batch["case_name"][0]
        for i, threshold in enumerate(thresholds):
            pred = probability >= threshold
            counts[i] += (np.count_nonzero(pred & gt), np.count_nonzero(pred & ~gt), np.count_nonzero(~pred & gt))
        if args.split == "test":
            if not cv2.imwrite(str(output_dir / "surface" / f"{case}_pred.png"), ((probability >= thresholds[0]) * 255).astype(np.uint8)):
                raise IOError(f"Could not save prediction {case}")
        if args.save_probabilities:
            np.save(output_dir / "probabilities" / f"{case}.npy", probability)
    rows = [dict(threshold=t, **metrics(*count)) for t, count in zip(thresholds, counts)]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["iou"])
    report = dict(best, split=args.split, checkpoint=str(Path(args.checkpoint).resolve()), epoch=epoch,
                  images=len(dataset), tile_size=tile_size, overlap_stride=stride,
                  precision_dtype="fp32", checkpoint_weights="model_state_dict (EMA when training enabled EMA)",
                  aggregation="global TP/FP/FN", fusion="weighted logits, then sigmoid", tta=False,
                  postprocessing="none", evaluation_size=saved["source_patch_size"])
    filename = "best_threshold.json" if args.split == "val" else "test_results.json"
    (output_dir / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
