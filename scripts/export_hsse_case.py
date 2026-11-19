"""Export aligned source data for the H0/H1/H2 state evolution figure."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet, TOPOLOGY_ATTENTION_VERSION


TILE = 512
SOURCE_SIZE = 1500
H_STATES = ("H0", "H1", "H2")
CFG = ROOT / "configs/swin_tiny_patch4_window7_224_lite.yaml"
BASE_REQUIRED = {
    "img_size": TILE,
    "source_patch_size": SOURCE_SIZE,
    "use_ema": False,
    "structure_profile": "stage23_boundary_0626",
    "enable_highres_structure_stream": True,
    "highres_structure_channels": 64,
    "highres_structure_fuse_stages": "stage23",
    "highres_structure_fusion_mode": "stage23",
    "stage_skeleton_mode": "direct",
    "enable_global_topology": True,
    "enable_h3_surface_fusion": True,
    "remove_stage2_pre_topology_source": True,
}
MODEL_OPTIONS = (
    "bottleneck_type", "global_topology_max_nodes", "global_topology_heads",
    "global_topology_alpha_max", "final_topology_eta_init", "final_gap_rho_init",
    "stage2_skeleton_gradient_ratio", "stage3_skeleton_gradient_ratio",
    "stage3_gate_topology_gradient_ratio", "final_skeleton_gradient_ratio",
    "stage_skeleton_bias_init", "stage_skeleton_positive_prior",
)
TRAINING_FIELDS = (
    "seed", "max_epochs", "batch_size", "accumulation_steps", "use_ema",
    "amp_dtype", "random_crop_train", "random_crops_per_image", "overlap_stride",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path, static, device):
    print(f"Loading {'Static' if static else 'Dynamic'} H checkpoint: {path}", flush=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    required = dict(BASE_REQUIRED, static_h0_stage_fusion=static)
    actual = {"static_h0_stage_fusion": False, **saved}
    mismatches = {key: (actual.get(key), value)
                  for key, value in required.items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"{path}: checkpoint is outside this H0 protocol: {mismatches}")
    if saved.get("ablate_h0_stream") or saved.get("enable_post_refine_structure_interaction"):
        raise ValueError(f"{path}: unexpected H stream ablation or post-refine interaction")
    if checkpoint["topology_attention_version"] != TOPOLOGY_ATTENTION_VERSION:
        raise ValueError(f"{path}: incompatible topology attention version")

    cfg = get_config(SimpleNamespace(
        cfg=str(CFG), opts=saved.get("opts"), batch_size=1, img_size=TILE,
        zip=False, cache_mode=None, resume=None, accumulation_steps=1,
        use_checkpoint=False, amp_opt_level=None, tag=None, eval=True, throughput=False,
    ))
    options = {key: value for key, value in required.items()
               if key not in ("img_size", "source_patch_size", "use_ema")}
    options.update({key: saved[key] for key in MODEL_OPTIONS if key in saved})
    model = SwinUnet(cfg, img_size=TILE, num_classes=1, return_skeleton=True, **options)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    info = dict(
        path=str(path.resolve()), sha256=sha256(path),
        epoch=checkpoint.get("epoch"), weights="model_state_dict (raw)",
        model_options=options, training={key: saved.get(key) for key in TRAINING_FIELDS},
    )
    return model.to(device).eval(), info


def read_window(dataset, case, top, left):
    matches = [name for name in dataset.image_files if case in (name, Path(name).stem)]
    if len(matches) != 1:
        raise ValueError(f"Case {case!r} matched {len(matches)} images. Use --list-cases.")
    image_name = matches[0]
    mask_name = dataset._find_label_name(image_name, dataset.mask_dir)
    image_path = Path(dataset.image_dir) / image_name
    mask_path = Path(dataset.mask_dir) / mask_name
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None:
        raise ValueError(f"Cannot read RGB or GT: {image_path}, {mask_path}")
    shape = (SOURCE_SIZE, SOURCE_SIZE)
    if bgr.shape != (*shape, 3) or mask.shape != shape:
        raise ValueError(f"Expected aligned native 1500 RGB and GT, got {bgr.shape}, {mask.shape}")
    window = np.s_[top:top + TILE, left:left + TILE]
    rgb = cv2.cvtColor(bgr[window], cv2.COLOR_BGR2RGB)
    gt = mask[window] > 127
    return rgb, gt, image_path, mask_path


def model_input(rgb, device):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


@torch.inference_mode()
def run_window(path, static, image):
    model, info = load_model(path, static, image.device)
    result = model(image)
    logits = result[0] if isinstance(result, tuple) else result
    if logits.shape != (1, 1, TILE, TILE) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError(f"Invalid surface logits from {path}: {tuple(logits.shape)}")
    features = model.swin_unet.last_stage_features
    norms = {}
    expected = ("H0",) if static else H_STATES
    present = tuple(name for name in H_STATES if name in features)
    if present != expected:
        raise RuntimeError(f"{path}: expected states {expected}, got {present}")
    if not static:
        for name in H_STATES:
            feature = features[name]
            if feature.shape != (1, 64, TILE // 4, TILE // 4):
                raise RuntimeError(f"Unexpected {name} shape: {tuple(feature.shape)}")
            norm = torch.linalg.vector_norm(feature[0].float(), dim=0)
            if not bool(torch.isfinite(norm).all()):
                raise RuntimeError(f"Nonfinite {name} norm")
            norms[name] = norm.cpu().numpy()
    return logits[0, 0].float().sigmoid().cpu().numpy(), norms, info


def save_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write PNG: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--top", type=int, default=512, help="Zero-based native y coordinate")
    parser.add_argument("--left", type=int, default=512, help="Zero-based native x coordinate")
    parser.add_argument("--dynamic-checkpoint", type=Path)
    parser.add_argument("--static-checkpoint", type=Path)
    parser.add_argument("--dynamic-threshold", type=float, default=0.45)
    parser.add_argument("--static-threshold", type=float, default=0.40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    dataset = RoadSkeletonDataset(str(args.root_path), split=args.split,
                                  image_size=TILE, source_patch_size=SOURCE_SIZE)
    if args.list_cases:
        print("\n".join(dataset.image_files))
        return
    if not all((args.case, args.dynamic_checkpoint, args.static_checkpoint, args.output_dir)):
        parser.error("Export requires --case, both checkpoints and --output_dir")
    if not (0 <= args.top <= SOURCE_SIZE - TILE and 0 <= args.left <= SOURCE_SIZE - TILE):
        parser.error(f"--top and --left must be in [0, {SOURCE_SIZE - TILE}]")
    if not (0 <= args.static_threshold <= 1 and 0 <= args.dynamic_threshold <= 1):
        parser.error("Thresholds must be in [0, 1]")
    for path in (args.dynamic_checkpoint, args.static_checkpoint):
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
    if args.output_dir.exists():
        parser.error(f"Use a new output directory: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select a GPU or use --device cpu.")

    rgb, gt, image_path, mask_path = read_window(dataset, args.case, args.top, args.left)
    image = model_input(rgb, torch.device(args.device))
    dynamic, norms, dynamic_info = run_window(args.dynamic_checkpoint, False, image)
    static, _, static_info = run_window(args.static_checkpoint, True, image)
    dynamic_mask = dynamic >= args.dynamic_threshold
    static_mask = static >= args.static_threshold
    gained, lost = dynamic_mask & ~static_mask, static_mask & ~dynamic_mask

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    save_png(out / "rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    save_png(out / "gt.png", gt.astype(np.uint8) * 255)
    save_png(out / "static_mask.png", static_mask.astype(np.uint8) * 255)
    save_png(out / "dynamic_mask.png", dynamic_mask.astype(np.uint8) * 255)
    np.savez_compressed(out / "h_norms.npz", **norms)
    np.savez_compressed(out / "surface_probabilities.npz", static=static, dynamic=dynamic)
    metadata = dict(
        case_id=image_path.stem, split=args.split, image=str(image_path), gt=str(mask_path),
        window=dict(top=args.top, left=args.left, height=TILE, width=TILE),
        protocol="single native 512x512 window; RGB ImageNet normalization; no TTA or stitching",
        config=dict(file=CFG.name, sha256=sha256(CFG)),
        dynamic=dict(threshold=args.dynamic_threshold, **dynamic_info),
        static=dict(threshold=args.static_threshold, **static_info),
        feature_response=dict(states=list(H_STATES), shape=[64, 128, 128],
                              statistic="channel L2 norm in FP32", source="Dynamic H"),
        changed_pixels=dict(
            gt_foreground=int(gt.sum()), true_road_gained=int((gained & gt).sum()),
            true_road_lost=int((lost & gt).sum()), false_positive_removed=int((lost & ~gt).sum()),
            false_positive_added=int((gained & ~gt).sum()),
        ),
    )
    (out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {image_path.stem} to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
