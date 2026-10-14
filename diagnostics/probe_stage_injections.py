"""Paired, checkpoint-only Stage 2/3 high-resolution injection probe."""

import argparse
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


MODES = ("baseline", "stage2_off", "stage3_off", "both_off")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MODEL_FILES = (
    "networks/vision_transformer.py",
    "networks/swin_transformer_unet_skip_expand_decoder_sys.py",
    "networks/skeleton_guided_head.py",
)
MODEL_BLOBS = {
    "h0": (
        "9bea85f391344b8803f4efa82b6e55e0aa7cfd72",
        "4aa0379d9316cd019a29517151f8ae0dc305af6d",
        "078474a6c3907d66c617989b324a17744fa5c2e7",
    ),
    "e202": (
        "551739b9106a666129c55b823f4c89f1adf0a1d9",
        "a928cb6998517a0f6aa395d9848161e8c62bec5f",
        "e7bb52e1ed23888a6c4eb249ddf2d752c4582345",
    ),
}


def git_blob_id(path):
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def matches_model_code(root, family):
    return all((root / path).is_file() and git_blob_id(root / path) == expected
               for path, expected in zip(MODEL_FILES, MODEL_BLOBS[family]))


def discover_code_root(saved_cfg, family):
    candidates = []
    if saved_cfg.is_absolute():
        candidates.append(saved_cfg.parent.parent)
    home = Path.home()
    for pattern in ("SwinNet*/networks/vision_transformer.py",
                    "SwinNet*/*/networks/vision_transformer.py"):
        candidates.extend(path.parent.parent for path in home.glob(pattern))
    for root in dict.fromkeys(candidates):
        if matches_model_code(root, family):
            return root
    raise FileNotFoundError(
        f"No checkout matching {family} model code under {home}/SwinNet*. "
        "Pass --code_root pointing to the exact training code checkout."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code_root", type=Path,
                        help="Defaults to the checkout containing checkpoint args.cfg")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap_stride", type=int, default=256)
    parser.add_argument("--weight_floor", type=float, required=True,
                        help="Use 0.05 for 599a410 H0; 0.10 for e2020d9")
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.max_images < 1:
        parser.error("threshold must be in (0,1) and max_images must be positive")
    if not 0 < args.overlap_stride <= args.tile_size:
        parser.error("overlap_stride must be in [1,tile_size]")
    if not 0 < args.weight_floor < 1:
        parser.error("weight_floor must be in (0,1)")
    return args


def load_model(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a training checkpoint with model_state_dict and args")
    weights = checkpoint["model_state_dict"]
    saved = checkpoint.get("args") or {}
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint args are not a dictionary")
    saved_cfg = Path(saved.get("cfg") or "")
    family = "h0" if any("highres_skeleton_adapter." in key for key in weights) else "e202"
    if args.code_root is None:
        args.code_root = discover_code_root(saved_cfg, family)
    args.code_root = args.code_root.resolve()
    if not (args.code_root / "networks/vision_transformer.py").is_file():
        raise FileNotFoundError(f"Not a model code checkout: {args.code_root}")
    print(f"[PROBE] model family={family}; model code={args.code_root}", flush=True)
    sys.path.insert(0, str(args.code_root))
    from config import get_config
    from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state

    cfg = args.code_root / "configs/swin_tiny_patch4_window7_224_lite.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"Missing model config in {args.code_root}: {cfg}")
    options = SimpleNamespace(
        cfg=str(cfg), opts=None, batch_size=1, img_size=args.tile_size,
        zip=False, cache_mode="", resume="", accumulation_steps=1,
        use_checkpoint=False, amp_opt_level="", tag="", eval=False,
        throughput=False,
    )
    config = get_config(options)
    signature = inspect.signature(SwinUnet)
    architecture = {name: saved[name] for name in signature.parameters
                    if name in saved and name not in {"config", "img_size", "num_classes"}}
    architecture.update(img_size=args.tile_size, num_classes=1, return_skeleton=True)
    architecture["structure_profile"] = checkpoint.get(
        "structure_profile", saved.get("structure_profile", "stage23_boundary_0626")
    )
    model = SwinUnet(config=config, **architecture)
    try:
        from networks.checkpoint_compat import adapt_connectivity_modules_for_checkpoint
    except ImportError:
        from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint
    adapt_connectivity_modules_for_checkpoint(model, weights, "standard")
    load_topology_checkpoint_state(
        model, weights, checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=True,
    )
    net = model.swin_unet
    for stage in (2, 3):
        prefix = f"swin_unet.highres_structure_fusion.{stage}."
        if not any(key.startswith(prefix) for key in weights):
            raise RuntimeError(f"Checkpoint lacks Stage {stage} injection weights: {prefix}")
        if str(stage) not in net.highres_structure_fusion:
            raise RuntimeError(f"Runtime lacks Stage {stage} injection")
    if not net.enable_highres_structure_stream or net.highres_structure_fusion_mode not in (
        "stage23", "stage23_final_correction"
    ):
        raise RuntimeError("Checkpoint did not enable both high-resolution Stage injections")
    source = "H0 adapter" if any("highres_skeleton_adapter." in key for key in weights) else "z_struct"
    return model.to(args.device).eval(), checkpoint, source


def positions(length, tile, stride):
    if length <= tile:
        return [0]
    return list(range(0, length - tile + 1, stride)) + (
        [length - tile] if (length - tile) % stride else []
    )


def load_cases(root, split, maximum):
    image_dir = root / split / "image"
    label_dir = root / split / "label"
    if not image_dir.is_dir():
        image_dir = root / split
    if not label_dir.is_dir():
        label_dir = root / f"{split}_labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"No label directory for {split}: {label_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in images[:maximum]:
        candidates = [image_path.name]
        for stem in (image_path.stem, image_path.stem.replace("_sat", "_mask")):
            candidates.extend(f"{stem}{extension}" for extension in (".png", ".tif", ".tiff"))
        label_path = next((label_dir / name for name in candidates
                           if (label_dir / name).is_file()), None)
        if label_path is None:
            raise FileNotFoundError(f"No label for {image_path.name} in {label_dir}")
        yield image_path, label_path


class InjectionHooks:
    def __init__(self, model):
        self.mode = "baseline"
        self.stats = {2: [], 3: []}
        self.handles = []
        for stage in (2, 3):
            module = model.swin_unet.highres_structure_fusion[str(stage)]
            self.handles.append(module.register_forward_hook(self.make_hook(stage)))

    def make_hook(self, stage):
        def hook(module, inputs, output):
            base = inputs[0]
            delta = output - base
            ratio = torch.linalg.vector_norm(delta.float()) / (
                torch.linalg.vector_norm(base.float()) + 1e-8
            )
            self.stats[stage].append((float(ratio), float(delta.float().abs().mean())))
            if self.mode == f"stage{stage}_off" or self.mode == "both_off":
                return base
            return None
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def infer_full(model, image, args, hooks, mode, tile_weight):
    hooks.mode = mode
    hooks.stats = {2: [], 3: []}
    height, width = image.shape[:2]
    logit_sum = torch.zeros((height, width), device=args.device)
    weight_sum = torch.zeros_like(logit_sum)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for top in positions(height, args.tile_size, args.overlap_stride):
        for left in positions(width, args.tile_size, args.overlap_stride):
            crop = image[top:top + args.tile_size, left:left + args.tile_size]
            valid_h, valid_w = crop.shape[:2]
            crop = np.pad(crop, ((0, args.tile_size - valid_h),
                                 (0, args.tile_size - valid_w), (0, 0)))
            normalized = ((crop.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
            inputs = torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0).to(args.device)
            result = model(inputs)
            logits = result[0] if isinstance(result, (tuple, list)) else result
            weight = tile_weight[:valid_h, :valid_w]
            logit_sum[top:top + valid_h, left:left + valid_w] += logits[0, 0, :valid_h, :valid_w] * weight
            weight_sum[top:top + valid_h, left:left + valid_w] += weight
    if not torch.all(weight_sum > 0):
        raise RuntimeError("Sliding window left uncovered pixels")
    probability = torch.sigmoid(logit_sum / weight_sum.clamp_min(1e-8)).cpu().numpy()
    stats = {f"stage{stage}_{key}": float(np.mean([sample[index] for sample in hooks.stats[stage]]))
             for stage in (2, 3) for index, key in enumerate(("relative_l2", "delta_mean_abs"))
             if hooks.stats[stage]}
    for stage in (2, 3):
        if not hooks.stats[stage]:
            raise RuntimeError(f"Stage {stage} injection was not called")
    return probability, stats


def gap_components(gt_skeleton, pred):
    missed = np.asarray(gt_skeleton & ~pred, dtype=np.uint8)
    return int(cv2.connectedComponents(missed, connectivity=8)[0] - 1)


def zhang_suen_skeletonize(mask):
    image = np.asarray(mask, dtype=np.uint8).copy()
    center = image[1:-1, 1:-1]
    if not center.size:
        return image.astype(bool)
    while True:
        changed = False
        for phase in (0, 1):
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = np.zeros_like(center)
            for index in range(8):
                transitions += (neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)
            if phase == 0:
                keep_clear = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                keep_clear = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = ((center == 1) & (count >= 2) & (count <= 6)
                      & (transitions == 1) & keep_clear)
            if np.any(remove):
                center[remove] = 0
                changed = True
        if not changed:
            return image.astype(bool)


def skeletonize(mask):
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning") and hasattr(ximgproc, "THINNING_ZHANGSUEN"):
        return ximgproc.thinning(
            np.asarray(mask, dtype=np.uint8) * 255,
            thinningType=ximgproc.THINNING_ZHANGSUEN,
        ) > 127
    return zhang_suen_skeletonize(mask)


def summarize(rows):
    numeric = sorted({key for row in rows for key, value in row.items()
                      if isinstance(value, (int, float))})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in numeric}


def main():
    args = parse_args()
    ximgproc = getattr(cv2, "ximgproc", None)
    skeleton_method = ("OpenCV Zhang-Suen" if ximgproc is not None
                       and hasattr(ximgproc, "thinning")
                       and hasattr(ximgproc, "THINNING_ZHANGSUEN")
                       else "NumPy Zhang-Suen fallback")
    print(f"[PROBE] GT skeleton method: {skeleton_method}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, source = load_model(args)
    hooks = InjectionHooks(model)
    weight_1d = (1 - torch.linspace(-1, 1, args.tile_size, device=args.device).abs()).clamp_min(
        args.weight_floor
    )
    tile_weight = torch.outer(weight_1d, weight_1d)
    rows = []
    try:
        with torch.inference_mode():
            for index, (image_path, label_path) in enumerate(
                load_cases(args.root_path, args.split, args.max_images), start=1
            ):
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                if bgr is None or label is None:
                    raise RuntimeError(f"Cannot read image/label: {image_path}, {label_path}")
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if image.shape[:2] != label.shape:
                    raise ValueError(f"Image/label size mismatch for {image_path.name}")
                gt_skeleton = skeletonize(label > 127)
                results = {}
                for mode in MODES:
                    probability, stats = infer_full(model, image, args, hooks, mode, tile_weight)
                    prediction = probability >= args.threshold
                    results[mode] = (probability, prediction)
                    rows.append(dict(case=image_path.stem, mode=mode, **stats,
                                     break_pixels=int((gt_skeleton & ~prediction).sum()),
                                     gap_components=gap_components(gt_skeleton, prediction),
                                     foreground_pixels=int(prediction.sum())))
                base_prob, base_pred = results["baseline"]
                for mode in MODES[1:]:
                    prob, pred = results[mode]
                    row = rows[-len(MODES) + MODES.index(mode)]
                    row["probability_abs_mean_vs_baseline"] = float(np.abs(prob - base_prob).mean())
                    row["changed_pixel_fraction_vs_baseline"] = float((pred != base_pred).mean())
                    row["new_break_pixels_vs_baseline"] = int((gt_skeleton & base_pred & ~pred).sum())
                    row["recovered_break_pixels_vs_baseline"] = int((gt_skeleton & ~base_pred & pred).sum())
                    changed_breaks = (gt_skeleton & (pred != base_pred)).astype(np.uint8) * 255
                    cv2.imwrite(str(args.output_dir / f"{image_path.stem}_{mode}_changed_breaks.png"),
                                changed_breaks)
                print(f"[{index}/{args.max_images}] {image_path.name} done", flush=True)
    finally:
        hooks.close()
    if not rows:
        raise RuntimeError("No cases evaluated")
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_epoch=checkpoint.get("epoch"),
                  code_root=str(args.code_root), split=args.split, threshold=args.threshold,
                  skeleton_method=skeleton_method,
                  source=source, images=len(rows) // len(MODES), tile_size=args.tile_size,
                  overlap_stride=args.overlap_stride, weight_floor=args.weight_floor,
                  modes={mode: summarize([
                      row for row in rows if row["mode"] == mode]) for mode in MODES},
                  note="Paired inference-only intervention; not a retrained ablation")
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
