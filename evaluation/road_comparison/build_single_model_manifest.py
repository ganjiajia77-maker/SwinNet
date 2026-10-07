"""Build a unified-metrics manifest for one random512 Swin checkpoint."""

import argparse
import json
import re
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--predictions_root", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training_commit", required=True)
    parser.add_argument("--inference_commit", required=True)
    parser.add_argument("--selection", type=Path, help="val_selection.json after validation")
    parser.add_argument("--check_only", action="store_true", help="verify checkpoint metadata before inference")
    args = parser.parse_args()

    model_name = "swin_random512_fp32_ema80"
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    expected = {
        "img_size": 512,
        "source_patch_size": 1024,
        "random_crop_train": True,
        "random_crops_per_image": 1,
        "amp_dtype": "none",
        "use_ema": True,
        "max_epochs": 80,
        "structure_profile": "stage23_boundary_0626",
        "enable_highres_structure_stream": True,
        "enable_global_topology": True,
        "enable_e128_stage_fusion": True,
        "enable_h3_surface_fusion": True,
        "remove_stage2_pre_topology_source": True,
        "stage_skeleton_mode": "direct",
    }
    mismatches = {key: (saved_args.get(key), value) for key, value in expected.items()
                  if saved_args.get(key) != value}
    if mismatches or checkpoint.get("ema_state_dict") is None:
        parser.error(f"Checkpoint is not the stated random512 FP32 EMA 80e run: {mismatches}; "
                     f"EMA state present={checkpoint.get('ema_state_dict') is not None}")
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    if args.check_only:
        print(f"Checkpoint verified: random512 FP32 EMA, max_epochs=80, saved_epoch={checkpoint_epoch}")
        return
    val_root = args.predictions_root / "val"
    candidates = []
    for directory in sorted(val_root.iterdir()):
        match = re.fullmatch(r"threshold_(\d+\.\d{2})", directory.name)
        if match and (directory / "surface").is_dir():
            candidates.append({"threshold": float(match.group(1)), "path": str(directory / "surface")})
    if not candidates:
        parser.error(f"No validation threshold mask directories in {val_root}")

    test = []
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection.get("selected_on") != "val" or model_name not in selection.get("models", {}):
            parser.error("Selection must be the validation result for this model")
        threshold = float(selection["models"][model_name]["threshold"])
        directory = args.predictions_root / "test" / f"threshold_{threshold:.2f}" / "surface"
        if not directory.is_dir():
            parser.error(f"Selected test mask directory not found: {directory}")
        test = [{"threshold": threshold, "path": str(directory)}]

    manifest = {
        "data_root": str(args.data_root),
        "protocol": {
            "image_size": 1024,
            "short_area_threshold": 20,
            "apls_max_nodes": 64,
            "apls_snap_radius": 5.0,
        },
        "models": [{
            "name": model_name,
            "provenance": {
                "checkpoint": str(args.checkpoint),
                "checkpoint_args_verified": True,
                "training_code_commit": args.training_commit,
                "inference_code_commit": args.inference_commit,
                "training_crop": "one random 512 crop per 1024 image per epoch",
                "training_precision": "FP32",
                "training_max_epochs": 80,
                "checkpoint_epoch": checkpoint_epoch,
                "weights": "EMA",
                "inference_tile_size": 512,
                "overlap_stride": 256,
                "merge": "taper-weighted logits -> sigmoid; denominator clamp 1e-8",
                "tta": "none",
                "postprocessing": "surface threshold only",
            },
            "predictions": {"val": candidates, "test": test},
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(candidates)} val thresholds, {len(test)} test threshold")


if __name__ == "__main__":
    main()
