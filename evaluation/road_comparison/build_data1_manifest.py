"""Build a single-model manifest for the unchanged unified mask comparison tool."""

import argparse
import json
from pathlib import Path

import torch


def directory_name(threshold):
    return f"thr{round(float(threshold) * 1000):03d}"


def _metadata(directory, checkpoint, split):
    metadata = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
    if metadata["split"] != split or Path(metadata["checkpoint"]).resolve() != checkpoint:
        raise ValueError(f"{directory} was generated for a different split or checkpoint")
    if (metadata["source_size"], metadata["tile_size"], metadata["stride"]) != (1024, 512, 256):
        raise ValueError("Inference metadata does not use the unified 1024/512/256 protocol")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--val_predictions", required=True)
    parser.add_argument("--test_predictions")
    parser.add_argument("--selected_threshold", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.test_predictions) != (args.selected_threshold is not None):
        parser.error("Provide both --test_predictions and --selected_threshold for test")
    checkpoint_path = Path(args.model_path).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("architecture") != "original_swin_unet_swin_t_window8_512_two_class":
        raise ValueError("Wrong checkpoint architecture")
    train_args = checkpoint["args"]
    val_root = Path(args.val_predictions).resolve()
    val_info = _metadata(val_root, checkpoint_path, "val")
    val_sources = [{"threshold": float(value),
                    "path": str(val_root / directory_name(value) / "surface")}
                   for value in val_info["thresholds"]]
    test_sources = []
    if args.test_predictions:
        test_root = Path(args.test_predictions).resolve()
        test_info = _metadata(test_root, checkpoint_path, "test")
        if (len(test_info["thresholds"]) != 1 or
                abs(test_info["thresholds"][0] - args.selected_threshold) > 1e-8):
            raise ValueError("Test masks must use only the validation-selected threshold")
        test_sources = [{"threshold": args.selected_threshold,
                         "path": str(test_root / directory_name(args.selected_threshold) / "surface")}]
    provenance = {
        "checkpoint": str(checkpoint_path),
        "code_commit": args.code_commit,
        "pretrained_source": train_args.get("pretrain_ckpt") or "none; trained from scratch",
        "pretrained_load": checkpoint.get("pretrain_info"),
        "training_crop": "one distinct 512x512 crop per 1024x1024 image and epoch",
        "inference_tile_size": 512,
        "overlap_stride": 256,
        "merge": "taper-weighted road logit; divide by weight sum; sigmoid",
        "tta": "none",
        "postprocessing": "binary surface threshold only",
        "weights": "raw; no EMA",
        "actual_run_verified": False,
    }
    manifest = {
        "data_root": str(Path(args.data_root).resolve()),
        "protocol": {"image_size": 1024, "short_area_threshold": 20,
                     "apls_max_nodes": 64, "apls_snap_radius": 5.0},
        "models": [{"name": "swin_unet_original_random512", "provenance": provenance,
                    "predictions": {"val": val_sources, "test": test_sources}}],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"Manifest: {output}; val thresholds={len(val_sources)}; "
          f"test thresholds={len(test_sources)}", flush=True)


if __name__ == "__main__":
    main()
