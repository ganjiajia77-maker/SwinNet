"""Build the unchanged unified comparison tool's single-model manifest."""

import argparse
import json
from pathlib import Path

import torch


def threshold_dir(value):
    return f"thr{round(float(value) * 1000):03d}"


def read_export(directory, checkpoint, split):
    data = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
    if data["split"] != split or Path(data["checkpoint"]).resolve() != checkpoint:
        raise ValueError(f"{directory} has different split/checkpoint")
    if (data["source_size"], data["tile_size"], data["stride"]) != (1024, 512, 256):
        raise ValueError("Inference metadata violates unified Data1 protocol")
    return data


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
        parser.error("Supply both --test_predictions and --selected_threshold for test")
    checkpoint_path = Path(args.model_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "upstream_darenet_lmz_512_binary":
        raise ValueError("Wrong checkpoint architecture")
    val_root = Path(args.val_predictions).resolve()
    val_data = read_export(val_root, checkpoint_path, "val")
    val_sources = [{"threshold": float(value),
                    "path": str(val_root / threshold_dir(value) / "surface")}
                   for value in val_data["thresholds"]]
    test_sources = []
    if args.test_predictions:
        test_root = Path(args.test_predictions).resolve()
        test_data = read_export(test_root, checkpoint_path, "test")
        if (len(test_data["thresholds"]) != 1 or
                abs(test_data["thresholds"][0] - args.selected_threshold) > 1e-8):
            raise ValueError("Test masks must use only validation-selected threshold")
        test_sources = [{"threshold": args.selected_threshold,
                         "path": str(test_root / threshold_dir(args.selected_threshold) / "surface")}]
    provenance = {
        "checkpoint": str(checkpoint_path), "code_commit": args.code_commit,
        "model_source": checkpoint.get("model_file"),
        "pretrained_source": "torchvision ResNet34 IMAGENET1K_V1 via upstream model",
        "training_crop": "one distinct 512x512 crop per native 1024 image per epoch",
        "inference_tile_size": 512, "overlap_stride": 256,
        "merge": "taper-weighted logits, then sigmoid", "tta": "none",
        "postprocessing": "binary surface threshold only", "weights": "raw; no EMA",
        "actual_run_verified": False,
    }
    manifest = {
        "data_root": str(Path(args.data_root).resolve()),
        "protocol": {"image_size": 1024, "short_area_threshold": 20,
                     "apls_max_nodes": 64, "apls_snap_radius": 5.0},
        "models": [{"name": "darenet_random512", "provenance": provenance,
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
