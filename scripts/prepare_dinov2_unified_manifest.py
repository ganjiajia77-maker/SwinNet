"""Connect DINOv2-H0 prediction masks to the shared Data1 metric protocol."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


MODEL_NAME = "dinov2_h0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--selection", type=Path, help="selection produced by unified val evaluation")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / "best.pth"
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    val_dir = run_dir / "unified_val_predictions"
    val_report = json.loads((val_dir / "best_threshold.json").read_text(encoding="utf-8"))
    if val_report.get("split") != "val" or Path(val_report["checkpoint"]).resolve() != checkpoint:
        parser.error("Validation masks must come from this best.pth")
    if val_report.get("tile_size") != 512 or val_report.get("overlap_stride") != 256:
        parser.error("Unexpected validation tiling; review provenance before comparing")

    candidates = []
    with (val_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            threshold = float(row["threshold"])
            mask_dir = val_dir / f"threshold_{threshold:.12g}" / "surface"
            if not mask_dir.is_dir() or not any(mask_dir.glob("*_pred.png")):
                parser.error(f"Missing validation masks: {mask_dir}")
            candidates.append({"threshold": threshold, "path": str(mask_dir)})
    if not candidates:
        parser.error("No validation thresholds found")

    provenance = {
        "checkpoint": str(checkpoint),
        "training_code_commit": "da30147",
        "pretrained_source": "/home/gjj/Swin-Unet-main/pretrained_ckpt/dinov2_converted.pth",
        "training_crop": "one random 512 crop per 1024 image",
        "encoder": "frozen DINOv2-L/16",
        "checkpoint_weights": "EMA model_state_dict",
        "inference_tile_size": 512,
        "overlap_stride": 256,
        "merge": "taper-weighted logits -> sigmoid",
        "tta": "none",
        "postprocessing": "surface threshold only",
    }
    model = {"name": MODEL_NAME, "provenance": provenance,
             "predictions": {"val": candidates, "test": []}}
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        selected = selection.get("models", {}).get(MODEL_NAME)
        if selection.get("selected_on") != "val" or not selected:
            parser.error("Selection must come from unified validation evaluation")
        identity = digest({"name": MODEL_NAME, "provenance": provenance})
        if selected.get("identity") != identity:
            parser.error("Checkpoint or inference provenance differs from validation selection")
        threshold = float(selected["threshold"])
        if threshold not in {item["threshold"] for item in candidates}:
            parser.error("Selected threshold has no validation masks")
        test_mask_dir = run_dir / "unified_test_predictions" / "surface"
        existing_test_dir = run_dir / "test_selected_threshold"
        existing_report_path = existing_test_dir / "test_results.json"
        if existing_report_path.is_file():
            existing = json.loads(existing_report_path.read_text(encoding="utf-8"))
            if (existing.get("split") == "test"
                    and Path(existing["checkpoint"]).resolve() == checkpoint
                    and abs(float(existing["threshold"]) - threshold) < 1e-12
                    and existing.get("tile_size") == 512
                    and existing.get("overlap_stride") == 256
                    and (existing_test_dir / "surface").is_dir()):
                test_mask_dir = existing_test_dir / "surface"
        model["predictions"]["test"] = [
            {"threshold": threshold, "path": str(test_mask_dir)}
        ]
        write_json(run_dir / "unified_selected_threshold.json", {
            "split": "val", "checkpoint": str(checkpoint), "threshold": threshold,
            "selection": str(args.selection.resolve()),
        })
        print(f"Selected test mask directory: {test_mask_dir}", flush=True)

    manifest = {
        "data_root": str(args.data_root.resolve()),
        "protocol": {"image_size": int(val_report["evaluation_size"]),
                     "short_area_threshold": 20, "apls_max_nodes": 64,
                     "apls_snap_radius": 5.0},
        "models": [model],
    }
    write_json(run_dir / "unified_manifest.json", manifest)
    print(f"Wrote {run_dir / 'unified_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
