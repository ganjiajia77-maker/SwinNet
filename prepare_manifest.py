"""Bind WeavingUnet binary predictions to the existing unified Data1 metric tool."""

import argparse
import hashlib
import json
from pathlib import Path


NAME = "weavingunet_random512"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    checkpoint = run / "best.pth"
    report = json.loads((run / "unified_val_predictions" / "inference_report.json").read_text(encoding="utf-8"))
    if (report["split"] != "val" or Path(report["checkpoint"]).resolve() != checkpoint.resolve()
            or report["tile_size"] != 512 or report["overlap_stride"] != 256):
        parser.error("Validation predictions must be 512/256 tiles from this best.pth")
    thresholds = [float(value) for value in report["threshold_scores"]]
    val_sources = [{"threshold": threshold,
                    "path": str(run / "unified_val_predictions" / f"threshold_{threshold:.12g}" / "surface")}
                   for threshold in thresholds]
    for source in val_sources:
        if not Path(source["path"]).is_dir():
            parser.error(f"Missing masks: {source['path']}")
    provenance = {"checkpoint": str(checkpoint), "checkpoint_epoch": report["checkpoint_epoch"],
                  "pretrained_source": "torchvision EfficientNet-V2-S ImageNet-1K",
                  "training_crop": "one native-resolution random 512 crop per Data1 image per epoch",
                  "encoder": "trainable EfficientNet-V2-S", "checkpoint_weights": "model_state_dict (no EMA)",
                  "inference_tile_size": 512, "overlap_stride": 256,
                  "merge": "taper-weighted probabilities", "tta": False,
                  "postprocessing": "threshold only"}
    test_sources = []
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        picked = selection.get("models", {}).get(NAME)
        if (selection.get("selected_on") != "val" or picked is None or
                picked["identity"] != digest({"name": NAME, "provenance": provenance})):
            parser.error("Validation selection does not match this checkpoint and inference protocol")
        threshold = float(picked["threshold"])
        if threshold not in thresholds:
            parser.error("Selected threshold is not among saved val masks")
        test_sources = [{"threshold": threshold,
                         "path": str(run / "unified_test_predictions" / "surface")}]
        (run / "selected_threshold.json").write_text(json.dumps({"threshold": threshold,
             "selection": str(args.selection.resolve()), "checkpoint": str(checkpoint) }, indent=2) + "\n", encoding="utf-8")
        print(f"VAL_SELECTED_THRESHOLD={threshold}", flush=True)
    manifest = {"data_root": str(args.data_root.resolve()),
                "protocol": {"image_size": 1024, "short_area_threshold": 20,
                             "apls_max_nodes": 64, "apls_snap_radius": 5.0},
                "models": [{"name": NAME, "provenance": provenance,
                            "predictions": {"val": val_sources, "test": test_sources}}]}
    (run / "unified_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
