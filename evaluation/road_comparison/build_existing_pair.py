"""Reuse two experiments' original val selections and test masks for graph evaluation."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


DEFAULT_NAMES = (
    "h0_shared_patch_mass_random512_fp32_80e_bs2",
    "h0_no_cnn_no_h0_global_mass_random512_fp32_80e_bs2",
)
SKIP = {"miniconda3", ".conda", ".cache", ".git", "data1", "Massachusetts",
        "predictions", "pretrained_ckpt", "node_modules", "__pycache__"}


def identity(model):
    content = {"name": model["name"], "provenance": model["provenance"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def absolute(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def find_candidates(search_root, data_root, names):
    found = {name: [] for name in names}
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [name for name in dirs if name not in SKIP]
        if "comparison.json" not in files:
            continue
        manifest_path = Path(root) / "comparison.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if absolute(manifest["data_root"], manifest_path.parent) != data_root:
                continue
            if manifest.get("protocol", {}).get("image_size") != 1500:
                continue
            choices = (manifest_path.parent / "val" / "val_selection.json",
                       manifest_path.parent / "val_selection.json")
            for selection_path in choices:
                if not selection_path.is_file():
                    continue
                selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
                if selection.get("selected_on") != "val" or selection.get("criterion") not in {"global_iou", "global_f1"}:
                    continue
                for model in manifest["models"]:
                    name = model.get("name")
                    if name not in found or name not in selection.get("models", {}):
                        continue
                    chosen = selection["models"][name]
                    if chosen.get("identity") != identity(model):
                        continue
                    threshold = float(chosen["threshold"])
                    # These are the two previously reported Massachusetts runs.
                    if abs(threshold - 0.45) > 1e-12:
                        continue
                    matches = [item for item in model["predictions"]["test"]
                               if abs(float(item["threshold"]) - threshold) < 1e-12]
                    if len(matches) != 1 or not absolute(matches[0]["path"], manifest_path.parent).is_dir():
                        continue
                    found[name].append((manifest_path, selection_path, model, threshold,
                                        selection["criterion"]))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_root", type=Path, default=Path("/home/gjj"))
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    found = find_candidates(args.search_root, data_root, DEFAULT_NAMES)
    models = []
    for name in DEFAULT_NAMES:
        candidates = found[name]
        if not candidates:
            parser.error(f"No validated original manifest, val selection and test masks found for {name}")
        checkpoints = {str(item[2]["provenance"].get("checkpoint", "")) for item in candidates}
        if len(checkpoints) != 1 or "" in checkpoints:
            parser.error(f"Multiple or missing checkpoints for {name}; inspect: " +
                         ", ".join(str(item[0]) for item in candidates))
        source, selection, original, threshold, criterion = max(
            candidates, key=lambda item: item[0].stat().st_mtime)
        model = copy.deepcopy(original)
        for split in ("val", "test"):
            for candidate in model["predictions"].get(split, []):
                candidate["path"] = str(absolute(candidate["path"], source.parent))
        model["validation_selection"] = {
            "selected_on": "val", "criterion": criterion,
            "threshold": threshold, "evidence": str(selection),
        }
        models.append(model)
        print(f"{name}: threshold={threshold} checkpoint={checkpoints.pop()}\n"
              f"  manifest={source}\n  selection={selection}\n"
              f"  masks={next(p['path'] for p in model['predictions']['test'] if abs(float(p['threshold']) - threshold) < 1e-12)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"data_root": str(data_root),
              "protocol": {"image_size": 1500, "short_area_threshold": 20,
                           "apls_max_control_nodes": 150, "apls_snap_radius": 5.0,
                           "topo_seed_density_deg": 0.002, "topo_step_deg": 0.0002},
              "models": models}
    path = args.output_dir / "comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {path}")


if __name__ == "__main__":
    main()
