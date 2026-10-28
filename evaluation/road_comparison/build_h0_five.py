"""Rebuild the five H0 comparisons using the exact inputs of existing reports."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from compare_predictions import (digest, label_index, model_identity, prediction_index,
                                 read_binary, threshold_protocol_hash, validate_manifest)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def content_hash(index, gt=False):
    result = hashlib.sha256()
    for case, path in sorted(index.items()):
        mask = read_binary(path, 1500, allow_replicated_rgb=gt)
        result.update(case.encode() + b"\0" + np.packbits(mask).tobytes())
    return result.hexdigest()


def candidate_files(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"predictions", ".git", "__pycache__"}]
        for name in files:
            if name.startswith("comparison") and name.endswith(".json"):
                yield Path(base) / name


def find_prediction(files, data_root, name, selection, expected_hash, labels):
    for path in files:
        try:
            manifest = load(path)
            root = Path(manifest["data_root"])
            root = (root if root.is_absolute() else path.parent / root).resolve()
            if root != data_root:
                continue
            for original in manifest["models"]:
                if original["name"] != name or model_identity(original) != selection["identity"]:
                    continue
                for candidate in original["predictions"]["test"]:
                    if abs(float(candidate["threshold"]) - float(selection["threshold"])) > 1e-12:
                        continue
                    index = prediction_index(candidate, path.parent, labels)
                    if content_hash(index) != expected_hash:
                        continue
                    model = copy.deepcopy(original)
                    source = Path(candidate["path"])
                    if not source.is_absolute():
                        source = path.parent / source
                    model["predictions"] = {"val": [], "test": [{
                        "threshold": float(selection["threshold"]), "path": str(source.resolve())}]}
                    return model, path
        except (OSError, ValueError, TypeError, KeyError):
            continue
    raise ValueError(f"Could not find original manifest and matching mask content for {name} in {list(files)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--four_dir", type=Path, required=True)
    parser.add_argument("--no_direction_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    four_report = args.four_dir / "test_after_kd_fix_20261027_081428" / "test_report.json"
    no_dir_report = (args.no_direction_run / "unified_four_metrics_v7" /
                     "test_20261027_113946" / "test_report.json")
    reports = [(load(four_report), four_report, ["H0", "H0_A", "H0_B", "H0_C"],
                list(candidate_files(args.four_dir))),
               (load(no_dir_report), no_dir_report, ["h0_no_direction_mass"],
                list(candidate_files(args.no_direction_run)))]
    labels = label_index(data_root, "test")
    if len(labels) != 49:
        raise ValueError(f"Expected 49 test images, found {len(labels)}")
    gt_hash = content_hash(labels, gt=True)
    cases_hash = digest(sorted(labels))
    models, selections, evidence = [], {}, {}
    for report, report_path, names, candidates in reports:
        if report.get("split") != "test" or report.get("threshold_selection", {}).get("selected_on") != "val":
            raise ValueError(f"Not a test report with validation selection: {report_path}")
        if report.get("case_ids_sha256") != cases_hash or report.get("binary_content_sha256", {}).get("gt") != gt_hash:
            raise ValueError(f"Test cases or GT content changed: {report_path}")
        for name in names:
            selection = report["threshold_selection"]["models"][name]
            expected = report["binary_content_sha256"][name]
            model, source = find_prediction(candidates, data_root, name, selection, expected, labels)
            models.append(model)
            selections[name] = selection
            evidence[name] = str(report_path)
            print(f"{name}: threshold={selection['threshold']} checkpoint={model['provenance'].get('checkpoint')}\n"
                  f"  masks={model['predictions']['test'][0]['path']}\n  manifest={source}", flush=True)
    manifest = {"data_root": str(data_root),
                "protocol": {"image_size": 1500, "short_area_threshold": 20,
                             "apls_max_control_nodes": 150, "apls_snap_radius": 5.0,
                             "topo_seed_density_deg": 0.002, "topo_step_deg": 0.0002},
                "models": models}
    selection = {"schema": 2, "selected_on": "val", "criterion": "reused_saved_validation_selection",
                 "threshold_protocol_hash": threshold_protocol_hash(validate_manifest(manifest)),
                 "models": selections, "source_test_reports": evidence}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in (("comparison.json", manifest), ("val_selection.json", selection)):
        (args.output_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified GT, cases and all five prediction hashes. Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
