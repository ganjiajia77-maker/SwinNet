"""Compare saved binary road masks with a shared, versioned protocol."""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import time

import cv2
import networkx
import numpy as np
import scipy
import shapely

from road_metrics import (DEFAULT_PROTOCOL, confusion, image_metrics, prepare_target,
                          segmentation, summarize)
from official_graph_metrics import prepare_apls_reference


EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".npy"}
METRIC_SOURCE_FILES = (
    "compare_predictions.py", "road_metrics.py", "official_graph_metrics.py",
    "vendor/sat2graph/graph.py", "vendor/sat2graph/topo.py",
    "vendor/sat2graph/main.upstream.py.txt",
    "vendor/sat2graph/graph.upstream.py.txt", "vendor/sat2graph/topo.upstream.py.txt",
    "vendor/cosmiq/apls.upstream.py.txt", "vendor/cosmiq/apls_utils.upstream.py.txt",
    "vendor/metrics_delin/toolong_tooshort_metric.upstream.py.txt",
    "vendor/metrics_delin/utils.upstream.py.txt",
)

# Existing validation sweeps selected thresholds by pixel IoU only. Their
# selections remain usable when the topology protocol is upgraded.
LEGACY_V1_PROTOCOL = {
    "metric_version": "road-mask-v1.1", "image_size": 1024,
    "short_area_threshold": 20, "apls_max_nodes": 64, "apls_snap_radius": 5.0,
    "connectivity": 8, "skeleton": "numpy_zhang_suen_zero_padded",
    "snap_rule": "distance <= radius; ties use lowest row-major node",
    "segmentation_aggregation": "global TP/FP/FN",
    "topology_aggregation": "per-image mean; bidirectional APLS harmonic per image",
    "postprocessing": "none; use supplied binary masks without resizing",
    "gt_encoding": "single-channel or identical-channel RGB binary mask; predictions single-channel",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def case_id(path):
    stem = path.stem
    return re.sub(r"(?:_(?:surface_pred|mask_pred|pred|sat|image|img|mask|label))+$", "", stem)


def index_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if "skeleton" in path.stem.lower():
            raise ValueError(f"Use the surface mask directory, not skeleton predictions: {path}")
        key = case_id(path)
        if key in result:
            raise ValueError(f"Duplicate case {key}: {result[key]} and {path}")
        result[key] = path
    if not result:
        raise ValueError(f"No mask files in {directory}")
    return result


def read_binary(path, size, allow_replicated_rgb=False):
    array = (np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else
             cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED))
    if array is None:
        raise ValueError(f"Could not read mask: {path}")
    if allow_replicated_rgb and array.ndim == 3 and array.shape[2] == 3:
        if not (np.array_equal(array[:, :, 0], array[:, :, 1]) and
                np.array_equal(array[:, :, 0], array[:, :, 2])):
            raise ValueError(f"GT RGB channels differ; refusing color conversion: {path}")
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel mask: {path}")
    if array.shape != (size, size):
        raise ValueError(f"No implicit resizing: {path} is {array.shape}, expected {(size, size)}")
    values = np.unique(array)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1, 255]).all():
        raise ValueError(f"Not a binary 0/1 or 0/255 mask: {path}; values={values[:12]}")
    if 1 in values and 255 in values:
        raise ValueError(f"Mixed mask encodings: {path}")
    return array > 0


def resolve_path(value, base):
    path = Path(value)
    return path if path.is_absolute() else base / path


def label_index(root, split):
    for directory in (root / split / "mask", root / split / "label", root / f"{split}_labels"):
        if directory.is_dir():
            labels = index_files(directory)
            image_dir = root / split / "image"
            if not image_dir.is_dir():
                image_dir = root / split
            if image_dir.is_dir():
                images = {case_id(p) for p in image_dir.iterdir() if p.is_file() and
                          p.suffix.lower() in EXTENSIONS | {".jpg", ".jpeg"}}
                if images != set(labels):
                    raise ValueError("Image and GT case sets differ; repair the dataset index first")
            return labels
    raise FileNotFoundError(f"No {split} GT directory under {root}")


def prediction_index(source, base, labels):
    directory = resolve_path(source["path"], base)
    if (directory / "surface").is_dir():
        directory = directory / "surface"
    predictions = index_files(directory)
    missing, extra = set(labels) - set(predictions), set(predictions) - set(labels)
    if missing or extra:
        raise ValueError(f"Case set mismatch in {directory}: missing={sorted(missing)[:8]}, "
                         f"extra={sorted(extra)[:8]}; no silent intersection")
    return predictions


def threshold_of(source):
    threshold = float(source["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {source['threshold']}")
    return threshold


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(manifest):
    protocol = dict(DEFAULT_PROTOCOL)
    overrides = dict(manifest.get("protocol", {}))
    adjustable = {"image_size", "short_area_threshold", "apls_max_nodes", "apls_snap_radius",
                  "apls_control_spacing_px", "apls_min_path_px", "tlts_n_paths",
                  "tlts_max_node_dist_px", "tlts_length_tolerance", "tlts_seed",
                  "topo_seed_density_deg", "topo_step_deg"}
    if set(overrides) - adjustable:
        raise ValueError(f"Unsupported protocol settings: {set(overrides) - adjustable}")
    if "apls_max_nodes" in overrides:
        protocol["legacy_apls_max_nodes_ignored"] = overrides.pop("apls_max_nodes")
    if "apls_control_spacing_px" in overrides:
        protocol["legacy_apls_control_spacing_px_ignored"] = overrides.pop(
            "apls_control_spacing_px")
    protocol.update(overrides)
    for key in ("image_size", "short_area_threshold", "legacy_apls_max_nodes_ignored",
                "tlts_n_paths"):
        value = protocol[key]
        if not isinstance(value, int) or value < (2 if key == "legacy_apls_max_nodes_ignored" else 1):
            raise ValueError(f"Invalid {key}: {value}")
    for key in ("apls_snap_radius", "legacy_apls_control_spacing_px_ignored",
                "apls_min_path_px",
                "topo_seed_density_deg", "topo_step_deg"):
        if not np.isfinite(protocol[key]) or protocol[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(protocol["tlts_max_node_dist_px"]) or protocol["tlts_max_node_dist_px"] < 3:
        raise ValueError("tlts_max_node_dist_px must be at least 3 pixels")
    if not np.isfinite(protocol["tlts_length_tolerance"]) or not 0 <= protocol["tlts_length_tolerance"] < 1:
        raise ValueError("tlts_length_tolerance must be in [0, 1)")
    if not isinstance(protocol["tlts_seed"], int) or not 0 <= protocol["tlts_seed"] < 2**64:
        raise ValueError("tlts_seed must be a nonnegative 64-bit integer")
    names = [m["name"] for m in manifest["models"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Model names must be unique and nonempty")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Model names may contain only ASCII letters, digits, _ and -")
    return protocol


def model_identity(model):
    return digest({"name": model["name"], "provenance": model["provenance"]})


def threshold_protocol_hash(protocol):
    return digest({"image_size": protocol["image_size"],
                   "gt_encoding": protocol["gt_encoding"],
                   "segmentation_aggregation": protocol["segmentation_aggregation"]})


def compatible_selection(selection, protocol):
    if selection.get("selected_on") != "val":
        return False
    if selection.get("protocol_hash") == digest(protocol):
        return True
    if selection.get("schema") == 2 and selection.get("threshold_protocol_hash") == threshold_protocol_hash(protocol):
        return True
    old = dict(LEGACY_V1_PROTOCOL)
    for key in ("image_size", "short_area_threshold", "apls_snap_radius"):
        old[key] = protocol[key]
    old["apls_max_nodes"] = protocol["legacy_apls_max_nodes_ignored"]
    return selection.get("protocol_hash") == digest(old)


def select_validation(manifest, base, protocol, output, progress):
    labels = label_index(resolve_path(manifest["data_root"], base), "val")
    selection = {"schema": 2, "selected_on": "val", "criterion": "global_iou",
                 "protocol_hash": digest(protocol),
                 "threshold_protocol_hash": threshold_protocol_hash(protocol),
                 "val_cases_hash": digest(sorted(labels)),
                 "val_images": len(labels), "models": {}}
    sweep = []
    for model in manifest["models"]:
        candidates = model["predictions"]["val"]
        if not candidates:
            raise ValueError(f"No validation masks supplied for {model['name']}")
        thresholds = [threshold_of(candidate) for candidate in candidates]
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("Duplicate validation threshold")
        indexed = [(candidate, prediction_index(candidate, base, labels)) for candidate in candidates]
        totals = [{k: 0 for k in ("tp", "fp", "fn")} for _ in indexed]
        for i, (case, path) in enumerate(sorted(labels.items()), 1):
            target = read_binary(path, protocol["image_size"], allow_replicated_rgb=True)
            for total, (_, predictions) in zip(totals, indexed):
                counts = confusion(read_binary(predictions[case], protocol["image_size"]), target)
                for key in total:
                    total[key] += counts[key]
            if i == 1 or i % progress == 0 or i == len(labels):
                print(f"[val] {model['name']} {i}/{len(labels)}", flush=True)
        rows = [{"name": model["name"], "threshold": threshold_of(candidate),
                 **total, **segmentation(total)} for (candidate, _), total in zip(indexed, totals)]
        best = max(rows, key=lambda row: (row["iou"], -row["threshold"]))
        selection["models"][model["name"]] = {
            "identity": model_identity(model), "threshold": best["threshold"],
            "val_iou": best["iou"], "candidates_evaluated": len(rows),
            "note": "best among supplied masks only; not an optimum over unsaved thresholds",
        }
        sweep.extend(rows)
    write_csv(output / "val_threshold_scores.csv", sweep)
    write_json(output / "val_selection.json", selection)
    return selection


def import_declared_selection(manifest, base, protocol):
    # This checks evidence existence, not the semantic content of a handwritten log.
    selection = {"schema": 2, "selected_on": "val", "criterion": "declared_validation_result",
                 "protocol_hash": digest(protocol),
                 "threshold_protocol_hash": threshold_protocol_hash(protocol), "models": {}}
    for model in manifest["models"]:
        declared = model["validation_selection"]
        if declared["selected_on"] != "val" or declared["criterion"] not in {"global_iou", "global_f1"}:
            raise ValueError("Imported threshold must have been selected on val by global IoU/F1")
        evidence = resolve_path(declared["evidence"], base)
        if not evidence.is_file() or not evidence.stat().st_size:
            raise FileNotFoundError(f"Validation threshold evidence missing: {evidence}")
        selection["models"][model["name"]] = {
            "identity": model_identity(model), "threshold": threshold_of(declared),
            "evidence": str(evidence), "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "note": "user-declared val selection; log contents not parsed or re-evaluated",
        }
    return selection


def metric_source_hashes():
    return {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in METRIC_SOURCE_FILES
    }


def input_signature(paths):
    signature = hashlib.sha256()
    for path in paths:
        path = Path(path)
        signature.update(str(path.resolve()).encode("utf-8") + b"\0")
        signature.update(hashlib.sha256(path.read_bytes()).digest())
    return signature.hexdigest()


def evaluate_case(job):
    case, label_path, model_paths, protocol, partial_path, run_id = job
    partial_path = Path(partial_path)
    paths = [label_path, *(path for _, path in model_paths)]
    source_hash = input_signature(paths)
    if partial_path.is_file():
        saved = json.loads(partial_path.read_text(encoding="utf-8"))
        if (saved.get("run_id") != run_id or saved.get("case_id") != case or
                saved.get("input_sha256") != source_hash or
                set(saved.get("rows", {})) != {name for name, _ in model_paths}):
            raise ValueError(f"Partial result does not match current inputs: {partial_path}")
        return saved, True

    target = read_binary(Path(label_path), protocol["image_size"],
                         allow_replicated_rgb=True)
    started = time.perf_counter()
    prepared = prepare_target(target, protocol)
    gt_seconds = time.perf_counter() - started
    print(f"[TIMING] {case} GT_graph={gt_seconds:.1f}s", flush=True)
    started = time.perf_counter()
    apls_reference = prepare_apls_reference(prepared["graph"])
    apls_gt_seconds = time.perf_counter() - started
    print(f"[TIMING] {case} APLS_GT_NATIVE={apls_gt_seconds:.1f}s "
          f"controls={len(apls_reference['controls'])} reused_for={len(model_paths)}_models",
          flush=True)
    rows = {}
    for name, path in model_paths:
        pred = read_binary(Path(path), protocol["image_size"])
        rows[name] = {"case_id": case, **image_metrics(
            pred, prepared, protocol, timing_label=f"{case} {name}")}
    result = {
        "case_id": case, "run_id": run_id, "input_sha256": source_hash,
        "timing_gt_graph_seconds": gt_seconds,
        "timing_apls_gt_native_seconds": apls_gt_seconds, "rows": rows,
    }
    temporary = partial_path.with_name(partial_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, partial_path)
    return result, False


def evaluate(manifest, base, protocol, split, selection, output, progress,
             workers=1, resume=False):
    if not compatible_selection(selection, protocol):
        raise ValueError("Selection must originate from val with this metric protocol or verified v1 protocol")
    labels = label_index(resolve_path(manifest["data_root"], base), split)
    models = []
    for model in manifest["models"]:
        selected = selection["models"][model["name"]]
        if selected["identity"] != model_identity(model):
            raise ValueError(f"Checkpoint/provenance changed for {model['name']}; reselect on val")
        matches = [source for source in model["predictions"][split]
                   if abs(threshold_of(source) - selected["threshold"]) < 1e-12]
        if len(matches) != 1:
            raise ValueError(f"Need exactly one {split} mask directory at val-selected threshold "
                             f"{selected['threshold']} for {model['name']}; cannot rethreshold binary PNG")
        models.append((model, prediction_index(matches[0], base, labels), selected))

    source_hashes = metric_source_hashes()
    run_state = {
        "split": split, "case_ids_sha256": digest(sorted(labels)),
        "manifest_sha256": digest(manifest), "selection_sha256": digest(selection),
        "protocol_sha256": digest(protocol), "metric_implementation_sha256": source_hashes,
    }
    run_id = digest(run_state)
    state_path = output / "run_state.json"
    if state_path.exists():
        if not resume:
            raise ValueError(f"Existing run requires --resume: {output}")
        if json.loads(state_path.read_text(encoding="utf-8")) != run_state:
            raise ValueError("Existing run has different inputs, protocol, or implementation")
    else:
        if any(path.name != "imported_val_selection.json" for path in output.iterdir()):
            raise ValueError(f"Output directory has files but no run_state.json: {output}")
        write_json(state_path, run_state)
    partial_dir = output / "partial"
    partial_dir.mkdir(exist_ok=True)

    jobs = []
    for case, label_path in sorted(labels.items()):
        paths = [(model["name"], str(predictions[case]))
                 for model, predictions, _ in models]
        partial_name = hashlib.sha256(case.encode("utf-8")).hexdigest() + ".json"
        jobs.append((case, str(label_path), paths, protocol,
                     str(partial_dir / partial_name), run_id))

    rows_by_model = {model["name"]: [] for model, _, _ in models}
    completed = {}
    started = time.perf_counter()
    if workers == 1:
        results = map(evaluate_case, jobs)
        for i, (result, cached) in enumerate(results, 1):
            completed[result["case_id"]] = result
            if i == 1 or i % progress == 0 or i == len(jobs):
                print(f"[{split}] {i}/{len(jobs)} {result['case_id']} "
                      f"{'cached' if cached else 'computed'}; "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    else:
        pool = multiprocessing.Pool(processes=workers)
        try:
            for i, (result, cached) in enumerate(pool.imap(evaluate_case, jobs), 1):
                completed[result["case_id"]] = result
                if i == 1 or i % progress == 0 or i == len(jobs):
                    print(f"[{split}] {i}/{len(jobs)} {result['case_id']} "
                          f"{'cached' if cached else 'computed'}; "
                          f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
        except BaseException:
            pool.terminate()
            raise
        else:
            pool.close()
        finally:
            pool.join()

    content_hashes = {"gt": hashlib.sha256(), **{m["name"]: hashlib.sha256()
                                                for m, _, _ in models}}
    for case, path in sorted(labels.items()):
        target = read_binary(path, protocol["image_size"], allow_replicated_rgb=True)
        content_hashes["gt"].update(case.encode() + b"\0" + np.packbits(target).tobytes())
        for model, predictions, _ in models:
            name = model["name"]
            pred = read_binary(predictions[case], protocol["image_size"])
            content_hashes[name].update(case.encode() + b"\0" + np.packbits(pred).tobytes())
            rows_by_model[name].append(completed[case]["rows"][name])
    for name, rows in rows_by_model.items():
        write_csv(output / f"{name}_{split}_per_image.csv", rows)

    summaries = []
    for model, _, selected in models:
        summaries.append({"name": model["name"], "threshold": selected["threshold"],
                          **summarize(rows_by_model[model["name"]])})
    report = {
        "split": split, "protocol": protocol, "threshold_selection": selection,
        "metric_implementation_sha256": source_hashes,
        "case_ids": sorted(labels), "case_ids_sha256": digest(sorted(labels)),
        "binary_content_sha256": {key: value.hexdigest() for key, value in content_hashes.items()},
        "provenance": {m["name"]: m["provenance"] for m, _, _ in models},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "opencv": cv2.__version__,
                        "networkx": networkx.__version__, "shapely": shapely.__version__},
        "summaries": summaries,
        "metric_runtime_seconds": time.perf_counter() - started,
        "workers": workers, "resumable_state_sha256": run_id,
        "timing_gt_graph_seconds_total": sum(
            completed[case]["timing_gt_graph_seconds"] for case in labels),
        "timing_note": "CPU mask evaluation only; no model inference timing can be obtained from PNG",
    }
    write_json(output / f"{split}_report.json", report)
    write_csv(output / f"{split}_comparison.csv", summaries)
    for row in summaries:
        print(f"{row['name']}: IoU={row['iou']:.6f} clDice={row['cldice']:.6f} "
              f"TOPO_F1={row['topo_f1']:.6f} APLS_graph={row['apls_bidirectional']:.6f} "
              f"TLTS_infeasible={row['tlts_infeasible']:.6f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--selection", type=Path, help="val_selection.json, required for test unless importing evidence")
    parser.add_argument("--import_val_selection", action="store_true", help="import existing val-selected thresholds and logs")
    parser.add_argument("--select_threshold_only", action="store_true",
                        help="on val, select threshold and write scores without computing topology")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent image processes; each process evaluates every model")
    parser.add_argument("--resume", action="store_true",
                        help="Continue matching partial image results in output_dir")
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress_every must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume and (args.split != "test" or args.select_threshold_only):
        parser.error("--resume is for test topology evaluation")
    if args.selection and args.import_val_selection:
        parser.error("Choose --selection or --import_val_selection, not both")
    if args.split == "val" and (args.selection or args.import_val_selection):
        parser.error("Val selects from supplied validation candidates; selection import is for test")
    if args.select_threshold_only and args.split != "val":
        parser.error("--select_threshold_only is valid only for val")
    if args.split == "test" and not (args.selection or args.import_val_selection):
        parser.error("Test never selects thresholds; provide --selection or --import_val_selection")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    base = args.manifest.resolve().parent
    protocol = validate_manifest(manifest)
    # Do not overwrite an earlier report, especially an interrupted partial run.
    if (args.output_dir.exists() and any(args.output_dir.iterdir()) and
            not args.resume):
        parser.error("output_dir is not empty; use a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "val":
        selection = select_validation(manifest, base, protocol, args.output_dir, args.progress_every)
        if args.select_threshold_only:
            print(f"Validation threshold selected; topology skipped. See {args.output_dir / 'val_selection.json'}", flush=True)
            return
    elif args.import_val_selection:
        selection = import_declared_selection(manifest, base, protocol)
        write_json(args.output_dir / "imported_val_selection.json", selection)
    else:
        selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
    evaluate(manifest, base, protocol, args.split, selection, args.output_dir,
             args.progress_every, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
