"""Run Massachusetts F-3 from archived, validation-selected test masks."""
import argparse
from collections import Counter
import csv
from functools import partial
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def candidates(roots):
    for root in roots:
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d not in {
                "partial", "predictions", "predictions_test", "masks", "maps",
                "cases", "checkpoints", ".git", "__pycache__", "locations",
            }]
            for name in files:
                if "comparison" in name and name.endswith(".json"):
                    yield Path(directory) / name


def archived_masks(cp, manifests, data, labels, checkpoints, protocol):
    found = {role: {} for role in checkpoints.values()}
    for path in manifests:
        try:
            original = load(path)
            if cp.resolve_path(original["data_root"], path.parent).resolve() != data:
                continue
            models = original["models"]
            selections = [path.parent / name for name in (
                "val/val_selection.json", "val_selection.json", "test/imported_val_selection.json",
                "test_six/imported_val_selection.json",
            )]
            for model in models:
                role = checkpoints.get(
                    cp.resolve_path(model["provenance"]["checkpoint"], path.parent).resolve()
                )
                if role is None:
                    continue
                for selection_path in selections:
                    if not selection_path.is_file():
                        continue
                    selection = load(selection_path)
                    if not cp.compatible_selection(selection, protocol):
                        continue
                    selected = selection["models"].get(model["name"], {})
                    if selected.get("identity") != cp.model_identity(model):
                        continue
                    threshold = selected["threshold"]
                    for source in model["predictions"]["test"]:
                        if abs(cp.threshold_of(source) - threshold) > 1e-8:
                            continue
                        try:
                            index = cp.prediction_index(source, path.parent, labels)
                        except (OSError, ValueError):
                            continue
                        mask_dir = next(iter(index.values())).parent.resolve()
                        found[role][(threshold, mask_dir)] = (model, path, selection_path, index)
        except (OSError, ValueError, TypeError, KeyError):
            continue

    result = []
    for expected_checkpoint, role in checkpoints.items():
        options = found.get(role, {})
        if not options:
            raise ValueError(f"{role}: no complete 49-image test mask set bound to a saved "
                             f"validation selection for {expected_checkpoint}")
        unique = {}
        for (threshold, folder), (model, manifest, selection, index) in options.items():
            signature = hashlib.sha256()
            for case, mask in sorted(index.items()):
                pred = cp.read_binary(mask, 1500)
                signature.update(case.encode() + b"\0" + np.packbits(pred).tobytes())
            unique[(threshold, signature.hexdigest())] = (folder, model, manifest, selection)
        if len(unique) != 1:
            raise ValueError(f"{role}: differing masks or thresholds in "
                             f"{[str(folder) for _, folder in options]}; use --manifest to choose")
        (threshold, content_hash), (folder, model, manifest, selection) = next(iter(unique.items()))
        provenance = dict(model["provenance"], original_model=model["name"],
                          comparison=str(manifest), selection=str(selection),
                          binary_mask_sha256=content_hash)
        print(f"[MASKS] {role} threshold={threshold} {folder}", flush=True)
        result.append(dict(name=role, provenance=provenance, predictions={
            "val": [], "test": [dict(threshold=threshold, path=str(folder))],
        }))
    return result


def native_xy(latlon):
    lat, lon = latlon[:2]
    return ((lon + 71.0) * 111111.0 * math.cos(math.radians(41)),
            (41.0 - lat) * 111111.0)


def captured_topo(original_score, vendor, gt, pred, protocol, rows):
    capture = dict(seeds=[], pairs={}, before=[], kept=[])

    def record(name, key):
        original = getattr(vendor, name)

        def call(*args, **kwargs):
            if name == "TOPO121":
                capture["before"] = list(args[0])
            result = original(*args, **kwargs)
            capture[key] = result
            return result
        return call

    hooks = {name: record(name, key) for name, key in (
        ("TOPOGenerateStartingPoints", "seeds"), ("TOPOGeneratePairs", "pairs"),
        ("TOPO121", "kept"),
    )}
    with patch.multiple(vendor, **hooks):
        score = original_score(gt, pred, protocol)
    before = {tuple(row[:2]): row for row in capture["before"]}
    kept = {tuple(row[:2]): row for row in capture["kept"]}
    for i, seed in enumerate(capture["seeds"]):
        key = tuple(seed[:2])
        pair = capture["pairs"].get(i)
        x, y = native_xy(seed)
        pred_x, pred_y = native_xy(pair[5:7]) if pair is not None else (None, None)
        local = kept.get(key, before.get(key))
        status = (
            "accepted" if key in kept else "excluded_by_one_to_one" if key in before
            else "empty_local_walk" if pair is not None else "no_nearby_seed_match"
        )
        rows.append(dict(seed_id=i, gt_x=x, gt_y=y, pred_x=pred_x, pred_y=pred_y,
                         status=status, local_precision=float(local[2]) if local is not None else None,
                         local_recall=float(local[3]) if local is not None else None))
    return score


def point_on_image(img, x, y, color):
    cv2.circle(img, (round(x), round(y)), 4, color, -1, cv2.LINE_AA)


def write_preview(directory, name, gt_skeleton, overlays, center, route=None):
    size = 160
    y, x = (int(round(v)) for v in center)
    y = min(max(y - size // 2, 0), max(gt_skeleton.shape[0] - size, 0))
    x = min(max(x - size // 2, 0), max(gt_skeleton.shape[1] - size, 0))
    region = np.s_[y:y + size, x:x + size]
    gt = cv2.cvtColor(gt_skeleton[region].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    panels = []
    for label, image in [("GT skeleton", gt), *((k, img[region]) for k, img in overlays.items())]:
        panel = image.copy()
        if route is not None:
            points = [(int(px) - x, int(py) - y) for py, px in route]
            for point in (points[0], points[-1]):
                cv2.circle(panel, point, 4, (20, 215, 255), 1)
        panel = cv2.copyMakeBorder(panel, 22, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(directory / f"{name}.png"), np.hstack(panels)):
        raise OSError(f"Could not write preview {name}")
    return dict(file=f"{name}.png", y0=y, x0=x, height=gt.shape[0], width=gt.shape[1])


def load_location_probes(diagnostics):
    legacy_path = diagnostics.parent / "evaluation" / "road_comparison" / "road_metrics.py"
    spec = spec_from_file_location("scgra_raster_metrics", legacy_path)
    raster = module_from_spec(spec)
    spec.loader.exec_module(raster)
    with patch.dict(sys.modules, {"road_metrics": raster}):
        spec = spec_from_file_location("scgra_location_probes", diagnostics / "scgra_location_probes.py")
        probes = module_from_spec(spec)
        spec.loader.exec_module(probes)
    return probes


def add_bridge_candidates(segments, skeleton, distance, nearest_gt_component):
    unsupported = skeleton & (distance > 5)
    _, segment_labels, _, _ = cv2.connectedComponentsWithStats(
        unsupported.astype(np.uint8), connectivity=8
    )
    for segment in segments:
        y = segment["bbox_y"]
        x = segment["bbox_x"]
        region = np.s_[max(0, y - 1):y + segment["bbox_height"] + 1,
                       max(0, x - 1):x + segment["bbox_width"] + 1]
        component = segment_labels[region] == segment["segment_id"]
        edge = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        supported_touch = edge & skeleton[region] & ~unsupported[region]
        labels = np.unique(nearest_gt_component[region][supported_touch])
        labels = labels[labels > 0]
        segment["gt_components_touched"] = ",".join(map(str, labels))
        segment["cross_gt_component_candidate"] = int(len(labels) >= 2)


def location_data(case, label_path, model_paths, protocol, topo, output):
    import compare_predictions as cp
    from road_metrics import skeletonize

    gt = cp.read_binary(Path(label_path), protocol["image_size"], allow_replicated_rgb=True)
    gt_skeleton = skeletonize(gt)
    paths = local_probes.gt_paths(gt, spacing=48, length=64)
    if gt.any():
        distance, nearest = distance_transform_edt(~gt, return_indices=True)
        _, gt_components = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        nearest_gt_component = gt_components[tuple(nearest)]
    else:
        distance = np.full(gt.shape, np.inf)
        nearest_gt_component = np.zeros(gt.shape, np.int32)
    endpoints = np.stack([path[[0, -1]] for path, _ in paths]).reshape(-1, 2) if paths else np.empty((0, 2))
    directory = output / "locations" / case
    directory.mkdir(parents=True, exist_ok=True)
    path_rows, segment_rows, seed_rows, counts = [], [], [], []
    overlays = {}
    for role, source in model_paths:
        mask = cp.read_binary(Path(source), protocol["image_size"])
        skeleton = skeletonize(mask)
        segments, unsupported = local_probes.unsupported_segments(
            skeleton, distance, tolerance=5, minimum=10
        )
        if segments:
            add_bridge_candidates(segments, skeleton, distance, nearest_gt_component)
        segment_rows.extend(dict(case_id=case, model=role, **item) for item in segments)
        rows = []
        overlay = np.full((*gt.shape, 3), 33, np.uint8)
        overlay[cv2.dilate(gt_skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (190, 190, 190)
        overlay[skeleton] = (100, 230, 90)
        overlay[unsupported] = (255, 70, 220)
        points = np.argwhere(skeleton)
        snapped = local_probes.snap_nodes(endpoints, points, 5).reshape(-1, 2)
        _, components = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
        for i, (path, length) in enumerate(paths):
            a, b = snapped[i]
            if a < 0 or b < 0:
                reachable = "endpoint_unmatched"
            elif a == b:
                reachable = "endpoints_merged"
            else:
                reachable = ("reachable" if components[tuple(points[a])] == components[tuple(points[b])]
                             else "disconnected")
            row = dict(case_id=case, model=role, path_id=i, start_y=int(path[0, 0]),
                       start_x=int(path[0, 1]), end_y=int(path[-1, 0]), end_x=int(path[-1, 1]),
                       gt_path_px=length, whole_image_reachability=reachable,
                       **local_probes.check_path(path, length, skeleton, 5, 3))
            rows.append(row)
        path_rows.extend(rows)
        seeds = topo[role]
        seed_rows.extend(dict(case_id=case, model=role, **row) for row in seeds)
        for row in seeds:
            if row["pred_x"] is not None:
                cv2.line(overlay, (round(row["gt_x"]), round(row["gt_y"])),
                         (round(row["pred_x"]), round(row["pred_y"])), (255, 210, 0), 1)
            color = {"accepted": (255, 210, 0), "no_nearby_seed_match": (0, 70, 255),
                     "excluded_by_one_to_one": (0, 175, 255), "empty_local_walk": (160, 120, 255)}
            point_on_image(overlay, row["gt_x"], row["gt_y"], color[row["status"]])
        if not cv2.imwrite(str(directory / f"{role}_topo_overlay.png"), overlay):
            raise OSError(f"Could not write overlay for {case} / {role}")
        overlays[role] = overlay
        counts.append(dict(case_id=case, model=role, routes=len(rows),
                           **{f"paths_{key}": value for key, value in Counter(r["status"] for r in rows).items()},
                           **{f"whole_{key}": value for key, value in Counter(r["whole_image_reachability"] for r in rows).items()},
                           **{f"seeds_{key}": value for key, value in Counter(r["status"] for r in seeds).items()},
                           cross_gt_component_candidates=sum(s["cross_gt_component_candidate"] for s in segments),
                           **Counter(r["kind"] for r in segments)))
    paired = []
    full = {row["path_id"]: row for row in path_rows if row["model"] == "full"}
    for row in path_rows:
        if row["model"] == "full":
            continue
        control, normal = row["status"], full[row["path_id"]]["status"]
        if control != normal:
            paired.append(dict(case_id=case, control_model=row["model"], path_id=row["path_id"],
                               start_y=row["start_y"], start_x=row["start_x"],
                               end_y=row["end_y"], end_x=row["end_x"],
                               full_status=normal, control_status=control))
    previews = []
    for role in ("fixed_h", "no_scgra"):
        for normal, control in (("matched", "gap_disconnected"), ("gap_disconnected", "matched")):
            row = next((r for r in paired if r["control_model"] == role
                        and r["full_status"] == normal and r["control_status"] == control), None)
            if row is not None:
                route = paths[row["path_id"]][0]
                example = write_preview(directory, f"{role}_full_{normal}", gt_skeleton,
                                        overlays, route[len(route) // 2], route)
                previews.append(dict(**row, **example))
    for role in overlays:
        unsupported = [s for s in segment_rows if s["model"] == role and s["kind"] == "attached_unsupported"]
        if unsupported:
            example = max(unsupported, key=lambda s: s["skeleton_pixels"])
            preview = write_preview(directory, f"{role}_unsupported_branch", gt_skeleton, overlays,
                                    (example["center_y"], example["center_x"]))
            previews.append(dict(**example, **preview))
    result = dict(paths=path_rows, unsupported=segment_rows, topo_seeds=seed_rows,
                  counts=counts, paired_paths=paired, previews=previews)
    cp.write_json(directory / "evidence.json", result)


def evaluate_case(output, job):
    import official_graph_metrics as graph

    case, label, models, protocol, partial_path, _ = job
    trace = {role: [] for role, _ in models}
    roles = iter(trace)
    original_score = graph._sat_topo_score

    def score(gt, pred, protocol):
        return captured_topo(original_score, graph.sat_topo, gt, pred, protocol, trace[next(roles)])

    if Path(partial_path).is_file() and (output / "locations" / case / "evidence.json").is_file():
        return evaluate_original(job)
    temporary = Path(partial_path).with_suffix(".in_progress")
    temporary.unlink(missing_ok=True)
    with patch.object(graph, "_sat_topo_score", score):
        result = evaluate_original((*job[:4], str(temporary), job[5]))
    try:
        location_data(case, label, models, protocol, trace, output)
        temporary.replace(partial_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "output", "tool", "diagnostics", "full", "fixed-h", "no-scgra"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--search", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(args.tool.resolve()), str(args.diagnostics.resolve())]
    import compare_predictions as cp
    global evaluate_original, local_probes
    evaluate_original = cp.evaluate_case
    local_probes = load_location_probes(args.diagnostics.resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    multiprocessing.set_start_method("fork")
    cv2.setNumThreads(1)
    data = args.data.resolve()
    labels = cp.label_index(data, "test")
    if len(labels) != 49:
        parser.error(f"Expected 49 Massachusetts test images, got {len(labels)}")
    roles = {"full": args.full, "fixed_h": args.fixed_h, "no_scgra": args.no_scgra}
    checkpoints = {file.resolve(): role for role, file in roles.items()}
    if len(checkpoints) != len(roles):
        parser.error("Supply three distinct archived model checkpoints")
    protocol_overrides = dict(image_size=1500, apls_max_control_nodes=150,
                              topo_seed_density_deg=0.002, topo_step_deg=0.0002)
    manifest = dict(data_root=str(data), protocol=protocol_overrides,
                    models=[dict(name=role) for role in roles])
    protocol = cp.validate_manifest(manifest)
    roots = [*args.search, *(file.parent for file in checkpoints)]
    sources = args.manifest or sorted(set(candidates(roots)))
    manifest["models"] = archived_masks(cp, sources, data, labels, checkpoints, protocol)
    selection = dict(schema=2, selected_on="val", criterion="reused_saved_validation_selection",
                     threshold_protocol_hash=cp.threshold_protocol_hash(protocol),
                     models={model["name"]: {
                         "identity": cp.model_identity(model),
                         "threshold": model["predictions"]["test"][0]["threshold"],
                     } for model in manifest["models"]})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    for name, record in (("comparison.json", manifest), ("val_selection.json", selection)):
        path = output / name
        if path.exists() and load(path) != record:
            parser.error(f"Existing {path} uses different masks or selections")
        cp.write_json(path, record)
    topology = output / "topology"
    topology.mkdir(exist_ok=args.resume)
    with patch.object(cp, "evaluate_case", partial(evaluate_case, output)):
        cp.evaluate(manifest, output, protocol, "test", selection, topology,
                    1, workers=args.workers, resume=args.resume)
    results = [load(output / "locations" / case / "evidence.json") for case in sorted(labels)]
    exports = {name: [row for result in results for row in result[name]] for name in
               ("paths", "unsupported", "topo_seeds", "counts", "paired_paths", "previews")}
    totals = {role: Counter() for role in roles}
    for row in exports["counts"]:
        totals[row["model"]].update({k: v for k, v in row.items() if k not in ("case_id", "model")})
    columns = sorted(set().union(*(counter.keys() for counter in totals.values())))
    exports["summary_counts"] = [dict(model=role, **{column: counter[column] for column in columns})
                                 for role, counter in totals.items()]
    for name, rows in exports.items():
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
    print(f"[ALL DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
