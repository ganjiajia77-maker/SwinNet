"""Paired road, error and path evidence from a trained decoder checkpoint."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from h0_location import file_digest, read_json, source_digest, write_json
from mass_protocol import SOURCE_SIZE, STRIDE, TILE_SIZE, sliding_positions, stitch_logits, validate_training_args
from networks.h0_location_contract import ARCHITECTURE_ID, make_contract
from networks.vision_transformer import SwinUnet
from scripts.decoder_evidence_geometry import (
    COHORTS, error_rows, official_topo_rows, path_rows, road_geometry, road_rows,
)

ORIGINAL_ID = "h0_stage2_location_v1"
TOPO = dict(image_size=1500, topo_seed_density_deg=.002, topo_step_deg=.0002)
INT64_FIX_PREVIOUS_SCRIPTS = {
    "diagnose_decoder_evidence.py": "66178141cc4e5787548121d15d0faad37b488a0bd3fae525b6df63d4136af629",
    "decoder_evidence_geometry.py": "1f54c23f9307a428b6384b1e6442a1384705ef4ffd31b9fa53bc1bee17ca0c6c",
}


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["architecture_contract"]
    architecture = saved["architecture_id"]
    if architecture not in (ORIGINAL_ID, ARCHITECTURE_ID) or saved["stage2_structure_location"] != "decoder":
        raise ValueError(f"Expected a decoder control checkpoint, received {architecture}")
    validate_training_args(checkpoint["args"])
    cfg = get_config(SimpleNamespace(cfg=str(ROOT / "configs/swin_tiny_patch4_window7_224_lite.yaml"),
        opts=None, batch_size=2, img_size=512, zip=False, cache_mode=None, resume=None,
        accumulation_steps=1, use_checkpoint=False, amp_opt_level=None, tag=None, eval=True, throughput=False))
    options = saved["constructor"]
    expected = make_contract(cfg, options)
    if architecture == ORIGINAL_ID:
        expected["architecture_id"] = ORIGINAL_ID
        expected.pop("global_topology_location")
    # The original zip and Git checkout can have different line endings.
    source = (ROOT / "mass_protocol.py").read_bytes().replace(b"\r\n", b"\n")
    source_digests = {hashlib.sha256(value).hexdigest() for value in (source, source.replace(b"\n", b"\r\n"))}
    key = "protocol_implementation_sha256"
    if saved[key] in source_digests:
        expected[key] = saved[key]
    mismatches = [name for name in sorted(set(saved) | set(expected)) if saved.get(name) != expected.get(name)]
    if mismatches or options["architecture_id"] != architecture:
        raise ValueError(f"Checkpoint contract mismatch: {mismatches or ['constructor.architecture_id']}")
    model = SwinUnet(cfg, **dict(options, architecture_id=None if architecture == ORIGINAL_ID else architecture))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), architecture, checkpoint.get("epoch")


@torch.inference_mode()
def replay_pair(model, image):
    core = model.swin_unet
    module = core.global_topology
    starts = sliding_positions(SOURCE_SIZE, TILE_SIZE, STRIDE)
    windows = iter(product(starts, starts))
    records = []
    enabled = core.enable_global_topology
    capture = getattr(module, "capture_anchor_selection", False)
    last = getattr(module, "last_anchor_selection", None)

    def observed(tile):
        top, left = next(windows)
        module.last_anchor_selection = None
        output = model(tile)
        trace = module.last_anchor_selection
        if trace is None:
            raise RuntimeError("Global topology did not select anchors in this window")
        candidates = trace["candidate_coords"][0][trace["candidate_valid"][0]].numpy().copy()
        selected = trace["selected_coords"][0][trace["selected_valid"][0]].numpy().copy()
        records.append((top, left, *trace["anchor_hw"], candidates, selected))
        return output

    try:
        module.capture_anchor_selection = True
        on, _ = stitch_logits(observed, image)
        if len(records) != len(starts)**2:
            raise RuntimeError("Anchor trace does not cover all windows")
        module.capture_anchor_selection = False
        core.enable_global_topology = False
        off, _ = stitch_logits(model, image)
    finally:
        core.enable_global_topology = enabled
        module.capture_anchor_selection = capture
        module.last_anchor_selection = last
    return on[0, 0], off[0, 0], records


def load_annotation_boxes(path):
    if path is None:
        return {}
    boxes = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            boxes.setdefault(row["case_id"], []).append(tuple(int(row[key]) for key in ("x0", "y0", "x1", "y1")))
    for case, regions in boxes.items():
        if any(not (0 <= x0 < x1 <= SOURCE_SIZE and 0 <= y0 < y1 <= SOURCE_SIZE) for x0, y0, x1, y1 in regions):
            raise ValueError(f"Invalid native-pixel annotation box in {case}")
    return boxes


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record_protocol(metadata, protocol):
    if metadata.exists():
        saved_protocol = read_json(metadata)
        if saved_protocol != protocol:
            # The exact 4ba55e1 reports differ only in JSON serialization of
            # bounding boxes. Any complete case from that version is reusable.
            previous = dict(protocol, scripts_sha256=INT64_FIX_PREVIOUS_SCRIPTS)
            if saved_protocol != previous:
                raise ValueError("This output belongs to a different run; use a different --output_dir")
            print("[RESUME] Reusing compatible cases from 4ba55e1", flush=True)
    write_json(metadata, protocol)


def comparison_index(directory):
    if directory is None:
        return {}
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in sorted(directory.glob("*.png")):
        case = path.stem.removesuffix("_pred").removesuffix("_surface").removesuffix("_sat")
        if case in result:
            raise ValueError(f"Duplicate comparison mask case {case}")
        result[case] = path
    if not result:
        raise ValueError(f"No comparison masks found in {directory}")
    return result


def save_png(path, data):
    if not cv2.imwrite(str(path), data):
        raise OSError(f"Failed writing {path}")


def save_visual(case_dir, sample, gt, masks, anchors, paths, events, changed_routes):
    rgb = sample["image"].numpy().transpose(1, 2, 0)
    rgb = np.clip((rgb * [.229, .224, .225] + [.485, .456, .406]) * 255, 0, 255).astype(np.uint8)
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panels = []
    for name, mask in (("GT", gt), *masks.items()):
        canvas = base.copy()
        canvas[mask] = (.45 * canvas[mask] + .55 * np.array([0, 215, 255])).astype(np.uint8)
        cv2.putText(canvas, name, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
        panels.append(canvas)
    changed = [row for row in paths if row["on_status"] != row["off_status"]]
    for row in changed[:12]:
        a, b = (row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])
        xy = np.ascontiguousarray(changed_routes[row["path_id"]][:, ::-1], dtype=np.int32)
        for panel in panels:
            cv2.polylines(panel, [xy], False, (255, 140, 60), 1)
            for pt in (a, b):
                cv2.circle(panel, pt, 6, (255, 255, 0), -1)
    for row in events[:30]:
        if row["on_category"] != row["off_category"]:
            for panel in panels:
                cv2.rectangle(panel, (row["left"], row["top"]), (row["right"], row["bottom"]), (0, 80, 255), 2)
    save_png(case_dir / "overview.png", np.hstack(panels))
    if changed:
        # Keep each anchor panel tied to an actual window. Never pool anchors
        # from different forwards into a single attention context.
        for index, row in enumerate(changed[:4], 1):
            for top, left, h, w, cand, fps in anchors:
                ys = (row["start_y"]-top, row["end_y"]-top)
                xs = (row["start_x"]-left, row["end_x"]-left)
                if all(0 <= value < TILE_SIZE for value in (*ys, *xs)):
                    tile = panels[1][top:top+TILE_SIZE, left:left+TILE_SIZE].copy()
                    for nodes, color, size in ((cand, (0, 255, 255), 1), (fps, (255, 255, 0), 4)):
                        for y, x in np.rint((nodes+.5) * (TILE_SIZE/np.array([h, w])) - .5).astype(int):
                            cv2.circle(tile, (x, y), size, color, -1)
                    save_png(case_dir / f"{row['path_id']}_window_{index}_{top}_{left}.png", tile)
                    break


def paired_summary(cases):
    summaries = {"images": len(cases), "roads": {}, "errors": {}, "paths": {}, "topo_seed_cohorts": {}}
    for cohort in COHORTS.values():
        roads = [row for case in cases for row in case["roads"] if row["cohort"] == cohort]
        comparable = [row for row in roads if row["on_local_connected"] is not None]
        denom = sum(row["gt_skeleton_px"] for row in roads)
        summaries["roads"][cohort] = dict(segments=len(roads), gt_skeleton_px=denom,
            restored_px=sum(row["restored_px"] for row in roads), lost_px=sum(row["lost_px"] for row in roads),
            connected_evaluable=len(comparable),
            on_centerline_coverage=sum(row["on_covered_px"] for row in roads)/denom if denom else None,
            off_centerline_coverage=sum(row["off_covered_px"] for row in roads)/denom if denom else None,
            on_connected=sum(row["on_local_connected"] for row in comparable),
            off_connected=sum(row["off_local_connected"] for row in comparable))
        if roads and "external_covered_px" in roads[0]:
            summaries["roads"][cohort].update(
                external_centerline_coverage=sum(row["external_covered_px"] for row in roads)/denom,
                external_connected=sum(row["external_local_connected"] for row in comparable))
        seeds = [row for case in cases for row in case["topo_seeds"] if row["cohort"] == cohort]
        aligned = {mode: {(row["case_id"], row["seed_id"], row["y"], row["x"]): row for row in seeds
                          if row["mode"] == mode} for mode in ("on", "off")}
        common = aligned["on"].keys() & aligned["off"].keys()
        summaries["topo_seed_cohorts"][cohort] = dict(matched_gt_seed_locations=len(common),
            on_recall_contribution=float(np.mean([aligned["on"][key]["recall_contribution"] for key in sorted(common)])) if common else None,
            off_recall_contribution=float(np.mean([aligned["off"][key]["recall_contribution"] for key in sorted(common)])) if common else None)
    errors = [row for case in cases for row in case["errors"]]
    summaries["errors"] = dict(spatial_events=len(errors),
        images_with_on_event=len({row["case_id"] for row in errors if row["on_category"] != "absent"}),
        images_with_off_event=len({row["case_id"] for row in errors if row["off_category"] != "absent"}),
        on_only=sum(row["on_category"] != "absent" and row["off_category"] == "absent" for row in errors),
        off_only=sum(row["off_category"] != "absent" and row["on_category"] == "absent" for row in errors),
        on_categories=dict(Counter(row["on_category"] for row in errors)),
        off_categories=dict(Counter(row["off_category"] for row in errors)))
    if errors and "external_category" in errors[0]:
        summaries["errors"]["external_categories"] = dict(Counter(row["external_category"] for row in errors))
    paths = [row for case in cases for row in case["paths"]]
    for key, subset in [
        ("all", paths),
        *((name, [row for row in paths if row["length_bin"] == name]) for name in ("32-127", "128-255", "256+")),
        ("no_joint_window", [row for row in paths if not row["joint_windows"]]),
        ("same_window_selected_pair", [row for row in paths if row["selected_pair_windows"]]),
        ("joint_window_without_selected_pair", [row for row in paths if row["joint_windows"] and not row["selected_pair_windows"]]),
    ]:
        both = [row for row in subset if row["off_status"] == row["on_status"] == "reachable"]
        summaries["paths"][key] = dict(paths=len(subset), both_reachable=len(both),
            recovered=sum(row["off_status"] != "reachable" and row["on_status"] == "reachable" for row in subset),
            lost=sum(row["off_status"] == "reachable" and row["on_status"] != "reachable" for row in subset),
            transitions=dict(Counter(f"{row['off_status']} -> {row['on_status']}" for row in subset)),
            prediction_shorter_than_off=sum(row["on_length_px"] < row["off_length_px"] for row in both),
            prediction_longer_than_off=sum(row["on_length_px"] > row["off_length_px"] for row in both),
            shorter_gt_relative_error=sum(abs(row["on_length_relative_error"]) < abs(row["off_length_relative_error"]) for row in both),
            larger_gt_relative_error=sum(abs(row["on_length_relative_error"]) > abs(row["off_length_relative_error"]) for row in both))
    per_image = [row for case in cases for row in case["per_image"]]
    for mode in sorted({row["mode"] for row in per_image}):
        items = [row for row in per_image if row["mode"] == mode]
        tp, fp, fn = (sum(row[key] for row in items) for key in ("tp", "fp", "fn"))
        summaries.setdefault("overall", {})[mode] = dict(
            iou=tp/max(tp+fp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1),
            precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
            topo_precision=float(np.mean([row["topo_precision"] for row in items])),
            topo_recall=float(np.mean([row["topo_recall"] for row in items])))
    if len(cases) > 1:
        rng = np.random.default_rng(1234)
        picks = rng.integers(0, len(cases), size=(2000, len(cases)))
        intervals = {}
        for metric in ("topo_precision", "topo_recall"):
            changes = np.array([row["on"][metric]-row["off"][metric] for row in
                ({value["mode"]: value for value in case["per_image"]} for case in cases)])
            intervals[metric] = dict(on_minus_off=float(changes.mean()),
                image_paired_bootstrap_95=[float(value) for value in np.quantile(changes[picks].mean(1), [.025, .975])])
        summaries["paired_topo"] = intervals
    return summaries


def evaluate_case(sample, model, args, protocol, road_metrics, official, read_binary, external, boxes, case_dir):
    case = sample["case_name"]
    target = sample["mask"][0].numpy() > .5
    on, off, anchors = replay_pair(model, sample["image"][None].to(args.device))
    masks = dict(on=on.sigmoid().cpu().numpy() >= args.threshold,
                 off=off.sigmoid().cpu().numpy() >= args.threshold)
    if external:
        masks["external"] = read_binary(external[case], SOURCE_SIZE)
    skel = road_metrics.skeletonize(target)
    skeletons = {name: road_metrics.skeletonize(mask) for name, mask in masks.items()}
    graphs = {name: road_metrics.skeleton_graph(value) for name, value in skeletons.items()}
    points, graph = road_metrics.skeleton_graph(skel)
    occluded = np.zeros_like(target)
    for x0, y0, x1, y1 in boxes.get(case, []):
        occluded[y0:y1, x0:x1] = True
    kind, width = road_geometry(target, skel, args.narrow_width, args.junction_radius, occluded)
    roads = road_rows(case, kind, width, masks, args.endpoint_tolerance, args.minimum_segment)
    errors = error_rows(case, target, skeletons, args.error_tolerance, args.minimum_error)
    changed_routes = {} if args.visuals else None
    paths = path_rows(case, points, graph, graphs, masks, (on-off).cpu().numpy(), anchors, protocol, changed_routes)
    topo_seeds, topo_precision_points, per_image = [], [], []
    gt_graph = dict(points=points, csr=graph)
    for mode, pred in masks.items():
        nodes, edges = graphs[mode]
        seeds, matches, row = official_topo_rows(case, gt_graph, dict(points=nodes, csr=edges), kind, TOPO, official, mode)
        topo_seeds.extend(seeds)
        topo_precision_points.extend(matches)
        per_image.append(dict(**row, **road_metrics.confusion(pred, target),
                              alpha_global=float(model.swin_unet.global_topology.alpha_global)))
        save_png(case_dir / f"{mode}.png", pred.astype(np.uint8)*255)
    flat_nodes = [(window, top, left, h, w, name, int(y), int(x))
                  for window, (top, left, h, w, cand, fps) in enumerate(anchors)
                  for name, nodes in ((0, cand), (1, fps)) for y, x in nodes]
    np.savez_compressed(case_dir / "anchors.npz", grid=np.asarray(flat_nodes, dtype=np.int32).reshape(-1, 8))
    if args.visuals:
        save_visual(case_dir, sample, target, masks, anchors, paths, errors, changed_routes)
    return dict(case_id=case, roads=roads, errors=errors, paths=paths, topo_seeds=topo_seeds,
                topo_precision_points=topo_precision_points, per_image=per_image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Use the existing val-selected threshold")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=0, help="0 evaluates the entire split")
    parser.add_argument("--topo_tool", type=Path, default=Path(
        "/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py"))
    parser.add_argument("--comparison_masks", type=Path, help="Optional aligned external binary PNGs")
    parser.add_argument("--occluded_boxes", type=Path, help="Optional blinded CSV, columns case_id,x0,y0,x1,y1")
    parser.add_argument("--visuals", action="store_true")
    for name, default in dict(narrow_width=8, junction_radius=16, minimum_segment=8,
            error_tolerance=4, minimum_error=5, endpoint_tolerance=5, anchor_tolerance=12,
            controls=48, paths_per_bin=25, seed=1234).items():
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.max_images < 0 or any(
            getattr(args, name) < 1 for name in ("narrow_width", "junction_radius", "minimum_segment", "error_tolerance",
            "minimum_error", "endpoint_tolerance", "anchor_tolerance", "controls", "paths_per_bin")) or args.seed < 0:
        parser.error("Threshold must be in (0,1), counts/tolerances positive, seed and max_images nonnegative")
    tool = args.topo_tool.resolve()
    if not tool.is_file() or tool.name != "compare_predictions.py":
        raise FileNotFoundError(tool)
    expected = read_json(Path(__file__).with_name("decoder_evidence_metric_sha256.json"))
    metric_hashes = {name: source_digest(tool.parent / name) for name in expected}
    if metric_hashes != expected:
        raise ValueError(f"Use the exact 4b7d37a metric source, differs at: "
                         f"{[name for name in expected if metric_hashes[name] != expected[name]]}")
    sys.path.insert(0, str(tool.parent))
    import road_metrics
    import official_graph_metrics as official
    from compare_predictions import read_binary
    data = RoadSkeletonDataset(str(args.root_path), args.split, image_size=None,
                               source_patch_size=SOURCE_SIZE, augment=False, return_full_image=True)
    if len(data) != {"val": 14, "test": 49}[args.split]:
        raise ValueError(f"Unexpected Massachusetts {args.split} split size: {len(data)}")
    external = comparison_index(args.comparison_masks)
    case_ids = [Path(name).stem.replace("_sat", "") for name in data.image_files]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate dataset case IDs")
    if external and set(external) != set(case_ids):
        raise ValueError("Comparison masks must cover exactly the same cases")
    boxes = load_annotation_boxes(args.occluded_boxes)
    out = args.output_dir.resolve()
    model, architecture, epoch = load_model(args.checkpoint, args.device)
    protocol = json.loads(json.dumps(vars(args), default=str))
    protocol.update(checkpoint_sha256=file_digest(args.checkpoint), checkpoint_epoch=epoch, architecture=architecture,
        placement="after_decoder_stage2" if model.swin_unet.global_topology_stage2 else "after_final_x4",
        topo=TOPO, metric_hashes=metric_hashes, scripts_sha256={path.name: file_digest(path) for path in (
            Path(__file__), Path(__file__).with_name("decoder_evidence_geometry.py"))},
        external_sha256={name: file_digest(path) for name, path in external.items()}, case_ids=case_ids,
        model_source_sha256={name: file_digest(ROOT / name) for name in (
            "mass_protocol.py", "networks/vision_transformer.py", "networks/keypoint_global_topology.py",
            "networks/swin_transformer_unet_skip_expand_decoder_sys.py")},
        occlusion_sha256=file_digest(args.occluded_boxes) if args.occluded_boxes else None,
        road_cohort_priority=["occluded_marked", "junction", "narrow", "ordinary"],
        source_size=SOURCE_SIZE, tile=TILE_SIZE, stride=STRIDE)
    out.mkdir(parents=True, exist_ok=True)
    record_protocol(out / "protocol.json", protocol)
    total = min(len(data), args.max_images) if args.max_images else len(data)
    results = []
    for index in range(total):
        sample = data[index]
        case_dir = out / "cases" / sample["case_name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        cache = case_dir / "evidence.json"
        cached = cache.is_file()
        digest = hashlib.sha256()
        for key in ("image", "mask"):
            digest.update(sample[key].contiguous().numpy().tobytes())
        sample_sha = digest.hexdigest()
        if cached:
            result = read_json(cache)
            if result["case_id"] != sample["case_name"] or result["sample_sha256"] != sample_sha:
                raise ValueError(f"Cached case mismatch: {cache}")
        else:
            result = evaluate_case(sample, model, args, protocol, road_metrics, official, read_binary, external, boxes, case_dir)
            result["sample_sha256"] = sample_sha
            write_json(cache, result)
        print(f"[{args.split}] {index+1}/{total} {sample['case_name']} {'cached' if cached else 'computed'}", flush=True)
        results.append(result)
    for group in ("roads", "errors", "paths", "topo_seeds", "topo_precision_points", "per_image"):
        write_csv(out / f"{group}.csv", [row for case in results for row in case[group]])
    write_json(out / "summary.json", paired_summary(results))
    print(out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
