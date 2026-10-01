import argparse
import csv
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-image topology metrics and synchronized inference latency."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--repo_root", default="", help="Code tree matching this checkpoint")
    run.add_argument("--root_path", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--cfg", required=True)
    run.add_argument("--split", default="test", choices=("val", "test"))
    run.add_argument("--threshold", type=float, required=True)
    run.add_argument("--img_size", type=int, default=256)
    run.add_argument("--source_patch_size", type=int, default=1024)
    run.add_argument("--warmup_iters", type=int, default=10)
    run.add_argument("--apls_max_nodes", type=int, default=64)
    run.add_argument("--apls_snap_radius", type=float, default=5.0)
    run.add_argument("--output_dir", required=True)
    run.add_argument("--name", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline_csv", required=True)
    compare.add_argument("--current_csv", required=True)
    compare.add_argument("--baseline_name", default="local_baseline")
    compare.add_argument("--current_name", default="server_p64")
    compare.add_argument("--output_csv", required=True)
    return parser.parse_args()


def resolve_code_root(args):
    if args.repo_root:
        root = Path(args.repo_root).resolve()
    else:
        root = Path(__file__).resolve().parents[1]
    if not (root / "networks" / "vision_transformer.py").is_file():
        raise FileNotFoundError(f"Not a SwinNet source tree: {root}")
    sys.path.insert(0, str(root))
    return root


def _checkpoint_args(checkpoint):
    saved = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved, SimpleNamespace):
        return vars(saved)
    return saved if isinstance(saved, dict) else {}


def load_model_for_checkpoint(repo_root, args, device):
    from config import get_config
    from networks import vision_transformer as vision

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}

    saved = _checkpoint_args(checkpoint)
    model_args = SimpleNamespace(
        cfg=str(Path(args.cfg).resolve()),
        opts=None,
        batch_size=1,
        img_size=args.img_size,
        zip=False,
        cache_mode="",
        resume="",
        accumulation_steps=0,
        use_checkpoint=False,
        amp_opt_level="",
        tag="runtime_topology_eval",
        eval=True,
        throughput=False,
    )
    config = get_config(model_args)

    profile = checkpoint.get(
        "structure_profile", saved.get("structure_profile", "stage23_boundary_0626")
    )
    constructor_values = {
        "config": config,
        "img_size": args.img_size,
        "num_classes": 1,
        "return_skeleton": True,
        "bottleneck_type": saved.get("bottleneck_type", "global_local"),
        "final_topology_eta_init": saved.get("final_topology_eta_init", 0.005),
        "final_gap_rho_init": saved.get("final_gap_rho_init", 0.005),
        "structure_profile": profile,
        "stage2_skeleton_gradient_ratio": saved.get("stage2_skeleton_gradient_ratio", 0.5),
        "stage3_skeleton_gradient_ratio": saved.get("stage3_skeleton_gradient_ratio", 0.5),
        "stage3_gate_topology_gradient_ratio": saved.get("stage3_gate_topology_gradient_ratio", 0.0),
        "final_skeleton_gradient_ratio": saved.get("final_skeleton_gradient_ratio", 0.0),
        "enable_highres_structure_stream": saved.get("enable_highres_structure_stream", False),
        "highres_structure_channels": saved.get("highres_structure_channels", 64),
        "highres_structure_fuse_stages": saved.get("highres_structure_fuse_stages", "stage23"),
        "highres_structure_fusion_mode": saved.get("highres_structure_fusion_mode", "stage23"),
        "enable_post_refine_structure_interaction": saved.get("enable_post_refine_structure_interaction", False),
        "enable_h3_surface_fusion": saved.get("enable_h3_surface_fusion", False),
        "enable_global_topology": saved.get("enable_global_topology", False),
        "global_topology_max_nodes": saved.get("global_topology_max_nodes", 32),
        "global_topology_heads": saved.get("global_topology_heads", 4),
        "global_topology_alpha_max": saved.get("global_topology_alpha_max", 0.05),
        "stage_skeleton_mode": saved.get("stage_skeleton_mode", "prior_residual"),
        "enable_e128_stage_fusion": saved.get("enable_e128_stage_fusion", False),
        "enable_coarse_road_mask": saved.get("enable_coarse_road_mask", False),
        "enable_psi_directional_descriptor": saved.get("enable_psi_directional_descriptor", False),
        "sparse_window_compute": saved.get(
            "enable_sparse_window_compute", saved.get("sparse_window_compute", False)
        ),
        "stage2_window_threshold": saved.get("stage2_window_threshold", 0.25),
        "stage3_window_threshold": saved.get("stage3_window_threshold", 0.25),
        "coarse_candidate_window_size": saved.get("coarse_candidate_window_size", 8),
        "coarse_corridor_window_radius": saved.get("coarse_corridor_window_radius", 0),
        "coarse_routing_mode": saved.get("coarse_routing_mode", "dense"),
        "bottleneck_coarse_road_mask": saved.get("bottleneck_coarse_road_mask", False),
        "bottleneck_window_threshold": saved.get("bottleneck_window_threshold", 0.25),
        "bottleneck_route_warmup_epochs": saved.get("bottleneck_route_warmup_epochs", 0),
        "bottleneck_route_warmup_mode": saved.get("bottleneck_route_warmup_mode", "dense"),
        "coarse_route_warmup_epochs": saved.get("coarse_route_warmup_epochs", 0),
        "routing_warmup_epochs": saved.get("routing_warmup_epochs", 10),
        "stage_skeleton_bias_init": saved.get("stage_skeleton_bias_init", "zero"),
        "stage_skeleton_positive_prior": saved.get("stage_skeleton_positive_prior", 0.05),
        "remove_stage2_pre_topology_source": saved.get("remove_stage2_pre_topology_source", False),
    }
    constructor = vision.SwinUnet
    signature = inspect.signature(constructor.__init__)
    constructor_values = {
        key: value for key, value in constructor_values.items()
        if key in signature.parameters
    }
    model = constructor(**constructor_values).to(device)

    loader = getattr(vision, "load_topology_checkpoint_state", None)
    if loader is not None:
        loader(
            model,
            state,
            checkpoint.get("topology_attention_version", "legacy-unrecorded"),
            strict=(saved.get("bottleneck_type", "global_local") == "global_local"),
        )
    else:
        model.load_state_dict(state, strict=True)

    route_state = None
    route_state_source = "checkpoint.routing_state"
    restore_routing = getattr(vision, "restore_routing_checkpoint_state", None)
    if restore_routing is not None:
        route_state = restore_routing(model, checkpoint)
    uses_sparse = bool(
        saved.get("enable_sparse_window_compute", saved.get("sparse_window_compute", False))
        and saved.get("coarse_routing_mode", "dense") == "p64"
    )
    route_is_active = (
        isinstance(route_state, dict)
        and route_state.get("sparse_enabled")
        and route_state.get("calibration_done")
    )
    if uses_sparse and not route_is_active:
        calibration = checkpoint.get("routing_calibration")
        if isinstance(calibration, dict) and calibration.get("sparse_enabled"):
            stage2_threshold = calibration.get("selected_stage2_threshold")
            stage3_threshold = calibration.get("selected_stage3_threshold")
            thresholds_valid = all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in (stage2_threshold, stage3_threshold)
            )
            core_model = model.module if hasattr(model, "module") else model
            swin_unet = getattr(core_model, "swin_unet", None)
            if thresholds_valid and swin_unet is not None:
                swin_unet.set_routing_thresholds(
                    float(stage2_threshold), float(stage3_threshold)
                )
                swin_unet.set_routing_calibration_done(True)
                swin_unet.sparse_window_compute = True
                swin_unet.set_route_epoch(max(
                    int(checkpoint.get("epoch", 0)),
                    int(calibration.get("epoch", 0)),
                    int(saved.get("routing_warmup_epochs", 0)),
                ))
                route_state = swin_unet.routing_state()
                route_state_source = "checkpoint.routing_calibration"
                route_is_active = bool(
                    route_state.get("sparse_enabled")
                    and route_state.get("calibration_done")
                )
    if uses_sparse and not route_is_active:
        raise RuntimeError(
            "Checkpoint has no recoverable active calibrated P64 route: neither "
            "routing_state nor routing_calibration contains valid active thresholds; "
            "timing would not include sparse window selection/writeback."
        )

    model.eval()
    metadata = {
        "name": args.name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "code_root": str(repo_root),
        "split": args.split,
        "threshold": args.threshold,
        "img_size": args.img_size,
        "source_patch_size": args.source_patch_size,
        "precision": "fp32",
        "route_state": route_state,
        "route_state_source": route_state_source,
    }
    print(f"Model: {metadata}", flush=True)
    return model, metadata


def resolve_dataset_dirs(root, split):
    split_root = Path(root) / split
    image_dir = split_root / "image"
    if not image_dir.is_dir():
        image_dir = split_root
    label_dir = next(
        (split_root / name for name in ("mask", "label") if (split_root / name).is_dir()),
        None,
    )
    if not image_dir.is_dir() or label_dir is None:
        raise FileNotFoundError(f"Expected {split_root}/image and mask or label directories.")
    return image_dir, label_dir


def find_label(image_name, label_dir):
    stem = Path(image_name).stem
    bases = [stem, stem.replace("_sat", "_mask"), stem.replace("_image", "_mask"), stem.replace("_img", "_mask")]
    candidates = [image_name]
    for base in bases:
        candidates.extend(base + ext for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"))
    for name in dict.fromkeys(candidates):
        path = label_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No matching ground-truth label for {image_name} in {label_dir}")


def center_crop_or_pad(array, side):
    height, width = array.shape[:2]
    if height > side:
        top = (height - side) // 2
        array = array[top:top + side, ...]
    elif height < side:
        before = (side - height) // 2
        after = side - height - before
        pads = ((before, after), (0, 0)) if array.ndim == 2 else ((before, after), (0, 0), (0, 0))
        array = np.pad(array, pads, mode="constant", constant_values=0)
    height, width = array.shape[:2]
    if width > side:
        left = (width - side) // 2
        array = array[:, left:left + side, ...]
    elif width < side:
        before = (side - width) // 2
        after = side - width - before
        pads = ((0, 0), (before, after)) if array.ndim == 2 else ((0, 0), (before, after), (0, 0))
        array = np.pad(array, pads, mode="constant", constant_values=0)
    return array


def load_sample(image_path, label_path, source_side, image_side):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise OSError(f"Could not read {image_path} or {label_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = center_crop_or_pad(image, source_side)
    mask = center_crop_or_pad(mask, source_side)
    mask = (mask > 127).astype(np.float32)
    image = cv2.resize(image, (image_side, image_side), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (image_side, image_side), interpolation=cv2.INTER_NEAREST)
    mask = mask > 0.5
    image = image.astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    image = (image - mean) / std
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).float().unsqueeze(0)
    return image, mask


def skeletonize(mask):
    image = mask.astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return skeleton > 0


def component_count(mask):
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(int(count) - 1, 0)


def graph_from_skeleton(skeleton):
    graph = nx.Graph()
    points = [tuple(point) for point in np.argwhere(skeleton)]
    point_set = set(points)
    for y, x in points:
        graph.add_node((y, x))
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor = (y + dy, x + dx)
            if neighbor in point_set:
                graph.add_edge((y, x), neighbor, weight=math.hypot(dy, dx))
    return graph


def sampled_nodes(graph, maximum):
    nodes = list(graph.nodes)
    if len(nodes) <= maximum:
        return nodes
    special = [node for node in nodes if graph.degree[node] != 2]
    if len(special) >= maximum:
        indices = np.linspace(0, len(special) - 1, maximum).astype(int)
        return [special[index] for index in indices]
    special_set = set(special)
    regular = [node for node in nodes if node not in special_set]
    remaining = maximum - len(special)
    indices = np.linspace(0, len(regular) - 1, remaining).astype(int)
    return special + [regular[index] for index in indices]


def approximate_apls(gt_skeleton, pred_skeleton, maximum, snap_radius):
    gt_graph = graph_from_skeleton(gt_skeleton)
    pred_graph = graph_from_skeleton(pred_skeleton)
    if gt_graph.number_of_nodes() < 2 or pred_graph.number_of_nodes() < 2:
        return 0.0
    gt_nodes = sampled_nodes(gt_graph, maximum)
    pred_points = np.asarray(list(pred_graph.nodes), dtype=np.float32)
    snapped = []
    for node in gt_nodes:
        delta = pred_points - np.asarray(node, dtype=np.float32)
        distances = np.sqrt((delta * delta).sum(axis=1))
        nearest = int(distances.argmin())
        snapped.append(tuple(pred_points[nearest].astype(int)) if distances[nearest] <= snap_radius else None)
    gt_paths = {
        node: nx.single_source_dijkstra_path_length(gt_graph, node, weight="weight")
        for node in gt_nodes
    }
    pred_paths = {
        node: nx.single_source_dijkstra_path_length(pred_graph, node, weight="weight")
        for node in dict.fromkeys(item for item in snapped if item is not None)
    }
    scores = []
    for i, source in enumerate(gt_nodes):
        for j in range(i + 1, len(gt_nodes)):
            gt_distance = gt_paths[source].get(gt_nodes[j])
            if gt_distance is None or gt_distance <= 0:
                continue
            pred_source, pred_target = snapped[i], snapped[j]
            pred_distance = None
            if pred_source is not None and pred_target is not None:
                pred_distance = pred_paths.get(pred_source, {}).get(pred_target)
            scores.append(
                0.0 if pred_distance is None else max(0.0, 1.0 - abs(pred_distance - gt_distance) / gt_distance)
            )
    return float(np.mean(scores)) if scores else 0.0


def topology_metrics(pred, gt, apls_max_nodes, apls_snap_radius):
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)
    pred_skel_n = int(pred_skel.sum())
    gt_skel_n = int(gt_skel.sum())
    topo_precision = float((pred_skel & gt).sum()) / (pred_skel_n + 1e-8)
    topo_recall = float((gt_skel & pred).sum()) / (gt_skel_n + 1e-8)
    cldice = 2.0 * topo_precision * topo_recall / (topo_precision + topo_recall + 1e-8)
    missing = gt_skel & ~pred
    pred_components = component_count(pred)
    gt_components = component_count(gt)
    return {
        "cldice": cldice,
        "break_pixels": int(missing.sum()),
        "break_rate": float(missing.sum()) / (gt_skel_n + 1e-8),
        "gap_components": component_count(missing),
        "pred_components": pred_components,
        "gt_components": gt_components,
        "frag_idx": pred_components / max(gt_components, 1),
        "apls_approx": approximate_apls(gt_skel, pred_skel, apls_max_nodes, apls_snap_radius),
    }


def run_eval(args):
    repo_root = resolve_code_root(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synchronized per-image latency measurement.")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    model, metadata = load_model_for_checkpoint(repo_root, args, device)
    image_dir, label_dir = resolve_dataset_dirs(args.root_path, args.split)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    first_image, _ = load_sample(
        image_paths[0], find_label(image_paths[0].name, label_dir),
        args.source_patch_size, args.img_size,
    )
    first_image = first_image.to(device)
    with torch.inference_mode():
        for _ in range(max(args.warmup_iters, 0)):
            model(first_image)
    torch.cuda.synchronize(device)

    records = []
    totals = {key: 0.0 for key in ("tp", "fp", "fn")}
    latencies = []
    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            label_path = find_label(image_path.name, label_dir)
            image, gt = load_sample(
                image_path, label_path, args.source_patch_size, args.img_size
            )
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            image = image.to(device, non_blocking=True)
            outputs = model(image)
            torch.cuda.synchronize(device)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if logits.shape[-2:] != gt.shape:
                logits = F.interpolate(logits.float(), size=gt.shape, mode="bilinear", align_corners=False)
            probability = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
            pred = probability >= args.threshold
            tp = int(np.logical_and(pred, gt).sum())
            fp = int(np.logical_and(pred, ~gt).sum())
            fn = int(np.logical_and(~pred, gt).sum())
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            metrics = topology_metrics(pred, gt, args.apls_max_nodes, args.apls_snap_radius)
            case_name = Path(image_path.name).stem.replace("_sat", "")
            records.append({
                "case_name": case_name,
                "threshold": args.threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                **metrics,
                "forward_wall_ms": latency_ms,
            })
            latencies.append(latency_ms)
            if index % 50 == 0 or index == len(image_paths):
                print(f"{args.name}: {index}/{len(image_paths)} images", flush=True)

    precision = totals["tp"] / (totals["tp"] + totals["fp"] + 1e-8)
    recall = totals["tp"] / (totals["tp"] + totals["fn"] + 1e-8)
    summary = {
        **metadata,
        "n_images": len(records),
        "global_iou": totals["tp"] / (totals["tp"] + totals["fp"] + totals["fn"] + 1e-8),
        "global_f1": 2.0 * precision * recall / (precision + recall + 1e-8),
        "global_precision": precision,
        "global_recall": recall,
        "mean_cldice": float(np.mean([row["cldice"] for row in records])),
        "mean_break_rate": float(np.mean([row["break_rate"] for row in records])),
        "mean_gap_components": float(np.mean([row["gap_components"] for row in records])),
        "mean_frag_idx": float(np.mean([row["frag_idx"] for row in records])),
        "mean_apls_approx": float(np.mean([row["apls_approx"] for row in records])),
        "forward_wall_ms_mean": float(np.mean(latencies)),
        "forward_wall_ms_median": float(np.median(latencies)),
        "forward_wall_ms_p90": float(np.percentile(latencies, 90)),
        "timing_scope": "H2D copy plus synchronized model forward; includes P64 selection/gather/sparse compute/scatter when enabled; excludes disk decode and CPU topology metrics.",
        "skeleton_method": "OpenCV morphological skeleton; identical script/definition for both runs.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Per-image report: {per_image_path}")
    print(f"Summary: {summary_path}")


def compare_reports(args):
    def read_report(path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = {row["case_name"]: row for row in csv.DictReader(handle)}
        if not rows:
            raise RuntimeError(f"Empty per-image CSV: {path}")
        return rows

    baseline = read_report(args.baseline_csv)
    current = read_report(args.current_csv)
    missing_from_current = sorted(set(baseline) - set(current))
    missing_from_baseline = sorted(set(current) - set(baseline))
    if missing_from_current or missing_from_baseline:
        raise RuntimeError(
            "Image sets differ: "
            f"only baseline={missing_from_current[:5]}, only current={missing_from_baseline[:5]}"
        )
    metrics = (
        "cldice", "break_rate", "gap_components", "frag_idx", "apls_approx",
        "forward_wall_ms",
    )
    paired = []
    for case in sorted(baseline):
        base_row, cur_row = baseline[case], current[case]
        base_threshold = float(base_row["threshold"])
        cur_threshold = float(cur_row["threshold"])
        if not math.isclose(base_threshold, cur_threshold, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"Threshold differs for {case}: {base_threshold} vs {cur_threshold}")
        item = {"case_name": case, "threshold": base_threshold}
        for metric in metrics:
            left, right = float(base_row[metric]), float(cur_row[metric])
            item[f"{args.baseline_name}_{metric}"] = left
            item[f"{args.current_name}_{metric}"] = right
            item[f"delta_{metric}"] = right - left
        paired.append(item)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0].keys()))
        writer.writeheader()
        writer.writerows(paired)
    print(f"Paired images: {len(paired)}; identical image set and threshold confirmed.")
    print(f"Saved paired report: {args.output_csv}")
    for metric in metrics:
        deltas = [float(row[f"delta_{metric}"]) for row in paired]
        print(f"mean delta {metric} (current - baseline): {np.mean(deltas):+.6f}")


def main():
    args = parse_args()
    if args.command == "run":
        run_eval(args)
    else:
        compare_reports(args)


if __name__ == "__main__":
    main()
