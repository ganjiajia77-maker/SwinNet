import argparse
import csv
import os
from types import SimpleNamespace

import cv2
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint, resize_like
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state
from config import get_config


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare segmentation and connectivity/topology metrics for two checkpoints."
    )
    p.add_argument("--root_path", required=True)
    p.add_argument("--baseline_model_path", required=True)
    p.add_argument("--current_model_path", required=True)
    p.add_argument("--baseline_name", default="baseline")
    p.add_argument("--current_name", default="current")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--baseline_threshold", type=float, default=None)
    p.add_argument("--current_threshold", type=float, default=None)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--source_patch_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--short_area_threshold", type=int, default=20)
    p.add_argument("--apls_max_nodes", type=int, default=64)
    p.add_argument("--apls_snap_radius", type=float, default=5.0)
    p.add_argument("--output_csv", default="./analysis_out/connectivity_topology_compare.csv")
    p.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    p.add_argument("--dataset", default="ImageData")
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--n_class", type=int, default=1)
    p.add_argument("--structure_profile", default="stage23_boundary_0626")
    p.add_argument("--bottleneck_type", default="global_local")
    p.add_argument("--disable_msfe_skip", action="store_true")
    p.add_argument("--enable_highres_structure_stream", action="store_true")
    p.add_argument("--highres_structure_channels", type=int, default=64)
    p.add_argument("--highres_structure_fuse_stages", default="stage23")
    p.add_argument("--highres_structure_fusion_mode", default="stage23")
    p.add_argument("--enable_global_topology", action="store_true")
    p.add_argument("--global_topology_max_nodes", type=int, default=32)
    p.add_argument("--global_topology_heads", type=int, default=4)
    p.add_argument("--global_topology_alpha_max", type=float, default=0.05)
    p.add_argument("--model_impl", default="auto", choices=("auto", "standard", "selective"))
    p.add_argument("--final_topology_eta_init", type=float, default=0.005)
    p.add_argument("--final_gap_rho_init", type=float, default=0.005)
    p.add_argument("--stage_topology_stages", default="none")
    p.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    p.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    p.add_argument("--stage_topology_bias_mode", default="pairwise_skeleton")
    p.add_argument("--stage_topology_ratio", type=float, default=0.08)
    p.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    p.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    p.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    p.add_argument("--stage3_gate_topology_gradient_ratio", type=float, default=0.0)
    p.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    p.add_argument("--zip", action="store_true")
    p.add_argument("--cache_mode", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--accumulation_steps", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--amp_opt_level", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--throughput", action="store_true")
    return p.parse_args()


def zhang_suen_skeletonize(mask):
    image = mask.astype(np.uint8).copy()
    h, w = image.shape
    while True:
        changed = False
        for phase in (0, 1):
            remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if image[y, x] == 0:
                        continue
                    p2, p3, p4 = image[y - 1, x], image[y - 1, x + 1], image[y, x + 1]
                    p5, p6, p7 = image[y + 1, x + 1], image[y + 1, x], image[y + 1, x - 1]
                    p8, p9 = image[y, x - 1], image[y - 1, x - 1]
                    n = [p2, p3, p4, p5, p6, p7, p8, p9]
                    count = sum(n)
                    transitions = sum(a == 0 and b == 1 for a, b in zip(n, n[1:] + n[:1]))
                    if not (2 <= count <= 6 and transitions == 1):
                        continue
                    if phase == 0:
                        keep_clear = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                    else:
                        keep_clear = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                    if keep_clear:
                        remove.append((y, x))
            if remove:
                changed = True
                for y, x in remove:
                    image[y, x] = 0
        if not changed:
            return image.astype(bool)


def skeletonize(mask):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(
            mask.astype(np.uint8) * 255,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        ) > 127
    return zhang_suen_skeletonize(mask)


def component_stats(mask, short_area_threshold):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int64)
    total = float(areas.sum())
    largest = float(areas.max()) if areas.size else 0.0
    return {
        "components": float(max(count - 1, 0)),
        "short_components": float((areas < short_area_threshold).sum()) if areas.size else 0.0,
        "largest_ratio": largest / (total + 1e-8),
    }


def component_count(mask):
    count, _, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return max(count - 1, 0)


def topology_scores(pred, gt):
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)
    pred_n = float(pred_skel.sum())
    gt_n = float(gt_skel.sum())
    topo_precision = float((pred_skel & gt).sum()) / (pred_n + 1e-8)
    topo_recall = float((gt_skel & pred).sum()) / (gt_n + 1e-8)
    topo_f1 = 2.0 * topo_precision * topo_recall / (topo_precision + topo_recall + 1e-8)
    missing = gt_skel & ~pred
    gap_stats = component_stats(missing, 2)
    return {
        "topo_precision": topo_precision,
        "topo_recall": topo_recall,
        "topo_f1": topo_f1,
        "cldice": topo_f1,
        "pred_skeleton_pixels": pred_n,
        "gt_skeleton_pixels": gt_n,
        "break_pixels": float(missing.sum()),
        "break_rate": float(missing.sum()) / (gt_n + 1e-8),
        "gap_components": gap_stats["components"],
        "max_gap_pixels": 0.0,
        "pred_skel": pred_skel,
        "gt_skel": gt_skel,
    }


def max_component_area(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max())


def skeleton_graph(skel):
    graph = nx.Graph()
    points = [tuple(p) for p in np.argwhere(skel)]
    point_set = set(points)
    for point in points:
        graph.add_node(point)
        y, x = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                other = (y + dy, x + dx)
                if other in point_set:
                    graph.add_edge(point, other, weight=float(np.hypot(dy, dx)))
    return graph


def sample_graph_nodes(graph, max_nodes):
    nodes = list(graph.nodes)
    if len(nodes) <= max_nodes:
        return nodes
    special = [node for node in nodes if graph.degree[node] != 2]
    if len(special) >= max_nodes:
        indices = np.linspace(0, len(special) - 1, max_nodes).astype(int)
        return [special[i] for i in indices]
    remaining = max_nodes - len(special)
    regular = [node for node in nodes if node not in set(special)]
    indices = np.linspace(0, len(regular) - 1, remaining).astype(int)
    return special + [regular[i] for i in indices]


def apls_score(gt_skel, pred_skel, max_nodes, snap_radius):
    """Approximate APLS using sampled skeleton graph nodes and pixel geodesics."""
    gt_graph = skeleton_graph(gt_skel)
    pred_graph = skeleton_graph(pred_skel)
    if gt_graph.number_of_nodes() < 2 or pred_graph.number_of_nodes() < 2:
        return 0.0
    gt_nodes = sample_graph_nodes(gt_graph, max_nodes)
    pred_points = np.asarray(list(pred_graph.nodes), dtype=np.float32)
    snapped = []
    for node in gt_nodes:
        delta = pred_points - np.asarray(node, dtype=np.float32)
        distance = np.sqrt((delta * delta).sum(axis=1))
        index = int(distance.argmin())
        snapped.append(tuple(pred_points[index].astype(int))) if distance[index] <= snap_radius else snapped.append(None)

    gt_paths = {node: nx.single_source_dijkstra_path_length(gt_graph, node, weight="weight") for node in gt_nodes}
    pred_paths = {
        node: nx.single_source_dijkstra_path_length(pred_graph, node, weight="weight")
        for node in dict.fromkeys(node for node in snapped if node is not None)
    }
    scores = []
    for i, source in enumerate(gt_nodes):
        for j in range(i + 1, len(gt_nodes)):
            target = gt_nodes[j]
            gt_distance = gt_paths[source].get(target)
            if gt_distance is None or gt_distance <= 0:
                continue
            pred_source, pred_target = snapped[i], snapped[j]
            pred_distance = None
            if pred_source is not None and pred_target is not None:
                pred_distance = pred_paths.get(pred_source, {}).get(pred_target)
            if pred_distance is None:
                scores.append(0.0)
            else:
                scores.append(max(0.0, 1.0 - abs(pred_distance - gt_distance) / gt_distance))
    return float(np.mean(scores)) if scores else 0.0


def empty_accumulator():
    return {"images": 0, "tp": 0, "fp": 0, "fn": 0, "values": []}


def load_metric_model(model_path, args, device):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_args = SimpleNamespace(**vars(args))
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        for name in (
            "structure_profile",
            "bottleneck_type",
            "final_topology_eta_init",
            "final_gap_rho_init",
            "stage2_skeleton_gradient_ratio",
            "stage3_skeleton_gradient_ratio",
            "stage3_gate_topology_gradient_ratio",
            "final_skeleton_gradient_ratio",
            "enable_highres_structure_stream",
            "highres_structure_channels",
            "highres_structure_fuse_stages",
            "highres_structure_fusion_mode",
            "enable_post_refine_structure_interaction",
            "enable_global_topology",
            "global_topology_max_nodes",
            "global_topology_heads",
            "global_topology_alpha_max",
        ):
            if name in saved_args:
                setattr(model_args, name, saved_args[name])
    if checkpoint.get("structure_profile"):
        model_args.structure_profile = checkpoint["structure_profile"]

    config = get_config(model_args)
    model = SwinUnet(
        config=config,
        img_size=model_args.img_size,
        num_classes=model_args.num_classes,
        return_skeleton=True,
        bottleneck_type=model_args.bottleneck_type,
        final_topology_eta_init=model_args.final_topology_eta_init,
        final_gap_rho_init=model_args.final_gap_rho_init,
        structure_profile=model_args.structure_profile,
        stage2_skeleton_gradient_ratio=model_args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=model_args.stage3_skeleton_gradient_ratio,
        stage3_gate_topology_gradient_ratio=model_args.stage3_gate_topology_gradient_ratio,
        final_skeleton_gradient_ratio=model_args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=model_args.enable_highres_structure_stream,
        highres_structure_channels=model_args.highres_structure_channels,
        highres_structure_fuse_stages=model_args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=model_args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=model_args.enable_post_refine_structure_interaction,
        enable_global_topology=model_args.enable_global_topology,
        global_topology_max_nodes=model_args.global_topology_max_nodes,
        global_topology_heads=model_args.global_topology_heads,
        global_topology_alpha_max=model_args.global_topology_alpha_max,
    ).to(device)
    adapt_connectivity_modules_for_checkpoint(model, state_dict, "standard")
    load_topology_checkpoint_state(
        model,
        state_dict,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=(model_args.bottleneck_type == "global_local"),
    )
    return model


def evaluate(name, model_path, threshold, args, loader, device):
    model = load_metric_model(model_path, args, device)
    model.eval()
    acc = empty_accumulator()
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=name)):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device)
            masks = resize_like(batch["mask"].to(device).float(), images, mode="nearest")
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            masks = resize_like(batch["mask"].to(device).float(), logits, mode="nearest")
            pred_batch = torch.sigmoid(logits) >= threshold
            gt_batch = masks > 0.5
            acc["tp"] += int((pred_batch & gt_batch).sum())
            acc["fp"] += int((pred_batch & ~gt_batch).sum())
            acc["fn"] += int((~pred_batch & gt_batch).sum())
            for pred_t, gt_t in zip(pred_batch[:, 0].cpu().numpy(), gt_batch[:, 0].cpu().numpy()):
                topo = topology_scores(pred_t, gt_t)
                gt_skel = topo.pop("gt_skel")
                missing = gt_skel & ~pred_t
                if missing.any():
                    topo["max_gap_pixels"] = max_component_area(missing)
                pred_components = component_stats(pred_t, args.short_area_threshold)
                gt_components = component_stats(gt_t, args.short_area_threshold)
                fp_components = component_count(pred_t & ~gt_t)
                foreground = float(pred_t.sum())
                topo.update(
                    {
                        "pred_components": pred_components["components"],
                        "short_pred_components": pred_components["short_components"],
                        "fragment_density_per_1000_px": 1000.0 * pred_components["components"] / (foreground + 1e-8),
                        "largest_component_ratio": pred_components["largest_ratio"],
                        "gt_components": gt_components["components"],
                        "extra_components": max(pred_components["components"] - gt_components["components"], 0.0),
                        "false_positive_components": float(fp_components),
                        "apls": apls_score(gt_skel, topo["pred_skel"], args.apls_max_nodes, args.apls_snap_radius),
                    }
                )
                topo.pop("pred_skel", None)
                acc["values"].append(topo)
                acc["images"] += 1
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    eps = 1e-8
    precision = acc["tp"] / (acc["tp"] + acc["fp"] + eps)
    recall = acc["tp"] / (acc["tp"] + acc["fn"] + eps)
    summary = {"name": name, "checkpoint": model_path, "threshold": threshold, "images": acc["images"]}
    summary.update(
        {
            "iou": acc["tp"] / (acc["tp"] + acc["fp"] + acc["fn"] + eps),
            "f1": 2.0 * precision * recall / (precision + recall + eps),
            "precision": precision,
            "recall": recall,
        }
    )
    keys = [
        "cldice", "topo_precision", "topo_recall", "topo_f1",
        "pred_components", "fragment_density_per_1000_px", "largest_component_ratio",
        "short_pred_components", "extra_components", "false_positive_components",
        "break_pixels", "break_rate", "gap_components", "max_gap_pixels", "apls",
    ]
    for key in keys:
        summary[key] = float(np.mean([row[key] for row in acc["values"]])) if acc["values"] else float("nan")
    return summary


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    baseline_threshold = args.baseline_threshold if args.baseline_threshold is not None else args.threshold
    current_threshold = args.current_threshold if args.current_threshold is not None else args.threshold
    rows = [
        evaluate(args.baseline_name, args.baseline_model_path, baseline_threshold, args, loader, device),
        evaluate(args.current_name, args.current_model_path, current_threshold, args, loader, device),
    ]
    fields = list(rows[0].keys())
    print("\nConnectivity/topology comparison")
    for row in rows:
        print("\n" + row["name"])
        for key in fields:
            if key not in {"name", "checkpoint"}:
                print(f"{key:38s} {row[key]}")
    print("\nDelta current - baseline")
    for key in fields:
        if key not in {"name", "checkpoint", "threshold", "images"}:
            print(f"{key:38s} {rows[1][key] - rows[0][key]:+.6f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {args.output_csv}")
    print("APLS is computed on sampled skeleton graph nodes; see --apls_max_nodes and --apls_snap_radius.")


if __name__ == "__main__":
    main()
