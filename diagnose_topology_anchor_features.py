import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_topology_anchor_heatmap import (
    ConfigArgs,
    anchor_distance_to_gt,
    build_model,
    coverage_from_anchors,
    extract_topology_anchors,
    find_stage3_output,
    gaussian_smooth,
    parse_int_list,
    save_visualization,
    summarize_anchor_distances,
    unnormalize_image,
    update_args_from_checkpoint,
)


def minmax_normalize(score, eps=1e-6):
    flat = score.flatten(start_dim=1)
    min_value = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_value = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (score - min_value) / (max_value - min_value).clamp_min(eps)


def connectivity_entropy_score(conn_prob, target_size):
    if conn_prob.shape[-2:] != target_size:
        conn_prob = F.interpolate(
            conn_prob,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    total = conn_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
    prob = conn_prob / total
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1, keepdim=True)
    return (entropy / math.log(max(conn_prob.shape[1], 2))).clamp(0.0, 1.0)


def direction_variation_score(direction_logits, target_size):
    if direction_logits is None:
        return None
    if direction_logits.shape[-2:] != target_size:
        direction_logits = F.interpolate(
            direction_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    direction = F.normalize(direction_logits.float(), dim=1, eps=1e-6)
    local_mean = F.avg_pool2d(direction, kernel_size=3, stride=1, padding=1)
    coherence = torch.linalg.vector_norm(local_mean, dim=1, keepdim=True).clamp(0.0, 1.0)
    confidence = torch.linalg.vector_norm(direction_logits.float(), dim=1, keepdim=True)
    confidence = minmax_normalize(confidence)
    return ((1.0 - coherence) * confidence).clamp(0.0, 1.0)


def build_feature_scores(z_struct, surface_prob, conn_prob, direction_logits, lambda_boost):
    z = z_struct.float()
    norm_score = torch.linalg.vector_norm(z, dim=1, keepdim=True)
    norm_score = minmax_normalize(norm_score)
    if surface_prob.shape[-2:] != norm_score.shape[-2:]:
        surface_prob = F.interpolate(
            surface_prob,
            size=norm_score.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    a0 = norm_score * surface_prob
    conn_entropy = connectivity_entropy_score(conn_prob, norm_score.shape[-2:])
    direction_variation = direction_variation_score(direction_logits, norm_score.shape[-2:])
    if direction_variation is None:
        direction_variation = torch.zeros_like(a0)

    return {
        "A0_norm_x_surface": a0,
        "A0_conn_entropy": a0 * (1.0 + float(lambda_boost) * conn_entropy),
        "A0_direction_variation": a0 * (1.0 + float(lambda_boost) * direction_variation),
    }, conn_entropy, direction_variation


def tensor_stats(tensor):
    values = tensor.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "max": float(values.max().item()),
    }


def get_z_struct(model):
    z_struct = getattr(model.swin_unet, "last_highres_z_struct", None)
    if z_struct is None:
        raise RuntimeError(
            "No last_highres_z_struct found. The checkpoint/model may not enable "
            "highres structure stream."
        )
    return z_struct


def anchor_pair_distance_stats(anchors):
    if len(anchors) < 2:
        return float("inf"), float("inf")
    coords = np.asarray([(x, y) for x, y, _ in anchors], dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt((diff * diff).sum(axis=-1))
    upper = distance[np.triu_indices(len(anchors), k=1)]
    return float(upper.mean()), float(upper.min())


def gt_endpoint_junction_masks(gt_skeleton):
    skel = (gt_skeleton > 0.5).astype(np.uint8)
    if skel.sum() == 0:
        return np.zeros_like(skel), np.zeros_like(skel)
    padded = np.pad(skel, 1, mode="constant")
    neighbors = (
        padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, 2:]
        + padded[2:, 2:]
        + padded[2:, 1:-1]
        + padded[2:, :-2]
        + padded[1:-1, :-2]
        + padded[:-2, :-2]
    )
    endpoint = ((skel > 0) & (neighbors == 1)).astype(np.uint8)
    junction = ((skel > 0) & (neighbors >= 3)).astype(np.uint8)
    return endpoint, junction


def anchor_distance_to_mask(anchors, mask):
    if not anchors:
        return np.array([], dtype=np.float32)
    target = (mask > 0).astype(np.uint8)
    if target.sum() == 0:
        return np.full((len(anchors),), np.inf, dtype=np.float32)
    distance = cv2.distanceTransform((1 - target).astype(np.uint8), cv2.DIST_L2, 3)
    height, width = target.shape
    values = []
    for x, y, _ in anchors:
        x = min(max(int(x), 0), width - 1)
        y = min(max(int(y), 0), height - 1)
        values.append(float(distance[y, x]))
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate topology anchors from highres structure feature scores."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--topk", default="32,64")
    parser.add_argument("--coverage_radii", default="5,10,20")
    parser.add_argument("--smooth_kernel", type=int, default=5)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--local_max_window", type=int, default=5)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--lambda_boost", type=float, default=1.0)
    parser.add_argument(
        "--max_visuals",
        type=int,
        default=8,
        help="Maximum number of images to visualize. Set 0 to save every image.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    topk_values = parse_int_list(args.topk)
    coverage_radii = parse_int_list(args.coverage_radii)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[feature-anchor] device={device}", flush=True)
    print(f"[feature-anchor] loading checkpoint: {args.checkpoint}", flush=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[feature-anchor] checkpoint loaded in {time.perf_counter() - start_time:.2f}s", flush=True)

    config_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    print("[feature-anchor] building model", flush=True)
    start_time = time.perf_counter()
    model = build_model(config_args, checkpoint, device)
    print(f"[feature-anchor] model ready in {time.perf_counter() - start_time:.2f}s", flush=True)

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[feature-anchor] dataset size={len(dataset)}", flush=True)
    print(
        "[feature-anchor] scores: A0_norm_x_surface, A0_conn_entropy, A0_direction_variation",
        flush=True,
    )

    rows = []
    anchor_rows = []
    visual_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            should_print = args.print_freq > 0 and batch_index % args.print_freq == 0
            if should_print:
                print(f"[feature-anchor] batch {batch_index + 1} forward start", flush=True)

            images = batch["image"].to(device)
            gt_skeleton = batch["skeleton"].to(device).float()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            stage_outputs = outputs[4] if len(outputs) > 4 else []
            stage3 = find_stage3_output(stage_outputs)
            z_struct = get_z_struct(model)
            surface_prob = torch.sigmoid(outputs[0])
            conn_prob = torch.sigmoid(stage3["connectivity"])
            direction_logits = stage3.get("direction")
            scores, conn_entropy, direction_variation = build_feature_scores(
                z_struct,
                surface_prob,
                conn_prob,
                direction_logits,
                lambda_boost=args.lambda_boost,
            )
            target_size = gt_skeleton.shape[-2:]
            scores = {
                mode: gaussian_smooth(
                    F.interpolate(
                        score,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    kernel_size=args.smooth_kernel,
                    sigma=args.smooth_sigma,
                )
                for mode, score in scores.items()
            }
            z_stats = tensor_stats(z_struct)
            conn_entropy_stats = tensor_stats(conn_entropy)
            direction_variation_stats = tensor_stats(direction_variation)

            image_names = batch.get("image_name", [""] * images.shape[0])
            for sample_index in range(images.shape[0]):
                image_id = os.path.splitext(str(image_names[sample_index]))[0]
                image_rgb = unnormalize_image(images[sample_index])
                gt_np = gt_skeleton[sample_index, 0].detach().cpu().numpy()
                gt_endpoint, gt_junction = gt_endpoint_junction_masks(gt_np)
                save_this_visual = args.max_visuals == 0 or visual_count < args.max_visuals

                for mode, score_map in scores.items():
                    sample_score = score_map[sample_index:sample_index + 1]
                    anchors_by_topk = extract_topology_anchors(
                        sample_score,
                        topk_values,
                        nms_radius=args.nms_radius,
                        local_max_window=args.local_max_window,
                    )
                    score_stats = tensor_stats(sample_score)

                    for topk, anchors in anchors_by_topk.items():
                        anchor_scores = np.asarray(
                            [score for _, _, score in anchors],
                            dtype=np.float32,
                        )
                        distances = anchor_distance_to_gt(anchors, gt_np)
                        mean_dist, median_dist, p90_dist = summarize_anchor_distances(distances)
                        pair_mean_dist, pair_min_dist = anchor_pair_distance_stats(anchors)
                        endpoint_distances = anchor_distance_to_mask(anchors, gt_endpoint)
                        junction_distances = anchor_distance_to_mask(anchors, gt_junction)
                        endpoint_mean_dist, endpoint_median_dist, endpoint_p90_dist = (
                            summarize_anchor_distances(endpoint_distances)
                        )
                        junction_mean_dist, junction_median_dist, junction_p90_dist = (
                            summarize_anchor_distances(junction_distances)
                        )
                        coverage = coverage_from_anchors(anchors, gt_np, coverage_radii)

                        rows.append(
                            {
                                "image_id": image_id,
                                "score_type": mode,
                                "topK": topk,
                                "num_anchor": len(anchors),
                                "mean_anchor_score": (
                                    float(anchor_scores.mean()) if anchor_scores.size else 0.0
                                ),
                                "median_anchor_score": (
                                    float(np.median(anchor_scores)) if anchor_scores.size else 0.0
                                ),
                                "coverage_5": coverage.get(5, 0.0),
                                "coverage_10": coverage.get(10, 0.0),
                                "coverage_20": coverage.get(20, 0.0),
                                "mean_gt_distance": mean_dist,
                                "median_gt_distance": median_dist,
                                "p90_gt_distance": p90_dist,
                                "mean_pair_distance": pair_mean_dist,
                                "min_pair_distance": pair_min_dist,
                                "gt_endpoint_count": int(gt_endpoint.sum()),
                                "gt_junction_count": int(gt_junction.sum()),
                                "mean_endpoint_distance": endpoint_mean_dist,
                                "median_endpoint_distance": endpoint_median_dist,
                                "p90_endpoint_distance": endpoint_p90_dist,
                                "mean_junction_distance": junction_mean_dist,
                                "median_junction_distance": junction_median_dist,
                                "p90_junction_distance": junction_p90_dist,
                                "score_mean": score_stats["mean"],
                                "score_std": score_stats["std"],
                                "score_max": score_stats["max"],
                                "z_struct_mean": z_stats["mean"],
                                "z_struct_std": z_stats["std"],
                                "z_struct_max": z_stats["max"],
                                "z_struct_height": int(z_struct.shape[-2]),
                                "z_struct_width": int(z_struct.shape[-1]),
                                "z_struct_channels": int(z_struct.shape[1]),
                                "connectivity_entropy_mean": conn_entropy_stats["mean"],
                                "connectivity_entropy_std": conn_entropy_stats["std"],
                                "direction_variation_mean": direction_variation_stats["mean"],
                                "direction_variation_std": direction_variation_stats["std"],
                                "lambda_boost": float(args.lambda_boost),
                            }
                        )
                        for anchor_index, (x, y, anchor_score) in enumerate(anchors):
                            anchor_rows.append(
                                {
                                    "image_id": image_id,
                                    "score_type": mode,
                                    "topK": topk,
                                    "anchor_index": anchor_index,
                                    "x": x,
                                    "y": y,
                                    "score": anchor_score,
                                    "distance_to_gt_skeleton": (
                                        float(distances[anchor_index])
                                        if anchor_index < len(distances)
                                        else float("inf")
                                    ),
                                    "distance_to_gt_endpoint": (
                                        float(endpoint_distances[anchor_index])
                                        if anchor_index < len(endpoint_distances)
                                        else float("inf")
                                    ),
                                    "distance_to_gt_junction": (
                                        float(junction_distances[anchor_index])
                                        if anchor_index < len(junction_distances)
                                        else float("inf")
                                    ),
                                }
                            )

                        if topk in (32, 64) and save_this_visual:
                            vis_path = os.path.join(
                                args.output_dir,
                                "visualizations",
                                f"{image_id}_{mode}_topK{topk}.png",
                            )
                            save_visualization(image_rgb, anchors, vis_path)

                if save_this_visual:
                    visual_count += 1

            if should_print:
                print(
                    "[feature-anchor] batch {} done: z_shape={} z_mean={:.6f} z_std={:.6f} "
                    "conn_entropy_mean={:.6f} direction_variation_mean={:.6f}".format(
                        batch_index + 1,
                        tuple(z_struct.shape),
                        z_stats["mean"],
                        z_stats["std"],
                        conn_entropy_stats["mean"],
                        direction_variation_stats["mean"],
                    ),
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No batches were processed.")

    diag_path = os.path.join(args.output_dir, "topology_feature_anchor_diagnostic.csv")
    with open(diag_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    anchors_path = os.path.join(args.output_dir, "topology_feature_anchor_coordinates.csv")
    anchor_fieldnames = [
        "image_id",
        "score_type",
        "topK",
        "anchor_index",
        "x",
        "y",
        "score",
        "distance_to_gt_skeleton",
        "distance_to_gt_endpoint",
        "distance_to_gt_junction",
    ]
    with open(anchors_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=anchor_fieldnames)
        writer.writeheader()
        writer.writerows(anchor_rows)

    print("\nTopology feature anchor diagnostic", flush=True)
    for mode in ("A0_norm_x_surface", "A0_conn_entropy", "A0_direction_variation"):
        for topk in topk_values:
            selected = [
                row for row in rows if row["score_type"] == mode and row["topK"] == topk
            ]
            if not selected:
                continue
            mean_anchors = np.mean([row["num_anchor"] for row in selected])
            cov5 = np.mean([row["coverage_5"] for row in selected])
            cov10 = np.mean([row["coverage_10"] for row in selected])
            cov20 = np.mean([row["coverage_20"] for row in selected])
            med_dist = np.mean([row["median_gt_distance"] for row in selected])
            min_pair = np.mean([row["min_pair_distance"] for row in selected])
            mean_pair = np.mean([row["mean_pair_distance"] for row in selected])
            med_endpoint = np.mean([row["median_endpoint_distance"] for row in selected])
            med_junction = np.mean([row["median_junction_distance"] for row in selected])
            print(
                "{} topK={}: anchors={:.2f} coverage@5/10/20={:.4f}/{:.4f}/{:.4f} "
                "median_dist={:.3f} pair_mean/min={:.3f}/{:.3f} endpoint/junction_median={:.3f}/{:.3f}".format(
                    mode,
                    topk,
                    mean_anchors,
                    cov5,
                    cov10,
                    cov20,
                    med_dist,
                    mean_pair,
                    min_pair,
                    med_endpoint,
                    med_junction,
                ),
                flush=True,
            )
    print("saved:", diag_path, flush=True)
    print("saved:", anchors_path, flush=True)
    print("saved visualizations under:", os.path.join(args.output_dir, "visualizations"), flush=True)


if __name__ == "__main__":
    main()
