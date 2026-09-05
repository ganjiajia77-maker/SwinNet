import argparse
import csv
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
from diagnose_topology_anchor_features import (
    anchor_distance_to_mask,
    anchor_pair_distance_stats,
    get_z_struct,
    gt_endpoint_junction_masks,
    minmax_normalize,
)
from diagnose_topology_anchor_heatmap import (
    ConfigArgs,
    anchor_distance_to_gt,
    build_model,
    coverage_from_anchors,
    extract_topology_anchors,
    gaussian_smooth,
    parse_int_list,
    save_visualization,
    summarize_anchor_distances,
    unnormalize_image,
    update_args_from_checkpoint,
)


def feature_surface_score(z_struct, surface_prob):
    norm_score = torch.linalg.vector_norm(z_struct.float(), dim=1, keepdim=True)
    norm_score = minmax_normalize(norm_score)
    if surface_prob.shape[-2:] != norm_score.shape[-2:]:
        surface_prob = F.interpolate(
            surface_prob,
            size=norm_score.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return norm_score * surface_prob


def fps_from_mask(mask, score_map, topk_values):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {topk: [] for topk in topk_values}

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    scores = score_map[ys, xs].astype(np.float32)
    max_topk = min(max(topk_values), coords.shape[0])

    selected = []
    first = int(np.argmax(scores))
    selected.append(first)
    min_dist_sq = np.sum((coords - coords[first]) ** 2, axis=1)

    for _ in range(1, max_topk):
        min_dist_sq[selected] = -1.0
        next_index = int(np.argmax(min_dist_sq))
        if min_dist_sq[next_index] < 0:
            break
        selected.append(next_index)
        dist_sq = np.sum((coords - coords[next_index]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    anchors = [
        (int(xs[index]), int(ys[index]), float(scores[index]))
        for index in selected
    ]
    return {topk: anchors[: min(topk, len(anchors))] for topk in topk_values}


def predicted_surface_skeleton(surface_prob_np, threshold):
    mask = (surface_prob_np >= float(threshold)).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    return (RoadSkeletonDataset._skeletonize_binary(mask) > 127).astype(np.uint8)


def evaluate_anchors(anchors, gt_skeleton, gt_endpoint, gt_junction, coverage_radii):
    distances = anchor_distance_to_gt(anchors, gt_skeleton)
    mean_dist, median_dist, p90_dist = summarize_anchor_distances(distances)
    endpoint_distances = anchor_distance_to_mask(anchors, gt_endpoint)
    junction_distances = anchor_distance_to_mask(anchors, gt_junction)
    endpoint_mean_dist, endpoint_median_dist, endpoint_p90_dist = (
        summarize_anchor_distances(endpoint_distances)
    )
    junction_mean_dist, junction_median_dist, junction_p90_dist = (
        summarize_anchor_distances(junction_distances)
    )
    pair_mean_dist, pair_min_dist = anchor_pair_distance_stats(anchors)
    coverage = coverage_from_anchors(anchors, gt_skeleton, coverage_radii)
    return {
        "coverage_5": coverage.get(5, 0.0),
        "coverage_10": coverage.get(10, 0.0),
        "coverage_20": coverage.get(20, 0.0),
        "mean_gt_distance": mean_dist,
        "median_gt_distance": median_dist,
        "p90_gt_distance": p90_dist,
        "mean_pair_distance": pair_mean_dist,
        "min_pair_distance": pair_min_dist,
        "mean_endpoint_distance": endpoint_mean_dist,
        "median_endpoint_distance": endpoint_median_dist,
        "p90_endpoint_distance": endpoint_p90_dist,
        "mean_junction_distance": junction_mean_dist,
        "median_junction_distance": junction_median_dist,
        "p90_junction_distance": junction_p90_dist,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare current anchors with GT-skeleton and predicted-surface-skeleton FPS anchors."
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
    parser.add_argument("--surface_threshold", type=float, default=0.5)
    parser.add_argument("--smooth_kernel", type=int, default=5)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--local_max_window", type=int, default=5)
    parser.add_argument("--nms_radius", type=int, default=4)
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
    print(f"[anchor-source] device={device}", flush=True)
    print(f"[anchor-source] loading checkpoint: {args.checkpoint}", flush=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[anchor-source] checkpoint loaded in {time.perf_counter() - start_time:.2f}s", flush=True)

    config_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    print("[anchor-source] building model", flush=True)
    start_time = time.perf_counter()
    model = build_model(config_args, checkpoint, device)
    print(f"[anchor-source] model ready in {time.perf_counter() - start_time:.2f}s", flush=True)

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
    print(f"[anchor-source] dataset size={len(dataset)}", flush=True)
    print(
        "[anchor-source] sources: current_feature_surface, gt_skeleton_fps, pred_surface_skeleton_fps",
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
                print(f"[anchor-source] batch {batch_index + 1} forward start", flush=True)

            images = batch["image"].to(device)
            gt_skeleton = batch["skeleton"].to(device).float()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            z_struct = get_z_struct(model)
            surface_prob = torch.sigmoid(outputs[0])
            target_size = gt_skeleton.shape[-2:]
            surface_prob_input = surface_prob
            if surface_prob_input.shape[-2:] != target_size:
                surface_prob_input = F.interpolate(
                    surface_prob_input,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            current_score = feature_surface_score(z_struct, surface_prob)
            current_score = F.interpolate(
                current_score,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            current_score = gaussian_smooth(
                current_score,
                kernel_size=args.smooth_kernel,
                sigma=args.smooth_sigma,
            )

            image_names = batch.get("image_name", [""] * images.shape[0])
            for sample_index in range(images.shape[0]):
                image_id = os.path.splitext(str(image_names[sample_index]))[0]
                image_rgb = unnormalize_image(images[sample_index])
                gt_np = gt_skeleton[sample_index, 0].detach().cpu().numpy()
                gt_endpoint, gt_junction = gt_endpoint_junction_masks(gt_np)
                surface_np = surface_prob_input[sample_index, 0].detach().cpu().numpy()
                current_np = current_score[sample_index, 0].detach().cpu().numpy()
                pred_skel = predicted_surface_skeleton(surface_np, args.surface_threshold)
                save_this_visual = args.max_visuals == 0 or visual_count < args.max_visuals

                sources = {
                    "current_feature_surface": extract_topology_anchors(
                        current_score[sample_index:sample_index + 1],
                        topk_values,
                        nms_radius=args.nms_radius,
                        local_max_window=args.local_max_window,
                    ),
                    "gt_skeleton_fps": fps_from_mask(
                        (gt_np > 0.5).astype(np.uint8),
                        np.ones_like(gt_np, dtype=np.float32),
                        topk_values,
                    ),
                    "pred_surface_skeleton_fps": fps_from_mask(
                        pred_skel,
                        surface_np.astype(np.float32),
                        topk_values,
                    ),
                }

                for source_name, anchors_by_topk in sources.items():
                    for topk, anchors in anchors_by_topk.items():
                        anchor_scores = np.asarray(
                            [score for _, _, score in anchors],
                            dtype=np.float32,
                        )
                        metrics = evaluate_anchors(
                            anchors,
                            gt_np,
                            gt_endpoint,
                            gt_junction,
                            coverage_radii,
                        )
                        row = {
                            "image_id": image_id,
                            "anchor_source": source_name,
                            "topK": topk,
                            "num_anchor": len(anchors),
                            "mean_anchor_score": (
                                float(anchor_scores.mean()) if anchor_scores.size else 0.0
                            ),
                            "median_anchor_score": (
                                float(np.median(anchor_scores)) if anchor_scores.size else 0.0
                            ),
                            "gt_skeleton_pixels": int((gt_np > 0.5).sum()),
                            "pred_surface_skeleton_pixels": int(pred_skel.sum()),
                            "pred_surface_threshold": float(args.surface_threshold),
                            "current_score_mean": float(current_np.mean()),
                            "current_score_max": float(current_np.max()),
                        }
                        row.update(metrics)
                        rows.append(row)

                        distances = anchor_distance_to_gt(anchors, gt_np)
                        endpoint_distances = anchor_distance_to_mask(anchors, gt_endpoint)
                        junction_distances = anchor_distance_to_mask(anchors, gt_junction)
                        for anchor_index, (x, y, anchor_score) in enumerate(anchors):
                            anchor_rows.append(
                                {
                                    "image_id": image_id,
                                    "anchor_source": source_name,
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
                                f"{image_id}_{source_name}_topK{topk}.png",
                            )
                            save_visualization(image_rgb, anchors, vis_path)

                if save_this_visual:
                    visual_count += 1

            if should_print:
                print(
                    "[anchor-source] batch {} done: surface_mean={:.6f} surface>thr={:.4f} current_score_mean={:.6f}".format(
                        batch_index + 1,
                        float(surface_prob_input.mean().item()),
                        float((surface_prob_input >= args.surface_threshold).float().mean().item()),
                        float(current_score.mean().item()),
                    ),
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No batches were processed.")

    diag_path = os.path.join(args.output_dir, "anchor_source_upper_bound_diagnostic.csv")
    with open(diag_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    coords_path = os.path.join(args.output_dir, "anchor_source_upper_bound_coordinates.csv")
    coord_fields = [
        "image_id",
        "anchor_source",
        "topK",
        "anchor_index",
        "x",
        "y",
        "score",
        "distance_to_gt_skeleton",
        "distance_to_gt_endpoint",
        "distance_to_gt_junction",
    ]
    with open(coords_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=coord_fields)
        writer.writeheader()
        writer.writerows(anchor_rows)

    print("\nAnchor source upper-bound diagnostic", flush=True)
    for source_name in (
        "current_feature_surface",
        "gt_skeleton_fps",
        "pred_surface_skeleton_fps",
    ):
        for topk in topk_values:
            selected = [
                row
                for row in rows
                if row["anchor_source"] == source_name and row["topK"] == topk
            ]
            if not selected:
                continue
            print(
                "{} topK={}: anchors={:.2f} coverage@10={:.4f} coverage@20={:.4f} "
                "median_skel={:.3f} median_junction={:.3f} pair_min={:.3f}".format(
                    source_name,
                    topk,
                    float(np.mean([row["num_anchor"] for row in selected])),
                    float(np.mean([row["coverage_10"] for row in selected])),
                    float(np.mean([row["coverage_20"] for row in selected])),
                    float(np.mean([row["median_gt_distance"] for row in selected])),
                    float(np.mean([row["median_junction_distance"] for row in selected])),
                    float(np.mean([row["min_pair_distance"] for row in selected])),
                ),
                flush=True,
            )
    print("saved:", diag_path, flush=True)
    print("saved:", coords_path, flush=True)
    print("saved visualizations under:", os.path.join(args.output_dir, "visualizations"), flush=True)


if __name__ == "__main__":
    main()
