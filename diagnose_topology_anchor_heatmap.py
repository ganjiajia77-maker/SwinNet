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

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_transformer import load_topology_checkpoint_state


class ConfigArgs:
    root_path = "./data1"
    dataset = "ImageData"
    list_dir = "./lists/lists_Synapse"
    num_classes = 2
    cfg = "./configs/swin_tiny_patch4_window7_224_lite.yaml"
    img_size = 256
    batch_size = 1
    num_workers = 0
    zip = False
    cache_mode = ""
    resume = ""
    accumulation_steps = 0
    use_checkpoint = False
    amp_opt_level = ""
    tag = ""
    eval = True
    throughput = False
    n_class = 2
    opts = None


def update_args_from_checkpoint(args, checkpoint):
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        for name, value in saved_args.items():
            setattr(args, name, value)
    return args


def build_model(args, checkpoint, device):
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=getattr(args, "bottleneck_type", "global_local"),
        structure_profile=getattr(args, "structure_profile", "full"),
        use_msfe_skip=not getattr(args, "disable_msfe_skip", False),
        enable_highres_structure_stream=getattr(args, "enable_highres_structure_stream", False),
        highres_structure_channels=getattr(args, "highres_structure_channels", 64),
        highres_structure_fuse_stages=getattr(args, "highres_structure_fuse_stages", "stage23"),
        highres_structure_fusion_mode=getattr(args, "highres_structure_fusion_mode", "stage23"),
        enable_post_refine_structure_interaction=getattr(
            args,
            "enable_post_refine_structure_interaction",
            False,
        ),
        enable_global_topology=getattr(args, "enable_global_topology", True),
        global_topology_max_nodes=getattr(args, "global_topology_max_nodes", 32),
        global_topology_heads=getattr(args, "global_topology_heads", 4),
        global_topology_reach_hops=getattr(args, "global_topology_reach_hops", 12),
        global_topology_nms_radius=getattr(args, "global_topology_nms_radius", 2),
        global_topology_skeleton_threshold=getattr(args, "global_topology_skeleton_threshold", 0.5),
        global_topology_connectivity_threshold=getattr(args, "global_topology_connectivity_threshold", 0.25),
        global_topology_bend_angle_threshold=getattr(args, "global_topology_bend_angle_threshold", 45.0),
        global_topology_alpha_max=getattr(args, "global_topology_alpha_max", 0.05),
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=True,
    )
    return model.to(device).eval()


def find_stage3_output(stage_outputs):
    for item in reversed(stage_outputs):
        if item.get("stage") == 3 and item.get("connectivity") is not None:
            return item
    raise RuntimeError("No stage-3 connectivity output found.")


def find_highres_structure_skeleton(stage_outputs):
    for item in reversed(stage_outputs):
        if item.get("stage") == "highres_structure":
            skeleton = item.get("highres_structure_skeleton")
            if skeleton is not None:
                return skeleton
    raise RuntimeError("No highres_structure_skeleton found in stage outputs.")


def parse_int_list(raw):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def gaussian_smooth(score, kernel_size=5, sigma=1.0):
    if kernel_size <= 1:
        return score
    if kernel_size % 2 == 0:
        raise ValueError("--smooth_kernel must be odd.")
    radius = kernel_size // 2
    coords = torch.arange(
        -radius,
        radius + 1,
        device=score.device,
        dtype=score.dtype,
    )
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(score, kernel_2d, padding=radius)


def local_maxima_candidates(score, window_size=5):
    if window_size % 2 == 0:
        raise ValueError("--local_max_window must be odd.")
    pooled = F.max_pool2d(score, kernel_size=window_size, stride=1, padding=window_size // 2)
    maxima = (score >= pooled) & (score > 0)
    values = score[0, 0][maxima[0, 0]]
    ys, xs = torch.where(maxima[0, 0])
    if values.numel() == 0:
        return []
    order = torch.argsort(values, descending=True)
    candidates = []
    for idx in order.tolist():
        candidates.append((int(xs[idx].item()), int(ys[idx].item()), float(values[idx].item())))
    return candidates


def greedy_nms(candidates, max_count, radius):
    selected = []
    radius_sq = float(radius * radius)
    for x, y, score in candidates:
        keep = True
        for sx, sy, _ in selected:
            dx = float(x - sx)
            dy = float(y - sy)
            if dx * dx + dy * dy <= radius_sq:
                keep = False
                break
        if keep:
            selected.append((x, y, score))
            if len(selected) >= max_count:
                break
    return selected


def extract_topology_anchors(anchor_score, topk_values, nms_radius, local_max_window):
    max_topk = max(topk_values)
    smoothed = anchor_score
    candidates = local_maxima_candidates(smoothed, window_size=local_max_window)
    selected = greedy_nms(candidates, max_topk, nms_radius)
    return {topk: selected[:topk] for topk in topk_values}


def fps_from_threshold(anchor_score, threshold, topk_values):
    score_2d = anchor_score[0, 0]
    ys, xs = torch.where(score_2d >= float(threshold))
    if ys.numel() == 0:
        return {topk: [] for topk in topk_values}

    scores = score_2d[ys, xs]
    coords = torch.stack([xs.float(), ys.float()], dim=1)
    max_topk = min(max(topk_values), coords.shape[0])
    selected_indices = []

    first = int(torch.argmax(scores).item())
    selected_indices.append(first)
    min_dist_sq = ((coords - coords[first]) ** 2).sum(dim=1)

    for _ in range(1, max_topk):
        min_dist_sq[selected_indices] = -1.0
        next_index = int(torch.argmax(min_dist_sq).item())
        if min_dist_sq[next_index] < 0:
            break
        selected_indices.append(next_index)
        dist_sq = ((coords - coords[next_index]) ** 2).sum(dim=1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    selected = [
        (
            int(xs[index].item()),
            int(ys[index].item()),
            float(scores[index].item()),
        )
        for index in selected_indices
    ]
    return {topk: selected[: min(topk, len(selected))] for topk in topk_values}


def tensor_stats(tensor):
    values = tensor.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "max": float(values.max().item()),
    }


def normalize_direction_confidence(direction_logits):
    if direction_logits is None:
        return None
    confidence = torch.linalg.vector_norm(direction_logits, dim=1, keepdim=True)
    flat = confidence.flatten(start_dim=1)
    denom = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (confidence / denom).clamp(0.0, 1.0)


def build_anchor_scores(skeleton_prob, conn_prob, direction_logits, alpha, beta):
    degree_score = conn_prob.sum(dim=1, keepdim=True) / 8.0
    direction_score = normalize_direction_confidence(direction_logits)
    scores = {
        "A": skeleton_prob,
        "B": skeleton_prob * (1.0 + alpha * degree_score),
    }
    if direction_score is not None:
        scores["C"] = skeleton_prob * (1.0 + alpha * degree_score + beta * direction_score)
    else:
        scores["C"] = scores["B"]
    return scores, degree_score, direction_score


def unnormalize_image(image_tensor):
    image = image_tensor.detach().float().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)
    image = np.transpose(image, (1, 2, 0))
    return (image * 255.0).astype(np.uint8)


def overlay_anchors(image_rgb, anchors):
    canvas = image_rgb.copy()
    for x, y, _ in anchors:
        cv2.circle(canvas, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)
        cv2.circle(canvas, (int(x), int(y)), radius=5, color=(255, 255, 255), thickness=1)
    return canvas


def save_visualization(image_rgb, anchors, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    overlay = overlay_anchors(image_rgb, anchors)
    cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def anchor_distance_to_gt(anchors, gt_skeleton):
    if not anchors:
        return np.array([], dtype=np.float32)
    gt = (gt_skeleton > 0.5).astype(np.uint8)
    if gt.sum() == 0:
        return np.full((len(anchors),), np.inf, dtype=np.float32)
    distance = cv2.distanceTransform((1 - gt).astype(np.uint8), cv2.DIST_L2, 3)
    height, width = gt.shape
    values = []
    for x, y, _ in anchors:
        x = min(max(int(x), 0), width - 1)
        y = min(max(int(y), 0), height - 1)
        values.append(float(distance[y, x]))
    return np.asarray(values, dtype=np.float32)


def coverage_from_anchors(anchors, gt_skeleton, radii):
    gt = (gt_skeleton > 0.5).astype(np.uint8)
    gt_count = int(gt.sum())
    if gt_count == 0:
        return {radius: 0.0 for radius in radii}
    anchor_mask = np.zeros_like(gt, dtype=np.uint8)
    for x, y, _ in anchors:
        x = min(max(int(x), 0), gt.shape[1] - 1)
        y = min(max(int(y), 0), gt.shape[0] - 1)
        anchor_mask[y, x] = 1
    if anchor_mask.sum() == 0:
        return {radius: 0.0 for radius in radii}
    distance_to_anchor = cv2.distanceTransform((1 - anchor_mask).astype(np.uint8), cv2.DIST_L2, 3)
    return {
        radius: float(((distance_to_anchor <= radius) & (gt > 0)).sum() / max(gt_count, 1))
        for radius in radii
    }


def summarize_anchor_distances(distances):
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("inf"), float("inf"), float("inf")
    return (
        float(np.mean(finite)),
        float(np.median(finite)),
        float(np.percentile(finite, 90)),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether highres structure skeleton heatmaps plus stage3 "
            "connectivity can produce stable topology anchors."
        )
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
    parser.add_argument("--topk", default="16,32,64")
    parser.add_argument("--fps_thresholds", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--coverage_radii", default="5,10,20")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
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
    fps_thresholds = [float(value) for value in args.fps_thresholds.split(",") if value.strip()]
    coverage_radii = parse_int_list(args.coverage_radii)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[anchor-diag] device={device}", flush=True)
    print(f"[anchor-diag] loading checkpoint: {args.checkpoint}", flush=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[anchor-diag] checkpoint loaded in {time.perf_counter() - start_time:.2f}s", flush=True)

    config_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    print("[anchor-diag] building model", flush=True)
    start_time = time.perf_counter()
    model = build_model(config_args, checkpoint, device)
    print(f"[anchor-diag] model ready in {time.perf_counter() - start_time:.2f}s", flush=True)

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
    print(f"[anchor-diag] dataset size={len(dataset)}", flush=True)
    print(
        "[anchor-diag] modes: A=skeleton, B=skeleton+connectivity, C=skeleton+connectivity+direction",
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
                print(f"[anchor-diag] batch {batch_index + 1} forward start", flush=True)

            images = batch["image"].to(device)
            gt_skeleton = batch["skeleton"].to(device).float()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            stage_outputs = outputs[4] if len(outputs) > 4 else []
            highres_logits = find_highres_structure_skeleton(stage_outputs)
            stage3 = find_stage3_output(stage_outputs)
            skeleton_prob = torch.sigmoid(highres_logits)
            conn_prob = torch.sigmoid(stage3["connectivity"])
            direction_logits = stage3.get("direction")

            target_size = gt_skeleton.shape[-2:]
            if skeleton_prob.shape[-2:] != target_size:
                skeleton_prob = F.interpolate(
                    skeleton_prob,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if conn_prob.shape[-2:] != target_size:
                conn_prob = F.interpolate(
                    conn_prob,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if direction_logits is not None and direction_logits.shape[-2:] != target_size:
                direction_logits = F.interpolate(
                    direction_logits,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            scores, degree_score, direction_score = build_anchor_scores(
                skeleton_prob,
                conn_prob,
                direction_logits,
                alpha=args.alpha,
                beta=args.beta,
            )
            scores = {
                mode: gaussian_smooth(
                    score,
                    kernel_size=args.smooth_kernel,
                    sigma=args.smooth_sigma,
                )
                for mode, score in scores.items()
            }

            skeleton_stats = tensor_stats(skeleton_prob)
            conn_stats = tensor_stats(conn_prob)
            direction_mean = (
                float(direction_score.mean().item()) if direction_score is not None else float("nan")
            )
            skeleton_pos_ratios = {
                threshold: float((skeleton_prob > threshold).float().mean().item())
                for threshold in (0.1, 0.3, 0.5)
            }

            image_names = batch.get("image_name", [""] * images.shape[0])
            for sample_index in range(images.shape[0]):
                image_id = os.path.splitext(str(image_names[sample_index]))[0]
                image_rgb = unnormalize_image(images[sample_index])
                gt_np = gt_skeleton[sample_index, 0].detach().cpu().numpy()

                for mode, score_map in scores.items():
                    sample_score = score_map[sample_index:sample_index + 1]
                    anchors_by_topk = extract_topology_anchors(
                        sample_score,
                        topk_values,
                        nms_radius=args.nms_radius,
                        local_max_window=args.local_max_window,
                    )
                    for topk, anchors in anchors_by_topk.items():
                        anchor_scores = np.asarray(
                            [score for _, _, score in anchors],
                            dtype=np.float32,
                        )
                        distances = anchor_distance_to_gt(anchors, gt_np)
                        mean_dist, median_dist, p90_dist = summarize_anchor_distances(distances)
                        coverage = coverage_from_anchors(anchors, gt_np, coverage_radii)

                        rows.append(
                            {
                                "image_id": image_id,
                                "extraction": "peak",
                                "mode": mode,
                                "threshold": "",
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
                                "skeleton_prob_mean": skeleton_stats["mean"],
                                "skeleton_prob_std": skeleton_stats["std"],
                                "skeleton_positive_ratio_0.1": skeleton_pos_ratios[0.1],
                                "skeleton_positive_ratio_0.3": skeleton_pos_ratios[0.3],
                                "skeleton_positive_ratio_0.5": skeleton_pos_ratios[0.5],
                                "connectivity_mean": conn_stats["mean"],
                                "connectivity_std": conn_stats["std"],
                                "direction_confidence_mean": direction_mean,
                            }
                        )
                        for anchor_index, (x, y, anchor_score) in enumerate(anchors):
                            anchor_rows.append(
                                {
                                    "image_id": image_id,
                                    "extraction": "peak",
                                    "mode": mode,
                                    "threshold": "",
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
                                }
                            )

                        if topk in (32, 64) and (
                            args.max_visuals == 0 or visual_count < args.max_visuals
                        ):
                            vis_path = os.path.join(
                                args.output_dir,
                                "visualizations",
                                f"{image_id}_mode_{mode}_topK{topk}.png",
                            )
                            save_visualization(image_rgb, anchors, vis_path)

                    for threshold in fps_thresholds:
                        anchors_by_topk = fps_from_threshold(
                            sample_score,
                            threshold=threshold,
                            topk_values=topk_values,
                        )
                        for topk, anchors in anchors_by_topk.items():
                            anchor_scores = np.asarray(
                                [score for _, _, score in anchors],
                                dtype=np.float32,
                            )
                            distances = anchor_distance_to_gt(anchors, gt_np)
                            mean_dist, median_dist, p90_dist = summarize_anchor_distances(distances)
                            coverage = coverage_from_anchors(anchors, gt_np, coverage_radii)

                            rows.append(
                                {
                                    "image_id": image_id,
                                    "extraction": "threshold_fps",
                                    "mode": mode,
                                    "threshold": threshold,
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
                                    "skeleton_prob_mean": skeleton_stats["mean"],
                                    "skeleton_prob_std": skeleton_stats["std"],
                                    "skeleton_positive_ratio_0.1": skeleton_pos_ratios[0.1],
                                    "skeleton_positive_ratio_0.3": skeleton_pos_ratios[0.3],
                                    "skeleton_positive_ratio_0.5": skeleton_pos_ratios[0.5],
                                    "connectivity_mean": conn_stats["mean"],
                                    "connectivity_std": conn_stats["std"],
                                    "direction_confidence_mean": direction_mean,
                                }
                            )
                            for anchor_index, (x, y, anchor_score) in enumerate(anchors):
                                anchor_rows.append(
                                    {
                                        "image_id": image_id,
                                        "extraction": "threshold_fps",
                                        "mode": mode,
                                        "threshold": threshold,
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
                                    }
                                )

                            if topk in (32, 64) and (
                                args.max_visuals == 0 or visual_count < args.max_visuals
                            ):
                                vis_path = os.path.join(
                                    args.output_dir,
                                    "visualizations",
                                    f"{image_id}_mode_{mode}_threshold_fps_thr{threshold:g}_topK{topk}.png",
                                )
                                save_visualization(image_rgb, anchors, vis_path)

                    if args.max_visuals == 0 or visual_count < args.max_visuals:
                        visual_count += 1

            if should_print:
                print(
                    "[anchor-diag] batch {} done: skel_mean={:.6f} skel_std={:.6f} "
                    "pos>0.1/0.3/0.5={:.4f}/{:.4f}/{:.4f} conn_mean={:.6f} "
                    "conn_std={:.6f} dir_conf_mean={:.6f}".format(
                        batch_index + 1,
                        skeleton_stats["mean"],
                        skeleton_stats["std"],
                        skeleton_pos_ratios[0.1],
                        skeleton_pos_ratios[0.3],
                        skeleton_pos_ratios[0.5],
                        conn_stats["mean"],
                        conn_stats["std"],
                        direction_mean,
                    ),
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No batches were processed.")

    diag_path = os.path.join(args.output_dir, "topology_anchor_diagnostic.csv")
    with open(diag_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    anchors_path = os.path.join(args.output_dir, "topology_anchor_coordinates.csv")
    anchor_fieldnames = [
        "image_id",
        "extraction",
        "mode",
        "threshold",
        "topK",
        "anchor_index",
        "x",
        "y",
        "score",
        "distance_to_gt_skeleton",
    ]
    with open(anchors_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=anchor_fieldnames)
        writer.writeheader()
        writer.writerows(anchor_rows)

    print("\nTopology anchor diagnostic", flush=True)
    for extraction in ("peak", "threshold_fps"):
        thresholds = [""] if extraction == "peak" else fps_thresholds
        for mode in ("A", "B", "C"):
            for threshold in thresholds:
                for topk in topk_values:
                    selected = [
                        row
                        for row in rows
                        if row["extraction"] == extraction
                        and row["mode"] == mode
                        and row["threshold"] == threshold
                        and row["topK"] == topk
                    ]
                    if not selected:
                        continue
                    mean_anchors = np.mean([row["num_anchor"] for row in selected])
                    cov5 = np.mean([row["coverage_5"] for row in selected])
                    cov10 = np.mean([row["coverage_10"] for row in selected])
                    cov20 = np.mean([row["coverage_20"] for row in selected])
                    med_dist = np.mean([row["median_gt_distance"] for row in selected])
                    threshold_text = "" if threshold == "" else f" threshold={threshold:g}"
                    print(
                        "{} mode={}{} topK={}: anchors={:.2f} coverage@5/10/20={:.4f}/{:.4f}/{:.4f} median_dist={:.3f}".format(
                            extraction,
                            mode,
                            threshold_text,
                            topk,
                            mean_anchors,
                            cov5,
                            cov10,
                            cov20,
                            med_dist,
                        ),
                        flush=True,
                    )
    print("saved:", diag_path, flush=True)
    print("saved:", anchors_path, flush=True)
    print("saved visualizations under:", os.path.join(args.output_dir, "visualizations"), flush=True)


if __name__ == "__main__":
    main()
