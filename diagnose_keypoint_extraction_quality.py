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

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_connectivity_target
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_transformer import load_topology_checkpoint_state


CONNECTIVITY_DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


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
        enable_global_topology=True,
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


def raw_keypoint_scores(module, skeleton_prob, symmetric_connectivity):
    degree = (symmetric_connectivity >= module.connectivity_threshold).sum(dim=1)
    top2 = symmetric_connectivity.topk(k=min(2, symmetric_connectivity.shape[1]), dim=1).indices
    vectors = torch.tensor(
        CONNECTIVITY_DIRECTIONS,
        device=skeleton_prob.device,
        dtype=torch.float32,
    )
    top2_vectors = vectors[top2.permute(0, 2, 3, 1)]
    first = top2_vectors[..., 0, :]
    second = top2_vectors[..., 1, :]
    dot = (first * second).sum(dim=-1).abs()

    bend = (degree == 2) & (dot < math.cos(math.radians(module.bend_angle_threshold)))
    endpoint = degree == 1
    junction = degree >= 3
    scores = torch.stack(
        [
            skeleton_prob[:, 0] * endpoint.float(),
            skeleton_prob[:, 0] * junction.float(),
            skeleton_prob[:, 0] * bend.float(),
        ],
        dim=1,
    )
    return scores * (skeleton_prob[:, 0:1] >= module.skeleton_threshold).float()


def extract_with_counts(module, skeleton_prob, connectivity_prob, direction, connectivity_mode):
    if connectivity_mode == "symmetric":
        keypoint_connectivity = module.build_symmetric_connectivity(connectivity_prob)
    elif connectivity_mode == "raw":
        keypoint_connectivity = connectivity_prob
    else:
        raise ValueError(f"Unsupported connectivity_mode: {connectivity_mode}")

    scores = raw_keypoint_scores(module, skeleton_prob, keypoint_connectivity)
    pre_counts = (scores > 0).sum(dim=(2, 3))
    coords, node_types, valid, values = module.extract_keypoints(
        skeleton_prob,
        keypoint_connectivity,
        direction,
    )
    post_counts = torch.stack(
        [
            ((node_types == 0) & valid).sum(dim=1),
            ((node_types == 1) & valid).sum(dim=1),
            ((node_types == 2) & valid).sum(dim=1),
        ],
        dim=1,
    )
    return coords, node_types, valid, pre_counts, post_counts


def soft_keypoint_scores(module, skeleton_prob, keypoint_connectivity):
    topk = keypoint_connectivity.topk(k=min(3, keypoint_connectivity.shape[1]), dim=1)
    top_values = topk.values
    top_indices = topk.indices
    top1 = top_values[:, 0]
    top2 = top_values[:, 1] if top_values.shape[1] > 1 else torch.zeros_like(top1)
    top3 = top_values[:, 2] if top_values.shape[1] > 2 else torch.zeros_like(top1)

    vectors = torch.tensor(
        CONNECTIVITY_DIRECTIONS,
        device=skeleton_prob.device,
        dtype=torch.float32,
    )
    first = vectors[top_indices[:, 0]]
    second = vectors[top_indices[:, 1]] if top_indices.shape[1] > 1 else torch.zeros_like(first)
    bendness = 1.0 - (first * second).sum(dim=-1).abs().clamp(0.0, 1.0)

    skeleton = skeleton_prob[:, 0]
    endpoint_score = skeleton * top1 * (1.0 - top2).clamp_min(0.0)
    junction_score = skeleton * torch.minimum(torch.minimum(top1, top2), top3)
    bend_score = skeleton * torch.minimum(top1, top2) * bendness
    return torch.stack([endpoint_score, junction_score, bend_score], dim=1)


def extract_soft_topk_with_counts(module, skeleton_prob, connectivity_prob, direction, connectivity_mode, score_threshold):
    if connectivity_mode == "symmetric":
        keypoint_connectivity = module.build_symmetric_connectivity(connectivity_prob)
    elif connectivity_mode == "raw":
        keypoint_connectivity = connectivity_prob
    else:
        raise ValueError(f"Unsupported connectivity_mode: {connectivity_mode}")

    batch, _, height, width = skeleton_prob.shape
    scores = soft_keypoint_scores(module, skeleton_prob, keypoint_connectivity)
    scores = scores * (scores >= score_threshold).float()
    pre_counts = (scores > 0).sum(dim=(2, 3))
    pooled = F.max_pool2d(
        scores.reshape(batch * 3, 1, height, width),
        2 * module.nms_radius + 1,
        1,
        module.nms_radius,
    ).reshape(batch, 3, height, width)
    candidates = scores * (scores >= pooled).float()
    flat_scores = candidates.reshape(batch, -1)
    count = min(module.max_nodes, flat_scores.shape[1])
    values, indices = flat_scores.topk(count, dim=1)
    valid = values > 0
    node_types = indices // (height * width)
    flat = indices % (height * width)
    ys = flat // width
    xs = flat % width
    coords = torch.stack([ys, xs], dim=-1)
    post_counts = torch.stack(
        [
            ((node_types == 0) & valid).sum(dim=1),
            ((node_types == 1) & valid).sum(dim=1),
            ((node_types == 2) & valid).sum(dim=1),
        ],
        dim=1,
    )
    return coords, node_types, valid, pre_counts, post_counts


def extract_keypoints_for_diagnostic(
    module,
    skeleton_prob,
    connectivity_prob,
    direction,
    connectivity_mode,
    extraction_mode,
    soft_score_threshold,
):
    if extraction_mode == "hard":
        return extract_with_counts(module, skeleton_prob, connectivity_prob, direction, connectivity_mode)
    if extraction_mode == "soft_topk":
        return extract_soft_topk_with_counts(
            module,
            skeleton_prob,
            connectivity_prob,
            direction,
            connectivity_mode,
            soft_score_threshold,
        )
    raise ValueError(f"Unsupported extraction_mode: {extraction_mode}")


def thinning_binary(mask):
    mask_uint8 = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return (cv2.ximgproc.thinning(mask_uint8) > 0).astype(np.uint8)

    skeleton = np.zeros_like(mask_uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    working = mask_uint8.copy()
    while cv2.countNonZero(working) > 0:
        eroded = cv2.erode(working, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(working, opened))
        working = eroded
    return (skeleton > 0).astype(np.uint8)


def thin_skeleton_keypoints_single(module, skeleton_prob_2d, connectivity_prob, score_threshold):
    height, width = skeleton_prob_2d.shape
    binary = (skeleton_prob_2d >= module.skeleton_threshold).detach().cpu().numpy().astype(np.uint8)
    thin = thinning_binary(binary)
    thin_t = torch.from_numpy(thin).to(device=connectivity_prob.device, dtype=torch.bool)

    degree = torch.zeros((height, width), device=connectivity_prob.device, dtype=torch.long)
    edge_count = 0
    edge_weight_sum = 0.0
    for direction_index, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        neighbor = torch.zeros_like(thin_t)
        y_src_start = max(-dy, 0)
        y_src_end = height - max(dy, 0)
        x_src_start = max(-dx, 0)
        x_src_end = width - max(dx, 0)
        y_dst_start = max(dy, 0)
        y_dst_end = height - max(-dy, 0)
        x_dst_start = max(dx, 0)
        x_dst_end = width - max(-dx, 0)
        neighbor[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = thin_t[
            y_src_start:y_src_end,
            x_src_start:x_src_end,
        ]
        linked = thin_t & neighbor
        degree += linked.long()
        if direction_index in (2, 3, 4, 5):
            edge_weight = connectivity_prob[direction_index]
            edge_mask = linked & (edge_weight >= module.connectivity_threshold)
            edge_count += int(edge_mask.sum().item())
            edge_weight_sum += float(edge_weight[edge_mask].sum().item()) if edge_mask.any() else 0.0

    endpoint = thin_t & (degree == 1)
    junction = thin_t & (degree >= 3)
    bend = torch.zeros_like(thin_t)
    vectors = torch.tensor(CONNECTIVITY_DIRECTIONS, device=connectivity_prob.device, dtype=torch.float32)
    ys, xs = torch.where(thin_t & (degree == 2))
    for y, x in zip(ys.tolist(), xs.tolist()):
        dirs = []
        for index, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            yy = y + dy
            xx = x + dx
            if 0 <= yy < height and 0 <= xx < width and bool(thin_t[yy, xx]):
                dirs.append(index)
        if len(dirs) == 2:
            dot = float(torch.abs((vectors[dirs[0]] * vectors[dirs[1]]).sum()).item())
            if dot < math.cos(math.radians(module.bend_angle_threshold)):
                bend[y, x] = True

    scores = torch.stack(
        [
            skeleton_prob_2d * endpoint.float(),
            skeleton_prob_2d * junction.float(),
            skeleton_prob_2d * bend.float(),
        ],
        dim=0,
    )
    scores = scores * (scores >= score_threshold).float()
    return scores, edge_count, edge_weight_sum


def extract_thin_skeleton_256_with_counts(module, skeleton_prob, connectivity_prob, score_threshold):
    batch, _, height, width = skeleton_prob.shape
    if connectivity_prob.shape[-2:] != (height, width):
        connectivity_prob = F.interpolate(
            connectivity_prob,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    score_items = []
    edge_counts = []
    edge_weight_sums = []
    for batch_index in range(batch):
        scores, edge_count, edge_weight_sum = thin_skeleton_keypoints_single(
            module,
            skeleton_prob[batch_index, 0],
            connectivity_prob[batch_index],
            score_threshold,
        )
        score_items.append(scores)
        edge_counts.append(edge_count)
        edge_weight_sums.append(edge_weight_sum)

    scores = torch.stack(score_items, dim=0)
    pre_counts = (scores > 0).sum(dim=(2, 3))
    pooled = F.max_pool2d(
        scores.reshape(batch * 3, 1, height, width),
        2 * module.nms_radius + 1,
        1,
        module.nms_radius,
    ).reshape(batch, 3, height, width)
    candidates = scores * (scores >= pooled).float()
    flat_scores = candidates.reshape(batch, -1)
    count = min(module.max_nodes, flat_scores.shape[1])
    values, indices = flat_scores.topk(count, dim=1)
    valid = values > 0
    node_types = indices // (height * width)
    flat = indices % (height * width)
    ys = flat // width
    xs = flat % width
    coords = torch.stack([ys, xs], dim=-1)
    post_counts = torch.stack(
        [
            ((node_types == 0) & valid).sum(dim=1),
            ((node_types == 1) & valid).sum(dim=1),
            ((node_types == 2) & valid).sum(dim=1),
        ],
        dim=1,
    )
    edge_counts = torch.tensor(edge_counts, device=skeleton_prob.device, dtype=torch.long)
    edge_weight_sums = torch.tensor(edge_weight_sums, device=skeleton_prob.device, dtype=skeleton_prob.dtype)
    return coords, node_types, valid, pre_counts, post_counts, edge_counts, edge_weight_sums


def coords_by_type(coords, node_types, valid, type_index):
    mask = (node_types == type_index) & valid
    return coords[mask].float()


def nearest_neighbor_stats(points):
    if points.shape[0] <= 1:
        return float("nan"), float("nan"), float("nan")
    distances = torch.cdist(points.unsqueeze(0), points.unsqueeze(0))[0]
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values
    return float(nearest.mean()), float(nearest.median()), float(nearest.min())


def precision_recall_match(pred_points, gt_points, tolerance):
    if pred_points.numel() == 0 and gt_points.numel() == 0:
        return float("nan"), float("nan"), 0, 0, 0
    if pred_points.numel() == 0:
        return 0.0, 0.0, 0, 0, int(gt_points.shape[0])
    if gt_points.numel() == 0:
        return 0.0, float("nan"), 0, int(pred_points.shape[0]), 0

    distances = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0))[0]
    candidates = []
    for pred_index in range(distances.shape[0]):
        gt_index = int(distances[pred_index].argmin())
        distance = float(distances[pred_index, gt_index])
        if distance <= tolerance:
            candidates.append((distance, pred_index, gt_index))
    candidates.sort()

    used_pred = set()
    used_gt = set()
    true_positive = 0
    for _, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        true_positive += 1

    false_positive = int(pred_points.shape[0]) - true_positive
    false_negative = int(gt_points.shape[0]) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return precision, recall, true_positive, false_positive, false_negative


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
    return None


def resolve_predicted_skeleton(stage3, stage_outputs, final_skeleton_logits, target_size):
    if stage3.get("skeleton") is not None:
        return torch.sigmoid(stage3["skeleton"]), "stage3_skeleton"

    highres_skeleton = find_highres_structure_skeleton(stage_outputs)
    if highres_skeleton is not None:
        skeleton = torch.sigmoid(
            F.interpolate(
                highres_skeleton,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        return skeleton, "highres_structure_skeleton"

    if final_skeleton_logits is not None:
        skeleton = torch.sigmoid(
            F.interpolate(
                final_skeleton_logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        return skeleton, "final_skeleton_fallback"

    raise RuntimeError(
        "No predicted skeleton source found: stage3 has no 'skeleton', "
        "stage_outputs has no highres_structure_skeleton, and final skeleton "
        "logits are unavailable."
    )


def main():
    parser = argparse.ArgumentParser(description="Diagnose whether extracted topology nodes are real road keypoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--match_tol", type=float, default=3.0)
    parser.add_argument(
        "--skeleton_threshold_override",
        type=float,
        default=None,
        help="Temporarily override global_topology.skeleton_threshold for diagnostics only.",
    )
    parser.add_argument(
        "--connectivity_threshold_override",
        type=float,
        default=None,
        help="Temporarily override global_topology.connectivity_threshold for diagnostics only.",
    )
    parser.add_argument(
        "--connectivity_mode",
        choices=("symmetric", "raw"),
        default="symmetric",
        help=(
            "Use symmetric two-sided C8 connectivity, or raw one-sided C8 "
            "connectivity, for keypoint extraction diagnostics."
        ),
    )
    parser.add_argument(
        "--extraction_mode",
        choices=("hard", "soft_topk", "thin_skeleton_256"),
        default="hard",
        help=(
            "Use the original hard rule extraction, a diagnostic-only soft-score "
            "topK extractor, or 256px threshold/thinning skeleton nodes."
        ),
    )
    parser.add_argument(
        "--soft_score_threshold",
        type=float,
        default=1e-6,
        help="Minimum soft keypoint score used only when --extraction_mode soft_topk.",
    )
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}", flush=True)
    print(f"[diag] loading checkpoint: {args.checkpoint}", flush=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[diag] checkpoint loaded in {time.perf_counter() - start_time:.2f}s", flush=True)

    config_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    print("[diag] building model", flush=True)
    start_time = time.perf_counter()
    model = build_model(config_args, checkpoint, device)
    print(f"[diag] model ready in {time.perf_counter() - start_time:.2f}s", flush=True)
    module = model.swin_unet.global_topology
    original_skeleton_threshold = float(module.skeleton_threshold)
    original_connectivity_threshold = float(module.connectivity_threshold)
    if args.skeleton_threshold_override is not None:
        module.skeleton_threshold = float(args.skeleton_threshold_override)
    if args.connectivity_threshold_override is not None:
        module.connectivity_threshold = float(args.connectivity_threshold_override)
    effective_skeleton_threshold = float(module.skeleton_threshold)
    effective_connectivity_threshold = float(module.connectivity_threshold)
    print(
        "keypoint thresholds: skeleton {:.4f} -> {:.4f}, connectivity {:.4f} -> {:.4f}; connectivity_mode={}; extraction_mode={}".format(
            original_skeleton_threshold,
            effective_skeleton_threshold,
            original_connectivity_threshold,
            effective_connectivity_threshold,
            args.connectivity_mode,
            args.extraction_mode,
        ),
        flush=True,
    )
    print(f"[diag] loading dataset: root={args.root_path} split={args.split}", flush=True)
    start_time = time.perf_counter()
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
    )
    print(f"[diag] dataset size={len(dataset)} loaded in {time.perf_counter() - start_time:.2f}s", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows = []
    aggregate = {
        "pred_nodes": 0,
        "gt_nodes": 0,
        "pred_pre_endpoint": 0,
        "pred_pre_junction": 0,
        "pred_pre_bend": 0,
        "pred_post_endpoint": 0,
        "pred_post_junction": 0,
        "pred_post_bend": 0,
        "gt_pre_endpoint": 0,
        "gt_pre_junction": 0,
        "gt_pre_bend": 0,
        "gt_post_endpoint": 0,
        "gt_post_junction": 0,
        "gt_post_bend": 0,
        "pred_edge_count": 0,
        "gt_edge_count": 0,
        "endpoint_tp": 0,
        "endpoint_fp": 0,
        "endpoint_fn": 0,
        "junction_tp": 0,
        "junction_fp": 0,
        "junction_fn": 0,
        "bend_tp": 0,
        "bend_fp": 0,
        "bend_fn": 0,
    }
    nn_means = []
    nn_medians = []
    nn_mins = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch_start = time.perf_counter()
            should_print = args.print_freq > 0 and batch_index % args.print_freq == 0
            if should_print:
                print(f"[diag] batch {batch_index + 1} start", flush=True)

            images = batch["image"].to(device)
            gt_skeleton = batch["skeleton"].to(device)
            if should_print:
                print(f"[diag] batch {batch_index + 1} forward start", flush=True)
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if should_print:
                print(f"[diag] batch {batch_index + 1} forward done", flush=True)
            stage3 = find_stage3_output(outputs[4])

            pred_connectivity = torch.sigmoid(stage3["connectivity"])
            pred_direction = stage3["direction"]
            height, width = pred_connectivity.shape[-2:]
            final_skeleton_logits = outputs[2] if len(outputs) > 2 else None
            pred_edge_count = 0
            gt_edge_count = 0
            if args.extraction_mode == "thin_skeleton_256":
                target_size = gt_skeleton.shape[-2:]
                pred_skeleton, pred_skeleton_source = resolve_predicted_skeleton(
                    stage3,
                    outputs[4],
                    final_skeleton_logits,
                    target_size,
                )
                gt_skeleton_stage = (gt_skeleton > 0.5).float()
                gt_connectivity = build_connectivity_target(
                    gt_skeleton_stage,
                    erode_kernel_size=1,
                ).to(device)
                (
                    pred_coords,
                    pred_types,
                    pred_valid,
                    pred_pre,
                    pred_post,
                    pred_edges,
                    pred_edge_weights,
                ) = extract_thin_skeleton_256_with_counts(
                    module,
                    pred_skeleton,
                    pred_connectivity,
                    args.soft_score_threshold,
                )
                (
                    gt_coords,
                    gt_types,
                    gt_valid,
                    gt_pre,
                    gt_post,
                    gt_edges,
                    gt_edge_weights,
                ) = extract_thin_skeleton_256_with_counts(
                    module,
                    gt_skeleton_stage,
                    gt_connectivity,
                    args.soft_score_threshold,
                )
                pred_edge_count = int(pred_edges.sum().item())
                gt_edge_count = int(gt_edges.sum().item())
            else:
                pred_skeleton, pred_skeleton_source = resolve_predicted_skeleton(
                    stage3,
                    outputs[4],
                    final_skeleton_logits,
                    (height, width),
                )
                gt_skeleton_stage = F.interpolate(
                    (gt_skeleton > 0.5).float(),
                    size=(height, width),
                    mode="nearest",
                )
                gt_connectivity = build_connectivity_target(
                    gt_skeleton_stage,
                    erode_kernel_size=1,
                ).to(device)
                gt_direction = torch.zeros(gt_skeleton_stage.shape[0], 2, height, width, device=device)

                pred_coords, pred_types, pred_valid, pred_pre, pred_post = extract_keypoints_for_diagnostic(
                    module,
                    pred_skeleton,
                    pred_connectivity,
                    pred_direction,
                    args.connectivity_mode,
                    args.extraction_mode,
                    args.soft_score_threshold,
                )
                gt_coords, gt_types, gt_valid, gt_pre, gt_post = extract_keypoints_for_diagnostic(
                    module,
                    gt_skeleton_stage,
                    gt_connectivity,
                    gt_direction,
                    args.connectivity_mode,
                    args.extraction_mode,
                    args.soft_score_threshold,
                )

            pred_points = pred_coords[pred_valid].float()
            nn_mean, nn_median, nn_min = nearest_neighbor_stats(pred_points)
            if not math.isnan(nn_mean):
                nn_means.append(nn_mean)
                nn_medians.append(nn_median)
                nn_mins.append(nn_min)

            row = {
                "batch_index": batch_index,
                "pred_skeleton_source": pred_skeleton_source,
                "pred_node_count": int(pred_valid.sum().item()),
                "pred_endpoint_count": int(pred_post[:, 0].sum().item()),
                "pred_junction_count": int(pred_post[:, 1].sum().item()),
                "pred_bend_count": int(pred_post[:, 2].sum().item()),
                "pred_pre_endpoint": int(pred_pre[:, 0].sum().item()),
                "pred_pre_junction": int(pred_pre[:, 1].sum().item()),
                "pred_pre_bend": int(pred_pre[:, 2].sum().item()),
                "pred_edge_count": pred_edge_count,
                "gt_node_count": int(gt_valid.sum().item()),
                "gt_endpoint_count": int(gt_post[:, 0].sum().item()),
                "gt_junction_count": int(gt_post[:, 1].sum().item()),
                "gt_bend_count": int(gt_post[:, 2].sum().item()),
                "gt_pre_endpoint": int(gt_pre[:, 0].sum().item()),
                "gt_pre_junction": int(gt_pre[:, 1].sum().item()),
                "gt_pre_bend": int(gt_pre[:, 2].sum().item()),
                "gt_edge_count": gt_edge_count,
                "pred_nn_mean": nn_mean,
                "pred_nn_median": nn_median,
                "pred_nn_min": nn_min,
            }

            aggregate["pred_nodes"] += row["pred_node_count"]
            aggregate["gt_nodes"] += row["gt_node_count"]
            aggregate["pred_edge_count"] += row["pred_edge_count"]
            aggregate["gt_edge_count"] += row["gt_edge_count"]
            for name in ("endpoint", "junction", "bend"):
                aggregate[f"pred_pre_{name}"] += row[f"pred_pre_{name}"]
                aggregate[f"pred_post_{name}"] += row[f"pred_{name}_count"]
                aggregate[f"gt_pre_{name}"] += row[f"gt_pre_{name}"]
                aggregate[f"gt_post_{name}"] += row[f"gt_{name}_count"]

                type_index = {"endpoint": 0, "junction": 1, "bend": 2}[name]
                pred_type_points = coords_by_type(pred_coords, pred_types, pred_valid, type_index)
                gt_type_points = coords_by_type(gt_coords, gt_types, gt_valid, type_index)
                precision, recall, tp, fp, fn = precision_recall_match(
                    pred_type_points,
                    gt_type_points,
                    args.match_tol,
                )
                row[f"{name}_precision"] = precision
                row[f"{name}_recall"] = recall
                row[f"{name}_tp"] = tp
                row[f"{name}_fp"] = fp
                row[f"{name}_fn"] = fn
                aggregate[f"{name}_tp"] += tp
                aggregate[f"{name}_fp"] += fp
                aggregate[f"{name}_fn"] += fn

            rows.append(row)
            if should_print:
                print(
                    "[diag] batch {} done in {:.2f}s: nodes={} endpoints={} junctions={} bends={} edges={}".format(
                        batch_index + 1,
                        time.perf_counter() - batch_start,
                        row["pred_node_count"],
                        row["pred_endpoint_count"],
                        row["pred_junction_count"],
                        row["pred_bend_count"],
                        row["pred_edge_count"],
                    ),
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No batches were processed.")

    cases_path = os.path.join(args.output_dir, "keypoint_extraction_cases.csv")
    with open(cases_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(aggregate)
    for name in ("endpoint", "junction", "bend"):
        tp = aggregate[f"{name}_tp"]
        fp = aggregate[f"{name}_fp"]
        fn = aggregate[f"{name}_fn"]
        summary[f"{name}_precision"] = tp / max(tp + fp, 1)
        summary[f"{name}_recall"] = tp / max(tp + fn, 1)
    summary["pred_nn_mean_avg"] = sum(nn_means) / max(len(nn_means), 1)
    summary["pred_nn_median_avg"] = sum(nn_medians) / max(len(nn_medians), 1)
    summary["pred_nn_min_avg"] = sum(nn_mins) / max(len(nn_mins), 1)
    summary["match_tol"] = args.match_tol
    summary["original_skeleton_threshold"] = original_skeleton_threshold
    summary["effective_skeleton_threshold"] = effective_skeleton_threshold
    summary["original_connectivity_threshold"] = original_connectivity_threshold
    summary["effective_connectivity_threshold"] = effective_connectivity_threshold
    summary["connectivity_mode"] = args.connectivity_mode
    summary["extraction_mode"] = args.extraction_mode
    summary["soft_score_threshold"] = args.soft_score_threshold

    summary_path = os.path.join(args.output_dir, "keypoint_extraction_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    for key, value in summary.items():
        print(f"{key}={value}")
    print("saved:", summary_path)
    print("saved:", cases_path)


if __name__ == "__main__":
    main()
