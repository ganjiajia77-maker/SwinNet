"""Evaluate saved data1 masks with the Swin comparison's topology definitions.

The skeleton, component and sampled APLS calculations follow
diagnostics/compare_prediction_topology_metrics.py and
diagnostics/compare_connectivity_topology_metrics.py in the Swin worktree.
Run this on the masks written by test_data1.py, without another model pass.
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from eval_data1_common import center_crop_or_pad, image_names, mask_path


def zhang_suen_skeletonize(mask):
    image = mask.astype(np.uint8).copy()
    while True:
        changed = False
        for phase in (0, 1):
            center = image[1:-1, 1:-1]
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum((a == 0) & (b == 1) for a, b in
                              zip(neighbors, neighbors[1:] + neighbors[:1]))
            remove = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if phase == 0:
                remove &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if np.any(remove):
                changed = True
                center[remove] = 0
        if not changed:
            return image.astype(bool)


def skeletonize(mask):
    if (hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning')
            and hasattr(cv2.ximgproc, 'THINNING_ZHANGSUEN')):
        return cv2.ximgproc.thinning(
            mask.astype(np.uint8) * 255,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        ) > 127
    return zhang_suen_skeletonize(mask)


def component_stats(mask, short_area_threshold):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int64)
    total = float(areas.sum())
    largest = float(areas.max()) if areas.size else 0.0
    return {
        'components': float(max(count - 1, 0)),
        'short_components': float((areas < short_area_threshold).sum()) if areas.size else 0.0,
        'largest_ratio': largest / (total + 1e-8),
    }


def max_component_area(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0.0


def skeleton_graph(skeleton):
    """Build the same eight-neighbor weighted pixel graph as the Swin diagnostic."""
    points = np.argwhere(skeleton)
    height, width = skeleton.shape
    indices = np.full((height, width), -1, dtype=np.int32)
    indices[skeleton] = np.arange(len(points), dtype=np.int32)
    rows, columns, weights = [], [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        first = indices[y0:y1, x0:x1].ravel()
        second = indices[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        valid = (first >= 0) & (second >= 0)
        source, target = first[valid], second[valid]
        weight = float(np.hypot(dy, dx))
        rows.extend((source, target))
        columns.extend((target, source))
        weights.extend((np.full(len(source), weight), np.full(len(source), weight)))
    if rows:
        graph = csr_matrix((np.concatenate(weights),
                            (np.concatenate(rows), np.concatenate(columns))),
                           shape=(len(points), len(points)))
    else:
        graph = csr_matrix((len(points), len(points)))
    return points, graph


def sample_graph_nodes(graph, max_nodes):
    nodes = np.arange(graph.shape[0])
    if len(nodes) <= max_nodes:
        return nodes
    special = nodes[np.diff(graph.indptr) != 2]
    if len(special) >= max_nodes:
        indices = np.linspace(0, len(special) - 1, max_nodes).astype(int)
        return special[indices]
    remaining = max_nodes - len(special)
    regular = nodes[np.diff(graph.indptr) == 2]
    indices = np.linspace(0, len(regular) - 1, remaining).astype(int)
    return np.concatenate((special, regular[indices]))


def apls_score(gt_skeleton, pred_skeleton, max_nodes, snap_radius):
    """Approximate APLS from sampled skeleton nodes, matching Swin diagnostics."""
    gt_points, gt_graph = skeleton_graph(gt_skeleton)
    pred_points, pred_graph = skeleton_graph(pred_skeleton)
    if len(gt_points) < 2 or len(pred_points) < 2:
        return 0.0
    gt_nodes = sample_graph_nodes(gt_graph, max_nodes)
    pred_points = pred_points.astype(np.float32)
    snapped = []
    for node in gt_nodes:
        delta = pred_points - gt_points[node].astype(np.float32)
        distance = np.sqrt((delta * delta).sum(axis=1))
        index = int(distance.argmin())
        snapped.append(index if distance[index] <= snap_radius else None)
    gt_paths = np.atleast_2d(dijkstra(gt_graph, directed=False, indices=gt_nodes))
    pred_sources = list(dict.fromkeys(node for node in snapped if node is not None))
    pred_paths = (np.atleast_2d(dijkstra(pred_graph, directed=False, indices=pred_sources))
                  if pred_sources else None)
    pred_source_rows = {node: row for row, node in enumerate(pred_sources)}
    scores = []
    for i in range(len(gt_nodes)):
        for j in range(i + 1, len(gt_nodes)):
            gt_distance = gt_paths[i, gt_nodes[j]]
            if not np.isfinite(gt_distance) or gt_distance <= 0:
                continue
            pred_source, pred_target = snapped[i], snapped[j]
            pred_distance = None
            if pred_source is not None and pred_target is not None:
                pred_distance = pred_paths[pred_source_rows[pred_source], pred_target]
            if pred_distance is None or not np.isfinite(pred_distance):
                scores.append(0.0)
            else:
                scores.append(max(0.0, 1.0 - abs(pred_distance - gt_distance) / gt_distance))
    return float(np.mean(scores)) if scores else 0.0


def image_metrics(prediction, target, short_area_threshold, apls_max_nodes, apls_snap_radius):
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    pred_stats = component_stats(prediction, short_area_threshold)
    gt_stats = component_stats(target, short_area_threshold)
    pred_skeleton = skeletonize(prediction)
    gt_skeleton = skeletonize(target)
    pred_count = float(pred_skeleton.sum())
    gt_count = float(gt_skeleton.sum())
    topo_precision = float((pred_skeleton & target).sum()) / (pred_count + 1e-8)
    topo_recall = float((gt_skeleton & prediction).sum()) / (gt_count + 1e-8)
    topo_f1 = 2.0 * topo_precision * topo_recall / (topo_precision + topo_recall + 1e-8)
    missing = gt_skeleton & ~prediction
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'iou': tp / max(tp + fp + fn, 1),
        'pred_components': pred_stats['components'],
        'gt_components': gt_stats['components'],
        'frag_idx': pred_stats['components'] / max(gt_stats['components'], 1.0),
        'extra_components': max(pred_stats['components'] - gt_stats['components'], 0.0),
        'short_pred_components': pred_stats['short_components'],
        'short_gt_components': gt_stats['short_components'],
        'largest_component_ratio': pred_stats['largest_ratio'],
        'gt_largest_component_ratio': gt_stats['largest_ratio'],
        'fragment_density_per_1000_px': 1000.0 * pred_stats['components'] /
                                        (float(prediction.sum()) + 1e-8),
        'topo_precision': topo_precision, 'topo_recall': topo_recall,
        'topo_f1': topo_f1, 'cldice': topo_f1,
        'break_pixels': float(missing.sum()),
        'break_rate': float(missing.sum()) / (gt_count + 1e-8),
        'gap_components': component_stats(missing, 2)['components'],
        'max_gap_pixels': max_component_area(missing),
        'apls': apls_score(gt_skeleton, pred_skeleton, apls_max_nodes, apls_snap_radius)
                if apls_max_nodes > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Topology metrics from saved data1 predictions')
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--pred_dir', required=True, help='test_data1.py output_dir/masks')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--short_area_threshold', type=int, default=20)
    parser.add_argument('--apls_max_nodes', type=int, default=64)
    parser.add_argument('--apls_snap_radius', type=float, default=5.0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    names = image_names(args.root_path, args.split)
    if not names:
        raise ValueError('No images found in {}/{}/image'.format(args.root_path, args.split))
    rows = []
    csv_path = os.path.join(args.output_dir, 'topology_per_image.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = None
        for index, name in enumerate(names, 1):
            started = time.monotonic()
            print('[{}/{}] {} started'.format(index, len(names), name), flush=True)
            prediction_path = os.path.join(args.pred_dir, os.path.splitext(name)[0] + '.png')
            if not os.path.isfile(prediction_path):
                raise FileNotFoundError(prediction_path)
            with Image.open(prediction_path) as prediction_image:
                prediction = np.asarray(prediction_image.convert('L')) > 127
            target_path = mask_path(args.root_path, args.split, name)
            with Image.open(target_path) as target_image:
                target = np.asarray(center_crop_or_pad(
                    target_image.convert('L'), args.source_patch_size)) > 127
            if prediction.shape != target.shape:
                raise ValueError('Prediction/target shape mismatch for {}: {} vs {}'.format(
                    name, prediction.shape, target.shape))
            row = {'image_id': name, **image_metrics(prediction, target,
                       args.short_area_threshold, args.apls_max_nodes, args.apls_snap_radius)}
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print('[{}/{}] {} done in {:.1f}s'.format(
                index, len(names), name, time.monotonic() - started), flush=True)
    tp = sum(row['tp'] for row in rows)
    fp = sum(row['fp'] for row in rows)
    fn = sum(row['fn'] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    summary = {
        'images': len(rows),
        'iou': tp / max(tp + fp + fn, 1),
        'f1': 2.0 * precision * recall / max(precision + recall, 1e-8),
        'precision': precision, 'recall': recall,
    }
    for key in rows[0]:
        if key in ('image_id', 'tp', 'fp', 'fn', 'iou'):
            continue
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary.update({'split': args.split, 'short_area_threshold': args.short_area_threshold,
                    'apls_max_nodes': args.apls_max_nodes,
                    'apls_snap_radius': args.apls_snap_radius})
    summary_path = os.path.join(args.output_dir, 'topology_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    for key, value in summary.items():
        print('{}: {}'.format(key, value))
    print('Per-image CSV: {}'.format(csv_path))
    print('Summary JSON: {}'.format(summary_path))


if __name__ == '__main__':
    main()
