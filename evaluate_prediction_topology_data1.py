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

import cv2
import networkx as nx
import numpy as np
from PIL import Image

from eval_data1_common import center_crop_or_pad, image_names, mask_path


def zhang_suen_skeletonize(mask):
    image = mask.astype(np.uint8).copy()
    height, width = image.shape
    while True:
        changed = False
        for phase in (0, 1):
            remove = []
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if image[y, x] == 0:
                        continue
                    p2, p3, p4 = image[y - 1, x], image[y - 1, x + 1], image[y, x + 1]
                    p5, p6, p7 = image[y + 1, x + 1], image[y + 1, x], image[y + 1, x - 1]
                    p8, p9 = image[y, x - 1], image[y - 1, x - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    count = sum(neighbors)
                    transitions = sum(a == 0 and b == 1 for a, b in
                                      zip(neighbors, neighbors[1:] + neighbors[:1]))
                    if not (2 <= count <= 6 and transitions == 1):
                        continue
                    if phase == 0:
                        remove_pixel = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                    else:
                        remove_pixel = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                    if remove_pixel:
                        remove.append((y, x))
            if remove:
                changed = True
                for y, x in remove:
                    image[y, x] = 0
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
    graph = nx.Graph()
    points = [tuple(point) for point in np.argwhere(skeleton)]
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
        return [special[index] for index in indices]
    remaining = max_nodes - len(special)
    special_set = set(special)
    regular = [node for node in nodes if node not in special_set]
    indices = np.linspace(0, len(regular) - 1, remaining).astype(int)
    return special + [regular[index] for index in indices]


def apls_score(gt_skeleton, pred_skeleton, max_nodes, snap_radius):
    """Approximate APLS from sampled skeleton nodes, matching Swin diagnostics."""
    gt_graph = skeleton_graph(gt_skeleton)
    pred_graph = skeleton_graph(pred_skeleton)
    if gt_graph.number_of_nodes() < 2 or pred_graph.number_of_nodes() < 2:
        return 0.0
    gt_nodes = sample_graph_nodes(gt_graph, max_nodes)
    pred_points = np.asarray(list(pred_graph.nodes), dtype=np.float32)
    snapped = []
    for node in gt_nodes:
        delta = pred_points - np.asarray(node, dtype=np.float32)
        distance = np.sqrt((delta * delta).sum(axis=1))
        index = int(distance.argmin())
        snapped.append(tuple(pred_points[index].astype(int)) if distance[index] <= snap_radius else None)
    gt_paths = {node: nx.single_source_dijkstra_path_length(gt_graph, node, weight='weight')
                for node in gt_nodes}
    pred_paths = {
        node: nx.single_source_dijkstra_path_length(pred_graph, node, weight='weight')
        for node in dict.fromkeys(node for node in snapped if node is not None)
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
            if pred_distance is None:
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
    for index, name in enumerate(names, 1):
        prediction_path = os.path.join(args.pred_dir, os.path.splitext(name)[0] + '.png')
        if not os.path.isfile(prediction_path):
            raise FileNotFoundError(prediction_path)
        prediction = np.asarray(Image.open(prediction_path).convert('L')) > 127
        target_path = mask_path(args.root_path, args.split, name)
        target_image = Image.open(target_path).convert('L')
        target = np.asarray(center_crop_or_pad(target_image, args.source_patch_size)) > 127
        if prediction.shape != target.shape:
            raise ValueError('Prediction/target shape mismatch for {}: {} vs {}'.format(
                name, prediction.shape, target.shape))
        row = {'image_id': name, **image_metrics(prediction, target,
                   args.short_area_threshold, args.apls_max_nodes, args.apls_snap_radius)}
        rows.append(row)
        if index % 50 == 0 or index == len(names):
            print('[{}/{}] {}'.format(index, len(names), name), flush=True)
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
    csv_path = os.path.join(args.output_dir, 'topology_per_image.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.join(args.output_dir, 'topology_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    for key, value in summary.items():
        print('{}: {}'.format(key, value))
    print('Per-image CSV: {}'.format(csv_path))
    print('Summary JSON: {}'.format(summary_path))


if __name__ == '__main__':
    main()
