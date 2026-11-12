"""Offline native-coordinate gap status and conservative GT road-side matching."""
import heapq
import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

NEIGHBORS = [(dy, dx, math.hypot(dy, dx)) for dy in (-1, 0, 1)
             for dx in (-1, 0, 1) if dy or dx]


def points(indices, shape):
    value = np.asarray(indices, dtype=np.int64)
    if value.ndim != 1 or np.any(value < 0) or np.any(value >= np.prod(shape)):
        raise ValueError('Invalid original fixed-gap flat pixel coordinates')
    return np.column_stack(np.divmod(value, shape[1]))


def bounded_geodesic(road, seeds, limit):
    """8-neighbor Euclidean step lengths, restricted to local GT road pixels."""
    distance = np.full(road.shape, np.inf, dtype=np.float32)
    queue = []
    for y, x in np.argwhere(seeds & road):
        distance[y, x] = 0.
        heapq.heappush(queue, (0., int(y), int(x)))
    h, w = road.shape
    while queue:
        value, y, x = heapq.heappop(queue)
        if value > float(distance[y, x])+1e-5:
            continue
        for dy, dx, cost in NEIGHBORS:
            yy, xx = y+dy, x+dx
            proposed = value+cost
            if (0 <= yy < h and 0 <= xx < w and road[yy, xx]
                    and proposed <= limit and proposed < distance[yy, xx]-1e-5):
                distance[yy, xx] = proposed
                heapq.heappush(queue, (proposed, yy, xx))
    return distance


def gap_geometry(gt, gap, tolerance=12., margin=16):
    pixels = points(gap['pixels'], gt.shape)
    anchors = [points(group, gt.shape) for group in gap['anchors']]
    if not len(pixels) or len(anchors) < 2 or any(not len(group) for group in anchors):
        raise ValueError('Fixed gap must have pixels and at least two nonempty anchor groups')
    all_points = np.concatenate([pixels]+anchors)
    padding = max(margin, int(math.ceil(tolerance))+2)
    y0, x0 = np.maximum(all_points.min(0)-padding, 0)
    y1, x1 = np.minimum(all_points.max(0)+padding+1, gt.shape)
    origin = np.array([y0, x0])
    road = gt[y0:y1, x0:x1].astype(bool)
    gap_mask = np.zeros_like(road)
    gap_mask[tuple((pixels-origin).T)] = True
    seeds = []
    for group in anchors:
        mask = np.zeros_like(road)
        mask[tuple((group-origin).T)] = True
        seeds.append(mask)
    geometry = dict(bbox=(int(y0), int(y1), int(x0), int(x1)), pixels=pixels,
                    anchors=anchors, road=road, gap_mask=gap_mask, seeds=seeds,
                    valid=False, reason='', geometric=[], strict=[], tolerance=tolerance)
    if len(anchors) != 2:
        geometry['reason'] = 'not_two_anchor_groups'
        return geometry
    geometry['geometric'] = [distance_transform_edt(~seed) <= tolerance for seed in seeds]
    if not np.all(road[gap_mask]) or any(not np.all(road[seed]) for seed in seeds):
        geometry['reason'] = 'fixed_reference_pixels_not_GT_road'
        return geometry
    _, labels = cv2.connectedComponents(road.astype(np.uint8), connectivity=8)
    groups = [set(labels[seed])-{0} for seed in seeds]
    if any(len(group) != 1 for group in groups):
        geometry['reason'] = 'anchor_group_mixes_local_GT_roads'
        return geometry
    if groups[0] != groups[1] or set(labels[gap_mask]) != groups[0]:
        geometry['reason'] = 'no_single_local_GT_path_through_gap'
        return geometry
    distances = [bounded_geodesic(road, seed, tolerance) for seed in seeds]
    to_gap = bounded_geodesic(road, gap_mask, tolerance)
    # A side is closer along GT to its original anchors than to the other side
    # or to the missing centerline. Parallel roads cannot cross GT background.
    geometry['strict'] = [road & geometry['geometric'][side]
        & (distances[side] <= tolerance)
        & (distances[side] < distances[1-side]) & (distances[side] < to_gap)
        for side in (0, 1)]
    if any(not region.any() for region in geometry['strict']):
        geometry['reason'] = 'empty_or_ambiguous_GT_side'
        return geometry
    geometry['valid'] = True
    return geometry


def local_status(pred, gap, margin=16, coverage_threshold=.8):
    pixels = points(gap['pixels'], pred.shape)
    anchors = [points(group, pred.shape) for group in gap['anchors']]
    all_points = np.concatenate([pixels]+anchors)
    y0, x0 = np.maximum(all_points.min(0)-margin, 0)
    y1, x1 = np.minimum(all_points.max(0)+margin+1, pred.shape)
    _, labels = cv2.connectedComponents(pred[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    groups = [set(labels[group[:, 0]-y0, group[:, 1]-x0])-{0} for group in anchors]
    connected = bool(set.intersection(*groups))
    coverage = float(pred[pixels[:, 0], pixels[:, 1]].mean())
    return dict(anchor_connected_local=connected, still_disconnected=not connected,
                gap_coverage=coverage, closed_proxy=connected and coverage >= coverage_threshold)


def membership(coords, geometry, kind):
    y0, y1, x0, x1 = geometry['bbox']
    result = np.zeros((len(coords), 2), dtype=bool)
    if not len(coords) or len(geometry[kind]) != 2:
        return result
    y, x = coords[:, 0], coords[:, 1]
    valid = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
    for side, region in enumerate(geometry[kind]):
        result[valid, side] = region[y[valid]-y0, x[valid]-x0]
    return result


def distinct_pair(hits):
    for a in np.flatnonzero(hits[:, 0]):
        for b in np.flatnonzero(hits[:, 1]):
            if a != b:
                return int(a), int(b)
    return None


def window_hits(record, geometry):
    offset = np.array([record['top'], record['left']])
    candidates = record['candidate_coords'][record['candidate_valid']]+offset
    selected = record['selected_coords'][record['selected_valid']]+offset
    ch = membership(candidates, geometry, 'geometric')
    sh = membership(selected, geometry, 'geometric')
    strict_c = membership(candidates, geometry, 'strict')
    strict_s = membership(selected, geometry, 'strict')
    pair = distinct_pair(strict_s)
    y0, y1, x0, x1 = geometry['bbox']
    tile = record.get('tile', 512)
    ya, yb = max(record['top'], y0), min(record['top']+tile, y1)
    xa, xb = max(record['left'], x0), min(record['left']+tile, x1)
    visible = (geometry['valid'] and ya < yb and xa < xb and
               all(region[ya-y0:yb-y0, xa-x0:xb-x0].any() for region in geometry['strict']))
    return dict(candidates_both_geometric=distinct_pair(ch) is not None,
        selected_both_geometric=distinct_pair(sh) is not None,
        candidates_both_correct=distinct_pair(strict_c) is not None,
        selected_correct_pair=pair is not None,
        GT_sides_visible_in_window=bool(visible) if geometry['valid'] else None,
        candidate_side0=int(ch[:, 0].sum()), candidate_side1=int(ch[:, 1].sum()),
        selected_side0=int(sh[:, 0].sum()), selected_side1=int(sh[:, 1].sum()),
        correct_candidate_side0=int(strict_c[:, 0].sum()), correct_candidate_side1=int(strict_c[:, 1].sum()),
        correct_selected_side0=int(strict_s[:, 0].sum()), correct_selected_side1=int(strict_s[:, 1].sum()),
        selected_geometric_rejected_by_GT_side=(
            int(np.any(sh, 1).sum()-np.any(strict_s, 1).sum()) if geometry['valid'] else None),
        pair_nodes=[] if pair is None else selected[list(pair)].tolist())


def summarize_gap(records, geometry):
    rows = [dict(window=i, top=record['top'], left=record['left'], **window_hits(record, geometry))
            for i, record in enumerate(records)]
    keys = ('candidates_both_geometric', 'selected_both_geometric',
            'candidates_both_correct', 'selected_correct_pair')
    result = {key: any(row[key] for row in rows) for key in keys}
    result['windows_with_correct_pair'] = sum(row['selected_correct_pair'] for row in rows)
    result['windows_seeing_both_GT_sides'] = (sum(bool(row['GT_sides_visible_in_window']) for row in rows)
                                            if geometry['valid'] else None)
    result['no_joint_window'] = (not result['windows_seeing_both_GT_sides'] if geometry['valid'] else None)
    # Union-across-window diagnostics are secondary and never count as a pair.
    union0 = any(row['correct_selected_side0'] for row in rows)
    union1 = any(row['correct_selected_side1'] for row in rows)
    result['both_correct_only_across_windows'] = bool(union0 and union1 and not result['selected_correct_pair'])
    result['selected_GT_rejected_occurrences'] = (sum(row['selected_geometric_rejected_by_GT_side'] for row in rows)
                                               if geometry['valid'] else None)
    if not geometry['valid']:
        result['candidates_both_correct'] = result['selected_correct_pair'] = None
        result['both_correct_only_across_windows'] = None
    if len(geometry['anchors']) != 2:
        result['candidates_both_geometric'] = result['selected_both_geometric'] = None
    return result, rows
