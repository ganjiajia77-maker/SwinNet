"""Coordinate-resolved probes on raster skeletons, in native image pixels."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import dijkstra

from road_metrics import skeleton_graph, skeletonize, snap_nodes


def gt_paths(gt, spacing, length):
    """Select one seed per pixel grid cell and one deterministic local target."""
    points, graph = skeleton_graph(skeletonize(gt))
    if not len(points):
        return []
    cells = points // spacing
    stride = (gt.shape[1] + spacing - 1) // spacing
    _, seeds = np.unique(cells[:, 0] * stride + cells[:, 1], return_index=True)
    result, seen = [], set()
    for seed in seeds:
        distance, predecessors = dijkstra(graph, directed=False, indices=seed,
                                          limit=length * 1.25, return_predecessors=True)
        candidates = np.flatnonzero((distance >= length * .75) & np.isfinite(distance))
        if not len(candidates):
            continue
        target = candidates[np.argmin(abs(distance[candidates] - length))]
        identity = tuple(sorted((int(seed), int(target))))
        if identity in seen:
            continue
        seen.add(identity)
        path = [int(target)]
        while path[-1] != seed:
            path.append(int(predecessors[path[-1]]))
        result.append((points[path[::-1]], float(distance[target])))
    return result


def longest_miss(missed):
    edges = np.flatnonzero(np.diff(np.r_[False, missed, False].astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max(initial=0))


def check_path(path, gt_length, skeleton, tolerance, min_gap):
    low = np.maximum(path.min(0) - tolerance, 0)
    high = np.minimum(path.max(0) + tolerance + 1, skeleton.shape)
    region = tuple(slice(a, b) for a, b in zip(low, high))
    local_path = path - low
    trace = np.zeros(tuple(high - low), dtype=bool)
    trace[tuple(local_path.T)] = True
    prediction = skeleton[region]
    corridor = prediction & (distance_transform_edt(~trace) <= tolerance)
    points, graph = skeleton_graph(corridor)
    ends = snap_nodes(local_path[[0, -1]], points, tolerance)
    miss = (distance_transform_edt(~prediction)[tuple(local_path.T)] > tolerance
            if prediction.any() else np.ones(len(path), dtype=bool))
    interior = miss[1:-1]
    missing_run = longest_miss(interior)
    miss_y, miss_x = None, None
    if missing_run:
        edges = np.flatnonzero(np.diff(np.r_[False, interior, False].astype(np.int8)))
        span = int(np.argmax(edges[1::2] - edges[::2]))
        midpoint = 1 + (edges[2*span] + edges[2*span+1]-1)//2
        miss_y, miss_x = map(int, path[midpoint])
    result = dict(status='endpoint_unmatched', max_missing_centerline_steps=missing_run,
                  missing_run_center_y=miss_y, missing_run_center_x=miss_x, predicted_path_px=None)
    if (ends < 0).any():
        return result
    if ends[0] == ends[1]:
        return dict(result, status='endpoints_merged')
    distance = dijkstra(graph, directed=False, indices=ends[0], limit=gt_length * 2)[ends[1]]
    if not np.isfinite(distance):
        # An 8-connected component check distinguishes disconnection from a detour
        # exceeding the bounded Dijkstra search.
        _, components = cv2.connectedComponents(corridor.astype(np.uint8), connectivity=8)
        joined = components[tuple(points[ends[0]])] == components[tuple(points[ends[1]])]
        status = ('path_length_mismatch' if joined else
                  'gap_disconnected' if missing_run >= min_gap else 'local_disconnected')
    else:
        status = 'matched' if .75 <= distance / gt_length <= 1.25 else 'path_length_mismatch'
    return dict(result, status=status, predicted_path_px=float(distance) if np.isfinite(distance) else None)


def unsupported_segments(skeleton, gt_distance, tolerance, minimum):
    unsupported = skeleton & (gt_distance > tolerance)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(unsupported.astype(np.uint8), connectivity=8)
    touches = cv2.dilate((skeleton & ~unsupported).astype(np.uint8), np.ones((3, 3), np.uint8))
    attached = set(np.unique(labels[touches.astype(bool)]))
    rows, retained = [], np.zeros(n, dtype=bool)
    for label in range(1, n):
        x, y, w, h, size = map(int, stats[label])
        if size >= minimum:
            retained[label] = True
            rows.append(dict(segment_id=label, center_y=float(centers[label, 1]),
                             center_x=float(centers[label, 0]), bbox_y=y, bbox_x=x,
                             bbox_height=h, bbox_width=w, skeleton_pixels=size,
                             kind='attached_unsupported' if label in attached else 'isolated_unsupported'))
    return rows, retained[labels]
