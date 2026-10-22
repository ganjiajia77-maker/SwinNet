"""Prediction-only deterministic endpoint proposals. No ground truth is accepted."""
import cv2
import numpy as np

from evaluation.road_comparison.road_metrics import skeletonize

OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def neighbors(skeleton, point):
    y, x = point
    h, w = skeleton.shape
    return [(y + dy, x + dx) for dy, dx in OFFSETS
            if 0 <= y + dy < h and 0 <= x + dx < w and skeleton[y + dy, x + dx]]


def endpoints(probability, cfg):
    skeleton = skeletonize(probability >= cfg.threshold)
    count, components, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_component
    skeleton &= keep[components]
    padded = np.pad(skeleton.astype(np.uint8), 1)
    h, w = skeleton.shape
    degree = sum(padded[1+dy:1+dy+h, 1+dx:1+dx+w] for dy, dx in OFFSETS)
    found = []
    for y, x in np.argwhere(skeleton & (degree == 1)):
        margin = cfg.boundary_margin
        if not (margin <= y < h - margin and margin <= x < w - margin):
            continue
        start = (int(y), int(x))
        current, previous, length = start, None, 0
        for _ in range(cfg.trace_steps):
            options = [p for p in neighbors(skeleton, current) if p != previous]
            if len(options) != 1:
                break
            previous, current = current, options[0]
            length += 1
        if length < cfg.min_trace:
            continue  # suppress short terminal spurs without changing the segmentation mask
        tangent = np.asarray(start, np.float32) - np.asarray(current, np.float32)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        found.append((start, tangent, float(probability[y, x])))
    found.sort(key=lambda item: (-item[2], *item[0]))
    return found[:cfg.max_endpoints], len(found)


def generate_candidates(probabilities, cfg):
    results = []
    for probability in probabilities:
        points, raw_count = endpoints(probability, cfg)
        possible = []
        for i, (u, tu, pu) in enumerate(points):
            for j in range(i + 1, len(points)):
                v, tv, pv = points[j]
                displacement = np.asarray(v, np.float32) - np.asarray(u, np.float32)
                distance = float(np.linalg.norm(displacement))
                if not cfg.min_distance <= distance <= cfg.max_distance:
                    continue
                unit = displacement / distance
                cu, cv = float(tu @ unit), float(tv @ -unit)
                if min(cu, cv) < cfg.tangent_cosine:
                    continue
                geometry = [distance / cfg.max_distance, cu, cv, float(tu @ tv),
                            float(unit[1]), float(unit[0]), (pu + pv) / 2, min(pu, pv)]
                rank = min(cu, cv) + 0.25 * (pu + pv) - 0.25 * distance / cfg.max_distance
                possible.append((-rank, i, j, geometry))
        possible.sort(key=lambda edge: (edge[0], points[edge[1]][0], points[edge[2]][0]))
        degree = np.zeros(len(points), np.int32)
        edges, geometry = [], []
        for _, i, j, feature in possible:
            if degree[i] >= cfg.per_endpoint or degree[j] >= cfg.per_endpoint:
                continue
            degree[i] += 1
            degree[j] += 1
            edges.append([points[i][0], points[j][0]])
            geometry.append(feature)
            if len(edges) == cfg.max_edges:
                break
        results.append({'edges': np.asarray(edges, np.float32).reshape(-1, 2, 2),
                        'geometry': np.asarray(geometry, np.float32).reshape(-1, 8),
                        'endpoint_count': raw_count, 'retained_endpoints': len(points),
                        'eligible_pairs': len(possible)})
    return results
