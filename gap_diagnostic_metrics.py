"""Shared verified-gap definitions from the existing H0 gate diagnostic."""
from itertools import combinations
import cv2
import numpy as np


def nearby_ids(anchor, pred, labels, kernel):
    nearby = cv2.dilate(anchor.astype(np.uint8), kernel).astype(bool) & pred
    return set(int(i) for i in np.unique(labels[nearby]) if i > 0)


def verified_gaps(gt_skeleton, baseline, tolerance, margin, min_pixels):
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    covered = cv2.dilate(baseline.astype(np.uint8), kernel).astype(bool)
    missing = gt_skeleton & ~covered
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8)
    gaps = []
    for gap_id in range(1, count):
        left, top, width, height, area = map(int, stats[gap_id])
        if area < min_pixels:
            continue
        y0, y1 = max(0, top - margin), min(missing.shape[0], top + height + margin)
        x0, x1 = max(0, left - margin), min(missing.shape[1], left + width + margin)
        region = components[y0:y1, x0:x1] == gap_id
        boundary = (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    & gt_skeleton[y0:y1, x0:x1] & covered[y0:y1, x0:x1] & ~region)
        anchor_count, anchor_labels, _, centers = cv2.connectedComponentsWithStats(
            boundary.astype(np.uint8), connectivity=8)
        local_baseline = baseline[y0:y1, x0:x1]
        _, baseline_labels = cv2.connectedComponents(local_baseline.astype(np.uint8), connectivity=8)
        anchors = []
        for anchor_id in range(1, anchor_count):
            anchor = anchor_labels == anchor_id
            ids = nearby_ids(anchor, local_baseline, baseline_labels, kernel)
            if ids:
                anchors.append((anchor, centers[anchor_id], ids))
        pairs = [(i, j) for i, j in combinations(range(len(anchors)), 2)
                 if np.linalg.norm(anchors[i][1] - anchors[j][1]) >= 4
                 and anchors[i][2].isdisjoint(anchors[j][2])]
        if pairs:
            gaps.append((gap_id, (y0, y1, x0, x1), region, anchors, pairs, area))
    return gaps, missing


def gap_metrics(gaps, floor, baseline_logit, floor_logit, tolerance):
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    covered_floor = cv2.dilate(floor.astype(np.uint8), kernel).astype(bool)
    records = []
    for gap_id, (y0, y1, x0, x1), region, anchors, pairs, area in gaps:
        local_floor = floor[y0:y1, x0:x1]
        _, floor_labels = cv2.connectedComponents(local_floor.astype(np.uint8), connectivity=8)
        floor_anchor_ids = [nearby_ids(anchor, local_floor, floor_labels, kernel)
                            for anchor, _, _ in anchors]
        connected = any(not floor_anchor_ids[i].isdisjoint(floor_anchor_ids[j])
                        and floor_anchor_ids[i] for i, j in pairs)
        center_covered = covered_floor[y0:y1, x0:x1] & region
        # Require the whole missed GT centerline and both flanks, not just edge pixels.
        full = bool(connected and np.all(center_covered[region]))
        boundary = np.zeros_like(region)
        for anchor, _, _ in anchors:
            boundary |= anchor
        edge = region & (cv2.distanceTransform((~boundary).astype(np.uint8),
                                                cv2.DIST_L2, 3) <= 4)
        interior = region & ~edge
        difference = floor_logit[y0:y1, x0:x1] - baseline_logit[y0:y1, x0:x1]
        records.append({"gap_id": gap_id, "x": x0, "y": y0, "pixels": area,
                        "connected_flanks": int(bool(connected)), "complete_closure": int(full),
                        "centerline_recovered_pixels": int(center_covered.sum()),
                        "centerline_recovered_fraction": float(center_covered.sum() / area),
                        "edge_recovered_pixels": int((center_covered & edge).sum()),
                        "interior_recovered_pixels": int((center_covered & interior).sum()),
                        "interior_pixels": int(interior.sum()),
                        "mean_gap_logit_delta": float(difference[region].mean()),
                        "mean_interior_logit_delta": float(difference[interior].mean())
                        if interior.any() else None})
    return records


def bbox_near(first, second, distance):
    x1, y1, w1, h1 = map(int, first)
    x2, y2, w2, h2 = map(int, second)
    dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
    dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
    return np.hypot(dx, dy) <= distance


def adjacent_gt_joins(pred, gt_labels, gt_stats, near_pixels):
    """Conservative screen: predicted components linking distinct nearby GT components."""
    _, labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    joined = set()
    overlapping = (pred & (gt_labels > 0))
    component_pairs = np.unique(np.stack((labels[overlapping], gt_labels[overlapping]), axis=1), axis=0)
    by_pred = {}
    for pred_id, gt_id in component_pairs:
        by_pred.setdefault(int(pred_id), set()).add(int(gt_id))
    for ids in by_pred.values():
        ids = sorted(ids)
        for first, second in combinations(ids, 2):
            if bbox_near(gt_stats[first, :4], gt_stats[second, :4], near_pixels):
                joined.add((first, second))
    return joined
