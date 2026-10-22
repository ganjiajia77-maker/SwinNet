"""Native-resolution local-path labels; GT never enters model.forward()."""
import heapq
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from gap_candidates import OFFSETS
from local_gap_config import LocalGapConfig


def snap_point(skeleton, mask, point, radius):
    y, x = point
    h, w = skeleton.shape
    y0, y1 = max(0, int(y-radius)), min(h, int(y+radius)+2)
    x0, x1 = max(0, int(x-radius)), min(w, int(x+radius)+2)
    patch = skeleton[y0:y1, x0:x1]
    points = np.argwhere(patch) + [y0, x0]
    if not len(points):
        return None, 'background' if not mask[y0:y1, x0:x1].any() else 'unmatched'
    distance = np.linalg.norm(points - point, axis=1)
    order = np.argsort(distance, kind='stable')
    if distance[order[0]] > radius:
        return None, 'unmatched'
    _, components = cv2.connectedComponents(patch.astype(np.uint8), connectivity=8)
    local = points - [y0, x0]
    ids = components[local[:, 0], local[:, 1]]
    if np.any((ids != ids[order[0]]) & (distance <= distance[order[0]] + 2)):
        return None, 'ambiguous_snap'
    return tuple(points[order[0]]), 'matched'


def short_path(skeleton, start, finish, allowed, limit):
    queue, distance, parent = [(0., start)], {start: 0.}, {}
    h, w = skeleton.shape
    while queue:
        cost, point = heapq.heappop(queue)
        if cost > distance[point] or cost > limit:
            continue
        if point == finish:
            path = [point]
            while point != start:
                point = parent[point]
                path.append(point)
            return path
        for dy, dx in OFFSETS:
            q = point[0] + dy, point[1] + dx
            if not (0 <= q[0] < h and 0 <= q[1] < w and skeleton[q] and allowed[q]):
                continue
            new_cost = cost + np.hypot(dy, dx)
            if new_cost <= limit and new_cost < distance.get(q, np.inf):
                distance[q], parent[q] = new_cost, point
                heapq.heappush(queue, (new_cost, q))
    return None


def has_junction(skeleton, path):
    padded = np.pad(skeleton, 1)
    for y, x in path[2:-2]:
        values = [bool(padded[y+1+dy, x+1+dx]) for dy, dx in OFFSETS]
        transitions = sum(not a and b for a, b in zip(values, values[1:] + values[:1]))
        if transitions >= 3:
            return True
    return False


def label_edges(skeleton, mask, edges_yx, feature_shape, cfg):
    """1=verified bounded local GT path; 0=clear negative; -1=ambiguous."""
    scale = np.asarray(skeleton.shape) / np.asarray(feature_shape)
    native = (np.asarray(edges_yx) + 0.5) * scale - 0.5
    labels, reasons = [], []
    for u, v in native:
        start, sr = snap_point(skeleton, mask, u, cfg.gt_snap_radius)
        finish, fr = snap_point(skeleton, mask, v, cfg.gt_snap_radius)
        if sr == 'background' or fr == 'background':
            labels.append(0); reasons.append('background_endpoint'); continue
        if start is None or finish is None or start == finish:
            labels.append(-1); reasons.append('ambiguous_endpoint'); continue
        length = float(np.linalg.norm(v-u))
        margin = max(32., cfg.gt_corridor_radius * 2, cfg.gt_snap_radius * 2)
        lower = np.maximum(0, np.floor(np.minimum(u, v)-margin).astype(int))
        upper = np.minimum(skeleton.shape, np.ceil(np.maximum(u, v)+margin+1).astype(int))
        y0, x0 = lower; y1, x1 = upper
        region = skeleton[y0:y1, x0:x1]
        a, b = tuple(np.asarray(start)-lower), tuple(np.asarray(finish)-lower)
        yy, xx = np.indices(region.shape)
        points = np.stack([yy+y0, xx+x0], -1)
        vector = v-u
        t = ((points-u)*vector).sum(-1) / max(length*length, 1e-6)
        nearest = u + np.clip(t, 0, 1)[..., None]*vector
        allowed = np.linalg.norm(points-nearest, axis=-1) <= cfg.gt_corridor_radius
        allowed[a] = allowed[b] = True
        limit = cfg.gt_detour_ratio * length + 2 * cfg.gt_snap_radius
        path = short_path(region, a, b, allowed, limit)
        if path is not None:
            if has_junction(region, path):
                labels.append(-1); reasons.append('junction_ambiguity')
            else:
                labels.append(1); reasons.append('local_path')
            continue
        # A nearby bent route outside a straight corridor is not a trustworthy negative.
        detour = short_path(region, a, b, np.ones_like(region), limit * 2)
        if detour is not None:
            labels.append(-1); reasons.append('bent_or_detouring_path')
        else:
            labels.append(0); reasons.append('no_local_path')
    return np.asarray(labels, np.int64), reasons


def proposal_gap_coverage(skeleton, p2, edges, labels, cfg):
    """Coverage of P2 misses on GT centerlines with two locally disconnected predicted flanks.

    This measures the P2 candidate-search ceiling, not final Surface repair success.
    Only gaps within the configured search-length range are counted.
    """
    h, w = skeleton.shape
    predicted = cv2.resize((p2 >= cfg.threshold).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    covered = cv2.dilate(predicted, np.ones((5, 5), np.uint8)) > 0
    count, components = cv2.connectedComponents((skeleton & ~covered).astype(np.uint8), connectivity=8)
    scale = np.asarray([h, w]) / np.asarray(p2.shape)
    native_edges = (np.asarray(edges) + .5) * scale - .5
    verified, detected = 0, 0
    for index in range(1, count):
        gap = components == index
        points = np.argwhere(gap)
        if len(points) < cfg.min_distance * min(scale) or len(points) > cfg.max_distance * max(scale) * 1.5:
            continue
        lo, hi = points.min(0), points.max(0)
        if np.any(lo < 8) or np.any(hi >= np.asarray([h, w])-8):
            continue
        # Work locally so distant loops cannot erase a genuine local disconnection.
        lower = np.maximum(0, lo-12); upper = np.minimum([h, w], hi+13)
        y0, x0 = lower; y1, x1 = upper
        local_gap = gap[y0:y1, x0:x1]
        ring = cv2.dilate(local_gap.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        local_skeleton = skeleton[y0:y1, x0:x1]
        flanks = ring & local_skeleton & covered[y0:y1, x0:x1] & ~local_gap
        n, flank_id = cv2.connectedComponents(flanks.astype(np.uint8), connectivity=8)
        if n != 3:
            continue
        _, local_gt_id = cv2.connectedComponents(local_skeleton.astype(np.uint8), connectivity=8)
        if len(set(local_gt_id[ring & local_skeleton].tolist()) - {0}) != 1:
            continue
        _, pred_id = cv2.connectedComponents(predicted[y0:y1, x0:x1], connectivity=8)
        ids = []
        for k in (1, 2):
            around = cv2.dilate((flank_id == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            ids.append(set(pred_id[around].tolist()) - {0})
        if not ids[0] or not ids[1] or ids[0] & ids[1]:
            continue
        verified += 1
        for edge, label in zip(native_edges, labels):
            if label != 1:
                continue
            vector = edge[1]-edge[0]
            t = np.clip(((points-edge[0])*vector).sum(-1) / max(float(vector@vector), 1e-6), 0, 1)
            distance = np.linalg.norm(points-(edge[0]+t[:, None]*vector), axis=1)
            if np.mean(distance <= cfg.gt_corridor_radius) >= .8:
                detected += 1
                break
    return verified, detected


def gap_auxiliary_loss(stage_outputs, surface_gt, skeleton_gt, audit_coverage=False):
    zero = surface_gt.new_zeros(())
    for item in stage_outputs or []:
        if item.get('stage') != 'gap_connection':
            continue
        cfg = LocalGapConfig.parse(item['gap_config'])
        p2_logits = item['p2_logits']
        target = F.adaptive_avg_pool2d(surface_gt.float(), p2_logits.shape[-2:])
        bce = F.binary_cross_entropy_with_logits(p2_logits.float(), target)
        probability = p2_logits.float().sigmoid()
        dice = 1 - ((2*(probability*target).sum((1,2,3))+1e-6) /
                    ((probability+target).sum((1,2,3))+1e-6)).mean()
        p2_loss = bce + .5 * dice
        positive_losses, negative_losses, labels_all = [], [], []
        correct_positive = predicted_positive = true_positive = 0
        verified_gaps = detected_gaps = 0
        reasons = {}
        skels = skeleton_gt.detach().cpu().numpy()[:, 0] > .5
        masks = surface_gt.detach().cpu().numpy()[:, 0] > .5
        p2_arrays = probability.detach().cpu().numpy()[:, 0]
        for b, (logits, edges) in enumerate(zip(item['gap_logits'], item['gap_edges'])):
            edge_array = edges.detach().cpu().numpy()
            labels, why = label_edges(skels[b], masks[b], edge_array, p2_logits.shape[-2:], cfg)
            labels_all.extend(labels.tolist())
            for reason in why:
                reasons[reason] = reasons.get(reason, 0) + 1
            # Bounded sparse audit at the actual training proposals, also valid during tile validation.
            if audit_coverage:
                a, c = proposal_gap_coverage(skels[b], p2_arrays[b], edge_array, labels, cfg)
                verified_gaps += a; detected_gaps += c
            y = torch.as_tensor(labels, device=logits.device)
            for label, destination in ((1, positive_losses), (0, negative_losses)):
                valid = y == label
                if valid.any():
                    destination.append(F.binary_cross_entropy_with_logits(logits[valid], y[valid].float(), reduction='none'))
            predicted = logits.detach() >= 0
            correct_positive += int(((y == 1) & predicted).sum())
            predicted_positive += int(((y >= 0) & predicted).sum())
            true_positive += int((y == 1).sum())
        # Each class averages over its valid edges; one-class batches still train.
        terms = [torch.cat(losses).mean() for losses in (positive_losses, negative_losses) if losses]
        gap_loss = torch.stack(terms).mean() if terms else p2_logits.sum() * 0
        stats = {'p2_loss': p2_loss.detach(), 'gap_loss': gap_loss.detach(),
                 'gap_candidates': len(labels_all), 'gap_positive': labels_all.count(1),
                 'gap_negative': labels_all.count(0), 'gap_ignored': labels_all.count(-1),
                 'gap_tp': correct_positive, 'gap_predicted_positive': predicted_positive,
                 'gap_true_positive': true_positive, 'verified_p2_gaps': verified_gaps,
                 'candidate_covered_p2_gaps': detected_gaps,
                 'coverage_audited_images': len(skels) if audit_coverage else 0,
                 'gap_residual_abs_mean': item['gap_residual_abs_mean'],
                 'gap_writeback_scale': item['gap_writeback_scale'], 'gap_label_reasons': reasons}
        return cfg.p2_weight*p2_loss + cfg.gap_weight*gap_loss, stats
    return zero, {}
