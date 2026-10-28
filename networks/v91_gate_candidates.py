"""Training-only balanced supervision for executable positive transport actions.

GT labels and matching are never used by the model forward. Component size is
an action-support area proxy, not a candidate path length or GT connection.
"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F


def distance_to_true(binary):
    if not binary.any():
        return np.full(binary.shape, 1e6, dtype=np.float32)
    return cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, 3)


def candidate_component_areas(candidate):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    return stats[labels, cv2.CC_STAT_AREA]


def added_error_proxies(p0, p1, background, threshold, output_threshold):
    """Train-only local component proxy; not proof of false GT connectivity."""
    old = p0 >= threshold
    added = (p1 >= output_threshold) & ~old & background
    _, old_labels = cv2.connectedComponents(old.astype(np.uint8), connectivity=8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(added.astype(np.uint8), connectivity=8)
    bridge = np.zeros_like(old); isolated = np.zeros_like(old)
    for index in range(1, count):
        x, y, width, height = (int(x) for x in stats[index, :4])
        x0, x1 = max(0, x-2), min(old.shape[1], x+width+2)
        y0, y1 = max(0, y-2), min(old.shape[0], y+height+2)
        component = labels[y0:y1, x0:x1] == index
        neighborhood = cv2.dilate(component.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sources = np.unique(old_labels[y0:y1, x0:x1][neighborhood]); sources = sources[sources > 0]
        if len(sources) == 0:
            isolated[y0:y1, x0:x1] |= component
        elif len(sources) >= 2:
            bridge[y0:y1, x0:x1] |= component
    return bridge, isolated


def missed_segment_counts(skeleton, p0, covered_before, covered_after, threshold):
    """Measure GT-missing skeleton components hit by >=1 candidate, not path recall."""
    by_length = {key: {'missing_segments': 0, 'covered_before': 0, 'covered_after': 0}
                 for key in ('3-7', '8-15', '16-31', '32+')}
    for b in range(skeleton.shape[0]):
        road = p0[b, 0] >= threshold
        missing = (skeleton[b, 0] > .5) & ~(cv2.dilate(road.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
        for i in range(1, count):
            length = int(stats[i, cv2.CC_STAT_AREA])
            if length < 3:
                continue
            key = '3-7' if length < 8 else '8-15' if length < 16 else '16-31' if length < 32 else '32+'
            x, y, width, height = (int(x) for x in stats[i, :4])
            seg = labels[y:y+height, x:x+width] == i
            by_length[key]['missing_segments'] += 1
            by_length[key]['covered_before'] += int((seg & covered_before[b, y:y+height, x:x+width]).any())
            by_length[key]['covered_after'] += int((seg & covered_after[b, y:y+height, x:x+width]).any())
    return by_length


def gate_candidates(result, mask, skeleton, *, threshold=.45, min_action=1e-5,
                    near_gt_work_pixels=8., max_pairs_per_stratum=256, max_pairs_per_step=256, seed=1234,
                    include_samples=False, output_threshold=.68):
    """Match actual positive actions in identical V/action/distance/area strata.

    Each step/image is matched separately. Inputs are detached for sample
    selection; only the returned BCE on G retains gradients. Distances use the
    predicted P0 road for both classes. Negatives are GT background close to GT
    roads or in high-Q regions. Ambiguous mixed GT pixels are excluded.
    """
    if (not 0 < threshold < 1 or not 0 < output_threshold < 1 or not 0 < min_action < .001 or
            max_pairs_per_stratum < 1 or max_pairs_per_step < 1):
        raise ValueError('Invalid candidate configuration')
    gate = result['g_step']
    if gate.ndim != 5 or gate.shape[1:3] != (3, 1):
        raise ValueError('Expected 3 gate maps shaped Bx3x1xHxW')
    bsz, _, _, h, w = gate.shape
    gt_fraction = F.interpolate(mask.float(), size=(h, w), mode='area').detach().cpu().numpy()[:, 0]
    p0 = result['u0'].detach().cpu().numpy()[:, 0]
    p1 = F.interpolate(result['p1'].detach(), size=(h, w), mode='area').cpu().numpy()[:, 0]
    q = result['q'].detach().cpu().numpy()[:, 0]
    availability = result['source_available'].detach().cpu().numpy()[:, :, 0]
    # Select transport opportunities before G is applied. If candidate selection
    # used the realized update (which already includes G), a low G would hide
    # exactly the useful actions that L_gate needs to learn to accept.
    rate = result['transport_rate'].detach().cpu().numpy()[:, :, 0]
    potential_action = np.stack([
        rate[:, k] * (result['m'][:, k].detach().cpu().numpy()[:, 0] -
                      result[f'u{k}'].detach().cpu().numpy()[:, 0])
        for k in range(3)], axis=1)
    realized_action = result['transport_delta'].detach().cpu().numpy()[:, :, 0]
    gate_values = gate.detach().cpu().numpy()[:, :, 0]
    if not all(np.isfinite(value).all() for value in
               (gt_fraction, p0, p1, q, availability, rate, potential_action,
                realized_action, gate_values)):
        raise ValueError('Nonfinite gate candidate input')
    candidate = ((potential_action > min_action) & (availability > 0))
    random = np.random.default_rng(seed)
    selected = np.zeros((bsz, 3, h, w), dtype=bool)
    labels = np.zeros((bsz, 3, h, w), dtype=np.float32)
    positive_before = np.zeros_like(selected)
    negatives_before = np.zeros_like(selected)
    detail, differences, samples = [], [], []
    gt_pos_bins = np.array([0., .02, .1, .3, .6, 1.01])
    action_bins = np.array([min_action, .001, .01, .05, .2, 1.01])
    for b in range(bsz):
        road_gt, background_gt = gt_fraction[b] >= .75, gt_fraction[b] <= .05
        dist_gt = distance_to_true(road_gt)
        dist_prediction = distance_to_true(p0[b] >= threshold)
        bridge, isolated = added_error_proxies(p0[b], p1[b], background_gt, threshold, output_threshold)
        for k in range(3):
            pos = candidate[b, k] & road_gt & (p0[b] < threshold)
            neg = candidate[b, k] & background_gt & ((dist_gt <= near_gt_work_pixels) |
                                                     (q[b] >= .5) | bridge | isolated)
            positive_before[b, k] = pos; negatives_before[b, k] = neg
            areas = candidate_component_areas(candidate[b, k])
            vbin = np.digitize(availability[b, k], gt_pos_bins[1:-1])
            abin = np.digitize(potential_action[b, k], action_bins[1:-1])
            dbin = np.digitize(dist_prediction, [1., 2., 4., 8., 16.])
            cbin = np.digitize(areas, [4, 8, 16, 32, 64])
            keys = (((vbin*8 + abin)*8 + dbin)*8 + cbin)
            count_pairs = 0
            strata = np.intersect1d(np.unique(keys[pos]), np.unique(keys[neg]))
            for key in random.permutation(strata):
                p_indices = np.flatnonzero(pos.ravel() & (keys.ravel() == key))
                n_indices = np.flatnonzero(neg.ravel() & (keys.ravel() == key))
                amount = min(len(p_indices), len(n_indices), max_pairs_per_stratum,
                             max_pairs_per_step-count_pairs)
                if not amount:
                    break
                p_indices = random.choice(p_indices, amount, replace=False)
                priority = np.where((bridge | isolated).ravel()[n_indices], 4., 1.)
                n_indices = random.choice(n_indices, amount, replace=False, p=priority/priority.sum())
                selected[b, k].flat[p_indices] = True; labels[b, k].flat[p_indices] = 1.
                selected[b, k].flat[n_indices] = True
                for pair_number, (index_p, index_n) in enumerate(zip(p_indices, n_indices)):
                    differences.append({'v': abs(float(availability[b, k].flat[index_p]-availability[b, k].flat[index_n])),
                        'potential_action': abs(float(potential_action[b, k].flat[index_p]-
                                                       potential_action[b, k].flat[index_n])),
                        'realized_action': abs(float(realized_action[b, k].flat[index_p]-
                                                     realized_action[b, k].flat[index_n])),
                        'prediction_distance_work': abs(float(dist_prediction.flat[index_p]-dist_prediction.flat[index_n])),
                        'component_area_work': abs(int(areas.flat[index_p])-int(areas.flat[index_n]))})
                    if include_samples:
                        for label, flat_index in ((1, index_p), (0, index_n)):
                            yy, xx = divmod(int(flat_index), w)
                            samples.append({'batch_index': b, 'step': k+1, 'stratum': int(key),
                                'pair_number': pair_number, 'label': label, 'y_work': yy, 'x_work': xx,
                                'availability': float(availability[b, k, yy, xx]),
                                'potential_transport_effect': float(potential_action[b, k, yy, xx]),
                                'realized_transport_effect': float(realized_action[b, k, yy, xx]),
                                'prediction_distance_work': float(dist_prediction[yy, xx]),
                                'component_area_work': int(areas[yy, xx]),
                                'near_gt': bool(dist_gt[yy, xx] <= near_gt_work_pixels),
                                'high_q': bool(q[b, yy, xx] >= .5),
                                'false_bridge_proxy': bool(bridge[yy, xx]),
                                'isolated_fp_proxy': bool(isolated[yy, xx]),
                                'g': float(gate_values[b, k, yy, xx])})
                count_pairs += amount
            detail.append({'batch_index': b, 'step': k+1, 'executable_candidates': int(candidate[b, k].sum()),
                'positive_before_matching': int(pos.sum()), 'negative_before_matching': int(neg.sum()),
                'matched_positive': count_pairs, 'matched_negative': count_pairs,
                'negative_near_gt_before': int((neg & (dist_gt <= near_gt_work_pixels)).sum()),
                'negative_high_q_before': int((neg & (q[b] >= .5)).sum()),
                'negative_bridge_proxy_before': int((neg & bridge).sum()),
                'negative_isolated_proxy_before': int((neg & isolated).sum()),
                'negative_bridge_proxy_matched': int((selected[b, k] & (labels[b, k] == 0) & bridge).sum()),
                'negative_isolated_proxy_matched': int((selected[b, k] & (labels[b, k] == 0) & isolated).sum())})
    sel = torch.as_tensor(selected[:, :, None], device=gate.device)
    target = torch.as_tensor(labels[:, :, None], device=gate.device)
    if bool(sel.any()):
        loss = F.binary_cross_entropy(gate[sel].clamp(1e-6, 1.-1e-6), target[sel])
    else:
        loss = gate.sum() * 0.
    full_size = mask.shape[-2:]
    pre = F.interpolate(torch.as_tensor(positive_before.any(1)[:, None].astype(np.float32)),
                        size=full_size, mode='nearest').numpy()[:, 0] > .5
    post = F.interpolate(torch.as_tensor((selected & (labels > 0)).any(1)[:, None].astype(np.float32)),
                         size=full_size, mode='nearest').numpy()[:, 0] > .5
    counts = missed_segment_counts(skeleton.detach().cpu().numpy(), result['p0'].detach().cpu().numpy(),
                                   pre, post, threshold)
    quality = {key: float(np.mean([item[key] for item in differences])) if differences else None
        for key in ('v', 'potential_action', 'realized_action',
                    'prediction_distance_work', 'component_area_work')}
    pos_scores = gate_values[selected & (labels > 0)]
    neg_scores = gate_values[selected & (labels == 0)]
    report = {'details': detail, 'executable_candidates': int(candidate.sum()),
        'positive_before_matching': int(positive_before.sum()), 'negative_before_matching': int(negatives_before.sum()),
        'matched_positive': int((selected & (labels > 0)).sum()),
        'matched_negative': int((selected & (labels == 0)).sum()),
        'g_positive_histogram': np.histogram(pos_scores, bins=100, range=(0., 1.))[0].tolist(),
        'g_negative_histogram': np.histogram(neg_scores, bins=100, range=(0., 1.))[0].tolist(),
        'mean_absolute_pair_differences': quality, 'gt_missing_segment_coverage': counts,
        'warning': 'GT skeleton component hit is an approximate local coverage proxy, not positive path-pair recall. '
                   'Action-component area in the work grid is a length proxy, not a measured connection length. '
                   'Isolated and bridge labels are prediction-component proxies, not certified topology. '
                   'Parallel-road and visually similar object subtypes are not automatically certified. '
                   'Caps apply per image and step, after stratified matching. Positive and negative refer to local '
                   'transport corrections, not guaranteed final pixel/whole-path outcomes.'}
    if include_samples:
        report['samples'] = samples
    return loss, report
