"""Validation-only causal audit for the v8 semantic transport gate.

This script never opens the test split, trains parameters, or writes checkpoints.
It emits threshold-score, gate-region, feature-probe, gradient, intervention,
and per-step dynamics evidence for the exact v7 -> v8 checkpoint pair.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.compare_v6_v7_matched import (  # noqa: E402
    bridge_mask, write_csv,
)
from evaluation.road_comparison.road_metrics import prepare_target  # noqa: E402
from mass_protocol import blend_weight, sliding_positions  # noqa: E402
from networks.reliable_directional_repair import DIRECTIONS, dynamic_losses  # noqa: E402
from run_q3_r3_mass import PROTOCOL, full_dataset  # noqa: E402
from run_reliable_directional_mass import evaluate_thresholds  # noqa: E402
from run_reliable_v8_mass import load_experiment  # noqa: E402


def _summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: None for key in ('n', 'mean', 'p10', 'p50', 'p90', 'p99')}
    return {'n': int(values.size), 'mean': float(values.mean()),
            'p10': float(np.percentile(values, 10)),
            'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)),
            'p99': float(np.percentile(values, 99))}


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _probe_metrics(y, score):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64) + 1
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = sums[inverse] / counts[inverse]
    positives, negatives = int(y.sum()), int((1-y).sum())
    auc = ((ranks[y == 1].sum() - positives * (positives + 1) / 2) /
           max(positives * negatives, 1)) if positives and negatives else 0.
    order = np.argsort(-score, kind='mergesort')
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positives, 1)
    ap = float((precision * sorted_y).sum() / max(positives, 1))
    fpr_at = {}
    for target in (.8, .9):
        eligible = np.flatnonzero(recall >= target)
        fpr_at[str(target)] = float(fp[eligible[0]] / max(negatives, 1)) if eligible.size else 1.
    return {'roc_auc': float(auc), 'pr_auc_average_precision': ap,
            'fpr_at_recall_0.8': fpr_at['0.8'],
            'fpr_at_recall_0.9': fpr_at['0.9'],
            'positives': positives, 'negatives': negatives}


def _fit_logistic(train_x, train_y, test_x, steps=250, lr=.25):
    mean = train_x.mean(0, keepdims=True)
    scale = train_x.std(0, keepdims=True).clip(1e-4)
    x = np.clip((train_x - mean) / scale, -10, 10)
    xt = np.clip((test_x - mean) / scale, -10, 10)
    y = train_y.astype(np.float64)[:, None]
    weight = np.zeros((x.shape[1], 1), dtype=np.float64)
    bias = np.zeros((1,), dtype=np.float64)
    for _ in range(steps):
        logits = np.clip(x @ weight + bias, -30, 30)
        pred = 1. / (1. + np.exp(-logits))
        weight -= lr * (x.T @ (pred-y) / len(y) + 1e-4 * weight)
        bias -= lr * (pred-y).mean(0)
    return (xt @ weight + bias).ravel()


def _grouped_probe(train_rows, val_rows, output):
    if not train_rows or not val_rows:
        return {'note': 'Train-fit/validation-eval probe needs matched candidates in both splits.'}
    x_groups = ('structure', 'structure_plus_H', 'structure_plus_H3',
                'structure_plus_H_H3', 'plus_bilateral_relation')
    train_y = np.asarray([r['label'] for r in train_rows], dtype=np.int8)
    val_y = np.asarray([r['label'] for r in val_rows], dtype=np.int8)
    report, csv_rows = {}, []
    for model_name in x_groups:
        train_x = np.stack([np.asarray(row[model_name]).ravel()
                            for row in train_rows]).astype(np.float32)
        val_x = np.stack([np.asarray(row[model_name]).ravel()
                          for row in val_rows]).astype(np.float32)
        if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            report[model_name] = {'features': model_name,
                                  'note': 'Both classes are required in train and val candidates.'}
            continue
        scores = _fit_logistic(train_x, train_y, val_x)
        report[model_name] = {
            'features': model_name, 'candidate_unit': 'GT missing-road connection pair',
            'fit_split': 'train', 'evaluation_split': 'val',
            'train_images': len({row['image'] for row in train_rows}),
            'val_images': len({row['image'] for row in val_rows}),
            'train_candidates_per_class': int(min(train_y.sum(), (1-train_y).sum())),
            **_probe_metrics(val_y, scores),
        }
        for candidate_index, (row, score) in enumerate(zip(val_rows, scores)):
            csv_rows.append({'feature_set': model_name, 'image': row['image'],
                             'split': 'val', 'candidate_id': candidate_index,
                             'candidate_unit': 'GT missing-road connection pair',
                             'label': int(row['label']),
                             'pair_span_pixels': row['length'],
                             'gap_length_pixels': row['gap_length'],
                             'direction_index': row['direction'],
                             'score': float(score)})
    write_csv(output / 'feature_probe_oof_scores.csv', csv_rows)
    return report


def _sample_feature(feature, y, x):
    feature = np.asarray(feature)
    if feature.ndim == 2:
        return np.asarray([feature[y, x]], dtype=np.float32)
    return feature[:, y, x].astype(np.float32, copy=False)


def _pair_feature_rows(image_id, maps, gt, p0_threshold, max_per_class=256):
    """Build path-level positive gaps and hard-background connection candidates."""
    from evaluation.road_comparison.road_metrics import skeletonize

    p0_full = np.asarray(maps['p0_full'])
    p0_work, q_work, reliable = maps['p0'], maps['q'], maps['r']
    work_h, work_w = p0_work.shape
    gt_native = gt.astype(bool)
    gt_work = cv2.resize(gt_native.astype(np.float32), (work_w, work_h),
                         interpolation=cv2.INTER_AREA)
    p0_binary = p0_full >= p0_threshold
    p0_near = cv2.dilate(p0_binary.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gt_skeleton = skeletonize(gt_native)
    missing = gt_skeleton & ~p0_near
    distance_to_p0 = cv2.distanceTransform(
        (~(p0_work >= p0_threshold)).astype(np.uint8), cv2.DIST_L2, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8)
    source_mask = p0_binary & gt_native
    positives = []

    def work_xy(y_native, x_native):
        return (int(np.clip(round(y_native * work_h / gt_native.shape[0]), 0, work_h - 1)),
                int(np.clip(round(x_native * work_w / gt_native.shape[1]), 0, work_w - 1)))

    def flank(point, axis, sign):
        py, px = point
        candidates = []
        for distance in range(3, 33, 2):
            ty = py + sign * axis[0] * distance
            tx = px + sign * axis[1] * distance
            radius = 6
            y0, y1 = max(0, int(ty) - radius), min(gt_native.shape[0], int(ty) + radius + 1)
            x0, x1 = max(0, int(tx) - radius), min(gt_native.shape[1], int(tx) + radius + 1)
            local = source_mask[y0:y1, x0:x1]
            yy, xx = np.where(local)
            if yy.size:
                yy, xx = yy + y0, xx + x0
                distance2 = (yy - ty) ** 2 + (xx - tx) ** 2
                index = int(distance2.argmin())
                candidates.append((float(distance2[index]), int(yy[index]), int(xx[index])))
                break
        return (candidates[0][1], candidates[0][2]) if candidates else None

    positive_quality = {'components': 0, 'with_two_predicted_flanks': 0,
                        'without_two_flanks': 0}
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < 3:
            continue
        yy, xx = np.where(labels == component)
        coords = np.stack((yy, xx), axis=1).astype(np.float64)
        centered = coords - coords.mean(0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        axis = vectors[:, -1]
        if axis[1] < 0:
            axis = -axis
        projection = centered @ axis
        left_point = coords[int(projection.argmin())]
        right_point = coords[int(projection.argmax())]
        left_flank = flank(left_point, axis, -1.)
        right_flank = flank(right_point, axis, 1.)
        positive_quality['components'] += 1
        if left_flank is None or right_flank is None:
            positive_quality['without_two_flanks'] += 1
            continue
        positive_quality['with_two_predicted_flanks'] += 1
        center_native = ((left_point + right_point) * .5)
        cy, cx = work_xy(*center_native)
        ly, lx = work_xy(*left_flank)
        ry, rx = work_xy(*right_flank)
        vector = np.asarray([right_flank[0] - left_flank[0],
                             right_flank[1] - left_flank[1]], dtype=np.float64)
        vector /= max(np.linalg.norm(vector), 1e-8)
        direction = max(range(len(DIRECTIONS)), key=lambda i:
                        (-vector[0] * DIRECTIONS[i][0] -
                         vector[1] * DIRECTIONS[i][1]) / np.hypot(*DIRECTIONS[i]))
        length = float(np.linalg.norm(np.asarray(right_flank) - np.asarray(left_flank)))
        gap_length = float(projection.max() - projection.min() + 1.)
        positives.append({'label': 1, 'image': str(image_id), 'center': (cy, cx),
                          'left': (ly, lx), 'right': (ry, rx),
                          'direction': direction, 'length': length,
                          'gap_length': gap_length,
                          'p0': float(p0_work[cy, cx]), 'q': float(q_work[cy, cx]),
                          'distance': float(distance_to_p0[cy, cx])})

    # Negatives are plausible two-sided proposals in Q3-high, GT-background areas.
    q_high = q_work > .5
    hard_centers = np.argwhere(q_high & (gt_work < .05) & (p0_work < .8))
    rng = np.random.default_rng(811 + int(image_id))
    if len(hard_centers) > max_per_class * 12:
        hard_centers = hard_centers[rng.choice(len(hard_centers), max_per_class * 12,
                                               replace=False)]
    negatives = []
    offsets = tuple((int(dy), int(dx)) for dy, dx in DIRECTIONS)
    for cy, cx in hard_centers:
        best = None
        for direction, (dy, dx) in enumerate(offsets):
            for radius in (2, 4, 8, 16):
                y1, x1 = int(cy + dy * radius), int(cx + dx * radius)
                y2, x2 = int(cy - dy * radius), int(cx - dx * radius)
                if not (0 <= y1 < work_h and 0 <= x1 < work_w and
                        0 <= y2 < work_h and 0 <= x2 < work_w):
                    continue
                support = min(float(reliable[y1, x1]), float(reliable[y2, x2]))
                if best is None or support > best[0]:
                    best = (support, direction, radius, (y1, x1), (y2, x2))
        if best is None or best[0] < .15:
            continue
        _, direction, radius, left, right = best
        negatives.append({'label': 0, 'image': str(image_id),
                          'center': (int(cy), int(cx)), 'left': left, 'right': right,
                          'direction': direction, 'length': float(2 * radius *
                                                                  np.hypot(dy, dx) *
                                                                  gt_native.shape[0] / work_h),
                          'gap_length': float(2 * radius * np.hypot(dy, dx) *
                                              gt_native.shape[0] / work_h),
                          'p0': float(p0_work[cy, cx]), 'q': float(q_work[cy, cx]),
                          'distance': float(distance_to_p0[cy, cx])})

    def strata(candidate):
        return (int(np.digitize(candidate['p0'], [.05, .15, .3, .45, .6, .8])),
                int(np.digitize(candidate['q'], [.6, .75, .9])),
                int(np.digitize(candidate['distance'], [2, 5, 10, 20])),
                int(np.digitize(candidate['gap_length'], [7, 15, 31])),
                candidate['direction'] // 2)

    pos_by, neg_by = {}, {}
    for candidate in positives:
        pos_by.setdefault(strata(candidate), []).append(candidate)
    for candidate in negatives:
        neg_by.setdefault(strata(candidate), []).append(candidate)
    matched_pos, matched_neg = [], []
    for key in sorted(set(pos_by) & set(neg_by)):
        pos, neg = pos_by[key], neg_by[key]
        n = min(len(pos), len(neg), max_per_class // max(len(set(pos_by) & set(neg_by)), 1))
        if n:
            matched_pos.extend(pos[int(i)] for i in rng.choice(len(pos), n, replace=False))
            matched_neg.extend(neg[int(i)] for i in rng.choice(len(neg), n, replace=False))

    def vector_at(key, point):
        return _sample_feature(maps[key], *point)

    rows = []
    matched = matched_pos + matched_neg
    for candidate in matched:
        cy, cx = candidate['center']
        left, right = candidate['left'], candidate['right']
        direction = candidate['direction']
        opposite = direction ^ 1
        center_structure = np.concatenate([
            _sample_feature(maps[key], cy, cx) for key in
            ('p0', 'q', 's2', 'c2', 'd2', 'r', 'w')])
        h_center, h_left, h_right = (vector_at('highres', point)
                                     for point in ((cy, cx), left, right))
        h3_center, h3_left, h3_right = (vector_at('decoder', point)
                                        for point in ((cy, cx), left, right))
        direction_relation = maps.get('direction_relation')
        relation_width = maps['relation'].shape[0]
        rel = (_sample_feature(direction_relation[
            direction * relation_width:(direction + 1) * relation_width], cy, cx)
               if direction_relation is not None else np.zeros(1, np.float32))
        rel_opp = (_sample_feature(direction_relation[
            opposite * relation_width:(opposite + 1) * relation_width], cy, cx)
                   if direction_relation is not None else np.zeros(1, np.float32))
        direction_pair_raw = maps.get('direction_pair_raw')
        raw_rel = (_sample_feature(direction_pair_raw[
            direction * relation_width:(direction + 1) * relation_width], cy, cx)
                   if direction_pair_raw is not None else np.zeros(1, np.float32))
        raw_rel_opp = (_sample_feature(direction_pair_raw[
            opposite * relation_width:(opposite + 1) * relation_width], cy, cx)
                       if direction_pair_raw is not None else np.zeros(1, np.float32))
        relation_geom = np.asarray([candidate['length'] / 64.,
                                    DIRECTIONS[direction][0],
                                    DIRECTIONS[direction][1]], dtype=np.float32)
        bilateral = np.concatenate((
            h_left, h_right, np.abs(h_left - h_right), h_left * h_right,
            h3_left, h3_right, np.abs(h3_left - h3_right), h3_left * h3_right,
            raw_rel, raw_rel_opp, rel, rel_opp,
            _sample_feature(maps['w'], cy, cx),
            np.asarray([maps['coverage'][cy, cx]], np.float32), relation_geom))
        rows.append({'image': candidate['image'], 'label': candidate['label'],
                     'structure': center_structure,
                     'structure_plus_H': np.concatenate((center_structure, h_center)),
                     'structure_plus_H3': np.concatenate((center_structure, h3_center)),
                     'structure_plus_H_H3': np.concatenate((center_structure, h_center, h3_center)),
                     'plus_bilateral_relation': np.concatenate((center_structure, bilateral)),
                     'length': candidate['length'], 'gap_length': candidate['gap_length'],
                     'direction': direction})
    quality = {
        'image': str(image_id), 'positive_gap_components': positive_quality['components'],
        'positive_with_two_predicted_flanks': positive_quality['with_two_predicted_flanks'],
        'positive_without_two_flanks': positive_quality['without_two_flanks'],
        'eligible_negative_background_candidates': len(hard_centers),
        'negative_with_two_reliable_sources': len(negatives),
        'matched_per_class': len(matched_pos),
        'matched_pairs': len(matched_pos),
        'matched_strata': len(set(strata(c) for c in matched_pos)),
    }
    return rows, quality


@torch.no_grad()
def _stitch_work_maps(model, image, device):
    positions = sliding_positions(1500, 512, 256)
    full_weight = blend_weight(512, device)
    names = ('p0', 'q', 'r', 'a', 'b', 'w', 's2', 'c2', 'd2', 'highres', 'decoder',
             'highres_projected', 'decoder_projected', 'relation', 'coverage',
             'direction_relation', 'direction_pair_raw',
             'g7', 'g8', 'delta_g', 'p1', 'p7',
             'proposal_raw_plus', 'proposal_raw_minus',
             'alpha_1', 'alpha_2', 'alpha_3', 'beta_1', 'beta_2', 'beta_3',
             'gamma_1', 'gamma_2', 'gamma_3', 'transport_delta_1',
             'transport_delta_2', 'transport_delta_3', 'proposal_delta_1',
             'proposal_delta_2', 'proposal_delta_3', 'anchor_delta_1',
             'anchor_delta_2', 'anchor_delta_3', 'delta_z_plus', 'delta_z_minus')
    canvases = {}
    mass = weight = full_mass = None
    full_canvases = {}
    work_size = full_work_size = None
    for top in positions:
        for left in positions:
            tile = image[:, :, top:top+512, left:left+512]
            frozen = model.frozen(tile, capture_diagnostics=True)
            out = model(tile, capture_diagnostics=True, frozen_override=frozen)
            saved_gate = model.gate_relation
            model.gate_relation = None
            try:
                v7_out = model(tile, frozen_override=frozen)
            finally:
                model.gate_relation = saved_gate
            relation = out['gate_relation_diagnostics']
            if work_size is None:
                work_size = int(out['r'].shape[-1])
                scale = work_size / 512.
                full_work_size = round(1500 * scale)
                weight = blend_weight(work_size, device)
                mass = torch.zeros((1, 1, full_work_size, full_work_size), device=device)
                full_mass = torch.zeros((1, 1, 1500, 1500), device=device)

            def align_work(value):
                return (value if value.shape[-2:] == (work_size, work_size) else
                        F.interpolate(value, size=(work_size, work_size),
                                      mode='bilinear', align_corners=False))

            values = {
                'p0': F.adaptive_avg_pool2d(out['p0'], (work_size, work_size)),
                'q': align_work(out['q']), 'r': out['r'], 'a': out['a'],
                'b': out['b'], 'w': out['w'],
                's2': align_work(out['stage2_s'].sigmoid()),
                'c2': align_work(out['stage2_c'].sigmoid()),
                'd2': (align_work(out['stage2_d']) if out['stage2_d'] is not None else
                       out['r'].new_zeros((1, 1, work_size, work_size))),
                'highres': relation['highres_projected'],
                'decoder': relation['decoder_projected'],
                'highres_projected': relation['highres_projected'],
                'decoder_projected': relation['decoder_projected'],
                'relation': relation['relation_embedding'],
                'direction_relation': relation['per_direction_relation'].flatten(1, 2),
                'direction_pair_raw': relation['per_direction_pair_raw'].flatten(1, 2),
                'coverage': out['gate_source_coverage'],
                'g7': out['g_v7'], 'g8': out['g_transport'],
                'delta_g': out['delta_g_logit'],
                'p1': F.adaptive_avg_pool2d(out['p1'], (work_size, work_size)),
                'p7': F.adaptive_avg_pool2d(v7_out['p1'], (work_size, work_size)),
                'proposal_raw_plus': out['proposal_raw'][:, :1],
                'proposal_raw_minus': out['proposal_raw'][:, 1:2],
            }
            values['highres'] = out['highres_h']
            values['decoder'] = out['stage3_h']
            for key in ('alpha', 'beta', 'gamma', 'transport_delta',
                        'proposal_delta', 'anchor_delta'):
                values.update({f'{key}_{step+1}': out[key][:, step]
                               for step in range(3)})
            values['delta_z_plus'] = out['delta_z_plus']
            values['delta_z_minus'] = out['delta_z_minus']
            full_values = {
                'p0_full': out['z0'], 'p1_full': out['z1'], 'p7_full': v7_out['z1'],
                'q_full': F.interpolate(out['q'], size=(512, 512), mode='bilinear',
                                         align_corners=False),
            }
            full_region = (..., slice(top, top+512), slice(left, left+512))
            full_mass[full_region] += full_weight
            for name, value in full_values.items():
                if name not in full_canvases:
                    full_canvases[name] = torch.zeros((1, value.shape[1], 1500, 1500),
                                                       device=device)
                full_canvases[name][full_region] += value.float() * full_weight
            y0, x0 = round(top * scale), round(left * scale)
            region = (..., slice(y0, y0+work_size), slice(x0, x0+work_size))
            mass[region] += weight
            for name in names:
                value = values[name].float()
                if name not in canvases:
                    canvases[name] = torch.zeros((1, value.shape[1], full_work_size,
                                                  full_work_size),
                                                  device=device)
                canvases[name][region] += value * weight
    if (mass <= 0).any():
        raise RuntimeError('Incomplete validation work-grid stitch')
    if (full_mass <= 0).any():
        raise RuntimeError('Incomplete validation full-resolution stitch')
    output = {name: (value / mass).cpu().numpy()[0] for name, value in canvases.items()}
    for name, value in full_canvases.items():
        stitched = value / full_mass
        if name in ('p0_full', 'p1_full', 'p7_full'):
            stitched = stitched.sigmoid()
        output[name] = stitched.cpu().numpy()[0]
    return output


def _region_rows(record, threshold):
    gt_native = record['gt']
    work_size = (record['p0'].shape[1], record['p0'].shape[0])
    gt = gt_native
    prepared = prepare_target(gt, PROTOCOL)
    p0 = record['p0_full'] >= threshold
    p7 = record['p7_full'] >= threshold
    p8 = record['p1_full'] >= threshold
    q = record['q_full']
    skeleton = prepared['skeleton']
    missing = skeleton & ~(cv2.dilate(p0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    groups = {
        'recovered_gt': missing & p8 & ~p7,
        'unrecovered_gt': missing & ~p8 & ~p7,
        'hard_background': ~gt & (q > .5),
        'new_false_bridge': bridge_mask(p0, p8, gt) & ~bridge_mask(p0, p7, gt),
        'new_tp': p8 & ~p7 & gt,
        'new_fp': p8 & ~p7 & ~gt,
    }
    rows = []
    for name, mask in groups.items():
        if not mask.any():
            continue
        mask_work = cv2.resize(mask.astype(np.uint8), work_size,
                               interpolation=cv2.INTER_AREA) > .25
        row = {'image': record['case'], 'region': name,
               'pixels_native': int(mask.sum()), 'pixels_workgrid': int(mask_work.sum())}
        for key in ('g7', 'g8', 'delta_g', 'r', 'a', 'delta_z_plus', 'delta_z_minus',
                    'proposal_raw_plus', 'proposal_raw_minus',
                    *(f'{name}_{step}' for name in
                      ('alpha', 'beta', 'gamma', 'transport_delta',
                       'proposal_delta', 'anchor_delta') for step in (1, 2, 3))):
            if key not in record:
                continue
            val = record[key]
            if val.ndim == 3:
                val = val.mean(0)
            arr = val[mask_work]
            row[f'{key}_mean'] = float(arr.mean()) if arr.size else None
            row[f'{key}_p90'] = float(np.percentile(arr, 90)) if arr.size else None
        b_delta = record['b'][mask_work] - record['p0'][mask_work]
        row['b_minus_p0_mean'] = float(b_delta.mean()) if b_delta.size else None
        for direction in range(record['w'].shape[0]):
            row[f'w_direction_{direction}_mean'] = float(record['w'][direction][mask_work].mean())
        w_values = np.clip(record['w'][:, mask_work], 1e-8, 1.)
        w_norm = w_values / np.maximum(w_values.sum(0, keepdims=True), 1e-8)
        entropy = -(w_norm * np.log(w_norm)).sum(0)
        row['w_entropy_mean'] = float(entropy.mean())
        row['w_max_direction_mean'] = float(w_norm.max(0).mean())
        if 'g7' in record:
            row['g7_saturated_low'] = float((record['g7'][mask_work] < .001).mean())
            row['g7_saturated_high'] = float((record['g7'][mask_work] > .999).mean())
            row['g8_saturated_low'] = float((record['g8'][mask_work] < .001).mean())
            row['g8_saturated_high'] = float((record['g8'][mask_work] > .999).mean())
        for key, limits in (('delta_z_plus', (7.9, 7.99, 7.999)),
                            ('delta_z_minus', (3.9, 3.99, 3.999))):
            if key in record:
                values = record[key][mask_work]
                for limit_value in limits:
                    row[f'{key}_ge_{limit_value}'] = float((values >= limit_value).mean())
        rows.append(row)
    return rows


def _gradient_audit(model, sample, device, loss_weights, top, left, gap_pixels):
    # Keep frozen BN/statistics paths in eval mode; only the new relation module
    # needs gradients, and this audit must not mutate running buffers.
    model.eval()
    model.gate_relation.train()
    model.set_accept_floor(0.)
    image = sample['image'][None, :, top:top+512, left:left+512].to(device)
    mask = sample['mask'][None, :, top:top+512, left:left+512].to(device)
    skeleton = sample['skeleton'][None, :, top:top+512, left:left+512].to(device)
    model.zero_grad(set_to_none=True)
    result = model(image, capture_diagnostics=True)
    loss, terms = dynamic_losses(
        result, mask, skeleton,
        lambda_bg=loss_weights['lambda_bg'],
        lambda_missing=loss_weights['lambda_missing'],
        lambda_cont=loss_weights['lambda_cont'])
    loss.backward()
    layer_norms = {}
    for name, module in model.gate_relation.named_children():
        grads = [p.grad.detach().float().norm().square() for p in module.parameters()
                 if p.grad is not None]
        layer_norms[name] = float(torch.stack(grads).sum().sqrt()) if grads else 0.
    parameter_norms = {
        name: (float(parameter.grad.detach().float().norm())
               if parameter.grad is not None else 0.)
        for name, parameter in model.gate_relation.named_parameters()
    }
    model.eval()
    return {'loss': float(loss.detach()), 'loss_terms': {k: float(v) for k, v in terms.items()
                                                          if isinstance(v, (float, int, torch.Tensor))
                                                          and (not isinstance(v, torch.Tensor) or v.numel() == 1)},
            'per_module_gradient_l2': layer_norms,
            'per_parameter_gradient_l2': parameter_norms,
            'gradient_tile': {'top': top, 'left': left,
                              'reference_gap_skeleton_pixels': gap_pixels}}


def _gradient_tile(gt, p0, threshold):
    skeleton = prepare_target(gt, PROTOCOL)['skeleton']
    road_near = cv2.dilate((p0 >= threshold).astype(np.uint8),
                           np.ones((5, 5), np.uint8)) > 0
    missing = skeleton & ~road_near
    positions = sliding_positions(1500, 512, 256)
    return max(((int(missing[top:top+512, left:left+512].sum()), top, left)
                for top in positions for left in positions))


@torch.no_grad()
def _measure_interventions(model, tile, intervention_names):
    frozen = model.frozen(tile, capture_diagnostics=True)
    baseline = model(tile, capture_diagnostics=True, frozen_override=frozen)
    gate, dynamics = {}, {}
    for name in intervention_names:
        varied = model(tile, capture_diagnostics=True, gate_intervention=name,
                       frozen_override=frozen)
        gate[name] = (varied['z1'], varied['g_transport'])
    for name, kwargs in (('transport_off', {'disable_transport': True}),
                         ('proposal_off', {'disable_proposal': True})):
        varied = model(tile, frozen_override=frozen, **kwargs)
        dynamics[name] = varied['z1']
    return baseline['z1'], baseline['g_transport'], gate, dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--v7_checkpoint', type=Path, required=True)
    parser.add_argument('--v8_checkpoint', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path,
                        help='Optional recovered copy of the exact frozen base checkpoint')
    parser.add_argument('--q3_checkpoint', type=Path,
                        help='Optional recovered copy of the exact frozen Q3/R3 checkpoint')
    parser.add_argument('--v7_selection', type=Path, required=True)
    parser.add_argument('--v8_selection', type=Path, required=True)
    parser.add_argument('--relation_mode', choices=('local', 'bilateral', 'directional', 'full'), default='full')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--probe_train_images', type=int, default=64,
                        help='Deterministically spaced train images used to fit feature probes')
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.probe_train_images < 0:
        parser.error('--probe_train_images must be zero (all train images) or positive')
    if not args.v7_checkpoint.is_file() or not args.v8_checkpoint.is_file():
        parser.error('Both v7 and v8 checkpoint paths must be files')
    v7_payload = torch.load(args.v7_checkpoint, map_location='cpu', weights_only=False)
    base_path = args.base_checkpoint or Path(v7_payload['base_checkpoint'])
    q3_path = args.q3_checkpoint or Path(v7_payload['q3_checkpoint'])
    missing_sources = [str(path) for path in (base_path, q3_path) if not path.is_file()]
    if missing_sources:
        parser.error('Frozen inference weights are required and were not found: ' +
                     ', '.join(missing_sources) +
                     '. Recover the exact files or pass their recovered paths with '
                     '--base_checkpoint/--q3_checkpoint; v7/v8 files do not embed them.')
    selection7 = json.loads(args.v7_selection.read_text(encoding='utf-8'))
    selection8 = json.loads(args.v8_selection.read_text(encoding='utf-8'))
    if selection7.get('selected_on') != 'val' or selection8.get('selected_on') != 'val':
        parser.error('Threshold selections must be validation selections')
    if selection7.get('dynamic_checkpoint') != str(args.v7_checkpoint.resolve()):
        parser.error('v7 selection does not belong to the supplied v7 checkpoint')
    if selection8.get('v7_checkpoint') != str(args.v7_checkpoint.resolve()):
        parser.error('v8 checkpoint selection does not reference this v7 checkpoint')
    if selection8.get('v8_checkpoint') != str(args.v8_checkpoint.resolve()):
        parser.error('v8 selection does not belong to the supplied v8 checkpoint')
    config = SimpleNamespace(v7_checkpoint=args.v7_checkpoint, v8_checkpoint=args.v8_checkpoint,
                             relation_mode=args.relation_mode, device=args.device,
                             base_checkpoint_override=args.base_checkpoint,
                             q3_checkpoint_override=args.q3_checkpoint)
    model, v7_payload = load_experiment(config)
    model.eval()
    dataset = full_dataset(args.root_path, 'val')
    if len(dataset) != 14 and args.max_images == 0:
        print(f'[NOTICE] val split has {len(dataset)} images (expected 14); using all.', flush=True)
    limit = min(len(dataset), args.max_images or len(dataset))
    args.output_dir.mkdir(parents=True, exist_ok=False)

    # Full threshold sweep is deliberately recomputed from validation predictions.
    thresholds = tuple(round(float(x), 2) for x in np.arange(.35, .751, .01))
    threshold_reports, _, _, _ = evaluate_thresholds(model, dataset, args.device, thresholds, limit)
    threshold_rows = []
    score_function = ('IoU + 0.15*clDice + 0.10*APLS_GT_to_Pred + '
                      '0.05*APLS_bidirectional - 0.001*short_pred_components - '
                      '0.01*background_bridge_proxy/images')
    for threshold, report in threshold_reports.items():
        m, diag = report['p1'], report['paired']
        threshold_rows.append({'threshold': threshold, 'selection_score': report['selection_score'],
                               'score_iou_term': m['iou'],
                               'score_cldice_term': .15 * m['cldice'],
                               'score_gt_apls_term': .1 * m['apls_gt_to_pred_approx'],
                               'score_bidir_apls_term': .05 * m['apls_bidirectional_approx'],
                               'score_short_component_penalty': -.001 * m['short_pred_components'],
                               'score_bridge_penalty': -.01 * diag['background_bridge_proxy'] / max(limit, 1),
                               'iou': m['iou'], 'cldice': m['cldice'],
                               'apls_gt_to_pred_approx': m['apls_gt_to_pred_approx'],
                               'apls_bidirectional_approx': m['apls_bidirectional_approx'],
                               'short_pred_components': m['short_pred_components'],
                               'background_bridge_proxy': diag['background_bridge_proxy'],
                               'closed_gaps': diag['closed_p0_gaps'],
                               'eligible_gaps': diag['eligible_p0_gaps'],
                               'recall': m['recall'], 'foreground_pixels': m['tp'] + m['fp']})
    write_csv(args.output_dir / 'threshold_scores.csv', threshold_rows)

    # Capture maps on the same full validation cases; use reference threshold from v7 val.
    tau = float(selection7['threshold'])
    records, probes, candidate_quality = [], [], []
    intervention_names = ('uniform_w', 'rotate_w_90', 'shuffle_sources', 'mask_relations',
                          *(f'drop_dir_{i}' for i in range(8)))
    intervention_effects = {name: [] for name in intervention_names}
    dynamics_effects = {'transport_off': [], 'proposal_off': []}
    region_rows = []
    for index in range(limit):
        sample = dataset[index]
        image = sample['image'][None].to(args.device)
        maps = _stitch_work_maps(model, image, args.device)
        record_keys = ('p0', 'q', 'g7', 'g8', 'delta_g', 'p1', 'p7',
                       'p0_full', 'p1_full', 'p7_full', 'q_full',
                       'r', 'a', 'b', 'w',
                       'delta_z_plus', 'delta_z_minus',
                       'proposal_raw_plus', 'proposal_raw_minus',
                       *(f'{name}_{step}' for name in
                         ('alpha', 'beta', 'gamma', 'transport_delta',
                          'proposal_delta', 'anchor_delta') for step in (1, 2, 3)))
        records.append({'case': sample['case_name'], 'gt': sample['mask'][0].numpy() > .5,
                        **{k: maps[k][0] if maps[k].ndim == 3 and maps[k].shape[0] == 1
                           else maps[k]
                           for k in record_keys}})
        probe_rows, quality = _pair_feature_rows(
            index, maps, sample['mask'][0].numpy() > .5, tau)
        probes.extend(probe_rows)
        candidate_quality.append(quality)
        # Stitch counterfactual logits using the same overlap weights as evaluation.
        baseline_logits = torch.zeros((1, 1, 1500, 1500), device=args.device)
        baseline_gate = torch.zeros_like(baseline_logits)
        intervention_logits = {
            name: torch.zeros_like(baseline_logits)
            for name in (*intervention_names, *dynamics_effects)}
        intervention_gates = {name: torch.zeros_like(baseline_gate)
                               for name in intervention_names}
        stitch_mass = torch.zeros_like(baseline_logits)
        tile_weight = blend_weight(512, args.device)
        positions = sliding_positions(1500, 512, 256)
        for top in positions:
            for left in positions:
                tile = image[:, :, top:top+512, left:left+512]
                base_z, base_g, gate_values, dynamics_values = _measure_interventions(
                    model, tile, intervention_names)
                region = (..., slice(top, top + 512), slice(left, left + 512))
                baseline_logits[region] += base_z.float() * tile_weight
                baseline_gate[region] += F.interpolate(
                    base_g.float(), size=(512, 512), mode='bilinear',
                    align_corners=False) * tile_weight
                stitch_mass[region] += tile_weight
                for name, (z_value, g_value) in gate_values.items():
                    intervention_logits[name][region] += z_value.float() * tile_weight
                    intervention_gates[name][region] += F.interpolate(
                        g_value.float(), size=(512, 512), mode='bilinear',
                        align_corners=False) * tile_weight
                for name, z_value in dynamics_values.items():
                    intervention_logits[name][region] += z_value.float() * tile_weight
        gt_full = sample['mask'][None].to(args.device) > .5
        base_probability = (baseline_logits / stitch_mass).sigmoid()
        base_prediction = base_probability >= tau
        for name in intervention_names:
            probability = (intervention_logits[name] / stitch_mass).sigmoid()
            prediction = probability >= tau
            intervention_effects[name].append({
                'image': sample['case_name'],
                'mean_abs_delta_p1': float((probability - base_probability).abs().mean()),
                'mean_abs_delta_g': float(((intervention_gates[name] - baseline_gate) /
                                           stitch_mass).abs().mean()),
                'delta_tp': int((prediction & gt_full).sum() - (base_prediction & gt_full).sum()),
                'delta_fp': int((prediction & ~gt_full).sum() - (base_prediction & ~gt_full).sum()),
                'pixels': int(gt_full.numel()),
            })
        for name in dynamics_effects:
            probability = (intervention_logits[name] / stitch_mass).sigmoid()
            prediction = probability >= tau
            dynamics_effects[name].append({
                'image': sample['case_name'],
                'mean_abs_delta_p1': float((probability - base_probability).abs().mean()),
                'delta_tp': int((prediction & gt_full).sum() - (base_prediction & gt_full).sum()),
                'delta_fp': int((prediction & ~gt_full).sum() - (base_prediction & ~gt_full).sum()),
                'pixels': int(gt_full.numel()),
            })
        region_rows.extend(_region_rows(records[-1], tau))
        print(f'[DIAG] val {index + 1}/{limit}: {sample["case_name"]}', flush=True)

    train_dataset = full_dataset(args.root_path, 'train')
    train_limit = min(len(train_dataset), args.probe_train_images or len(train_dataset))
    train_indices = np.unique(np.linspace(0, len(train_dataset) - 1, train_limit,
                                          dtype=np.int64))
    train_probes, train_candidate_quality = [], []
    model.eval()
    for train_index in train_indices:
        sample = train_dataset[int(train_index)]
        maps = _stitch_work_maps(model, sample['image'][None].to(args.device), args.device)
        rows, quality = _pair_feature_rows(
            int(train_index), maps, sample['mask'][0].numpy() > .5, tau)
        train_probes.extend(rows)
        train_candidate_quality.append(quality)
        print(f'[PROBE-TRAIN] {len(train_candidate_quality)}/{len(train_indices)}', flush=True)

    probes_report = _grouped_probe(train_probes, probes, args.output_dir)
    probes_report['matching_quality'] = {
        'probe_fit_split': 'train',
        'probe_evaluation_split': 'val',
        'train_images_with_matched_rows': sum(q['matched_per_class'] > 0
                                               for q in train_candidate_quality),
        'images_with_matched_rows': sum(q['matched_per_class'] > 0 for q in candidate_quality),
        'train_positive_gap_components': sum(q['positive_gap_components']
                                              for q in train_candidate_quality),
        'train_positive_with_two_flanks': sum(q['positive_with_two_predicted_flanks']
                                               for q in train_candidate_quality),
        'train_negative_with_two_sources': sum(q['negative_with_two_reliable_sources']
                                                     for q in train_candidate_quality),
        'train_matched_pair_total': sum(q['matched_pairs']
                                              for q in train_candidate_quality),
        'positive_gap_components': sum(q['positive_gap_components'] for q in candidate_quality),
        'positive_with_two_flanks': sum(q['positive_with_two_predicted_flanks']
                                        for q in candidate_quality),
        'negative_with_two_sources': sum(q['negative_with_two_reliable_sources']
                                         for q in candidate_quality),
        'matched_pair_total': sum(q['matched_pairs'] for q in candidate_quality),
        'matched_strata_total': sum(q['matched_strata'] for q in candidate_quality),
        'train_per_image': train_candidate_quality,
        'per_image': candidate_quality,
    }
    write_csv(args.output_dir / 'candidate_matching_quality.csv', candidate_quality)
    write_csv(args.output_dir / 'gate_regions.csv', region_rows)
    _write_json(args.output_dir / 'interventions.json',
                {'gate_relation_interventions': intervention_effects,
                 'dynamics_counterfactuals': dynamics_effects,
                 'scope': 'fixed parameters; validation only; per-variant logits stitched '
                          'with the inference overlap weights before full-image metrics'})
    # One validation batch, no optimizer step: confirms which relation blocks receive signal.
    gradient_by_image = []
    loss_weights = {
        'lambda_bg': float(v7_payload['lambda_bg']),
        'lambda_missing': float(v7_payload['lambda_missing']),
        'lambda_cont': float(v7_payload['lambda_cont']),
    }
    for i in range(limit):
        sample = dataset[i]
        missing_pixels, top, left = _gradient_tile(
            records[i]['gt'], records[i]['p0_full'], tau)
        gradient_by_image.append({
            'image': sample['case_name'],
            **_gradient_audit(model, sample, args.device, loss_weights,
                              top, left, missing_pixels),
        })
    gradient_report = {'images': gradient_by_image, 'aggregate_module_l2': {}}
    module_names = tuple(gradient_by_image[0]['per_module_gradient_l2']) if gradient_by_image else ()
    for name in module_names:
        vals = np.asarray([item['per_module_gradient_l2'][name]
                           for item in gradient_by_image], dtype=np.float64)
        gradient_report['aggregate_module_l2'][name] = {
            'mean': float(vals.mean()), 'min': float(vals.min()), 'max': float(vals.max())}
    _write_json(args.output_dir / 'feature_probe.json', probes_report)
    _write_json(args.output_dir / 'gradient_audit.json', gradient_report)

    best = max(threshold_reports, key=lambda t: threshold_reports[t]['selection_score'])
    score055 = threshold_reports.get(.55)
    probe_base = probes_report.get('structure', {})
    h_probe = probes_report.get('structure_plus_H', {})
    h3_probe = probes_report.get('structure_plus_H3', {})
    bridge_probe = probes_report.get('plus_bilateral_relation', {})
    intervention_means = {
        name: {
            'mean_abs_delta_p1': float(np.mean([row['mean_abs_delta_p1'] for row in rows])) if rows else None,
            'mean_abs_delta_g': float(np.mean([row['mean_abs_delta_g'] for row in rows])) if rows else None,
            'mean_delta_tp': float(np.mean([row['delta_tp'] for row in rows])) if rows else None,
            'mean_delta_fp': float(np.mean([row['delta_fp'] for row in rows])) if rows else None,
        }
        for name, rows in intervention_effects.items()}
    dynamics_means = {
        name: {
            'mean_abs_delta_p1': float(np.mean([row['mean_abs_delta_p1'] for row in rows])) if rows else None,
            'mean_delta_tp': float(np.mean([row['delta_tp'] for row in rows])) if rows else None,
            'mean_delta_fp': float(np.mean([row['delta_fp'] for row in rows])) if rows else None,
        }
        for name, rows in dynamics_effects.items()}
    fp_transport = -(dynamics_means['transport_off']['mean_delta_fp'] or 0.)
    fp_proposal = -(dynamics_means['proposal_off']['mean_delta_fp'] or 0.)
    if fp_transport <= 0. and fp_proposal <= 0.:
        fp_conclusion = (
            'Neither single-path removal reduced false positives at this operating point '
            f'(transport-off ΔFP={-fp_transport:.1f}, proposal-off ΔFP={-fp_proposal:.1f} '
            'pixels/image); interactions or threshold crossings prevent a unique attribution.')
        next_module = 'Inspect per-region counterfactuals before changing either update path.'
    elif fp_proposal > fp_transport:
        fp_conclusion = (f'Proposal removal reduces more FP ({fp_proposal:.1f} pixels/image) '
                         f'than transport removal ({fp_transport:.1f}); this is an '
                         'interaction-inclusive counterfactual, not an additive decomposition.')
        next_module = 'Inspect B proposal and A acceptance coupling before changing G/W.'
    else:
        fp_conclusion = (f'Transport removal reduces more FP ({fp_transport:.1f} pixels/image) '
                         f'than proposal removal ({fp_proposal:.1f}); this is an '
                         'interaction-inclusive counterfactual, not an additive decomposition.')
        next_module = 'Refine transport permission/source selection; leave B/A unchanged for isolation.'
    h_conclusion = (
        f"Structure PR-AUC={probe_base.get('pr_auc_average_precision', float('nan')):.4f}; "
        f"+H={h_probe.get('pr_auc_average_precision', float('nan')):.4f}; "
        f"+H3={h3_probe.get('pr_auc_average_precision', float('nan')):.4f}; "
        f"bilateral={bridge_probe.get('pr_auc_average_precision', float('nan')):.4f}. "
        f"Probe fit used {probes_report.get('matching_quality', {}).get('train_images_with_matched_rows', 0)} "
        'train images; evaluation is held-out val only, not a significance test.')
    relation_conclusion = (
        f"mask_relations mean |ΔG|={intervention_means.get('mask_relations', {}).get('mean_abs_delta_g')}; "
        f"uniform_W mean |ΔG|={intervention_means.get('uniform_w', {}).get('mean_abs_delta_g')}; "
        f"shuffle_sources mean |ΔG|={intervention_means.get('shuffle_sources', {}).get('mean_abs_delta_g')}; "
        f"pair-layer gradient L2={gradient_report['aggregate_module_l2'].get('pair', {}).get('mean')}. "
        'Compare these measured perturbations with PR-AUC increments to judge whether the features are used.')
    answers = {
        '1_H_H3_discriminative_value': h_conclusion,
        '2_relation_and_directional_use': relation_conclusion,
        '3_primary_false_positive_path': fp_conclusion,
        '4_next_module_and_threshold_selection': next_module +
        f" Selector chose {best:.2f} because it maximized the printed composite score; "
        f"score@0.55={threshold_reports[.55]['selection_score']:.6f}, "
        f"score@{best:.2f}={threshold_reports[best]['selection_score']:.6f}. "
        'Gap closure is not part of the selection score, so its 14-to-0 change cannot directly lower the chosen threshold score.'}
    report = {
        'split': 'val', 'images': limit, 'v7_threshold': tau,
        'v8_selection_threshold_file': selection8.get('threshold'),
        'v8_recomputed_selection_threshold': best, 'selection_score_formula': score_function,
        'frozen_checkpoint_paths_used': {'base': str(base_path.resolve()),
                                         'q3': str(q3_path.resolve()),
                                         'overrides_supplied': bool(args.base_checkpoint or
                                                                    args.q3_checkpoint)},
        'threshold_0.55': score055, 'selected_threshold_report': threshold_reports[best],
        'v7_vs_v8_feature_probe': probes_report,
        'relation_gate_interventions': intervention_effects,
        'relation_gate_intervention_means': intervention_means,
        'transport_vs_proposal_counterfactuals': dynamics_effects,
        'transport_vs_proposal_means': dynamics_means,
        'relation_encoder_gradients': gradient_report,
        'answers_to_requested_four_questions': answers,
        'notes': [
            'Primary probe unit is a path-level connection pair: missed GT skeleton segment with two P0/GT road flanks versus Q3-high GT-background candidate with two reliable R sources.',
            'Pair candidates are matched within image by P0, Q3, distance to existing P0 roads, gap length, and undirected axis; matching yield is reported per image.',
            'Feature probes are fit only on matched train candidates and evaluated only on matched val candidates; no test split is opened.',
            'Feature groups use S2/C2/D2/R/W, H/H3 center and flank features, raw per-direction relation embeddings, and W-weighted embeddings.',
            'Threshold scoring follows current validation selector; gap closure is diagnostic and is not a term in selection_score.',
            'Counterfactuals are inference-only and not deployable metrics.'
        ]}
    _write_json(args.output_dir / 'diagnosis.json', report)
    lines = ['# v8 Failure Diagnosis (Validation Only)', '',
             f'Images: {limit}; fixed candidate reference threshold={tau:.3f}.', '',
             f'Selection score: `{score_function}`.', '',
             f'Recomputed best threshold: {best:.2f}; checkpoint selection: '
             f'{selection8.get("threshold")}.', '',
             '| Threshold | Score | IoU | clDice | GT→Pred APLS | BiAPLS | Short comps | Closed/eligible gaps |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in threshold_rows:
        lines.append(f'| {row["threshold"]:.2f} | {row["selection_score"]:.5f} | '
                     f'{row["iou"]:.4f} | {row["cldice"]:.4f} | '
                     f'{row["apls_gt_to_pred_approx"]:.4f} | '
                     f'{row["apls_bidirectional_approx"]:.4f} | '
                     f'{row["short_pred_components"]:.2f} | '
                     f'{row["closed_gaps"]}/{row["eligible_gaps"]} |')
    lines += ['', '## Feature Probe', '',
              '| Feature set | PR-AUC | ROC-AUC | FPR@80% recall | FPR@90% recall | Pos | Neg |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, item in probes_report.items():
        if name == 'matching_quality' or not isinstance(item, dict):
            continue
        lines.append(f'| {name} | {item.get("pr_auc_average_precision", 0):.4f} | '
                     f'{item.get("roc_auc", 0):.4f} | '
                     f'{item.get("fpr_at_recall_0.8", 1):.4f} | '
                     f'{item.get("fpr_at_recall_0.9", 1):.4f} | '
                     f'{item.get("positives", 0)} | {item.get("negatives", 0)} |')
    lines += ['', '## Requested Answers', '',
              f'1. H/H3 discriminative value: {h_conclusion}',
              f'2. Relation/directional use: {relation_conclusion}',
              f'3. Main FP path: {fp_conclusion}',
              f'4. Next module: {next_module} Selector behavior: score@0.55='
              f'{threshold_reports[.55]["selection_score"]:.6f}; '
              f'score@{best:.2f}={threshold_reports[best]["selection_score"]:.6f}. '
              'Gap closure is not in the score formula.', '',
              'Detailed region distributions are in `gate_regions.csv`; threshold values in '
              '`threshold_scores.csv`; fixed-weight interventions in `interventions.json`; '
              'gradient evidence in `gradient_audit.json`.', '']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
