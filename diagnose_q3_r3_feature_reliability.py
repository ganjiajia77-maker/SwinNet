"""Frozen train-fit/validation-only reliability audit for Q3/R3 conditions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diagnose_q3_r3_additions import component_rows
from direction_target_utils import build_continuous_direction_target
from losses.road_losses import (build_stage_connectivity_target,
                               build_stage_skeleton_target)
from mass_protocol import blend_weight, sliding_positions
from run_q3_r3_mass import full_dataset, load_experiment
from topology_direction_constants import CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS


def probability(logits):
    return 1. / (1. + np.exp(-np.clip(logits, -80., 80.)))


def describe(values):
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if not array.size:
        return {'n': 0, 'mean': None, 'p10': None, 'median': None, 'p90': None}
    return {'n': int(array.size), 'mean': float(array.mean()),
            'p10': float(np.quantile(array, .1)),
            'median': float(np.median(array)), 'p90': float(np.quantile(array, .9))}


def safe_ap(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    return float(average_precision_score(labels, scores)) if np.unique(labels).size == 2 else None


def calibration(labels, scores, bins=10):
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not labels.size:
        return {'n': 0, 'brier': None, 'ece': None, 'bins': []}
    rows, ece = [], 0.
    edges = np.linspace(0., 1., bins + 1)
    for index in range(bins):
        selected = ((scores >= edges[index]) &
                    ((scores <= edges[index + 1]) if index == bins - 1
                     else (scores < edges[index + 1])))
        if not selected.any():
            continue
        confidence = float(scores[selected].mean())
        frequency = float(labels[selected].mean())
        ece += float(selected.mean()) * abs(confidence - frequency)
        rows.append({'bin_low': float(edges[index]), 'bin_high': float(edges[index + 1]),
                     'count': int(selected.sum()), 'mean_probability': confidence,
                     'positive_frequency': frequency})
    return {'n': int(labels.size), 'brier': float(np.mean((scores - labels) ** 2)),
            'ece': float(ece), 'bins': rows}


def coarse_mask(mask, size, mode='max'):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    if mode == 'max':
        return torch.nn.functional.adaptive_max_pool2d(tensor, size)[0, 0].numpy() > .5
    return torch.nn.functional.adaptive_avg_pool2d(tensor, size)[0, 0].numpy() > .5


@torch.inference_mode()
def stitched_features(model, image):
    """Use the same tile positions and logit blending as run_q3_r3_mass.stitch_three."""
    side = image.shape[-1]
    if image.shape[-2] != side or side % 4:
        raise ValueError(f'Expected a square input divisible by four, got {tuple(image.shape)}')
    weight = blend_weight(512, image.device)[0, 0].cpu().numpy()
    small_weight = cv2.resize(weight, (128, 128), interpolation=cv2.INTER_AREA)
    full = {key: np.zeros((side, side), np.float32) for key in ('z0', 'z1', 'q')}
    full_den = np.zeros((side, side), np.float32)
    small = {}
    small_den = np.zeros((side // 4, side // 4), np.float32)
    for top in sliding_positions(side, 512, 256):
        for left in sliding_positions(side, 512, 256):
            result = model(image[:, :, top:top+512, left:left+512],
                           capture_diagnostics=True)
            a, b = slice(top, top + 512), slice(left, left + 512)
            for key in full:
                if key == 'q':
                    tile = torch.nn.functional.interpolate(
                        result['q'].float(), size=(512, 512), mode='bilinear',
                        align_corners=False)[0, 0].cpu().numpy()
                else:
                    tile = result[key][0, 0].float().cpu().numpy()
                full[key][a, b] += tile * weight
            full_den[a, b] += weight
            a4, b4 = slice(top // 4, top // 4 + 128), slice(left // 4, left // 4 + 128)
            for key in ('q', 'stage2_s', 'stage2_c', 'stage2_d', 'highres_h', 'stage3_h'):
                value = result[key]
                if value is None:
                    continue
                value = torch.nn.functional.interpolate(
                    value.float(), size=(128, 128), mode='bilinear',
                    align_corners=False)[0].cpu().numpy()
                if key not in small:
                    small[key] = np.zeros((value.shape[0], side // 4, side // 4), np.float32)
                small[key][:, a4, b4] += value * small_weight
            small_den[a4, b4] += small_weight
    if (full_den <= 0).any() or (small_den <= 0).any():
        raise RuntimeError('Tile stitching left uncovered pixels')
    stitched = {key: value / full_den for key, value in full.items()}
    small_maps = {key: value / small_den[None] for key, value in small.items()}
    small_maps['q'] = cv2.resize(stitched['q'], (side // 4, side // 4),
                                 interpolation=cv2.INTER_AREA)[None]
    return {**stitched, **small_maps}


def gap_regions(gt, skeleton, p0, q, p1, threshold, q_threshold):
    old = p0 >= threshold
    new = p1 >= threshold
    covered = cv2.dilate(old.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    missing = skeleton & ~covered
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8)
    retained = np.zeros_like(missing)
    flanks = np.zeros_like(missing)
    lengths = {'3-7': 0, '8-15': 0, '16-31': 0, '32+': 0}
    gap_rows = []
    _, gt_parts = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        x, y, width, height = (int(value) for value in stats[index, :4])
        x0, y0 = max(x - 12, 0), max(y - 12, 0)
        x1, y1 = min(x + width + 12, gt.shape[1]), min(y + height + 12, gt.shape[0])
        segment = labels[y0:y1, x0:x1] == index
        retained[y0:y1, x0:x1] |= segment
        name = '3-7' if area < 8 else '8-15' if area < 16 else '16-31' if area < 32 else '32+'
        lengths[name] += 1
        local = cv2.dilate(segment.astype(np.uint8), kernel) > 0
        part_ids = np.unique(gt_parts[y0:y1, x0:x1][segment])
        part_ids = part_ids[part_ids > 0]
        side = (local & skeleton[y0:y1, x0:x1] & covered[y0:y1, x0:x1] &
                np.isin(gt_parts[y0:y1, x0:x1], part_ids))
        flanks[y0:y1, x0:x1] |= side
        gap_rows.append({'label': index, 'length': area, 'bin': name,
                         'flank_pixels': int(side.sum())})
    flanks &= ~retained
    normal = skeleton & covered & ~flanks & ~(
        cv2.dilate(retained.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    q_full = cv2.resize(q[0], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    added_fp = new & ~old & ~gt
    far_from_gt = ~(cv2.dilate(gt.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    hard_background = ~gt & far_from_gt & (q_full >= q_threshold) & ~added_fp
    return {'normal_road': normal, 'gap_center': retained,
            'reliable_flank': flanks, 'q_hard_background': hard_background,
            'new_false_positive': added_fp}, lengths, gap_rows


def input_features(maps):
    parts = {'p0_q3': np.stack((cv2.resize(probability(maps['z0']),
                                                 maps['q'].shape[-2:][::-1],
                                                 interpolation=cv2.INTER_AREA),
                              maps['q'][0])),
             'h': maps['highres_h'], 'h3': maps['stage3_h'],
             'h_h3': np.concatenate((maps['highres_h'], maps['stage3_h'])),
             's2': probability(maps['stage2_s']),
             'c2': probability(maps['stage2_c'])}
    if 'stage2_d' in maps:
        parts['d2'] = maps['stage2_d']
    return parts


def select_samples(maps, regions, rng, per_class):
    shape = maps['q'].shape[-2:]
    positive = coarse_mask(regions['gap_center'], shape)
    negative = coarse_mask(regions['q_hard_background'], shape, mode='mean')
    negative &= ~positive
    indices = []
    targets = []
    counts = {}
    for label, mask in ((1, positive), (0, negative)):
        candidates = np.flatnonzero(mask)
        counts[str(label)] = int(candidates.size)
        if candidates.size > per_class:
            candidates = rng.choice(candidates, per_class, replace=False)
        indices.append(candidates)
        targets.append(np.full(candidates.size, label, np.uint8))
    indices = np.concatenate(indices)
    targets = np.concatenate(targets)
    features = {key: value.reshape(value.shape[0], -1)[:, indices].T.astype(np.float32)
                for key, value in input_features(maps).items()}
    return features, targets, counts


def append_samples(store, features, labels):
    if labels.size:
        store['targets'].append(labels)
        for key, value in features.items():
            store[key].append(value)


def make_targets(skeleton, size):
    tensor = torch.from_numpy(skeleton.astype(np.float32))[None, None]
    stage = build_stage_skeleton_target(tensor, size)[0, 0].numpy() > .5
    conn = build_stage_connectivity_target(tensor, size, 'fine_edges')[0].numpy() > .5
    return stage, conn


def output_regions(maps, regions, skeleton, audit):
    size = maps['q'].shape[-2:]
    stage, conn = make_targets(skeleton, size)
    s2 = probability(maps['stage2_s'])[0]
    c2 = probability(maps['stage2_c'])
    audit['s2_all_scores'].append(s2.ravel())
    audit['s2_all_targets'].append(stage.ravel().astype(np.uint8))
    degree = (cv2.filter2D(stage.astype(np.uint8), cv2.CV_16S,
                           np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
              - stage.astype(np.int16))
    junction = stage & (degree >= 3)
    tangent = build_continuous_direction_target(
        torch.from_numpy(stage.astype(np.float32))[None, None], radius=3)[0].numpy()
    tangent_norm = np.linalg.norm(tangent, axis=0)
    d2 = maps.get('stage2_d')
    if d2 is not None:
        norm = np.linalg.norm(d2, axis=0)
        cosine = np.sum(d2 * tangent, axis=0) / np.maximum(norm * tangent_norm, 1e-6)
        error = np.degrees(np.arccos(np.clip(cosine, -1, 1))) * .5
        valid_dir = stage & (tangent_norm > 1e-6) & (norm > 1e-6)
    for name, full_mask in regions.items():
        mask = coarse_mask(full_mask, size, 'mean' if 'background' in name else 'max')
        audit['s2'][name].append(s2[mask])
        audit['s2_targets'][name].append(stage[mask].astype(np.uint8))
        audit['s2_coverage'][name].extend((int((mask & stage & (s2 >= .5)).sum()),
                                           int((mask & stage).sum())))
        center = mask & stage
        if center.any():
            bounds = np.broadcast_to(center, c2.shape).copy()
            for direction, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
                if dy < 0:
                    bounds[direction, :abs(dy)] = False
                elif dy > 0:
                    bounds[direction, -dy:] = False
                if dx < 0:
                    bounds[direction, :, :abs(dx)] = False
                elif dx > 0:
                    bounds[direction, :, -dx:] = False
            predicted = c2[bounds] >= .5
            expected = conn[bounds]
            counts = audit['c2_agreement'][name]
            counts[0] += int((predicted & expected).sum())
            counts[1] += int((predicted & ~expected).sum())
            counts[2] += int((~predicted & expected).sum())
            counts[3] += int(np.any(bounds, axis=0).sum())
        if name == 'reliable_flank':
            gap = coarse_mask(regions['gap_center'], size)
            flank = mask & stage
            height, width = gap.shape
            for direction, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
                valid = flank.copy()
                if dy < 0:
                    valid[:abs(dy)] = False
                elif dy > 0:
                    valid[-dy:] = False
                if dx < 0:
                    valid[:, :abs(dx)] = False
                elif dx > 0:
                    valid[:, -dx:] = False
                neighbor_gap = np.zeros_like(gap)
                sy0, sy1 = max(0, -dy), min(height, height - dy)
                sx0, sx1 = max(0, -dx), min(width, width - dx)
                neighbor_gap[sy0:sy1, sx0:sx1] = gap[sy0+dy:sy1+dy, sx0+dx:sx1+dx]
                audit['gap_c2_scores'][direction].append(c2[direction][valid])
                audit['gap_c2_targets'][direction].append(
                    (conn[direction] & neighbor_gap)[valid].astype(np.uint8))
        for direction, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            valid = mask & stage
            if dy < 0:
                valid[:abs(dy)] = False
            elif dy > 0:
                valid[-dy:] = False
            if dx < 0:
                valid[:, :abs(dx)] = False
            elif dx > 0:
                valid[:, -dx:] = False
            audit['c2'][name][direction].append(c2[direction][valid])
            audit['c2_gt'][name][direction].append(conn[direction][valid].astype(np.uint8))
            if valid.any():
                audit['c2_all'][direction].append(c2[direction][valid])
                audit['c2_all_gt'][direction].append(conn[direction][valid].astype(np.uint8))
        if d2 is not None:
            select = mask & valid_dir
            audit['d2'][name].append(error[select])
            audit['d2_valid'][name].extend((int(select.sum()), int((mask & stage).sum())))
    if d2 is not None:
        audit['d2']['junction'].append(error[junction & valid_dir])
        audit['d2_valid']['junction'].extend((int((junction & valid_dir).sum()), int(junction.sum())))
    return stage, tangent, tangent_norm > 1e-6


def directional_feature_similarity(maps, stage, tangent, valid_tangent,
                                    regions, audit):
    height, width = stage.shape
    yy, xx = np.nonzero(stage & valid_tangent)
    theta = .5 * np.arctan2(tangent[1], tangent[0])
    dy_all, dx_all = np.sin(theta), np.cos(theta)
    qbg = coarse_mask(regions['q_hard_background'], stage.shape, 'mean')
    flanks = coarse_mask(regions['reliable_flank'], stage.shape)
    features = {'H': maps['highres_h'], 'H3': maps['stage3_h']}
    normalized = {name: value / np.maximum(np.linalg.norm(value, axis=0, keepdims=True), 1e-6)
                  for name, value in features.items()}
    sim_maps = {}
    for name, feature in normalized.items():
        sum_map = np.zeros((height, width), np.float32)
        count_map = np.zeros((height, width), np.float32)
        for radius in (1, 2, 4):
            for sign in (-1, 1):
                oy = np.rint(sign * dy_all[yy, xx] * radius).astype(np.int32)
                ox = np.rint(sign * dx_all[yy, xx] * radius).astype(np.int32)
                ny, nx = yy + oy, xx + ox
                inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
                y0, x0, y1, x1 = yy[inside], xx[inside], ny[inside], nx[inside]
                cosine = np.sum(feature[:, y0, x0] * feature[:, y1, x1], axis=0)
                onroad = stage[y1, x1]
                negative = qbg[y1, x1]
                audit['h_similarity'][name]['along_gt'].append(cosine[onroad])
                audit['h_similarity'][name]['nonroad'].append(cosine[~onroad])
                audit['h_similarity'][name]['q_hard_background'].append(cosine[negative])
                sum_map[y0, x0] += cosine
                count_map[y0, x0] += 1

        fy, fx = np.nonzero(flanks & stage)
        wrong_scores = []
        for radius in (1, 2, 4):
            for dy, dx in CONNECTIVITY_DIRECTIONS:
                ny, nx = fy + dy * radius, fx + dx * radius
                inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
                y0, x0, y1, x1 = fy[inside], fx[inside], ny[inside], nx[inside]
                wrong = qbg[y1, x1] & ~stage[y1, x1]
                if wrong.any():
                    wrong_scores.append(np.sum(feature[:, y0[wrong], x0[wrong]] *
                                               feature[:, y1[wrong], x1[wrong]], axis=0))
        audit['h_similarity'][name]['wrong_connection_candidate'].append(
            np.concatenate(wrong_scores) if wrong_scores else np.empty(0, np.float32))
        sim_maps[name] = sum_map / np.maximum(count_map, 1.)
    return sim_maps


def concat(chunks):
    return np.concatenate(chunks) if chunks else np.empty(0, np.float32)


def pr_artifact(path, labels, scores, title):
    if np.unique(labels).size != 2:
        return None
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('recall', 'precision', 'threshold'))
        writer.writerows(zip(recall, precision, np.r_[thresholds, np.nan]))
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path.with_suffix('.png'), dpi=160)
    plt.close()
    return safe_ap(labels, scores)


def visualize(path, sample, maps, regions, threshold, similarity_maps):
    image = sample['image'].numpy().transpose(1, 2, 0)
    image = np.clip((image * [0.229, 0.224, 0.225] +
                     [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    p0 = probability(maps['z0'])
    p1 = probability(maps['z1'])
    q = cv2.resize(maps['q'][0], p0.shape[::-1], interpolation=cv2.INTER_LINEAR)
    s = cv2.resize(probability(maps['stage2_s'])[0], p0.shape[::-1],
                   interpolation=cv2.INTER_LINEAR)
    overlay = image.copy()
    overlay[regions['normal_road']] = (40, 160, 40)
    overlay[regions['reliable_flank']] = (20, 200, 230)
    overlay[regions['gap_center']] = (20, 20, 250)
    overlay[regions['q_hard_background']] = (220, 60, 180)
    overlay[regions['new_false_positive']] = (20, 140, 255)
    legend = (('normal GT road', (40, 160, 40)),
              ('gap flank', (20, 200, 230)),
              ('gap center', (20, 20, 250)),
              ('Q-high hard BG', (220, 60, 180)),
              ('new FP', (20, 140, 255)))
    for line, (label, color) in enumerate(legend):
        cv2.putText(overlay, label, (12, 66 + line * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, color, 2)
    c2_vis = image.copy()
    c2 = probability(maps['stage2_c'])
    dirs = CONNECTIVITY_DIRECTIONS
    for y in range(8, c2.shape[1], 12):
        for x in range(8, c2.shape[2], 12):
            direction = int(np.argmax(c2[:, y, x]))
            score = float(c2[direction, y, x])
            if score < .5:
                continue
            dy, dx = dirs[direction]
            start = (x * 4, y * 4)
            end = (int((x + dx * 5) * 4), int((y + dy * 5) * 4))
            cv2.arrowedLine(c2_vis, start, end, (20, 230, 230), 2, tipLength=.3)
    d2_vis = image.copy()
    d2 = maps.get('stage2_d')
    if d2 is not None:
        d2 = d2 / np.maximum(np.linalg.norm(d2, axis=0, keepdims=True), 1e-6)
        sgrid = probability(maps['stage2_s'])[0]
        for y in range(8, d2.shape[1], 12):
            for x in range(8, d2.shape[2], 12):
                if sgrid[y, x] < .5:
                    continue
                theta = .5 * np.arctan2(d2[1, y, x], d2[0, y, x])
                dx, dy = int(np.cos(theta) * 18), int(np.sin(theta) * 18)
                cx, cy = x * 4, y * 4
                cv2.line(d2_vis, (cx-dx, cy-dy), (cx+dx, cy+dy), (255, 180, 20), 2)
    def heat(value):
        value = cv2.resize(value, p0.shape[::-1], interpolation=cv2.INTER_LINEAR)
        return cv2.applyColorMap(np.uint8(np.clip((value + 1.) * 127.5, 0, 255)),
                                 cv2.COLORMAP_TURBO)
    q_rgb = cv2.cvtColor(np.uint8(np.clip(q, 0, 1) * 255), cv2.COLOR_GRAY2BGR)
    p0_rgb = cv2.cvtColor(np.uint8(np.clip(p0, 0, 1) * 255), cv2.COLOR_GRAY2BGR)
    s_rgb = cv2.cvtColor(np.uint8(np.clip(s, 0, 1) * 255), cv2.COLOR_GRAY2BGR)
    maps9 = [image, overlay, p0_rgb, q_rgb, s_rgb, c2_vis, d2_vis,
             heat(similarity_maps['H']), heat(similarity_maps['H3'])]
    panels = []
    names = ('image', 'GT regions', 'P0', 'Q3', 'S2', 'C2 arrows',
             'D2 axial field', 'H along-road cosine', 'H3 along-road cosine')
    for name, bitmap in zip(names, maps9):
        bitmap = cv2.resize(bitmap, (480, 480), interpolation=cv2.INTER_AREA)
        cv2.putText(bitmap, name, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3)
        panels.append(bitmap)
    rows = [np.concatenate(panels[offset:offset+3], axis=1) for offset in (0, 3, 6)]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--repair_checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', type=float, default=.6)
    parser.add_argument('--q_threshold', type=float, default=.5)
    parser.add_argument('--train_images', type=int, default=96)
    parser.add_argument('--train_per_class', type=int, default=300)
    parser.add_argument('--val_per_class', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 0 < args.q_threshold < 1:
        parser.error('Thresholds must lie in (0, 1)')
    if min(args.train_images, args.train_per_class, args.val_per_class) < 1:
        parser.error('Training/image and sample limits must be positive')
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.repair_checkpoint = args.repair_checkpoint.resolve()
    args.init_repair_checkpoint = None
    args.eta_plus = args.eta_minus = .5
    args.enable_stage2_local_propagation = False
    args.enable_gap_refinement = False
    args.device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = load_experiment(args).eval().requires_grad_(False)
    if model.base.swin_unet.local_gap_config.c_target != 'fine_edges':
        raise ValueError('This audit expects the checkpoint fine_edges C2 target')
    if model.base.swin_unet.decoder_structure_blocks['2'].enable_local_propagation:
        raise RuntimeError('Stage 2 propagation must remain disabled')
    if model.base.swin_unet.enable_gap_refinement:
        raise RuntimeError('Legacy gap refinement must remain disabled')
    rng = np.random.default_rng(args.seed)
    train = full_dataset(args.root_path, 'train')
    val = full_dataset(args.root_path, 'val')
    if len(val) != 14:
        raise ValueError(f'Expected exactly 14 validation images, got {len(val)}')
    train_ids = rng.permutation(len(train))[:min(args.train_images, len(train))]
    train_store = defaultdict(list)
    train_counts = defaultdict(int)
    for ordinal, index in enumerate(train_ids):
        sample = train[int(index)]
        side = sample['image'].shape[-1]
        if side != 1500:
            raise ValueError('Expected native 1500 Mass training images')
        top = int(rng.integers(0, side - 512 + 1)) // 4 * 4
        left = int(rng.integers(0, side - 512 + 1)) // 4 * 4
        image = sample['image'][None, :, top:top+512, left:left+512].to(args.device)
        with torch.inference_mode():
            output = model(image, capture_diagnostics=True)
        maps = {key: value[0].float().cpu().numpy() if key not in ('z0', 'z1') else
                value[0, 0].float().cpu().numpy() for key, value in output.items()
                if key in ('z0', 'z1', 'q', 'stage2_s', 'stage2_c', 'stage2_d',
                           'highres_h', 'stage3_h') and value is not None}
        gt = sample['mask'][0, top:top+512, left:left+512].numpy() > .5
        skel = sample['skeleton'][0, top:top+512, left:left+512].numpy() > .5
        regions, _, _ = gap_regions(gt, skel, probability(maps['z0']), maps['q'],
                                    probability(maps['z1']), args.threshold, args.q_threshold)
        features, labels, counts = select_samples(maps, regions, rng, args.train_per_class)
        append_samples(train_store, features, labels)
        for key, value in counts.items():
            train_counts[key] += value
        print(f'[TRAIN PROBE] {ordinal + 1}/{len(train_ids)} +={counts["1"]} -={counts["0"]}',
              flush=True)
    train_y = concat(train_store['targets'])
    if np.unique(train_y).size != 2:
        raise RuntimeError('Training crops did not contain both true gaps and Q-high hard background')

    audit = {'s2': defaultdict(list), 's2_coverage': defaultdict(list),
             's2_targets': defaultdict(list),
             'c2': defaultdict(lambda: defaultdict(list)),
             'c2_gt': defaultdict(lambda: defaultdict(list)),
             'c2_agreement': defaultdict(lambda: [0, 0, 0, 0]),
             'c2_all': defaultdict(list), 'c2_all_gt': defaultdict(list),
             'gap_c2_scores': defaultdict(list), 'gap_c2_targets': defaultdict(list),
             'd2': defaultdict(list), 'd2_valid': defaultdict(list),
             'h_similarity': defaultdict(lambda: defaultdict(list)),
             's2_all_scores': [], 's2_all_targets': []}
    val_store = defaultdict(list)
    val_counts = defaultdict(int)
    gap_bins = defaultdict(lambda: {'count': 0, 'pixels': 0, 's2_covered': 0,
                                     'q_high_pixels': 0})
    fp_categories = defaultdict(lambda: {'components_with_fp': 0, 'fp_pixels': 0})
    image_rows = []
    representative = []
    for index in range(len(val)):
        sample = val[index]
        maps = stitched_features(model, sample['image'][None].to(args.device))
        gt = sample['mask'][0].numpy() > .5
        skel = sample['skeleton'][0].numpy() > .5
        p0, p1 = probability(maps['z0']), probability(maps['z1'])
        regions, lengths, segments = gap_regions(
            gt, skel, p0, maps['q'], p1, args.threshold, args.q_threshold)
        stage_grid, tangent, valid_tangent = output_regions(maps, regions, skel, audit)
        similarity_maps = directional_feature_similarity(
            maps, stage_grid, tangent, valid_tangent, regions, audit)
        features, labels, counts = select_samples(maps, regions, rng, args.val_per_class)
        append_samples(val_store, features, labels)
        for key, value in counts.items():
            val_counts[key] += value
        s2_full = cv2.resize(probability(maps['stage2_s'])[0], p0.shape[::-1],
                             interpolation=cv2.INTER_LINEAR)
        q_full = cv2.resize(maps['q'][0], p0.shape[::-1], interpolation=cv2.INTER_LINEAR)
        missing = regions['gap_center']
        gap_count, gap_labels, gap_stats, _ = cv2.connectedComponentsWithStats(
            missing.astype(np.uint8), connectivity=8)
        s2_positive = np.bincount(gap_labels[s2_full >= .5].ravel(), minlength=gap_count)
        q_positive = np.bincount(gap_labels[q_full >= args.q_threshold].ravel(),
                                 minlength=gap_count)
        for part in range(1, gap_count):
            length = int(gap_stats[part, cv2.CC_STAT_AREA])
            name = '3-7' if length < 8 else '8-15' if length < 16 else '16-31' if length < 32 else '32+'
            gap_bins[name]['count'] += 1
            gap_bins[name]['pixels'] += length
            gap_bins[name]['s2_covered'] += int(s2_positive[part])
            gap_bins[name]['q_high_pixels'] += int(q_positive[part])
        components = component_rows(sample['case_name'], 'dual_recovery',
                                    (p1 >= args.threshold) & (p0 < args.threshold),
                                    p0 >= args.threshold, gt)
        for item in components:
            fp_pixels = item['area_px'] - item['gt_overlap_px']
            if fp_pixels:
                category = item['category']
                fp_categories[category]['components_with_fp'] += 1
                fp_categories[category]['fp_pixels'] += fp_pixels
        row = {'case_id': sample['case_name'], 'gap_segments': len(segments),
               'gap_pixels': int(missing.sum()),
               'gap_center_s2_mean': describe(s2_full[missing])['mean'],
               'flank_s2_mean': describe(s2_full[regions['reliable_flank']])['mean'],
               'new_fp_pixels': int(regions['new_false_positive'].sum()),
               'probe_positive_candidates': counts['1'],
               'probe_negative_candidates': counts['0'], **lengths}
        image_rows.append(row)
        case_path = args.output_dir / f'case_{index:02d}_{sample["case_name"]}.png'
        visualize(case_path, sample, maps, regions, args.threshold, similarity_maps)
        representative.append((row['gap_pixels'] + row['new_fp_pixels'], str(case_path)))
        if len(representative) > 8:
            representative.sort(reverse=True)
            _, discard = representative.pop()
            Path(discard).unlink(missing_ok=True)
        print(f'[VAL] {index + 1}/14 {sample["case_name"]} gaps={len(segments)} '
              f'new_fp={row["new_fp_pixels"]}', flush=True)

    with (args.output_dir / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)
    representative.sort(reverse=True)

    s2_labels = concat(audit['s2_all_targets'])
    s2_scores = concat(audit['s2_all_scores'])
    s2_ap = pr_artifact(args.output_dir / 's2_pr', s2_labels, s2_scores,
                        'S2 versus stage skeleton target')
    s2_global_calibration = calibration(s2_labels, s2_scores)
    global_reliability_bins = s2_global_calibration.pop('bins')
    region_report = {}
    hard_negative = concat(audit['s2']['q_hard_background'])
    reliability_rows = []
    for region, values in audit['s2'].items():
        hits, total = audit['s2_coverage'][region][::2], audit['s2_coverage'][region][1::2]
        scores = concat(values)
        labels = concat(audit['s2_targets'][region])
        region_calibration = calibration(labels, scores)
        reliability_rows.extend({'region': region, **row}
                                for row in region_calibration.pop('bins'))
        region_report[region] = {'s2': describe(scores),
                                 's2_skeleton_coverage': sum(hits) / sum(total) if sum(total) else None,
                                 's2_positive_gt_pixels': sum(total),
                                 's2_recall_at_0.5': (float((scores[labels > 0] >= .5).mean())
                                                      if np.any(labels > 0) else None),
                                 's2_specificity_at_0.5': (float((scores[labels == 0] < .5).mean())
                                                           if np.any(labels == 0) else None),
                                 's2_calibration': region_calibration, 'c2': {}}
        positives = concat(values)
        if region in ('normal_road', 'gap_center', 'reliable_flank') and positives.size and hard_negative.size:
            contrast_scores = np.concatenate((positives, hard_negative))
            contrast_labels = np.r_[np.ones(positives.size, np.uint8),
                                    np.zeros(hard_negative.size, np.uint8)]
            region_report[region]['s2_pr_auc_vs_q_hard_background'] = pr_artifact(
                args.output_dir / f's2_{region}_vs_hard_background_pr',
                contrast_labels, contrast_scores, f'S2 {region} versus hard background')
            region_report[region]['s2_contrast_prevalence'] = float(contrast_labels.mean())
        tp, fp, fn, centers = audit['c2_agreement'][region]
        region_report[region]['c2_eight_direction_agreement'] = {
            'valid_centers': centers, 'tp': tp, 'fp': fp, 'fn': fn,
            'micro_precision_at_0.5': tp / (tp + fp) if tp + fp else None,
            'micro_recall_at_0.5': tp / (tp + fn) if tp + fn else None}
        for direction, name in enumerate(CONNECTIVITY_DIR_NAMES):
            scores = concat(audit['c2'][region][direction])
            targets = concat(audit['c2_gt'][region][direction])
            region_report[region]['c2'][name] = {
                'probability': describe(scores), 'gt_positive_rate': float(targets.mean()) if targets.size else None,
                'pr_auc': safe_ap(targets, scores) if scores.size else None,
                'precision_at_0.5': (float((targets[scores >= .5]).mean())
                                     if np.any(scores >= .5) else None),
                'recall_at_0.5': (float(np.mean(scores[targets > 0] >= .5))
                                  if np.any(targets > 0) else None)}
        if 'd2' in audit:
            region_report[region]['d2_axial_error_degrees'] = describe(concat(audit['d2'][region]))
            d_counts = audit['d2_valid'][region]
            region_report[region]['d2_valid_fraction'] = (
                sum(d_counts[::2]) / sum(d_counts[1::2]) if sum(d_counts[1::2]) else None)
    reliability_rows.extend({'region': 'all_stage_cells', **row}
                            for row in global_reliability_bins)
    with (args.output_dir / 's2_calibration_bins.csv').open(
            'w', newline='', encoding='utf-8') as handle:
        fields = ('region', 'bin_low', 'bin_high', 'count', 'mean_probability',
                  'positive_frequency')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reliability_rows)
    plt.figure(figsize=(5, 5))
    plt.plot((0, 1), (0, 1), '--', color='gray')
    for region in sorted({row['region'] for row in reliability_rows}):
        rows = [row for row in reliability_rows if row['region'] == region]
        if rows:
            plt.plot([row['mean_probability'] for row in rows],
                     [row['positive_frequency'] for row in rows], marker='o', label=region)
    plt.xlabel('Mean S2 probability')
    plt.ylabel('GT skeleton frequency')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output_dir / 's2_calibration.png', dpi=160)
    plt.close()
    c2_ap = {}
    gap_c2_report = {}
    for direction, name in enumerate(CONNECTIVITY_DIR_NAMES):
        y = concat(audit['c2_all_gt'][direction])
        score = concat(audit['c2_all'][direction])
        c2_ap[name] = pr_artifact(args.output_dir / f'c2_{name}_pr', y, score,
                                 f'C2 {name} valid-center adjacency')
        flank_y = concat(audit['gap_c2_targets'][direction])
        flank_score = concat(audit['gap_c2_scores'][direction])
        flank_ap = pr_artifact(args.output_dir / f'c2_flank_to_gap_{name}_pr',
                               flank_y, flank_score,
                               f'C2 {name}: GT continuation from flank into gap')
        gap_c2_report[name] = {
            'valid_flank_centers': int(flank_y.size),
            'gt_edges_toward_gap': int(flank_y.sum()),
            'mean_probability_on_gt_continuation':
                float(flank_score[flank_y > 0].mean()) if np.any(flank_y > 0) else None,
            'recall_at_0.5': (float((flank_score[flank_y > 0] >= .5).mean())
                              if np.any(flank_y > 0) else None),
            'false_direction_rate_at_0.5':
                float((flank_score[flank_y == 0] >= .5).mean()) if np.any(flank_y == 0) else None,
            'pr_auc': flank_ap}
    h_similarity_report = {
        name: {kind: describe(concat(chunks)) for kind, chunks in kinds.items()}
        for name, kinds in audit['h_similarity'].items()}

    val_y = concat(val_store['targets'])
    if np.unique(val_y).size != 2:
        raise RuntimeError('Validation subset lacks one comparison class')
    has_d2 = 'd2' in train_store and 'd2' in val_store
    structural = ('s2', 'c2', 'd2') if has_d2 else ('s2', 'c2')
    groups = [('s2',), ('c2',)]
    if has_d2:
        groups.append(('d2',))
    groups.extend([('s2', 'c2') + (('d2',) if has_d2 else ()),
                   ('h',), ('h3',), ('h', 'h3'),
                   ('h', 'h3') + structural,
                   ('p0_q3',), ('p0_q3', 's2'), ('p0_q3', 'c2'),
                   ('p0_q3', 'h'), ('p0_q3', 'h3'),
                   ('p0_q3', 's2', 'h', 'h3'),
                   ('p0_q3', 's2', 'c2'),
                   ('p0_q3', 'h', 'h3'),
                   ('p0_q3',) + structural,
                   ('p0_q3', 'h', 'h3') + structural])
    if has_d2:
        groups.extend([('p0_q3', 'd2'), ('p0_q3', 's2', 'd2')])
    probe_results = {}
    for group in groups:
        name = '+'.join(group)
        train_x = np.concatenate([concat(train_store[key]) for key in group], axis=1)
        val_x = np.concatenate([concat(val_store[key]) for key in group], axis=1)
        probe = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=args.seed))
        probe.fit(train_x, train_y)
        score = probe.predict_proba(val_x)[:, 1]
        ap = pr_artifact(args.output_dir / f'probe_{name}_pr', val_y, score, name)
        probe_results[name] = {'pr_auc_sampled': ap,
                               'roc_auc_sampled': float(roc_auc_score(val_y, score)),
                               'n_features': int(train_x.shape[1])}

    for values in gap_bins.values():
        values['s2_coverage'] = (values['s2_covered'] / values['pixels'] if values['pixels'] else None)
        values['q_high_fraction'] = (values['q_high_pixels'] / values['pixels'] if values['pixels'] else None)
    baseline_ap = probe_results['p0_q3']['pr_auc_sampled']
    for values in probe_results.values():
        values['delta_pr_auc_vs_p0_q3'] = values['pr_auc_sampled'] - baseline_ap
    normal_c2 = region_report['normal_road']['c2_eight_direction_agreement']
    flank_c2 = region_report['reliable_flank']['c2_eight_direction_agreement']
    cue_assessment = {
        'h_h3': {'incremental_pr_auc_vs_p0_q3':
                 probe_results['p0_q3+h+h3']['delta_pr_auc_vs_p0_q3'],
                 'h_only_increment': probe_results['p0_q3+h']['delta_pr_auc_vs_p0_q3'],
                 'h3_only_increment': probe_results['p0_q3+h3']['delta_pr_auc_vs_p0_q3'],
                 'with_s2_increment':
                 probe_results['p0_q3+s2+h+h3']['pr_auc_sampled'] -
                 probe_results['p0_q3+s2']['pr_auc_sampled']},
        's2': {'incremental_pr_auc_vs_p0_q3':
               probe_results['p0_q3+s2']['delta_pr_auc_vs_p0_q3'],
               'gap_coverage': region_report['gap_center']['s2_skeleton_coverage'],
               'flank_coverage': region_report['reliable_flank']['s2_skeleton_coverage']},
        'c2': {'incremental_pr_auc_vs_p0_q3':
               probe_results['p0_q3+c2']['delta_pr_auc_vs_p0_q3'],
               'incremental_pr_auc_given_s2':
               probe_results['p0_q3+s2+c2']['pr_auc_sampled'] -
               probe_results['p0_q3+s2']['pr_auc_sampled'],
               'normal_direction_precision': normal_c2['micro_precision_at_0.5'],
               'flank_direction_precision': flank_c2['micro_precision_at_0.5']}}
    if has_d2:
        cue_assessment['d2'] = {
            'incremental_pr_auc_vs_p0_q3': probe_results['p0_q3+d2']['delta_pr_auc_vs_p0_q3'],
            'incremental_pr_auc_scd_vs_sc':
                probe_results['p0_q3+s2+c2+d2']['pr_auc_sampled'] -
                probe_results['p0_q3+s2+c2']['pr_auc_sampled'],
            'incremental_pr_auc_given_s2':
                probe_results['p0_q3+s2+d2']['pr_auc_sampled'] -
                probe_results['p0_q3+s2']['pr_auc_sampled'],
            'flank_median_axial_error_degrees':
                region_report['reliable_flank']['d2_axial_error_degrees']['median'],
            'junction_median_axial_error_degrees': describe(concat(audit['d2']['junction']))['median']}
    report = {'base_checkpoint': str(args.base_checkpoint),
              'repair_checkpoint': str(args.repair_checkpoint), 'split': 'val',
              'validation_images': len(val), 'probe_fit_split': 'train',
              'probe_train_images': len(train_ids), 'seed': args.seed,
              'surface_threshold': args.threshold, 'q_threshold': args.q_threshold,
              'train_candidate_counts': dict(train_counts),
              'val_candidate_counts': dict(val_counts),
              'probe_train_samples': int(train_y.size), 'probe_val_samples': int(val_y.size),
              'probe_val_prevalence_sampled': float(val_y.mean()),
              'probe_groups': probe_results, 'regions': region_report,
              'cue_assessment': cue_assessment,
              's2_pr_auc_global': s2_ap, 'c2_pr_auc_valid_by_direction': c2_ap,
              's2_global_calibration': s2_global_calibration,
              'c2_gap_flank_continuation_by_direction': gap_c2_report,
              'h_gt_direction_similarity': h_similarity_report,
              'd2_junction_error_degrees': describe(concat(audit['d2']['junction'])),
              'gap_length_bins': dict(gap_bins),
              'new_fp_components_and_pixels': dict(fp_categories),
              'notes': [
                  'Probe PR-AUC uses the same capped, sampled validation points for every group; '
                  'it does not estimate natural pixel prevalence.',
                  'C2 is evaluated on GT stage skeleton centers and in-bounds directions '
                  'using the existing fine_edges target, not on arbitrary background.',
                  'Pure gap/flank regions contain positive GT skeleton only; S2 PR-AUC there '
                  'is undefined, so coverage and distributions are reported instead.',
                  'Gap lengths are 8-connected missed-centerline pixel counts, a proxy for path length.',
                  'Reliable flanks are covered GT skeleton within 12 pixels of a gap and in the '
                  'same GT connected component; proximity does not prove two-sided path continuity.',
                  'Junctions use coarse 8-neighbor degree >=3; diagonal bends can be overcounted.',
                  'FP expansion/island/bridge labels are geometric proxies, not verified road graph errors.',
                  'GT is used for train probe labels and validation scoring only, never as model input.',
                  'No model weights are updated; Stage 2 propagation and legacy Gap remain off.'
              ]}
    best = max(probe_results, key=lambda key: probe_results[key]['pr_auc_sampled'])
    c2_gain = (probe_results['p0_q3+s2+c2']['pr_auc_sampled'] -
               probe_results['p0_q3+s2']['pr_auc_sampled'])
    if c2_gain > .01:
        c2_message = ('C2 adds more than 0.01 sampled PR-AUC; it merits a controlled '
                      'directional-propagation experiment, subject to valid-center '
                      'direction agreement and false-bridge review.')
    else:
        c2_message = ('C2 adds at most 0.01 sampled PR-AUC in this probe; keep it '
                      'as a weak auxiliary cue until a controlled ablation demonstrates benefit.')
    increments = ', '.join(f'{name}: {values["delta_pr_auc_vs_p0_q3"]:+.4f}'
                           for name, values in probe_results.items() if name != 'p0_q3')
    conclusion = ('The highest sampled validation PR-AUC came from ' + best + '. ' +
                  'Sequential incremental PR-AUC changes were ' + increments + '. ' +
                  c2_message + ' This is diagnostic association, not causal evidence. '
                  'Geometry-based FP bridge labels require visual review.')
    report['representative_cases'] = [Path(path).name for _, path in representative[-8:]]
    report['conclusion'] = conclusion
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output_dir / 'conclusion.txt').write_text(conclusion + '\n', encoding='utf-8')
    print(json.dumps({'summary': str(args.output_dir / 'summary.json'),
                      'probe_groups': probe_results, 'conclusion': conclusion}, indent=2), flush=True)


if __name__ == '__main__':
    main()
