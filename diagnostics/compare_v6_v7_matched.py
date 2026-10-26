"""Compare v6/v7 at validation-matched operating points on native Mass images."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.road_comparison.road_metrics import image_metrics, prepare_target, summarize  # noqa: E402
from mass_protocol import sliding_positions  # noqa: E402
from run_q3_r3_mass import PROTOCOL, full_dataset, stitch_three  # noqa: E402
from run_reliable_directional_mass import gap_lengths, load_model  # noqa: E402


def probability(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -80.0, 80.0)))


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_dynamic(path, device, expected_version):
    path = path.resolve()
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('version') != expected_version:
        raise ValueError(f'{path} has version {payload.get("version")!r}, expected {expected_version!r}')
    base, q3 = Path(payload['base_checkpoint']), Path(payload['q3_checkpoint'])
    if not base.is_file() or not q3.is_file():
        raise FileNotFoundError(f'Missing base or Q3 checkpoint from {path}: {base}, {q3}')
    config = SimpleNamespace(
        mode='audit', dynamic_checkpoint=path, init_dynamic_checkpoint=None,
        base_checkpoint=base.resolve(), q3_checkpoint=q3.resolve(), device=device,
        width=payload['width'], accept_floor_initial=payload['accept_floor_initial'],
        accept_floor_anneal_epochs=payload['accept_floor_anneal_epochs'],
        missing_threshold=payload['missing_threshold'], lambda_bg=payload['lambda_bg'],
        lambda_missing=payload.get('lambda_missing', 2.0),
        lambda_cont=payload['lambda_cont'], transport_gate_mode='v6')
    return load_model(config).eval(), (base.resolve(), q3.resolve())


@torch.no_grad()
def predictions(models, dataset, device, limit):
    records = []
    for index in range(limit):
        sample = dataset[index]
        image = sample['image'][None].to(device)
        z0_6, z1_6, _ = stitch_three(models['v6'], image)
        z0_7, z1_7, _ = stitch_three(models['v7'], image)
        if not np.allclose(z0_6, z0_7, rtol=0, atol=1e-5):
            raise RuntimeError(f'P0 mismatch on {sample["case_name"]}; checkpoints do not share a base')
        records.append({
            'index': index, 'case': sample['case_name'],
            'gt': (sample['mask'][0].numpy() > .5),
            'p0': probability(z0_6), 'v6': probability(z1_6),
            'v7': probability(z1_7)})
        print(f'[PREDICT] {index + 1}/{limit}: {sample["case_name"]}', flush=True)
    return records


def threshold_for_count(scores, desired):
    """Choose a threshold whose >= count is as close as possible to desired."""
    scores = np.asarray(scores, dtype=np.float32).ravel()
    desired = int(desired)
    if not 0 <= desired <= scores.size:
        raise ValueError('Desired count is outside the available score array')
    if desired == 0:
        return float(np.nextafter(float(scores.max()), np.inf))
    if desired == scores.size:
        return float(scores.min())
    split = scores.size - desired
    partition = np.partition(scores, (split - 1, split))
    lower, upper = float(partition[split - 1]), float(partition[split])
    candidates = [upper, float(np.nextafter(upper, np.inf))]
    if lower < upper:
        candidates.append((lower + upper) / 2.0)
    return min(candidates, key=lambda value: (abs(int(np.count_nonzero(scores >= value)) - desired), value))


def choose_thresholds(records, reference_threshold, grid_step):
    target_tp = sum(int((record['v6'] >= reference_threshold)[record['gt']].sum())
                    for record in records)
    target_area = sum(int((record['v6'] >= reference_threshold).sum()) for record in records)
    v7_all = np.concatenate([record['v7'].ravel() for record in records])
    v7_road = np.concatenate([record['v7'][record['gt']] for record in records])
    recall_threshold = threshold_for_count(v7_road, target_tp)
    area_threshold = threshold_for_count(v7_all, target_area)
    grid = np.round(np.arange(.30, .800001, grid_step), 6)
    if grid[-1] < .8:
        grid = np.append(grid, .8)
    edges = np.r_[grid, 1.000001]
    all_hist = np.zeros(len(grid), dtype=np.int64)
    road_hist = np.zeros_like(all_hist)
    for record in records:
        all_hist += np.histogram(record['v7'], bins=edges)[0]
        road_hist += np.histogram(record['v7'][record['gt']], bins=edges)[0]
    areas = np.cumsum(all_hist[::-1])[::-1]
    true_positives = np.cumsum(road_hist[::-1])[::-1]
    total_road = int(sum(record['gt'].sum() for record in records))
    joint_error = (np.abs(true_positives - target_tp) / max(total_road, 1) +
                   np.abs(areas - target_area) / max(target_area, 1))
    joint_threshold = float(grid[np.argmin(joint_error)])
    scan = [dict(threshold=float(t), tp=int(tp), area=int(area),
                 recall=float(tp / max(total_road, 1)),
                 recall_difference=float((tp - target_tp) / max(total_road, 1)),
                 foreground_area_difference=int(area - target_area))
            for t, tp, area in zip(grid, true_positives, areas)]
    points = {'v6_reference': reference_threshold, 'v7_same_threshold': reference_threshold,
              'v7_match_recall': recall_threshold, 'v7_match_area': area_threshold,
              'v7_joint_grid': joint_threshold}
    residuals = {}
    for name, threshold in points.items():
        model = 'v6' if name == 'v6_reference' else 'v7'
        tp = sum(int((record[model] >= threshold)[record['gt']].sum()) for record in records)
        area = sum(int((record[model] >= threshold).sum()) for record in records)
        residuals[name] = {'tp': tp, 'recall': tp / max(total_road, 1),
                           'foreground_area': area, 'tp_difference': tp - target_tp,
                           'foreground_area_difference': area - target_area}
    return points, residuals, scan


def evaluate_points(records, points):
    rows, paired = defaultdict(list), defaultdict(list)
    for record in records:
        gt = record['gt']
        prepared = prepare_target(gt, PROTOCOL)
        p0 = record['p0'] >= points['v6_reference']
        for name, threshold in points.items():
            model = 'v6' if name == 'v6_reference' else 'v7'
            pred = record[model] >= threshold
            rows[name].append({'case_id': record['case'], **image_metrics(pred, prepared, PROTOCOL)})
            bins, bridges = gap_lengths(p0, pred, prepared['skeleton'], gt)
            paired[name].append({
                'case_id': record['case'], 'eligible_gaps': sum(v['eligible'] for v in bins.values()),
                'closed_gaps': sum(v['closed'] for v in bins.values()),
                'background_bridge_proxy': bridges,
                'new_gt_pixels_vs_p0': int((pred & ~p0 & gt).sum()),
                'new_background_pixels_vs_p0': int((pred & ~p0 & ~gt).sum())})
    summary = {}
    for name, values in rows.items():
        metric = summarize(values)
        counts = {key: sum(row[key] for row in paired[name]) for key in paired[name][0]
                  if key != 'case_id'}
        summary[name] = {
            'threshold': points[name], 'metrics': metric, 'paired': counts,
            'gap_closure_rate_proxy': counts['closed_gaps'] / max(counts['eligible_gaps'], 1),
            'predicted_foreground_area': metric['tp'] + metric['fp']}
    return summary, rows, paired


def owner_bounds(positions, index, tile_size, image_size):
    start = positions[index]
    lower = (0 if index == 0 else
             round((positions[index - 1] + start + tile_size) / 2))
    upper = (image_size if index == len(positions) - 1 else
             round((start + positions[index + 1] + tile_size) / 2))
    return max(lower - start, 0), min(upper - start, tile_size)


def closure_gain_mask(p0, v6, v7, skeleton, gt):
    missing = skeleton & ~(cv2.dilate(p0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8)
    _, old_labels = cv2.connectedComponents(p0.astype(np.uint8), connectivity=8)
    _, labels6 = cv2.connectedComponents(v6.astype(np.uint8), connectivity=8)
    _, labels7 = cv2.connectedComponents(v7.astype(np.uint8), connectivity=8)
    gain = np.zeros_like(p0)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        x, y, width, height = (int(value) for value in stats[index, :4])
        x0, x1 = max(x - 7, 0), min(x + width + 7, p0.shape[1])
        y0, y1 = max(y - 7, 0), min(y + height + 7, p0.shape[0])
        region = labels[y0:y1, x0:x1] == index
        flank = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        old, road = old_labels[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        sources = np.unique(old[flank & road])
        sources = sources[sources > 0]
        if len(sources) < 2:
            continue

        def closed(pred, new_labels):
            if pred[y0:y1, x0:x1][region].mean() < .8:
                return False
            local = new_labels[y0:y1, x0:x1]
            connected = [set(np.unique(local[flank & (old == label) & road])) - {0}
                         for label in sources]
            return bool(set.intersection(*connected))

        if closed(v7, labels7) and not closed(v6, labels6):
            gain[y0:y1, x0:x1] |= region
    return gain


def bridge_mask(p0, pred, gt):
    excluded = cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    new_background = pred & ~p0 & ~excluded
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        new_background.astype(np.uint8), connectivity=8)
    _, old_labels = cv2.connectedComponents(p0.astype(np.uint8), connectivity=8)
    result = np.zeros_like(pred)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        x, y, width, height = (int(value) for value in stats[index, :4])
        x0, x1 = max(x - 2, 0), min(x + width + 2, pred.shape[1])
        y0, y1 = max(y - 2, 0), min(y + height + 2, pred.shape[0])
        local = labels[y0:y1, x0:x1] == index
        near = cv2.dilate(local.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if (np.unique(old_labels[y0:y1, x0:x1][near]) > 0).sum() >= 2:
            result[y0:y1, x0:x1] |= local
    return result


def diagnostic_maps(result):
    scalar = ('g_transport', 'c_bridge', 'gate_source_evidence',
              'path_continuity_evidence', 'r', 'b', 'a', 'u0')
    stepped = ('source_available', 'transport_delta', 'proposal_delta')
    maps = {}
    for name in (*scalar, 'w', *stepped):
        value = result[name]
        if name in stepped:
            value = value[:, :, 0]
        maps[name] = F.interpolate(value.float(), size=(512, 512),
                                   mode='bilinear', align_corners=False)[0].cpu().numpy()
    return maps


def region_values(m6, m7, y, x, record):
    values = {}
    for label, maps in (('v6', m6), ('v7', m7)):
        for name in ('g_transport', 'c_bridge', 'gate_source_evidence',
                     'path_continuity_evidence', 'r', 'b', 'a'):
            values[f'{name}_{label}'] = maps[name][0, y, x]
        values[f'w_direction_{label}'] = maps['w'][:, y, x].argmax(axis=0)
        values[f'w_max_{label}'] = maps['w'][:, y, x].max(axis=0)
        for direction in range(8):
            values[f'w{direction}_{label}'] = maps['w'][direction, y, x]
        values[f'b_minus_p0_{label}'] = maps['b'][0, y, x] - maps['u0'][0, y, x]
        for step in range(3):
            for name in ('source_available', 'transport_delta', 'proposal_delta'):
                values[f'{name}{step + 1}_{label}'] = maps[name][step, y, x]
        values[f'transport_total_{label}'] = maps['transport_delta'][:, y, x].sum(axis=0)
        values[f'proposal_total_{label}'] = maps['proposal_delta'][:, y, x].sum(axis=0)
    values['delta_g'] = values['g_transport_v7'] - values['g_transport_v6']
    values['p1_minus_p0_v6'] = record['v6'][y, x] - record['p0'][y, x]
    values['p1_minus_p0_v7'] = record['v7'][y, x] - record['p0'][y, x]
    return values


@torch.no_grad()
def diagnose_positions(models, dataset, records, points, device, samples_per_region):
    accum = defaultdict(lambda: defaultdict(list))
    sampled_rows = []
    positions = sliding_positions(1500, 512, 256)
    rng = np.random.default_rng(1234)
    for record in records:
        sample = dataset[record['index']]
        image = sample['image'][None].to(device)
        prepared = prepare_target(record['gt'], PROTOCOL)
        p0 = record['p0'] >= points['v6_reference']
        for comparison in ('v7_match_recall', 'v7_match_area'):
            p6 = record['v6'] >= points['v6_reference']
            p7 = record['v7'] >= points[comparison]
            bridge6 = bridge_mask(p0, p6, record['gt'])
            bridge7 = bridge_mask(p0, p7, record['gt'])
            masks = {
                'v7_only_gt_road': p7 & ~p6 & record['gt'],
                'v7_only_p0_missing_gt': p7 & ~p6 & ~p0 & record['gt'],
                'v7_only_background': p7 & ~p6 & ~record['gt'],
                'v7_closed_gap_center': closure_gain_mask(
                    p0, p6, p7, prepared['skeleton'], record['gt']),
                'v7_only_bridge_proxy': bridge7 & ~bridge6 & p7 & ~p6,
            }
            record.setdefault('regions', {})[comparison] = masks
        image_samples = defaultdict(list)
        for iy, top in enumerate(positions):
            for ix, left in enumerate(positions):
                owned_y = owner_bounds(positions, iy, 512, 1500)
                owned_x = owner_bounds(positions, ix, 512, 1500)
                full = (slice(top, top + 512), slice(left, left + 512))
                masks = {}
                for comparison, regions in record['regions'].items():
                    for name, full_mask in regions.items():
                        mask = full_mask[full].copy()
                        mask[:owned_y[0]] = False
                        mask[owned_y[1]:] = False
                        mask[:, :owned_x[0]] = False
                        mask[:, owned_x[1]:] = False
                        if mask.any():
                            masks[(comparison, name)] = mask
                if not masks:
                    continue
                tile = image[:, :, top:top + 512, left:left + 512]
                m6 = diagnostic_maps(models['v6'](tile, capture_diagnostics=True))
                m7 = diagnostic_maps(models['v7'](tile, capture_diagnostics=True))
                local_record = {'v6': record['v6'][full], 'v7': record['v7'][full],
                                'p0': record['p0'][full]}
                for key, mask in masks.items():
                    y, x = np.nonzero(mask)
                    values = region_values(m6, m7, y, x, local_record)
                    for field, value in values.items():
                        accum[key][field].append(np.asarray(value))
                    take = rng.choice(len(y), size=min(16, len(y)), replace=False)
                    for k in take:
                        image_samples[key].append({
                            'case': record['case'], 'comparison': key[0], 'region': key[1],
                            'y': int(top + y[k]), 'x': int(left + x[k]),
                            **{field: float(value[k]) for field, value in values.items()}})
        for key, rows in image_samples.items():
            take = rng.choice(len(rows), size=min(samples_per_region, len(rows)), replace=False)
            sampled_rows.extend(rows[int(index)] for index in take)
        del record['regions']
        print(f'[POSITIONS] {record["case"]}', flush=True)
    summaries = []
    delta_g_by_region = {}
    for (comparison, region), fields in sorted(accum.items()):
        count = len(np.concatenate(fields['delta_g']))
        row = {'comparison': comparison, 'region': region, 'pixels': count}
        for field, chunks in fields.items():
            values = np.concatenate(chunks)
            row[f'{field}_mean'] = float(values.mean())
            row[f'{field}_p10'] = float(np.quantile(values, .1))
            row[f'{field}_p50'] = float(np.quantile(values, .5))
            row[f'{field}_p90'] = float(np.quantile(values, .9))
            if field == 'delta_g':
                row['delta_g_positive_fraction'] = float((values > 0).mean())
                delta_g_by_region[(comparison, region)] = values
        summaries.append(row)
    return summaries, sampled_rows, delta_g_by_region


def delta_g_auc(positive, negative):
    if positive is None or negative is None or not len(positive) or not len(negative):
        return None
    values = np.concatenate((positive, negative))
    labels = np.r_[np.ones(len(positive), dtype=bool), np.zeros(len(negative), dtype=bool)]
    order = np.argsort(values, kind='mergesort')
    values, labels = values[order], labels[order]
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[start:end] = (start + end + 1) / 2.0
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def markdown_report(split, points, residuals, summary, region_rows, auc_rows):
    lines = [f'# v6/v7 matched comparison ({split})', '',
             'The reference is v6 at the predeclared fixed comparison threshold. '
             'Recall and foreground area are matched separately; the joint grid point '
             'is a compromise, not an exact simultaneous match.', '',
             '| Point | Threshold | Recall delta | Area delta | IoU | clDice | '
             'GT->pred APLS | pred->GT APLS | bidirectional APLS | Closed gaps | '
             'Bridge proxy |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    anchor = summary['v6_reference']['metrics']
    anchor_area = summary['v6_reference']['predicted_foreground_area']
    for name, item in summary.items():
        metric, paired = item['metrics'], item['paired']
        lines.append(f'| {name} | {item["threshold"]:.6f} | '
                     f'{metric["recall"] - anchor["recall"]:+.6f} | '
                     f'{item["predicted_foreground_area"] - anchor_area:+d} | '
                     f'{metric["iou"]:.4f} | {metric["cldice"]:.4f} | '
                     f'{metric["apls_gt_to_pred_approx"]:.4f} | '
                     f'{metric["apls_pred_to_gt_approx"]:.4f} | '
                     f'{metric["apls_bidirectional_approx"]:.4f} | '
                     f'{paired["closed_gaps"]}/{paired["eligible_gaps"]} | '
                     f'{paired["background_bridge_proxy"]} |')
    if residuals:
        lines += ['', 'Validation matching residuals (pixels):']
        lines.extend(f'- {name}: TP {value["tp_difference"]:+d}; area '
                     f'{value["foreground_area_difference"]:+d}'
                     for name, value in residuals.items())
    if region_rows:
        lines += ['', '| Comparison | Region | Pixels | Delta G mean | Delta G p50 | '
                  'Delta G positive |', '| --- | --- | ---: | ---: | ---: | ---: |']
        lines.extend(f'| {row["comparison"]} | {row["region"]} | {row["pixels"]} | '
                     f'{row["delta_g_mean"]:+.6f} | {row["delta_g_p50"]:+.6f} | '
                     f'{row["delta_g_positive_fraction"]:.3f} |' for row in region_rows)
        lines += ['', 'Delta G discrimination (v7-only P0-missed GT road versus '
                  'v7-only background):']
        lines.extend(f'- {name}: AUC {value:.4f}' if value is not None else
                     f'- {name}: undefined (one region has no pixels)'
                     for name, value in auc_rows.items())
    lines += ['', 'Gap closure and wrong bridging are raster proxies, not confirmed '
              'road graph edits. Internal values come from one owner tile per pixel; '
              'P1 decisions use all overlapping tiles. Differences between the two '
              'trained checkpoints include parameter updates as well as the G formula.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('val', 'test'))
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--v6_checkpoint', type=Path, required=True)
    parser.add_argument('--v7_checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path, help='Validation selection JSON, required for test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reference_threshold', type=float, default=.55)
    parser.add_argument('--grid_step', type=float, default=.001)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--samples_per_region', type=int, default=250)
    args = parser.parse_args()
    if not (0 < args.reference_threshold < 1 and 0 < args.grid_step <= .01):
        parser.error('Reference threshold or grid step is invalid')
    if args.max_images < 0 or args.samples_per_region < 0:
        parser.error('Image and sampling limits must be nonnegative')
    if args.split == 'test' and not args.selection:
        parser.error('--selection from val is required for test')
    if not args.root_path.is_dir():
        parser.error(f'Dataset not found: {args.root_path}')
    for checkpoint in (args.v6_checkpoint, args.v7_checkpoint):
        if not checkpoint.is_file():
            parser.error(f'Checkpoint not found: {checkpoint}')
    from run_reliable_directional_mass import V6_VERSION, V7_VERSION
    device = torch.device(args.device)
    models = {}
    models['v6'], inputs6 = load_dynamic(args.v6_checkpoint, device, V6_VERSION)
    models['v7'], inputs7 = load_dynamic(args.v7_checkpoint, device, V7_VERSION)
    if inputs6 != inputs7:
        raise ValueError('The v6 and v7 checkpoints must use the same base and Q3 checkpoints')
    dataset = full_dataset(args.root_path, args.split)
    limit = min(len(dataset), args.max_images or len(dataset))
    if args.split == 'val' and args.max_images:
        raise ValueError('Do not choose thresholds on a val subset; omit --max_images')
    if limit < 1:
        raise ValueError('No images available for comparison')
    records = predictions(models, dataset, device, limit)
    if args.split == 'val':
        points, residuals, scan = choose_thresholds(
            records, args.reference_threshold, args.grid_step)
        selection = {'selected_on': 'val', 'reference_threshold': args.reference_threshold,
                     'grid_step': args.grid_step, 'images': limit,
                     'v6_checkpoint': str(args.v6_checkpoint.resolve()),
                     'v7_checkpoint': str(args.v7_checkpoint.resolve()),
                     'case_names': [record['case'] for record in records],
                     'points': points, 'matching_residuals_val': residuals}
    else:
        selection = json.loads(args.selection.read_text(encoding='utf-8'))
        if (selection.get('selected_on') != 'val' or
                selection['v6_checkpoint'] != str(args.v6_checkpoint.resolve()) or
                selection['v7_checkpoint'] != str(args.v7_checkpoint.resolve())):
            raise ValueError('The selection file does not match these checkpoints')
        points, residuals, scan = selection['points'], None, None
    summary, rows, paired = evaluate_points(records, points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.split == 'val':
        (args.output_dir / 'selection.json').write_text(
            json.dumps(selection, indent=2), encoding='utf-8')
        write_csv(args.output_dir / 'fine_threshold_scan.csv', scan)
    region_rows, samples, auc_rows = [], [], {}
    if args.split == 'val':
        region_rows, samples, delta_g = diagnose_positions(
            models, dataset, records, points, device, args.samples_per_region)
        for name in ('v7_match_recall', 'v7_match_area'):
            auc_rows[name] = delta_g_auc(
                delta_g.get((name, 'v7_only_p0_missing_gt')),
                delta_g.get((name, 'v7_only_background')))
        write_csv(args.output_dir / 'region_summary.csv', region_rows)
        write_csv(args.output_dir / 'sampled_positions.csv', samples)
    report = {'split': args.split, 'images': limit, 'selection': selection,
              'summary': summary, 'region_summary': region_rows,
              'delta_g_auc_gt_vs_background': auc_rows,
              'metric_note': 'Macro topology and approximate raster APLS use the same road_metrics implementation.'}
    (args.output_dir / f'{args.split}_comparison.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    (args.output_dir / f'{args.split}_comparison.md').write_text(
        markdown_report(args.split, points, residuals, summary, region_rows, auc_rows),
        encoding='utf-8')
    for name in points:
        write_csv(args.output_dir / f'{args.split}_{name}_per_image.csv', rows[name])
        write_csv(args.output_dir / f'{args.split}_{name}_paired.csv', paired[name])
    print(f'[DONE] {args.output_dir / f"{args.split}_comparison.md"}', flush=True)


if __name__ == '__main__':
    main()
