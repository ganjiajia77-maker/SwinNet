"""Analyze saved V7/V8 probability maps without loading any model checkpoint.

Manifest CSV columns: case_id,gt,p0,v7,v8. Optional p0_kind/v7_kind/v8_kind
columns accept `probability` or `binary`. Paths may be absolute or relative to
the manifest. NPZ paths can select an array with `archive.npz#key`.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.compare_v6_v7_matched import write_csv
from evaluation.road_comparison.road_metrics import (
    image_metrics, prepare_target, segmentation, summarize,
)
from run_q3_r3_mass import PROTOCOL
from run_reliable_directional_mass import gap_lengths, selection_score


GRID = tuple(round(float(x), 2) for x in np.arange(.35, .751, .01))


def _load_array(spec, base_dir):
    path_text, separator, key = str(spec).partition('#')
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        array = np.load(path, allow_pickle=False)
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            if separator:
                if key not in archive.files:
                    raise KeyError(f'{key!r} not in {path}; keys={archive.files}')
                array = archive[key].copy()
            elif len(archive.files) == 1:
                array = archive[archive.files[0]].copy()
            else:
                raise ValueError(f'{path} has multiple arrays; use path.npz#key')
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ValueError(f'Could not read image: {path}')
        if array.ndim == 3:
            if array.shape[2] == 4:
                array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
            else:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    array = np.asarray(array)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D map at {path}, got {array.shape}')
    return array


def _as_probability(array, kind, name):
    if kind == 'binary':
        return (array > 0).astype(np.float32)
    if kind != 'probability':
        raise ValueError(f'{name}_kind must be probability or binary, got {kind!r}')
    result = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer) and result.max(initial=0) > 1:
        result /= float(np.iinfo(array.dtype).max)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} contains NaN/Inf')
    if result.size and (result.min() < -1e-5 or result.max() > 1.00001):
        raise ValueError(f'{name} probability range is [{result.min()}, {result.max()}]')
    return np.clip(result, 0., 1.)


def _read_manifest(path):
    records = []
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'case_id', 'gt', 'p0', 'v7', 'v8'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'Manifest needs columns {sorted(required)}')
        for row in reader:
            case = row['case_id']
            loaded = {'case_id': case, 'kinds': {}}
            for name in ('gt', 'p0', 'v7', 'v8'):
                array = _load_array(row[name], path.parent)
                if name == 'gt':
                    if np.issubdtype(array.dtype, np.integer) and array.max(initial=0) > 1:
                        array = array.astype(np.float32) / np.iinfo(array.dtype).max
                    loaded[name] = array > .5
                else:
                    kind = row.get(f'{name}_kind', 'probability') or 'probability'
                    loaded[name] = _as_probability(array, kind, name)
                    loaded['kinds'][name] = kind
            shapes = {name: value.shape for name, value in loaded.items()
                      if name not in ('case_id', 'kinds')}
            if len(set(shapes.values())) != 1:
                raise ValueError(f'{case}: map shapes differ: {shapes}')
            records.append(loaded)
    if not records:
        raise ValueError('Prediction manifest is empty')
    return records


def _count_at(records, name, threshold):
    tp = area = gt_count = 0
    for row in records:
        gt = row['gt']
        pred = row[name] >= threshold
        tp += int((pred & gt).sum())
        area += int(pred.sum())
        gt_count += int(gt.sum())
    fp, fn = area - tp, gt_count - tp
    return {'tp': tp, 'fp': fp, 'fn': fn, 'area': area, 'gt': gt_count,
            'recall': tp / max(gt_count, 1), 'iou': tp / max(tp + fp + fn, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1)}


def _matched_v8_threshold(records, reference_threshold):
    target = _count_at(records, 'v7', reference_threshold)
    candidates = []
    road_scores = np.concatenate([row['v8'][row['gt']] for row in records])
    all_scores = np.concatenate([row['v8'].ravel() for row in records])
    for threshold in GRID:
        current = _count_at(records, 'v8', threshold)
        candidates.append((abs(current['tp'] - target['tp']) / max(target['gt'], 1) +
                           abs(current['area'] - target['area']) / max(target['area'], 1),
                           threshold, current))
    return target, min(candidates, key=lambda item: (item[0], item[1])), {
        'v8_road_score_min': float(road_scores.min()),
        'v8_road_score_max': float(road_scores.max()),
        'v8_all_score_min': float(all_scores.min()),
        'v8_all_score_max': float(all_scores.max())}


def _evaluate(records, model_name, threshold, reference_threshold):
    metric_rows, per_image = [], []
    paired = {'eligible_gaps': 0, 'closed_gaps': 0, 'bridge_components': 0,
              'bridge_pixels': 0, 'isolated_added_fp_components': 0,
              'isolated_added_fp_pixels': 0, 'attached_added_fp_components': 0,
              'attached_added_fp_pixels': 0,
              'added_gt_pixels_vs_p0': 0, 'added_background_pixels_vs_p0': 0,
              'predicted_foreground_pixels': 0}
    bins_total = {name: {'eligible': 0, 'closed': 0}
                  for name in ('3-7', '8-15', '16-31', '32+')}
    for row in records:
        gt = row['gt']
        prepared = prepare_target(gt, PROTOCOL)
        p0 = row['p0'] >= reference_threshold
        prediction = row[model_name] >= threshold
        metric = {'case_id': row['case_id'],
                  **image_metrics(prediction, prepared, PROTOCOL)}
        pixel_metrics = segmentation(metric)
        bins, bridges = gap_lengths(p0, prediction, prepared['skeleton'], gt)
        eligible = sum(value['eligible'] for value in bins.values())
        closed = sum(value['closed'] for value in bins.values())
        added_gt = int((prediction & ~p0 & gt).sum())
        added_bg = int((prediction & ~p0 & ~gt).sum())
        outside_gt = cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
        added_bg_map = prediction & ~p0 & outside_gt
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            added_bg_map.astype(np.uint8), connectivity=8)
        _, p0_labels = cv2.connectedComponents(p0.astype(np.uint8), connectivity=8)
        bridge_pixels = isolated_pixels = attached_pixels = 0
        bridge_count = isolated_count = attached_count = 0
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < 3:
                continue
            x, y, width, height = (int(value) for value in stats[component, :4])
            x0, x1 = max(x - 2, 0), min(x + width + 2, gt.shape[1])
            y0, y1 = max(y - 2, 0), min(y + height + 2, gt.shape[0])
            local = labels[y0:y1, x0:x1] == component
            near = cv2.dilate(local.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            sources = np.unique(p0_labels[y0:y1, x0:x1][near])
            sources = sources[sources > 0]
            if len(sources) >= 2:
                bridge_count += 1
                bridge_pixels += area
            elif len(sources) == 1:
                attached_count += 1
                attached_pixels += area
            else:
                isolated_count += 1
                isolated_pixels += area
        for name, values in bins.items():
            bins_total[name]['eligible'] += values['eligible']
            bins_total[name]['closed'] += values['closed']
        per_image.append({'case_id': row['case_id'], 'threshold': threshold,
                          'tp': metric['tp'], 'fp': metric['fp'], 'fn': metric['fn'],
                          'iou': pixel_metrics['iou'], 'f1': pixel_metrics['f1'],
                          'cldice': metric['cldice'],
                          'apls_gt_to_pred_approx': metric['apls_gt_to_pred_approx'],
                          'apls_pred_to_gt_approx': metric['apls_pred_to_gt_approx'],
                          'apls_bidirectional_approx': metric['apls_bidirectional_approx'],
                          'eligible_p0_gaps': eligible, 'closed_p0_gaps': closed,
                          'background_bridge_components': bridges,
                          'background_bridge_pixels': bridge_pixels,
                          'isolated_added_fp_components': isolated_count,
                          'isolated_added_fp_pixels': isolated_pixels,
                          'attached_added_fp_components': attached_count,
                          'attached_added_fp_pixels': attached_pixels,
                          'added_gt_pixels_vs_p0': added_gt,
                          'added_background_pixels_vs_p0': added_bg})
        metric_rows.append(metric)
        paired['eligible_gaps'] += eligible
        paired['closed_gaps'] += closed
        paired['bridge_components'] += bridges
        paired['bridge_pixels'] += bridge_pixels
        paired['isolated_added_fp_components'] += isolated_count
        paired['isolated_added_fp_pixels'] += isolated_pixels
        paired['attached_added_fp_components'] += attached_count
        paired['attached_added_fp_pixels'] += attached_pixels
        paired['added_gt_pixels_vs_p0'] += added_gt
        paired['added_background_pixels_vs_p0'] += added_bg
        paired['predicted_foreground_pixels'] += int(prediction.sum())
    summary = summarize(metric_rows)
    paired['closure_rate_proxy'] = paired['closed_gaps'] / max(paired['eligible_gaps'], 1)
    summary['selection_score'] = selection_score(
        summary, {'background_bridge_proxy': paired['bridge_components']}, len(records))
    summary['paired_diagnostics'] = paired
    return summary, per_image, bins_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--v7_threshold', type=float, required=True,
                        help='Validation-selected V7 threshold used as the recall/area reference')
    parser.add_argument('--v8_threshold', type=float, required=True,
                        help='Validation-selected V8 threshold, reported alongside matched point')
    parser.add_argument('--p0_reference_threshold', type=float,
                        help='Threshold for P0 reference gaps; defaults to v7_threshold')
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f'Prediction manifest does not exist: {args.manifest}')
    records = _read_manifest(args.manifest.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    p0_threshold = (args.p0_reference_threshold if args.p0_reference_threshold is not None
                    else args.v7_threshold)
    target = _count_at(records, 'v7', args.v7_threshold)
    v8_is_probability = all(row['kinds']['v8'] == 'probability' for row in records)
    if v8_is_probability:
        target, matched, ranges = _matched_v8_threshold(records, args.v7_threshold)
        matched_threshold = matched[1]
        matched_info = {'threshold': matched_threshold,
                        'joint_normalized_error': matched[0], 'counts': matched[2],
                        'score_ranges': ranges}
    else:
        matched_threshold = .5
        matched_info = {'threshold': None, 'available': False,
                        'reason': 'V8 maps are binary; threshold matching cannot be performed'}
    points = [
        ('v7_reference', 'v7', args.v7_threshold),
        ('v8_selected', 'v8', args.v8_threshold),
    ]
    if v8_is_probability:
        points.append(('v8_matched_recall_area_grid', 'v8', matched_threshold))
    reports, per_image_rows = {}, []
    for label, model_name, threshold in points:
        report, rows, bins = _evaluate(records, model_name, threshold, p0_threshold)
        report['gap_by_length'] = bins
        reports[label] = {'model': model_name, 'threshold': threshold, **report}
        per_image_rows.extend({'operating_point': label, **row} for row in rows)
    if v8_is_probability:
        scan = []
        for threshold in GRID:
            counts = _count_at(records, 'v8', threshold)
            scan.append({'threshold': threshold, **counts,
                         'tp_difference_vs_v7_reference': counts['tp'] - target['tp'],
                         'foreground_difference_vs_v7_reference': counts['area'] - target['area']})
        write_csv(args.output_dir / 'v8_cache_v8_threshold_scan.csv', scan)
    write_csv(args.output_dir / 'v8_cache_per_image.csv', per_image_rows)
    result = {
        'source': 'saved prediction maps only; no model or checkpoint loaded',
        'manifest': str(args.manifest.resolve()), 'images': len(records),
        'p0_reference_threshold': p0_threshold,
        'v7_reference_counts': target,
        'v8_matched_point': matched_info,
        'operating_points': reports,
        'metric_note': ('IoU/F1 use aggregated TP/FP/FN through summarize; topology metrics '
                        'are means across images. Gap closure and background bridges are '
                        'the existing diagnostic proxies, not official graph APLS.')}
    (args.output_dir / 'v8_cache_comparison.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
