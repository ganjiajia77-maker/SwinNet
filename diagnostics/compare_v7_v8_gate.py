"""Paired native-image v7/v8 evaluation with validation-only matched thresholds."""

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
    bridge_mask, choose_thresholds, closure_gain_mask, evaluate_points,
    probability, write_csv,
)
from evaluation.road_comparison.road_metrics import prepare_target  # noqa: E402
from mass_protocol import blend_weight, sliding_positions  # noqa: E402
from run_q3_r3_mass import PROTOCOL, full_dataset  # noqa: E402
from run_reliable_v8_mass import load_experiment  # noqa: E402


def load_pair(args):
    config = SimpleNamespace(v7_checkpoint=args.v7_checkpoint,
                             v8_checkpoint=args.v8_checkpoint,
                             relation_mode=args.relation_mode, device=args.device)
    model, _ = load_experiment(config)
    model.eval()
    return model


@torch.no_grad()
def stitch_pair(model, image):
    if tuple(image.shape) != (1, 3, 1500, 1500):
        raise ValueError(f'Expected a native 1500 image, got {tuple(image.shape)}')
    names = ('p0', 'v7', 'v8', 'g7', 'g8', 'q')
    canvases = {name: torch.zeros((1, 1, 1500, 1500), device=image.device)
                for name in names}
    mass = torch.zeros_like(canvases['p0'])
    weight = blend_weight(512, image.device)
    saved = model.gate_relation
    for top in sliding_positions(1500, 512, 256):
        for left in sliding_positions(1500, 512, 256):
            tile = image[:, :, top:top+512, left:left+512]
            v8 = model(tile, capture_diagnostics=True)
            model.gate_relation = None
            try:
                v7 = model(tile)
            finally:
                model.gate_relation = saved
            values = {'p0': v8['z0'], 'v7': v7['z1'], 'v8': v8['z1'],
                      'g7': v7['g_transport'], 'g8': v8['g_transport'],
                      'q': v8['q']}
            region = (slice(None), slice(None), slice(top, top+512), slice(left, left+512))
            for name, value in values.items():
                if value.shape[-2:] != (512, 512):
                    value = F.interpolate(value, size=(512, 512), mode='bilinear',
                                          align_corners=False)
                canvases[name][region] += value.float() * weight
            mass[region] += weight
    if bool((mass <= 0).any()):
        raise RuntimeError('Uncovered pixels in sliding-window reconstruction')
    return {name: (canvas / mass).cpu().numpy()[0, 0]
            for name, canvas in canvases.items()}


def region_summary(records, thresholds):
    rows = []
    v7_t, v8_t = thresholds['v7_reference'], thresholds['v8_val_selected']
    for record in records:
        gt = record['gt']
        prepared = prepare_target(gt, PROTOCOL)
        p0 = record['p0'] >= v7_t
        old = record['v6'] >= v7_t
        new = record['v7'] >= v8_t
        skeleton = prepared['skeleton']
        missing = skeleton & ~(cv2.dilate(p0.astype(np.uint8),
                                            np.ones((5, 5), np.uint8)) > 0)
        gained_closure = closure_gain_mask(p0, old, new, skeleton, gt)
        extra_bridge = bridge_mask(p0, new, gt) & ~bridge_mask(p0, old, gt)
        groups = {
            'newly_closed_gap': gained_closure,
            'recovered_gt': missing & new & ~old,
            'unrecovered_gt': missing & ~new,
            'similar_background_new_fp': new & ~old & ~gt & (record['q'] > .5),
            'extra_false_bridge_proxy': extra_bridge,
        }
        for name, region in groups.items():
            if not region.any():
                continue
            old_g, new_g = record['g7'][region], record['g8'][region]
            delta = new_g - old_g
            rows.append({'case': record['case'], 'region': name, 'pixels': int(region.sum()),
                         'g7_mean': float(old_g.mean()), 'g8_mean': float(new_g.mean()),
                         'g7_p10': float(np.percentile(old_g, 10)),
                         'g7_p90': float(np.percentile(old_g, 90)),
                         'g8_p10': float(np.percentile(new_g, 10)),
                         'g8_p90': float(np.percentile(new_g, 90)),
                         'delta_g_mean': float(delta.mean()),
                         'delta_g_p10': float(np.percentile(delta, 10)),
                         'delta_g_p50': float(np.percentile(delta, 50)),
                         'delta_g_p90': float(np.percentile(delta, 90))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--v7_checkpoint', type=Path, required=True)
    parser.add_argument('--v8_checkpoint', type=Path, required=True)
    parser.add_argument('--v7_selection', type=Path, required=True)
    parser.add_argument('--v8_selection', type=Path, required=True)
    parser.add_argument('--relation_mode',
                        choices=('local', 'bilateral', 'directional', 'full'), required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--val_match_selection', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_images', type=int, default=0)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.split == 'test' and not args.val_match_selection:
        parser.error('Test requires --val_match_selection from a completed val comparison')
    selection7 = json.loads(args.v7_selection.read_text(encoding='utf-8'))
    selection8 = json.loads(args.v8_selection.read_text(encoding='utf-8'))
    if selection7.get('selected_on') != 'val' or selection8.get('selected_on') != 'val':
        raise ValueError('Both model thresholds must be selected on validation data')
    if selection7.get('dynamic_checkpoint') != str(args.v7_checkpoint.resolve()):
        raise ValueError('v7 selection does not match its checkpoint')
    if selection8.get('v8_checkpoint') != str(args.v8_checkpoint.resolve()):
        raise ValueError('v8 selection does not match its checkpoint')
    if selection8.get('v7_checkpoint') != str(args.v7_checkpoint.resolve()):
        raise ValueError('v8 is not initialized from this v7 checkpoint')
    model = load_pair(args)
    dataset = full_dataset(args.root_path, args.split)
    limit = min(len(dataset), args.max_images or len(dataset))
    records = []
    for index in range(limit):
        sample = dataset[index]
        outputs = stitch_pair(model, sample['image'][None].to(args.device))
        records.append({'case': sample['case_name'], 'gt': sample['mask'][0].numpy() > .5,
                        'p0': probability(outputs['p0']),
                        'v6': probability(outputs['v7']),
                        'v7': probability(outputs['v8']),
                        'g7': outputs['g7'], 'g8': outputs['g8'], 'q': outputs['q']})
        print(f'[PAIRED] {index+1}/{limit} {sample["case_name"]}', flush=True)
    reference = float(selection7['threshold'])
    current = float(selection8['threshold'])
    if args.split == 'val':
        raw_points, residuals, scan = choose_thresholds(records, reference, .001)
        points = {'v7_reference': raw_points['v6_reference'],
                  'v8_same_threshold': raw_points['v7_same_threshold'],
                  'v8_val_selected': current,
                  'v8_match_recall': raw_points['v7_match_recall'],
                  'v8_match_area': raw_points['v7_match_area'],
                  'v8_joint_grid': raw_points['v7_joint_grid']}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selection = {'split': 'val', 'points': points, 'residuals': residuals,
                     'v7_checkpoint': str(args.v7_checkpoint.resolve()),
                     'v8_checkpoint': str(args.v8_checkpoint.resolve()),
                     'relation_mode': args.relation_mode}
        (args.output_dir / 'matched_selection.json').write_text(
            json.dumps(selection, indent=2), encoding='utf-8')
        write_csv(args.output_dir / 'fine_threshold_scan.csv', scan)
    else:
        selection = json.loads(args.val_match_selection.read_text(encoding='utf-8'))
        expected = {'split': 'val',
                    'v7_checkpoint': str(args.v7_checkpoint.resolve()),
                    'v8_checkpoint': str(args.v8_checkpoint.resolve()),
                    'relation_mode': args.relation_mode}
        if any(selection.get(key) != value for key, value in expected.items()):
            raise ValueError('Matched thresholds do not belong to these checkpoints')
        points = selection['points']
    aliases = {'v6_reference': points['v7_reference'],
               **{name: threshold for name, threshold in points.items()
                  if name != 'v7_reference'}}
    summary, _, _ = evaluate_points(records, aliases)
    summary['v7_reference'] = summary.pop('v6_reference')
    regions = region_summary(records, points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f'{args.split}_gate_regions.csv', regions)
    (args.output_dir / f'{args.split}_comparison.json').write_text(
        json.dumps({'split': args.split, 'images': limit,
                    'threshold_source': 'validation', 'operating_points': summary,
                    'region_note': 'Background bridge is a raster proxy; closure uses P0 gaps.'},
                   indent=2), encoding='utf-8')
    lines = [f'# v7 versus v8 ({args.split})', '',
             f'Images: {limit}; relation mode: {args.relation_mode}.', '',
             '| Operating point | Threshold | IoU | F1 | Precision | Recall | clDice | '
             'GT->Pred APLS | Pred->GT APLS | Bidirectional APLS | Closed gaps | '
             'Bridge proxy | Short components | Added GT | Added background |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, entry in summary.items():
        metric, paired = entry['metrics'], entry['paired']
        lines.append(f'| {name} | {entry["threshold"]:.4f} | {metric["iou"]:.4f} | '
                     f'{metric["f1"]:.4f} | {metric["precision"]:.4f} | '
                     f'{metric["recall"]:.4f} | {metric["cldice"]:.4f} | '
                     f'{metric["apls_gt_to_pred_approx"]:.4f} | '
                     f'{metric["apls_pred_to_gt_approx"]:.4f} | '
                     f'{metric["apls_bidirectional_approx"]:.4f} | '
                     f'{paired["closed_gaps"]} | {paired["background_bridge_proxy"]} | '
                     f'{metric["short_pred_components"]:.2f} | '
                     f'{paired["new_gt_pixels_vs_p0"]} | '
                     f'{paired["new_background_pixels_vs_p0"]} |')
    lines += ['', 'Threshold matching was performed on val only. APLS and bridge counts are '
              'raster approximations, not official SpaceNet APLS.']
    (args.output_dir / f'{args.split}_report.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
