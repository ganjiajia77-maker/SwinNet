"""Paired Q3 localization and R3 counterfactual audit on native Mass images."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from evaluation.road_comparison.road_metrics import image_metrics, prepare_target, summarize
from mass_protocol import blend_weight, sliding_positions
from networks.q3_r3_repair import make_q_target
from run_q3_r3_mass import PROTOCOL, full_dataset, load_experiment, reference_gaps


VARIANTS = ('p0', 'predicted_q', 'constant_q', 'oracle_q')
LOGIT_THRESHOLD = lambda threshold: float(np.log(threshold / (1. - threshold)))


def sigmoid(logits):
    return 1. / (1. + np.exp(-np.clip(logits, -80., 80.)))


def length_bin(length):
    if length < 8:
        return '3-7'
    if length < 16:
        return '8-15'
    if length < 32:
        return '16-31'
    return '32+'


@torch.no_grad()
def stitch_counterfactuals(model, image, skeleton, threshold):
    if tuple(image.shape) != (1, 3, 1500, 1500) or skeleton.shape != (1500, 1500):
        raise ValueError('Expected a native 1500 image and matched GT skeleton')
    weight = blend_weight(512, image.device)
    canvases = {key: torch.zeros((1, 1, 1500, 1500), device=image.device)
                for key in (*VARIANTS, 'q')}
    divisor = torch.zeros_like(canvases['q'])
    skeleton = torch.from_numpy(np.ascontiguousarray(skeleton[None, None])).to(
        image.device, dtype=image.dtype)
    for top in sliding_positions(1500, 512, 256):
        for left in sliding_positions(1500, 512, 256):
            result = model(image[:, :, top:top+512, left:left+512], return_aux=True)
            oracle = make_q_target(skeleton[:, :, top:top+512, left:left+512],
                                   result['z0'], result['q'].shape[-2:], threshold=threshold)
            constant = torch.full_like(result['q'], 0.5)
            z_constant, _ = model.refine_with_q(result['z0'], result['q_features'],
                                                 result['p0_q'], constant)
            z_oracle, _ = model.refine_with_q(result['z0'], result['q_features'],
                                               result['p0_q'], oracle)
            q = F.interpolate(result['q'], size=(512, 512), mode='bilinear',
                              align_corners=False)
            region = (slice(None), slice(None), slice(top, top+512), slice(left, left+512))
            tiles = dict(p0=result['z0'], predicted_q=result['z1'], constant_q=z_constant,
                         oracle_q=z_oracle, q=q)
            for name, tile in tiles.items():
                canvases[name][region] += tile.float() * weight
            divisor[region] += weight
    if bool((divisor <= 0).any()):
        raise RuntimeError('Sliding inference left uncovered pixels')
    output = {name: (canvas / divisor).cpu().numpy()[0, 0]
              for name, canvas in canvases.items()}
    if not all(np.isfinite(value).all() for value in output.values()):
        raise RuntimeError('Counterfactual inference produced nonfinite values')
    return output


def closure_proxy(region, flank, flank_labels, p0_labels, gt, pred, pred_labels):
    if pred[region].mean() < 0.8:
        return False
    linked = [set(np.unique(pred_labels[flank & (p0_labels == label) & gt])) - {0}
              for label in flank_labels]
    return bool(set.intersection(*linked))


def gap_rows(case_id, p0, gt, gt_skeleton, logits, masks, q, threshold):
    labels, p0_labels, gaps = reference_gaps(p0, gt_skeleton, gt)
    pred_labels = {name: cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[1]
                   for name, mask in masks.items() if name != 'p0'}
    distance = cv2.distanceTransform((~p0).astype(np.uint8), cv2.DIST_L2, 3)
    rows = []
    threshold_logit = LOGIT_THRESHOLD(threshold)
    for component, length, flank_labels in gaps:
        region = labels == component
        center = region & (distance >= np.percentile(distance[region], 75))
        flank = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        side_scores = [float((q[flank & (p0_labels == label) & gt] >= 0.5).mean())
                       for label in flank_labels]
        center_high = float((q[center] >= 0.5).mean())
        if center_high >= 0.5:
            status = 'center_high'
        elif all(score >= 0.5 for score in side_scores):
            status = 'sides_only'
        elif float(q[region].max()) < 0.5:
            status = 'whole_gap_low'
        else:
            status = 'mixed_or_one_side'
        margin = threshold_logit - logits['p0'][center]
        row = {'case_id': case_id, 'component_id': component, 'length_pixels': length,
               'length_bin': length_bin(length), 'q_status': status,
               'center_pixels': int(center.sum()), 'q_center_mean': float(q[center].mean()),
               'q_center_max': float(q[center].max()), 'q_center_high_fraction': center_high,
               'q_gap_high_fraction': float((q[region] >= 0.5).mean()),
               'q_side_min_high_fraction': min(side_scores),
               'q_side_mean_high_fraction': float(np.mean(side_scores)),
               'p0_center_logit_mean': float(logits['p0'][center].mean()),
               'required_logit_margin_mean': float(margin.mean()),
               'required_logit_margin_median': float(np.median(margin))}
        for name in VARIANTS[1:]:
            delta = logits[name][center] - logits['p0'][center]
            crossable = delta >= margin
            increased = delta > 0
            row[f'{name}_delta_logit_center_mean'] = float(delta.mean())
            row[f'{name}_delta_logit_center_positive_fraction'] = float((delta > 0).mean())
            row[f'{name}_cross_threshold_fraction'] = float(crossable.mean())
            row[f'{name}_raised_but_below_threshold_fraction'] = float(
                (increased & ~crossable).mean())
            row[f'{name}_not_raised_fraction'] = float((~increased).mean())
            row[f'{name}_center_recovered_fraction'] = float(masks[name][center].mean())
            row[f'{name}_gap_recovered_fraction'] = float(masks[name][region].mean())
            row[f'{name}_closed'] = int(closure_proxy(region, flank, flank_labels, p0_labels,
                                                       gt, masks[name], pred_labels[name]))
        rows.append(row)
    return rows


def q_region_stats(q, gt, gt_skeleton, p0):
    missing = gt_skeleton & ~(cv2.dilate(p0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    away_missing = ~cv2.dilate(missing.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    edge = gt & ~cv2.erode(gt.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    normal_edge = edge & away_missing
    background = ~cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return {'normal_edge_pixels': int(normal_edge.sum()),
            'normal_edge_q_high_pixels': int((normal_edge & (q >= 0.5)).sum()),
            'background_pixels': int(background.sum()),
            'background_q_high_pixels': int((background & (q >= 0.5)).sum())}


def aggregate_gaps(rows):
    result = {}
    for field in ('q_status', 'length_bin'):
        result[field] = {}
        for key in sorted({row[field] for row in rows}):
            group = [row for row in rows if row[field] == key]
            entry = {'gaps': len(group),
                     'q_center_high_fraction': float(np.mean([r['q_center_high_fraction'] for r in group])),
                     'required_logit_margin_mean': float(np.mean([r['required_logit_margin_mean'] for r in group]))}
            for name in VARIANTS[1:]:
                entry[name] = {'closed': sum(r[f'{name}_closed'] for r in group),
                               'mean_delta_logit_center': float(np.mean([
                                   r[f'{name}_delta_logit_center_mean'] for r in group])),
                               'mean_cross_threshold_fraction': float(np.mean([
                                   r[f'{name}_cross_threshold_fraction'] for r in group])),
                               'mean_raised_but_below_threshold_fraction': float(np.mean([
                                   r[f'{name}_raised_but_below_threshold_fraction'] for r in group])),
                               'mean_not_raised_fraction': float(np.mean([
                                   r[f'{name}_not_raised_fraction'] for r in group])),
                               'mean_center_recovered_fraction': float(np.mean([
                                   r[f'{name}_center_recovered_fraction'] for r in group]))}
            result[field][key] = entry
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(args):
    selection = json.loads((args.run_dir / 'val_selection.json').read_text(encoding='utf-8'))
    if selection['selected_on'] != 'val' or Path(selection['repair_checkpoint']).resolve() != args.repair_checkpoint.resolve():
        raise ValueError('Use the validation-selected threshold from this repair checkpoint')
    threshold = float(selection['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Invalid validation threshold')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = load_experiment(args).eval()
    dataset = full_dataset(args.root_path, args.split)
    limit = min(len(dataset), args.max_images or len(dataset))
    metrics = {name: [] for name in VARIANTS}
    all_gaps, image_rows = [], []
    for index in range(limit):
        sample = dataset[index]
        gt = sample['mask'][0].numpy() > 0.5
        prepared = prepare_target(gt, PROTOCOL)
        logits = stitch_counterfactuals(model, sample['image'][None].to(args.device),
                                         prepared['skeleton'], threshold)
        q = logits.pop('q')
        masks = {name: sigmoid(value) >= threshold for name, value in logits.items()}
        for name in VARIANTS:
            metrics[name].append(image_metrics(masks[name], prepared, PROTOCOL))
        gaps = gap_rows(sample['case_name'], masks['p0'], gt, prepared['skeleton'],
                        logits, masks, q, threshold)
        all_gaps.extend(gaps)
        image_rows.append({'case_id': sample['case_name'], 'reference_gaps': len(gaps),
                           'q_high_area_fraction': float((q >= 0.5).mean()),
                           **q_region_stats(q, gt, prepared['skeleton'], masks['p0'])})
        print(f'[DIAG] {index+1}/{limit} {sample["case_name"]} gaps={len(gaps)}', flush=True)
    background = sum(row['background_pixels'] for row in image_rows)
    edge = sum(row['normal_edge_pixels'] for row in image_rows)
    report = {'split': args.split, 'images': limit, 'threshold': threshold,
              'reference_gaps': len(all_gaps), 'base_checkpoint': str(args.base_checkpoint),
              'repair_checkpoint': str(args.repair_checkpoint),
              'metrics': {name: summarize(rows) for name, rows in metrics.items()},
              'gap_strata': aggregate_gaps(all_gaps),
              'q_normal_road_edge_high_fraction': sum(
                  row['normal_edge_q_high_pixels'] for row in image_rows) / max(edge, 1),
              'q_nonroad_background_high_fraction': sum(
                  row['background_q_high_pixels'] for row in image_rows) / max(background, 1),
              'q_high_area_fraction_mean': float(np.mean([
                  row['q_high_area_fraction'] for row in image_rows])),
              'definitions': {
                  'reference_gap': 'Same P0 two-flank component proxy as run_q3_r3_mass.py',
                  'center': 'Farthest quartile of gap pixels from P0 foreground',
                  'q_high': 'Stitched Q >= 0.5; center_high requires >=50% center coverage',
                  'constant_q': 'Q replaced by 0.5 in R3 input and gate; Q features unchanged',
                  'oracle_q': 'GT skeleton missed by each tile P0 at validation threshold, '
                              'dilated and pooled using the training Q-target rule; diagnostic only',
                  'closure': '>=80% of reference gap covered and all P0 flank components connected',
                  'apls': 'Shared raster approximation, not official SpaceNet APLS'},
              'note': 'Oracle Q uses GT and is not a deployable result or strict performance upper bound.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_csv(args.output_dir / 'gaps.csv', all_gaps)
    write_csv(args.output_dir / 'images.csv', image_rows)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--repair_checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--enable_stage2_local_propagation', action='store_true')
    parser.add_argument('--enable_gap_refinement', action='store_true')
    args = parser.parse_args()
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.repair_checkpoint = args.repair_checkpoint.resolve()
    args.device = torch.device(args.device)
    diagnose(args)


if __name__ == '__main__':
    main()
