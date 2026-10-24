"""Paired P0/P1 addition audit for a trained dual-head Q3/R3 checkpoint."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from evaluation.road_comparison.road_metrics import skeletonize
from run_q3_r3_mass import full_dataset, load_experiment, stitch_three


def probability(logits):
    return 1. / (1. + np.exp(-np.clip(logits, -80., 80.)))


def component_rows(case_id, mode, added, old_road, gt):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        added.astype(np.uint8), connectivity=8)
    _, old_labels = cv2.connectedComponents(old_road.astype(np.uint8), connectivity=8)
    distance = (cv2.distanceTransform((~old_road).astype(np.uint8),
                                      cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                if old_road.any() else None)
    rows = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        x1 = min(x + width + 1, added.shape[1])
        y1 = min(y + height + 1, added.shape[0])
        region = labels[y0:y1, x0:x1] == index
        neighbors = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touched = np.unique(old_labels[y0:y1, x0:x1][neighbors & old_road[y0:y1, x0:x1]])
        touched = touched[touched > 0]
        mask = labels[y:y+height, x:x+width] == index
        gt_pixels = int(np.count_nonzero(mask & gt[y:y+height, x:x+width]))
        distances = distance[y:y+height, x:x+width][mask] if distance is not None else None
        max_distance = float(distances.max()) if distances is not None else None
        if len(touched) >= 2:
            category = 'gt_supported_bridge_proxy' if gt_pixels / area >= 0.5 else 'possible_false_bridge'
        elif len(touched) == 0:
            category = 'isolated'
        elif max_distance is not None and max_distance <= 2.:
            category = 'edge_expansion_proxy'
        else:
            category = 'attached_extension'
        rows.append({
            'case_id': case_id, 'mode': mode, 'component_id': index,
            'area_px': area, 'skeleton_length_px': int(skeletonize(mask).sum()),
            'min_distance_to_old_road_px': float(distances.min()) if distances is not None else None,
            'max_distance_to_old_road_px': max_distance,
            'touches_old_road': bool(len(touched)),
            'old_components_touched': int(len(touched)),
            'gt_overlap_px': gt_pixels, 'gt_overlap_fraction': gt_pixels / area,
            'category': category,
        })
    return rows


def image_row(case_id, mode, p0_prob, p1_prob, q, gt, threshold, q_threshold):
    old_road = p0_prob > threshold
    new_road = p1_prob > threshold
    added = new_road & ~old_road
    removed = old_road & ~new_road
    high = q >= q_threshold
    fp = added & ~gt
    tp = added & gt
    background = ~gt
    components = component_rows(case_id, mode, added, old_road, gt)
    row = {
        'case_id': case_id, 'mode': mode,
        'added_px': int(added.sum()), 'added_gt_px': int(tp.sum()),
        'added_fp_px': int(fp.sum()),
        'added_fp_q_high_px': int((fp & high).sum()),
        'added_fp_q_low_px': int((fp & ~high).sum()),
        'added_gt_q_high_px': int((tp & high).sum()),
        'added_gt_q_low_px': int((tp & ~high).sum()),
        'gt_background_q_high_px': int((background & high).sum()),
        'gt_background_q_low_px': int((background & ~high).sum()),
        'removed_gt_px': int((removed & gt).sum()),
        'removed_background_px': int((removed & background).sum()),
        'positive_probability_change_mean': float(np.maximum(p1_prob - p0_prob, 0.).mean()),
        'positive_probability_change_on_added_mean': (
            float((p1_prob - p0_prob)[added].mean()) if added.any() else 0.),
        'added_components': len(components),
        'isolated_components': sum(item['category'] == 'isolated' for item in components),
        'edge_expansion_proxy_components': sum(
            item['category'] == 'edge_expansion_proxy' for item in components),
        'gt_supported_bridge_proxy_components': sum(
            item['category'] == 'gt_supported_bridge_proxy' for item in components),
        'possible_false_bridge_components': sum(
            item['category'] == 'possible_false_bridge' for item in components),
    }
    return row, components, added


def visualization(sample, p0_prob, p1_prob, q, gt, added, threshold, delta_scale):
    p0_png = (p0_prob > threshold).astype(np.uint8) * 255
    q_png = np.round(np.clip(q, 0., 1.) * 255).astype(np.uint8)
    positive_delta = np.maximum(p1_prob - p0_prob, 0.)
    delta_png = np.round(np.clip(positive_delta / delta_scale, 0., 1.) * 255).astype(np.uint8)
    image = sample['image'].numpy().transpose(1, 2, 0)
    image = np.clip((image * [0.229, 0.224, 0.225] +
                     [0.485, 0.456, 0.406]) * 255., 0., 255.).astype(np.uint8)
    overlay = (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 0.25).astype(np.uint8)
    overlay[gt] = (170, 170, 170)
    overlay[p0_png > 0] = (230, 90, 20)
    overlay[added & gt] = (40, 235, 40)
    overlay[added & ~gt] = (30, 30, 255)
    return {'p0_road': p0_png, 'q3': q_png,
            'positive_probability_change': delta_png, 'added_road_gt_overlay': overlay}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    totals = {key: sum(row[key] for row in rows) for key in (
        'added_px', 'added_gt_px', 'added_fp_px', 'added_fp_q_high_px',
        'added_fp_q_low_px', 'added_gt_q_high_px', 'added_gt_q_low_px',
        'gt_background_q_high_px', 'gt_background_q_low_px',
        'removed_gt_px', 'removed_background_px', 'added_components',
        'isolated_components', 'edge_expansion_proxy_components',
        'gt_supported_bridge_proxy_components', 'possible_false_bridge_components')}
    fp_count = totals['added_fp_px']
    totals['fp_fraction_in_q_high'] = (
        totals['added_fp_q_high_px'] / fp_count if fp_count else 0.)
    for level in ('high', 'low'):
        fp = totals[f'added_fp_q_{level}_px']
        background = totals[f'gt_background_q_{level}_px']
        totals[f'fp_rate_in_q_{level}_background'] = fp / background if background else 0.
    totals['images'] = len(rows)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--repair_checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--q_threshold', type=float, default=0.5)
    parser.add_argument('--delta_scale', type=float, default=0.2)
    parser.add_argument('--max_images', type=int, default=0,
                        help='0 evaluates the full split')
    parser.add_argument('--max_visualizations', type=int, default=8,
                        help='save the cases with the most added pixels')
    parser.add_argument('--skip_gate_zero', action='store_true')
    args = parser.parse_args()
    if not (0. < args.threshold < 1. and 0. < args.q_threshold < 1.) or args.delta_scale <= 0:
        parser.error('Thresholds must be in (0,1) and delta_scale must be positive')
    if args.max_images < 0 or args.max_visualizations < 0:
        parser.error('Image limits must be nonnegative')
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.repair_checkpoint = args.repair_checkpoint.resolve()
    args.init_repair_checkpoint = None
    args.eta_plus = args.eta_minus = 0.5
    args.enable_stage2_local_propagation = False
    args.enable_gap_refinement = False
    args.device = torch.device(args.device)
    model = load_experiment(args).eval()
    dataset = full_dataset(args.root_path, args.split)
    limit = min(len(dataset), args.max_images or len(dataset))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_image, per_component, visuals = [], [], []

    for index in range(limit):
        sample = dataset[index]
        image = sample['image'][None].to(args.device)
        z0, z1, q = stitch_three(model, image)
        p0_prob, p1_prob = probability(z0), probability(z1)
        gt = sample['mask'][0].numpy() > 0.5
        case_id = sample['case_name']
        row, components, added = image_row(
            case_id, 'gate_0.2', p0_prob, p1_prob, q, gt,
            args.threshold, args.q_threshold)
        per_image.append(row)
        per_component.extend(components)
        if args.max_visualizations and (len(visuals) < args.max_visualizations or
                                        row['added_px'] > visuals[0][0]):
            panels = visualization(sample, p0_prob, p1_prob, q, gt, added,
                                   args.threshold, args.delta_scale)
            visuals.append((row['added_px'], case_id, panels))
            visuals.sort(key=lambda item: item[0])
            visuals = visuals[-args.max_visualizations:]

        if not args.skip_gate_zero:
            model.positive_gate_floor = 0.
            try:
                z0_zero, z1_zero, q_zero = stitch_three(model, image)
            finally:
                model.positive_gate_floor = 0.2
            if not np.allclose(z0, z0_zero, atol=1e-5) or not np.allclose(q, q_zero, atol=1e-5):
                raise RuntimeError('P0 or Q3 changed during the gate-only counterfactual')
            zero_row, zero_components, _ = image_row(
                case_id, 'gate_0', p0_prob, probability(z1_zero), q, gt,
                args.threshold, args.q_threshold)
            per_image.append(zero_row)
            per_component.extend(zero_components)
        print(f'[AUDIT] {index + 1}/{limit}: {case_id} added={row["added_px"]} '
              f'FP={row["added_fp_px"]}', flush=True)

    panels_dir = args.output_dir / 'aligned_maps'
    panels_dir.mkdir()
    for _, case_id, panels in visuals:
        for name, bitmap in panels.items():
            path = panels_dir / f'{case_id}_{name}.png'
            if not cv2.imwrite(str(path), bitmap):
                raise OSError(f'Cannot write {path}')
    write_csv(args.output_dir / 'per_image.csv', per_image)
    write_csv(args.output_dir / 'added_components.csv', per_component)
    report = {
        'split': args.split, 'images': limit,
        'checkpoint': str(args.repair_checkpoint),
        'threshold': args.threshold, 'q_threshold': args.q_threshold,
        'modes': {mode: summarize([row for row in per_image if row['mode'] == mode])
                  for mode in ('gate_0.2', 'gate_0')
                  if any(row['mode'] == mode for row in per_image)},
        'visualized_cases': [case_id for _, case_id, _ in visuals],
        'map_encoding': {'p0_road': 'white=predicted road at threshold',
                         'q3': 'grayscale 0..255 for Q3 0..1',
                         'positive_probability_change':
                         f'grayscale 0..255 for positive P1-P0, saturated at {args.delta_scale}',
                         'added_road_gt_overlay':
                         'dim RGB background; gray=GT road; blue=P0; green=new GT road; red=new FP'},
        'component_categories': '8-connected additions; contact with old road uses 8 neighbors. '
                                'Edge expansion and bridging are geometric proxies, not path validation.',
        'note': 'Inference-only paired diagnosis; GT is used for measurement and visualization.',
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
