"""Paired, validation-only audit of directional dynamics checkpoints."""

import csv
import json

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from networks.reliable_directional_repair import compose_surface_logits
from run_q3_r3_mass import full_dataset
from run_reliable_directional_mass import evaluate_thresholds, path_length_bin


def mean(value, mask):
    selected = value[mask]
    return float(selected.float().mean()) if selected.numel() else None


def quantiles(value, mask, name):
    selected = value[mask].float()
    if not selected.numel():
        return {f'{name}_{label}': None for label in ('p10', 'p50', 'p90')}
    return {f'{name}_{label}': float(torch.quantile(selected, quantile))
            for label, quantile in (('p10', .1), ('p50', .5), ('p90', .9))}


def summary(rows):
    keys = sorted({key for row in rows for key in row})
    result = {}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))
                  and row[key] is not None and np.isfinite(row[key])]
        result[key] = float(np.mean(values)) if values else None
    return result


def work_mask(mask, size):
    return F.adaptive_max_pool2d(mask.float(), size) > 0


def masks(result, mask, skeleton, threshold):
    p0 = result['p0']
    missing = (skeleton > .5) & (F.max_pool2d(p0, 5, 1, 2) < threshold)
    size = result['u0'].shape[-2:]
    road = (mask > .5) & (p0 >= threshold)
    background = (mask < .5) & (F.interpolate(result['q'], size=mask.shape[-2:],
                                               mode='bilinear', align_corners=False) > .5)
    new_fp = (result['p1'] >= threshold) & (p0 < threshold) & (mask < .5)
    isolated_fp = new_fp & (F.max_pool2d((p0 >= threshold).float(), 7, 1, 3) == 0)
    return {'all': torch.ones_like(result['u0'], dtype=torch.bool),
            'missing': work_mask(missing, size),
            'reliable_road': work_mask(road, size),
            'hard_background': work_mask(background, size),
            'isolated_new_fp': work_mask(isolated_fp, size)}


def dynamics_with(model, result, weights, accept=None):
    states, _, _ = model.dynamics(result['visual'], result['u0'], result['q'],
                                  result['r'], result['b'],
                                  result['a'] if accept is None else accept, weights)
    z1, _ = compose_surface_logits(result['z0'], result['u0'], states[-1])
    return z1.sigmoid(), states[-1]


def gap_rows(result, mask, skeleton, threshold, case_name):
    p0 = (result['p0'][0, 0].cpu().numpy() >= threshold).astype(np.uint8)
    missing = (skeleton[0, 0].cpu().numpy() > .5) & ~(
        cv2.dilate(p0, np.ones((5, 5), np.uint8)) > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8)
    work_size = result['u0'].shape[-2:]
    rows = []
    for index in range(1, count):
        length = int(stats[index, cv2.CC_STAT_AREA])
        if length < 3:
            continue
        segment = torch.from_numpy((labels == index)[None, None]).to(result['u0'].device)
        center = work_mask(segment, work_size)
        flank_full = ((F.max_pool2d(segment.float(), 17, 1, 8) > 0) & ~segment &
                      (mask > .5) & (result['p0'] >= threshold))
        flank = work_mask(flank_full, work_size)
        positions = np.column_stack(np.nonzero(labels == index)).astype(np.float32)
        center_yx = positions.mean(axis=0)
        _, _, axes = np.linalg.svd(positions - center_yx, full_matrices=False)
        direction = axes[0] if axes.shape[0] else np.array([0., 1.])
        yy, xx = np.indices(labels.shape)
        side = (yy - center_yx[0]) * direction[0] + (xx - center_yx[1]) * direction[1]
        negative = work_mask(flank_full & torch.from_numpy(
            side[None, None] < -1.).to(flank_full.device), work_size)
        positive = work_mask(flank_full & torch.from_numpy(
            side[None, None] > 1.).to(flank_full.device), work_size)
        p0_work = result['u0']
        row = {'case': case_name, 'length': length, 'bin': path_length_bin(length),
               'flank_r': mean(result['r'], flank),
               'flank_r_side_1': mean(result['r'], negative),
               'flank_r_side_2': mean(result['r'], positive),
               'best_bilateral': mean(result['e_two'].amax(1, keepdim=True), center),
               'directional_bilateral': mean(result['directional_bilateral'], center),
               'b_minus_p0': mean(result['b'] - p0_work, center),
               'b_above_threshold_fraction': mean((result['b'] >= threshold).float(), center),
               'a': mean(result['a'], center)}
        for step in (1, 2, 3):
            row[f'u{step}_recovered_fraction'] = mean(
                (result[f'u{step}'] >= threshold).float(), center)
            row[f'u{step}_signed_change'] = mean(result[f'u{step}'] - p0_work, center)
        center_full = segment.bool()
        row['p1_center_recovered_fraction'] = mean(
            (result['p1'] >= threshold).float(), center_full)
        rows.append(row)
    return rows


@torch.no_grad()
def audit(args, model):
    model.eval()
    dataset = full_dataset(args.root_path, 'val')
    threshold = args.audit_threshold
    reports, _, tiles, _ = evaluate_thresholds(
        model, dataset, args.device, (threshold,), args.max_images)
    rows, gaps = [], []
    for _, index, top, left in tiles:
        sample = dataset[index]
        image = sample['image'][None, :, top:top+512, left:left+512].to(args.device)
        mask = sample['mask'][None, :, top:top+512, left:left+512].to(args.device)
        skeleton = sample['skeleton'][None, :, top:top+512, left:left+512].to(args.device)
        result = model(image, capture_diagnostics=True)
        regions = masks(result, mask, skeleton, threshold)
        zero_p1, zero_u3 = dynamics_with(model, result, torch.zeros_like(result['w']))
        rotated_w = result['w'].roll(2, dims=1)
        rotated_p1, _ = dynamics_with(model, result, rotated_w)
        constant_a = torch.full_like(result['a'], float(result['a'].mean()))
        constant_a_p1, _ = dynamics_with(model, result, result['w'], constant_a)
        rotated_full = model(image, capture_diagnostics=True,
                             weights_override=rotated_w)
        for region_name, region in regions.items():
            row = {'case': sample['case_name'], 'region': region_name,
                   'pixels': int(region.sum()),
                   'r_mean': mean(result['r'], region),
                   'a_mean': mean(result['a'], region),
                   'b_minus_p0_signed': mean(result['b'] - result['u0'], region),
                   'b_minus_p0_abs': mean((result['b'] - result['u0']).abs(), region),
                   'b_above_threshold_fraction': mean((result['b'] >= threshold).float(), region),
                   'a_above_half_fraction': mean((result['a'] >= .5).float(), region),
                   'bilateral_mean': mean(result['directional_bilateral'], region),
                   'one_sided_mean': mean(result['e_one'].mean(1, keepdim=True), region),
                   'neighbor_minus_p0': mean(result['directional_neighbor'] -
                                             result['u0'], region),
                   'w_mass': mean(result['w'].sum(1, keepdim=True), region),
                   'u3_minus_u0': mean(result['u3'] - result['u0'], region),
                   'fixed_ba_no_w_u3_abs': mean((result['u3'] - zero_u3).abs(), region)}
            row.update(quantiles(result['b'] - result['u0'], region, 'b_minus_p0'))
            row.update(quantiles(result['a'], region, 'a'))
            full_region = F.interpolate(region.float(), size=result['p1'].shape[-2:],
                                        mode='nearest').bool()
            row['p1_minus_p0_signed'] = mean(result['p1'] - result['p0'], full_region)
            row['fixed_ba_no_transport_p1_abs'] = mean(
                (result['p1'] - zero_p1).abs(), full_region)
            row['fixed_ba_rotated_w_p1_abs'] = mean(
                (result['p1'] - rotated_p1).abs(), full_region)
            row['recomputed_ba_rotated_w_p1_abs'] = mean(
                (result['p1'] - rotated_full['p1']).abs(), full_region)
            row['constant_a_p1_signed'] = mean(
                constant_a_p1 - result['p1'], full_region)
            row['recomputed_b_rotated_w_abs'] = mean(
                (result['b'] - rotated_full['b']).abs(), region)
            row['recomputed_a_rotated_w_abs'] = mean(
                (result['a'] - rotated_full['a']).abs(), region)
            for direction in range(8):
                row[f'w{direction}_mean'] = mean(result['w'][:, direction:direction+1],
                                                 region)
            for step in range(3):
                previous = result[f'u{step}']
                row[f'alpha{step+1}'] = mean(result['alpha'][:, step], region)
                row[f'beta{step+1}'] = mean(result['beta'][:, step], region)
                row[f'gamma{step+1}'] = mean(result['gamma'][:, step], region)
                row[f'v{step+1}'] = mean(result['source_available'][:, step], region)
                row[f'm_minus_u{step}_abs'] = mean(
                    (result['m'][:, step] - previous).abs(), region)
                row.update(quantiles(result['alpha'][:, step], region,
                                     f'alpha{step+1}'))
                row.update(quantiles(result['beta'][:, step], region,
                                     f'beta{step+1}'))
                row.update(quantiles(result['source_available'][:, step], region,
                                     f'v{step+1}'))
                row[f'u{step+1}_minus_p0_abs'] = mean(
                    (result[f'u{step+1}'] - result['u0']).abs(), region)
            rows.append(row)
        gaps.extend(gap_rows(result, mask, skeleton, threshold, sample['case_name']))
        print(f'[AUDIT] {len(rows)//5}/{len(tiles)} tiles; gaps={len(gaps)}', flush=True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (('regions', rows), ('sampled_gaps', gaps)):
        if data:
            with (output / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
                fields = list(data[0])
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
    report = {'checkpoint_version': 'v2' if model.legacy_route else 'v3',
              'split': 'val', 'threshold': threshold,
              'full_image_comparison': reports[threshold],
              'sample_rule': 'one P0-gap-rich 512 tile per validation image',
              'w_to_p1_definition': 'mean |P1(W,B,A,R)-P1(W=0,B,A,R fixed)|',
              'region_summary': {name: summary([row for row in rows if row['region'] == name])
                                 for name in ('all', 'missing', 'reliable_road',
                                              'hard_background', 'isolated_new_fp')},
              'gap_bins': {name: {'count': len(selected), **summary(selected)}
                           for name in ('3-7', '8-15', '16-31', '32+')
                           for selected in [[row for row in gaps if row['bin'] == name]]},
              'gap_note': 'Sampled-tile center recovery is not a full path-closure test; '
                          'full-image paired gap closure above is also a proxy.'}
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'[AUDIT] {output / "audit.json"}', flush=True)
