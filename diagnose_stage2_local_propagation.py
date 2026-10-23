"""Audit Stage 2 local connectivity propagation and its paired surface effect."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_gap_candidates import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from evaluation.road_comparison.road_metrics import (
    DEFAULT_PROTOCOL, image_metrics, prepare_target, summarize,
)
from mass_protocol import sliding_positions, stitch_logits
from topology_direction_constants import (
    CONNECTIVITY_DIRECTIONS, CONNECTIVITY_DIR_NAMES, CONNECTIVITY_OPPOSITE,
)


def neighbor_at(value, dy, dx):
    height, width = value.shape[-2:]
    padded = F.pad(value, (1, 1, 1, 1))
    return padded[..., 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]


def add_region_distribution(store, name, values, region):
    selected = values[region]
    if selected.numel() == 0:
        return
    slot = store[name]
    flat = selected.detach().float().cpu().numpy().reshape(-1)
    slot['sum'] += float(flat.sum())
    slot['count'] += int(flat.size)
    stride = max(1, len(flat) // 128)
    slot['samples'].extend(flat[::stride][:128].tolist())


def summarize_regions(store):
    result = {}
    for name, slot in store.items():
        samples = np.asarray(slot['samples'], np.float64)
        result[name] = {
            'pixels': slot['count'],
            'mean': slot['sum'] / max(slot['count'], 1),
            'sampled_p10_p50_p90': np.quantile(samples, [.1, .5, .9]).tolist()
            if len(samples) else None,
            'quantile_sample_size': int(len(samples)),
        }
    return result


def propagation_weight_maps(skeleton, connectivity):
    raw_weights = []
    for direction, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        opposite = CONNECTIVITY_OPPOSITE[direction]
        source_weight = skeleton * connectivity[:, opposite:opposite+1]
        raw_weights.append(neighbor_at(source_weight, dy, dx))
    raw = torch.cat(raw_weights, dim=1)
    effective = raw / (1 + raw.sum(dim=1, keepdim=True))
    return raw, effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--threshold', type=float, default=.45)
    parser.add_argument('--max_images', type=int, default=0, help='0 evaluates the full split')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('--threshold must be in (0,1)')
    if args.output_dir.exists():
        parser.error('Choose a new output_dir')

    model, _ = load_model(args.model_path, args.device)
    stage2 = model.swin_unet.decoder_structure_blocks['2']
    propagation = stage2.local_propagation
    saved_beta = propagation.beta.detach().clone()
    dataset = RoadSkeletonDataset(args.root_path, args.split, image_size=None,
                                  source_patch_size=1500, return_full_image=True)
    args.output_dir.mkdir(parents=True)
    protocol = dict(DEFAULT_PROTOCOL, image_size=1500)
    metrics = {'propagation_off': [], 'propagation_on': []}
    per_image = []
    distributions = defaultdict(lambda: {'sum': 0., 'count': 0, 'samples': []})
    stage_calls, propagation_records = [], []

    def stage_hook(_, inputs, outputs):
        stage_calls.append(True)

    def propagation_hook(_, inputs, output):
        propagation_records.append({
            'skeleton': inputs[1].detach().float().cpu(),
            'connectivity': inputs[2].detach().float().cpu(),
        })

    stage_handle = stage.register_forward_hook(stage_hook)
    propagation_handle = propagation.register_forward_hook(propagation_hook)
    count = len(dataset) if args.max_images <= 0 else min(len(dataset), args.max_images)
    threshold_logit = float(np.log(args.threshold / (1 - args.threshold)))
    try:
        for index in range(count):
            sample = dataset[index]
            image = sample['image'][None].to(args.device)
            target_mask = sample['mask'][0].numpy() > .5
            target = prepare_target(target_mask, protocol)
            propagation.beta.zero_()
            off_logits, _ = stitch_logits(model, image)
            stage_calls.clear()
            propagation_records.clear()
            propagation.beta.copy_(saved_beta)
            on_logits, _ = stitch_logits(model, image)

            if len(stage_calls) != len(propagation_records):
                raise RuntimeError('Stage 2 and propagation hooks saw different tile counts')
            off = off_logits[0, 0].cpu().numpy()
            on = on_logits[0, 0].cpu().numpy()
            off_mask, on_mask = off >= threshold_logit, on >= threshold_logit
            metrics['propagation_off'].append(image_metrics(off_mask, target, protocol))
            metrics['propagation_on'].append(image_metrics(on_mask, target, protocol))
            name = Path(sample['image_name']).stem
            per_image.append({
                'image': name,
                **{f'off_{key}': value for key, value in metrics['propagation_off'][-1].items()},
                **{f'on_{key}': value for key, value in metrics['propagation_on'][-1].items()},
                'changed_pixel_fraction': float(np.mean(off_mask != on_mask)),
                'probability_abs_delta_mean': float(np.abs(
                    1 / (1 + np.exp(-np.clip(on, -80, 80))) -
                    1 / (1 + np.exp(-np.clip(off, -80, 80)))).mean()),
                'added_gt_road_pixels': int((on_mask & ~off_mask & target_mask).sum()),
                'added_background_pixels': int((on_mask & ~off_mask & ~target_mask).sum()),
                'removed_gt_road_pixels': int((off_mask & ~on_mask & target_mask).sum()),
                'removed_background_pixels': int((off_mask & ~on_mask & ~target_mask).sum()),
            })

            tile_index = 0
            row_positions = sliding_positions(1500, 512, 256)
            col_positions = sliding_positions(1500, 512, 256)
            for tile_index, prop_item in enumerate(propagation_records):
                # Hook order matches the protocol's row-major 512/256 sliding windows.
                tile_top = row_positions[tile_index // len(col_positions)]
                tile_left = col_positions[tile_index % len(col_positions)]
                tile_index += 1
                s2 = prop_item['skeleton']
                c2 = prop_item['connectivity']
                height, width = s2.shape[-2:]
                gt_tile = torch.as_tensor(target_mask[tile_top:tile_top+512,
                                                      tile_left:tile_left+512], dtype=torch.float32)[None, None]
                pred_tile_prob = torch.as_tensor(
                    1 / (1 + np.exp(-np.clip(on[tile_top:tile_top+512,
                                                 tile_left:tile_left+512], -80, 80))),
                    dtype=torch.float32)[None, None]
                gt_grid = F.adaptive_max_pool2d(gt_tile, (height, width))[0, 0] > 0
                pred_grid = F.adaptive_avg_pool2d(pred_tile_prob, (height, width))[0, 0] >= args.threshold
                regions = {
                    'normal_road_gt_and_predicted': gt_grid & pred_grid,
                    'true_missed_road_gt_only': gt_grid & ~pred_grid,
                    'ordinary_background': ~gt_grid,
                }
                raw, effective = propagation_weight_maps(s2, c2)
                signals = {
                    'S2': s2,
                    'C2_mean_over_8_directions': c2.mean(dim=1, keepdim=True),
                    'C2_max_over_8_directions': c2.max(dim=1, keepdim=True).values,
                    'raw_weight_mean_over_directions': raw.mean(dim=1, keepdim=True),
                    'effective_weight_sum': effective.sum(dim=1, keepdim=True),
                }
                for direction, direction_name in enumerate(CONNECTIVITY_DIR_NAMES):
                    signals[f'C2_{direction_name}'] = c2[:, direction:direction+1]
                    signals[f'raw_w_{direction_name}'] = raw[:, direction:direction+1]
                    signals[f'effective_w_{direction_name}'] = effective[:, direction:direction+1]
                for region_name, region in regions.items():
                    for signal_name, tensor in signals.items():
                        add_region_distribution(distributions,
                                                f'{region_name}/{signal_name}',
                                                tensor[0, 0] if tensor.shape[1] == 1 else tensor[0],
                                                region)
            print(f'[STAGE2-LOCAL] {index+1}/{count} {name}: beta={float(saved_beta):.6g}; '
                  f'changed={per_image[-1]["changed_pixel_fraction"]:.4f}', flush=True)
    finally:
        propagation.beta.copy_(saved_beta)
        stage_handle.remove()
        propagation_handle.remove()

    for name in metrics:
        metrics[name] = summarize(metrics[name])
    with (args.output_dir / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader(); writer.writerows(per_image)
    distribution_summary = summarize_regions(distributions)
    (args.output_dir / 'stage2_region_distributions.json').write_text(
        json.dumps(distribution_summary, indent=2), encoding='utf-8')
    summary = {
        'checkpoint': args.model_path, 'split': args.split, 'images': count,
        'threshold': args.threshold, 'stage2_local_propagation_beta': float(saved_beta),
        'stage2_region_distributions': 'stage2_region_distributions.json',
        'surface_metrics': metrics,
        'mean_changed_pixel_fraction': float(np.mean([r['changed_pixel_fraction'] for r in per_image])),
        'mean_probability_abs_delta': float(np.mean([r['probability_abs_delta_mean'] for r in per_image])),
        'region_definition': 'At Stage 2 feature resolution: normal road = any GT road in cell and pooled final '
                             'surface probability >= threshold; missed road = GT road and below threshold; '
                             'background = no GT road in cell.',
        'weight_definition': 'raw_w_d(p)=S2(p+delta_d)*C2_opposite(d)(p+delta_d); effective_w_d is raw weight '
                             'divided by 1+sum_d(raw_w_d), exactly matching the neighbor coefficients used by '
                             'LocalConnectivityPropagation.',
        'scope': 'Frozen checkpoint paired inference; propagation-off changes only Stage 2 beta to zero. '
                 'APLS is the shared raster approximation, not official SpaceNet APLS.'
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
