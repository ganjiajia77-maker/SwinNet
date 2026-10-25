"""Read-only audit of a trained reliable directional dynamics checkpoint."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from mass_protocol import sliding_positions
from networks.reliable_directional_repair import (DIRECTIONS, dynamic_losses,
                                                  shifted)
from run_q3_r3_mass import full_dataset, stitch_three
from run_reliable_directional_mass import (MODULE_NAMES, VERSION, load_model,
                                           selection_score)


REGIONS = ('all', 'q_high', 'gap_center', 'difficult_background')


def tensor_summary(values):
    if not values.size:
        return {'count': 0}
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite diagnostic values')
    return {'count': int(values.size), 'mean': float(values.mean()),
            'max': float(values.max()),
            'p10': float(np.percentile(values, 10)),
            'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)),
            'p99': float(np.percentile(values, 99))}


def module_state(model):
    return {name: {key: value.detach().cpu().clone()
                   for key, value in getattr(model, name).state_dict().items()}
            for name in MODULE_NAMES}


def module_distance(left, right):
    answer = {}
    for name in MODULE_NAMES:
        deltas = [(right[name][key].float() - value.float()).flatten()
                  for key, value in left[name].items()]
        flat = torch.cat(deltas)
        answer[name] = {'l2': float(flat.norm()),
                        'max_abs': float(flat.abs().max()),
                        'changed_values': int((flat != 0).sum())}
    return answer


def checkpoint_audit(args, model, initial):
    payload = torch.load(args.dynamic_checkpoint, map_location='cpu', weights_only=False)
    if payload.get('version') != VERSION:
        raise ValueError('Wrong checkpoint version')
    if set(payload.get('modules', {})) != set(MODULE_NAMES):
        raise ValueError('Checkpoint module list does not match current model')
    best = module_state(model)
    last_path = args.last_checkpoint or args.dynamic_checkpoint.with_name('last.pth')
    last = None
    if last_path.is_file():
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        for name in MODULE_NAMES:
            if set(last['modules'][name]) != set(best[name]):
                raise ValueError(f'Last checkpoint module mismatch: {name}')
    trainable = {name: sum(p.numel() for p in getattr(model, name).parameters())
                 for name in MODULE_NAMES}
    expected = sum(trainable.values())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=.01)
    optimizer_parameters = [p for group in optimizer.param_groups for p in group['params']]
    optimizer_count = sum(p.numel() for p in optimizer_parameters)
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    if (optimizer_count != expected or
            {id(p) for p in optimizer_parameters} != {id(p) for p in model_parameters}):
        raise RuntimeError('Trainable parameters are missing from optimizer')
    if last and 'optimizer' in last:
        optimizer.load_state_dict(last['optimizer'])
    report = {
        'best_epoch': payload['epoch'],
        'last_epoch': last.get('epoch') if last else None,
        'best_selection_score': payload.get('selection'),
        'best_has_optimizer': 'optimizer' in payload,
        'last_has_optimizer': bool(last and 'optimizer' in last),
        'last_optimizer_loads': bool(last and 'optimizer' in last),
        'trainable_parameters': trainable,
        'optimizer_parameter_count': optimizer_count,
        'initial_reconstruction_to_best': module_distance(initial, best),
        'best_to_last': (module_distance(best, last['modules']) if last else None),
        'selection_formula': 'IoU + 0.15*clDice + 0.10*APLS_GT + '
                             '0.05*APLS_bidir - 0.001*short_components - '
                             '0.01*background_bridge_proxy/images',
        'selection_function': selection_score.__name__,
        'initial_reference': 'Reconstructed with the training seed and current code; '
                             'original initial weights were not saved.',
    }
    return report


def gt_gap_mask(skeleton, p0, threshold):
    covered = cv2.dilate((p0 >= threshold).astype(np.uint8),
                         np.ones((5, 5), np.uint8)) > 0
    missing = (skeleton & ~covered).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, connectivity=8)
    kept = np.zeros_like(missing, dtype=bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 3:
            kept |= labels == index
    return kept, int(sum(stats[index, cv2.CC_STAT_AREA] >= 3
                         for index in range(1, count)))


def work_mask(full_mask, work_size, device):
    value = torch.from_numpy(full_mask.astype(np.float32))[None, None].to(device)
    return F.adaptive_max_pool2d(value, work_size).bool()


def collect(values, key, value, masks):
    arr = value.detach().float().cpu().numpy()[0, 0]
    for region, mask in masks.items():
        if mask.shape != arr.shape:
            raise ValueError(f'{key} mask shape mismatch: {mask.shape} != {arr.shape}')
        if mask.any():
            values[(region, key)].append(arr[mask].copy())


def region_masks(result, gt_mask, gap_mask, device):
    q = result['q'].detach()
    work_size = q.shape[-2:]
    q_full = F.interpolate(q, size=gt_mask.shape, mode='bilinear',
                           align_corners=False)[0, 0].cpu().numpy() >= .5
    full = {'all': np.ones_like(gt_mask), 'q_high': q_full,
            'gap_center': gap_mask,
            'difficult_background': q_full & ~gt_mask}
    work = {name: work_mask(mask, work_size, device)[0, 0].cpu().numpy()
            for name, mask in full.items()}
    return work, full


def collect_forward(values, result, gt_mask, gap_mask, device):
    work, full = region_masks(result, gt_mask, gap_mask, device)
    p0 = result['p0_work']
    collect(values, 'R', result['r'], work)
    collect(values, 'A', result['a'], work)
    collect(values, 'abs_B_minus_P0_work', (result['b'] - p0).abs(), work)
    collect(values, 'W_no_transport', 1. - result['w'].sum(1, keepdim=True), work)
    for direction in range(8):
        collect(values, f'W_direction_{direction}', result['w'][:, direction:direction+1], work)
    for step in range(3):
        state = result[f'u{step}']
        updated = result[f'u{step+1}']
        availability = result['source_available'][:, step]
        rates = result['rate_coefficients'][:, step]
        collect(values, f'V_step_{step+1}', availability, work)
        for channel, name in enumerate(('raw_stay', 'raw_transport',
                                        'raw_propose', 'gamma_anchor')):
            collect(values, f'{name}_step_{step+1}', rates[:, channel:channel+1], work)
        collect(values, f'alpha_transport_step_{step+1}',
                result['transport'][:, step], work)
        collect(values, f'beta_propose_step_{step+1}',
                result['propose'][:, step], work)
        collect(values, f'stay_step_{step+1}', result['stay'][:, step], work)
        support = result['r'] + (1. - result['r']) * result['a'] * state
        numerator = torch.zeros_like(state)
        for index, (dy, dx) in enumerate(DIRECTIONS):
            source = result['w'][:, index:index+1] * shifted(support, dy, dx)
            numerator += source * shifted(state, dy, dx)
        mass = result['source_mass'][:, step].clamp_min(1e-6)
        propagated = numerator / mass
        collect(values, f'abs_M_minus_U_previous_step_{step+1}',
                (propagated - state).abs(), work)
        collect(values, f'abs_M_minus_U_updated_step_{step+1}',
                (propagated - updated).abs(), work)
        collect(values, f'abs_U{step+1}_minus_P0_work', (updated - p0).abs(), work)
    collect(values, 'abs_T_full_minus_P0_full',
            (result['t_full'] - result['p0']).abs(), full)
    collect(values, 'A_full', result['accept_full'], full)
    collect(values, 'abs_P1_minus_P0_full',
            (result['p1'] - result['p0']).abs(), full)
    collect(values, 'signed_P1_minus_P0_full', result['p1'] - result['p0'], full)


@torch.no_grad()
def counterfactuals(model, image, baseline, gap_mask, gt_mask, threshold):
    modes = {'final_A_one': {'final_a_one': True},
             'internal_A_one': {'internal_a_one': True},
             'both_A_one': {'internal_a_one': True, 'final_a_one': True},
             'B_plus_0.5': {'proposal_shift': .5},
             'W_rotated': {'rotate_w': True}}
    report = {}
    gap = torch.from_numpy(gap_mask).to(image.device)[None, None]
    background = torch.from_numpy(~gt_mask).to(image.device)[None, None]
    for name, override in modes.items():
        result = model(image, diagnostic_override=override)
        d_p = (result['p1'] - baseline['p1']).abs()
        d_u = (result['u3'] - baseline['u3']).abs()
        report[name] = {
            'abs_P1_change_mean': float(d_p.mean()),
            'abs_P1_change_max': float(d_p.max()),
            'abs_U3_change_mean': float(d_u.mean()),
            'gap_center_signed_P1_change': (float((result['p1'] - baseline['p1'])[gap].mean())
                                            if gap.any() else None),
            'background_abs_P1_change': (float(d_p[background].mean())
                                         if background.any() else None),
            'binary_changed_pixels': int(((result['p1'] >= threshold) !=
                                          (baseline['p1'] >= threshold)).sum()),
        }
    return report


def norm(grads):
    found = [grad.detach().float().square().sum() for grad in grads if grad is not None]
    return float(torch.stack(found).sum().sqrt()) if found else 0.


def gradient_audit(model, args):
    from datasets.dataset_road_skeleton import RoadSkeletonDataset
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
                                  source_patch_size=1500, random_crop_train=True,
                                  random_crops_per_image=1, random_crop_seed=args.seed,
                                  augment=True)
    dataset.set_epoch(1)
    sample = dataset[0]
    image = sample['image'][None].to(args.device)
    mask = sample['mask'][None].to(args.device)
    skeleton = sample['skeleton'][None].to(args.device)
    model.train()
    result = model(image)
    loss, terms = dynamic_losses(result, mask, skeleton,
                                 lambda_bg=args.lambda_bg,
                                 lambda_cont=args.lambda_cont,
                                 return_live_terms=True)
    parameters = {name: tuple(getattr(model, name).parameters())
                  for name in MODULE_NAMES}
    gradients = {}
    for name, module_parameters in parameters.items():
        gradients[name] = {}
        for loss_name, live in (('L_seg', terms['seg_live']),
                                ('L_cont', terms['cont_live']), ('L_total', loss)):
            grads = torch.autograd.grad(live, module_parameters,
                                        retain_graph=True, allow_unused=True)
            gradients[name][loss_name] = norm(grads)
    before = module_state(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(p.grad is not None for p in model.frozen.parameters()):
        raise RuntimeError('Frozen parameters received gradients')
    optimizer.step()
    moved = module_distance(before, module_state(model))
    return {'loss_total': float(loss), 'L_seg': float(terms['seg']),
            'L_cont': float(terms['cont']), 'weighted_L_cont': float(args.lambda_cont * terms['cont']),
            'valid_gt_paths': terms['paths'], 'gradients': gradients,
            'optimizer_step_parameter_change': moved,
            'checkpoint_unchanged_on_disk': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--q3_checkpoint', type=Path, required=True)
    parser.add_argument('--dynamic_checkpoint', type=Path, required=True)
    parser.add_argument('--last_checkpoint', type=Path)
    parser.add_argument('--output_json', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_images', type=int, default=14)
    parser.add_argument('--threshold', type=float, default=.45)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_bg', type=float, default=1.)
    parser.add_argument('--lambda_cont', type=float, default=.2)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    initial_args = copy.copy(args)
    initial_args.dynamic_checkpoint = None
    model = load_model(initial_args)
    initial = module_state(model)
    payload = torch.load(args.dynamic_checkpoint, map_location='cpu', weights_only=False)
    expected = {'version': VERSION, 'base_checkpoint': str(args.base_checkpoint.resolve()),
                'q3_checkpoint': str(args.q3_checkpoint.resolve()), 'width': args.width,
                'steps': 3}
    mismatch = {key: (payload.get(key), value) for key, value in expected.items()
                if payload.get(key) != value}
    if mismatch or set(payload.get('modules', {})) != set(MODULE_NAMES):
        raise ValueError(f'Checkpoint mismatch: {mismatch}')
    for name in MODULE_NAMES:
        getattr(model, name).load_state_dict(payload['modules'][name], strict=True)
    checkpoint = checkpoint_audit(args, model, initial)
    model.eval()
    dataset = full_dataset(args.root_path, 'val')
    limit = min(len(dataset), args.max_images)
    values = defaultdict(list)
    counter = defaultdict(list)
    sample_rows = []
    for index in range(limit):
        sample = dataset[index]
        image = sample['image'][None].to(args.device)
        with torch.no_grad():
            z0, _, _ = stitch_three(model, image)
        p0 = 1. / (1. + np.exp(-np.clip(z0, -80., 80.)))
        skeleton = sample['skeleton'][0].numpy() > .5
        gt_mask = sample['mask'][0].numpy() > .5
        gap, gap_count = gt_gap_mask(skeleton, p0, args.threshold)
        positions = sliding_positions(1500, 512, 256)
        _, top, left = max((int(gap[y:y+512, x:x+512].sum()), y, x)
                           for y in positions for x in positions)
        tile = image[:, :, top:top+512, left:left+512]
        tile_gt = gt_mask[top:top+512, left:left+512]
        tile_gap = gap[top:top+512, left:left+512]
        with torch.no_grad():
            result = model(tile, capture_diagnostics=True)
            collect_forward(values, result, tile_gt, tile_gap, args.device)
            modes = counterfactuals(model, tile, result, tile_gap, tile_gt,
                                    args.threshold)
        for mode, metrics in modes.items():
            for key, value in metrics.items():
                if value is not None:
                    counter[(mode, key)].append(value)
        sample_rows.append({'case': sample['case_name'], 'reference_gaps': gap_count,
                            'selected_tile': [top, left],
                            'gap_pixels_in_tile': int(tile_gap.sum()),
                            'baseline_added_pixels': int(((result['p1'] >= args.threshold) &
                                                          (result['p0'] < args.threshold)).sum()),
                            'counterfactuals': modes})
        print(f'[AUDIT] {index+1}/{limit} {sample["case_name"]} gaps={gap_count}', flush=True)
    report = {'version': VERSION, 'validation_images': limit,
              'threshold': args.threshold,
              'sample_rule': 'one P0 reference-gap-rich 512 tile per validation image',
              'gap_definition': 'GT skeleton absent within 2px of P0 binary mask; '
                                '8-connected missing segments with at least 3 pixels',
              'checkpoint': checkpoint,
              'regions': {region: {key: tensor_summary(np.concatenate(parts))
                                   for (name, key), parts in values.items() if name == region}
                          for region in REGIONS},
              'counterfactual_mean': {mode: {key: float(np.mean(parts))
                                            for (name, key), parts in counter.items()
                                            if name == mode}
                                      for mode in ('final_A_one', 'internal_A_one',
                                                   'both_A_one', 'B_plus_0.5', 'W_rotated')},
              'per_image': sample_rows}
    report['training_batch'] = gradient_audit(model, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'[AUDIT] wrote {args.output_json}', flush=True)


if __name__ == '__main__':
    main()
