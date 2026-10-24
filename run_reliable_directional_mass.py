"""Train and evaluate frozen H0/Q3 with reliable directional dynamics on Mass."""

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from evaluation.road_comparison.road_metrics import image_metrics, prepare_target, summarize
from networks.reliable_directional_repair import ReliableDirectionalRepair, dynamic_losses
from run_q3_r3_mass import (DEFAULT_CHECKPOINT, PROTOCOL, full_dataset,
                            load_experiment, stitch_three)
from mass_protocol import sliding_positions


VERSION = 'reliable_directional_dynamics_v1'
MODULE_NAMES = ('visual', 'reliable_head', 'relations', 'proposal_head',
                'accept_head', 'dynamics', 'detail_projection', 'detail_head')
THRESHOLDS = tuple(round(float(x), 2) for x in np.arange(.35, .751, .05))


def load_model(args):
    old_args = SimpleNamespace(base_checkpoint=args.base_checkpoint.resolve(),
                               repair_checkpoint=args.q3_checkpoint.resolve(),
                               init_repair_checkpoint=None, eta_plus=.5, eta_minus=.5,
                               enable_stage2_local_propagation=False,
                               enable_gap_refinement=False, device=args.device)
    frozen = load_experiment(old_args)
    model = ReliableDirectionalRepair(frozen, width=args.width, steps=3).to(args.device)
    inner = model.frozen.base.swin_unet
    if (inner.decoder_structure_blocks['2'].enable_local_propagation or
            inner.enable_gap_refinement or not inner.enable_highres_structure_stream or
            not inner.enable_h3_surface_fusion):
        raise RuntimeError('Frozen structure routing does not match this experiment')
    if args.dynamic_checkpoint:
        payload = torch.load(args.dynamic_checkpoint, map_location='cpu', weights_only=False)
        if payload.get('version') != VERSION:
            raise ValueError('Wrong directional dynamics checkpoint version')
        expected = {'base_checkpoint': str(args.base_checkpoint.resolve()),
                    'q3_checkpoint': str(args.q3_checkpoint.resolve()),
                    'width': args.width, 'steps': 3}
        mismatches = {key: (payload.get(key), value) for key, value in expected.items()
                      if payload.get(key) != value}
        if mismatches or set(payload.get('modules', {})) != set(MODULE_NAMES):
            raise ValueError(f'Checkpoint source/config/module mismatch: {mismatches}')
        for name in MODULE_NAMES:
            getattr(model, name).load_state_dict(payload['modules'][name], strict=True)
        print(f'[LOAD] strict new modules epoch={payload["epoch"]}', flush=True)
    frozen_count = sum(parameter.numel() for parameter in model.frozen.parameters())
    new_count = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    print(f'[PARAMS] frozen={frozen_count:,} new_trainable={new_count:,}', flush=True)
    return model


def save_checkpoint(model, path, args, epoch, selection=None, optimizer=None):
    payload = {'version': VERSION, 'epoch': epoch,
               'base_checkpoint': str(args.base_checkpoint.resolve()),
               'q3_checkpoint': str(args.q3_checkpoint.resolve()),
               'width': args.width, 'steps': 3, 'selection': selection,
               'modules': {name: getattr(model, name).state_dict() for name in MODULE_NAMES}}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def gradient_norm(module):
    squared = [parameter.grad.detach().norm().square()
               for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(squared).sum().sqrt()) if squared else 0.


def batch_summary(result, threshold=.6):
    with torch.no_grad():
        p0, p1 = result['p0'], result['p1']
        return {'r_mean': float(result['r'].mean()),
                'r_near_zero': float((result['r'] < .01).float().mean()),
                'a_mean': float(result['a'].mean()),
                'a_near_one': float((result['a'] > .99).float().mean()),
                'b_minus_p0_mean_abs': float((result['b'] - result['u0']).abs().mean()),
                'w_direction_means': result['w'].mean((0, 2, 3)).cpu().tolist(),
                'w_sum_mean': float(result['w'].sum(1).mean()),
                'step_changes': [float((result[f'u{index}'] -
                                        result[f'u{index-1}']).abs().mean())
                                 for index in (1, 2, 3)],
                'added_fraction': float(((p1 >= threshold) & (p0 < threshold)).float().mean()),
                'removed_fraction': float(((p1 < threshold) & (p0 >= threshold)).float().mean())}


def smoke(args, model):
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
                                  source_patch_size=1500, random_crop_train=True)
    sample = dataset[0]
    image = sample['image'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    mask = sample['mask'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    skeleton = sample['skeleton'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    model.train()
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    inner = model.frozen.base.swin_unet
    calls = {'local': 0, 'gap': 0, 'p2': 0, 'h3_surface_fusion': 0}
    handles = []
    def register(module, name):
        handles.append(module.register_forward_hook(
            lambda unused_module, unused_input, unused_output:
            calls.__setitem__(name, calls[name] + 1)))
    register(inner.decoder_structure_blocks['2'].local_propagation, 'local')
    register(inner.gap_connection, 'gap')
    register(inner.decoder_structure_blocks['2'].p2_head, 'p2')
    register(inner.surface_h3_fusion, 'h3_surface_fusion')
    start = time.perf_counter()
    try:
        result = model(image, capture_diagnostics=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(calls[name] for name in ('local', 'gap', 'p2')) or calls['h3_surface_fusion'] != 1:
        raise RuntimeError(f'Unexpected frozen base path calls: {calls}')
    loss, terms = dynamic_losses(result, mask, skeleton,
                                 lambda_bg=args.lambda_bg, lambda_cont=args.lambda_cont)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Non-finite smoke loss')
    loss.backward()
    if args.device.type == 'cuda':
        torch.cuda.synchronize(args.device)
    gradients = {name: gradient_norm(getattr(model, name)) for name in MODULE_NAMES}
    required_gradients = ('visual', 'reliable_head', 'relations', 'proposal_head',
                          'accept_head', 'dynamics', 'detail_head')
    if any(gradients[name] <= 0 or not np.isfinite(gradients[name])
           for name in required_gradients):
        raise RuntimeError(f'New repair module has zero/nonfinite gradient: {gradients}')
    if any(parameter.grad is not None for parameter in model.frozen.parameters()):
        raise RuntimeError('Frozen encoder/Q3 received gradients')
    for key in ('r', 'w', 'b', 'a', 'u0', 'u1', 'u2', 'u3', 'p1'):
        value = result[key]
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool((value > 1).any()):
            raise RuntimeError(f'Invalid probability in {key}')
    if bool((result['w'].sum(1) > 1.000001).any()):
        raise RuntimeError('Directional weights have no reject/no-transport mass')
    if not bool((result['source_available'] >= 0).all()):
        raise RuntimeError('Invalid source availability')
    with torch.no_grad():
        without_w, _, _ = model.dynamics(
            result['visual'], result['u0'], result['q'], result['r'],
            result['b'], result['a'], torch.zeros_like(result['w']))
        w_effect = float((without_w[-1] - result['u3']).abs().mean())
    w_gradient = gradient_norm(model.relations.direction_head)
    if w_gradient <= 0 or w_effect <= 0:
        raise RuntimeError(f'W does not affect U3: gradient={w_gradient}, effect={w_effect}')
    report = {'shapes': {key: list(result[key].shape) for key in
                         ('z0', 'q', 'r', 'w', 'b', 'a', 'u0', 'u1', 'u2', 'u3', 'p1')},
              'loss': float(loss), 'seg': float(terms['seg']),
              'cont': float(terms['cont']), 'gt_paths': terms['paths'],
              'gradient_norms': gradients, 'initial': batch_summary(result),
              'w_head_gradient_norm': w_gradient, 'w_to_u3_mean_abs': w_effect,
              'peak_cuda_mib': (torch.cuda.max_memory_allocated(args.device) / 2**20
                                if args.device.type == 'cuda' else None),
              'batch_forward_backward_seconds': time.perf_counter() - start,
              'frozen_gradient_count': 0,
              'frozen_base_calls': calls,
              'old_stage2_propagation_enabled': False, 'old_gap_enabled': False,
              'old_r3_computed': False}
    print(json.dumps(report, indent=2), flush=True)


def path_length_bin(length):
    return '3-7' if length < 8 else '8-15' if length < 16 else '16-31' if length < 32 else '32+'


def gap_lengths(p0, p1, gt_skeleton, gt_mask):
    covered = cv2.dilate(p0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    missing = (gt_skeleton & ~covered).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, connectivity=8)
    _, p0_labels = cv2.connectedComponents(p0.astype(np.uint8), connectivity=8)
    _, p1_labels = cv2.connectedComponents(p1.astype(np.uint8), connectivity=8)
    result = {name: {'eligible': 0, 'closed': 0} for name in ('3-7', '8-15', '16-31', '32+')}
    for index in range(1, count):
        length = int(stats[index, cv2.CC_STAT_AREA])
        if length < 3:
            continue
        x, y, width, height = (int(value) for value in stats[index, :4])
        x0, x1 = max(x-7, 0), min(x+width+7, labels.shape[1])
        y0, y1 = max(y-7, 0), min(y+height+7, labels.shape[0])
        region = labels[y0:y1, x0:x1] == index
        flank = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        old_region = p0_labels[y0:y1, x0:x1]
        new_region = p1_labels[y0:y1, x0:x1]
        ground_region = gt_mask[y0:y1, x0:x1]
        sources = np.unique(old_region[flank & ground_region])
        sources = sources[sources > 0]
        if len(sources) < 2:
            continue
        name = path_length_bin(length)
        result[name]['eligible'] += 1
        if p1[y0:y1, x0:x1][region].mean() >= .8:
            connected = [set(np.unique(new_region[flank & (old_region == label) &
                                                  ground_region])) - {0}
                         for label in sources]
            result[name]['closed'] += int(bool(set.intersection(*connected)))
    new_background = (p1 & ~p0 & ~(cv2.dilate(
        gt_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        new_background.astype(np.uint8), connectivity=8)
    bridge_count = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        x, y, width, height = (int(value) for value in stats[index, :4])
        x0, x1 = max(x-2, 0), min(x+width+2, labels.shape[1])
        y0, y1 = max(y-2, 0), min(y+height+2, labels.shape[0])
        local = labels[y0:y1, x0:x1] == index
        near = cv2.dilate(local.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sources = np.unique(p0_labels[y0:y1, x0:x1][near])
        bridge_count += int((sources > 0).sum() >= 2)
    return result, bridge_count


def selection_score(metrics, diagnostics, images):
    return (metrics['iou'] + .15 * metrics['cldice'] +
            .1 * metrics['apls_gt_to_pred_approx'] +
            .05 * metrics['apls_bidirectional_approx'] -
            .001 * metrics['short_pred_components'] -
            .01 * diagnostics['background_bridge_proxy'] / max(images, 1))


@torch.no_grad()
def evaluate_thresholds(model, dataset, device, thresholds, max_images=0):
    model.eval()
    limit = min(len(dataset), max_images or len(dataset))
    storage = {threshold: {'p0': [], 'p1': [], 'paired': [], 'bins':
                           {name: {'eligible': 0, 'closed': 0}
                            for name in ('3-7', '8-15', '16-31', '32+')}} for threshold in thresholds}
    reference_tiles = []
    start = time.perf_counter()
    for index in range(limit):
        sample = dataset[index]
        z0, z1, _ = stitch_three(model, sample['image'][None].to(device))
        p0_prob = 1. / (1. + np.exp(-np.clip(z0, -80., 80.)))
        p1_prob = 1. / (1. + np.exp(-np.clip(z1, -80., 80.)))
        gt = sample['mask'][0].numpy() > .5
        prepared = prepare_target(gt, PROTOCOL)
        for threshold in thresholds:
            data = storage[threshold]
            p0, p1 = p0_prob >= threshold, p1_prob >= threshold
            data['p0'].append({'case_id': sample['case_name'],
                               **image_metrics(p0, prepared, PROTOCOL)})
            data['p1'].append({'case_id': sample['case_name'],
                               **image_metrics(p1, prepared, PROTOCOL)})
            by_length, bridges = gap_lengths(p0, p1, prepared['skeleton'], gt)
            eligible = sum(values['eligible'] for values in by_length.values())
            closed = sum(values['closed'] for values in by_length.values())
            data['paired'].append({'case_id': sample['case_name'],
                                   'eligible_p0_gaps': eligible, 'closed_p0_gaps': closed,
                                   'background_bridge_proxy': bridges,
                                   'added_gt_pixels': int((p1 & ~p0 & gt).sum()),
                                   'added_background_pixels': int((p1 & ~p0 & ~gt).sum()),
                                   'removed_gt_pixels': int((p0 & ~p1 & gt).sum()),
                                   'removed_background_pixels': int((p0 & ~p1 & ~gt).sum()),
                                   'probability_change_mean': float(np.abs(p1_prob-p0_prob).mean())})
            for name, values in by_length.items():
                data['bins'][name]['eligible'] += values['eligible']
                data['bins'][name]['closed'] += values['closed']
        if len(thresholds) == 1:
            threshold = thresholds[0]
            missing = prepared['skeleton'] & ~(cv2.dilate(
                (p0_prob >= threshold).astype(np.uint8),
                np.ones((5, 5), np.uint8)) > 0)
            positions = sliding_positions(1500, 512, 256)
            best_tile = max(((int(missing[y:y+512, x:x+512].sum()), y, x)
                             for y in positions for x in positions))
            reference_tiles.append((best_tile[0], index, best_tile[1], best_tile[2]))
        print(f'[EVAL] {index + 1}/{limit}', flush=True)
    reports = {}
    for threshold, data in storage.items():
        paired = {name: sum(item[name] for item in data['paired']) for name in
                  ('eligible_p0_gaps', 'closed_p0_gaps', 'background_bridge_proxy',
                   'added_gt_pixels', 'added_background_pixels',
                   'removed_gt_pixels', 'removed_background_pixels')}
        paired['gap_closure_rate_proxy'] = (paired['closed_p0_gaps'] /
                                             max(paired['eligible_p0_gaps'], 1))
        paired['probability_change_mean'] = float(np.mean(
            [item['probability_change_mean'] for item in data['paired']]))
        bins = {name: {**counts, 'closure_rate_proxy':
                       counts['closed'] / max(counts['eligible'], 1)}
                for name, counts in data['bins'].items()}
        p0_summary, p1_summary = summarize(data['p0']), summarize(data['p1'])
        reports[threshold] = {'threshold': threshold, 'images': limit,
                              'p0': p0_summary, 'p1': p1_summary,
                              'paired': paired, 'gap_by_length': bins,
                              'selection_score': selection_score(p1_summary, paired, limit),
                              'metric_note': 'Raster APLS approximation; gap closure and '
                                             'background bridge are diagnostic proxies.'}
    return reports, storage, sorted(reference_tiles, reverse=True), time.perf_counter() - start


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(output, report, details):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name in ('p0', 'p1', 'paired'):
        write_rows(output / f'{name}_per_image.csv', details[name])


def train(args, model):
    if args.mode == 'train':
        args.output_dir.mkdir(parents=True, exist_ok=False)
        start_epoch, best_score = 1, -float('inf')
    else:
        if not args.dynamic_checkpoint:
            raise ValueError('Resume needs --dynamic_checkpoint')
        payload = torch.load(args.dynamic_checkpoint, map_location='cpu', weights_only=False)
        start_epoch = int(payload['epoch']) + 1
        best_score = float(payload.get('selection') or -float('inf'))
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
                                  source_patch_size=1500, random_crop_train=True,
                                  random_crops_per_image=1, random_crop_seed=args.seed,
                                  augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=.01)
    if args.mode == 'resume':
        if 'optimizer' not in payload:
            raise ValueError('Resume checkpoint lacks optimizer state')
        optimizer.load_state_dict(payload['optimizer'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)
        sums = {'seg': 0., 'cont': 0., 'total': 0., 'r_mean': 0.,
                'r_near_zero': 0., 'a_mean': 0., 'a_near_one': 0.,
                'b_minus_p0_mean_abs': 0., 'w_sum_mean': 0.,
                'added_fraction': 0., 'removed_fraction': 0.}
        grad_sums = {name: 0. for name in MODULE_NAMES}
        w_gradient_sum = 0.
        direction_sums = np.zeros(8, dtype=np.float64)
        step_sums = [0., 0., 0.]
        start = time.perf_counter()
        for batch in loader:
            image = batch['image'].to(args.device, non_blocking=True)
            mask = batch['mask'].to(args.device, non_blocking=True)
            skeleton = batch['skeleton'].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = model(image)
            loss, terms = dynamic_losses(result, mask, skeleton,
                                         lambda_bg=args.lambda_bg,
                                         lambda_cont=args.lambda_cont)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            for name in MODULE_NAMES:
                grad_sums[name] += gradient_norm(getattr(model, name))
            w_gradient_sum += gradient_norm(model.relations.direction_head)
            optimizer.step()
            snapshot = batch_summary(result)
            for name in sums:
                sums[name] += (float(loss) if name == 'total'
                               else float(terms[name]) if name in terms else snapshot[name])
            for index, value in enumerate(snapshot['step_changes']):
                step_sums[index] += value
            direction_sums += np.asarray(snapshot['w_direction_means'])
        count = len(loader)
        report = {'epoch': epoch, **{key: value/count for key, value in sums.items()},
                  'gradients': {key: value/count for key, value in grad_sums.items()},
                  'w_head_gradient': w_gradient_sum/count,
                  'w_direction_means': (direction_sums/count).tolist(),
                  'step_changes': [value/count for value in step_sums],
                  'train_seconds': time.perf_counter() - start}
        print('[TRAIN] ' + json.dumps(report), flush=True)
        with (args.output_dir / 'epoch_metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(report) + '\n')
        save_checkpoint(model, args.output_dir / 'last.pth', args, epoch,
                        best_score, optimizer)
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            reports, _, _, elapsed = evaluate_thresholds(
                model, full_dataset(args.root_path, 'val'), args.device,
                (.6,), args.max_images)
            evaluation = reports[.6]
            score = evaluation['selection_score']
            evaluation['epoch'] = epoch
            evaluation['eval_seconds'] = elapsed
            (args.output_dir / f'val_epoch_{epoch:03d}.json').write_text(
                json.dumps(evaluation, indent=2), encoding='utf-8')
            print(f'[VAL] epoch={epoch} score={score:.6f} '
                  f'IoU={evaluation["p1"]["iou"]:.5f} '
                  f'clDice={evaluation["p1"]["cldice"]:.5f} '
                  f'APLS_GT={evaluation["p1"]["apls_gt_to_pred_approx"]:.5f} '
                  f'short={evaluation["p1"]["short_pred_components"]:.2f}', flush=True)
            if score > best_score:
                best_score = score
                save_checkpoint(model, args.output_dir / 'best.pth', args,
                                epoch, score)
            save_checkpoint(model, args.output_dir / 'last.pth', args, epoch,
                            best_score, optimizer)


def color_map(value, size=(512, 512)):
    value = cv2.resize(value.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(np.uint8(np.clip(value, 0., 1.) * 255), cv2.COLORMAP_TURBO)


@torch.no_grad()
def save_visualizations(model, dataset, tiles, output, device, threshold, count=4):
    output.mkdir(parents=True, exist_ok=True)
    for ordinal, (_, index, top, left) in enumerate(tiles[:count]):
        sample = dataset[index]
        result = model(sample['image'][None, :, top:top+512, left:left+512].to(device),
                       capture_diagnostics=True)
        arr = {key: value[0].float().cpu().numpy() for key, value in result.items()
               if torch.is_tensor(value) and value.ndim == 4}
        image = sample['image'][:, top:top+512, left:left+512].numpy().transpose(1, 2, 0)
        image = np.uint8(np.clip((image * [.229, .224, .225] +
                                  [.485, .456, .406]) * 255, 0, 255))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gt = sample['mask'][0, top:top+512, left:left+512].numpy()
        p0, p1 = arr['p0'][0], arr['p1'][0]
        gt_skeleton = sample['skeleton'][0, top:top+512, left:left+512].numpy() > .5
        missed = gt_skeleton & ~(cv2.dilate(
            (p0 >= threshold).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        work = lambda key: arr[key][0]
        panels = [('image', image), ('GT', color_map(gt)),
                  ('GT missed', color_map(missed.astype(np.float32))),
                  ('P0', color_map(p0)), ('Q3', color_map(work('q'))),
                  ('R', color_map(work('r'))),
                  ('E_two', color_map(arr['e_two'].mean(0))),
                  ('E_one', color_map(arr['e_one'].mean(0))),
                  ('W_sum', color_map(arr['w'].sum(0))),
                  ('B', color_map(work('b'))), ('A', color_map(work('a'))),
                  ('U0', color_map(work('u0'))), ('U1', color_map(work('u1'))),
                  ('U2', color_map(work('u2'))), ('U3', color_map(work('u3'))),
                  ('P1', color_map(p1)),
                  ('P1-P0', color_map(np.clip((p1-p0+1.)/2., 0., 1.))),
                  ('new road', color_map(((p1>=threshold)&(p0<threshold)).astype(np.float32)))]
        panel_images = []
        for name, bitmap in panels:
            bitmap = cv2.resize(bitmap, (320, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(bitmap, name, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        .75, (255, 255, 255), 2)
            panel_images.append(bitmap)
        while len(panel_images) % 5:
            panel_images.append(np.zeros_like(panel_images[0]))
        rows = [np.concatenate(panel_images[offset:offset+5], axis=1)
                for offset in range(0, len(panel_images), 5)]
        prefix = f'{ordinal:02d}_{sample["case_name"]}_{top}_{left}'
        cv2.imwrite(str(output / f'{prefix}.png'), np.concatenate(rows, axis=0))
        for direction in range(8):
            cv2.imwrite(str(output / f'{prefix}_W{direction}.png'),
                        color_map(arr['w'][direction]))
        for axis in range(4):
            cv2.imwrite(str(output / f'{prefix}_E_two_axis{axis}.png'),
                        color_map(arr['e_two'][axis]))
            cv2.imwrite(str(output / f'{prefix}_E_one_axis{axis}.png'),
                        color_map(arr['e_one'][axis]))


def evaluate(args, model):
    dataset = full_dataset(args.root_path, args.mode)
    if args.mode == 'val':
        reports, _, _, elapsed = evaluate_thresholds(
            model, dataset, args.device, THRESHOLDS, args.max_images)
        threshold = max(reports, key=lambda value: reports[value]['selection_score'])
        selection = {'selected_on': 'val', 'threshold': threshold,
                     'selection_score': reports[threshold]['selection_score'],
                     'dynamic_checkpoint': str(args.dynamic_checkpoint.resolve()),
                     'base_checkpoint': str(args.base_checkpoint.resolve()),
                     'q3_checkpoint': str(args.q3_checkpoint.resolve()),
                     'sweep': {str(key): value for key, value in reports.items()},
                     'sweep_seconds': elapsed}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'val_selection.json').write_text(
            json.dumps(selection, indent=2), encoding='utf-8')
    else:
        selection = json.loads((args.output_dir / 'val_selection.json').read_text())
        if (selection['selected_on'] != 'val' or
                selection['dynamic_checkpoint'] != str(args.dynamic_checkpoint.resolve()) or
                selection['base_checkpoint'] != str(args.base_checkpoint.resolve()) or
                selection['q3_checkpoint'] != str(args.q3_checkpoint.resolve())):
            raise ValueError('Test requires the matching validation threshold selection')
        threshold = float(selection['threshold'])
    reports, details, tiles, elapsed = evaluate_thresholds(
        model, dataset, args.device, (threshold,), args.max_images)
    report = reports[threshold]
    report['evaluation_seconds'] = elapsed
    report['version'] = VERSION
    report['base_checkpoint'] = str(args.base_checkpoint.resolve())
    report['q3_checkpoint'] = str(args.q3_checkpoint.resolve())
    report['checkpoint'] = str(args.dynamic_checkpoint.resolve())
    report['protocol'] = PROTOCOL
    write_evaluation(args.output_dir / args.mode, report, details[threshold])
    if args.mode == 'val' and args.max_images == 0:
        save_visualizations(model, dataset, tiles, args.output_dir / 'val' / 'dynamics',
                            args.device, threshold, args.visualizations)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'train', 'resume', 'val', 'test'))
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--q3_checkpoint', type=Path, required=True)
    parser.add_argument('--dynamic_checkpoint', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_bg', type=float, default=1.)
    parser.add_argument('--lambda_cont', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--visualizations', type=int, default=4)
    args = parser.parse_args()
    if (args.width % 8 or args.width < 8 or args.batch_size < 1 or args.epochs < 1 or
            args.val_interval < 1 or args.lambda_bg < 0 or args.lambda_cont < 0 or
            args.max_images < 0 or args.visualizations < 0):
        parser.error('Invalid model, training, or evaluation configuration')
    if args.mode in ('resume', 'val', 'test') and not args.dynamic_checkpoint:
        parser.error('--dynamic_checkpoint is required for resume/val/test')
    if args.mode == 'train' and args.dynamic_checkpoint:
        parser.error('Use resume to continue a directional dynamics checkpoint')
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.q3_checkpoint = args.q3_checkpoint.resolve()
    args.device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_model(args)
    if args.mode == 'smoke':
        smoke(args, model)
    elif args.mode in ('train', 'resume'):
        train(args, model)
    else:
        evaluate(args, model)


if __name__ == '__main__':
    main()
