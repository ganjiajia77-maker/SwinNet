"""Frozen-checkpoint evidence-flow diagnostics for the a4001b3 H0 model.

The training modes fit only lightweight readout heads; the a400 backbone is
strictly frozen. Test diagnostics use validation-selected thresholds and fixed
pixel sets derived from the direction-on checkpoint pass.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import a400_probe_experiment as base

CONTEXT_NAMES = ('H0_context3', 'H0_context5')
FUSION_NAMES = ('fusion2_decoder_in', 'fusion2_out',
                'fusion3_decoder_in', 'fusion3_out')
FUSION_H_NAMES = ('fusion2_H_input', 'fusion3_H_input')
READOUT_NAMES = ('H_E1_probe', *CONTEXT_NAMES, *FUSION_NAMES)
FLOW_H_NAMES = ('H0', 'H_E1', 'H_E2', 'H_E3')


class ContextHeads(nn.Module):
    def __init__(self, channels, device):
        super().__init__()
        self.context = nn.ModuleDict({
            'H0_context3': nn.Sequential(nn.Conv2d(channels['H0'], 32, 3, padding=1),
                                         nn.GELU(), nn.Conv2d(32, 1, 1)),
            'H0_context5': nn.Sequential(nn.Conv2d(channels['H0'], 32, 5, padding=2),
                                         nn.GELU(), nn.Conv2d(32, 1, 1)),
        })
        self.pointwise = nn.ModuleDict({name: nn.Conv2d(channels[name], 1, 1)
                                        for name in ('H_E1_probe', *FUSION_NAMES)})
        self.to(device)

    def forward(self, features):
        result = {name: self.context[name](features['H0']) for name in CONTEXT_NAMES}
        result['H_E1_probe'] = self.pointwise['H_E1_probe'](features['H_E1'])
        result.update({name: self.pointwise[name](features[name]) for name in FUSION_NAMES})
        return result


def _threshold_row(prob, target):
    pred = prob >= 0.5
    tp, fp, fn = int((pred & target).sum()), int((pred & ~target).sum()), int((~pred & target).sum())
    return base.scores(tp, fp, fn)


def fit_readouts(args, tap, backbone_sha, device):
    out = args.readout_dir
    out.mkdir(parents=True, exist_ok=False)
    train_set = base.RoadOnlyCrop(args.root_path, 'train', args.seed, args.max_train_images)
    val_set = base.RoadOnlyCrop(args.root_path, 'val', args.seed, 0)
    workers = args.num_workers
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True,
                            persistent_workers=workers > 0)
    sample, _ = train_set[0]
    sample_features, _ = tap(sample[None].to(device))
    channels = {name: int(sample_features[name].shape[1])
                for name in ('H0', 'H_E1', *FUSION_NAMES)}
    heads = ContextHeads(channels, device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    best_loss, best_epoch, best_state = float('inf'), 0, None
    history = []
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        heads.train()
        train_loss = 0.0
        count = 0
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            with torch.no_grad():
                features, _ = tap(images)
            optimizer.zero_grad(set_to_none=True)
            logits = heads(features)
            losses = [base.road_loss(base.F.interpolate(logits[name], size=base.SIZE,
                                                        mode='bilinear', align_corners=False), target)
                      for name in READOUT_NAMES]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            count += 1
        heads.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for images, target in val_loader:
                images, target = images.to(device), target.to(device)
                features, _ = tap(images)
                logits = heads(features)
                val_loss += float(torch.stack([
                    base.road_loss(base.F.interpolate(logits[name], size=base.SIZE,
                                                      mode='bilinear', align_corners=False), target)
                    for name in READOUT_NAMES]).mean())
                val_count += 1
        val_loss /= max(val_count, 1)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss / max(count, 1),
                        'val_loss': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in heads.state_dict().items()}
        print(f'readout epoch={epoch + 1} train={history[-1]["train_loss"]:.5f} '
              f'val={val_loss:.5f} best={best_epoch}', flush=True)
    if best_state is None:
        raise RuntimeError('No readout checkpoint was selected on validation loss')
    heads.load_state_dict(best_state, strict=True)
    heads.eval()

    # Select a separate global-IoU threshold for every new readout on native
    # validation imagery, using the same weighted-logit 512/256 stitcher.
    totals = {name: np.zeros((len(base.THRESHOLDS), 3), dtype=np.int64) for name in READOUT_NAMES}
    for index, (case, rgb, target) in enumerate(base.native_cases(args.root_path, 'val'), 1):
        maps = native_readouts(tap, heads, rgb, device)
        for name in READOUT_NAMES:
            for ti, threshold in enumerate(base.THRESHOLDS):
                pred = maps[name] >= threshold
                totals[name][ti] += (int((pred & target).sum()), int((pred & ~target).sum()),
                                     int((~pred & target).sum()))
        print(f'readout validation {index}: {case}', flush=True)
    selected = {}
    for name in READOUT_NAMES:
        records = [(base.scores(*row), threshold)
                   for row, threshold in zip(totals[name], base.THRESHOLDS)]
        metric, threshold = max(records, key=lambda item: (item[0]['iou'], -item[1]))
        selected[name] = {'threshold': threshold, 'val_iou': metric['iou'], 'val_f1': metric['f1']}
    payload = {'schema': 1, 'base_commit': base.BASE_COMMIT,
               'backbone_sha256': backbone_sha, 'feature_channels': channels,
               'context_kernel_sizes': [3, 5], 'heads_state': best_state,
               'best_epoch': best_epoch, 'best_val_loss': best_loss,
               'thresholds_selected_on_val': selected,
               'protocol': 'frozen backbone; train-split crops; global IoU thresholds on native val tiles'}
    torch.save(payload, out / 'readouts_best.pth')
    base.write_csv(out / 'readout_training_history.csv', history)
    base.write_json(out / 'readout_val_selection.json', {
        'backbone_sha256': backbone_sha, 'selected_on': 'val', 'criterion': 'global_iou',
        'thresholds': selected, 'best_epoch': best_epoch, 'best_val_loss': best_loss})
    print(f'Saved frozen-backbone readouts to {out}', flush=True)


@torch.no_grad()
def native_readouts(tap, heads, rgb, device, direction_off=False):
    tap.direction_off = direction_off
    weight = base.blend_weight(base.SIZE, device)
    denominator = torch.zeros((1, 1, base.NATIVE, base.NATIVE), device=device)
    canvases = {name: torch.zeros_like(denominator) for name in READOUT_NAMES}
    image = base.normalize(rgb).unsqueeze(0).to(device)
    for top in base.sliding_positions(base.NATIVE, base.SIZE, base.STRIDE):
        for left in base.sliding_positions(base.NATIVE, base.SIZE, base.STRIDE):
            features, _ = tap(image[:, :, top:top + base.SIZE, left:left + base.SIZE])
            logits = heads(features)
            region = (slice(None), slice(None), slice(top, top + base.SIZE), slice(left, left + base.SIZE))
            denominator[region] += weight
            for name in READOUT_NAMES:
                value = logits[name]
                if value.shape[-2:] != (base.SIZE, base.SIZE):
                    value = F.interpolate(value, size=base.SIZE, mode='bilinear', align_corners=False)
                canvases[name][region] += value.float() * weight
    tap.direction_off = False
    return {name: torch.sigmoid(canvas / denominator)[0, 0].cpu().numpy()
            for name, canvas in canvases.items()}


def _native_all(tap, probe_heads, readouts, rgb, device, direction_off=False):
    tap.direction_off = direction_off
    weight = base.blend_weight(base.SIZE, device)
    denominator = torch.zeros((1, 1, base.NATIVE, base.NATIVE), device=device)
    names = (*base.ALL_NAMES, 'trace_H_E1', 'trace_H_E2', 'trace_H_E3',
             *FUSION_H_NAMES, *READOUT_NAMES)
    canvases = {name: torch.zeros_like(denominator) for name in names}
    image = base.normalize(rgb).unsqueeze(0).to(device)
    for top in base.sliding_positions(base.NATIVE, base.SIZE, base.STRIDE):
        for left in base.sliding_positions(base.NATIVE, base.SIZE, base.STRIDE):
            features, surface = tap(image[:, :, top:top + base.SIZE, left:left + base.SIZE])
            outputs = {name: base.probe_logits(probe_heads[name], features[name]) for name in base.PROBES}
            outputs['original_surface'] = surface
            readout_logits = readouts(features)
            outputs['trace_H_E1'] = F.interpolate(readout_logits['H_E1_probe'], size=base.SIZE,
                                                   mode='bilinear', align_corners=False)
            outputs['trace_H_E2'] = base.probe_logits(probe_heads['H_E2'], features['H_E2'])
            outputs['trace_H_E3'] = base.probe_logits(probe_heads['H_E3'], features['H_E3'])
            outputs['fusion2_H_input'] = base.probe_logits(
                probe_heads['H_E3'], features['fusion2_h_in'])
            outputs['fusion3_H_input'] = base.probe_logits(
                probe_heads['H_E3'], features['fusion3_h_in'])
            outputs.update({name: readout_logits[name]
                            for name in (*CONTEXT_NAMES, *FUSION_NAMES)})
            region = (slice(None), slice(None), slice(top, top + base.SIZE), slice(left, left + base.SIZE))
            denominator[region] += weight
            for name, logits in outputs.items():
                if logits.shape[-2:] != (base.SIZE, base.SIZE):
                    logits = F.interpolate(logits, size=base.SIZE, mode='bilinear', align_corners=False)
                canvases[name][region] += logits.float() * weight
    tap.direction_off = False
    return {name: torch.sigmoid(canvas / denominator)[0, 0].cpu().numpy()
            for name, canvas in canvases.items()}


def trace_test(args, tap, probe_heads, probe_payload, backbone_sha, device):
    old_selection_path = args.run_dir / 'val_selection.json'
    old_selection = json.loads(old_selection_path.read_text(encoding='utf-8'))
    if (old_selection.get('backbone_sha256') != backbone_sha
            or old_selection.get('selected_on') != 'val'
            or old_selection.get('criterion') != 'global_iou'
            or set(old_selection.get('models', {})) != set(base.ALL_NAMES)):
        raise ValueError('Existing 1x1 probe thresholds do not match the pinned checkpoint')
    readout_path = args.readout_dir / 'readouts_best.pth'
    payload = torch.load(readout_path, map_location='cpu', weights_only=False)
    if payload.get('backbone_sha256') != backbone_sha:
        raise ValueError('Context/fusion readouts belong to another checkpoint')
    channels = payload['feature_channels']
    readouts = ContextHeads(channels, device)
    readouts.load_state_dict(payload['heads_state'], strict=True)
    readouts.eval()
    thresholds = {name: float(old_selection['models'][name]['threshold']) for name in base.ALL_NAMES}
    thresholds.update({name: float(value['threshold'])
                       for name, value in payload['thresholds_selected_on_val'].items()})
    from evaluation.road_comparison.road_metrics import skeletonize
    out = args.run_dir / 'evidence_flow'
    out.mkdir(parents=True, exist_ok=False)
    summary = {'schema': 1, 'backbone_sha256': backbone_sha,
               'probe_checkpoint_sha256': base.file_sha256(args.run_dir / 'probe_best.pth'),
               'readout_checkpoint_sha256': base.file_sha256(readout_path),
               'thresholds_selected_on_val': thresholds,
               'definitions': {
                   'H0_exclusive': 'baseline_missed AND H0 >= H0 val threshold AND every other existing 1x1 probe < its own val threshold',
                   'all_unrecognized': 'baseline_missed AND every existing 1x1 probe, including H0, < its own val threshold',
                   'sets_are_fixed_from': 'direction-on pass of this exact a4001b3 checkpoint',
                   'direction_off': 'forward hook bypasses E2DirectionAggregation by returning its input; all weights/checkpoint unchanged'},
               'expected_pixel_counts_reference_only': {'H0_exclusive': 13930, 'all_unrecognized': 41181},
               'per_image': []}
    trace_rows, set_rows = [], []
    review_candidates = []
    test_pred = out / 'direction_predictions'
    for mode in ('on', 'off'):
        (test_pred / f'direction_{mode}' / 'surface').mkdir(parents=True, exist_ok=True)
    total = {mode: {'baseline_missed': 0, 'H0_exclusive': 0, 'all_unrecognized': 0,
                    }
             for mode in ('on', 'off')}
    for image_index, (case, rgb, target) in enumerate(base.native_cases(args.root_path, 'test'), 1):
        on = _native_all(tap, probe_heads, readouts, rgb, device, direction_off=False)
        off = _native_all(tap, probe_heads, readouts, rgb, device, direction_off=True)
        skeleton = skeletonize(target)
        base_missed = skeleton & (on['original_surface'] < thresholds['original_surface'])
        other = [name for name in base.PROBES if name != 'H0']
        h0_exclusive = base_missed & (on['H0'] >= thresholds['H0'])
        for name in other:
            h0_exclusive &= on[name] < thresholds[name]
        all_unrecognized = base_missed.copy()
        for name in base.PROBES:
            all_unrecognized &= on[name] < thresholds[name]
        sets = {'H0_exclusive': h0_exclusive, 'all_unrecognized': all_unrecognized}
        for mode in ('on', 'off'):
            total[mode]['baseline_missed'] += int(base_missed.sum())
        for name, mask in sets.items():
            n = int(mask.sum())
            total['on'][name] += n
            total['off'][name] += n
            idx = np.flatnonzero(mask)
            per_set = {'image': case, 'set': name, 'pixels': n,
                       'coordinate_sha256': hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()}
            for mode, maps in (('on', on), ('off', off)):
                for probe_name in ('H0', 'trace_H_E1', 'trace_H_E2', 'trace_H_E3',
                                   'F2', 'F3', 'original_surface',
                                   *FUSION_NAMES, *FUSION_H_NAMES, *CONTEXT_NAMES):
                    vals = maps[probe_name].ravel()[idx]
                    th = (thresholds['H_E1_probe'] if probe_name == 'trace_H_E1' else
                          thresholds['H_E2'] if probe_name == 'trace_H_E2' else
                          thresholds['H_E3'] if probe_name == 'trace_H_E3' else
                          thresholds['H_E3'] if probe_name in FUSION_H_NAMES else
                          thresholds.get(probe_name, .5))
                    per_set[f'{mode}_{probe_name}_threshold'] = th
                    hits = int((vals >= th).sum())
                    per_set[f'{mode}_{probe_name}_mean'] = float(vals.mean()) if n else None
                    per_set[f'{mode}_{probe_name}_hit_fraction'] = hits / n if n else None
            set_rows.append(per_set)

        # Deterministically collect spatially separated examples for human
        # visual review. The category is deliberately left blank; it is not
        # inferred from model scores or used for threshold/model selection.
        unresolved = all_unrecognized
        context_rescued = unresolved & (
            (on['H0_context3'] >= thresholds['H0_context3']) |
            (on['H0_context5'] >= thresholds['H0_context5']))
        still_missed = unresolved & ~context_rescued
        for group_name, group_mask in (('context_rescued', context_rescued),
                                       ('context_unresolved', still_missed)):
            ys, xs = np.nonzero(group_mask)
            if len(xs):
                rng = np.random.RandomState(int.from_bytes(
                    hashlib.sha256(f'{case}:{group_name}'.encode()).digest()[:4], 'little'))
                order = rng.permutation(len(xs))
                taken, last = 0, []
                for pos in order:
                    y, x = int(ys[pos]), int(xs[pos])
                    if any((y - py) ** 2 + (x - px) ** 2 < 96 ** 2 for py, px in last):
                        continue
                    review_candidates.append((case, rgb, target, on, group_name, y, x))
                    last.append((y, x)); taken += 1
                    if taken >= 4:
                        break

        # Stage-transition traces share fixed masks and val-selected thresholds.
        for set_name, mask in sets.items():
            idx = np.flatnonzero(mask)
            for mode, maps in (('on', on), ('off', off)):
                row = {'image': case, 'set': set_name, 'direction': mode, 'pixels': int(idx.size)}
                for name in ('H0', 'trace_H_E1', 'trace_H_E2', 'trace_H_E3', 'F2', 'F3',
                             'fusion2_decoder_in', 'fusion2_out', 'fusion3_decoder_in', 'fusion3_out',
                             'fusion2_H_input', 'fusion3_H_input',
                             'surface_prehead', 'original_surface', 'H0_context3', 'H0_context5'):
                    values = maps[name].ravel()[idx]
                    row[f'{name}_threshold'] = (
                        thresholds['H_E1_probe'] if name == 'trace_H_E1' else
                        thresholds['H_E2'] if name == 'trace_H_E2' else
                        thresholds['H_E3'] if name == 'trace_H_E3' else
                        thresholds['H_E3'] if name in FUSION_H_NAMES else
                        thresholds.get(name, .5))
                    row[f'{name}_mean'] = float(values.mean()) if idx.size else None
                    row[f'{name}_p50'] = float(np.quantile(values, .50)) if idx.size else None
                    row[f'{name}_p90'] = float(np.quantile(values, .90)) if idx.size else None
                # Background contrast uses every native GT-background pixel,
                # not the fixed missed-road sets.
                bg = ~target
                row['background_pixels'] = int(bg.sum())
                for name in ('fusion2_decoder_in', 'fusion2_out', 'fusion3_decoder_in', 'fusion3_out'):
                    row[f'{name}_background_mean'] = float(maps[name][bg].mean())
                    row[f'{name}_background_false_positive_rate'] = float((maps[name][bg] >= thresholds[name]).mean())
                trace_rows.append(row)

        # Save the two same-checkpoint E2 direction outputs for the standard
        # shared topology tool; threshold is inherited from validation.
        for mode, maps in (('on', on), ('off', off)):
            prediction = maps['original_surface'] >= thresholds['original_surface']
            cv2.imwrite(str(test_pred / f'direction_{mode}' / 'surface' / f'{case}_pred.png'),
                        prediction.astype(np.uint8) * 255)
            tp = int((prediction & target).sum())
            total[mode].setdefault('tp', 0); total[mode].setdefault('fn', 0); total[mode].setdefault('gt', 0)
            total[mode]['tp'] += tp; total[mode]['fn'] += int((~prediction & target).sum()); total[mode]['gt'] += int(target.sum())
        summary['per_image'].append({'image': case, 'baseline_missed': int(base_missed.sum()),
                                     'H0_exclusive': int(h0_exclusive.sum()),
                                     'all_unrecognized': int(all_unrecognized.sum()),
                                     'sets_disjoint': not bool((h0_exclusive & all_unrecognized).any()),
                                     'coordinate_sha256': {
                                         name: hashlib.sha256(np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest()
                                         for name, mask in sets.items()}})
        print(f'evidence-flow {image_index}: {case} H0-exclusive={int(h0_exclusive.sum())} '
              f'all-unrecognized={int(all_unrecognized.sum())}', flush=True)
    summary['totals'] = {mode: {key: value for key, value in stats.items() if key != 'sets'}
                         for mode, stats in total.items()}
    summary['direction_on_off_global_recall'] = {
        mode: stats['tp'] / stats['gt'] if stats.get('gt') else None for mode, stats in total.items()}
    summary['observed_pixel_counts'] = {
        'H0_exclusive': total['on']['H0_exclusive'],
        'all_unrecognized': total['on']['all_unrecognized']}
    summary['expected_counts_match'] = {
        'H0_exclusive_13930': total['on']['H0_exclusive'] == 13930,
        'all_unrecognized_41181': total['on']['all_unrecognized'] == 41181}
    base.write_csv(out / 'fixed_set_stage_probability_traces.csv', trace_rows)
    base.write_csv(out / 'fixed_set_probe_response.csv', set_rows)
    review_dir = out / 'representative_cases'
    review_dir.mkdir()
    review_rows = []
    for case, rgb, target, maps, group_name, y, x in review_candidates:
        half = 96
        y0, y1 = max(0, y - half), min(base.NATIVE, y + half)
        x0, x1 = max(0, x - half), min(base.NATIVE, x + half)
        patch = rgb[y0:y1, x0:x1].copy()
        gt = target[y0:y1, x0:x1]
        pred3 = maps['H0_context3'][y0:y1, x0:x1] >= thresholds['H0_context3']
        pred5 = maps['H0_context5'][y0:y1, x0:x1] >= thresholds['H0_context5']
        # Overlay GT in red and either local-context prediction in green.
        overlay = patch.astype(np.float32)
        overlay[gt, 0] = np.minimum(255, overlay[gt, 0] * .45 + 140)
        overlay[pred3 | pred5, 1] = np.minimum(255, overlay[pred3 | pred5, 1] * .45 + 140)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.circle(overlay, (x - x0, y - y0), 5, (255, 255, 0), 1)
        filename = f'{case}_{group_name}_{y}_{x}.png'
        if not cv2.imwrite(str(review_dir / filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise OSError(f'Could not write representative example {filename}')
        review_rows.append({'image': case, 'group': group_name, 'row': y, 'column': x,
                            'patch_path': str(review_dir / filename),
                            'context3_probability': float(maps['H0_context3'][y, x]),
                            'context5_probability': float(maps['H0_context5'][y, x]),
                            'review_category': '',
                            'category_options': 'visible_thin_road;occlusion;intersection;possible_label_ambiguity;other'})
    base.write_csv(out / 'representative_cases.csv', review_rows)
    manifest = {
        'data_root': str(args.root_path),
        'protocol': {'image_size': base.NATIVE, 'short_area_threshold': 20,
                     'apls_max_nodes': 64, 'apls_snap_radius': 5.0},
        'models': []}
    for mode in ('on', 'off'):
        manifest['models'].append({
            'name': f'a400_direction_{mode}',
            'provenance': {'backbone_sha256': backbone_sha,
                           'direction_aggregation': mode,
                           'readout_probe_sha256': base.file_sha256(args.run_dir / 'probe_best.pth')},
            'validation_selection': {'selected_on': 'val', 'criterion': 'global_iou',
                                     'threshold': thresholds['original_surface'],
                                     'evidence': str(old_selection_path)},
            'predictions': {'test': [{'threshold': thresholds['original_surface'],
                                     'path': str(test_pred / f'direction_{mode}' / 'surface')}]}})
    base.write_json(out / 'direction_on_off_manifest.json', manifest)
    base.write_json(out / 'evidence_flow_summary.json', summary)
    print(json.dumps({'out': str(out), 'observed_pixel_counts': summary['observed_pixel_counts'],
                      'global_recall': summary['direction_on_off_global_recall']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_repo', type=Path, required=True,
                        help='exact a4001b3 source checkout')
    parser.add_argument('mode', choices=('fit_readouts', 'trace_test'))
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--run_dir', type=Path, required=True,
                        help='existing ten-probe run with probe_best.pth and val_selection.json')
    parser.add_argument('--readout_dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_train_images', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    args.probe_repo = args.probe_repo.resolve(strict=True)
    args.model_path = args.model_path.resolve(strict=True)
    args.root_path = args.root_path.resolve(strict=True)
    args.run_dir = args.run_dir.resolve(strict=True)
    args.readout_dir = args.readout_dir.resolve()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        parser.error('This a400 diagnostic expects CUDA')
    backbone_sha = base.file_sha256(args.model_path)
    model = base.load_backbone(args.model_path, device)
    tap = base.FeatureTap(model)
    try:
        if args.mode == 'fit_readouts':
            fit_readouts(args, tap, backbone_sha, device)
        else:
            heads, payload = base.load_heads(args.run_dir / 'probe_best.pth', backbone_sha, device)
            trace_test(args, tap, heads, payload, backbone_sha, device)
    finally:
        tap.close()


if __name__ == '__main__':
    main()
