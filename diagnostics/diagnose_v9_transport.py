"""Read-only V9 gradient/transport audit; full validation, fixed P0 reference.

Run val_direct first. This file never changes hashed training sources, weights,
optimizer state, or training/evaluation modes stored on disk.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import run_reliable_v9_mass as v9
from mass_protocol import blend_weight, sliding_positions
from networks.reliable_directional_repair import compose_surface_logits, dynamic_losses


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def distribution(value):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = a[np.isfinite(a)]
    return {'count': int(a.size), 'nonfinite': int(a.size-finite.size),
            **({'mean': float(finite.mean()), 'mean_abs': float(np.abs(finite).mean()),
                'min': float(finite.min()), 'max': float(finite.max()),
                'zero_fraction': float((finite == 0).mean()),
                **{f'p{p}': float(np.percentile(finite, p)) for p in (10, 50, 90, 99)}}
               if finite.size else {})}


def tensor_distribution(value):
    return None if value is None else distribution(value.detach().float().cpu().numpy())


def norm(values):
    live = [g.detach().double().square().sum() for g in values if g is not None]
    return float(torch.stack(live).sum().sqrt()) if live else 0.


def read_history(run, output, model):
    path = run/'stage_D/epoch_metrics.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.is_file() else []
    grad_rows, summary = [], {}
    for name in v9.STAGE_MODULES['D']:
        values = []
        for row in rows:
            value = row.get('gradient_norms', {}).get(name)
            grad_rows.append({'epoch': row['epoch'], 'module': name, 'total_gradient_l2': value,
                'continuity_gradient_first_batch_l2': row.get('continuity_gradient_norms_first_batch', {}).get(name)})
            if value is not None:
                values.append(value)
        summary[name] = {'recorded_epochs': len(values),
            'finite_positive_epochs': sum(int(np.isfinite(x) and x > 0) for x in values),
            'gradient_distribution': distribution(values)}
    write_csv(output/'historical_gradients.csv', grad_rows)
    write_csv(output/'training_curve.csv', [{k: val for k, val in row.items()
        if not isinstance(val, (dict, list))} for row in rows])
    val_rows = []
    for path in sorted((run/'stage_D').glob('val_epoch_*.json')):
        data = json.loads(path.read_text())
        val_rows.append({'epoch': int(path.stem.rsplit('_', 1)[1]),
            'threshold': data['threshold'], 'selection_score': data['selection_score'],
            **{'p0_'+k: v for k, v in data['p0'].items()},
            **{'p1_'+k: v for k, v in data['p1'].items()}})
    write_csv(output/'validation_curve.csv', val_rows)
    saved_optimizer = {}
    last = run/'stage_D/last.pth'
    if last.is_file():
        state = torch.load(last, map_location='cpu', weights_only=False)
        for group in state.get('optimizer', {}).get('param_groups', []):
            name = group.get('name', 'unnamed')
            saved_optimizer[name] = {'lr': group['lr'], 'parameter_tensors': len(group['params']),
                'expected_parameter_tensors': len(list(v9.module(model, name).parameters()))
                if name in v9.STAGE_MODULES['D'] else None}
    result = {'epochs': len(rows), 'module_gradients': summary, 'saved_optimizer_groups': saved_optimizer,
        'limitation': 'Historical norms are epoch averages, not proof that every parameter had useful gradients on every batch.'}
    write_json(output/'training_history.json', result)
    return result


def gradient_batch(model, image, mask, skeleton, source):
    """No optimizer or parameter writes; eval-mode gradients on a real batch."""
    model.eval()
    result = model(image, capture_diagnostics=True)
    total, terms = dynamic_losses(result, mask, skeleton,
        **{name: source[name] for name in ('lambda_bg', 'lambda_missing', 'lambda_cont', 'missing_threshold')},
        return_live_terms=True)
    if not torch.isfinite(total):
        raise RuntimeError('Nonfinite diagnostic loss')
    named = [(f'{group}.{name}', p) for group in v9.STAGE_MODULES['D']
             for name, p in v9.module(model, group).named_parameters()]
    params = [p for _, p in named]
    gradient_sets = {}
    for label, loss in (('total', total), ('seg', total-source['lambda_cont']*terms['cont_live']),
                        ('cont_unweighted', terms['cont_live'])):
        gradient_sets[label] = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    layers, modules = [], {}
    for i, (name, p) in enumerate(named):
        layers.append({'parameter': name, 'numel': p.numel(),
            **{label+'_l2': norm([grads[i]]) for label, grads in gradient_sets.items()},
            'total_connected': gradient_sets['total'][i] is not None,
            'total_finite': (bool(torch.isfinite(gradient_sets['total'][i]).all())
                             if gradient_sets['total'][i] is not None else None)})
    for group in v9.STAGE_MODULES['D']:
        indices = [i for i, (name, _) in enumerate(named) if name.startswith(group+'.')]
        modules[group] = {'trainable_parameters': sum(params[i].numel() for i in indices),
            **{label+'_l2': norm([grads[i] for i in indices]) for label, grads in gradient_sets.items()},
            'connected_tensors': sum(gradient_sets['total'][i] is not None for i in indices),
            'parameter_tensors': len(indices)}
    chain = {}
    for step in (1, 2, 3):
        u = result[f'u{step}']
        chain[f'u{step}'] = {'requires_grad': u.requires_grad, 'grad_fn': type(u.grad_fn).__name__,
            'loss_gradient': tensor_distribution(torch.autograd.grad(total, u, retain_graph=True,
                allow_unused=True)[0]) if u.requires_grad else None}
        if step < 3:
            next_u = result[f'u{step+1}']
            chain[f'u{step}_to_u{step+1}'] = tensor_distribution(torch.autograd.grad(
                next_u.mean(), u, retain_graph=True, allow_unused=True)[0]) if u.requires_grad and next_u.requires_grad else None
    u = result['u3']
    composition = {'u3_clamp_outside_fraction': float(((u < 1e-5) | (u > 1-1e-5)).float().mean()),
        'z1_wrt_u3': tensor_distribution(torch.autograd.grad(result['z1'].mean(), u,
            retain_graph=True, allow_unused=True)[0]),
        'p1_wrt_u3': tensor_distribution(torch.autograd.grad(result['p1'].mean(), u,
            retain_graph=True, allow_unused=True)[0])}
    with torch.no_grad():
        z_identity, _ = compose_surface_logits(result['z0'], result['u0'], result['u0'])
        z_changed, _ = compose_surface_logits(result['z0'], result['u0'],
            (result['u0']+.001).clamp(1e-5, 1-1e-5))
        composition.update(identity_logit_max_error=float((z_identity-result['z0']).abs().max()),
            positive_work_perturbation_surface_mean_abs=float((z_changed-result['z0']).abs().mean()))
    raw = result['proposal_raw']
    raw_grad = torch.autograd.grad(total, raw, retain_graph=True, allow_unused=True)[0]
    plus_s = raw[:, :1].sigmoid().detach(); minus_s = raw[:, 1:2].sigmoid().detach()
    proposal = {'parameterization': 'DeltaZ_plus=8*sigmoid(raw_plus); DeltaZ_minus=4*sigmoid(raw_minus); no hard clamp at 8/4',
        'raw_plus': tensor_distribution(raw[:, :1]), 'raw_minus': tensor_distribution(raw[:, 1:2]),
        'delta_plus': tensor_distribution(result['delta_z_plus']),
        'delta_minus': tensor_distribution(result['delta_z_minus']),
        'plus_ge_7_9_fraction': float((result['delta_z_plus'] >= 7.9).float().mean()),
        'plus_ge_7_99_fraction': float((result['delta_z_plus'] >= 7.99).float().mean()),
        'plus_exact_8_fraction': float((result['delta_z_plus'] == 8).float().mean()),
        'minus_ge_3_99_fraction': float((result['delta_z_minus'] >= 3.99).float().mean()),
        'minus_exact_4_fraction': float((result['delta_z_minus'] == 4).float().mean()),
        'plus_local_derivative': tensor_distribution(8*plus_s*(1-plus_s)),
        'minus_local_derivative': tensor_distribution(4*minus_s*(1-minus_s)),
        'loss_wrt_raw_plus': tensor_distribution(raw_grad[:, :1]) if raw_grad is not None else None,
        'loss_wrt_raw_minus': tensor_distribution(raw_grad[:, 1:2]) if raw_grad is not None else None}
    report = {'loss': float(total.detach()), 'L_seg': float(terms['seg']), 'L_cont': float(terms['cont']),
        'L_cont_paths': terms['paths'], 'missing_segments': terms['missing_segments'],
        'modules': modules, 'u_gradient_chain': chain, 'compose_surface_logits': composition,
        'proposal_saturation': proposal,
        'frozen_parameter_requires_grad_count': sum(p.requires_grad for p in model.frozen.parameters()),
        'frozen_parameter_grad_count': sum(p.grad is not None for p in model.frozen.parameters())}
    model.zero_grad(set_to_none=True)
    return report, layers


def gradient_audit(args, model, source):
    dataset = v9.RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
        source_patch_size=1500, random_crop_train=True, random_crops_per_image=1,
        random_crop_seed=args.seed, augment=False)
    count = args.gradient_batches*args.batch_size
    indices = np.linspace(0, len(dataset)-1, count).astype(int).tolist()
    reports, layer_rows = [], []
    for i in range(args.gradient_batches):
        selected = indices[i*args.batch_size:(i+1)*args.batch_size]
        samples = [dataset[j] for j in selected]
        tensors = [torch.stack([s[key] for s in samples]).to(args.device)
                   for key in ('image', 'mask', 'skeleton')]
        report, layers = gradient_batch(model, *tensors, source)
        report['training_indices'] = selected; reports.append(report)
        layer_rows.extend({'batch': i, **row} for row in layers)
        print(f'[GRADIENT] {i+1}/{args.gradient_batches} loss={report["loss"]:.6f} paths={report["L_cont_paths"]}', flush=True)
    write_json(args.output_dir/'gradients.json', {'mode': 'eval, read-only, no optimizer.step', 'batches': reports})
    write_csv(args.output_dir/'layer_gradients.csv', layer_rows)
    return reports


class Stats:
    def __init__(self):
        self.values = {}

    def add(self, region, key, array):
        a = np.asarray(array, dtype=np.float64)
        if not a.size:
            return
        if not np.isfinite(a).all():
            raise RuntimeError(f'Nonfinite output: {region}/{key}')
        low, high = (-1., 1.)
        if key == 'delta_plus': low, high = 0., 8.
        elif key == 'delta_minus': low, high = 0., 4.
        elif key == 'w_entropy': low, high = 0., math.log(8.)
        elif key.startswith('raw_'): low, high = -40., 40.
        s = self.values.setdefault((region, key), dict(n=0, total=0., absolute=0., positive=0,
            negative=0, hist=np.zeros(1024, dtype=np.int64), low=low, high=high, outside=0))
        s['n'] += a.size; s['total'] += float(a.sum()); s['absolute'] += float(np.abs(a).sum())
        s['positive'] += int((a > 1e-6).sum()); s['negative'] += int((a < -1e-6).sum())
        s['outside'] += int(((a < low) | (a > high)).sum())
        s['hist'] += np.histogram(np.clip(a, low, high), bins=1024, range=(low, high))[0]

    def rows(self):
        return [{'region': region, 'quantity': key, 'pixels': s['n'], 'mean': s['total']/s['n'],
            'mean_abs': s['absolute']/s['n'], 'positive_fraction_eps1e6': s['positive']/s['n'],
            'negative_fraction_eps1e6': s['negative']/s['n'], 'histogram_outside_fraction': s['outside']/s['n'],
            **{f'p{p}_hist_approx': s['low']+(min(np.searchsorted(s['hist'].cumsum(), p/100*s['n']), 1023)+.5)*
               (s['high']-s['low'])/1024 for p in (10, 50, 90, 99)}}
            for (region, key), s in self.values.items()]


def transport_maps(result, rates):
    """Exact update terms with checks against the existing forward implementation."""
    def a(t): return t.detach().float().cpu().numpy()[0, 0]
    w = result['w'].clamp_min(1e-8)
    entropy = -(w*w.log()).sum(1, keepdim=True)
    maps = {'r': a(result['r']), 'a': a(result['a']), 'b_minus_p0': a(result['b']-result['u0']),
        'delta_plus': a(result['delta_z_plus']), 'delta_minus': a(result['delta_z_minus']),
        'plus_ge_7_9': a((result['delta_z_plus'] >= 7.9).float()),
        'plus_ge_7_99': a((result['delta_z_plus'] >= 7.99).float()),
        'minus_le_0_01': a((result['delta_z_minus'] <= .01).float()),
        'raw_plus': a(result['proposal_raw'][:, :1]), 'raw_minus': a(result['proposal_raw'][:, 1:2]),
        'w_entropy': a(entropy), 'w_max': a(w.max(1, keepdim=True).values),
        'w_near_uniform': a((entropy/math.log(8.) > .99).float()), 'r_ge_0_99': a((result['r'] > .99).float())}
    for k in range(3):
        u = result[f'u{k}']; candidate = result['m'][:, k]-u
        transport = rates[k][:, 1:2]*result['g_step'][:, k]*candidate
        proposal = .2*result['a']*rates[k][:, 2:3]*(result['b']-u)
        anchor = rates[k][:, 3:4]*(result['u0']-u)
        torch.testing.assert_close(transport, result['transport_delta'][:, k], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(proposal, result['proposal_delta'][:, k], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(anchor, result['anchor_delta'][:, k], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(u+transport+proposal+anchor, result[f'u{k+1}'], rtol=1e-5, atol=2e-7)
        values = {'source_mass': result['source_mass'][:, k], 'V': result['source_available'][:, k],
            'candidate': candidate, 'transport': transport, 'proposal': proposal, 'anchor': anchor,
            'G': result['g_step'][:, k], 'alpha': result['alpha'][:, k],
            'beta': result['beta'][:, k], 'gamma': result['gamma'][:, k], 'stay': result['stay'][:, k],
            'U_change_from_P0': result[f'u{k+1}']-result['u0'],
            'high_G_negligible_transport': ((result['g_step'][:, k] > .9) & (transport.abs() < 1e-6)).float(),
            'G_ge_0_99': (result['g_step'][:, k] > .99).float(),
            'G_gt_0_9': (result['g_step'][:, k] > .9).float(),
            'V_lt_0_01': (result['source_available'][:, k] < .01).float(),
            'rate_max': rates[k].max(1, keepdim=True).values,
            'rate_concentrated': (rates[k].max(1, keepdim=True).values > .95).float()}
        values.update({f'rate_{j}': rates[k][:, j:j+1] for j in range(4)})
        for key, value in values.items():
            maps[f'{key}_{k+1}'] = a(value)
    return maps


@torch.no_grad()
def collect_validation(args, model):
    from run_q3_r3_mass import full_dataset
    dataset = full_dataset(args.root_path, 'val')
    if len(dataset) != 14:
        raise ValueError(f'Expected the same 14 val images, found {len(dataset)}')
    stats, caches = Stats(), []
    for index in range(len(dataset)):
        sample = dataset[index]; image = sample['image'][None].to(args.device)
        gt = sample['mask'][0].numpy() > .5
        height, width = gt.shape
        canvases = [torch.zeros((1, 1, height, width), device=args.device) for _ in range(4)]
        weight = blend_weight(512, args.device)
        records, rates = [], []
        handle = model.dynamics.rates.register_forward_hook(lambda m, i, out: rates.append(out.softmax(1)))
        try:
            for y in sliding_positions(height, 512, 256):
                for x in sliding_positions(width, 512, 256):
                    rates.clear()
                    result = model(image[:, :, y:y+512, x:x+512], capture_diagnostics=True)
                    q = F.interpolate(result['q'], size=(512, 512), mode='bilinear', align_corners=False)
                    for canvas, value in zip(canvases[:3], (result['z0'], result['z1'], q)):
                        canvas[:, :, y:y+512, x:x+512] += value.float()*weight
                    canvases[3][:, :, y:y+512, x:x+512] += weight
                    if len(rates) != 3:
                        raise RuntimeError('Expected exactly three rates calls')
                    records.append((y, x, transport_maps(result, rates)))
        finally:
            handle.remove()
        z0, z1, q = [(c/canvases[3]).cpu().numpy()[0, 0] for c in canvases[:3]]
        prob0 = 1/(1+np.exp(-np.clip(z0, -80, 80))); prob1 = 1/(1+np.exp(-np.clip(z1, -80, 80)))
        p0, p1 = prob0 >= args.reference_threshold, prob1 >= args.threshold
        isolated, bridge = v9.background_components(p0, p1, gt)
        from evaluation.road_comparison.road_metrics import skeletonize
        missing = skeletonize(gt) & ~(cv2.dilate(p0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        regions = {'all': np.ones_like(gt), 'recovered_gt': gt & ~p0 & p1,
            'unrecovered_gt': gt & ~p0 & ~p1, 'hard_background': ~gt & (q >= .5),
            'isolated_new_fp': isolated, 'false_bridge': bridge, 'missing_center': missing,
            'reliable_road': gt & p0}
        owned = np.zeros_like(gt)
        for y, x, maps in records:
            take = ~owned[y:y+512, x:x+512]; owned[y:y+512, x:x+512] = True
            masks = {name: mask[y:y+512, x:x+512] & take for name, mask in regions.items()}
            for key, value in maps.items():
                full = cv2.resize(value, (512, 512), interpolation=cv2.INTER_LINEAR)
                for name, selected in masks.items():
                    stats.add(name, key, full[selected])
        caches.append({'case': sample['case_name'], 'p0': prob0, 'p1': prob1, 'gt': gt})
        del records, result, image, canvases
        print(f'[VAL TRANSPORT] {index+1}/{len(dataset)} {sample["case_name"]}', flush=True)
    rows = stats.rows()
    write_csv(args.output_dir/'transport_regions.csv', rows)
    write_json(args.output_dir/'transport_regions.json', {'images': len(dataset),
        'reference_threshold': args.reference_threshold, 'p1_threshold': args.threshold, 'rows': rows,
        'method': 'All 512/256 windows; bilinear work-map projection; deterministic first-window ownership counts each full-image pixel once. Local tile associations, not an exact causal decomposition of stitched P1.',
        'false_bridge_note': 'Region proxy: new background component near two old components; metric proxy uses the unchanged gap_lengths routine and is a distinct quantity.',
        'G_high_epsilon': 'G>0.9 and abs(transport)<1e-6; masks use fixed P0 and selected P1.'})
    return caches, rows


def matched_threshold(caches, field, target, measure, bins=10000):
    hist = np.zeros(bins+1, dtype=np.int64)
    for item in caches:
        p = item[field][item['gt']] if measure == 'tp' else item[field].reshape(-1)
        hist += np.bincount(np.minimum((p*bins).astype(np.int64), bins), minlength=bins+1)
    counts = hist[::-1].cumsum()[::-1]
    errors = np.abs(counts-target)
    i = int(np.flatnonzero(errors == errors.min())[-1])
    # Verify nearby grid points against original floats (bin-boundary rounding).
    candidates = []
    for index in range(max(0, i-2), min(bins, i+2)+1):
        threshold = index/bins
        actual = sum(int(((item[field] >= threshold) & item['gt']).sum()) if measure == 'tp'
                     else int((item[field] >= threshold).sum()) for item in caches)
        candidates.append((abs(actual-target), -threshold, actual-target))
    _, minus_threshold, residual = min(candidates)
    return -minus_threshold, residual


def compare_outputs(args, caches, selection):
    from evaluation.road_comparison.road_metrics import prepare_target, image_metrics, summarize
    from run_reliable_directional_mass import gap_lengths, selection_score
    from run_q3_r3_mass import PROTOCOL
    curves = {float(t): value for t, value in selection['threshold_curves'].items()}
    p0_t = max(curves, key=lambda t: selection_score(curves[t]['p0'], {'background_bridge_proxy': 0}, len(caches)))
    p1_t = float(selection['threshold'])
    tp = sum(int(((c['p0'] >= p0_t) & c['gt']).sum()) for c in caches)
    area = sum(int((c['p0'] >= p0_t).sum()) for c in caches)
    match_recall, _ = matched_threshold(caches, 'p1', tp, 'tp', args.matching_bins)
    match_area, _ = matched_threshold(caches, 'p1', area, 'area', args.matching_bins)
    points = {'own_val_thresholds': (p0_t, p1_t), 'same_threshold': (p1_t, p1_t),
        'matched_recall': (p0_t, match_recall), 'matched_foreground_area': (p0_t, match_area)}
    stores = {name: {field: [] for field in ('p0', 'p1', 'paired', 'length')} for name in points}
    per_image = []
    for index, cache in enumerate(caches):
        gt = cache['gt']; prepared = prepare_target(gt, PROTOCOL)
        reference = cache['p0'] >= args.reference_threshold
        metric_cache, gap_cache = {}, {}
        for name, (t0, t1) in points.items():
            storage = stores[name]
            for field, threshold in (('p0', t0), ('p1', t1)):
                key = (field, threshold)
                if key not in metric_cache:
                    metric_cache[key] = image_metrics(cache[field] >= threshold, prepared, PROTOCOL)
                storage[field].append({'case_id': cache['case'], **metric_cache[key]})
            if t1 not in gap_cache:
                gap_cache[t1] = gap_lengths(reference, cache['p1'] >= t1, prepared['skeleton'], gt)
            by_length, bridges = gap_cache[t1]
            storage['length'].append(by_length)
            pred = cache['p1'] >= t1
            paired = {'eligible_p0_gaps': sum(v['eligible'] for v in by_length.values()),
                'closed_p0_gaps': sum(v['closed'] for v in by_length.values()), 'background_bridge_proxy': bridges,
                'added_gt_pixels': int((pred & ~reference & gt).sum()),
                'added_background_pixels': int((pred & ~reference & ~gt).sum()),
                'removed_gt_pixels': int((reference & ~pred & gt).sum()),
                'removed_background_pixels': int((reference & ~pred & ~gt).sum())}
            storage['paired'].append(paired)
            per_image.append({'case': cache['case'], 'point': name, 'p0_threshold': t0,
                'p1_threshold': t1, **paired})
        print(f'[COMPARE TOPOLOGY] {index+1}/{len(caches)}', flush=True)
    reports, flat = {}, []
    for name, (t0, t1) in points.items():
        data = stores[name]; p0, p1 = summarize(data['p0']), summarize(data['p1'])
        paired = {key: sum(row[key] for row in data['paired']) for key in data['paired'][0]}
        paired['gap_closure_rate_proxy'] = paired['closed_p0_gaps']/max(paired['eligible_p0_gaps'], 1)
        by_length = {length: {key: sum(row[length][key] for row in data['length']) for key in ('eligible', 'closed')}
                     for length in data['length'][0]}
        for value in by_length.values():
            value['closure_rate_proxy'] = value['closed']/max(value['eligible'], 1)
        reports[name] = {'p0_threshold': t0, 'p1_threshold': t1, 'p0': p0, 'p1': p1, 'paired': paired,
            'gap_by_length': by_length, 'recall_difference': p1['recall']-p0['recall'],
            'foreground_area_difference': p1['tp']+p1['fp']-p0['tp']-p0['fp']}
        for field, metrics in (('p0', p0), ('p1', p1)):
            flat.append({'point': name, 'output': field, 'threshold': t0 if field == 'p0' else t1,
                **metrics, **paired, 'recall_difference': reports[name]['recall_difference'],
                'foreground_area_difference': reports[name]['foreground_area_difference']})
    # Selection curves must come from precisely this checkpoint and reproduce counts.
    selected = curves[p1_t]
    for field in ('p0', 'p1'):
        actual = reports['same_threshold'][field]
        if any(actual[k] != selected[field][k] for k in ('tp', 'fp', 'fn')):
            raise ValueError(f'Cached threshold sweep does not reproduce current {field} pixel counts; rerun val_direct')
    write_json(args.output_dir/'operating_points.json', {'selected_on': 'val', 'reference_threshold': args.reference_threshold,
        'selection_formula': 'IoU + .15*clDice + .10*APLS_GT_to_Pred + .05*APLS_bidirectional - .001*short_components - .01*background_bridge_proxy/images; P0 selection uses zero bridge penalty.',
        'matching_grid': 1/args.matching_bins, 'matching_anchor': 'P0 at its independently selected val threshold',
        'comparisons': reports, 'note': 'Closure and bridges are unchanged diagnostic proxies; APLS is the raster approximation, not official SpaceNet APLS.'})
    write_csv(args.output_dir/'operating_points.csv', flat)
    write_csv(args.output_dir/'paired_per_image.csv', per_image)
    write_csv(args.output_dir/'threshold_curve.csv', [{'threshold': t,
        'p0_score': selection_score(v['p0'], {'background_bridge_proxy': 0}, len(caches)),
        'p1_score': v['selection_score'], **{'p0_'+k: value for k, value in v['p0'].items()},
        **{'p1_'+k: value for k, value in v['p1'].items()}} for t, v in sorted(curves.items())])
    return reports


def markdown_report(args, history, gradients, rows, comparisons):
    lookup = {(r['region'], r['quantity']): r for r in rows}
    lines = ['# V9 validation diagnosis', '',
        'No weights, formulas, optimizer state or losses were changed. All model selection uses val.', '',
        '## Gradient evidence', '', '| Module | Positive-gradient epochs | Current total / continuity L2 (first audit batch) |',
        '|---|---:|---:|']
    for name, data in history['module_gradients'].items():
        current = gradients[0]['modules'][name]
        lines.append(f'| {name} | {data["finite_positive_epochs"]}/{data["recorded_epochs"]} | {current["total_l2"]:.4g} / {current["cont_unweighted_l2"]:.4g} |')
    lines += ['', 'Current layer gradients, U1→U2→U3 Jacobian probes, compose/logit/clamp gradients and raw residual saturation are in gradients.json and layer_gradients.csv. These are current eval-mode real-training-batch gradients, not a reconstruction of historical per-layer gradients.', '',
        '## Update attribution', '', '| Region | Step | V | M-U | G | transport signed / abs | proposal signed / abs | high G, negligible transport fraction |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for region in ('recovered_gt', 'unrecovered_gt', 'hard_background', 'isolated_new_fp', 'false_bridge', 'missing_center'):
        for step in (1, 2, 3):
            def s(key): return lookup.get((region, f'{key}_{step}'), {})
            if not s('V'):
                continue
            lines.append(f'| {region} | {step} | {s("V")["mean"]:.5g} | {s("candidate")["mean"]:.5g} | {s("G")["mean"]:.5g} | {s("transport")["mean"]:.5g} / {s("transport")["mean_abs"]:.5g} | {s("proposal")["mean"]:.5g} / {s("proposal")["mean_abs"]:.5g} | {s("high_G_negligible_transport")["mean"]:.4g} |')
    lines += ['', '## Source and update bottleneck indicators', '']
    for region in ('missing_center', 'unrecovered_gt', 'isolated_new_fp', 'false_bridge'):
        for step in (1, 2, 3):
            def s(key): return lookup.get((region, f'{key}_{step}'), {})
            if not s('V'):
                continue
            total_abs = sum(s(key)['mean_abs'] for key in ('transport', 'proposal', 'anchor'))
            transport_share = s('transport')['mean_abs']/max(total_abs, 1e-30)
            proposal_share = s('proposal')['mean_abs']/max(total_abs, 1e-30)
            high_g = s('G_gt_0_9')['mean']
            conditional = s('high_G_negligible_transport')['mean']/high_g if high_g > 0 else None
            lines.append(f'- {region}, step {step}: V<.01 fraction={s("V_lt_0_01")["mean"]:.4g}; '
                f'candidate positive/negative fraction={s("candidate")["positive_fraction_eps1e6"]:.4g}/'
                f'{s("candidate")["negative_fraction_eps1e6"]:.4g}; transport/proposal share of absolute updates='
                f'{transport_share:.4g}/{proposal_share:.4g}; negligible transport conditional on G>.9={conditional}.')
    lines += ['', 'V=clamp(source_mass,0,1); alpha=rate[1]*G; proposal coefficient=.2*A*rate[2]; anchor=rate[3]. No extra multiplication by V is introduced in this audit.', '',
        'Low V and non-positive M-U can prevent useful transport even when G is high. The table separates source/candidate limitations from gate strength. Transport/proposal magnitudes describe local updates and do not by themselves prove a causal error source after nonlinear stitching.', '',
        '## Validation operating points', '', '| Point | Threshold P0/P1 | IoU P0/P1 | clDice P0/P1 | bidirectional APLS P0/P1 | Recall residual | Area residual | Closed / eligible | Background bridge proxy |', '|---|---|---|---|---|---:|---:|---|---:|']
    for name, data in comparisons.items():
        a, b, paired = data['p0'], data['p1'], data['paired']
        lines.append(f'| {name} | {data["p0_threshold"]:.4f}/{data["p1_threshold"]:.4f} | {a["iou"]:.5f}/{b["iou"]:.5f} | {a["cldice"]:.5f}/{b["cldice"]:.5f} | {a["apls_bidirectional_approx"]:.5f}/{b["apls_bidirectional_approx"]:.5f} | {data["recall_difference"]:+.6f} | {data["foreground_area_difference"]:+d} | {paired["closed_p0_gaps"]}/{paired["eligible_p0_gaps"]} | {paired["background_bridge_proxy"]} |')
    lines += ['', 'Full Precision/Recall/F1, both directional APLS, short components, TP/FP additions and length bins are in operating_points.json/CSV. G mean alone is not a success criterion. Fixed P0 reference is used for every gap calculation. Background bridge metrics and false_bridge region masks have distinct documented proxy definitions.', '',
        'Full validation regional maps cover every window and each full-image pixel once. They are conditional associations with binary outcomes, not an intervention experiment. No assertion of training success or a unique root cause is made from gradient nonzero status alone.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--root_path', type=Path, default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--gradient_batches', type=int, default=2); p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--matching_bins', type=int, default=10000)
    args = p.parse_args()
    if args.gradient_batches < 1 or args.batch_size < 1 or args.matching_bins < 1000:
        p.error('Invalid batch or matching configuration')
    checkpoint = args.run_dir/'stage_D/best.pth'
    selection_path = args.run_dir/'direct_val_selection.json'
    if not checkpoint.is_file() or not selection_path.is_file():
        p.error('Need stage_D/best.pth and direct_val_selection.json; run val_direct first')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    selection = json.loads(selection_path.read_text())
    if payload['identity'].get('initialization') != 'direct_random_downstream':
        p.error('This command audits the direct-initialization V9 run')
    if Path(payload['training_config']['root_path']).resolve() != args.root_path.resolve():
        p.error('The data root differs from the trained checkpoint protocol')
    if selection.get('selected_on') != 'val' or selection['report'].get('images') != 14:
        p.error('Need a complete 14-image val selection')
    if selection['identity'] != payload['identity'] or Path(selection['checkpoint']).resolve() != checkpoint.resolve():
        p.error('Threshold selection belongs to another checkpoint')
    args.reference_threshold = float(payload['training_config']['reference_threshold'])
    if float(selection['report']['p0_reference_threshold']) != args.reference_threshold:
        p.error('P0 reference threshold differs between training and selection')
    args.threshold = float(selection['threshold']); args.device = torch.device(args.device)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Use a new, empty output_dir to preserve prior diagnostics')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    load_args = SimpleNamespace(initialization='direct', q3_checkpoint=Path(payload['identity']['q3']['path']), device=args.device)
    model, source, identity = v9.load_initial(load_args)
    v9.restore_checkpoint(model, checkpoint, identity, 'D')
    model.requires_grad_(False)
    for name in v9.STAGE_MODULES['D']:
        v9.module(model, name).requires_grad_(True)
    model.eval()
    write_json(args.output_dir/'protocol.json', {'checkpoint': str(checkpoint), 'best_epoch': payload['epoch'],
        'best_score': payload['score'], 'p0_reference_threshold': args.reference_threshold,
        'p1_val_threshold': args.threshold, 'identity': identity, 'checkpoint_sha256': v9.digest(checkpoint),
        'loss_formula': 'L_seg + 0.2*L_cont', 'selection_path': str(selection_path),
        'no_optimizer_step': True, 'no_test_data': True})
    history = read_history(args.run_dir, args.output_dir, model)
    gradients = gradient_audit(args, model, source)
    caches, rows = collect_validation(args, model)
    comparisons = compare_outputs(args, caches, selection)
    markdown_report(args, history, gradients, rows, comparisons)
    if v9.digest(checkpoint) != json.loads((args.output_dir/'protocol.json').read_text())['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed on disk during audit')
    print(f'[DONE] {args.output_dir / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
