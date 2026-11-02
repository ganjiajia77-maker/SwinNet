"""Fixed-checkpoint, three-step source/proposal/anchor counterfactual audit.

The proposal B and its positive_support are produced by the unmodified model.
All interventions operate on the actual selector update, with a fresh state,
source support, selector and rates at each of the three steps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from diagnostics.audit_full_h0_saved_predictions import component_data, digest, gap_state, read_json, scores, write_csv
from diagnostics.diagnose_full_h0_optimistic_reachability import (
    atomic_json, atomic_npz, grid_gap, optimistic_sequence, rebuild_model, work_gap_state,
    signature, tile_gap, validate_saved_prediction, verify_recorded_code,
)
from mass_protocol import blend_weight, sliding_positions
from networks.reliable_directional_repair import DIRECTIONS, compose_surface_logits, shifted
from networks.v9_source_selector import source_support
from run_q3_r3_mass import full_dataset
from run_slim_end_to_end_mass import seed_all
from selector_transport_reference import bridge_proxy, exact_recall_threshold

MODES = ('baseline', 'redistribute', 'no_stay', 'transport_010',
         'proposal_no_a', 'proposal_no_point2', 'no_anchor')


def change_choice(choice, source_states, valid, mode):
    if mode not in ('redistribute', 'no_stay'):
        return choice
    # The current state's eight actual neighbours, never padded image values.
    best = source_states.masked_fill(~valid, -torch.inf).argmax(1, keepdim=True)
    has_source = valid.any(1, keepdim=True)
    mass = choice[:, 1:].sum(1, keepdim=True) if mode == 'redistribute' else has_source.to(choice.dtype)
    sources = torch.zeros_like(choice[:, 1:]).scatter(1, best, mass)
    return torch.cat((1. - mass, sources), 1)


def replay(model, inputs, mode):
    """Re-evaluate the original 3-step equations, modifying exactly one term."""
    visual, p0, q, reliable, proposal, accept, weights = inputs[:7]
    state = p0
    states = [state]
    traces = []
    for _ in range(3):
        support = source_support(state, p0, reliable)
        mass = torch.zeros_like(state)
        for d, (dy, dx) in enumerate(DIRECTIONS):
            mass = mass + weights[:, d:d+1] * shifted(support, dy, dx)
        available = mass.clamp(0., 1.)
        rates = model.original.dynamics.rates(torch.cat((visual, p0, q, reliable, proposal,
                                                          accept, state, available), 1)).softmax(1)
        scored = model.selector(model._context, state, support, available, rates[:, 1:2])
        choice = change_choice(scored['choice'], scored['source_states'], scored['valid'], mode)
        leave = choice[:, 1:].sum(1, keepdim=True)
        beta = .2 * accept * rates[:, 2:3]
        gamma = rates[:, 3:4]
        if mode == 'proposal_no_a': beta = .2 * rates[:, 2:3]
        if mode == 'proposal_no_point2': beta = accept * rates[:, 2:3]
        if mode == 'no_anchor': gamma = torch.zeros_like(gamma)
        rate = rates[:, 1:2]
        if mode == 'transport_010':
            # Transfer unused stay mass to transport; never take proposal/anchor budget.
            limit = (1. - beta - gamma).clamp_min(0.) / leave.clamp_min(1e-12)
            rate = torch.minimum(torch.maximum(rate, torch.full_like(rate, .1)), limit)
        transport = (rate * choice[:, 1:] * (scored['source_states'] - state)).sum(1, keepdim=True)
        proposed = beta * (proposal - state)
        anchored = gamma * (p0 - state)
        next_state = state + transport + proposed + anchored
        stay = 1. - rate * leave - beta - gamma
        if bool((stay < -1e-6).any()) or bool(((next_state < -1e-6) | (next_state > 1.+1e-6)).any()):
            raise RuntimeError('Invalid convex update in ' + mode)
        neighbour_max = scored['source_states'].masked_fill(~scored['valid'], -torch.inf).amax(1, keepdim=True)
        neighbour_max = torch.where(scored['valid'].any(1, keepdim=True), neighbour_max, state)
        weighted = (choice[:, 1:] * scored['source_states']).sum(1, keepdim=True)
        selected = torch.where(leave > 1e-8, weighted / leave.clamp_min(1e-8), state)
        useful = (choice[:, 1:] * (scored['source_states'] > state).to(choice.dtype)).sum(1, keepdim=True)
        traces.append({'U': state, 'P0': p0, 'B': proposal, 'A': accept,
                       'max_neighbour': neighbour_max, 'selected_mean': selected,
                       'stay_weight': choice[:, :1], 'higher_source_weight': useful,
                       'transport_rate': rate, 'proposal_rate': rates[:, 2:3], 'anchor_rate': gamma,
                       'available': available, 'transport_potential': selected-state,
                       'proposal_potential': proposal-state, 'proposal_coefficient': beta,
                       'transport_delta': transport, 'proposal_delta': proposed,
                       'anchor_delta': anchored, 'U_next': next_state})
        for d in range(8):
            traces[-1][f'source_{d}_probability'] = scored['source_states'][:, d:d+1]
            traces[-1][f'source_{d}_weight'] = choice[:, d+1:d+2]
            traces[-1][f'source_{d}_valid'] = scored['valid'][:, d:d+1].to(state.dtype)
        state = next_state
        states.append(state)
    return states, traces


def native_gap_regions(local, grid):
    mapped, reasons = grid_gap(local, 512, grid)
    if reasons: return None, reasons
    gap = mapped['pixels']
    # The centre is the half of missing cells farthest from their nearest anchor.
    anchor = np.concatenate(mapped['anchors'])
    mask = np.ones((grid, grid), np.uint8)
    mask.flat[anchor] = 0
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5).reshape(-1)[gap]
    centre = gap[distance >= np.quantile(distance, .5)]
    return {'gap': gap, 'centre_proxy': centre, 'anchors': np.unique(anchor)}, []


def make_trace_rows(case, entry, top, left, traces, threshold):
    rows = []
    candidates = []
    for gap in entry['gaps']:
        local = tile_gap(gap, tuple(entry['shape']), top, left, 512)
        if local is not None:
            candidates.append((gap, local))
    if not candidates:
        return rows
    # Transfer each step once; individual per-gap GPU-to-CPU copies would
    # otherwise synchronize the device thousands of times per window.
    arrays = {}
    for mode, steps in traces.items():
        arrays[mode] = []
        for fields in steps:
            keys = list(fields)
            batch = torch.cat([fields[key] for key in keys], 1)[0].detach().cpu().numpy()
            arrays[mode].append({key: batch[i].reshape(-1) for i, key in enumerate(keys)})
    for gap, local in candidates:
        grid = traces['baseline'][0]['U'].shape[-1]
        regions, reasons = native_gap_regions(local, grid)
        if reasons: continue
        for mode, steps in arrays.items():
            for step, fields in enumerate(steps, 1):
                for region, indices in regions.items():
                    row = {'case': case, 'gap_id': gap['id'], 'tile_top': top, 'tile_left': left,
                           'mode': mode, 'step': step, 'region': region, 'work_cells': len(indices),
                           'threshold': float(threshold)}
                    for key, value in fields.items():
                        values = value[indices]
                        row[key+'_mean'] = float(values.mean())
                        row[key+'_p10'], row[key+'_p50'], row[key+'_p90'] = map(float, np.quantile(values, [.1, .5, .9]))
                    before = fields['U'][indices]
                    after = fields['U_next'][indices]
                    row['fraction_below_threshold_before'] = float((before < threshold).mean())
                    row['fraction_below_threshold_after'] = float((after < threshold).mean())
                    bottleneck = int(indices[int(after.argmin())])
                    row['lowest_U_next_work_y'], row['lowest_U_next_work_x'] = divmod(bottleneck, grid)
                    rows.append(row)
    return rows


def make_tile_topology(case, entry, top, left, captured, result, threshold):
    candidates = []
    for gap in entry['gaps']:
        local = tile_gap(gap, tuple(entry['shape']), top, left, 512)
        if local is not None: candidates.append((gap, local))
    if not candidates: return []
    rows = []
    for mode, (states, _) in captured.items():
        for step in (1, 2, 3):
            work = states[step][0, 0].detach().cpu().numpy()
            full = compose_surface_logits(result['z0'], result['u0'], states[step])[0].sigmoid()[0, 0].detach().cpu().numpy()
            work_labels = component_data(work >= threshold)[1]
            full_mask = full >= threshold; full_labels = component_data(full_mask)[1]
            for gap, local in candidates:
                mapped, reasons = grid_gap(local, 512, work.shape[0])
                if reasons: continue
                w = work_gap_state(work, work_labels, mapped, threshold)
                f = gap_state(full_mask, full_labels, local, (512, 512))
                full_gap = full.reshape(-1)[np.asarray(local['pixels'],dtype=np.int64)]
                rows.append({'case':case, 'gap_id':gap['id'], 'tile_top':top, 'tile_left':left,
                             'mode':mode, 'step':step, 'threshold':float(threshold),
                             'work_coverage':w['coverage'], 'work_min_probability':w['min_probability'],
                             'work_local_closed':w['local_closed_proxy'],
                             'full_coverage':f['coverage'], 'full_min_probability':float(full_gap.min()),
                             'full_mean_probability':float(full_gap.mean()),
                             'full_lowest_probability_margin':float(full_gap.min()-threshold),
                             'full_local_closed':f['local_closed_proxy'],
                             'work_closed_full_not':bool(w['local_closed_proxy'] and not f['local_closed_proxy'])})
    return rows


@torch.inference_mode()
def infer_case(model, image, name, entry, threshold):
    h, w = image.shape[-2:]
    device = image.device
    weight = blend_weight(512, device)
    names = ('p0',) + MODES + ('upper3', 'baseline_step1', 'baseline_step2')
    canvas = torch.zeros((len(names), 1, h, w), dtype=torch.float32, device=device)
    divisor = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
    positions = [(y, x) for y in sliding_positions(h, 512, 256) for x in sliding_positions(w, 512, 256)]
    rows = []; topology = []
    original = model._source_dynamics
    for i, (top, left) in enumerate(positions):
        captured = {}
        def replacement(this, *args):
            actual = original(*args)
            for mode in MODES:
                captured[mode] = replay(this, args, mode)
            for k in range(4):
                if not torch.allclose(actual[0][k], captured['baseline'][0][k], atol=2e-6, rtol=2e-6):
                    raise RuntimeError('Replayed baseline differs from actual forward at step ' + str(k))
            return actual
        object.__setattr__(model, '_source_dynamics', types.MethodType(replacement, model))
        try:
            result = model(image[:, :, top:top+512, left:left+512])
        finally:
            object.__setattr__(model, '_source_dynamics', original)
        upper = optimistic_sequence(result['u0'], result['b'], 3)
        logits = [result['z0']]
        for mode in MODES:
            z, _ = compose_surface_logits(result['z0'], result['u0'], captured[mode][0][-1])
            logits.append(z)
        logits.append(compose_surface_logits(result['z0'], result['u0'], upper[-1])[0])
        for k in (1, 2):
            logits.append(compose_surface_logits(result['z0'], result['u0'], captured['baseline'][0][k])[0])
        canvas[:, :, top:top+512, left:left+512] += torch.cat(logits, 0).float() * weight
        divisor[:, :, top:top+512, left:left+512] += weight
        rows.extend(make_trace_rows(name, entry, top, left, {key: value[1] for key, value in captured.items()}, threshold))
        topology.extend(make_tile_topology(name, entry, top, left, captured, result, threshold))
        if i == 0 or (i+1) % 5 == 0 or i+1 == len(positions):
            print(f'[TILE] {name} {i+1}/{len(positions)}', flush=True)
    predictions = (canvas / divisor).sigmoid().cpu().numpy()[:, 0]
    maps = dict(zip(names, predictions))
    if not np.isfinite(predictions).all(): raise RuntimeError('Nonfinite predictions')
    if float(np.maximum(maps['baseline'] - maps['upper3'], 0).max()) > 1e-4:
        raise RuntimeError('Original output exceeds the max reachability bound')
    return maps, rows, topology


def analyze(cases, manifest, thresholds, output, traces, topology):
    summaries, gap_rows = {}, []
    step_gaps = {}
    for threshold in thresholds:
        for case in cases:
            entry = manifest['images'][case['case']]
            statuses = {}
            for key in ('p0','baseline_step1','baseline_step2'):
                mask = case[key] >= threshold; labels = component_data(mask)[1]
                statuses[key] = {gap['id']:gap_state(mask, labels, gap, mask.shape) for gap in entry['gaps']}
            step_gaps[(threshold,case['case'])] = statuses
    for target_threshold in thresholds:
        target_tp = sum(int(((c['baseline'] >= target_threshold) & c['gt']).sum()) for c in cases)
        for mode in MODES + ('upper3',):
            matched, mismatch = exact_recall_threshold(cases, mode, target_tp)
            for protocol, cutoff in (('fixed', target_threshold), ('matched_recall', matched)):
                totals = {'tp': 0, 'fp': 0, 'fn': 0, 'added_tp': 0, 'lost_tp': 0,
                          'added_fp': 0, 'removed_fp': 0, 'new_bridge_proxy_vs_baseline': 0,
                          'bridge_proxy_vs_current_p0': 0,
                          'new_local_closed': 0, 'lost_local_closed': 0, 'local_closed': 0}
                for case in cases:
                    gt = case['gt']; base = case['baseline'] >= target_threshold
                    pred = case[mode] >= cutoff
                    values = scores(pred, gt)
                    for key in ('tp', 'fp', 'fn'): totals[key] += values[key]
                    totals['added_tp'] += int((pred & ~base & gt).sum())
                    totals['lost_tp'] += int((base & ~pred & gt).sum())
                    totals['added_fp'] += int((pred & ~base & ~gt).sum())
                    totals['removed_fp'] += int((base & ~pred & ~gt).sum())
                    totals['new_bridge_proxy_vs_baseline'] += bridge_proxy(base, pred, gt)
                    totals['bridge_proxy_vs_current_p0'] += bridge_proxy(case['p0'] >= target_threshold, pred, gt)
                    base_labels = component_data(base)[1]; pred_labels = component_data(pred)[1]
                    upper = case['upper3'] >= target_threshold
                    upper_labels = component_data(upper)[1]
                    entry = manifest['images'][case['case']]
                    for gap in entry['gaps']:
                        b = gap_state(base, base_labels, gap, gt.shape)
                        p = gap_state(pred, pred_labels, gap, gt.shape)
                        u = gap_state(upper, upper_labels, gap, gt.shape)
                        bc, pc, uc = b['local_closed_proxy'], p['local_closed_proxy'], u['local_closed_proxy']
                        totals['local_closed'] += int(pc)
                        totals['new_local_closed'] += int(pc and not bc)
                        totals['lost_local_closed'] += int(bc and not pc)
                        gap_rows.append({'case': case['case'], 'gap_id': gap['id'], 'length_bin': gap.get('length_bin'),
                            'reference_threshold': target_threshold, 'mode': mode, 'protocol': protocol,
                            'threshold': cutoff, 'baseline_local_closed': bc, 'upper_local_closed': uc,
                            'comparison_group': ('both_closed' if bc and uc else 'upper_only' if uc else
                                                 'baseline_only' if bc else 'neither_closed'),
                            'mode_local_closed': pc, 'new_local_closed': bool(pc and not bc),
                            'lost_local_closed': bool(bc and not pc), 'coverage': p['coverage'],
                            'local_connected': p['local_connected'],
                            'p0_local_closed':step_gaps[(target_threshold,case['case'])]['p0'][gap['id']]['local_closed_proxy'],
                            'baseline_step1_local_closed':step_gaps[(target_threshold,case['case'])]['baseline_step1'][gap['id']]['local_closed_proxy'],
                            'baseline_step2_local_closed':step_gaps[(target_threshold,case['case'])]['baseline_step2'][gap['id']]['local_closed_proxy']})
                tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
                totals.update(iou=tp/max(tp+fp+fn, 1), recall=tp/max(tp+fn, 1),
                              precision=tp/max(tp+fp, 1), threshold=float(cutoff),
                              matched_tp_error=int(mismatch) if protocol == 'matched_recall' else None)
                summaries[f'{target_threshold}|{mode}|{protocol}'] = totals
    # One group per native gap, at the selected threshold; no arithmetic
    # subtraction of the number of closed gaps is used for set intersections.
    chosen = thresholds[-1]
    base_rows = [r for r in gap_rows if r['reference_threshold'] == chosen and
                 r['mode'] == 'baseline' and r['protocol'] == 'fixed']
    groups = {(r['case'], r['gap_id']):r['comparison_group'] for r in base_rows}
    group_counts = {g:sum(value == g for value in groups.values()) for g in
                    ('upper_only','both_closed','baseline_only','neither_closed')}
    trace_groups = {}
    for row in traces:
        group = groups.get((row['case'], row['gap_id']), 'unknown')
        key = (group, row['mode'], row['step'], row['region'])
        trace_groups.setdefault(key, []).append(row)
    trace_summary = []
    for (group, mode, step, region), rows in sorted(trace_groups.items()):
        result = {'group':group, 'mode':mode, 'step':step, 'region':region,
                  'gap_tiles':len(rows), 'unique_gaps':len({(r['case'],r['gap_id']) for r in rows})}
        for field in ('U_mean','U_next_mean','B_mean','P0_mean','max_neighbour_mean',
                      'selected_mean_mean','stay_weight_mean','higher_source_weight_mean',
                      'transport_rate_mean','transport_potential_mean','transport_delta_mean',
                      'proposal_potential_mean','proposal_coefficient_mean','proposal_delta_mean',
                      'anchor_delta_mean','fraction_below_threshold_after'):
            values = np.asarray([r[field] for r in rows])
            result[field+'_mean_over_gap_tiles'] = float(values.mean())
            result[field+'_p10_over_gap_tiles'], result[field+'_p50_over_gap_tiles'], result[field+'_p90_over_gap_tiles'] = map(float, np.quantile(values,[.1,.5,.9]))
        trace_summary.append(result)
    write_csv(output/'trace_summary.csv', trace_summary, list(trace_summary[0]) if trace_summary else ['group'])
    atomic_json(output/'summary.json', {'split':'val', 'complete': True, 'results':summaries,
        'group_threshold':chosen, 'actual_upper_intersection_counts':group_counts,
        'trace_rows':len(traces), 'tile_topology_rows':len(topology),
        'limitations':['B and positive_support stay fixed from the original forward.',
        'Interventions are non-deployable diagnostics; highest-neighbour choice is not ground-truth verified.',
        'Gap closure uses >=80% skeleton coverage and fixed-anchor connectivity within the local ROI; it is a proxy.',
        'Bridge counts are GT-buffered added-component proxies, relative to baseline and current P0 respectively.',
        'Matched Recall is selected on validation; probability ties and threshold differences are reported.',
        'Per-tile traces use fully-contained gaps and unambiguous work-grid mappings; overlapping tiles repeat gaps.',
        'Work-cell threshold comparisons are descriptive: original-resolution logit composition can change the effective threshold.',
        'Trace summary quantiles are over per-gap-per-tile means, not pooled pixel quantiles; overlapping windows repeat gaps.',
        'Higher source weight counts neighbours with U_neighbour > U_current; it is not a GT-confirmed correct relation.']})
    write_csv(output/'gap_comparison.csv', gap_rows, list(gap_rows[0]) if gap_rows else ['case'])
    lines = ['# Fixed-checkpoint update bottleneck audit', '',
             'Same B and positive_support; 8 neighbours, 3 steps; GT only for evaluation.',
             f'Source channel order (dy, dx): `{DIRECTIONS}`.', '',
             f'At reference threshold {chosen}: fixed-gap set intersections: `{json.dumps(group_counts)}`.',
             'See trace_summary.csv: quantiles are over gap/tile means, with unique-gap counts.', '']
    for threshold in thresholds:
        lines += [f'## Baseline threshold {threshold}', '',
                  '| Mode | Protocol | Threshold | IoU | Recall | FP | FN | +TP/lost TP | +FP/removed FP | New/lost closures | Bridges vs P0 | New bridges vs baseline | TP match error |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for mode in MODES + ('upper3',):
            for protocol in ('fixed', 'matched_recall'):
                x = summaries[f'{threshold}|{mode}|{protocol}']
                lines.append(f'| {mode} | {protocol} | {x["threshold"]:.6f} | {x["iou"]:.5f} | {x["recall"]:.5f} | {x["fp"]} | {x["fn"]} | {x["added_tp"]}/{x["lost_tp"]} | {x["added_fp"]}/{x["removed_fp"]} | {x["new_local_closed"]}/{x["lost_local_closed"]} | {x["bridge_proxy_vs_current_p0"]} | {x["new_bridge_proxy_vs_baseline"]} | {x["matched_tp_error"]} |')
        lines.append('')
    lines += ['Transport intervention raises its rate to at least 0.10 where the stay budget allows; this is a diagnostic setting, not an optimized value.',
              'Proposal interventions remove either the A multiplier or the 0.2 multiplier from the effective coefficient, while B remains unchanged.',
              'See gap_comparison.csv for actual gap-ID intersections and step_traces.csv for tile-level traces.',
              'Bridge and gap closure are proxies. Interventions do not establish deployable model quality.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--b_run_name', default='B_transport_0p01_proposal_0p001')
    parser.add_argument('--root_path', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected_gaps', type=int, default=198)
    parser.add_argument('--expected_images', type=int, default=14)
    args = parser.parse_args()
    if Path(args.b_run_name).name != args.b_run_name or not args.b_run_name.startswith('B'):
        raise ValueError('Invalid B run name')
    pair = args.pair_dir.resolve(); run = pair / args.b_run_name
    path = run/'best.pth'; manifest_path = pair/'reference/fixed_gaps_and_anchors.json'
    manifest = read_json(manifest_path); manifest_sha = digest(manifest_path)
    if manifest['count'] != args.expected_gaps or sum(len(v['gaps']) for v in manifest['images'].values()) != args.expected_gaps:
        raise ValueError('Unexpected fixed gap count')
    ckpt_sha = digest(path); checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if config['reference_sha256'] != manifest_sha: raise ValueError('Checkpoint/reference mismatch')
    selection_file = run/'eval/val_selection.json'
    thresholds = [.45]
    if selection_file.is_file():
        selection = read_json(selection_file)
        if selection['checkpoint_sha256'] != ckpt_sha or selection['reference_sha256'] != manifest_sha:
            raise ValueError('Validation selection metadata mismatch')
        selected = float(selection['threshold'])
        if abs(selected-.45) > 1e-8: thresholds.append(selected)
    root = (args.root_path or Path(config['root_path'])).resolve()
    verify_recorded_code(config)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    identity = {'checkpoint_sha256':ckpt_sha, 'reference_sha256':manifest_sha, 'dataset':str(root),
                'code_sha256':digest(Path(__file__)), 'thresholds':thresholds, 'modes':MODES,
                'device':torch.device(args.device).type, 'torch':str(torch.__version__)}
    identity = json.loads(json.dumps(identity))
    identity_path = output/'identity.json'
    if identity_path.exists() and read_json(identity_path) != identity: raise ValueError('Output identity mismatch')
    if not identity_path.exists() and any(output.iterdir()): raise ValueError('Output directory is not empty')
    atomic_json(identity_path, identity)
    cache = output/'cases'; cache.mkdir(exist_ok=True)
    seed_all(int(config['seed']))
    model = rebuild_model(checkpoint, torch.device(args.device)); del checkpoint
    dataset = full_dataset(root, 'val')
    if len(dataset) != args.expected_images or len(manifest['images']) != args.expected_images:
        raise ValueError('Validation image count mismatch')
    cases, all_traces, all_topology, seen = [], [], [], set()
    for i in range(len(dataset)):
        sample = dataset[i]; name = str(sample['case_name'])
        if name in seen or name not in manifest['images']: raise ValueError('Unexpected validation case: '+name)
        seen.add(name); gt = sample['mask'][0].detach().cpu().numpy() > .5
        entry = manifest['images'][name]
        if gt.shape != (1500, 1500) or list(gt.shape) != entry['shape']: raise ValueError('Unexpected image size')
        reference = pair/'reference'/(name+'.npz')
        if not reference.is_file(): reference = run/'eval/val'/(name+'.npz')
        with np.load(reference) as f:
            if not np.array_equal(f['gt'].astype(bool), gt): raise ValueError('GT mismatch: '+name)
        fingerprint = signature(sample); npz = cache/(name+'.npz'); meta = cache/(name+'.json')
        if npz.is_file() and meta.is_file():
            data = read_json(meta)
            if data['input_sha256'] != fingerprint: raise ValueError('Cached image changed')
            with np.load(npz) as f: maps = {key:f[key].copy() for key in f.files}
            traces = data['traces']; topology = data['topology']; print(f'[CACHE] {i+1}/{len(dataset)} {name}', flush=True)
        else:
            maps, traces, topology = infer_case(model, sample['image'][None].to(args.device), name, entry, thresholds[-1])
            validate_saved_prediction(pair, args.b_run_name, name, gt, {'p0':maps['p0'], 'p1':maps['baseline']})
            atomic_npz(npz, maps); atomic_json(meta, {'input_sha256':fingerprint, 'traces':traces, 'topology':topology})
        all_traces.extend(traces); all_topology.extend(topology); cases.append({'case':name, 'gt':gt, **maps})
        print(f'[DONE CASE] {i+1}/{len(dataset)} {name}', flush=True)
    if seen != set(manifest['images']): raise ValueError('Validation cases mismatch')
    write_csv(output/'step_traces.csv', all_traces, list(all_traces[0]) if all_traces else ['case'])
    write_csv(output/'tile_topology.csv', all_topology, list(all_topology[0]) if all_topology else ['case'])
    analyze(cases, manifest, thresholds, output, all_traces, all_topology)
    print('[DONE] '+str(output/'REPORT.md'), flush=True)


if __name__ == '__main__': main()
