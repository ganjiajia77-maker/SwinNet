"""Fixed native P0 gaps: whole paths, actual source choices and background shortcuts.

GT defines evaluation regions only. It is never passed to the source selector.
Internal maps use max-blend-window ownership, not a fictitious stitched latent.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import run_reliable_v9_mass as v9
from networks.v9_multiscale_transport import install, SOURCES
from networks.reliable_directional_repair import DIRECTIONS, shifted
from mass_protocol import blend_weight, sliding_positions
from diagnostics.v92_reachability import ordered_simple_path, longest_false_run, length_bin


def fixed_gaps(p0, skeleton, gt):
    """Reuse V9's 2px P0 tolerance, >=3 pixels and 7px flank convention."""
    missing = skeleton & ~(cv2.dilate(p0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
    _, old_labels = cv2.connectedComponents(p0.astype(np.uint8), connectivity=8)
    gaps = []
    for index in range(1, count):
        length = int(stats[index, cv2.CC_STAT_AREA])
        if length < 3:
            continue
        x, y, w, h = (int(v) for v in stats[index, :4])
        y0, y1 = max(0, y-7), min(gt.shape[0], y+h+7)
        x0, x1 = max(0, x-7), min(gt.shape[1], x+w+7)
        slices = np.s_[y0:y1, x0:x1]
        local = labels[slices] == index
        flank = cv2.dilate(local.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        old = old_labels[slices]
        anchors = np.unique(old[flank & gt[slices]])
        anchors = anchors[anchors > 0]
        # Keep exactly the same >=2 eligibility as original V9 gap_lengths.
        points = np.argwhere(local) + [y0, x0]
        order = ordered_simple_path(local)
        order = [(y0+yy, x0+xx) for yy, xx in order] if order is not None else None
        center = order[len(order)//2] if order else tuple(points[np.argmin(((points-points.mean(0))**2).sum(1))])
        gaps.append({'id': index, 'length': length, 'length_bin': length_bin(length),
            'slices': slices, 'mask': local, 'flank': flank, 'anchors': anchors,
            'old': old, 'points': points, 'order': order, 'center': center,
            'eligible': len(anchors) >= 2,
            'coordinate_sha256': hashlib.sha256(points.astype('<i4').tobytes()).hexdigest()})
    return gaps, old_labels


def connected(gap, prediction, gt):
    if not gap['eligible']:
        return None
    local_pred = prediction[gap['slices']]
    _, new = cv2.connectedComponents(local_pred.astype(np.uint8), connectivity=8)
    sets = [set(np.unique(new[gap['flank'] & (gap['old'] == label) & gt[gap['slices']]]))-{0}
            for label in gap['anchors']]
    return bool(local_pred[gap['mask']].mean() >= .8 and set.intersection(*sets))


def parallel_background_bridges(p0, prediction, gt, old_labels):
    """Conservative proxy, not authoritative wrong-edge GT or all parallel roads."""
    background = ~(cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    new_bg = prediction & ~p0 & background
    n, labels, stats, _ = cv2.connectedComponentsWithStats(new_bg.astype(np.uint8), connectivity=8)
    bridges, parallel = 0, 0
    for index in range(1, n):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        x, y, w, h = (int(v) for v in stats[index, :4])
        sl = np.s_[max(0, y-8):min(gt.shape[0], y+h+8), max(0, x-8):min(gt.shape[1], x+w+8)]
        region = labels[sl] == index
        near = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        components = np.unique(old_labels[sl][near]); components = components[components > 0]
        if len(components) < 2:
            continue
        bridges += 1
        tangents = []
        for label in components:
            points = np.argwhere((old_labels[sl] == label) & gt[sl])
            if len(points) < 5:
                continue
            values, vectors = np.linalg.eigh(np.cov(points.T))
            if (values[-1]-values[0])/max(values.sum(), 1e-6) < .5:
                continue
            tangents.append(vectors[:, -1])
        parallel += int(any(abs(float(a@b)) >= .9 for i, a in enumerate(tangents) for b in tangents[i+1:]))
    return bridges, parallel


@torch.inference_mode()
def infer(model, sample, device, min_effect):
    image = sample['image'][None].to(device); h, w = image.shape[-2:]
    gt = sample['mask'][0].numpy() > .5
    canvas = torch.zeros((2, 1, h, w), device=device); denominator = torch.zeros((1, 1, h, w), device=device)
    weight = blend_weight(512, device); native_weight = weight[0, 0].cpu().numpy()
    owner_score = np.full((h, w), -1., np.float32)
    names = ('transport', 'potential', 'available', 'stay', 'choice', 'choice_weight',
             'source_y', 'source_x', 'source_value', 'far_weight', 'generated_fp_weight', 'generated_tp_weight',
             'real_choice', 'real_choice_weight', 'repeated_generated_fp')
    maps = {name: np.zeros((3, h, w), np.float32) for name in names}
    for y in sliding_positions(h, 512, 256):
        for x in sliding_positions(w, 512, 256):
            result = model(image[:, :, y:y+512, x:x+512], capture_diagnostics=True)
            region = np.s_[y:y+512, x:x+512]
            canvas[:, :, y:y+512, x:x+512] += torch.cat((result['z0'], result['z1']), 0)*weight
            denominator[:, :, y:y+512, x:x+512] += weight
            owned = native_weight > owner_score[region]
            owner_score[region][owned] = native_weight[owned]
            wh, ww = result['u0'].shape[-2:]
            gt_work = F.interpolate(torch.from_numpy(gt[region].copy()).to(device).float()[None, None],
                                    size=(wh, ww), mode='area')
            yy, xx = torch.meshgrid(torch.arange(wh, device=device), torch.arange(ww, device=device), indexing='ij')
            fp_uses = []
            for step in range(3):
                state = result[f'u{step}']; p0 = result['u0']; reliable = result['r']
                if 'selector_weights' in result:
                    probs = result['selector_weights'][:, step]
                    source_values = result['selector_source_values'][:, step]
                else:
                    support = reliable+(1.-reliable)*((state-p0).clamp_min(0.)/(1.-p0).clamp_min(1e-5)).clamp_max(1.)
                    local_mass = result['source_mass'][:, step]
                    pieces = []
                    for i, (dy, dx, radius) in enumerate(SOURCES):
                        pieces.append(result['w'][:, i//4:i//4+1]*shifted(support, dy, dx)/local_mass.clamp_min(1e-6)
                                      if radius == 1 else torch.zeros_like(state))
                    source_weights = torch.cat(pieces, 1)
                    valid_mass = local_mass > 1e-6
                    source_weights = torch.where(valid_mass, source_weights, torch.zeros_like(source_weights))
                    probs = torch.cat(((~valid_mass).float(), source_weights), 1)
                    source_values = torch.cat([shifted(state, dy*r, dx*r) for dy, dx, r in SOURCES], 1)
                source_p0 = torch.cat([shifted(p0, dy*r, dx*r) for dy, dx, r in SOURCES], 1)
                source_gt = torch.cat([shifted(gt_work, dy*r, dx*r) for dy, dx, r in SOURCES], 1)
                newly_generated = source_values > source_p0+min_effect
                source_probs = probs[:, 1:]
                fp_uses.append(newly_generated & (source_gt < .05) & (source_probs >= .01))
                fp = (source_probs * newly_generated * (source_gt < .05)).sum(1, keepdim=True)
                tp = (source_probs * newly_generated * (source_gt >= .75)).sum(1, keepdim=True)
                choice_weight, choice = probs.max(1, keepdim=True)
                real_weight, real_choice = source_probs.max(1, keepdim=True)
                choices = choice[0, 0].cpu().numpy().astype(int)-1
                dy_table = torch.tensor([0]+[s[0]*s[2] for s in SOURCES], device=device)
                dx_table = torch.tensor([0]+[s[1]*s[2] for s in SOURCES], device=device)
                dy, dx = dy_table[choice], dx_table[choice]
                src_y = y+(yy[None, None]+dy+.5)*512/wh-.5
                src_x = x+(xx[None, None]+dx+.5)*512/ww-.5
                candidate_values = torch.cat((state, source_values), 1)
                selected_value = candidate_values.gather(1, choice)
                far = [1+i for i, (_, _, r) in enumerate(SOURCES) if r >= 4]
                values = {'transport': result['transport_delta'][:, step],
                    'potential': result['transport_rate'][:, step]*(result['m'][:, step]-state),
                    'available': result['source_available'][:, step], 'stay': probs[:, :1],
                    'choice': choices, 'choice_weight': choice_weight, 'source_y': src_y, 'source_x': src_x,
                    'source_value': selected_value, 'far_weight': probs[:, far].sum(1, keepdim=True),
                    'generated_fp_weight': fp, 'generated_tp_weight': tp,
                    'real_choice': real_choice, 'real_choice_weight': real_weight,
                    'repeated_generated_fp': (torch.stack(fp_uses, 1).sum(1) >= 2).any(1, keepdim=True).float()}
                for name, value in values.items():
                    array = value if isinstance(value, np.ndarray) else value[0, 0].cpu().numpy()
                    resized = cv2.resize(array.astype(np.float32), (512, 512), interpolation=cv2.INTER_NEAREST)
                    maps[name][step][region][owned] = resized[owned]
    if not bool((denominator > 0).all()):
        raise RuntimeError('Incomplete native inference')
    logits = (canvas/denominator).cpu().numpy()[:, 0]
    return 1./(1.+np.exp(-np.clip(logits, -80., 80.))), maps


def audit(args, model, config):
    from run_v9_multiscale import restore, write_csv, write_json
    from run_q3_r3_mass import full_dataset, PROTOCOL
    from evaluation.road_comparison.road_metrics import prepare_target, image_metrics, summarize
    # Both arms must already have validation selections. Fixed reference threshold
    # comes only from the original V9 validation result, never the test split.
    selection_path = args.source_checkpoint.parent.parent/'direct_val_selection.json'
    original_selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if Path(original_selection['checkpoint']).resolve() != args.source_checkpoint.resolve():
        raise ValueError('Original validation selection points to another checkpoint')
    if original_selection.get('selected_on') != 'val' or original_selection['identity'] != config['identity']:
        raise ValueError('Original validation selection identity differs')
    threshold = float(original_selection['threshold'])
    p0_threshold = float(config['training']['reference_threshold'])
    out = args.output_dir/('audit_'+args.audit_split+'_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    saved = {'original': torch.load(args.source_checkpoint, map_location='cpu', weights_only=False)}
    if v9.digest(args.source_checkpoint) != config['source_sha256']:
        raise ValueError('Source checkpoint fingerprint changed')
    selections = {'original': threshold}
    for mode in ('local', 'multiscale'):
        expected = dict(config, mode=mode)
        path = args.output_dir/mode/'best.pth'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['multiscale_config'] != expected:
            raise ValueError(f'{mode} does not share the same source, training budget and loss configuration')
        selected = json.loads((path.parent/'val_selection.json').read_text(encoding='utf-8'))
        if selected['checkpoint_sha256'] != v9.digest(path) or selected['config'] != expected:
            raise ValueError(f'{mode} validation selection is stale')
        selections[mode] = float(selected['threshold']); saved[mode] = payload
    data = full_dataset(args.root_path, args.audit_split)
    gap_rows, source_rows, image_rows, metric_rows = [], [], [], defaultdict(list)
    reference = hashlib.sha256(); bins = defaultdict(Counter); baseline_counts = Counter()
    for index in range(len(data)):
        sample = data[index]; case = str(sample['case_name']); gt = sample['mask'][0].numpy() > .5
        predictions, maps = {}, {}
        for name in ('original', 'local', 'multiscale'):
            install(model, 'local', args.seed)
            if name == 'original':
                for module in v9.MODULE_NAMES:
                    getattr(model, module).load_state_dict(saved[name]['modules'][module], strict=True)
                for module in ('gate_relation', 'state_gate'):
                    getattr(model, module).load_state_dict(saved[name][module], strict=True)
            else:
                install(model, name, args.seed)
                restore(model, args.output_dir/name/'best.pth', saved[name]['multiscale_config'])
            model.eval(); model.dynamics.blend = 1.
            probability, internal = infer(model, sample, args.device, args.min_effect)
            if predictions and not np.array_equal(probability[0], next(iter(predictions.values()))[0]):
                raise RuntimeError('Frozen P0 differs across arms')
            predictions[name] = probability; maps[name] = internal
        prepared = prepare_target(gt, PROTOCOL); p0 = predictions['original'][0] >= p0_threshold
        for value in (case.encode()+b'\0', np.asarray(gt.shape, dtype=np.int64).tobytes(),
                      *(np.packbits(mask, axis=None).tobytes() for mask in (p0, gt, prepared['skeleton']))):
            reference.update(value)
        gaps, old_labels = fixed_gaps(p0, prepared['skeleton'], gt)
        masks = {name: predictions[name][1] >= threshold for name in predictions}
        for field, probability in (('p0', predictions['original'][0]), ('p1', predictions['original'][1])):
            pred = probability >= threshold
            baseline_counts[field+'_tp'] += int((pred & gt).sum())
            baseline_counts[field+'_fp'] += int((pred & ~gt).sum())
            baseline_counts[field+'_fn'] += int((~pred & gt).sum())
        for name, mask in masks.items():
            metric_rows[name].append({'case_id': case, **image_metrics(mask, prepared, PROTOCOL)})
            bridges, parallel = parallel_background_bridges(p0, mask, gt, old_labels)
            bg = ~gt
            row = {'case': case, 'mode': name, 'threshold': threshold, 'background_bridge_proxy': bridges,
                   'parallel_background_bridge_proxy': parallel}
            for step in range(3):
                for kind in ('generated_fp_weight', 'generated_tp_weight', 'far_weight', 'stay'):
                    row[f'step{step+1}_{kind}_mean'] = float(maps[name][kind][step].mean())
                row[f'step{step+1}_background_positive_transport_mean'] = float(np.maximum(maps[name]['transport'][step][bg], 0).mean())
            row['same_generated_fp_source_used_twice_fraction'] = float(maps[name]['repeated_generated_fp'][-1].mean())
            image_rows.append(row)
        for gap in gaps:
            ys, xs = gap['points'].T; cy, cx = gap['center']
            for name, mask in masks.items():
                internal = maps[name]
                updated = (internal['transport'][:, ys, xs] > args.min_effect).any(0)
                potential = (internal['potential'][:, ys, xs] > args.min_effect).any(0)
                cumulative = internal['transport'][:, ys, xs].sum(0)
                closed = connected(gap, mask, gt)
                own_closed = connected(gap, predictions[name][1] >= selections[name], gt)
                row = {'case': case, 'gap_id': gap['id'], 'mode': name, 'length': gap['length'],
                    'length_bin': gap['length_bin'], 'eligible': gap['eligible'], 'simple_path': gap['order'] is not None,
                    'coordinate_sha256': gap['coordinate_sha256'], 'center_y': int(cy), 'center_x': int(cx),
                    'positive_potential_fraction': float(potential.mean()), 'positive_transport_fraction': float(updated.mean()),
                    'positive_transport_full_coverage': bool(updated.all()),
                    'center_has_positive_transport': bool((internal['transport'][:, cy, cx] > args.min_effect).any()),
                    'positive_net_transport_fraction': float((cumulative > args.min_effect).mean()),
                    'binary_gap_coverage': float(mask[ys, xs].mean()), 'anchors_connected_fixed_threshold': closed,
                    'anchors_connected_own_val_threshold': own_closed,
                    'longest_no_update_run': None}
                if gap['order'] is not None:
                    row['longest_no_update_run'] = longest_false_run([
                        bool((internal['transport'][:, y, x] > args.min_effect).any()) for y, x in gap['order']])
                gap_rows.append(row)
                counter = bins[(name, gap['length_bin'])]
                counter['all_gaps'] += 1; counter['eligible'] += gap['eligible']; counter['closed_fixed'] += int(closed or False)
                counter['center_positive_transport'] += row['center_has_positive_transport']
                counter['full_positive_transport_coverage'] += row['positive_transport_full_coverage']
                for step in range(3):
                    choice = int(internal['choice'][step, cy, cx])
                    dy, dx, radius = SOURCES[choice] if choice >= 0 else (0, 0, 0)
                    real_choice = int(internal['real_choice'][step, cy, cx])
                    rdy, rdx, rradius = SOURCES[real_choice]
                    source_rows.append({'case': case, 'gap_id': gap['id'], 'mode': name, 'step': step+1,
                        'length_bin': gap['length_bin'], 'center_y': int(cy), 'center_x': int(cx),
                        'source_choice': choice, 'dy': dy, 'dx': dx, 'radius_work': radius,
                        'source_native_y': float(internal['source_y'][step, cy, cx]),
                        'source_native_x': float(internal['source_x'][step, cy, cx]),
                        'source_U': float(internal['source_value'][step, cy, cx]),
                        'choice_weight': float(internal['choice_weight'][step, cy, cx]),
                        'top_real_source_dy': rdy, 'top_real_source_dx': rdx, 'top_real_source_radius': rradius,
                        'top_real_source_weight': float(internal['real_choice_weight'][step, cy, cx]),
                        'stay_weight': float(internal['stay'][step, cy, cx]),
                        'far_weight': float(internal['far_weight'][step, cy, cx]),
                        'positive_potential': float(internal['potential'][step, cy, cx]),
                        'actual_transport': float(internal['transport'][step, cy, cx])})
        print(f'[GAP AUDIT] {args.audit_split} {index+1}/{len(data)} {case} fixed_gaps={len(gaps)}', flush=True)
    reference_record = {'sha256': reference.hexdigest(), 'images': len(data),
        'eligible_p0_gaps': sum(row['mode']=='original' and row['eligible'] for row in gap_rows),
        'p0_threshold': p0_threshold, 'split': args.audit_split}
    if args.audit_split == 'val':
        for field in ('p0', 'p1'):
            for key in ('tp', 'fp', 'fn'):
                if baseline_counts[field+'_'+key] != original_selection['report'][field][key]:
                    raise ValueError(f'Original V9 validation counts no longer reproduce: {field}/{key}')
        if reference_record['eligible_p0_gaps'] != original_selection['report']['paired']['eligible_p0_gaps']:
            raise ValueError('Fixed-gap count differs from original V9 validation')
    if args.reference_set is not None:
        expected = json.loads(args.reference_set.read_text(encoding='utf-8'))
        expected_gaps = expected.get('eligible_reference_gaps', expected.get('eligible_p0_gaps'))
        if expected['sha256'] != reference_record['sha256'] or expected_gaps != reference_record['eligible_p0_gaps']:
            raise ValueError('Fixed P0/GT gap fingerprint differs from the provided reference set')
    write_json(out/'reference_set.json', reference_record)
    write_csv(out/'fixed_gaps.csv', gap_rows); write_csv(out/'gap_center_sources.csv', source_rows)
    write_csv(out/'generated_source_reuse_and_bridges.csv', image_rows)
    reports = [{'mode': name, 'threshold': threshold, **summarize(rows)} for name, rows in metric_rows.items()]
    write_csv(out/'fixed_threshold_comparison.csv', reports)
    write_json(out/'summary.json', {'source_sha256': config['source_sha256'], 'reference_set': reference_record,
        'same_threshold': threshold, 'own_val_thresholds': selections,
        'min_effect': args.min_effect,
        'gap_by_length': {f'{name}/{length}': dict(value) for (name, length), value in bins.items()},
        'metrics': reports,
        'limits': [
            'Gap length is the native GT skeleton missing-component pixel count, not Euclidean distance or a guaranteed repairable length.',
            'Closure uses original V9 >=2 P0 flank components, their shared predicted local component, and >=80% missing-centerline coverage; this is a fixed-gap proxy.',
            'Work-grid maps and chosen sources use the highest-blend-weight owning window. They are not a unique full-image latent state or a causal source attribution.',
            'Source argmax includes stay. Full softmax far mass and the weighted use of newly generated GT-background sources are reported separately.',
            'Background bridge/parallel bridge counts are geometric proxies; roads belonging to an already-connected P0 component may be missed. Inspect overlays before interpreting them as actual wrong links.',
            'APLS here is the existing raster approximation; topo_f1 equals clDice in this tool. No graph metric implementation was changed.',
            'Use val diagnostics for design. Test labels are only for a final confirmation, not model selection.']})
    (args.output_dir/('latest_audit_'+args.audit_split+'.txt')).write_text(str(out.resolve())+'\n', encoding='utf-8')
    print(f'[AUDIT DONE] {out}', flush=True)
