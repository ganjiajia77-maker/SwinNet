"""Audit the saved v8 gate and, when its frozen sources exist, its val behavior.

The artifact report is always written first. Forward interventions require the
exact frozen base/Q3 checkpoints referenced by the v7 dependency.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.compare_v6_v7_matched import bridge_mask, write_csv  # noqa: E402
from diagnostics.diagnose_v8_failure import (  # noqa: E402
    _gradient_audit, _gradient_tile, _region_rows, _stitch_work_maps,
    _summary, _write_json,
)
from evaluation.road_comparison.road_metrics import (  # noqa: E402
    image_metrics, prepare_target, summarize,
)
from mass_protocol import blend_weight, sliding_positions  # noqa: E402
from run_q3_r3_mass import PROTOCOL, full_dataset  # noqa: E402
from run_reliable_directional_mass import gap_lengths, selection_score  # noqa: E402
from run_reliable_v8_mass import VERSION, load_experiment  # noqa: E402

THRESHOLDS = tuple(round(float(x), 2) for x in np.arange(.35, .751, .01))
SCORE_FORMULA = ('IoU + 0.15*clDice + 0.10*APLS_GT_to_Pred + '
                 '0.05*APLS_bidirectional - 0.001*short_pred_components - '
                 '0.01*background_bridge_components/images')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _training_history(run_dir):
    history = []
    path = run_dir / 'epoch_metrics.jsonl'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                history.append(json.loads(line))
    validation = {}
    pattern = re.compile(r'\[VAL\] epoch=(\d+) score=([\d.eE+-]+) IoU=([\d.eE+-]+)')
    searched = []
    log_paths = sorted(set(run_dir.glob('*.log')) | set(run_dir.glob('*log*.txt')))
    for path in log_paths:
        searched.append(str(path))
        with path.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    epoch, score, iou = match.groups()
                    validation[int(epoch)] = {'selection_score_at_0.60': float(score),
                                              'iou_at_0.60': float(iou),
                                              'source': str(path)}
    for row in history:
        row['validation'] = validation.get(int(row['epoch']))
        row['delta_g_amplitude_saved'] = None
        row['gradient_norms_saved'] = None
    return {'epochs': history, 'validation_logs_searched': searched,
            'missing_history_note': ('The original trainer recorded train metrics each epoch, '
                                     'but val metrics only in console logs. It did not save '
                                     'per-epoch DeltaG maps or relation gradients.')}


def _artifact_report(args, checkpoint, selection, original):
    state = checkpoint['gate_relation']
    parameters = {}
    for name, value in state.items():
        if torch.is_tensor(value):
            data = value.detach().float()
            parameters[name] = {'shape': list(data.shape), 'finite': bool(torch.isfinite(data).all()),
                                'l2': float(data.norm()), 'mean_abs': float(data.abs().mean())}
    history = _training_history(args.run_dir)
    for row in history['epochs']:
        if (int(row['epoch']) == int(checkpoint['epoch']) and
                row['validation'] is None):
            row['validation'] = {'selection_score_at_0.60': checkpoint.get('best_score'),
                                 'iou_at_0.60': None, 'source': 'best.pth'}
    return {
        'scope': 'v8 saved checkpoint and validation artifacts; no model forward',
        'v8_checkpoint': str(args.v8_checkpoint), 'v8_sha256': _sha256(args.v8_checkpoint),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'training_best_score_at_threshold_0.60': checkpoint.get('best_score'),
        'checkpoint_relation_mode': checkpoint.get('relation_mode'),
        'v7_checkpoint_reference': checkpoint.get('v7_checkpoint'),
        'validation_selected_threshold': selection.get('threshold'),
        'validation_selection_score': original.get('selection_score'),
        'validation_summary_threshold': original.get('val_selected_threshold',
                                                    original.get('threshold')),
        'selection_score_formula': SCORE_FORMULA,
        'score_explanation': ('best_score was chosen during training using val at threshold '
                              '0.60; the saved validation selection scans 0.35 through 0.75. '
                              'These are different operating points, not duplicate scores.'),
        'saved_validation': {
            'images': original.get('images'), 'p1': original.get('p1'),
            'paired': original.get('paired')},
        'gate_relation_parameters': parameters,
        'training_history': history,
    }


@torch.no_grad()
def _intervention_probabilities(model, image):
    device = image.device
    mass = torch.zeros((1, 1, 1500, 1500), device=device)
    weighted = {name: torch.zeros_like(mass) for name in
                ('delta_zero', 'shuffle_h_h3', 'transport_off', 'proposal_off')}
    weight = blend_weight(512, device)
    positions = sliding_positions(1500, 512, 256)
    for top in positions:
        for left in positions:
            tile = image[:, :, top:top+512, left:left+512]
            frozen = model.frozen(tile, capture_diagnostics=True)
            variants = {
                'delta_zero': model(tile, frozen_override=frozen,
                                    gate_intervention='zero_delta')['z1'],
                'shuffle_h_h3': model(tile, frozen_override=frozen,
                                      gate_intervention='shuffle_h_h3')['z1'],
                'transport_off': model(tile, frozen_override=frozen,
                                       disable_transport=True)['z1'],
                'proposal_off': model(tile, frozen_override=frozen,
                                      disable_proposal=True)['z1'],
            }
            region = (..., slice(top, top+512), slice(left, left+512))
            mass[region] += weight
            for name, z1 in variants.items():
                weighted[name][region] += z1.float() * weight
    if (mass <= 0).any():
        raise RuntimeError('The diagnostic sliding windows did not cover the image')
    return {name: (value / mass).sigmoid()[0, 0].cpu().numpy()
            for name, value in weighted.items()}


def _metric_row(probability, gt, prepared, p0_probability, threshold, case, variant):
    p0 = p0_probability >= threshold
    prediction = probability >= threshold
    bins, bridges = gap_lengths(p0, prediction, prepared['skeleton'], gt)
    return {'case_id': case, 'variant': variant, 'threshold': threshold,
            **image_metrics(prediction, prepared, PROTOCOL),
            'eligible_p0_gaps': sum(x['eligible'] for x in bins.values()),
            'closed_p0_gaps': sum(x['closed'] for x in bins.values()),
            'background_bridge_components': bridges,
            'background_bridge_pixels': int(bridge_mask(p0, prediction, gt).sum()),
            'added_tp_vs_p0': int((prediction & ~p0 & gt).sum()),
            'added_fp_vs_p0': int((prediction & ~p0 & ~gt).sum())}


def _score_curve(rows, images):
    curves = []
    for variant in ('v7', 'v8'):
        for threshold in THRESHOLDS:
            subset = [row for row in rows if row['variant'] == variant and
                      row['threshold'] == threshold]
            metric_rows = [{key: value for key, value in row.items()
                            if key not in ('variant', 'threshold', 'eligible_p0_gaps',
                                           'closed_p0_gaps', 'background_bridge_components',
                                           'background_bridge_pixels', 'added_tp_vs_p0',
                                           'added_fp_vs_p0')}
                           for row in subset]
            metrics = summarize(metric_rows)
            bridge_count = sum(row['background_bridge_components'] for row in subset)
            curves.append({'variant': variant, 'threshold': threshold,
                           'images': len(subset), 'selection_score': selection_score(
                               metrics, {'background_bridge_proxy': bridge_count}, images),
                           'iou': metrics['iou'], 'f1': metrics['f1'],
                           'precision': metrics['precision'], 'recall': metrics['recall'],
                           'cldice': metrics['cldice'],
                           'apls_gt_to_pred_approx': metrics['apls_gt_to_pred_approx'],
                           'apls_pred_to_gt_approx': metrics['apls_pred_to_gt_approx'],
                           'apls_bidirectional_approx': metrics['apls_bidirectional_approx'],
                           'foreground_pixels': metrics['tp'] + metrics['fp'],
                           'short_pred_components': metrics['short_pred_components'],
                           'closed_p0_gaps': sum(row['closed_p0_gaps'] for row in subset),
                           'eligible_p0_gaps': sum(row['eligible_p0_gaps'] for row in subset),
                           'background_bridge_components': bridge_count,
                           'background_bridge_pixels': sum(
                               row['background_bridge_pixels'] for row in subset)})
    return curves


def _comparison_at_workpoint(curve, rows, threshold):
    v8 = next(row for row in curve if row['variant'] == 'v8' and
              abs(row['threshold'] - threshold) < 1e-6)
    v7_rows = [row for row in curve if row['variant'] == 'v7']
    v7 = min(v7_rows, key=lambda row: (abs(row['recall'] - v8['recall']),
                                      abs(row['foreground_pixels'] - v8['foreground_pixels'])))
    paired = [row for row in rows if row['variant'] == 'v8' and
              abs(row['threshold'] - threshold) < 1e-6]
    return {'v8': v8, 'v7_matched_recall': v7,
            'recall_difference': v8['recall'] - v7['recall'],
            'foreground_area_difference': (v8['foreground_pixels'] -
                                           v7['foreground_pixels']),
            'matched_by': 'nearest global recall on validation thresholds 0.35:0.01:0.75',
            'v8_per_image': [{key: row[key] for key in
                              ('case_id', 'closed_p0_gaps',
                               'background_bridge_components',
                               'background_bridge_pixels')}
                             for row in paired]}


def _reconcile(original, matched, run_dir):
    saved = original.get('paired') or {}
    current = matched['v8']
    per_image_file = run_dir / 'eval' / 'val' / 'paired_per_image.csv'
    saved_rows = {}
    if per_image_file.is_file():
        with per_image_file.open(newline='', encoding='utf-8') as handle:
            saved_rows = {row['case_id']: row for row in csv.DictReader(handle)}
    per_image = []
    for row in matched['v8_per_image']:
        old = saved_rows.get(row['case_id'], {})
        per_image.append({'case_id': row['case_id'],
                          'saved_closed': old.get('closed_p0_gaps'),
                          'recomputed_closed': row['closed_p0_gaps'],
                          'saved_bridge_components': old.get('background_bridge_proxy'),
                          'recomputed_bridge_components': row['background_bridge_components'],
                          'recomputed_bridge_pixels': row['background_bridge_pixels']})
    return {'saved_threshold': original.get('val_selected_threshold', original.get('threshold')),
            'saved_iou': (original.get('p1') or {}).get('iou'),
            'recomputed_iou': current['iou'],
            'saved_closed_p0_gaps': saved.get('closed_p0_gaps'),
            'recomputed_closed_p0_gaps': current['closed_p0_gaps'],
            'saved_bridge_components': saved.get('background_bridge_proxy'),
            'recomputed_bridge_components': current['background_bridge_components'],
            'recomputed_bridge_pixels': current['background_bridge_pixels'],
            'definitions': ('gap_lengths counts complete eligible P0 gap components and '
                            'background bridge components; bridge_mask.sum counts pixels. '
                            'All recomputed counts use the same val image, threshold, GT '
                            'preparation, and P0->P1 reference.'),
            'per_image': per_image}


def _region_contributions(maps, gt, v7_threshold, v8_threshold):
    p7 = maps['p7_full'][0] >= v7_threshold
    p8 = maps['p1_full'][0] >= v8_threshold
    new_fp = p8 & ~p7 & ~gt
    new_tp = p8 & ~p7 & gt
    work_size = maps['transport_delta_1'][0].shape
    rows = []
    for name, native_mask in (('v8_extra_tp', new_tp), ('v8_extra_fp', new_fp)):
        work_mask = cv2.resize(native_mask.astype(np.uint8),
                               (work_size[1], work_size[0]),
                               interpolation=cv2.INTER_AREA) > .25
        row = {'region': name, 'native_pixels': int(native_mask.sum()),
               'work_pixels': int(work_mask.sum())}
        for source, map_name in (('alpha_times_m_minus_u', 'transport_delta'),
                                 ('beta_times_b_minus_u', 'proposal_delta'),
                                 ('gamma_times_p0_minus_u', 'anchor_delta')):
            for step in (1, 2, 3):
                values = maps[f'{map_name}_{step}'][0][work_mask]
                row[f'{source}_{step}'] = _summary(values)
                row[f'{source}_{step}_mean_abs'] = (float(np.abs(values).mean())
                                                   if values.size else None)
        for name_map, near in (('delta_z_plus', 7.9), ('delta_z_minus', 3.9)):
            values = maps[name_map][0][work_mask]
            row[name_map] = _summary(values)
            row[f'{name_map}_saturation_fraction'] = (float((values >= near).mean())
                                                       if values.size else None)
        rows.append(row)
    return rows


def _forward_report(args, artifact, original, selection, checkpoint):
    config = SimpleNamespace(v7_checkpoint=args.v7_checkpoint,
                             v8_checkpoint=args.v8_checkpoint,
                             relation_mode=checkpoint['relation_mode'],
                             device=torch.device(args.device),
                             base_checkpoint_override=args.base_checkpoint,
                             q3_checkpoint_override=args.q3_checkpoint)
    model, v7_payload = load_experiment(config)
    model.eval()
    dataset = full_dataset(args.root_path, 'val')
    if len(dataset) != 14:
        raise ValueError(f'Expected 14 validation images, found {len(dataset)}')
    threshold = round(float(selection['threshold']), 2)
    if threshold not in THRESHOLDS:
        raise ValueError('The saved validation threshold is outside the audited grid')
    curve_rows, variants, region_rows, dynamics_rows, gradient_locations = [], [], [], [], {}
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample['image'][None].to(config.device)
        gt = sample['mask'][0].numpy() > .5
        prepared = prepare_target(gt, PROTOCOL)
        maps = _stitch_work_maps(model, image, config.device, minimal=True)
        p0_probability = maps['p0_full'][0]
        cases = {'v7': maps['p7_full'][0], 'v8': maps['p1_full'][0]}
        gradient_locations[sample['case_name']] = _gradient_tile(
            gt, p0_probability, threshold)
        for current in THRESHOLDS:
            for name in ('v7', 'v8'):
                curve_rows.append(_metric_row(cases[name], gt, prepared,
                                              p0_probability, current,
                                              sample['case_name'], name))
        interventions = _intervention_probabilities(model, image)
        zero_error = float(np.abs(interventions['delta_zero'] - cases['v7']).max())
        if zero_error > 2e-5:
            raise RuntimeError(f'DeltaG=0 differs from v7 by {zero_error} on '
                               f'{sample["case_name"]}')
        base_prediction = cases['v8'] >= threshold
        for name, probability in {'v8_original': cases['v8'],
                                  **interventions}.items():
            row = _metric_row(probability, gt, prepared, p0_probability,
                              threshold, sample['case_name'], name)
            row['p1_mean_abs_vs_v8'] = float(np.abs(probability - cases['v8']).mean())
            row['changed_pixels_vs_v8'] = int(np.count_nonzero(
                (probability >= threshold) != base_prediction))
            variants.append(row)
        record = {'case': sample['case_name'], 'gt': gt,
                  **{key: (value[0] if value.ndim == 3 and value.shape[0] == 1
                           else value) for key, value in maps.items()
                     if key in ('p0', 'q', 'g7_logit', 'g7', 'delta_g', 'g8',
                                'g8_minus_g7', 'p0_full', 'p1_full', 'p7_full',
                                'q_full', 'r', 'a', 'b', 'w', 'delta_z_plus',
                                'delta_z_minus', 'proposal_raw_plus',
                                'proposal_raw_minus') or key.startswith((
                                    'alpha_', 'beta_', 'gamma_', 'transport_delta_',
                                    'proposal_delta_', 'anchor_delta_'))}}
        region_rows.extend(_region_rows(record, threshold))
        print(f'[V8 DIAG] val {index+1}/{len(dataset)} {sample["case_name"]}', flush=True)
    curve = _score_curve(curve_rows, len(dataset))
    matched = _comparison_at_workpoint(curve, curve_rows, threshold)
    reconciliation = _reconcile(original, matched, args.run_dir)
    write_csv(args.output_dir / 'threshold_curve.csv', curve)
    write_csv(args.output_dir / 'variant_per_image.csv', variants)
    write_csv(args.output_dir / 'gate_regions_per_image.csv', region_rows)
    _write_json(args.output_dir / 'bridge_reconciliation.json', reconciliation)
    gradients = []
    loss_weights = {key: float(v7_payload[key]) for key in
                    ('lambda_bg', 'lambda_missing', 'lambda_cont')}
    for index in range(len(dataset)):
        sample = dataset[index]
        maps = _stitch_work_maps(model, sample['image'][None].to(config.device),
                                 config.device, minimal=True)
        dynamics_rows.extend({'case_id': sample['case_name'], **row}
                             for row in _region_contributions(
                                 maps, sample['mask'][0].numpy() > .5,
                                 matched['v7_matched_recall']['threshold'], threshold))
        case_row = next(row for row in curve_rows if row['case_id'] == sample['case_name']
                        and row['variant'] == 'v8' and row['threshold'] == threshold)
        # The tile choice is based on the frozen surface, never on v8's repaired output.
        pixels, top, left = gradient_locations[sample['case_name']]
        gradients.append({'case_id': sample['case_name'], 'reference_metric_iou':
                          case_row['iou'], **_gradient_audit(
                              model, sample, config.device, loss_weights,
                              top, left, pixels)})
    _write_json(args.output_dir / 'dynamic_contributions.json', dynamics_rows)
    _write_json(args.output_dir / 'gradient_audit.json', gradients)
    effect = {}
    for name in ('v8_original', 'delta_zero', 'shuffle_h_h3',
                 'transport_off', 'proposal_off'):
        rows = [row for row in variants if row['variant'] == name]
        effect[name] = {'p1_mean_abs_vs_v8_per_image': float(np.mean(
                            [row['p1_mean_abs_vs_v8'] for row in rows])),
                        'tp': sum(row['tp'] for row in rows),
                        'fp': sum(row['fp'] for row in rows),
                        'closed_p0_gaps': sum(row['closed_p0_gaps'] for row in rows),
                        'background_bridge_components': sum(
                            row['background_bridge_components'] for row in rows)}
    gate_gradients = {
        name: _summary([item['per_module_gradient_l2'][name] for item in gradients])
        for name in ('highres_project', 'decoder_project', 'pair', 'delta')}
    fp_dynamics = [row for row in dynamics_rows if row['region'] == 'v8_extra_fp']
    fp_update_magnitude = {}
    for key in ('alpha_times_m_minus_u', 'beta_times_b_minus_u',
                'gamma_times_p0_minus_u'):
        fp_update_magnitude[key] = _summary([
            row[f'{key}_{step}_mean_abs'] for row in fp_dynamics for step in (1, 2, 3)
            if row[f'{key}_{step}_mean_abs'] is not None])
    gate_removed_effect = effect['delta_zero']['p1_mean_abs_vs_v8_per_image']
    semantic_shuffle_effect = effect['shuffle_h_h3']['p1_mean_abs_vs_v8_per_image']
    fp_reduced_by_transport = (effect['v8_original']['fp'] - effect['transport_off']['fp'])
    fp_reduced_by_proposal = (effect['v8_original']['fp'] - effect['proposal_off']['fp'])
    if gate_removed_effect < 1e-6:
        semantic_answer = 'DeltaG removal barely changes P1; the v8 gate has no measured forward effect.'
    elif semantic_shuffle_effect < .1 * gate_removed_effect:
        semantic_answer = ('DeltaG changes P1, but shuffling H/H3 has less than 10% of '
                           'that effect; spatial semantic use is weak at this checkpoint.')
    else:
        semantic_answer = ('Both DeltaG removal and H/H3 shuffle change P1. Compare '
                           'recovered gaps against bridge/FP counts before claiming '
                           'better connection discrimination.')
    if fp_reduced_by_transport > fp_reduced_by_proposal and fp_reduced_by_transport > 0:
        priority = 'Inspect transport permission G and its source evidence first.'
    elif fp_reduced_by_proposal > fp_reduced_by_transport and fp_reduced_by_proposal > 0:
        priority = 'Inspect B proposal and A acceptance first.'
    else:
        priority = 'The two forward removals do not isolate one dominant FP path.'
    artifact['forward'] = {
        'split': 'val', 'images': len(dataset),
        'operating_threshold': threshold,
        'v8_selection_score_recomputed': matched['v8']['selection_score'],
        'matched_recall': matched,
        'intervention_effect': effect,
        'current_checkpoint_gradient_l2': gate_gradients,
        'matched_recall_extra_fp_update_mean_abs': fp_update_magnitude,
        'answers_to_four_questions': {
            'semantic_connection_decision': semantic_answer,
            'why_semantic_gain_is_limited': (
                'Use the H/H3 shuffle effect, regional DeltaG distributions, and '
                'current-checkpoint highres_project/decoder_project/pair gradients '
                'together; historical per-epoch gradients were not saved.'),
            'background_fp_source': (
                f'Transport-off removes {fp_reduced_by_transport} FP; proposal-off '
                f'removes {fp_reduced_by_proposal} FP at the selected threshold. '
                'Both are interaction-inclusive counterfactuals.'),
            'priority': priority},
        'bridge_reconciliation': reconciliation,
        'region_distribution_file': 'gate_regions_per_image.csv',
        'gradient_file': 'gradient_audit.json',
        'dynamics_file': 'dynamic_contributions.json',
        'gate_regions': {
            name: {key: _summary([row[f'{key}_mean'] for row in region_rows
                                  if row['region'] == name and
                                  row.get(f'{key}_mean') is not None])
                   for key in ('g7_logit', 'delta_g', 'g7', 'g8', 'g8_minus_g7')}
            for name in ('recovered_gt', 'unrecovered_gt', 'hard_background',
                         'isolated_new_fp', 'false_bridge')},
        'interpretation_note': ('Per-step signed deltas are exact forward terms. '
                                'Transport/proposal-off results include interactions; '
                                'neither is an additive causal partition.')}
    return artifact


def _markdown(report):
    history = report['training_history']['epochs']
    rows = ['# V8 Gate Diagnosis', '',
            f"Checkpoint epoch: {report['checkpoint_epoch']}; "
            f"best_score at 0.60: {report['training_best_score_at_threshold_0.60']}; "
            f"validation-selected threshold: {report['validation_selected_threshold']}; "
            f"saved selection_score: {report['validation_selection_score']}.", '',
            f"Selection formula: `{SCORE_FORMULA}`.", '',
            '## Training record', '',
            '| Epoch | Train loss | L_seg | L_cont | Val score at 0.60 | Val IoU at 0.60 |',
            '|---:|---:|---:|---:|---:|---:|']
    for row in history:
        val = row.get('validation') or {}
        rows.append(f"| {row['epoch']} | {row.get('loss')} | {row.get('L_seg')} | "
                    f"{row.get('L_cont')} | {val.get('selection_score_at_0.60')} | "
                    f"{val.get('iou_at_0.60')} |")
    rows.extend(['', report['training_history']['missing_history_note'], ''])
    if 'forward' not in report:
        rows.append('Forward evidence unavailable: the saved v8 checkpoint contains only '
                    'gate_relation weights. The exact frozen model dependencies or cached '
                    'feature maps are needed to reconstruct H/H3 and dynamics.')
        return '\n'.join(rows) + '\n'
    forward = report['forward']
    selected = forward['matched_recall']['v8']
    baseline = forward['matched_recall']['v7_matched_recall']
    effects = forward['intervention_effect']
    reconcile = forward['bridge_reconciliation']
    rows.extend(['## 1. V8 relation effect', '',
                 f"V8 at {selected['threshold']:.2f}: IoU={selected['iou']:.5f}, "
                 f"clDice={selected['cldice']:.5f}, foreground pixels="
                 f"{selected['foreground_pixels']}, "
                 f"closed gaps={selected['closed_p0_gaps']}, "
                 f"bridge components={selected['background_bridge_components']}.",
                 f"DeltaG=0 changes P1 by {effects['delta_zero']['p1_mean_abs_vs_v8_per_image']:.6g} "
                 f"on average; H/H3 spatial shuffle changes it by "
                 f"{effects['shuffle_h_h3']['p1_mean_abs_vs_v8_per_image']:.6g}.",
                 'The fixed-checkpoint counterfactuals are in `variant_per_image.csv`; '
                 'old G logit, DeltaG, G7, G8, and G8-G7 distributions are in '
                 '`gate_regions_per_image.csv`.', '',
                 '## 2. Gradients and training', '',
                 'Current best-checkpoint validation gradients by gate_relation layer are '
                 'in `gradient_audit.json`. They are not historical training gradients. '
                 'Per-epoch DeltaG maps and gradients were not saved by the original trainer.', '',
                 '## 3. Threshold and bridge counts', '',
                 f"Saved closed={reconcile['saved_closed_p0_gaps']}, recomputed "
                 f"closed={reconcile['recomputed_closed_p0_gaps']}; saved bridge "
                 f"components={reconcile['saved_bridge_components']}, recomputed bridge "
                 f"components={reconcile['recomputed_bridge_components']}, bridge "
                 f"pixels={reconcile['recomputed_bridge_pixels']}.",
                 'The per-threshold curve is `threshold_curve.csv`; per-image disagreements '
                 'and exact units are in `bridge_reconciliation.json`.', '',
                 '## 4. Dynamic update sources', '',
                 f"Matched-recall V7 threshold={baseline['threshold']:.2f}; "
                 f"V8 threshold={selected['threshold']:.2f}; global recall difference="
                 f"{forward['matched_recall']['recall_difference']:.6f}.",
                 'Signed alpha*(M-U), beta*(B-U), gamma*(P0-U) for the extra TP/FP '
                 'are in `dynamic_contributions.json`. Transport/proposal-off P1 and '
                 'FP contrasts are in `variant_per_image.csv`.', '',
                 '## Four Answers', ''])
    for question, answer in forward['answers_to_four_questions'].items():
        rows.append(f'- **{question}:** {answer}')
    rows.append('')
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--root_path', type=Path)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--v7_checkpoint', type=Path)
    parser.add_argument('--base_checkpoint', type=Path)
    parser.add_argument('--q3_checkpoint', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--artifacts_only', action='store_true')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.v8_checkpoint = args.run_dir / 'best.pth'
    selection_file = args.run_dir / 'eval' / 'val_selection.json'
    summary_file = args.run_dir / 'eval' / 'val' / 'summary.json'
    for path in (args.v8_checkpoint, selection_file, summary_file):
        if not path.is_file():
            parser.error(f'Missing V8 artifact: {path}')
    checkpoint = torch.load(args.v8_checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('version') != VERSION:
        parser.error(f'Unexpected V8 checkpoint version: {checkpoint.get("version")}')
    selection = json.loads(selection_file.read_text(encoding='utf-8'))
    original = json.loads(summary_file.read_text(encoding='utf-8'))
    if selection.get('selected_on') != 'val' or selection.get('relation_mode') != checkpoint.get('relation_mode'):
        parser.error('V8 selection does not match this checkpoint/mode')
    if selection.get('v8_checkpoint') != str(args.v8_checkpoint.resolve()):
        parser.error('V8 validation selection references a different checkpoint')
    if original.get('checkpoint') != str(args.v8_checkpoint.resolve()):
        parser.error('V8 validation summary references a different checkpoint')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact = _artifact_report(args, checkpoint, selection, original)
    _write_json(args.output_dir / 'v8_artifacts.json', artifact)
    (args.output_dir / 'REPORT.md').write_text(_markdown(artifact), encoding='utf-8')
    if args.artifacts_only:
        print(f'[V8] Saved artifact report: {args.output_dir / "v8_artifacts.json"}')
        return
    if args.root_path is None:
        parser.error('--root_path is required for validation forward diagnostics')
    args.v7_checkpoint = (args.v7_checkpoint or Path(checkpoint['v7_checkpoint'])).resolve()
    if not args.v7_checkpoint.is_file():
        parser.error(f'V8 forward dependency missing: {args.v7_checkpoint}')
    v7 = torch.load(args.v7_checkpoint, map_location='cpu', weights_only=False)
    base = (args.base_checkpoint or Path(v7['base_checkpoint'])).resolve()
    q3 = (args.q3_checkpoint or Path(v7['q3_checkpoint'])).resolve()
    missing = [str(path) for path in (base, q3) if not path.is_file()]
    if missing:
        parser.error('V8 forward requires its exact frozen checkpoint dependencies: ' +
                     ', '.join(missing) + '. v8_artifacts.json was written; feature '
                     'interventions and dynamics cannot be reconstructed from best.pth alone.')
    args.base_checkpoint = base if args.base_checkpoint else None
    args.q3_checkpoint = q3 if args.q3_checkpoint else None
    report = _forward_report(args, artifact, original, selection, checkpoint)
    _write_json(args.output_dir / 'v8_minimal_diagnosis.json', report)
    (args.output_dir / 'REPORT.md').write_text(_markdown(report), encoding='utf-8')
    print(f'[V8] Completed 14-image validation diagnosis: {args.output_dir}')


if __name__ == '__main__':
    main()
