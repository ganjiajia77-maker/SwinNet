"""Fixed-weight V9 A/B/C/D intervention on the complete validation split."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

import run_reliable_v9_mass as v9
from diagnostics.diagnose_v9_transport import matched_threshold
from mass_protocol import blend_weight, sliding_positions

MODES = ('A_original', 'B_state_zero', 'C_source_fixed', 'D_transport_off')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def tile_variants(model, tile):
    """Reuse identical frozen features; recompute every downstream intervention."""
    original_gate = model.state_gate
    frozen = model.frozen(tile, capture_diagnostics=True)
    try:
        variants = {'A_original': model(None, frozen_override=frozen)}
        model.state_gate = None
        variants['B_state_zero'] = model(None, frozen_override=frozen)
        model.state_gate = original_gate
        variants['C_source_fixed'] = model(None, frozen_override=frozen,
                                          freeze_generated_sources=True)
        variants['D_transport_off'] = model(None, frozen_override=frozen,
                                           disable_transport=True)
    finally:
        model.state_gate = original_gate
    for name, result in variants.items():
        if not torch.allclose(result['z0'], variants['A_original']['z0'], rtol=0, atol=0):
            raise RuntimeError(f'{name} changed frozen P0')
    return variants


@torch.no_grad()
def infer_cases(model, dataset, device):
    caches = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample['image'][None].to(device)
        gt = sample['mask'][0].numpy() > .5
        height, width = gt.shape
        canvases = {key: torch.zeros((1, 1, height, width), device=device)
                    for key in ('p0', *MODES)}
        divisor = torch.zeros_like(canvases['p0'])
        weight = blend_weight(512, device)
        for y in sliding_positions(height, 512, 256):
            for x in sliding_positions(width, 512, 256):
                variants = tile_variants(model, image[:, :, y:y+512, x:x+512])
                region = (slice(None), slice(None), slice(y, y+512), slice(x, x+512))
                canvases['p0'][region] += variants['A_original']['z0'].float() * weight
                for name in MODES:
                    canvases[name][region] += variants[name]['z1'].float() * weight
                divisor[region] += weight
        if not (divisor > 0).all():
            raise RuntimeError('Sliding inference left pixels uncovered')
        probabilities = {}
        for name, canvas in canvases.items():
            logits = (canvas/divisor).cpu().numpy()[0, 0]
            if not np.isfinite(logits).all():
                raise RuntimeError(f'Nonfinite logits for {name}')
            probabilities[name] = 1/(1+np.exp(-np.clip(logits, -80, 80)))
        caches.append({'case': sample['case_name'], 'gt': gt, **probabilities})
        print(f'[COUNTERFACTUAL] {index+1}/{len(dataset)} {sample["case_name"]}', flush=True)
    return caches


def operating_points(caches, threshold, bins):
    target_tp = sum(int(((c['A_original'] >= threshold) & c['gt']).sum()) for c in caches)
    target_area = sum(int((c['A_original'] >= threshold).sum()) for c in caches)
    points = {}
    for mode in MODES:
        if mode == 'A_original':
            recall_t = area_t = threshold
        else:
            recall_t, _ = matched_threshold(caches, mode, target_tp, 'tp', bins)
            area_t, _ = matched_threshold(caches, mode, target_area, 'area', bins)
        points[mode] = {'same_threshold': threshold, 'matched_recall': recall_t,
                        'matched_foreground_area': area_t}
    return points


def evaluate(caches, points, reference_threshold, expected_gaps, output, modes=MODES, p0_threshold=None):
    from evaluation.road_comparison.road_metrics import image_metrics, prepare_target, summarize
    from run_q3_r3_mass import PROTOCOL
    from run_reliable_directional_mass import gap_lengths
    store = {(name, point): {'rows': [], 'paired': [], 'bins': []}
             for name in modes for point in points[name]}
    per_image, p0_rows = [], []
    reference_digest = hashlib.sha256()
    for index, case in enumerate(caches):
        gt = case['gt']; prepared = prepare_target(gt, PROTOCOL)
        p0 = case['p0'] >= reference_threshold
        reference_digest.update(case['case'].encode('utf-8') + b'\0')
        reference_digest.update(np.asarray(gt.shape, dtype=np.int64).tobytes())
        for mask in (p0, gt, prepared['skeleton']):
            reference_digest.update(np.packbits(mask, axis=None).tobytes())
        if p0_threshold is not None:
            p0_rows.append({'case_id': case['case'], **image_metrics(case['p0'] >= p0_threshold, prepared, PROTOCOL)})
        metric_cache, gap_cache = {}, {}
        for name in modes:
            for point, threshold in points[name].items():
                key = (name, threshold)
                pred = case[name] >= threshold
                if key not in metric_cache:
                    metric_cache[key] = image_metrics(pred, prepared, PROTOCOL)
                    gap_cache[key] = gap_lengths(p0, pred, prepared['skeleton'], gt)
                buckets, bridge = gap_cache[key]
                original_pred = case['A_original'] >= points['A_original']['same_threshold']
                stats = {'eligible': sum(item['eligible'] for item in buckets.values()),
                    'closed': sum(item['closed'] for item in buckets.values()), 'background_bridge_proxy': bridge,
                    'added_gt_pixels': int((pred & ~p0 & gt).sum()),
                    'added_background_pixels': int((pred & ~p0 & ~gt).sum()),
                    'removed_gt_pixels': int((p0 & ~pred & gt).sum()),
                    'removed_background_pixels': int((p0 & ~pred & ~gt).sum()),
                    'new_gt_vs_original': int((pred & ~original_pred & gt).sum()),
                    'new_background_vs_original': int((pred & ~original_pred & ~gt).sum()),
                    'removed_gt_vs_original': int((original_pred & ~pred & gt).sum()),
                    'removed_background_vs_original': int((original_pred & ~pred & ~gt).sum()),
                    'changed_binary_vs_original': int((pred != original_pred).sum()),
                    'probability_abs_diff_sum_vs_original': float(np.abs(case[name]-case['A_original']).sum()),
                    'pixels': int(gt.size)}
                data = store[(name, point)]
                data['rows'].append({'case_id': case['case'], **metric_cache[key]})
                data['paired'].append(stats); data['bins'].append(buckets)
                per_image.append({'case': case['case'], 'mode': name, 'point': point,
                                  'threshold': threshold, **stats})
        print(f'[TOPOLOGY] {index+1}/{len(caches)}', flush=True)
    reports, flat = {}, []
    p0_metrics = summarize(p0_rows) if p0_rows else None
    for name in modes:
        reports[name] = {}
        for point, threshold in points[name].items():
            data = store[(name, point)]
            metrics = summarize(data['rows'])
            paired = {k: sum(row[k] for row in data['paired']) for k in data['paired'][0]}
            if expected_gaps is not None and paired['eligible'] != expected_gaps:
                raise ValueError(f'P0 reference has {paired["eligible"]} eligible gaps, not {expected_gaps}; '
                    'check exact BASE/Q3, data split and P0 reference threshold; no conclusions were generated')
            paired['closure_rate_proxy'] = paired['closed']/max(paired['eligible'], 1)
            paired['probability_abs_diff_mean_vs_original'] = paired['probability_abs_diff_sum_vs_original']/max(paired['pixels'], 1)
            buckets = {length: {k: sum(row[length][k] for row in data['bins'])
                                for k in ('eligible', 'closed')} for length in data['bins'][0]}
            reports[name][point] = {'threshold': threshold, 'metrics': metrics, 'paired': paired,
                                    'gap_by_length': buckets}
    reference = reports['A_original']['same_threshold']['metrics']
    for name in modes:
        for point in points[name]:
            report = reports[name][point]; metrics = report['metrics']
            report['recall_difference_vs_A'] = metrics['recall']-reference['recall']
            report['foreground_area_difference_vs_A'] = metrics['tp']+metrics['fp']-reference['tp']-reference['fp']
            if p0_metrics is not None:
                report['recall_difference_vs_P0'] = metrics['recall']-p0_metrics['recall']
                report['foreground_area_difference_vs_P0'] = (metrics['tp']+metrics['fp']-
                                                               p0_metrics['tp']-p0_metrics['fp'])
            flat.append({'mode': name, 'point': point, 'threshold': report['threshold'], **metrics,
                         **report['paired'], 'recall_difference_vs_A': report['recall_difference_vs_A'],
                         'foreground_area_difference_vs_A': report['foreground_area_difference_vs_A'],
                         'recall_difference_vs_P0': report.get('recall_difference_vs_P0'),
                         'foreground_area_difference_vs_P0': report.get('foreground_area_difference_vs_P0')})
    if p0_metrics is not None:
        write_json(output/'p0_own_metrics.json', {'threshold': p0_threshold, 'metrics': p0_metrics})
    write_json(output/'reference_set.json', {'sha256': reference_digest.hexdigest(),
        'images': len(caches), 'p0_reference_threshold': reference_threshold,
        'eligible_reference_gaps': reports['A_original']['same_threshold']['paired']['eligible'],
        'definition': 'SHA256 over ordered case names, shapes, fixed P0 masks, GT masks and GT skeletons. '
                      'Gap and bridge algorithms are inherited from the unchanged V9 evaluation.'})
    write_csv(output/'counterfactual_metrics.csv', flat)
    write_csv(output/'counterfactual_per_image.csv', per_image)
    return reports


def selection_analysis(selection, output):
    rows = []
    for threshold, item in selection['threshold_curves'].items():
        metrics = item['p1']; paired = item['paired']; images = item['images']
        bridge_penalty = .01*paired['background_bridge_proxy']/max(images, 1)
        computed = (metrics['iou']+.15*metrics['cldice']+.1*metrics['apls_gt_to_pred_approx']+
                    .05*metrics['apls_bidirectional_approx']-.001*metrics['short_pred_components']-bridge_penalty)
        if abs(computed-item['selection_score']) > 1e-8:
            raise ValueError('Saved threshold score does not match the current selection formula')
        rows.append({'threshold': float(threshold), 'score': computed,
            'score_without_bridge_penalty_diagnostic_only': computed+bridge_penalty,
            'bridge_penalty': bridge_penalty, 'iou': metrics['iou'], 'f1': metrics['f1'],
            'recall': metrics['recall'], 'cldice': metrics['cldice'],
            'apls_gt_to_pred_approx': metrics['apls_gt_to_pred_approx'],
            'apls_bidirectional_approx': metrics['apls_bidirectional_approx'],
            'short_pred_components': metrics['short_pred_components'],
            'background_bridge_proxy': paired['background_bridge_proxy'],
            'closed_p0_gaps': paired['closed_p0_gaps'], 'eligible_p0_gaps': paired['eligible_p0_gaps']})
    rows.sort(key=lambda r: r['threshold'])
    write_csv(output/'threshold_score_components.csv', rows)
    return {'selected_threshold': float(selection['threshold']),
        'score_winner': max(rows, key=lambda r: r['score'])['threshold'],
        'without_bridge_penalty_winner_diagnostic_only': max(rows, key=lambda r: r['score_without_bridge_penalty_diagnostic_only'])['threshold'],
        'note': 'The no-penalty winner is diagnostic only; do not change selection rules using test results.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--root_path', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--expected_reference_gaps', type=int, default=206)
    p.add_argument('--matching_bins', type=int, default=10000)
    args = p.parse_args()
    if args.matching_bins < 1000:
        p.error('matching_bins must be at least 1000')
    checkpoint = args.run_dir/'stage_D/best.pth'; selection_path = args.run_dir/'direct_val_selection.json'
    if not checkpoint.is_file() or not selection_path.is_file():
        p.error('Need existing V9 best.pth and direct_val_selection.json')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Use an empty new output directory')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    selection = json.loads(selection_path.read_text())
    if selection.get('selected_on') != 'val' or selection['report']['images'] != 14:
        p.error('Require the full 14-image val selection')
    if payload['identity'] != selection['identity'] or Path(selection['checkpoint']).resolve() != checkpoint.resolve():
        p.error('Selection and checkpoint do not match')
    if Path(payload['training_config']['root_path']).resolve() != args.root_path.resolve():
        p.error('Data root differs from training')
    reference = float(payload['training_config']['reference_threshold'])
    if float(selection['report']['p0_reference_threshold']) != reference:
        p.error('P0 reference threshold differs from training')
    saved_gaps = selection['report']['paired']['eligible_p0_gaps']
    if args.expected_reference_gaps >= 0 and saved_gaps != args.expected_reference_gaps:
        p.error(f'Saved val selection has {saved_gaps} eligible P0 gaps, expected {args.expected_reference_gaps}; verify protocol before running the audit')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checksum = v9.digest(checkpoint)
    model, _, identity = v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(payload['identity']['q3']['path']), device=torch.device(args.device)))
    v9.restore_checkpoint(model, checkpoint, identity, 'D'); model.eval()
    from run_q3_r3_mass import full_dataset
    dataset = full_dataset(args.root_path, 'val')
    if len(dataset) != 14:
        p.error('Expected exactly 14 val images')
    score_info = selection_analysis(selection, args.output_dir)
    cases = infer_cases(model, dataset, torch.device(args.device))
    threshold = float(selection['threshold'])
    # Guard against stale/mismatched predictions before interpreting interventions.
    for field in ('p0', 'A_original'):
        pred_key = 'p0' if field == 'p0' else 'p1'
        totals = {'tp': sum(int(((c[field] >= threshold) & c['gt']).sum()) for c in cases),
                  'fp': sum(int(((c[field] >= threshold) & ~c['gt']).sum()) for c in cases),
                  'fn': sum(int(((c[field] < threshold) & c['gt']).sum()) for c in cases)}
        if any(totals[k] != selection['report'][pred_key][k] for k in totals):
            raise ValueError(f'Original {field} does not reproduce the saved validation sweep: {totals}')
    points = operating_points(cases, threshold, args.matching_bins)
    reports = evaluate(cases, points, reference,
        None if args.expected_reference_gaps < 0 else args.expected_reference_gaps, args.output_dir)
    output = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum, 'identity': identity,
        'images': len(cases), 'selected_on': 'val', 'original_threshold': threshold,
        'p0_reference_threshold': reference, 'matching_grid': 1/args.matching_bins,
        'modes': {'A_original': 'unchanged V9', 'B_state_zero': 'state_gate logit contribution zero',
                  'C_source_fixed': 'SourceSupport_k=R', 'D_transport_off': 'alpha_k=0'},
        'selection_analysis': score_info, 'comparisons': reports,
        'note': 'Fixed-checkpoint forward interventions, not retrained ablations. APLS is a raster approximation; closed gaps and bridges are proxy counts.'}
    if v9.digest(checkpoint) != checksum:
        raise RuntimeError('Original checkpoint changed on disk during the audit')
    write_json(args.output_dir/'counterfactual.json', output)
    lines = ['# V9 fixed-weight counterfactuals', '', f'P0 reference threshold: {reference}; '
             f'original validation threshold: {threshold}; images: {len(cases)}.', '',
             '| Mode | Point | Threshold | Recall difference vs A | IoU | clDice | bidirectional APLS | closed/eligible | background bridges | added GT / background | short components |',
             '|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|']
    for name in MODES:
        for point, row in reports[name].items():
            m, pair = row['metrics'], row['paired']
            lines.append(f'| {name} | {point} | {row["threshold"]:.4f} | {row["recall_difference_vs_A"]:+.6f} | '
                f'{m["iou"]:.5f} | {m["cldice"]:.5f} | {m["apls_bidirectional_approx"]:.5f} | '
                f'{pair["closed"]}/{pair["eligible"]} | {pair["background_bridge_proxy"]} | '
                f'{pair["added_gt_pixels"]}/{pair["added_background_pixels"]} | {m["short_pred_components"]:.3f} |')
    lines += ['', f'Selection score winner: {score_info["score_winner"]}; '
              f'without bridge penalty (diagnostic only): {score_info["without_bridge_penalty_winner_diagnostic_only"]}.',
              '', 'Read counterfactual.json for GT→Pred/Pred→GT APLS, length bins, area and recall residuals. '
              'All modes use the same 14 images and fixed P0 reference; proxy gap/bridge definitions are inherited unchanged from V9.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'[DONE] {args.output_dir / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
