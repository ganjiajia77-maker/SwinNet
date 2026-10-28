"""Compare original V9, matched continuation control and L_gate on val only."""
import argparse
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
from diagnostics.v91_counterfactual import evaluate, write_json, write_csv
from mass_protocol import blend_weight, sliding_positions
from networks.reliable_directional_repair import ReliableDirectionalRepair

MODES = ('A_original', 'B_control', 'C_gate')


def create_variant(frozen, path, identity, device):
    model = ReliableDirectionalRepair(frozen, width=32, steps=3, bridge_route=True,
        conditional_transport=True, transport_gate_mode='v7', gate_relation_mode='full').to(device)
    model.state_gate = v9.StateGate().to(device)
    v9.restore_checkpoint(model, path, identity, 'D')
    model.eval()
    return model


def selection_for(model, checkpoint, name, args, identity, reference):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    path = args.output_dir/f'{name}_val_selection.json'
    thresholds = tuple(round(float(x), 2) for x in np.arange(.35, .751, .01))
    reports, _, _, _ = evaluate_thresholds(model, full_dataset(args.root_path, 'val'),
        args.device, thresholds, reference_threshold=reference)
    threshold = max(reports, key=lambda t: reports[t]['selection_score'])
    result = {'selected_on': 'val', 'threshold': threshold, 'identity': identity,
              'checkpoint': str(checkpoint.resolve()), 'report': reports[threshold], 'threshold_curves': reports}
    write_json(path, result)
    return result


@torch.no_grad()
def infer(models, dataset, device, region_context=None):
    cases, regional = [], {}
    for i, sample in enumerate(dataset):
        image = sample['image'][None].to(device); gt = sample['mask'][0].numpy() > .5
        h, w = gt.shape
        canvases = {key: torch.zeros((1, 1, h, w), device=device) for key in ('p0', *MODES)}
        divisor = torch.zeros_like(canvases['p0']); weight = blend_weight(512, device)
        q_canvas = torch.zeros_like(divisor) if region_context is not None else None
        records = []
        for y in sliding_positions(h, 512, 256):
            for x in sliding_positions(w, 512, 256):
                tile = image[:, :, y:y+512, x:x+512]
                frozen = models['A_original'].frozen(tile, capture_diagnostics=True)
                outputs = {name: (model(None, frozen_override=frozen, capture_diagnostics=True)
                                  if region_context is not None else model(None, frozen_override=frozen))
                           for name, model in models.items()}
                region = (slice(None), slice(None), slice(y, y+512), slice(x, x+512))
                canvases['p0'][region] += outputs['A_original']['z0'].float()*weight
                for name in MODES:
                    if not torch.equal(outputs[name]['z0'], outputs['A_original']['z0']):
                        raise RuntimeError('Comparison models do not share the same P0')
                    canvases[name][region] += outputs[name]['z1'].float()*weight
                if region_context is not None:
                    q = F.interpolate(outputs['A_original']['q'], size=(512, 512),
                                      mode='bilinear', align_corners=False)
                    q_canvas[region] += q.float()*weight
                    maps = {}
                    for name in MODES:
                        result = outputs[name]
                        for k in range(3):
                            for field, value in (('G', result['g_step'][:, k]),
                                    ('V', result['source_available'][:, k]),
                                    ('candidate', result['m'][:, k]-result[f'u{k}']),
                                    ('transport', result['transport_delta'][:, k]),
                                    ('proposal', result['proposal_delta'][:, k])):
                                maps[(name, k+1, field)] = value[0, 0].cpu().numpy()
                    records.append((y, x, maps))
                divisor[region] += weight
        probabilities = {}
        for key, canvas in canvases.items():
            z = (canvas/divisor).cpu().numpy()[0, 0]
            if not np.isfinite(z).all():
                raise RuntimeError('Nonfinite comparison logits')
            probabilities[key] = 1/(1+np.exp(-np.clip(z, -80, 80)))
        cases.append({'case': sample['case_name'], 'gt': gt, **probabilities})
        if region_context is not None:
            import cv2
            from evaluation.road_comparison.road_metrics import skeletonize
            ref = probabilities['p0'] >= region_context['reference_threshold']
            original = probabilities['A_original'] >= region_context['original_threshold']
            q = (q_canvas/divisor).cpu().numpy()[0, 0]
            isolated, bridges = v9.background_components(ref, original, gt)
            missing_center = skeletonize(gt) & ~(cv2.dilate(ref.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
            masks = {'missing_gt': gt & ~ref, 'missing_center': missing_center,
                'hard_background': ~gt & (q >= .5), 'original_isolated_fp': isolated,
                'original_false_bridge': bridges}
            owned = np.zeros_like(gt)
            for y, x, maps in records:
                take = ~owned[y:y+512, x:x+512]; owned[y:y+512, x:x+512] = True
                local = {name: value[y:y+512, x:x+512] & take for name, value in masks.items()}
                for (name, step, field), value in maps.items():
                    projected = cv2.resize(value, (512, 512), interpolation=cv2.INTER_LINEAR)
                    for region, mask in local.items():
                        if not mask.any(): continue
                        vals = projected[mask].astype(np.float64)
                        if not np.isfinite(vals).all(): raise ValueError('Nonfinite regional dynamics')
                        acc = regional.setdefault((name, step, region, field), [0, 0., 0.])
                        acc[0] += vals.size; acc[1] += float(vals.sum()); acc[2] += float(np.abs(vals).sum())
            del records
        print(f'[PAIRED] {i+1}/{len(dataset)} {sample["case_name"]}', flush=True)
    if region_context is not None:
        write_csv(region_context['output_dir']/'same_position_gate_regions.csv', [
            {'mode': key[0], 'step': key[1], 'region': key[2], 'quantity': key[3],
             'pixels': value[0], 'mean': value[1]/value[0], 'mean_abs': value[2]/value[0]}
            for key, value in regional.items()])
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True); p.add_argument('--experiment_dir', type=Path, required=True)
    p.add_argument('--root_path', type=Path, required=True); p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--reference_set', type=Path, required=True,
                   help='reference_set.json produced by the successful fixed-weight counterfactual run')
    p.add_argument('--device', default='cuda'); p.add_argument('--matching_bins', type=int, default=10000)
    p.add_argument('--expected_reference_gaps', type=int, default=206)
    args = p.parse_args(); args.device = torch.device(args.device)
    if args.matching_bins < 1000:
        p.error('matching_bins must be at least 1000')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Choose an empty output directory')
    original_path = args.run_dir/'stage_D/best.pth'
    checkpoints = {'A_original': original_path, 'B_control': args.experiment_dir/'control/best.pth',
                   'C_gate': args.experiment_dir/'gate/best.pth'}
    if not all(path.is_file() for path in checkpoints.values()):
        p.error('Need original, control and gate best.pth files')
    payload = torch.load(original_path, map_location='cpu', weights_only=False)
    if Path(payload['training_config']['root_path']).resolve() != args.root_path.resolve():
        p.error('Data root differs from V9 training')
    selections = {'A_original': json.loads((args.run_dir/'direct_val_selection.json').read_text())}
    selected = selections['A_original']
    if (selected['identity'] != payload['identity'] or Path(selected['checkpoint']).resolve() != original_path.resolve()
        or selected.get('selected_on') != 'val' or selected['report']['images'] != 14):
        p.error('Original val selection is inconsistent')
    ref = float(payload['training_config']['reference_threshold'])
    if float(selected['report']['p0_reference_threshold']) != ref:
        p.error('P0 reference differs')
    if args.expected_reference_gaps >= 0 and selected['report']['paired']['eligible_p0_gaps'] != args.expected_reference_gaps:
        p.error('Original reference gap count differs from the requested frozen collection')
    initial_sha = v9.digest(original_path)
    variants = {name: torch.load(checkpoints[name], map_location='cpu', weights_only=False)
                for name in ('B_control', 'C_gate')}
    confs = []
    for name, variant in variants.items():
        expected_variant = 'control' if name == 'B_control' else 'gate'
        if variant.get('v91_initial_checkpoint_sha256') != initial_sha or variant.get('v91_config', {}).get('variant') != expected_variant:
            p.error(f'{name} was not trained from the same original V9 checkpoint')
        confs.append({k: v for k, v in variant['v91_config'].items() if k != 'variant'})
    if confs[0] != confs[1]:
        p.error('Control and gate training settings differ')
    model, _, identity = v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(payload['identity']['q3']['path']), device=args.device))
    v9.restore_checkpoint(model, original_path, identity, 'D'); model.eval()
    models = {'A_original': model,
        **{name: create_variant(model.frozen, checkpoints[name], identity, args.device)
           for name in ('B_control', 'C_gate')}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('B_control', 'C_gate'):
        selections[name] = selection_for(models[name], checkpoints[name], name, args, identity, ref)
    from run_q3_r3_mass import full_dataset
    dataset = full_dataset(args.root_path, 'val')
    if len(dataset) != 14:
        p.error('Expected the same 14 validation images')
    original_t = float(selected['threshold'])
    cases = infer(models, dataset, args.device, region_context={'original_threshold': original_t,
        'reference_threshold': ref, 'output_dir': args.output_dir})
    for name in MODES:
        t = float(selections[name]['threshold']); saved = selections[name]['report']['p1']
        totals = {'tp': sum(int(((c[name] >= t) & c['gt']).sum()) for c in cases),
                  'fp': sum(int(((c[name] >= t) & ~c['gt']).sum()) for c in cases),
                  'fn': sum(int(((c[name] < t) & c['gt']).sum()) for c in cases)}
        if any(totals[k] != saved[k] for k in totals):
            raise ValueError(f'{name} does not reproduce its own saved validation counts: {totals}')
    target_tp = sum(int(((c['A_original'] >= original_t) & c['gt']).sum()) for c in cases)
    target_area = sum(int((c['A_original'] >= original_t).sum()) for c in cases)
    from run_reliable_directional_mass import selection_score
    curves = {float(t): value for t, value in selected['threshold_curves'].items()}
    p0_t = max(curves, key=lambda t: selection_score(curves[t]['p0'], {'background_bridge_proxy': 0}, len(cases)))
    p0_tp = sum(int(((c['p0'] >= p0_t) & c['gt']).sum()) for c in cases)
    p0_area = sum(int((c['p0'] >= p0_t).sum()) for c in cases)
    points = {}
    for name in MODES:
        recall_t = original_t if name == 'A_original' else matched_threshold(cases, name, target_tp, 'tp', args.matching_bins)[0]
        area_t = original_t if name == 'A_original' else matched_threshold(cases, name, target_area, 'area', args.matching_bins)[0]
        points[name] = {'same_threshold': original_t, 'own_val_threshold': float(selections[name]['threshold']),
                        'matched_recall': recall_t, 'matched_foreground_area': area_t,
                        'matched_P0_recall': matched_threshold(cases, name, p0_tp, 'tp', args.matching_bins)[0],
                        'matched_P0_foreground_area': matched_threshold(cases, name, p0_area, 'area', args.matching_bins)[0]}
    reports = evaluate(cases, points, ref, None if args.expected_reference_gaps < 0 else args.expected_reference_gaps,
                       args.output_dir, modes=MODES, p0_threshold=p0_t)
    original_reference_path = args.reference_set
    if not original_reference_path.is_file():
        raise ValueError('Missing reference_set.json from the fixed-weight counterfactual')
    original_reference = json.loads(original_reference_path.read_text())
    current_reference = json.loads((args.output_dir/'reference_set.json').read_text())
    if current_reference != original_reference:
        raise ValueError('Fixed P0/GT/skeleton reference differs from the original counterfactual')
    p0_own = json.loads((args.output_dir/'p0_own_metrics.json').read_text())
    result = {'selected_on': 'val', 'p0_reference_threshold': ref, 'original_checkpoint_sha256': initial_sha,
        'reference_set': current_reference, 'p0_own_threshold': p0_t, 'p0_own_metrics': p0_own['metrics'],
        'p0_at_original_common_threshold': selected['report']['p0'], 'comparison': reports,
        'note': 'Original V9 and both continuations have the same frozen P0 and fixed gap collection. '
        'Matched thresholds use a 0.0001 grid; residuals, approximate APLS, closure and bridge proxies are reported. '
        'same_position_gate_regions.csv uses masks fixed from original V9/P0, not model-specific positive-negative matched candidates.'}
    write_json(args.output_dir/'comparison.json', result)
    lines = ['# V9 versus L_gate continuation', '',
        '| mode | point | threshold | recall difference vs original | IoU | clDice | bidirectional APLS | gaps closed/eligible | background bridges | added GT / background |',
        '|---|---|---:|---:|---:|---:|---:|---|---:|---|']
    for name in MODES:
        for point, row in reports[name].items():
            m, s = row['metrics'], row['paired']
            lines.append(f'| {name} | {point} | {row["threshold"]:.4f} | {row["recall_difference_vs_A"]:+.6f} | '
                f'{m["iou"]:.5f} | {m["cldice"]:.5f} | {m["apls_bidirectional_approx"]:.5f} | '
                f'{s["closed"]}/{s["eligible"]} | {s["background_bridge_proxy"]} | '
                f'{s["added_gt_pixels"]}/{s["added_background_pixels"]} |')
    lines += ['', 'Additional metrics, individual recall/area residuals and gap-length bins: comparison.json / counterfactual_metrics.csv. '
              'A=original V9, B=continued control, C=continued with L_gate.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'[DONE] {args.output_dir / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
