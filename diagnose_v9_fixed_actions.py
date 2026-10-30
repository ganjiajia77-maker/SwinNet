"""Compare B/C gate decisions at equal helpful acceptance on identical val actions.

Read-only experiment: no optimizer, model update, new loss or GT state update.
Run from the repository root; upstream dependencies are resolved from original V9.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_reliable_v9_mass as v9
from run_v9_gate_benefit import atomic_json, load_original
from gate_benefit_protocol import create_variant
from networks.v9_transport_benefit import BenefitConfig, benefit_labels, match_labels
from networks.v9_fixed_action_audit import capture_bank, replay_gate, compare_scores, bootstrap_difference, acceptance_points
from mass_protocol import sliding_positions
from diagnostics.v91_counterfactual import write_csv

VERSION = 'v9_fixed_action_equal_helpful_v1'
SCORERS = ('B_control', 'C_gate3')
MODES = ('full_gate', 'residual_only')


def verify_continuations(experiment, source):
    """No extra training. Check existing two continuations and full data hashes."""
    paths = {'B_control': experiment/'control_3step/best.pth', 'C_gate3': experiment/'gate_3step/best.pth'}
    configs = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing completed continuation: {path}. This command does not start training.')
        state = torch.load(path, map_location='cpu', weights_only=False)
        config = state.get('benefit_config', {})
        if config.get('variant') != ('control' if name == 'B_control' else 'gate') or config.get('steps') != 3:
            raise ValueError(f'Wrong continuation/step identity: {path}')
        if config.get('original_sha256') != v9.digest(source):
            raise ValueError(f'Continuation does not start from this original V9: {path}')
        configs[name] = config
    left, right = ({k: v for k, v in configs[name].items() if k != 'variant'} for name in SCORERS)
    if left != right: raise ValueError('B/C continuation configurations differ beyond the gate-loss variant')
    schedules = []
    for path in paths.values():
        marker = path.parent/'training_complete.json'
        if not marker.is_file() or json.loads(marker.read_text())['epochs'] != left['epochs']:
            raise ValueError(f'Continuation is not complete: {path.parent}')
        records = {row['epoch']: row for line in (path.parent/'epochs.jsonl').read_text().splitlines()
                   if line.strip() for row in [json.loads(line)]}
        if set(records) != set(range(1, left['epochs']+1)): raise ValueError('Incomplete paired epoch records')
        schedules.append(records)
    for epoch in schedules[0]:
        for key in ('case_order_sha256', 'augmented_images_sha256'):
            if schedules[0][epoch][key] != schedules[1][epoch][key]:
                raise ValueError(f'B/C data schedule differs: epoch={epoch} {key}')
    return paths, left


def input_digest(sample):
    digest = hashlib.sha256(str(sample['case_name']).encode())
    for key in ('image', 'mask'):
        array = sample[key].detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def gather_image(sample, models, bank_names, benefit, image_index, device, seed):
    """Each bank defines ONE immutable state/action tape scored by both models."""
    height, width = sample['mask'].shape[-2:]
    image = sample['image'][None].to(device)
    pieces = {name: [] for name in bank_names}
    owned = np.zeros((height, width), dtype=bool)
    for y in sliding_positions(height, 512, 256):
        for x in sliding_positions(width, 512, 256):
            frozen = next(iter(models.values())).frozen(image[:, :, y:y+512, x:x+512], capture_diagnostics=True)
            own = ~owned[y:y+512, x:x+512]; owned[y:y+512, x:x+512] = True
            target = sample['mask'][None, :, y:y+512, x:x+512].to(device)
            for name in bank_names:
                bank = capture_bank(models[name], frozen_override=frozen)
                result = bank['result']; h, w = result['r'].shape[-2:]
                yy = np.minimum(((np.arange(h)+.5)*512/h).astype(int), 511)
                xx = np.minimum(((np.arange(w)+.5)*512/w).astype(int), 511)
                ownership = torch.as_tensor(own[yy[:,None], xx[None,:]], device=device)[None,None]
                labels = benefit_labels(result, target, benefit, ownership)
                selected, _ = match_labels(labels, benefit, seed+image_index*100000+y*1000+x)
                take = labels['valid'].cpu().numpy().astype(bool)
                indexes = np.nonzero(take); k, wy, wx = indexes[1], indexes[3], indexes[4]
                row = {'image_id': np.full(len(k), image_index, dtype=np.int16), 'step': (k+1).astype(np.int8),
                       'native_y': (y+yy[wy]).astype(np.int16), 'native_x': (x+xx[wx]).astype(np.int16),
                       'tile_y': np.full(len(k), y, dtype=np.int16), 'tile_x': np.full(len(k), x, dtype=np.int16),
                       'work_y': wy.astype(np.int16), 'work_x': wx.astype(np.int16),
                       'label': labels['target'].cpu().numpy()[take].astype(np.uint8),
                       'matched': selected.cpu().numpy()[take]}
                for field, tensor in (('action', labels['action']), ('V', labels['source']), ('gain', labels['gain']),
                                      ('off', labels['off']), ('on', labels['on']),
                                      ('U', torch.stack(result['states'][:-1], dim=1)), ('M', result['m']),
                                      ('rate', result['transport_rate'])):
                    row[field] = tensor.detach().cpu().numpy()[take].astype(np.float32)
                for mode in MODES:
                    for scorer in SCORERS:
                        logits = replay_gate(models[scorer], bank, mode)
                        if scorer == name:
                            torch.testing.assert_close(logits, result['gate_logits'], atol=2e-6, rtol=1e-6)
                        row[f'{scorer}_{mode}'] = logits.cpu().numpy()[take].astype(np.float32)
                # These exact action/label rows are shared, rather than independently sampled per scorer.
                pieces[name].append(row)
            print(f'[FIXED ACTION TILE] image={image_index+1} y={y} x={x}', flush=True)
    return {name: {field: np.concatenate([row[field] for row in rows]) for field in rows[0]}
            for name, rows in pieces.items()}


def save_tape(path, rows, provenance):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle: np.savez_compressed(handle, **rows)
    temp.replace(path)
    action_hash = hashlib.sha256()
    for field in sorted(rows):
        if not field.startswith(('B_control_', 'C_gate3_')):
            action_hash.update(field.encode()); action_hash.update(rows[field].tobytes())
    atomic_json(path.with_suffix('.json'), {**provenance, 'action_rows': len(rows['label']),
                'action_sha256': action_hash.hexdigest(), 'npz_sha256': v9.digest(path)})


def load_tapes(output, manifest):
    banks = {}
    for name in manifest['banks']:
        parts = []
        for image_index, case in enumerate(manifest['cases']):
            path = output/'actions'/name/f'{image_index:02d}.npz'
            meta_path = path.with_suffix('.json')
            if not path.is_file() or not meta_path.is_file(): raise FileNotFoundError(f'Missing action tape: {path}')
            meta = json.loads(meta_path.read_text())
            if meta['case'] != case or meta['npz_sha256'] != v9.digest(path):
                raise ValueError(f'Action tape content/identity differs: {path}')
            with np.load(path, allow_pickle=False) as data: parts.append({key: data[key] for key in data.files})
        banks[name] = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    return banks


def analyze(output, manifest, banks, targets, bootstrap, seed):
    report = {'version': VERSION, 'selected_on': 'val', 'weights_changed': False, 'banks': {},
              'scope': 'G decision discrimination on fixed actions, not final Surface or topology performance.',
              'full_gate': 'Both scorers read the bank raw H/H3/P0/R/Q and identical per-step state/M/V/rate/B inputs. Each uses its own learned visual projection and internal relation/W gate features, preserving its trained feature coordinate system; those never replace the action tape.',
              'residual_only': 'Also fix old gate logit and W/two/one relation inputs; only semantic/state gate weights differ.',
              'ownership': 'All val windows; first-window work-cell-center ownership. No row deduplication across steps: each step is a distinct action.',
              'limitations': ['Off-policy gate replay does not establish improved final road closure.',
                  'Each tape remains conditioned on its generating model; compare both B and C banks.',
                  'Benefit labels use the existing clear-GT area-fraction BCE proxy, not perfect topological benefit.',
                  'Acceptance thresholds are diagnostic only; validation is not independent test evidence.']}
    table, per_image, curves = [], [], []
    for bank, rows in banks.items():
        report['banks'][bank] = {}
        if len(rows['label']) == 0: raise ValueError(f'No valid fixed actions in {bank}; no successful result can be reported')
        for mode in MODES:
            stats = {}
            for population in ('natural', 'matched'):
                take = np.ones(len(rows['label']), bool) if population == 'natural' else rows['matched'].astype(bool)
                for subgroup, mask in [('all', take), *[(f'step{k}', take & (rows['step']==k)) for k in (1,2,3)],
                                       ('positive_action', take & (rows['action']>0)), ('negative_action', take & (rows['action']<0))]:
                    labels = rows['label'][mask].astype(bool)
                    b, c = (rows[f'{name}_{mode}'][mask] for name in SCORERS)
                    result = compare_scores(labels, b, c, targets)
                    if population == 'natural' and subgroup == 'all':
                        confidence_targets = [.8, .9]
                        if result['raw_at_G_0_5']['B_control']['helpful'] is not None:
                            confidence_targets.append(result['raw_at_G_0_5']['B_control']['helpful'])
                        result['paired_image_bootstrap'] = bootstrap_difference(labels, b, c, rows['image_id'][mask],
                            sorted(set(confidence_targets)), bootstrap, seed)
                    # Keep detailed per-step/sign threshold bounds in CSV, not another huge JSON dump.
                    if subgroup == 'all':
                        compact = {key:value for key,value in result.items() if key != 'matched_helpful'}
                        compact['matched_helpful'] = [
                            {'helpful_target': row['helpful_target'],
                             'B_harmful': row['B'].get('harmful_acceptance'),
                             'C_harmful': row['C'].get('harmful_acceptance'),
                             'harmful_C_minus_B': row['harmful_C_minus_B'],
                             'matches_B_default_helpful_rate': row['matches_B_default_helpful_rate']}
                            for row in result['matched_helpful']]
                        stats[population+'_'+subgroup] = compact
                    for row in result['matched_helpful']:
                        left, right = row['B'], row['C']
                        if not left['available'] or not right['available']: continue
                        table.append({'bank': bank, 'mode': mode, 'population': population, 'group': subgroup,
                            'helpful_actions': result['helpful'], 'harmful_actions': result['harmful'],
                            'helpful_target': row['helpful_target'], 'B_helpful': left['helpful_acceptance'],
                            'C_helpful': right['helpful_acceptance'], 'B_harmful': left['harmful_acceptance'],
                            'C_harmful': right['harmful_acceptance'], 'harmful_C_minus_B': row['harmful_C_minus_B'],
                            'B_logit_threshold': left['threshold_logit'], 'C_logit_threshold': right['threshold_logit'],
                            'B_tie_fraction': left['tie_fraction'], 'C_tie_fraction': right['tie_fraction'],
                            'B_harmful_strict': left['harmful_strict'], 'B_harmful_inclusive': left['harmful_inclusive'],
                            'C_harmful_strict': right['harmful_strict'], 'C_harmful_inclusive': right['harmful_inclusive']})
                    if subgroup == 'all':
                        for scorer, score in (('B_control', b), ('C_gate3', c)):
                            for point in acceptance_points(score, labels, np.linspace(0., 1., 101)):
                                curves.append({'bank': bank, 'mode': mode, 'population': population,
                                               'scorer': scorer, **point})
            for image_index, case in enumerate(manifest['cases']):
                mask = rows['image_id']==image_index
                result = compare_scores(rows['label'][mask].astype(bool), rows[f'B_control_{mode}'][mask],
                                        rows[f'C_gate3_{mode}'][mask], (.8,.9))
                for row in result['matched_helpful']:
                    if row['B']['available'] and row['C']['available']:
                        per_image.append({'bank': bank, 'mode': mode, 'case': case,
                            'helpful_target': row['helpful_target'], 'helpful': result['helpful'], 'harmful': result['harmful'],
                            'harmful_C_minus_B': row['harmful_C_minus_B']})
            report['banks'][bank][mode] = stats
            print(f'[FIXED ACTION ANALYSIS] bank={bank} mode={mode}', flush=True)
    atomic_json(output/'fixed_actions.json', report)
    write_csv(output/'matched_helpful.csv', table); write_csv(output/'per_image.csv', per_image)
    write_csv(output/'acceptance_curves.csv', curves)
    lines = ['# Fixed-action G selectivity diagnosis', '',
        'Read-only val experiment. B and C judge identical actions within each bank. A negative C-minus-B harmful rate is better.', '',
        '| Bank | Gate scoring | Helpful acceptance | Harmful B | Harmful C | C minus B |',
        '|---|---|---:|---:|---:|---:|']
    for row in table:
        if row['population']=='natural' and row['group']=='all' and row['helpful_target'] in (.8,.9):
            lines.append(f'| {row["bank"]} | {row["mode"]} | {row["helpful_target"]:.0%} | '
                         f'{row["B_harmful"]:.3%} | {row["C_harmful"]:.3%} | {row["harmful_C_minus_B"]:+.3%} |')
    lines += ['', '## Evidence and interpretation', '',
        'A lower harmful acceptance at the same helpful acceptance supports better ranking on this fixed proxy-label population. '
        'A lower raw acceptance at G>=0.5 alone does not establish that.',
        'Check both B/C action banks, natural/matched populations, step/action-sign strata and paired-image bootstrap intervals. '
        'A uniform monotonic logit shift has identical matched curves, even if its raw gate is much smaller.',
        'Full-gate scoring includes each model\'s visual projection and learned relation evidence; residual-only isolates semantic/state corrections on common visual/old-G/W evidence. '
        'Transport targets, rates and states are never recomputed by the scoring models.',
        'Tied scores use the SAME fractional boundary acceptance for both labels. Rates are exact expectations; deterministic strict/inclusive bounds are saved. '
        'These thresholds do not modify the deployed soft gate.',
        'No final Surface, closure, bridge or test-set improvement is inferred from this experiment. '
        'The two score vectors share one action tape, label mask and matching selection.', '',
        'Details: fixed_actions.json; matched_helpful.csv; per_image.csv; acceptance_curves.csv; actions/<bank>/*.npz.']
    for bank in report['banks']:
        result = report['banks'][bank]['full_gate']['natural_all']
        lines += ['', f'### {bank}: {result["helpful"]} helpful / {result["harmful"]} harmful actions', '',
                  'Paired image bootstrap, C-minus-B harmful acceptance (95% percentile interval):']
        for row in result['paired_image_bootstrap']:
            low, high = row['delta_harmful_C_minus_B_ci95']
            conclusion = 'lower harmful acceptance supported' if high < 0 else 'no clear difference supported' if low <= 0 else 'higher harmful acceptance supported'
            lines.append(f'- helpful={row["helpful_target"]:.3%}: [{low:+.3%}, {high:+.3%}], {conclusion}.')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'[DONE] {output / "REPORT.md"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, default=Path('/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407'))
    parser.add_argument('--experiment_dir', type=Path, required=True)
    parser.add_argument('--root_path', type=Path, default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--banks', nargs='+', choices=SCORERS, default=list(SCORERS))
    parser.add_argument('--targets', nargs='+', type=float, default=[.5,.7,.8,.9,.95])
    parser.add_argument('--bootstrap', type=int, default=100); parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cached_only', action='store_true')
    args = parser.parse_args(); args.device = torch.device(args.device)
    if args.bootstrap < 0 or any(not 0 <= t <= 1 for t in args.targets): parser.error('Invalid bootstrap budget or helpful targets')
    args.output_dir.mkdir(parents=True, exist_ok=True); marker = args.output_dir/'capture_identity.json'
    if args.cached_only:
        if not marker.is_file(): raise FileNotFoundError('No existing complete fixed-action cache')
        manifest = json.loads(marker.read_text())
        if manifest['version'] != VERSION or len(manifest['cases']) != 14: raise ValueError('Wrong cache protocol')
        analyze(args.output_dir, manifest, load_tapes(args.output_dir, manifest), args.targets, args.bootstrap, args.seed)
        return
    source = args.run_dir/'stage_D/best.pth'
    paths, config = verify_continuations(args.experiment_dir, source)
    # Preserve the source settings; these arguments are for strict loading, never training.
    args.batch_size = config['batch_size']; args.seed = config['seed']
    model, _, identity, _, _, _ = load_original(args)
    models = {name: create_variant(model.frozen, path, identity, args.device) for name, path in paths.items()}
    del model
    from run_q3_r3_mass import full_dataset
    dataset = full_dataset(args.root_path, 'val')
    if len(dataset) != 14: raise ValueError('Expected the same 14 full validation images')
    benefit = BenefitConfig(**config['labels']).validate()
    manifest = {'version': VERSION, 'experiment_dir': str(args.experiment_dir.resolve()),
        'source_sha256': v9.digest(source), 'identity': identity, 'root_path': str(args.root_path.resolve()),
        'banks': list(dict.fromkeys(args.banks)), 'checkpoint_sha256': {name: v9.digest(path) for name, path in paths.items()},
        'cases': [str(dataset[index]['case_name']) for index in range(len(dataset))],
        'label_config': config['labels'], 'seed': args.seed,
        'capture_code': {name: v9.digest(Path(__file__).parent/name) for name in (
            'diagnose_v9_fixed_actions.py', 'networks/v9_fixed_action_audit.py')}}
    if marker.is_file() and json.loads(marker.read_text()) != manifest: raise ValueError('Output cache identity changed; use a new output directory')
    atomic_json(marker, manifest)
    for index in range(len(dataset)):
        sample = dataset[index]; digest = input_digest(sample)
        complete = True
        for bank in manifest['banks']:
            file = args.output_dir/'actions'/bank/f'{index:02d}.npz'; meta = file.with_suffix('.json')
            if not file.is_file() or not meta.is_file(): complete = False; continue
            value = json.loads(meta.read_text())
            if value['input_sha256'] != digest or value['case'] != str(sample['case_name']): raise ValueError('Cached input data changed')
            if value['npz_sha256'] != v9.digest(file): raise ValueError('Cached action tape is corrupt')
        if complete:
            print(f'[FIXED ACTION CACHE] {index+1}/14 {sample["case_name"]}', flush=True); continue
        rows = gather_image(sample, models, manifest['banks'], benefit, index, args.device, args.seed)
        for bank, values in rows.items():
            file = args.output_dir/'actions'/bank/f'{index:02d}.npz'; file.parent.mkdir(parents=True, exist_ok=True)
            save_tape(file, values, {'case': str(sample['case_name']), 'input_sha256': digest})
        print(f'[FIXED ACTION IMAGE] {index+1}/14 {sample["case_name"]}', flush=True)
    analyze(args.output_dir, manifest, load_tapes(args.output_dir, manifest), args.targets, args.bootstrap, args.seed)
    if any(v9.digest(path) != manifest['checkpoint_sha256'][name] for name, path in paths.items()):
        raise RuntimeError('Checkpoint changed while diagnostic was running')


if __name__ == '__main__': main()
