"""Locate the recorded Mass Stage2-decoder or five DeepGlobe baseline predictions."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from compare_predictions import (digest, label_index, model_identity, prediction_index,
                                 read_binary, threshold_protocol_hash, validate_manifest)
from road_metrics import confusion


EXPERIMENTS = {
    'mass': {
        'h0_stage2_location_v1_decoder_mass_random512_fp32_80e_bs2':
            (0.45, (4116339, 1098161, 1064016)),
    },
    'data1': {
        'coanet_random512': (0.30, (43016607, 12253073, 12151309)),
        'oarenet_random512': (0.30, (43654277, 12434949, 11513639)),
        'segroadv2_b2_random512': (0.65, (43254100, 11073428, 11913816)),
        'weavingunet_random512': (0.35, (43183062, 12036177, 11984854)),
        'swin_unet_original_random512': (0.50, (41992202, 13752983, 13175714)),
    },
}
SKIP = {'miniconda3', '.conda', '.cache', '.git',
        'predictions', 'pretrained_ckpt', 'node_modules', '__pycache__'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def discover(search_root, data_root, size, experiments, labels):
    result = {name: {} for name in experiments}
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in SKIP
                   and (Path(root) / d).resolve() != data_root
                   and not (Path(root).name == 'Swin-Unet-main'
                            and d in {'data1', 'Massachusetts'})]
        for filename in files:
            if not (filename.startswith('comparison') and filename.endswith('.json')):
                continue
            source = Path(root) / filename
            try:
                manifest = load(source)
                if resolve(manifest['data_root'], source.parent) != data_root:
                    continue
                if manifest.get('protocol', {}).get('image_size') != size:
                    continue
                selections = [source.parent / 'val' / 'val_selection.json',
                              source.parent / 'val_selection.json']
                for sel_path in selections:
                    if not sel_path.is_file():
                        continue
                    selection = load(sel_path)
                    if selection.get('selected_on') != 'val':
                        continue
                    for original in manifest['models']:
                        name = original.get('name')
                        if name not in experiments or name not in selection.get('models', {}):
                            continue
                        chosen = selection['models'][name]
                        if chosen.get('identity') != model_identity(original):
                            continue
                        threshold = float(chosen['threshold'])
                        if abs(threshold - experiments[name][0]) > 1e-12:
                            continue
                        for pred in original.get('predictions', {}).get('test', []):
                            if abs(float(pred['threshold']) - threshold) > 1e-12:
                                continue
                            try:
                                index = prediction_index(pred, source.parent, labels)
                            except (OSError, ValueError):
                                continue
                            path = resolve(pred['path'], source.parent)
                            key = (str(path), chosen['identity'])
                            result[name][key] = (source, sel_path, original, chosen, path, index)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return result


def audit_candidates(candidates, labels, size, expected):
    totals = {key: np.zeros(3, dtype=np.int64) for key in candidates}
    hashes = {key: hashlib.sha256() for key in candidates}
    for number, (case, gt_path) in enumerate(sorted(labels.items()), 1):
        gt = read_binary(gt_path, size, allow_replicated_rgb=True)
        for key, item in candidates.items():
            pred = read_binary(item[5][case], size)
            counts = confusion(pred, gt)
            totals[key] += [counts['tp'], counts['fp'], counts['fn']]
            hashes[key].update(case.encode() + b'\0' + np.packbits(pred).tobytes())
        if number == 1 or number % 100 == 0 or number == len(labels):
            print(f'  mask audit {number}/{len(labels)}', flush=True)
    return [(candidates[key], hashes[key].hexdigest()) for key in candidates
            if tuple(map(int, totals[key])) == expected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mass', 'data1'], required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--search_root', type=Path, default=Path('/home/gjj'))
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', help='Optional exact model names to evaluate')
    parser.add_argument('--weaving_masks', type=Path,
                        help='Existing WeavingUnet binary surface directory')
    parser.add_argument('--weaving_selection', type=Path,
                        help='Original WeavingUnet validation selection JSON')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    size, nimages = (1500, 49) if args.dataset == 'mass' else (1024, 1246)
    labels = label_index(data_root, 'test')
    if len(labels) != nimages:
        raise ValueError(f'{args.dataset}: expected {nimages} test images, found {len(labels)}')
    experiments = EXPERIMENTS[args.dataset]
    if args.models:
        unknown = set(args.models) - set(experiments)
        if unknown or len(args.models) != len(set(args.models)):
            parser.error(f'Unknown or repeated model names: {sorted(unknown)}')
        experiments = {name: experiments[name] for name in args.models}
    if bool(args.weaving_masks) != bool(args.weaving_selection):
        parser.error('Specify both --weaving_masks and --weaving_selection')
    found = discover(args.search_root, data_root, size, experiments, labels)
    weaving = 'weavingunet_random512'
    if args.weaving_masks and weaving in experiments and not found[weaving]:
        original_selection = load(args.weaving_selection)
        if original_selection.get('selected_on') != 'val':
            raise ValueError('WeavingUnet selection is not from validation')
        original_chosen = original_selection.get('models', {}).get(weaving)
        if not isinstance(original_chosen, dict) or abs(float(original_chosen['threshold']) - 0.35) > 1e-12:
            raise ValueError('WeavingUnet validation threshold is not the recorded 0.35')
        selection_sha = hashlib.sha256(args.weaving_selection.read_bytes()).hexdigest()
        masks = args.weaving_masks.resolve()
        model = {'name': weaving,
                 'provenance': {'checkpoint': None,
                                'checkpoint_status': 'not verified from saved prediction',
                                'validation_selection': str(args.weaving_selection.resolve()),
                                'validation_selection_sha256': selection_sha,
                                'original_selection_identity': original_chosen.get('identity'),
                                'prediction_origin': 'user supplied saved binary masks'},
                 'predictions': {'val': [], 'test': [{'threshold': 0.35, 'path': str(masks)}]}}
        # Record the original identity and derive a new one for this explicitly
        # reconstructed, checkpoint-unverified manifest.
        chosen = copy.deepcopy(original_chosen)
        chosen['original_identity'] = original_chosen.get('identity')
        chosen['identity'] = model_identity(model)
        index = prediction_index(model['predictions']['test'][0], masks, labels)
        key = (str(masks), chosen['identity'])
        found[weaving][key] = (None, args.weaving_selection.resolve(), model, chosen, masks, index)
    missing = [name for name, candidates in found.items() if not candidates]
    if missing:
        raise ValueError('No validated manifest, validation selection and test masks for: ' + ', '.join(missing))
    models, selections, source_records = [], {}, {}
    for name, (_, expected) in experiments.items():
        candidates = found[name]
        if not candidates:
            raise ValueError(f'No manifest, validated val selection and test mask directory found for {name}')
        print(f'{name}: checking {len(candidates)} candidate(s)', flush=True)
        matched = audit_candidates(candidates, labels, size, expected)
        if not matched:
            raise ValueError(f'No predictions match the previously recorded TP/FP/FN for {name}')
        signatures = {(item[2]['provenance'].get('checkpoint'), pred_hash)
                      for item, pred_hash in matched}
        if any(not checkpoint for checkpoint, _ in signatures) and not (
            name == weaving and all(item[0] is None for item, _ in matched)
        ):
            raise ValueError(f'Matched {name} has no checkpoint in its provenance')
        if len(signatures) != 1:
            raise ValueError(f'Multiple checkpoints or different masks match {name}; inspect: '
                             + ', '.join(str(item[0]) for item, _ in matched))
        (source, sel_path, original, chosen, path, _), pred_hash = matched[0]
        model = copy.deepcopy(original)
        model['predictions'] = {'val': [], 'test': [{'threshold': float(chosen['threshold']), 'path': str(path)}]}
        models.append(model)
        selections[name] = chosen
        source_records[name] = {'manifest': str(source) if source else None, 'validation_selection': str(sel_path),
                                'prediction_sha256': pred_hash}
        print(f'  threshold={chosen["threshold"]} checkpoint={original["provenance"].get("checkpoint")}\n'
              f'  masks={path}\n  manifest={source}', flush=True)
    manifest = {'data_root': str(data_root), 'protocol': {
        'image_size': size, 'short_area_threshold': 20, 'apls_max_control_nodes': 150,
        'apls_snap_radius': 5.0, 'topo_seed_density_deg': 0.002, 'topo_step_deg': 0.0002,
    }, 'models': models}
    selection = {'schema': 2, 'selected_on': 'val', 'criterion': 'reused_saved_validation_selection',
                 'threshold_protocol_hash': threshold_protocol_hash(validate_manifest(manifest)),
                 'models': selections, 'source_records': source_records}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, data in (('comparison.json', manifest), ('val_selection.json', selection)):
        (args.output_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Created verified {args.dataset} input: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
