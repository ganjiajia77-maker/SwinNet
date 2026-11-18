"""Reuse original H0/Data1 test masks and their validation-selected threshold."""
import argparse
import copy
import json
from pathlib import Path

from compare_predictions import (label_index, model_identity, prediction_index,
                                 resolve_path, validate_manifest)

NAME = 'h0_599a410_random512_fp32_ema80'
PROTOCOL = dict(image_size=1024, short_area_threshold=20,
    topo_seed_density_deg=.002, topo_step_deg=.0002,
    apls_max_control_nodes=150, apls_control_sampling_seed=1234,
    apls_snap_radius=5., apls_min_path_px=10., tlts_n_paths=200, tlts_seed=1234)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def matched_model(manifest, source, selection, run_dir, data_root):
    if resolve_path(manifest['data_root'], source.parent).resolve() != data_root:
        raise ValueError('Manifest belongs to another dataset')
    if manifest.get('protocol', {}).get('image_size') != 1024:
        raise ValueError('Expected native1024 original H0/Data1 masks')
    if selection.get('selected_on') != 'val' or selection.get('criterion') not in ('global_iou', 'global_f1'):
        raise ValueError('Threshold was not selected by validation IoU/F1')
    original = next(m for m in manifest['models'] if m.get('name') == NAME)
    checkpoint = resolve_path(original['provenance']['checkpoint'], source.parent).resolve()
    if checkpoint != run_dir/'best.pth':
        raise ValueError('Manifest belongs to another checkpoint')
    chosen = selection['models'][NAME]
    if chosen.get('identity') != model_identity(original):
        raise ValueError('Validation selection does not match the original model identity')
    threshold = float(chosen['threshold'])
    matches = [p for p in original['predictions'].get('test', [])
               if abs(float(p['threshold'])-threshold) < 1e-12]
    if len(matches) != 1:
        raise ValueError('Expected exactly one test export at the validation threshold')
    prediction = resolve_path(matches[0]['path'], source.parent).resolve()
    if not prediction.is_dir():
        raise FileNotFoundError(f'Test masks no longer exist: {prediction}')
    model = copy.deepcopy(original)
    model['predictions'] = dict(test=[dict(threshold=threshold, path=str(prediction))])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir, data_root = args.run_dir.resolve(), args.data_root.resolve()
    if not (run_dir/'best.pth').is_file():
        parser.error(f'Original H0 checkpoint is missing: {run_dir / "best.pth"}')
    labels = label_index(data_root, 'test')
    if len(labels) != 1246:
        parser.error(f'Expected 1246 DeepGlobe test images, found {len(labels)}')
    candidates, errors = [], []
    for source in sorted(run_dir.rglob('comparison*.json')):
        try:
            manifest = load(source)
        except (OSError, ValueError):
            continue
        if not any(m.get('name') == NAME for m in manifest.get('models', [])):
            continue
        for selection_path in (source.parent/'val/val_selection.json', source.parent/'val_selection.json'):
            if not selection_path.is_file():
                continue
            try:
                selection = load(selection_path)
                model = matched_model(manifest, source, selection, run_dir, data_root)
                prediction_index(model['predictions']['test'][0], source.parent, labels)
                candidates.append((source, selection_path, model, selection['criterion']))
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                errors.append(f'{source}: {error}')
    if not candidates:
        parser.error('No complete original-H0 validation selection and matching test masks found.\n'
            + '\n'.join(errors[-8:])
            + '\nInspect existing exports with: find "'+str(run_dir)+'" -type f -name comparison.json -print')
    # A later validation audit may deliberately choose another threshold.
    # Select its recorded result, never search test scores for a better one.
    source, selection, model, criterion = max(candidates, key=lambda c: c[1].stat().st_mtime_ns)
    threshold = model['predictions']['test'][0]['threshold']
    model['validation_selection'] = dict(selected_on='val', criterion=criterion,
        threshold=threshold, evidence=str(selection))
    result = dict(data_root=str(data_root), protocol=PROTOCOL, models=[model])
    validate_manifest(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir/'comparison.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Checkpoint: {run_dir / "best.pth"}\nOriginal manifest: {source}\nValidation selection: {selection}'
          f'\nThreshold: {threshold}\nTest masks: {model["predictions"]["test"][0]["path"]}'
          f'\nTest images: 1246\nDiagnosis manifest: {path}', flush=True)


if __name__ == '__main__':
    main()
