"""Score already-exported diagnostic val masks without selecting thresholds.

The pinned six-metric CLI's val entry writes selection CSV/JSON before its
empty-output guard. Call its unchanged evaluate() with the already fixed
diagnostic threshold instead. All graph calculations remain external.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys


def run(diagnostic_dir, tool, workers=4, module=None, models=None):
    root = Path(diagnostic_dir).resolve()
    tool = Path(tool).resolve()
    if workers < 1:
        raise ValueError('workers must be positive')
    if module is None:
        if not tool.is_file():
            raise FileNotFoundError(tool)
        sys.path.insert(0, str(tool.parent))
        module = importlib.import_module(tool.stem)
        if Path(module.__file__).resolve() != tool:
            raise ValueError('Imported a different topology tool')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    evidence = json.loads((root/'protocol.json').read_text(encoding='utf-8-sig'))
    complete = json.loads((root/'inference_complete.json').read_text(encoding='utf-8-sig'))
    if not complete.get('complete') or complete.get('images') != 14:
        raise ValueError('Expected completed 14-image diagnostic inference')
    if evidence.get('split') != 'val' or complete['checkpoint_sha256'] != evidence['checkpoint_sha256']:
        raise ValueError('Inference provenance mismatch')
    protocol = module.validate_manifest(manifest)
    threshold = float(evidence['threshold'])
    selection = dict(schema=2, selected_on='val', criterion='fixed_shared_diagnostic_threshold',
        protocol_hash=module.digest(protocol), threshold_protocol_hash=module.threshold_protocol_hash(protocol),
        note='Previously fixed threshold for all modes; no threshold selection or sweep here.', models={})
    expected_modes = {'normal', 'c_prior_one', 'residual_off'}
    actual_modes = {model['name'] for model in manifest['models']}
    if actual_modes not in (expected_modes, expected_modes | {'bias_only'}):
        raise ValueError('Expected the three frozen-checkpoint modes with optional bias_only')
    if models is not None:
        if not models or not set(models) <= actual_modes:
            raise ValueError('Requested topology modes are not present in this diagnostic')
        manifest = dict(manifest, models=[item for item in manifest['models'] if item['name'] in models])
    for model in manifest['models']:
        sources = model['predictions'].get('val', [])
        if len(sources) != 1 or module.threshold_of(sources[0]) != threshold:
            raise ValueError('Every mode must use the identical fixed val threshold')
        if model['provenance']['sha256'] != evidence['checkpoint_sha256']:
            raise ValueError('Mask checkpoint differs from diagnostic provenance')
        selection['models'][model['name']] = dict(identity=module.model_identity(model), threshold=threshold)

    # Preserve the failed CLI's topology/ files. Never place selection files in
    # the evaluator's new output directory before it creates run_state.json.
    output = root/'topology_fixed_val'
    output.mkdir(exist_ok=True)
    module.write_json(root/'fixed_topology_selection.json', selection)
    (root/'topology_result_dir.txt').write_text(str(output)+'\n', encoding='utf-8')
    print(f'[FIXED VAL] threshold={threshold}; output={output}; no model inference', flush=True)
    result = module.evaluate(manifest, root, protocol, 'val', selection, output,
                             progress=1, workers=workers, resume=(output/'run_state.json').is_file())
    print('[TOPOLOGY DONE]', output, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic_dir', type=Path, required=True)
    parser.add_argument('--topo_tool', type=Path, default=Path(
        '/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--models', nargs='+', help='Only score these cached modes, e.g. bias_only')
    args = parser.parse_args()
    run(args.diagnostic_dir, args.topo_tool, args.workers, models=args.models)
