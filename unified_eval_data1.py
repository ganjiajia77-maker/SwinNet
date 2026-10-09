"""Export SegRoadv2 masks, then use the shared road_comparison tool for all scores."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data1_common import binary_prediction, load_case, load_model, predict_full, split_index


HERE = Path(__file__).resolve().parent
TOOL = HERE / 'evaluation' / 'road_comparison' / 'compare_predictions.py'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        path = HERE / 'SOURCE_REVISION.txt'
        return path.read_text().strip() if path.exists() else 'archive'


def threshold_directory(root, split, threshold):
    return root / (split + '_predictions') / ('thr_' + f'{threshold:.6f}'.replace('.', 'p')) / 'masks'


def export_masks(snapshot, root, split, thresholds, data_root, config):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for SegRoadv2 mask export')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = load_model(snapshot, 'cuda:0', config['weights'])
    directories = {value: threshold_directory(root, split, value) for value in thresholds}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    index = split_index(data_root, split)
    for number, paths in enumerate(index, 1):
        image, _ = load_case(paths, config['source_patch_size'])
        components = predict_full(model, image, 'cuda:0', config['tile_size'],
                                  config['overlap_stride'], config['tta'])
        for threshold, directory in directories.items():
            prediction = binary_prediction(components, threshold, config['prediction_mode'])
            Image.fromarray(prediction.astype(np.uint8) * 255).save(directory / (paths[0].stem + '.png'))
        print('[export {}] {}/{} {}'.format(split, number, len(index), paths[0].name), flush=True)
    del model
    torch.cuda.empty_cache()
    return [{'threshold': value, 'path': str(directories[value])} for value in thresholds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--root_path', type=Path, help='required on val; test reuses saved data root')
    parser.add_argument('--model_path', type=Path, help='required on val; test uses the frozen snapshot')
    parser.add_argument('--thresholds', default=None)
    parser.add_argument('--tile_size', type=int, default=None)
    parser.add_argument('--source_patch_size', type=int, default=None)
    parser.add_argument('--overlap_stride', type=int, default=None)
    parser.add_argument('--prediction_mode', choices=['source_fusion', 'surface'], default=None)
    parser.add_argument('--weights', choices=['ema', 'raw'], default=None)
    parser.add_argument('--no_tta', action='store_true', default=None)
    parser.add_argument('--select_threshold_only', action='store_true',
                        help='on val, select the threshold without calculating validation topology')
    parser.add_argument('--metrics_only', action='store_true',
                        help='reuse this run\'s already exported masks and write metrics in a fresh directory')
    parser.add_argument('--metrics_output_dir', type=Path,
                        help='optional new metric output dir, useful after interrupted topology calculation')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    manifest_path, config_path = root / 'comparison.json', root / 'evaluation_config.json'
    snapshot = root / 'evaluation_checkpoint.pth'
    metric_output = (args.metrics_output_dir or root / args.split).resolve()
    if metric_output.exists() and any(metric_output.iterdir()):
        parser.error('Metric output is not empty; choose a new --metrics_output_dir')
    if args.select_threshold_only and args.split != 'val':
        parser.error('--select_threshold_only is valid only on val')
    if args.split == 'val' and not args.metrics_only:
        if not args.root_path or not args.model_path or not args.model_path.is_file():
            parser.error('Validation requires --root_path and an existing --model_path')
        if root.exists() and any(root.iterdir()):
            parser.error('Evaluation run directory is not empty; choose a new --output_dir')
        thresholds = [float(value) for value in (args.thresholds or
                      '0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95').split(',')]
        if not thresholds or len(set(thresholds)) != len(thresholds) or not all(
                np.isfinite(value) and 0 <= value <= 1 for value in thresholds):
            parser.error('Thresholds must be finite, unique values in [0,1]')
        if len(set(f'{value:.6f}' for value in thresholds)) != len(thresholds):
            parser.error('Thresholds are too close for six-decimal directory names')
        config = {'data_root': str(args.root_path.resolve()), 'source_patch_size': args.source_patch_size or 1024,
                  'tile_size': args.tile_size or 512, 'overlap_stride': args.overlap_stride or 256,
                  'prediction_mode': args.prediction_mode or 'source_fusion',
                  'weights': args.weights or 'ema', 'tta': not args.no_tta, 'thresholds': thresholds}
        if config['source_patch_size'] != 1024 or config['tile_size'] != 512:
            parser.error('This comparison requires native 1024 masks and 512 tiles')
        if not 0 < config['overlap_stride'] <= config['tile_size']:
            parser.error('Invalid overlap_stride')
        split_index(config['data_root'], 'val')
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.model_path, snapshot)
        checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
        training = checkpoint['config']
        if training.get('source_patch_size') != 1024 or training.get('img_size') != 512:
            raise ValueError('Checkpoint is not a native1024/random512 data1 run')
        if config['weights'] == 'ema' and checkpoint.get('ema_state_dict') is None:
            raise ValueError('Checkpoint has no EMA weights')
        config['model_name'] = 'segroadv2_{}_random512'.format(training['phi'])
        config['checkpoint_sha256'] = sha256(snapshot)
        provenance = {
            'checkpoint': str(snapshot), 'original_checkpoint': str(args.model_path.resolve()),
            'checkpoint_sha256': config['checkpoint_sha256'], 'checkpoint_epoch': checkpoint['epoch'],
            'training_code_commit': checkpoint.get('training_code_commit'),
            'upstream_commit': checkpoint.get('upstream_commit'),
            'evaluation_code_commit': revision(), 'training_config': training,
            'weights': config['weights'], 'inference_tile_size': config['tile_size'],
            'overlap_stride': config['overlap_stride'], 'tta': '4 flips' if config['tta'] else 'none',
            'merge': 'uniform average of surface probabilities and raw connectivity-logit sums',
            'postprocessing': config['prediction_mode'],
            'source_fusion_thresholds': {'near_logit_sum': 3.0, 'far_logit_sum': 1.5},
        }
        del checkpoint
        manifest = {'data_root': config['data_root'],
                    'protocol': {'image_size': 1024, 'short_area_threshold': 20,
                                 'apls_max_nodes': 64, 'apls_snap_radius': 5.0},
                    'models': [{'name': config['model_name'], 'provenance': provenance,
                                'predictions': {'val': [], 'test': []}}]}
        write_json(config_path, config)
        manifest['models'][0]['predictions']['val'] = export_masks(
            snapshot, root, 'val', thresholds, config['data_root'], config)
        write_json(manifest_path, manifest)
    else:
        if not config_path.is_file() or not manifest_path.is_file():
            parser.error('Run validation export first; missing evaluation config/manifest')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if sha256(snapshot) != config['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed; validation and test must share weights')
        supplied = {'tile_size': args.tile_size, 'source_patch_size': args.source_patch_size,
                    'overlap_stride': args.overlap_stride, 'prediction_mode': args.prediction_mode,
                    'weights': args.weights}
        if args.no_tta is not None:
            supplied['tta'] = not args.no_tta
        if args.root_path:
            supplied['data_root'] = str(args.root_path.resolve())
        for key, value in supplied.items():
            if value is not None and value != config[key]:
                raise ValueError('Inference configuration differs from validation: ' + key)
        if args.model_path or args.thresholds:
            parser.error('This mode uses the frozen checkpoint and saved/val-selected thresholds')
        if args.split == 'test' and not args.metrics_only:
            selection = json.loads((root / 'val' / 'val_selection.json').read_text(encoding='utf-8'))
            threshold = float(selection['models'][config['model_name']]['threshold'])
            print('Using VAL-selected threshold={}'.format(threshold), flush=True)
            prediction_root = root / 'test_predictions'
            if prediction_root.exists() and any(prediction_root.iterdir()):
                parser.error('Test masks already exist; use --metrics_only to recompute metrics')
            manifest['models'][0]['predictions']['test'] = export_masks(
                snapshot, root, 'test', [threshold], config['data_root'], config)
            write_json(manifest_path, manifest)
    command = [sys.executable, '-u', str(TOOL), '--manifest', str(manifest_path),
               '--split', args.split, '--output_dir', str(metric_output), '--progress_every', '10']
    if args.split == 'test':
        command += ['--selection', str(root / 'val' / 'val_selection.json')]
    elif args.select_threshold_only:
        command += ['--select_threshold_only']
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
