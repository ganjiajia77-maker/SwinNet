"""Fixed Massachusetts protocol for the original H0 Stage 2 location experiment."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from mass_protocol import BASE_COMMIT, validate_training_args


ROOT = Path(__file__).resolve().parent
CFG = ROOT / 'configs' / 'swin_tiny_patch4_window7_224_lite.yaml'
EVAL = ROOT / 'evaluation' / 'road_comparison' / 'compare_predictions.py'
ARCHITECTURE_ID = 'h0_stage2_location_v1'
LOCATIONS = ('decoder', 'encoder_e1')
MODEL_NAMES = {location: f'{ARCHITECTURE_ID}_{location}_mass_random512_fp32_80e_bs2'
               for location in LOCATIONS}
PROTOCOL = {'image_size': 1500, 'short_area_threshold': 20,
            'apls_max_nodes': 64, 'apls_snap_radius': 5.0}

# These are the effective original mass_h0.py flags, including parser defaults
# that must not change silently when the training entry is adapted.
REFERENCE = {
    'img_size': 512, 'source_patch_size': 1500, 'overlap_stride': 256,
    'random_crop_train': True, 'random_crops_per_image': 1, 'max_epochs': 80,
    'batch_size': 2, 'accumulation_steps': 1, 'val_interval': 5,
    'amp_dtype': 'none', 'use_ema': False, 'trend_val_every_steps': 0,
    'threshold': 0.5, 'base_lr': 5e-4, 'pretrained_lr': 5e-5,
    'pretrained_min_lr': 5e-6, 'new_lr': 2e-4, 'new_min_lr': 1e-5,
    'warmup_epochs': 10, 'stage2_skeleton_weight': 0.008,
    'stage3_skeleton_weight': 0.012, 'stage_skeleton_loss_factor': 1.0,
    'stage_skeleton_only_loss_factor': 1.0, 'stage2_skeleton_gradient_ratio': 0.5,
    'stage3_skeleton_gradient_ratio': 0.5, 'stage3_gate_topology_gradient_ratio': 0.0,
    'skeleton_pos_weight': 2.0, 'highres_structure_skeleton_weight': 0.0,
    'stage_direction_factor': 0.2, 'stage_connectivity_factor': 3.0,
    'road_attention_weight': 0.0, 'masked_connectivity_center_experiment': True,
    'connectivity_pos_weight': 2.0, 'connectivity_focal_gamma': 1.5,
    'edge_contrastive_margin': 0.1, 'directional_pos_weight_cardinal': 1.0,
    'directional_pos_weight_diagonal': 2.5, 'surface_focal_gamma': 1.0,
    'structure_profile': 'stage23_boundary_0626', 'stage_skeleton_mode': 'direct',
    'enable_highres_structure_stream': True, 'highres_structure_channels': 64,
    'enable_h3_surface_fusion': True, 'remove_stage2_pre_topology_source': True,
    'highres_structure_fuse_stages': 'stage23', 'highres_structure_fusion_mode': 'stage23',
    'enable_global_topology': True, 'global_topology_max_nodes': 32,
    'global_topology_heads': 4, 'global_topology_alpha_max': 0.05,
}
DEFAULTS = {
    'augment': True, 'deterministic': 1, 'direct_resize_train': False,
    'val_crop_list': '', 'bottleneck_type': 'global_local',
    'enable_post_refine_structure_interaction': False,
    'freeze_post_refine_interaction_only': False, 'final_skeleton_gradient_ratio': 0.0,
    'stage_skeleton_bias_init': 'zero', 'stage_skeleton_positive_prior': 0.05,
    'final_topology_eta_init': 0.0, 'final_gap_rho_init': 0.0,
    'surface_loss': 'bce_dice', 'no_pretrain': False, 'warm_start_ckpt': '',
    'tiny_overfit_samples': 0, 'max_train_batches': 0,
    'disable_centerline_loss': False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def source_digest(path):
    """Hash text sources consistently across Git CRLF and LF checkouts."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_implementation_sources(path, required=True):
    """Verify current code separately from immutable baseline provenance."""
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'Finalize current-source digests first: {path}')
        return None
    hashes = read_json(path)
    needed = set(read_json(ROOT / 'BASE_MODEL_SHA256.json')) | {
        'h0_location.py', 'mass_protocol.py', 'train_image.py', 'test_image.py',
        'datasets/dataset_road_skeleton.py', 'networks/h0_location_contract.py',
        'config.py', 'configs/swin_tiny_patch4_window7_224_lite.yaml',
        'evaluation/road_comparison/compare_predictions.py',
        'evaluation/road_comparison/road_metrics.py',
        'BASE_MODEL_SHA256.json', 'PROTOCOL_SOURCE_SHA256.json',
    }
    if not isinstance(hashes, dict) or needed - set(hashes):
        raise ValueError(f'Implementation manifest must cover all experiment sources: {sorted(needed)}')
    for name, expected in hashes.items():
        source = (ROOT / name).resolve()
        if (not source.is_relative_to(ROOT) or not source.is_file()
                or source_digest(source) != expected):
            raise ValueError(f'Implementation source digest mismatch: {name}')
    return hashes


def validate_reference_args(saved):
    validate_training_args(saved)
    expected = dict(REFERENCE, **DEFAULTS)
    mismatches = {name: (saved.get(name), value) for name, value in expected.items()
                  if saved.get(name) != value}
    for name, value in {'stage2_direction_factor': 0.2, 'stage3_direction_factor': 0.2,
                        'stage2_connectivity_factor': 3.0, 'stage3_connectivity_factor': 3.0,
                        'final_skeleton_weight': 0.0, 'final_connectivity_weight': 0.0,
                        'boundary_weight': 0.0}.items():
        if saved.get(name) is not None and saved[name] != value:
            mismatches[name] = (saved[name], value)
    if mismatches:
        raise ValueError(f'Reference training settings changed (actual, required): {mismatches}')


def verify_checkpoint(path, location):
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model_state_dict'), dict):
        raise ValueError('Expected an experiment checkpoint containing model_state_dict')
    contract = checkpoint.get('architecture_contract')
    expected = {'architecture_id': ARCHITECTURE_ID, 'stage2_structure_location': location,
                'base_model_commit': BASE_COMMIT}
    if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Checkpoint architecture_contract must match {expected}; got {contract}')
    saved = checkpoint.get('args')
    if not isinstance(saved, dict):
        raise ValueError('Checkpoint must record resolved training args')
    validate_reference_args(saved)
    if saved.get('stage2_structure_location') != location:
        raise ValueError('Checkpoint args and requested location disagree')
    if saved.get('architecture_id') != ARCHITECTURE_ID:
        raise ValueError('Checkpoint args and architecture ID disagree')
    if checkpoint.get('ema_state_dict') is not None:
        raise ValueError('Reference requires raw FP32 weights without EMA')
    if (not isinstance(checkpoint.get('epoch'), int) or isinstance(checkpoint['epoch'], bool)
            or not 1 <= checkpoint['epoch'] <= 80):
        raise ValueError('Checkpoint epoch must be in 1..80')
    result = {'epoch': checkpoint['epoch'], 'args': saved, 'architecture_contract': contract,
              'checkpoint_sha256': file_digest(path),
              'can_resume': isinstance(checkpoint.get('optimizer_state_dict'), dict)
                            and isinstance(checkpoint.get('loaded_pretrained_names'), (list, tuple))}
    return result


def train_command(args, checkpoint_info=None):
    command = [sys.executable, '-u', str(ROOT / 'train_image.py'),
               '--root_path', str(args.root_path), '--output_dir', str(args.output_dir),
               '--run_name', args.run_name, '--cfg', str(CFG),
               '--num_workers', str(args.num_workers), '--seed', str(args.seed),
               '--architecture_id', ARCHITECTURE_ID,
               '--stage2_structure_location', args.location]
    for name, value in REFERENCE.items():
        if isinstance(value, bool):
            if value:
                command.append('--' + name)
            elif name == 'use_ema':
                command.append('--no-use_ema')
        else:
            command += ['--' + name, str(value)]
    if checkpoint_info is None:
        command += ['--pretrain_ckpt', str(args.pretrain_ckpt)]
    else:
        command += ['--resume', str(args.model_path)]
        pretrained = checkpoint_info['args'].get('pretrain_ckpt')
        if pretrained:
            command += ['--pretrain_ckpt', str(pretrained)]
    return command


def infer_command(args, split, thresholds):
    return [sys.executable, '-u', str(ROOT / 'test_image.py'),
            '--root_path', str(args.root_path), '--model_path', str(args.model_path),
            '--cfg', str(CFG), '--split', split, '--img_size', '512',
            '--source_patch_size', '1500', '--overlap_infer',
            '--architecture_id', ARCHITECTURE_ID, '--stage2_structure_location', args.location,
            '--output_dir', str(args.output_dir / 'predictions' / split),
            '--threshold', str(thresholds[0]), '--export_thresholds'] + [str(t) for t in thresholds]


def run(command, env):
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def provenance(args, info, implementation):
    return {'checkpoint': str(args.model_path), 'checkpoint_sha256': info['checkpoint_sha256'],
            'checkpoint_epoch': info['epoch'], 'architecture_contract': info['architecture_contract'],
            'base_model_commit': BASE_COMMIT, 'checkpoint_args_sha256': digest(info['args']),
            'reference_training': dict(REFERENCE, **DEFAULTS),
            'seed': info['args'].get('seed'), 'data_root': str(args.root_path),
            'pretrained_source': info['args'].get('pretrain_ckpt'),
            'training_crop': 'one random512 crop/native1500 image/epoch; shared epoch',
            'weights': 'raw; no EMA', 'inference_tile_size': 512, 'overlap_stride': 256,
            'merge': 'taper-weighted logits; divisor clamp1e-8; sigmoid', 'tta': 'none',
            'postprocessing': 'binary surface threshold only',
            'original_model_source_sha256': read_json(ROOT / 'BASE_MODEL_SHA256.json'),
            'original_protocol_source_sha256': read_json(ROOT / 'PROTOCOL_SOURCE_SHA256.json'),
            'implementation_source_sha256': implementation}


def manifest(args, info, implementation, thresholds=None):
    name = MODEL_NAMES[args.location]
    path = args.output_dir / 'comparison.json'
    identity = provenance(args, info, implementation)
    if thresholds is not None:
        candidates = [{'threshold': t, 'path': str(args.output_dir / 'predictions' / 'val'
                                                  / f'threshold_{t:.2f}' / 'surface')}
                      for t in thresholds]
        data = {'data_root': str(args.root_path), 'protocol': PROTOCOL,
                'models': [{'name': name, 'provenance': identity,
                            'predictions': {'val': candidates, 'test': []}}]}
        if path.exists():
            raise FileExistsError(f'Use a fresh report directory: {path}')
        return path, data
    data = read_json(path)
    if (data.get('data_root') != str(args.root_path) or data.get('protocol') != PROTOCOL
            or len(data.get('models', [])) != 1 or data['models'][0].get('name') != name
            or data['models'][0].get('provenance') != identity):
        raise ValueError('Checkpoint, location, data root, or source changed; sweep into a fresh directory')
    # Validation paths remain evidence in the manifest, but are never reopened on test.
    return path, data


def selected_threshold(args, data):
    selection_path = args.output_dir / 'val' / 'val_selection.json'
    selection = read_json(selection_path)
    model = data['models'][0]
    selected = selection.get('models', {}).get(MODEL_NAMES[args.location], {})
    threshold = selected.get('threshold')
    candidates = {candidate['threshold'] for candidate in model['predictions']['val']}
    if (selection.get('selected_on') != 'val' or selection.get('criterion') != 'global_iou'
            or selected.get('identity') != digest({'name': model['name'], 'provenance': model['provenance']})
            or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
            or threshold not in candidates):
        raise ValueError('Invalid validation selection or mismatched model identity')
    # The unchanged evaluator expands defaults before hashing the metric protocol.
    sys.path.insert(0, str(EVAL.parent))
    from compare_predictions import validate_manifest
    if selection.get('protocol_hash') != digest(validate_manifest(data)):
        raise ValueError('Validation selection uses a different metric protocol')
    return selection_path, selection, float(threshold)


def compare_command(args, split, path, output, selection=None):
    command = [sys.executable, '-u', str(EVAL), '--manifest', str(path), '--split', split,
               '--output_dir', str(output), '--progress_every', '10']
    if split == 'val':
        command.append('--select_threshold_only')
    else:
        command += ['--selection', str(selection)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'train', 'resume', 'sweep', 'test', 'topology'])
    parser.add_argument('--location', required=True, choices=LOCATIONS)
    parser.add_argument('--root_path', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--pretrain_ckpt', type=Path)
    parser.add_argument('--model_path', type=Path)
    parser.add_argument('--run_name', default='')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[round(i / 100, 2) for i in range(5, 100, 5)])
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='print train/resume command without executing or writing pointers')
    parser.add_argument('--implementation_manifest', type=Path, default=ROOT / 'IMPLEMENTATION_SHA256.json')
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error('num_workers must be nonnegative')
    if args.dry_run and args.mode not in ('train', 'resume'):
        parser.error('--dry_run is valid only for train or resume; inspect is already read-only')
    implementation = verify_implementation_sources(args.implementation_manifest,
                                                  required=args.mode != 'inspect' and not args.dry_run)
    info = None
    if args.model_path is not None:
        args.model_path = args.model_path.resolve()
        info = verify_checkpoint(args.model_path, args.location)
    if args.mode == 'inspect':
        preview = argparse.Namespace(**vars(args))
        preview.root_path = args.root_path.resolve() if args.root_path else '<DATA_ROOT>'
        preview.output_dir = args.output_dir.resolve() if args.output_dir else '<OUTPUT_ROOT>'
        preview.pretrain_ckpt = args.pretrain_ckpt.resolve() if args.pretrain_ckpt else '<IMAGENET_SWINV2_TINY>'
        preview.run_name = args.run_name or MODEL_NAMES[args.location] + '_RUN'
        preview.seed = info['args'].get('seed', 1234) if info else (1234 if args.seed is None else args.seed)
        print(json.dumps({'architecture_id': ARCHITECTURE_ID, 'location': args.location,
                          'model_name': MODEL_NAMES[args.location], 'checkpoint': info,
                          'gpu': args.gpu, 'command': shlex.join(train_command(preview, info)),
                          'implementation_verified': implementation is not None,
                          'reference_training': dict(REFERENCE, **DEFAULTS)}, indent=2))
        return
    if args.root_path is None or args.output_dir is None:
        parser.error('Provide --root_path and --output_dir')
    args.root_path, args.output_dir = args.root_path.resolve(), args.output_dir.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OPENCV_LOG_LEVEL='ERROR',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
    if args.mode in ('train', 'resume'):
        if args.mode == 'train':
            if info is not None:
                parser.error('Fresh train does not accept --model_path; use resume for a matching checkpoint')
            if args.pretrain_ckpt is None or (not args.dry_run and not args.pretrain_ckpt.is_file()):
                parser.error('Provide the ImageNet SwinV2 Tiny --pretrain_ckpt')
            args.pretrain_ckpt = args.pretrain_ckpt.resolve()
            args.seed = 1234 if args.seed is None else args.seed
            args.run_name = args.run_name or MODEL_NAMES[args.location] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
            if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
                parser.error('run_name must be a single directory name')
            run_dir = args.output_dir / args.run_name
            if run_dir.exists():
                parser.error(f'Fresh run already exists: {run_dir}')
        else:
            if info is None or not info['can_resume']:
                parser.error('Resume requires a matching checkpoint with optimizer and loaded_pretrained_names')
            if info['epoch'] >= 80:
                parser.error('This checkpoint already reached the 80-epoch budget')
            if args.pretrain_ckpt is not None:
                parser.error('Resume uses saved pretraining metadata; omit --pretrain_ckpt')
            saved_seed = info['args'].get('seed')
            if not isinstance(saved_seed, int) or (args.seed is not None and args.seed != saved_seed):
                parser.error('Resume seed must match checkpoint seed')
            args.seed = saved_seed
            run_dir = args.model_path.parent
            if run_dir.name.lower() == 'checkpoints':
                run_dir = run_dir.parent
            if args.run_name and args.run_name != run_dir.name:
                parser.error('Resume reuses the checkpoint run directory and name')
            args.run_name = run_dir.name
        if args.dry_run:
            print(f'GPU={args.gpu}; dry run; no process launched or run pointers written', flush=True)
            print(shlex.join(train_command(args, info)), flush=True)
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pointer = args.output_dir / f'{ARCHITECTURE_ID}_{args.location}_latest.json'
        write_json(pointer, {'architecture_id': ARCHITECTURE_ID, 'stage2_structure_location': args.location,
                             'run_dir': str(run_dir), 'best': str(run_dir / 'best.pth'),
                             'last': str(run_dir / 'last.pth'), 'seed': args.seed})
        run(train_command(args, info), env)
        print(f'Best checkpoint: {run_dir / "best.pth"}')
        return
    if info is None:
        parser.error('Provide --model_path from a matching location run')
    path, data = manifest(args, info, implementation,
                          sorted(args.thresholds) if args.mode == 'sweep' else None)
    if args.mode == 'sweep':
        thresholds = sorted(args.thresholds)
        if (len(thresholds) != len(set(thresholds)) or any(not math.isfinite(t) or not 0 < t < 1
                or abs(t - round(t, 2)) > 1e-8 for t in thresholds)):
            parser.error('Supply unique thresholds in (0,1) with at most two decimals')
        val_output = args.output_dir / 'val'
        if val_output.exists() and any(val_output.iterdir()):
            parser.error('Use a fresh validation report directory')
        run(infer_command(args, 'val', thresholds), env)
        write_json(path, data)
        run(compare_command(args, 'val', path, val_output), env)
        print((val_output / 'val_threshold_scores.csv').read_text(encoding='utf-8'))
        return
    selection_path, selection, threshold = selected_threshold(args, data)
    print(f'Validation-selected threshold: {threshold:.2f}', flush=True)
    metric_dir = args.output_dir / 'test'
    if args.recompute:
        metric_dir = args.output_dir / ('test_topology_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    report_path = metric_dir / 'test_report.json'
    csv_path = metric_dir / 'test_comparison.csv'
    if args.mode == 'topology' and not args.recompute and report_path.is_file() and csv_path.is_file():
        cached = read_json(report_path)
        if (cached.get('provenance') != {MODEL_NAMES[args.location]: data['models'][0]['provenance']}
                or cached.get('threshold_selection') != selection or cached.get('split') != 'test'):
            raise ValueError('Cached test report belongs to different provenance or threshold selection')
        print(csv_path.read_text(encoding='utf-8'))
        return
    if metric_dir.exists() and any(metric_dir.iterdir()):
        parser.error('Metric directory is nonempty; use topology --recompute for a fresh report')
    if args.mode == 'test':
        run(infer_command(args, 'test', [threshold]), env)
    test_dir = args.output_dir / 'predictions' / 'test' / f'threshold_{threshold:.2f}' / 'surface'
    if not test_dir.is_dir():
        raise FileNotFoundError(f'Generate selected test masks first: {test_dir}')
    data['models'][0]['predictions']['test'] = [{'threshold': threshold, 'path': str(test_dir)}]
    write_json(path, data)
    run(compare_command(args, 'test', path, metric_dir, selection_path), env)
    print(csv_path.read_text(encoding='utf-8'))
    print(f'Per-image topology: {metric_dir / (MODEL_NAMES[args.location] + "_test_per_image.csv")}')


if __name__ == '__main__':
    main()
