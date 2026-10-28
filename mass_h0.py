"""Run H0 surface-only plus global soft-gate on Massachusetts."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import torch

from mass_protocol import BASE_COMMIT, MODEL_NAME, validate_training_args


ROOT = Path(__file__).resolve().parent
EVAL = ROOT / 'evaluation' / 'road_comparison'
CFG = ROOT / 'configs' / 'swin_tiny_patch4_window7_224_lite.yaml'


def architecture_flags():
    return ['--structure_profile', 'stage23_boundary_0626',
            '--stage_skeleton_mode', 'direct',
            '--highres_structure_channels', '64',
            '--remove_stage2_pre_topology_source', '--highres_structure_fuse_stages', 'stage23',
            '--highres_structure_fusion_mode', 'stage23', '--enable_global_topology',
            '--global_topology_max_nodes', '32', '--global_topology_heads', '4',
            '--global_topology_alpha_max', '0.05']


def train_command(args):
    return [sys.executable, '-u', str(ROOT / 'train_image.py'),
            '--root_path', str(args.root_path), '--output_dir', str(args.output_dir),
            '--run_name', args.run_name, '--cfg', str(CFG),
            '--pretrain_ckpt', str(args.pretrain_ckpt),
            '--img_size', '512', '--source_patch_size', '1500', '--overlap_stride', '256',
            '--random_crop_train', '--random_crops_per_image', '1', '--max_epochs', '80',
            '--batch_size', '2', '--accumulation_steps', '1', '--val_interval', '5',
            '--num_workers', str(args.num_workers), '--seed', str(args.seed),
            '--amp_dtype', 'none', '--no-use_ema', '--trend_val_every_steps', '0',
            '--threshold', '0.5', '--base_lr', '5e-4',
            '--pretrained_lr', '5e-5', '--pretrained_min_lr', '5e-6',
            '--new_lr', '2e-4', '--new_min_lr', '1e-5', '--warmup_epochs', '10',
            '--stage2_skeleton_weight', '0.0', '--stage3_skeleton_weight', '0.0',
            '--stage_skeleton_loss_factor', '1.0', '--stage_skeleton_only_loss_factor', '1.0',
            '--stage2_skeleton_gradient_ratio', '0.5', '--stage3_skeleton_gradient_ratio', '0.5',
            '--stage3_gate_topology_gradient_ratio', '0.0', '--skeleton_pos_weight', '2.0',
            '--highres_structure_skeleton_weight', '0.0', '--stage_direction_factor', '0.0',
            '--stage_connectivity_factor', '3.0', '--road_attention_weight', '0.0',
            '--connectivity_pos_weight', '2.0',
            '--connectivity_focal_gamma', '1.5', '--edge_contrastive_margin', '0.1',
            '--directional_pos_weight_cardinal', '1.0', '--directional_pos_weight_diagonal', '2.5',
            '--surface_focal_gamma', '1.0'] + architecture_flags()


def run(command, env):
    print('\n' + shlex.join([str(value) for value in command]), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def verify_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('Expected a checkpoint produced by this Massachusetts H0 training entry')
    if checkpoint.get('experiment_variant') != 'h0_surface_global_softgate_v1':
        raise ValueError('Checkpoint is not from this surface-global experiment')
    if checkpoint.get('experiment_base_commit') != BASE_COMMIT:
        raise ValueError('Checkpoint does not record the original 599a410 H0 architecture')
    validate_training_args(checkpoint.get('args', {}))
    if checkpoint.get('ema_state_dict') is not None:
        raise ValueError('This experiment requires raw weights without EMA')
    result = {'epoch': checkpoint.get('epoch'), 'args': checkpoint['args']}
    del checkpoint
    return result


def manifest(args, checkpoint_info, selection=None):
    prediction_root = args.output_dir / 'predictions'
    val = []
    for directory in sorted((prediction_root / 'val').glob('threshold_*')):
        if (directory / 'surface').is_dir():
            val.append({'threshold': float(directory.name.split('_', 1)[1]),
                        'path': str(directory / 'surface')})
    if not val:
        raise ValueError('Run sweep first to generate validation threshold masks')
    test = []
    if selection is not None:
        if selection.get('selected_on') != 'val' or MODEL_NAME not in selection.get('models', {}):
            raise ValueError('Selection must come from this model on the validation split')
        threshold = float(selection['models'][MODEL_NAME]['threshold'])
        directory = prediction_root / 'test' / f'threshold_{threshold:.2f}' / 'surface'
        if not directory.is_dir():
            raise FileNotFoundError(f'Generate the selected test masks first: {directory}')
        test = [{'threshold': threshold, 'path': str(directory)}]
    source_files = ['train_image.py', 'test_image.py', 'mass_h0.py', 'mass_protocol.py',
                    'datasets/dataset_road_skeleton.py', 'config.py',
                    'evaluation/road_comparison/compare_predictions.py',
                    'evaluation/road_comparison/road_metrics.py']
    source_files += ['networks/surface_only.py', 'networks/keypoint_global_topology.py', 'networks/swin_transformer_unet_skip_expand_decoder_sys.py', 'networks/vision_transformer.py']
    data = {'data_root': str(args.root_path),
            'protocol': {'image_size': 1500, 'short_area_threshold': 20, 'apls_max_control_nodes': 150, 'topo_seed_density_deg': 0.002, 'topo_step_deg': 0.0002},
            'models': [{'name': MODEL_NAME, 'provenance': {
                'checkpoint': str(args.model_path), 'checkpoint_epoch': checkpoint_info['epoch'],
                'base_model_commit': BASE_COMMIT, 'checkpoint_args_verified': True,
                'training_crop': 'one random512 crop per native1500 image per epoch; shared epoch',
                'training_precision': 'FP32', 'training_max_epochs': 80,
                'micro_batch_size': 2, 'accumulation_steps': 1, 'weights': 'raw; no EMA',
                'pretrained_source': checkpoint_info['args'].get('pretrain_ckpt'),
                'inference_tile_size': 512, 'overlap_stride': 256,
                'merge': 'same train/test taper-weighted logits; divisor clamp1e-8; sigmoid',
                'tta': 'none', 'postprocessing': 'binary surface threshold only',
                'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in source_files}},
                'predictions': {'val': val, 'test': test}}]}
    destination = args.output_dir / 'comparison.json'
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return destination


def infer_command(args, split, thresholds):
    return [sys.executable, '-u', str(ROOT / 'test_image.py'),
            '--root_path', str(args.root_path), '--model_path', str(args.model_path),
            '--cfg', str(CFG), '--split', split, '--img_size', '512', '--source_patch_size', '1500',
            '--overlap_infer', '--output_dir', str(args.output_dir / 'predictions' / split),
            '--threshold', str(thresholds[0]), '--export_thresholds'] + [str(t) for t in thresholds]


def compare_command(args, split, manifest_path, selection_path=None, output_dir=None):
    command = [sys.executable, '-u', str(EVAL / 'compare_predictions.py'),
               '--manifest', str(manifest_path), '--split', split,
               '--output_dir', str(output_dir or args.output_dir / split), '--progress_every', '1', '--workers', '4']
    if split == 'val':
        command += ['--select_threshold_only']
    if selection_path is not None:
        command += ['--selection', str(selection_path)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'sweep', 'test', 'topology'])
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--pretrain_ckpt', type=Path)
    parser.add_argument('--model_path', type=Path)
    parser.add_argument('--run_name', default='')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[round(i / 100, 2) for i in range(5, 100, 5)])
    parser.add_argument('--recompute', action='store_true', help='recompute cached test topology metrics')
    args = parser.parse_args()
    args.root_path, args.output_dir = args.root_path.resolve(), args.output_dir.resolve()
    if args.num_workers < 0:
        parser.error('num_workers must be nonnegative')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OPENCV_LOG_LEVEL='ERROR',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.mode == 'train':
        if args.pretrain_ckpt is None or not args.pretrain_ckpt.is_file():
            parser.error('Provide the ImageNet SwinV2 Tiny --pretrain_ckpt')
        args.pretrain_ckpt = args.pretrain_ckpt.resolve()
        args.run_name = args.run_name or 'mass_h0_surface_global_softgate_random512_fp32_80e_bs2_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = args.output_dir / args.run_name
        if run_dir.exists():
            parser.error(f'Fresh run directory already exists: {run_dir}')
        print(f'GPU={args.gpu}; FP32; no EMA; 80 epochs; batch2; accumulation1; run={run_dir}', flush=True)
        run(train_command(args), env)
        print(f'Best checkpoint: {run_dir / "best.pth"}')
        return
    if args.model_path is None or not args.model_path.is_file():
        parser.error('Provide --model_path from this Massachusetts H0 run')
    args.model_path = args.model_path.resolve()
    checkpoint_info = verify_checkpoint(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'sweep':
        thresholds = sorted(set(args.thresholds))
        if (len(thresholds) != len(args.thresholds) or
                any(not 0 < t < 1 or abs(t - round(t, 2)) > 1e-8 for t in thresholds)):
            parser.error('Supply unique thresholds in (0,1) with at most two decimals')
        run(infer_command(args, 'val', thresholds), env)
        destination = manifest(args, checkpoint_info)
        run(compare_command(args, 'val', destination), env)
        print((args.output_dir / 'val' / 'val_threshold_scores.csv').read_text(encoding='utf-8'))
        return
    selection_path = args.output_dir / 'val' / 'val_selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    threshold = float(selection['models'][MODEL_NAME]['threshold'])
    print(f'Validation-selected threshold: {threshold:.2f}', flush=True)
    if args.mode == 'test':
        run(infer_command(args, 'test', [threshold]), env)
    destination = manifest(args, checkpoint_info, selection)
    if args.mode == 'test':
        print(f'Test predictions exported; manifest: {destination}')
        return
    metric_dir = args.output_dir / 'test'
    if args.recompute:
        metric_dir = args.output_dir / ('test_topology_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    report = metric_dir / 'test_comparison.csv'
    if args.recompute or not report.is_file():
        run(compare_command(args, 'test', destination, selection_path, metric_dir), env)
    print(report.read_text(encoding='utf-8'))
    print(f'Per-image topology: {metric_dir / (MODEL_NAME + "_test_per_image.csv")}')


if __name__ == '__main__':
    main()
