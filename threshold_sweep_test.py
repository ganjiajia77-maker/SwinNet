import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    restore_routing_checkpoint_state,
)
from config import get_config


def compute_metrics_all_samples(logits_list, targets_list, threshold):
    tp = fp = fn = 0
    for logits, targets in zip(logits_list, targets_list):
        pred = torch.sigmoid(logits) >= threshold
        gt = targets > 0.5
        tp += int((pred & gt).sum().item())
        fp += int((pred & (~gt)).sum().item())
        fn += int(((~pred) & gt).sum().item())
    return metrics_from_counts(tp, fp, fn)


def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def run_fixed_crop_sweep(
    model,
    loader,
    thresholds,
    tile_size,
    stride,
    device,
    inference_dtype=None,
    use_inference_amp=False,
):
    positions = RoadSkeletonDataset.sliding_positions
    weight_1d = torch.linspace(-1.0, 1.0, steps=tile_size, device=device).abs()
    weight_1d = (1.0 - weight_1d).clamp_min(0.1)
    tile_weight = (weight_1d[:, None] * weight_1d[None, :]).view(
        1, 1, tile_size, tile_size
    )
    counts = {
        threshold: {'tp': 0, 'fp': 0, 'fn': 0}
        for threshold in thresholds
    }
    with torch.no_grad():
        for batch in tqdm(loader, desc='Fixed-crop inference'):
            images = batch['image'].to(device)
            masks = (batch['mask'].to(device) > 0.5)
            if images.shape[0] != 1:
                raise ValueError('--fixed_crop_eval expects batch_size=1.')
            _, _, height, width = images.shape
            logit_canvas = torch.zeros((1, 1, height, width), device=device)
            weight_canvas = torch.zeros_like(logit_canvas)

            for top in positions(height, tile_size, stride):
                for left in positions(width, tile_size, stride):
                    bottom = min(top + tile_size, height)
                    right = min(left + tile_size, width)
                    tile = images[:, :, top:bottom, left:right]
                    pad_h = tile_size - tile.shape[-2]
                    pad_w = tile_size - tile.shape[-1]
                    if pad_h > 0 or pad_w > 0:
                        tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h))
                    with torch.autocast(
                        device_type=device.type,
                        dtype=inference_dtype if inference_dtype is not None else torch.float32,
                        enabled=use_inference_amp,
                    ):
                        outputs = model(tile)
                    surface_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    tile_height = bottom - top
                    tile_width = right - left
                    tile_logits = surface_logits[:, :, :tile_height, :tile_width]
                    weight = tile_weight[:, :, :tile_height, :tile_width]
                    logit_canvas[:, :, top:bottom, left:right] += tile_logits * weight
                    weight_canvas[:, :, top:bottom, left:right] += weight

            if weight_canvas.min().item() <= 0:
                raise RuntimeError('Fixed-crop sweep left uncovered pixels.')
            prob = torch.sigmoid(logit_canvas / weight_canvas.clamp_min(1e-8))
            for threshold in thresholds:
                pred = prob >= threshold
                counts[threshold]['tp'] += int((pred & masks).sum().item())
                counts[threshold]['fp'] += int((pred & (~masks)).sum().item())
                counts[threshold]['fn'] += int(((~pred) & masks).sum().item())

    return {
        threshold: metrics_from_counts(
            values['tp'],
            values['fp'],
            values['fn'],
        )
        for threshold, values in counts.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./data1')
    parser.add_argument('--model_path', type=str, 
                       default='./model_out/train_skeleton_20260521_200553/checkpoints/epoch_100.pth')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--fixed_crop_eval', action='store_true',
                        help='evaluate full source patches by tiled img_size crops and global TP/FP/FN')
    parser.add_argument('--overlap_stride', type=int, default=0,
                        help='stride for --fixed_crop_eval; default uses img_size')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml')
    parser.add_argument(
        '--structure_profile',
        type=str,
        default='full',
        choices=['full', 'stage23_boundary_0626', 'stage23_boundary_0626_final_ske'],
    )
    parser.add_argument('--bottleneck_type', type=str, default='global_local', choices=['global_local', 'legacy_global_local', 'g2l2'])
    parser.add_argument('--disable_msfe_skip', action='store_true')
    parser.add_argument('--enable_highres_structure_stream', action='store_true')
    parser.add_argument('--highres_structure_channels', type=int, default=64)
    parser.add_argument(
        '--highres_structure_fuse_stages',
        type=str,
        default='stage23',
        choices=['stage2', 'stage3', 'stage23'],
    )
    parser.add_argument(
        '--highres_structure_fusion_mode',
        type=str,
        default='stage23',
        choices=[
            'stage23',
            'final_correction',
            'stage23_final_correction',
            'post_refine_interaction',
            'none',
        ],
    )
    parser.add_argument('--enable_global_topology', action='store_true')
    parser.add_argument('--global_topology_max_nodes', type=int, default=32)
    parser.add_argument('--global_topology_heads', type=int, default=4)
    parser.add_argument('--global_topology_alpha_max', type=float, default=0.05)
    parser.add_argument('--stage_skeleton_mode', type=str, default='prior_residual', choices=['direct', 'prior_residual'])
    parser.add_argument('--enable_e128_stage_fusion', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_h3_surface_fusion', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_coarse_road_mask', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_psi_directional_descriptor', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_sparse_window_compute', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--remove_stage2_pre_topology_source', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--stage2_window_threshold', type=float, default=0.25)
    parser.add_argument('--stage3_window_threshold', type=float, default=0.25)
    parser.add_argument('--routing_warmup_epochs', type=int, default=10)
    parser.add_argument('--coarse_candidate_window_size', type=int, default=8)
    parser.add_argument('--coarse_corridor_window_radius', type=int, default=0)
    parser.add_argument('--coarse_routing_mode', type=str, default='dense', choices=['dense', 'p64', 'bottleneck', 'bottleneck_no_psi'])
    parser.add_argument('--bottleneck_coarse_road_mask', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--bottleneck_window_threshold', type=float, default=0.25)
    parser.add_argument('--stage2_skeleton_gradient_ratio', type=float, default=0.5)
    parser.add_argument('--stage3_skeleton_gradient_ratio', type=float, default=0.5)
    parser.add_argument('--stage3_gate_topology_gradient_ratio', type=float, default=0.0)
    parser.add_argument('--final_skeleton_gradient_ratio', type=float, default=0.0)
    parser.add_argument('--final_topology_eta_init', type=float, default=0.005)
    parser.add_argument('--final_gap_rho_init', type=float, default=0.005)
    parser.add_argument('--zip', action='store_true', help='use zipped dataset')
    parser.add_argument('--cache_mode', type=str, default='', help='cache mode for dataset')
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--accumulation_steps', type=int, default=0)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--amp_opt_level', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument(
        '--amp_dtype',
        type=str,
        choices=['bfloat16', 'float16', 'none'],
        default='none',
        help='inference autocast dtype; use float16 on Turing GPUs',
    )
    parser.add_argument('--dataset', type=str, default='ImageData')
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inference_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }.get(args.amp_dtype)
    use_inference_amp = device.type == 'cuda' and inference_dtype is not None
    print('Using device: {}'.format(device))
    
    print('\nLoading model from: {}'.format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    saved_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        for name in (
            'structure_profile', 'bottleneck_type', 'enable_highres_structure_stream',
            'highres_structure_channels', 'highres_structure_fuse_stages',
            'highres_structure_fusion_mode', 'enable_global_topology',
            'global_topology_max_nodes', 'global_topology_heads',
            'global_topology_alpha_max', 'stage_skeleton_mode',
            'enable_e128_stage_fusion', 'enable_h3_surface_fusion',
            'enable_coarse_road_mask', 'enable_psi_directional_descriptor',
            'enable_sparse_window_compute', 'remove_stage2_pre_topology_source',
            'stage2_window_threshold', 'stage3_window_threshold',
            'routing_warmup_epochs',
            'coarse_candidate_window_size', 'coarse_corridor_window_radius',
            'coarse_routing_mode', 'bottleneck_coarse_road_mask',
            'bottleneck_window_threshold', 'stage2_skeleton_gradient_ratio',
            'stage3_skeleton_gradient_ratio', 'stage3_gate_topology_gradient_ratio',
            'final_skeleton_gradient_ratio', 'final_topology_eta_init',
            'final_gap_rho_init',
        ):
            if name in saved_args:
                setattr(args, name, saved_args[name])
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        structure_profile=args.structure_profile,
        stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
        stage3_gate_topology_gradient_ratio=args.stage3_gate_topology_gradient_ratio,
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_global_topology=args.enable_global_topology,
        global_topology_max_nodes=args.global_topology_max_nodes,
        global_topology_heads=args.global_topology_heads,
        global_topology_alpha_max=args.global_topology_alpha_max,
        stage_skeleton_mode=args.stage_skeleton_mode,
        enable_e128_stage_fusion=args.enable_e128_stage_fusion,
        enable_h3_surface_fusion=args.enable_h3_surface_fusion,
        enable_coarse_road_mask=args.enable_coarse_road_mask,
        enable_psi_directional_descriptor=args.enable_psi_directional_descriptor,
        sparse_window_compute=args.enable_sparse_window_compute,
        stage2_window_threshold=args.stage2_window_threshold,
        stage3_window_threshold=args.stage3_window_threshold,
        routing_warmup_epochs=args.routing_warmup_epochs,
        coarse_candidate_window_size=args.coarse_candidate_window_size,
        coarse_corridor_window_radius=args.coarse_corridor_window_radius,
        coarse_routing_mode=args.coarse_routing_mode,
        bottleneck_coarse_road_mask=args.bottleneck_coarse_road_mask,
        bottleneck_window_threshold=args.bottleneck_window_threshold,
        remove_stage2_pre_topology_source=args.remove_stage2_pre_topology_source,
    )
    load_topology_checkpoint_state(
        model,
        checkpoint['model_state_dict'],
        checkpoint.get('topology_attention_version', 'legacy-unrecorded'),
        strict=(args.bottleneck_type == 'global_local'),
    )
    route_state = restore_routing_checkpoint_state(model, checkpoint)
    if route_state is not None:
        print(f'[P64 ROUTING] {route_state}', flush=True)
    model = model.to(device)
    model.eval()
    print('Model loaded')
    
    print('\nLoading {} dataset'.format(args.split))
    if args.fixed_crop_eval:
        test_dataset = RoadSkeletonDataset(
            root_dir=args.root_path,
            split=args.split,
            image_size=None,
            source_patch_size=args.source_patch_size,
            return_full_image=True,
        )
        if args.batch_size != 1:
            print('[INFO] --fixed_crop_eval uses batch_size=1 for full-image tiling.')
        args.batch_size = 1
    else:
        test_dataset = RoadSkeletonDataset(
            root_dir=args.root_path,
            split=args.split,
            image_size=args.img_size,
            source_patch_size=args.source_patch_size,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print('{} set size: {}'.format(args.split.capitalize(), len(test_dataset)))
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    if args.fixed_crop_eval:
        stride = args.overlap_stride if args.overlap_stride > 0 else args.img_size
        print('\nRunning fixed-crop sweep on {} set: source={} tile={} stride={}'.format(
            args.split,
            args.source_patch_size,
            args.img_size,
            stride,
        ))
        surface_results = run_fixed_crop_sweep(
            model,
            test_loader,
            thresholds,
            args.img_size,
            stride,
            device,
            inference_dtype=inference_dtype,
            use_inference_amp=use_inference_amp,
        )
        print('\n' + '='*80)
        print('SURFACE SEGMENTATION - FIXED-CROP THRESHOLD SWEEP ({} SET)'.format(args.split.upper()))
        print('='*80)
        print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Threshold', 'IoU', 'F1', 'Precision', 'Recall'))
        print('-'*60)
        for threshold in thresholds:
            metrics = surface_results[threshold]
            print('{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(
                threshold, metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall']
            ))
        best_threshold_iou = max(surface_results.keys(), key=lambda t: surface_results[t]['iou'])
        best_threshold_f1 = max(surface_results.keys(), key=lambda t: surface_results[t]['f1'])
        print('\nBest threshold (IoU): {:.2f} -> IoU: {:.4f}'.format(
            best_threshold_iou, surface_results[best_threshold_iou]['iou']))
        print('Best threshold (F1):  {:.2f} -> F1: {:.4f}'.format(
            best_threshold_f1, surface_results[best_threshold_f1]['f1']))
        print('\n' + '='*80)
        print('Threshold sweep complete!')
        print('='*80)
        return

    print('\nRunning inference on {} set'.format(args.split))
    all_surface_logits = []
    all_skeleton_logits = []
    all_surface_targets = []
    all_skeleton_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            skeleton_masks = batch['skeleton'].to(device)
            
            with torch.autocast(
                device_type=device.type,
                dtype=inference_dtype if inference_dtype is not None else torch.float32,
                enabled=use_inference_amp,
            ):
                outputs = model(images)
            skeleton_logits = None
            
            if isinstance(outputs, tuple):
                surface_logits = outputs[0]
                if len(outputs) > 1 and torch.is_tensor(outputs[1]):
                    skeleton_logits = outputs[1]
            else:
                surface_logits = outputs
            
            all_surface_logits.append(surface_logits.cpu())
            all_surface_targets.append(masks.cpu())
            
            if skeleton_logits is not None:
                all_skeleton_logits.append(skeleton_logits.cpu())
                all_skeleton_targets.append(skeleton_masks.cpu())
    
    print('Inference complete')
    
    print('\n' + '='*80)
    print('SURFACE SEGMENTATION - THRESHOLD SWEEP ({} SET)'.format(args.split.upper()))
    print('='*80)
    print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Threshold', 'IoU', 'F1', 'Precision', 'Recall'))
    print('-'*60)
    
    surface_results = {}
    
    for threshold in thresholds:
        metrics = compute_metrics_all_samples(
            all_surface_logits, 
            all_surface_targets, 
            threshold
        )
        surface_results[threshold] = metrics
        print('{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(
            threshold, metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall']
        ))
    
    best_threshold_iou = max(surface_results.keys(), 
                             key=lambda t: surface_results[t]['iou'])
    best_threshold_f1 = max(surface_results.keys(), 
                            key=lambda t: surface_results[t]['f1'])
    
    print('\nBest threshold (IoU): {:.2f} -> IoU: {:.4f}'.format(
        best_threshold_iou, surface_results[best_threshold_iou]['iou']))
    print('Best threshold (F1):  {:.2f} -> F1: {:.4f}'.format(
        best_threshold_f1, surface_results[best_threshold_f1]['f1']))
    
    if all_skeleton_logits:
        print('\n' + '='*80)
        print('SKELETON SEGMENTATION - THRESHOLD SWEEP ({} SET)'.format(args.split.upper()))
        print('='*80)
        print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Threshold', 'IoU', 'F1', 'Precision', 'Recall'))
        print('-'*60)
        
        skeleton_results = {}
        
        for threshold in thresholds:
            metrics = compute_metrics_all_samples(
                all_skeleton_logits, 
                all_skeleton_targets, 
                threshold
            )
            skeleton_results[threshold] = metrics
            print('{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(
                threshold, metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall']
            ))
        
        best_threshold_skeleton_iou = max(skeleton_results.keys(), 
                                         key=lambda t: skeleton_results[t]['iou'])
        best_threshold_skeleton_f1 = max(skeleton_results.keys(), 
                                        key=lambda t: skeleton_results[t]['f1'])
        
        print('\nBest threshold (IoU): {:.2f} -> IoU: {:.4f}'.format(
            best_threshold_skeleton_iou, skeleton_results[best_threshold_skeleton_iou]['iou']))
        print('Best threshold (F1):  {:.2f} -> F1: {:.4f}'.format(
            best_threshold_skeleton_f1, skeleton_results[best_threshold_skeleton_f1]['f1']))
    
    print('\n' + '='*80)
    print('Threshold sweep complete!')
    print('='*80)


if __name__ == '__main__':
    main()
