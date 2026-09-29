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
)
from losses.road_losses import binary_metrics_from_logits
from config import get_config


def compute_metrics_all_samples(logits_list, targets_list, threshold):
    all_metrics = {
        'iou': [],
        'f1': [],
        'precision': [],
        'recall': [],
    }
    
    for logits, targets in zip(logits_list, targets_list):
        metrics = binary_metrics_from_logits(logits, targets, threshold=threshold)
        all_metrics['iou'].append(metrics['iou'])
        all_metrics['f1'].append(metrics['f1'])
        all_metrics['precision'].append(metrics['precision'])
        all_metrics['recall'].append(metrics['recall'])
    
    return {
        'iou': np.mean(all_metrics['iou']),
        'f1': np.mean(all_metrics['f1']),
        'precision': np.mean(all_metrics['precision']),
        'recall': np.mean(all_metrics['recall']),
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
    parser.add_argument('--dataset', type=str, default='ImageData')
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    model = model.to(device)
    model.eval()
    print('Model loaded')
    
    print('\nLoading {} dataset'.format(args.split))
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
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
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
