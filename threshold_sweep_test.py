import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)
from losses.cldice_loss import soft_skeletonize
from losses.road_losses import build_connectivity_target
from config import get_config


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


def compute_global_surface_metrics(logits_list, targets_list, threshold):
    tp = fp = fn = 0
    cldice_values = []
    for logits, targets in zip(logits_list, targets_list):
        probs = torch.sigmoid(logits).float()
        targets = (targets > 0.5).float()
        pred = (probs >= threshold).float()
        tp += int((pred * targets).sum().item())
        fp += int((pred * (1.0 - targets)).sum().item())
        fn += int(((1.0 - pred) * targets).sum().item())
        cldice_values.extend(hard_cldice_scores(pred, targets).tolist())
    metrics = metrics_from_counts(tp, fp, fn)
    metrics['cldice'] = float(np.mean(cldice_values)) if cldice_values else float('nan')
    metrics['tp'] = tp
    metrics['fp'] = fp
    metrics['fn'] = fn
    return metrics


def hard_cldice_scores(pred, targets, iter_num=10):
    pred = pred.float().clamp(0.0, 1.0)
    targets = targets.float().clamp(0.0, 1.0)
    pred_skel = soft_skeletonize(pred, iter_num=iter_num)
    target_skel = soft_skeletonize(targets, iter_num=iter_num)
    tprec = (pred_skel * targets).sum(dim=(1, 2, 3)) / (
        pred_skel.sum(dim=(1, 2, 3)) + 1e-8
    )
    tsens = (target_skel * pred).sum(dim=(1, 2, 3)) / (
        target_skel.sum(dim=(1, 2, 3)) + 1e-8
    )
    return (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)


def compute_connectivity_metrics(connectivity_logits_list, skeleton_targets_list, threshold):
    tp = fp = fn = 0
    for logits, skeleton_targets in zip(connectivity_logits_list, skeleton_targets_list):
        targets = build_connectivity_target(skeleton_targets.float())
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(targets, size=logits.shape[-2:], mode='nearest')
        pred = (torch.sigmoid(logits) >= threshold).float()
        targets = (targets > 0.5).float()
        tp += int((pred * targets).sum().item())
        fp += int((pred * (1.0 - targets)).sum().item())
        fn += int(((1.0 - pred) * targets).sum().item())
    metrics = metrics_from_counts(tp, fp, fn)
    metrics['tp'] = tp
    metrics['fp'] = fp
    metrics['fn'] = fn
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./data1')
    parser.add_argument('--model_path', type=str, 
                       default='./model_out/train_skeleton_20260521_200553/checkpoints/epoch_100.pth')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--skeleton_threshold', type=float, default=0.15)
    parser.add_argument('--connectivity_threshold', type=float, default=0.5)
    parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml')
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
    parser.add_argument('--enable_global_topology', action='store_true')
    parser.add_argument('--global_topology_max_nodes', type=int, default=32)
    parser.add_argument('--global_topology_heads', type=int, default=4)
    parser.add_argument('--global_topology_reach_hops', type=int, default=12)
    parser.add_argument('--global_topology_nms_radius', type=int, default=2)
    parser.add_argument('--global_topology_skeleton_threshold', type=float, default=0.5)
    parser.add_argument('--global_topology_connectivity_threshold', type=float, default=0.25)
    parser.add_argument('--global_topology_bend_angle_threshold', type=float, default=45.0)
    parser.add_argument('--global_topology_alpha_max', type=float, default=0.05)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    
    print('\nLoading model from: {}'.format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    saved_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
    for name in (
        'structure_profile', 'bottleneck_type', 'enable_highres_structure_stream',
        'highres_structure_channels', 'highres_structure_fuse_stages',
        'highres_structure_fusion_mode', 'disable_msfe_skip',
        'enable_global_topology', 'global_topology_max_nodes', 'global_topology_heads',
        'global_topology_reach_hops', 'global_topology_nms_radius',
        'global_topology_skeleton_threshold', 'global_topology_connectivity_threshold',
        'global_topology_bend_angle_threshold', 'global_topology_alpha_max',
    ):
        if isinstance(saved_args, dict) and name in saved_args:
            setattr(args, name, saved_args[name])
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=getattr(args, 'bottleneck_type', 'global_local'),
        structure_profile=getattr(args, 'structure_profile', 'full'),
        use_msfe_skip=not getattr(args, 'disable_msfe_skip', False),
        enable_highres_structure_stream=getattr(args, 'enable_highres_structure_stream', False),
        highres_structure_channels=getattr(args, 'highres_structure_channels', 64),
        highres_structure_fuse_stages=getattr(args, 'highres_structure_fuse_stages', 'stage23'),
        highres_structure_fusion_mode=getattr(args, 'highres_structure_fusion_mode', 'stage23'),
        enable_global_topology=getattr(args, 'enable_global_topology', False),
        global_topology_max_nodes=getattr(args, 'global_topology_max_nodes', 32),
        global_topology_heads=getattr(args, 'global_topology_heads', 4),
        global_topology_reach_hops=getattr(args, 'global_topology_reach_hops', 12),
        global_topology_nms_radius=getattr(args, 'global_topology_nms_radius', 2),
        global_topology_skeleton_threshold=getattr(args, 'global_topology_skeleton_threshold', 0.5),
        global_topology_connectivity_threshold=getattr(args, 'global_topology_connectivity_threshold', 0.25),
        global_topology_bend_angle_threshold=getattr(args, 'global_topology_bend_angle_threshold', 45.0),
        global_topology_alpha_max=getattr(args, 'global_topology_alpha_max', 0.05),
    )
    load_topology_checkpoint_state(
        model,
        checkpoint['model_state_dict'],
        checkpoint.get('topology_attention_version', 'legacy-unrecorded'),
        strict=True,
    )
    model = model.to(device)
    model.eval()
    print('Model loaded')
    
    print('\nLoading test dataset')
    test_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split='test',
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print('Test set size: {}'.format(len(test_dataset)))
    
    print('\nRunning inference on test set')
    all_surface_logits = []
    all_skeleton_logits = []
    all_connectivity_logits = []
    all_surface_targets = []
    all_skeleton_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            skeleton_masks = batch['skeleton'].to(device)
            
            outputs = model(images)
            skeleton_logits = None
            connectivity_logits = None
            
            if isinstance(outputs, tuple):
                surface_logits = outputs[0]
                if len(outputs) > 2 and torch.is_tensor(outputs[2]):
                    skeleton_logits = outputs[2]
                if len(outputs) > 3 and torch.is_tensor(outputs[3]):
                    connectivity_logits = outputs[3]
            else:
                surface_logits = outputs
            
            all_surface_logits.append(surface_logits.cpu())
            all_surface_targets.append(masks.cpu())
            
            if skeleton_logits is not None:
                all_skeleton_logits.append(skeleton_logits.cpu())
                all_skeleton_targets.append(skeleton_masks.cpu())
            if connectivity_logits is not None:
                all_connectivity_logits.append(connectivity_logits.cpu())
    
    print('Inference complete')
    
    print('\n' + '='*80)
    print('SURFACE SEGMENTATION - THRESHOLD SWEEP (TEST SET)')
    print('='*80)
    print('Input pipeline: 1024 center crop/pad -> resize256 -> model -> threshold -> global TP/FP/FN')
    print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('Threshold', 'IoU', 'F1', 'Precision', 'Recall', 'clDice'))
    print('-'*74)
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    surface_results = {}
    
    for threshold in thresholds:
        metrics = compute_global_surface_metrics(
            all_surface_logits, 
            all_surface_targets, 
            threshold
        )
        surface_results[threshold] = metrics
        print('{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(
            threshold, metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall'], metrics['cldice']
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
        print('SKELETON SEGMENTATION - THRESHOLD SWEEP (TEST SET)')
        print('='*80)
        print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Threshold', 'IoU', 'F1', 'Precision', 'Recall'))
        print('-'*60)
        
        skeleton_results = {}
        
        for threshold in thresholds:
            metrics = compute_global_surface_metrics(
                all_skeleton_logits, 
                all_skeleton_targets, 
                threshold
            )
            skeleton_results[threshold] = metrics
            print('{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(
                threshold, metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall']
            ))

    if all_connectivity_logits:
        conn_metrics = compute_connectivity_metrics(
            all_connectivity_logits,
            all_skeleton_targets,
            args.connectivity_threshold,
        )
        print('\n' + '='*80)
        print('FINAL CONNECTIVITY - GLOBAL EDGE METRICS')
        print('='*80)
        print('Threshold: {:.2f}'.format(args.connectivity_threshold))
        print(
            'IoU: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(
                conn_metrics['iou'],
                conn_metrics['f1'],
                conn_metrics['precision'],
                conn_metrics['recall'],
            )
        )
        
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
