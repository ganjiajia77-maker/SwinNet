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
    print_topology_coefficients,
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
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
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(
                targets.float(),
                size=logits.shape[-2:],
                mode='nearest',
            )
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
                       default='./model_out/train_skeleton_20260521_094935/best.pth')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--final_topology_eta_init', type=float, default=0.005)
    parser.add_argument('--final_gap_rho_init', type=float, default=0.005)
    parser.add_argument(
        '--stage_topology_stages',
        type=str,
        default='none',
        choices=['none', 'stage3', 'stage23'],
    )
    parser.add_argument('--stage_topology_alpha_max', type=float, default=1.0)
    parser.add_argument('--stage_topology_alpha_init', type=float, default=0.1)
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
    parser.add_argument(
        '--structure_profile',
        type=str,
        default=STRUCTURE_PROFILE_FULL,
        choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626],
    )
    parser.add_argument(
        '--enable_graph_prop',
        action='store_true',
        help='enable final soft skeleton graph propagation',
    )
    parser.add_argument(
        '--disable_msfe_skip',
        action='store_true',
        help='ablate MSFE blocks on decoder skip stages inx=2,3; auto-read from checkpoint when omitted',
    )
    
    args = parser.parse_args()
    
    enable_graph_prop = args.enable_graph_prop
    checkpoint = None
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            saved_profile = checkpoint.get("structure_profile")
            if saved_profile:
                args.structure_profile = saved_profile
            elif isinstance(checkpoint.get("args"), dict):
                args.structure_profile = checkpoint["args"].get(
                    "structure_profile",
                    args.structure_profile,
                )
            if not enable_graph_prop and isinstance(checkpoint.get("args"), dict):
                enable_graph_prop = bool(
                    checkpoint["args"].get("enable_graph_prop", False)
                )
            if isinstance(checkpoint.get("args"), dict) and "disable_msfe_skip" in checkpoint["args"]:
                args.disable_msfe_skip = bool(checkpoint["args"]["disable_msfe_skip"])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    
    print('\nLoading model from: {}'.format(args.model_path))
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        structure_profile=args.structure_profile,
        enable_final_graph_prop=enable_graph_prop,
        use_msfe_skip=not args.disable_msfe_skip,
    )
    if checkpoint is None:
        checkpoint = torch.load(args.model_path, map_location=device)
    load_topology_checkpoint_state(
        model,
        checkpoint['model_state_dict'],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model = model.to(device)
    model.eval()
    print('Model loaded')
    print("Using topology attention constrained version", flush=True)
    print_topology_coefficients(model)
    
    print('\nLoading validation dataset')
    val_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split='val',
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print('Validation set size: {}'.format(len(val_dataset)))
    
    print('\nRunning inference on validation set')
    all_surface_logits = []
    all_surface_targets = []
    all_skeleton_logits = []
    all_skeleton_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Inference'):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                surface_logits = outputs[0]
                skeleton_logits = outputs[2] if len(outputs) > 2 else None
            else:
                raise RuntimeError("Structure-guided threshold sweep requires auxiliary outputs.")
            
            all_surface_logits.append(surface_logits.cpu())
            all_surface_targets.append(masks.cpu())
            if skeleton_logits is not None:
                all_skeleton_logits.append(skeleton_logits.cpu())
                all_skeleton_targets.append(batch["skeleton"].cpu())
    
    print('Inference complete')
    
    print('\n' + '='*80)
    print('SURFACE SEGMENTATION - THRESHOLD SWEEP')
    print('='*80)
    print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Threshold', 'IoU', 'F1', 'Precision', 'Recall'))
    print('-'*60)
    
    thresholds = [
        0.10,
        0.15,
        0.20,
        0.22,
        0.24,
        0.25,
        0.26,
        0.28,
        0.30,
        0.32,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ]
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

    print('\n' + '='*80)
    if all_skeleton_logits:
        print('FINAL SKELETON (256x256) - THRESHOLD SWEEP')
    else:
        print('FINAL SKELETON (256x256) - SKIPPED (final skeleton head disabled)')
    print('='*80)
    if all_skeleton_logits:
        print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format(
            'Threshold', 'IoU', 'F1', 'Precision', 'Recall'
        ))
        print('-'*60)

        skeleton_results = {}
        for threshold in thresholds:
            metrics = compute_metrics_all_samples(
                all_skeleton_logits,
                all_skeleton_targets,
                threshold,
            )
            skeleton_results[threshold] = metrics
            print('{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(
                threshold,
                metrics['iou'],
                metrics['f1'],
                metrics['precision'],
                metrics['recall'],
            ))

        best_skeleton_threshold_iou = max(
            skeleton_results.keys(),
            key=lambda t: skeleton_results[t]['iou'],
        )
        best_skeleton_threshold_f1 = max(
            skeleton_results.keys(),
            key=lambda t: skeleton_results[t]['f1'],
        )
        print('\nBest final skeleton threshold (IoU): {:.2f} -> IoU: {:.4f}'.format(
            best_skeleton_threshold_iou,
            skeleton_results[best_skeleton_threshold_iou]['iou'],
        ))
        print('Best final skeleton threshold (F1):  {:.2f} -> F1: {:.4f}'.format(
            best_skeleton_threshold_f1,
            skeleton_results[best_skeleton_threshold_f1]['f1'],
        ))
    else:
        print('No final skeleton logits; use stage2/3 structure for skeleton quality.')
    
    print('\n' + '='*80)
    print('Threshold sweep complete!')
    print('='*80)


if __name__ == '__main__':
    main()
