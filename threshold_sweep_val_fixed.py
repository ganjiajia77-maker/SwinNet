import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet as ViT_seg
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
                       default='./model_out/train_skeleton_20260521_094935/best.pth')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
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
    
    args = parser.parse_args()
    
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
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print('Model loaded')
    
    print('\nLoading validation dataset')
    val_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split='val',
        image_size=args.img_size,
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
                skeleton_logits = outputs[2]
            else:
                raise RuntimeError("Structure-guided threshold sweep requires auxiliary outputs.")
            
            all_surface_logits.append(surface_logits.cpu())
            all_surface_targets.append(masks.cpu())
            all_skeleton_logits.append(skeleton_logits.cpu())
            all_skeleton_targets.append(batch["skeleton"].cpu())
    
    print('Inference complete')
    
    print('\n' + '='*80)
    print('SURFACE SEGMENTATION - THRESHOLD SWEEP')
    print('='*80)
    print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Threshold', 'IoU', 'F1', 'Precision', 'Recall'))
    print('-'*60)
    
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
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
    print('FINAL SKELETON (256x256) - THRESHOLD SWEEP')
    print('='*80)
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
    
    print('\n' + '='*80)
    print('Threshold sweep complete!')
    print('='*80)


if __name__ == '__main__':
    main()
