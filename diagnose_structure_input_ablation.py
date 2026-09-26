import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_structure_residual_scale_sweep import (
    add_runtime_residual_stats,
    build_parser as build_base_parser,
    decoder_blocks,
    enable_residual_diagnostics,
    metrics,
    update_confusion,
    update_topology_stats,
)

DEFAULT_MODES = ('full', 'no_skeleton', 'no_connectivity', 'gate_feature_only')
MODES = DEFAULT_MODES + ('oracle_gate_skeleton', 'oracle_connectivity_skeleton')


def ablation_blocks(model):
    blocks = dict(decoder_blocks(model))
    module = model.module if hasattr(model, "module") else model
    source = getattr(module.swin_unet, "stage2_topology_source", None)
    if source is not None:
        blocks["stage2_topology_source"] = source
    return blocks

def build_parser():
    parser = build_base_parser()
    parser.description = 'Runtime ablation for structure gate inputs.'
    parser.add_argument('--modes', type=str, default=','.join(DEFAULT_MODES))
    parser.add_argument('--output_csv', type=str, default='./analysis_out/structure_input_ablation.csv')
    parser.add_argument('--enable_global_topology', action='store_true')
    parser.add_argument('--global_topology_max_nodes', type=int, default=32)
    parser.add_argument('--global_topology_heads', type=int, default=4)
    parser.add_argument('--global_topology_alpha_max', type=float, default=0.05)
    return parser

def set_ablation_mode(model, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown mode: {mode}')
    ablate_skeleton = mode in {'no_skeleton', 'gate_feature_only'}
    ablate_connectivity = mode in {'no_connectivity', 'gate_feature_only'}
    for block in ablation_blocks(model).values():
        block.runtime_ablate_skeleton_prior = ablate_skeleton
        block.runtime_ablate_connectivity_gate = ablate_connectivity
        block.runtime_connectivity_skeleton_override = None
        block.runtime_gate_skeleton_override = None

def set_oracle_skeleton_input(model, mode, skeletons):
    if mode not in {'oracle_gate_skeleton', 'oracle_connectivity_skeleton'}:
        return
    attr = (
        'runtime_gate_skeleton_override'
        if mode == 'oracle_gate_skeleton'
        else 'runtime_connectivity_skeleton_override'
    )
    for block in ablation_blocks(model).values():
        setattr(block, attr, skeletons)

def new_stats():
    return {
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
        'cldice_sum': 0.0, 'image_count': 0, 'break_pixels': 0,
        'pred_components': 0, 'gt_components': 0,
        'false_positive_components': 0, 'extra_components': 0,
        'gamma1_sum_stage2': 0.0, 'gamma1_sum_stage3': 0.0,
        'residual_norm_sum_stage2': 0.0, 'residual_norm_sum_stage3': 0.0,
        'runtime_count': 0, 'delta_abs_sum': 0.0, 'delta_count': 0,
        'delta_max': 0.0, 'changed_pixels': 0,
    }

def summarize(mode, stats):
    iou, f1, precision, recall = metrics(stats)
    images = max(stats['image_count'], 1)
    runs = max(stats['runtime_count'], 1)
    row = dict(mode=mode, iou=iou, f1=f1, precision=precision, recall=recall)
    row.update(stats)
    row['cldice'] = stats['cldice_sum'] / images
    row['break_pixels_per_image'] = stats['break_pixels'] / images
    row['false_positive_components_per_image'] = stats['false_positive_components'] / images
    row['extra_components_per_image'] = stats['extra_components'] / images
    row['gamma1_stage2'] = stats['gamma1_sum_stage2'] / runs
    row['gamma1_stage3'] = stats['gamma1_sum_stage3'] / runs
    row['gate_residual_relative_norm_stage2'] = stats['residual_norm_sum_stage2'] / runs
    row['gate_residual_relative_norm_stage3'] = stats['residual_norm_sum_stage3'] / runs
    row['mean_abs_logit_delta_vs_full'] = stats['delta_abs_sum'] / max(stats['delta_count'], 1)
    row['max_abs_logit_delta_vs_full'] = stats['delta_max']
    row['changed_pixel_count_vs_full'] = stats['changed_pixels']
    return row

def main():
    args = build_parser().parse_args()
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    if 'full' not in modes:
        modes = ['full'] + modes
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f'Unknown mode in --modes: {mode}')
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    model.eval()
    enable_residual_diagnostics(model)
    dataset = RoadSkeletonDataset(root_dir=args.root_path, split=args.split, image_size=args.img_size, source_patch_size=args.source_patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    rows = []
    full_logits_cache = []
    print('\nStructure input runtime ablation')
    print('mode                 IoU      F1       Precision  Recall   clDice   break_px  fp_comp  extra_comp  dlogit    dmax      changed')
    with torch.no_grad():
        for mode in modes:
            set_ablation_mode(model, mode)
            stats = new_stats()
            for index, batch in enumerate(tqdm(loader, desc=mode)):
                if args.max_batches and index >= args.max_batches:
                    break
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                skeletons = batch['skeleton'].to(device, non_blocking=True)
                set_oracle_skeleton_input(model, mode, skeletons)
                surface_logits = model(images)[0]
                if mode == 'full':
                    full_logits_cache.append(surface_logits.detach().cpu())
                else:
                    full_logits = full_logits_cache[index].to(device=device, dtype=surface_logits.dtype)
                    delta = surface_logits - full_logits
                    stats['delta_abs_sum'] += float(delta.abs().sum().item())
                    stats['delta_count'] += int(delta.numel())
                    stats['delta_max'] = max(stats['delta_max'], float(delta.abs().max().item()))
                    changed = (torch.sigmoid(surface_logits) >= args.threshold) != (torch.sigmoid(full_logits) >= args.threshold)
                    stats['changed_pixels'] += int(changed.sum().item())
                add_runtime_residual_stats(stats, model)
                update_confusion(stats, surface_logits, masks, args.threshold)
                update_topology_stats(stats, surface_logits, masks, skeletons, args.threshold)
            row = summarize(mode, stats)
            rows.append(row)
            print(mode, row['iou'], row['f1'], row['precision'], row['recall'], row['cldice'], row['break_pixels_per_image'], row['false_positive_components_per_image'], row['mean_abs_logit_delta_vs_full'], row['changed_pixel_count_vs_full'])

    set_ablation_mode(model, 'full')
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    main()
