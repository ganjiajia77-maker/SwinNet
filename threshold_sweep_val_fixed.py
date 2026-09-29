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
    SwinUnet as ViTStandard,
    load_topology_checkpoint_state as load_topology_checkpoint_state_standard,
    print_topology_coefficients as print_topology_coefficients_standard,
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE,
)
from networks.vision_transformer_selective_fusion import (
    SwinUnet as ViTSelective,
    load_topology_checkpoint_state as load_topology_checkpoint_state_selective,
    print_topology_coefficients as print_topology_coefficients_selective,
)


def _cli_has(flag_name):
    flag = "--" + flag_name
    negative_flag = "--no-" + flag_name
    return any(
        argument == flag
        or argument == negative_flag
        or argument.startswith(flag + "=")
        for argument in sys.argv[1:]
    )
from losses.road_losses import binary_metrics_from_logits
from config import get_config
from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint


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


def select_skeleton_logits(outputs):
    if not isinstance(outputs, tuple):
        return None, "none"

    if len(outputs) > 2 and outputs[2] is not None:
        return outputs[2], "final"

    structure_outputs = outputs[-1] if outputs and isinstance(outputs[-1], list) else []
    for item in reversed(structure_outputs):
        if not isinstance(item, dict):
            continue
        if item.get("highres_structure_skeleton") is not None:
            return item["highres_structure_skeleton"], "highres_structure"
        if item.get("skeleton") is not None:
            return item["skeleton"], "stage{}".format(item.get("stage", "unknown"))

    return None, "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./data1')
    parser.add_argument('--model_path', type=str, 
                       default='./model_out/train_skeleton_20260521_094935/best.pth')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--crop_list', type=str, default='', help='fixed crop list for the selected split')
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
    parser.add_argument('--stage2_skeleton_gradient_ratio', type=float, default=0.5)
    parser.add_argument('--stage3_skeleton_gradient_ratio', type=float, default=0.5)
    parser.add_argument('--stage3_gate_topology_gradient_ratio', type=float, default=0.0)
    parser.add_argument('--final_skeleton_gradient_ratio', type=float, default=0.0)
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
    parser.add_argument(
        '--model_impl',
        type=str,
        default='standard',
        choices=['standard', 'selective'],
        help='model implementation used by the checkpoint',
    )
    parser.add_argument(
        '--bottleneck_type',
        type=str,
        default='global_local',
        choices=['global_local', 'legacy_global_local', 'g2l2'],
    )
    parser.add_argument(
        '--structure_profile',
        type=str,
        default=STRUCTURE_PROFILE_FULL,
        choices=[
            STRUCTURE_PROFILE_FULL,
            STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
            STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE,
        ],
    )
    parser.add_argument(
        '--disable_msfe_skip',
        action='store_true',
        help='ablate MSFE blocks on decoder skip stages inx=2,3; auto-read from checkpoint when omitted',
    )
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
    parser.add_argument('--enable_post_refine_structure_interaction', action='store_true')
    parser.add_argument('--enable_h3_surface_fusion', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--enable_global_topology', action='store_true')
    parser.add_argument('--global_topology_max_nodes', type=int, default=32)
    parser.add_argument('--global_topology_heads', type=int, default=4)
    parser.add_argument('--global_topology_alpha_max', type=float, default=0.05)
    parser.add_argument('--stage_skeleton_mode', type=str, default=None, choices=['direct', 'prior_residual'])
    parser.add_argument('--enable_e128_stage_fusion', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--enable_coarse_road_mask', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--enable_psi_directional_descriptor', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--enable_sparse_window_compute', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--remove_stage2_pre_topology_source', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--stage2_window_threshold', type=float, default=None)
    parser.add_argument('--stage3_window_threshold', type=float, default=None)
    parser.add_argument('--coarse_candidate_window_size', type=int, default=None)
    parser.add_argument('--coarse_corridor_window_radius', type=int, default=None)
    parser.add_argument('--coarse_routing_mode', type=str, default=None, choices=['dense', 'p64', 'bottleneck', 'bottleneck_no_psi'])
    parser.add_argument('--bottleneck_coarse_road_mask', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--bottleneck_window_threshold', type=float, default=None)
    args = parser.parse_args()
    
    checkpoint = None
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            saved_profile = checkpoint.get("structure_profile")
            if saved_profile and not _cli_has("structure_profile"):
                args.structure_profile = saved_profile
            elif isinstance(checkpoint.get("args"), dict) and not _cli_has("structure_profile"):
                args.structure_profile = checkpoint["args"].get(
                    "structure_profile",
                    args.structure_profile,
                )
            if isinstance(checkpoint.get("args"), dict):
                saved_args = checkpoint["args"]
                if "disable_msfe_skip" in saved_args and not _cli_has("disable_msfe_skip"):
                    args.disable_msfe_skip = bool(saved_args["disable_msfe_skip"])
                for name in (
                    "stage_topology_stages",
                    "stage_topology_alpha_max",
                    "stage_topology_alpha_init",
                    "stage2_skeleton_gradient_ratio",
                    "stage3_skeleton_gradient_ratio",
                    "stage3_gate_topology_gradient_ratio",
                    "final_skeleton_gradient_ratio",
                    "enable_highres_structure_stream",
                    "highres_structure_channels",
                    "highres_structure_fuse_stages",
                    "highres_structure_fusion_mode",
                    "enable_post_refine_structure_interaction",
                    "enable_h3_surface_fusion",
                    "enable_global_topology",
                    "global_topology_max_nodes",
                    "global_topology_heads",
                    "global_topology_alpha_max",
                    "bottleneck_type",
                    "stage_skeleton_mode",
                    "enable_e128_stage_fusion",
                    "enable_coarse_road_mask",
                    "enable_psi_directional_descriptor",
                    "enable_sparse_window_compute",
                    "remove_stage2_pre_topology_source",
                    "stage2_window_threshold",
                    "stage3_window_threshold",
                    "coarse_candidate_window_size",
                    "coarse_corridor_window_radius",
                    "coarse_routing_mode",
                    "bottleneck_coarse_road_mask",
                    "bottleneck_window_threshold",
                ):
                    if name in saved_args and not _cli_has(name):
                        setattr(args, name, saved_args[name])

    for name, default in (
        ("stage_skeleton_mode", "prior_residual"),
        ("enable_e128_stage_fusion", False),
        ("enable_h3_surface_fusion", False),
        ("enable_coarse_road_mask", False),
        ("enable_psi_directional_descriptor", False),
        ("enable_sparse_window_compute", False),
        ("remove_stage2_pre_topology_source", False),
        ("stage2_window_threshold", 0.10),
        ("stage3_window_threshold", 0.10),
        ("coarse_candidate_window_size", 8),
        ("coarse_corridor_window_radius", 0),
        ("coarse_routing_mode", "dense"),
        ("bottleneck_coarse_road_mask", False),
        ("bottleneck_window_threshold", 0.25),
    ):
        if getattr(args, name) is None:
            setattr(args, name, default)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inference_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }.get(args.amp_dtype)
    use_inference_amp = device.type == 'cuda' and inference_dtype is not None
    print('Using device: {}'.format(device))
    
    print('\nLoading model from: {}'.format(args.model_path))
    config = get_config(args)
    if args.model_impl == 'selective':
        vit_cls = ViTSelective
        loader = load_topology_checkpoint_state_selective
        printer = print_topology_coefficients_selective
    else:
        vit_cls = ViTStandard
        loader = load_topology_checkpoint_state_standard
        printer = print_topology_coefficients_standard

    model_kwargs = dict(
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
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
        enable_h3_surface_fusion=args.enable_h3_surface_fusion,
    )
    if args.model_impl == 'standard':
        model_kwargs.update(
            stage3_gate_topology_gradient_ratio=args.stage3_gate_topology_gradient_ratio,
            enable_global_topology=args.enable_global_topology,
            global_topology_max_nodes=args.global_topology_max_nodes,
            global_topology_heads=args.global_topology_heads,
            global_topology_alpha_max=args.global_topology_alpha_max,
            stage_skeleton_mode=args.stage_skeleton_mode,
            enable_e128_stage_fusion=args.enable_e128_stage_fusion,
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
    model = vit_cls(**model_kwargs)
    if checkpoint is None:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    adapt_connectivity_modules_for_checkpoint(
        model,
        checkpoint['model_state_dict'],
        args.model_impl,
    )
    loader(
        model,
        checkpoint['model_state_dict'],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    model = model.to(device)
    model.eval()
    print('Model loaded')
    print("Using topology attention constrained version", flush=True)
    printer(model)
    
    print(f'\nLoading {args.split} dataset')
    val_dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.crop_list,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print('{} set size: {}'.format(args.split.capitalize(), len(val_dataset)))
    
    print(f'\nRunning inference on {args.split} set')
    all_surface_logits = []
    all_surface_targets = []
    all_skeleton_logits = []
    all_skeleton_targets = []
    skeleton_source = None
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Inference'):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            with torch.autocast(
                device_type=device.type,
                dtype=inference_dtype if inference_dtype is not None else torch.float32,
                enabled=use_inference_amp,
            ):
                outputs = model(images)
            
            if isinstance(outputs, tuple):
                surface_logits = outputs[0]
                skeleton_logits, current_skeleton_source = select_skeleton_logits(outputs)
                if skeleton_logits is not None and skeleton_source is None:
                    skeleton_source = current_skeleton_source
            else:
                raise RuntimeError("Structure-guided threshold sweep requires auxiliary outputs.")
            
            all_surface_logits.append(surface_logits.cpu())
            all_surface_targets.append(masks.cpu())
            if skeleton_logits is not None:
                all_skeleton_logits.append(skeleton_logits.cpu())
                all_skeleton_targets.append(batch["skeleton"].cpu())
    
    print('Inference complete')
    if skeleton_source is not None:
        print('Skeleton logits source: {}'.format(skeleton_source))
    
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
        print('SKELETON ({}) - THRESHOLD SWEEP'.format(skeleton_source))
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
