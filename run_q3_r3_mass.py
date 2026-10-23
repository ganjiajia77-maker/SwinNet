"""Train and evaluate frozen H0 plus one Q3/R3 repair on native Mass images."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from evaluation.road_comparison.road_metrics import DEFAULT_PROTOCOL, image_metrics, prepare_target, summarize
from local_gap_config import EXPERIMENT_PROTOCOL
from mass_protocol import blend_weight, sliding_positions, validate_training_args
from networks.q3_r3_repair import Q3R3Repair, repair_losses
from networks.vision_transformer import SwinUnet


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path('/home/gjj/SwinNet-roadbias/model_out/'
                          'h0_local_gap_mass1500_fp32_noema80_b2_acc1_s1234_20261022_204742/best.pth')
PROTOCOL = dict(DEFAULT_PROTOCOL, image_size=1500)


def config_for_512():
    args = SimpleNamespace(cfg=str(ROOT / 'configs/swin_tiny_patch4_window7_224_lite.yaml'),
                           opts=None, batch_size=2, img_size=512, zip=False, cache_mode=None,
                           resume=None, accumulation_steps=1, use_checkpoint=False,
                           amp_opt_level=None, tag=None, eval=False, throughput=False)
    return get_config(args)


def load_experiment(args):
    checkpoint = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in checkpoint or not isinstance(checkpoint.get('args'), dict):
        raise ValueError('Expected the original H0 model_state_dict and training args')
    if checkpoint.get('experiment_protocol') != EXPERIMENT_PROTOCOL:
        raise ValueError('Base checkpoint is not the 39e1398 Mass local-gap experiment')
    saved = checkpoint['args']
    validate_training_args(saved)
    if checkpoint.get('ema_state_dict') is not None:
        raise ValueError('Expected raw, non-EMA base weights')
    model = SwinUnet(
        config=config_for_512(), img_size=512, num_classes=1, return_skeleton=True,
        bottleneck_type=saved.get('bottleneck_type', 'global_local'),
        final_topology_eta_init=saved.get('final_topology_eta_init', 0.005),
        final_gap_rho_init=saved.get('final_gap_rho_init', 0.005),
        structure_profile=saved['structure_profile'],
        stage2_skeleton_gradient_ratio=saved['stage2_skeleton_gradient_ratio'],
        stage3_skeleton_gradient_ratio=saved['stage3_skeleton_gradient_ratio'],
        stage3_gate_topology_gradient_ratio=saved['stage3_gate_topology_gradient_ratio'],
        final_skeleton_gradient_ratio=saved.get('final_skeleton_gradient_ratio', 0.0),
        enable_highres_structure_stream=True, highres_structure_channels=64,
        highres_structure_fuse_stages='stage23', highres_structure_fusion_mode='stage23',
        enable_post_refine_structure_interaction=saved.get('enable_post_refine_structure_interaction', False),
        enable_h3_surface_fusion=True, enable_global_topology=False,
        global_topology_max_nodes=saved.get('global_topology_max_nodes', 32),
        global_topology_heads=saved.get('global_topology_heads', 4),
        global_topology_alpha_max=saved.get('global_topology_alpha_max', 0.05),
        stage_skeleton_mode='direct',
        stage_skeleton_bias_init=saved.get('stage_skeleton_bias_init', 'zero'),
        stage_skeleton_positive_prior=saved.get('stage_skeleton_positive_prior', 0.05),
        remove_stage2_pre_topology_source=True, local_gap_config=saved['local_gap_config'],
        enable_stage2_local_propagation=args.enable_stage2_local_propagation,
        enable_gap_refinement=args.enable_gap_refinement,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f'[LOAD] strict checkpoint: {len(model.state_dict())} tensors, '
          'missing=0, unexpected=0, shape_mismatch=0', flush=True)
    repair = Q3R3Repair(model, eta=args.eta,
                        enable_stage2_local_propagation=args.enable_stage2_local_propagation,
                        enable_gap_refinement=args.enable_gap_refinement).to(args.device)
    frozen = sum(p.numel() for p in repair.base.parameters())
    trainable = sum(p.numel() for p in repair.parameters() if p.requires_grad)
    print(f'[PARAMS] frozen={frozen:,} trainable_Q3_R3={trainable:,}', flush=True)
    if args.repair_checkpoint:
        saved_repair = torch.load(args.repair_checkpoint, map_location='cpu', weights_only=False)
        if Path(saved_repair['base_checkpoint']) != args.base_checkpoint.resolve():
            raise ValueError('Repair checkpoint references a different base checkpoint')
        for name in ('q_features', 'q_head', 'r_features', 'r_head'):
            getattr(repair, name).load_state_dict(saved_repair[name], strict=True)
    return repair


def save_repair(model, path, base_checkpoint, epoch, val_iou):
    state = {'base_checkpoint': str(base_checkpoint.resolve()), 'epoch': epoch,
             'val_iou': val_iou, 'eta': model.eta}
    for name in ('q_features', 'q_head', 'r_features', 'r_head'):
        state[name] = getattr(model, name).state_dict()
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def stitch_three(model, image):
    if tuple(image.shape) != (1, 3, 1500, 1500):
        raise ValueError(f'Expected one native 1500 image, got {tuple(image.shape)}')
    weight = blend_weight(512, image.device)
    canvases = [torch.zeros((1, 1, 1500, 1500), device=image.device) for _ in range(4)]
    positions = sliding_positions(1500, 512, 256)
    for top in positions:
        for left in positions:
            result = model(image[:, :, top:top+512, left:left+512])
            q = F.interpolate(result['q'], size=(512, 512), mode='bilinear', align_corners=False)
            region = (slice(None), slice(None), slice(top, top+512), slice(left, left+512))
            for canvas, tile in zip(canvases[:3], (result['z0'], result['z1'], q)):
                canvas[region] += tile.float() * weight
            canvases[3][region] += weight
    if bool((canvases[3] <= 0).any()):
        raise RuntimeError('Uncovered pixels in the 512/256 stitch')
    return tuple((canvas / canvases[3]).cpu().numpy()[0, 0] for canvas in canvases[:3])


def full_dataset(root, split):
    return RoadSkeletonDataset(str(root), split, image_size=None, source_patch_size=1500,
                               return_full_image=True)


def val_iou(model, root, device, max_images=0):
    model.eval()
    dataset = full_dataset(root, 'val')
    total_tp = total_fp = total_fn = 0
    for index in range(min(len(dataset), max_images or len(dataset))):
        sample = dataset[index]
        _, z1, _ = stitch_three(model, sample['image'][None].to(device))
        pred = z1 >= 0
        gt = sample['mask'][0].numpy() > 0.5
        total_tp += int((pred & gt).sum())
        total_fp += int((pred & ~gt).sum())
        total_fn += int((~pred & gt).sum())
    return total_tp / max(total_tp + total_fp + total_fn, 1)


def train(args, model):
    if args.repair_checkpoint:
        raise ValueError('Training resumes are not enabled; start with base checkpoint only')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
                                  source_patch_size=1500, random_crop_train=True,
                                  random_crops_per_image=1, random_crop_seed=args.seed,
                                  augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0.01)
    best = -1.
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        loss_sum = q_fraction = q_high = 0.
        for batch in loader:
            image = batch['image'].to(args.device, non_blocking=True)
            mask = batch['mask'].to(args.device, non_blocking=True)
            skeleton = batch['skeleton'].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = model(image)
            loss, stats = repair_losses(result, mask, skeleton, lambda_q=args.lambda_q)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Nonfinite repair loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            q_fraction += float(stats['q_positive_fraction'])
            q_high += float(stats['q_high_fraction'])
        mean_loss = loss_sum / len(loader)
        print(f'[TRAIN] epoch={epoch} loss={mean_loss:.6f} '
              f'q_pos={q_fraction/len(loader):.6f} q_high={q_high/len(loader):.6f}', flush=True)
        save_repair(model, args.output_dir / 'last.pth', args.base_checkpoint, epoch, None)
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            score = val_iou(model, args.root_path, args.device, args.max_images)
            print(f'[VAL] epoch={epoch} IoU@0.50={score:.6f}', flush=True)
            if score > best:
                best = score
                save_repair(model, args.output_dir / 'best.pth', args.base_checkpoint, epoch, score)


def gap_proxy(p0, p1, gt_skeleton, gt_mask):
    """Count P0 two-flank missed segments closed by P1; not vector-road truth."""
    kernel = np.ones((5, 5), np.uint8)
    near_p0 = cv2.dilate(p0.astype(np.uint8), kernel) > 0
    missing = (gt_skeleton & ~near_p0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, connectivity=8)
    _, p0_labels = cv2.connectedComponents(p0.astype(np.uint8), connectivity=8)
    _, p1_labels = cv2.connectedComponents(p1.astype(np.uint8), connectivity=8)
    eligible = closed = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        region = labels == index
        flank = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        flank_labels = np.unique(p0_labels[flank & gt_mask])
        flank_labels = flank_labels[flank_labels > 0]
        if len(flank_labels) < 2:
            continue
        eligible += 1
        if p1[region].mean() >= 0.8:
            linked = [set(np.unique(p1_labels[flank & (p0_labels == label) & gt_mask])) - {0}
                      for label in flank_labels]
            closed += int(bool(set.intersection(*linked)))
    new_background = (p1 & ~p0 & ~cv2.dilate(gt_mask.astype(np.uint8),
                                               np.ones((7, 7), np.uint8)).astype(bool))
    bridge_count = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(new_background.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        near = cv2.dilate((labels == index).astype(np.uint8), kernel) > 0
        sources = np.unique(p0_labels[near])
        bridge_count += int((sources > 0).sum() >= 2)
    return eligible, closed, bridge_count


def evaluate(args, model):
    model.eval()
    dataset = full_dataset(args.root_path, args.mode)
    limit = min(len(dataset), args.max_images or len(dataset))
    if args.mode == 'val':
        thresholds = np.arange(0.05, 1., 0.05)
        counts = np.zeros((len(thresholds), 3), dtype=np.int64)
        for index in range(limit):
            sample = dataset[index]
            _, z1, _ = stitch_three(model, sample['image'][None].to(args.device))
            gt = sample['mask'][0].numpy() > 0.5
            prob = 1 / (1 + np.exp(-np.clip(z1, -80, 80)))
            for row, threshold in zip(counts, thresholds):
                pred = prob >= threshold
                row += ((pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum())
            print(f'[SWEEP] {index+1}/{limit}', flush=True)
        iou = counts[:, 0] / np.maximum(counts.sum(axis=1), 1)
        choice = int(np.argmax(iou))
        threshold = float(round(thresholds[choice], 2))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selection = {'threshold': threshold, 'selected_on': 'val', 'images': limit,
                     'repair_checkpoint': str(args.repair_checkpoint.resolve()),
                     'val_iou': float(iou[choice])}
        (args.output_dir / 'val_selection.json').write_text(json.dumps(selection, indent=2))
        print(f'[SWEEP] selected threshold={threshold:.2f} IoU={iou[choice]:.6f}', flush=True)
    else:
        selection = json.loads((args.output_dir / 'val_selection.json').read_text())
        if (selection['selected_on'] != 'val' or
                selection['repair_checkpoint'] != str(args.repair_checkpoint.resolve())):
            raise ValueError('Selection must come from val with the same repair checkpoint')
        threshold = selection['threshold']
    rows = {'p0': [], 'p1': []}
    extras = []
    for index in range(limit):
        sample = dataset[index]
        z0, z1, q = stitch_three(model, sample['image'][None].to(args.device))
        gt = sample['mask'][0].numpy() > 0.5
        prepared = prepare_target(gt, PROTOCOL)
        p0_prob = 1 / (1 + np.exp(-np.clip(z0, -80, 80)))
        p1_prob = 1 / (1 + np.exp(-np.clip(z1, -80, 80)))
        p0, p1 = p0_prob >= threshold, p1_prob >= threshold
        for name, mask in (('p0', p0), ('p1', p1)):
            row = image_metrics(mask, prepared, PROTOCOL)
            row['case_id'] = sample['case_name']
            rows[name].append(row)
        missing = prepared['skeleton'] & ~(cv2.dilate(p0.astype(np.uint8),
                                                       np.ones((5, 5), np.uint8)) > 0)
        eligible, closed, false_bridge = gap_proxy(p0, p1, prepared['skeleton'], gt)
        extras.append({'case_id': sample['case_name'], 'q_missing_recall':
                       float((q[missing] >= 0.5).mean()) if missing.any() else 1.,
                       'q_high_area_fraction': float((q >= 0.5).mean()),
                       'mean_probability_change': float(np.abs(p1_prob-p0_prob).mean()),
                       'binary_change_fraction': float((p0 != p1).mean()),
                       'probability_increase_mean': float(np.maximum(p1_prob-p0_prob, 0).mean()),
                       'probability_decrease_mean': float(np.maximum(p0_prob-p1_prob, 0).mean()),
                       'added_gt_pixels': int((p1 & ~p0 & gt).sum()),
                       'added_background_pixels': int((p1 & ~p0 & ~gt).sum()),
                       'removed_gt_pixels': int((p0 & ~p1 & gt).sum()),
                       'removed_background_pixels': int((p0 & ~p1 & ~gt).sum()),
                       'eligible_p0_gaps': eligible, 'closed_p0_gaps': closed,
                       'background_bridge_proxy': false_bridge})
        print(f'[{args.mode.upper()}] {index+1}/{limit}', flush=True)
    report = {'split': args.mode, 'threshold': threshold, 'images': limit,
              'p0': summarize(rows['p0']), 'p1': summarize(rows['p1']),
              'paired_diagnostics': {
                  'q_missing_recall': float(np.mean([r['q_missing_recall'] for r in extras])),
                  'q_high_area_fraction': float(np.mean([r['q_high_area_fraction'] for r in extras])),
                  'mean_probability_change': float(np.mean([r['mean_probability_change'] for r in extras])),
                  'binary_change_fraction': float(np.mean([r['binary_change_fraction'] for r in extras])),
                  'probability_increase_mean': float(np.mean([r['probability_increase_mean'] for r in extras])),
                  'probability_decrease_mean': float(np.mean([r['probability_decrease_mean'] for r in extras])),
                  'added_gt_pixels': sum(r['added_gt_pixels'] for r in extras),
                  'added_background_pixels': sum(r['added_background_pixels'] for r in extras),
                  'removed_gt_pixels': sum(r['removed_gt_pixels'] for r in extras),
                  'removed_background_pixels': sum(r['removed_background_pixels'] for r in extras),
                  'p0_gt_skeleton_uncovered_rate': rows['p0'][0]['break_rate'],
                  'p1_gt_skeleton_uncovered_rate': rows['p1'][0]['break_rate'],
                  'eligible_p0_gaps': sum(r['eligible_p0_gaps'] for r in extras),
                  'closed_p0_gaps': sum(r['closed_p0_gaps'] for r in extras),
                  'gap_closure_rate_proxy': sum(r['closed_p0_gaps'] for r in extras) /
                                            max(sum(r['eligible_p0_gaps'] for r in extras), 1),
                  'background_bridge_proxy': sum(r['background_bridge_proxy'] for r in extras)},
              'note': 'APLS is the shared raster approximation. Gap closure and background bridges '
                      'are component-based diagnostic proxies, not official graph edge labels.'}
    output = args.output_dir / args.mode
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    for name, data in (('p0', rows['p0']), ('p1', rows['p1']), ('paired', extras)):
        with (output / f'{name}_per_image.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(report, indent=2), flush=True)


def smoke(args, model):
    model.train()
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
                                  source_patch_size=1500, random_crop_train=True)
    sample = dataset[0]
    image = sample['image'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    mask = sample['mask'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    skeleton = sample['skeleton'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    calls = {'local': 0, 'gap': 0, 'p2': 0}
    def local_hook(*unused):
        calls['local'] += 1
    def gap_hook(*unused):
        calls['gap'] += 1
    def p2_hook(*unused):
        calls['p2'] += 1
    local = model.base.swin_unet.decoder_structure_blocks['2'].local_propagation
    handles = [local.register_forward_hook(local_hook),
               model.base.swin_unet.gap_connection.register_forward_hook(gap_hook),
               model.base.swin_unet.decoder_structure_blocks['2'].p2_head.register_forward_hook(p2_hook)]
    try:
        output = model(image)
        loss, stats = repair_losses(output, mask, skeleton, args.lambda_q)
        loss.backward()
        initial_delta = float((output['z1'] - output['z0']).detach().abs().sum())
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        surface_only, _ = repair_losses(output, mask, skeleton, lambda_q=0.)
        surface_only.backward()
    finally:
        for handle in handles:
            handle.remove()
    if calls != {'local': 0, 'gap': 0, 'p2': 0}:
        raise RuntimeError(f'Disabled propagation/gap/P2 path still ran: {calls}')
    if any(parameter.grad is not None for parameter in model.base.parameters()):
        raise RuntimeError('Frozen base received gradients')
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite loss')
    gradients = {name: float(sum(p.grad.abs().sum() for p in module.parameters()
                                 if p.grad is not None))
                 for name, module in (('q_features', model.q_features), ('q_head', model.q_head),
                                      ('r_features', model.r_features), ('r_head', model.r_head))}
    if gradients['q_head'] <= 0 or gradients['r_head'] <= 0 or gradients['r_features'] <= 0:
        raise RuntimeError(f'Q3/R3 head gradients are zero: {gradients}')
    report = {'shapes': {k: list(v.shape) for k, v in output.items() if torch.is_tensor(v)},
              'loss': float(loss.detach()), 'gradients': gradients,
              'initial_delta_l1': initial_delta,
              'learned_delta_l1_after_one_step': float((output['z1'] - output['z0']).detach().abs().sum()),
              'disabled_calls': calls, 'q_positive_fraction': float(stats['q_positive_fraction']),
              'gpu_peak_mib': torch.cuda.max_memory_allocated(args.device) / 2**20
              if args.device.type == 'cuda' else None}
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'train', 'val', 'test'))
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--repair_checkpoint', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--lambda_q', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--disable_q3_r3', action='store_true',
                        help='run the frozen P0 baseline for a paired evaluation')
    parser.add_argument('--enable_stage2_local_propagation', action='store_true')
    parser.add_argument('--enable_gap_refinement', action='store_true')
    parser.add_argument('--max_images', type=int, default=0, help='0 uses the entire split')
    args = parser.parse_args()
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.mode in ('val', 'test') and not args.repair_checkpoint:
        parser.error('--repair_checkpoint is required for val/test')
    if args.mode == 'train' and args.disable_q3_r3:
        parser.error('Q3/R3 cannot be disabled during its training run')
    model = load_experiment(args)
    model.enable_q3_r3 = not args.disable_q3_r3
    if args.mode == 'smoke':
        smoke(args, model)
    elif args.mode == 'train':
        train(args, model)
    else:
        evaluate(args, model)


if __name__ == '__main__':
    main()
