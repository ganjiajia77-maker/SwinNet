"""Read-only E2 direction audit for the a4001b3 H0 architecture.

The paired run disables only the E2 aggregation output of the SAME checkpoint.
It measures checkpoint dependence, not the benefit of retraining without E2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation' / 'road_comparison'))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from evaluation.road_comparison import road_metrics
from compare_predictions import digest, model_identity, validate_manifest
from h0_e2_config import validate_checkpoint
from h0_encoder_protocol import MODEL_NAME, SOURCE_SIZE, STRIDE, TILE_SIZE, blend_weight, sliding_positions
from losses.encoder_direction_loss import direction_target, encoder_direction_loss
from losses.road_losses import BCEDiceLoss
from networks.vision_transformer import SwinUnet


def summarize_log(path: Path):
    epochs = defaultdict(list)
    malformed = 0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or 'epoch' not in row:
                    raise ValueError('missing epoch')
                epochs[int(row['epoch'])].append(row)
            except (ValueError, TypeError):
                malformed += 1
    fields = ('gamma', 'q_mean', 'q_std', 'delta_abs_mean', 'delta_relative_abs',
              'encoder_direction_raw', 'encoder_direction_weighted',
              'encoder_direction_valid', 'D_grad_norm_sum', 'q_grad_norm_sum',
              'projection_grad_norm_sum')
    result = []
    for epoch, rows in sorted(epochs.items()):
        item = {'epoch': epoch, 'batches': len(rows)}
        for key in fields:
            values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
            item[key + '_mean'] = float(np.mean(values)) if values else None
            if key.endswith('grad_norm_sum'):
                item[key + '_nonzero_fraction'] = sum(v > 0 for v in values) / len(values) if values else None
        result.append(item)
    return {'source': str(path), 'epochs': result, 'malformed_lines': malformed,
            'interpretation': 'Gradient magnitudes alone cannot establish alignment or conflict with surface loss.'}


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = validate_checkpoint(checkpoint)
    args = SimpleNamespace(**saved)
    args.cfg = str(ROOT / 'configs' / 'swin_tiny_patch4_window7_224_lite.yaml')
    args.opts = None
    config = get_config(args)
    model = SwinUnet(
        config=config, img_size=args.img_size, num_classes=1, return_skeleton=True,
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
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
        enable_global_topology=args.enable_global_topology,
        global_topology_max_nodes=args.global_topology_max_nodes,
        global_topology_heads=args.global_topology_heads,
        global_topology_alpha_max=args.global_topology_alpha_max,
        stage_skeleton_mode=args.stage_skeleton_mode,
        enable_h3_surface_fusion=args.enable_h3_surface_fusion,
        stage_skeleton_bias_init=args.stage_skeleton_bias_init,
        stage_skeleton_positive_prior=args.stage_skeleton_positive_prior,
        remove_stage2_pre_topology_source=args.remove_stage2_pre_topology_source,
        encoder_structure_exchange=args.encoder_structure_exchange,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    return model.to(device).eval(), saved


def load_threshold(selection: Path, checkpoint: Path, root_path: Path):
    data = json.loads(selection.read_text(encoding='utf-8'))
    if data.get('selected_on') != 'val' or MODEL_NAME not in data.get('models', {}):
        raise ValueError('Use a validation selection for this architecture')
    candidates = (selection.parent.parent / 'comparison.json',
                  selection.parent / 'comparison.json')
    manifest_path = next((path for path in candidates if path.is_file()), candidates[0])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol = validate_manifest(manifest)
    if data.get('protocol_hash') != digest(protocol):
        raise ValueError('Validation selection metric protocol differs from this manifest')
    if protocol['image_size'] != SOURCE_SIZE or any(
            protocol[key] != road_metrics.DEFAULT_PROTOCOL[key] for key in
            ('short_area_threshold', 'apls_max_nodes', 'apls_snap_radius')):
        raise ValueError('Validation selection does not use the unified native1500 topology protocol')
    models = [model for model in manifest['models'] if model['name'] == MODEL_NAME]
    if len(models) != 1 or data['models'][MODEL_NAME].get('identity') != model_identity(models[0]):
        raise ValueError('Validation selection does not match the model provenance')
    if (Path(models[0]['provenance']['checkpoint']).resolve() != checkpoint.resolve()
            or Path(manifest['data_root']).resolve() != root_path.resolve()):
        raise ValueError('Validation selection checkpoint or data root differs from this diagnostic')
    if protocol['image_size'] != SOURCE_SIZE:
        raise ValueError('Expected native1500 validation protocol')
    threshold = float(data['models'][MODEL_NAME]['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Invalid validation threshold')
    return threshold


def read_native(dataset: RoadSkeletonDataset, image_name: str):
    label_name = dataset._find_label_name(image_name, dataset.mask_dir)
    rgb = cv2.imread(str(Path(dataset.image_dir) / image_name), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(Path(dataset.mask_dir) / label_name), cv2.IMREAD_GRAYSCALE)
    if rgb is None or gt is None:
        raise FileNotFoundError(f'Cannot read image/GT for {image_name}')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = dataset._center_crop_or_pad(rgb, (SOURCE_SIZE, SOURCE_SIZE))
    gt = dataset._center_crop_or_pad(gt, (SOURCE_SIZE, SOURCE_SIZE)) > 127
    if rgb.shape[:2] != (SOURCE_SIZE, SOURCE_SIZE) or gt.shape != (SOURCE_SIZE, SOURCE_SIZE):
        raise ValueError(f'Expected native {SOURCE_SIZE} image: {image_name}')
    skeleton = dataset._skeletonize_binary(gt.astype(np.uint8)) > 127
    image = rgb.astype(np.float32) / 255.0
    image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    image = torch.from_numpy(image.transpose(2, 0, 1).copy())[None]
    return image, gt, skeleton


class Moment:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def add(self, values):
        values = values.detach().float()
        self.n += values.numel()
        self.sum += float(values.sum())
        self.sumsq += float(values.square().sum())

    def report(self):
        mean = self.sum / self.n if self.n else None
        return {'count': self.n, 'mean': mean,
                'std': math.sqrt(max(self.sumsq / self.n - mean * mean, 0)) if self.n else None}


def gradient_audit(model, image, road, skeleton, direction_cfg):
    """One frozen-checkpoint crop; compare surface and D gradients on shared E2 input."""
    feature_box = []
    module = model.swin_unet.e2_direction
    hook = module.register_forward_pre_hook(lambda _module, inputs: feature_box.append(inputs[0]))
    try:
        with torch.enable_grad():
            outputs = model(image)
            if len(feature_box) != 1:
                raise RuntimeError('E2 aggregation was not called exactly once')
            feature = feature_box[0]
            raw = outputs[5][1]['encoder_direction_raw']
            surface = BCEDiceLoss(dice_weight=.5, focal_gamma=1.0)(outputs[0], road)[0]
            d_loss, valid, _ = encoder_direction_loss(raw, skeleton, road, direction_cfg)
            if not int(valid):
                return {'valid_direction_pixels': 0, 'surface_loss': float(surface.detach()),
                        'direction_loss': 0.0, 'gradient_cosine': None}
            surface_grad = torch.autograd.grad(surface, feature, retain_graph=True)[0]
            direction_grad = torch.autograd.grad(direction_cfg['lambda_encoder_direction'] * d_loss,
                                                feature)[0]
            a, b = surface_grad.float().flatten(), direction_grad.float().flatten()
            a_norm, b_norm = a.norm(), b.norm()
            return {'valid_direction_pixels': int(valid),
                    'surface_loss': float(surface.detach()), 'direction_loss': float(d_loss.detach()),
                    'surface_e2_grad_norm': float(a_norm), 'weighted_D_e2_grad_norm': float(b_norm),
                    'weighted_D_to_surface_grad_ratio': float(b_norm / a_norm.clamp_min(1e-12)),
                    'gradient_cosine': float(F.cosine_similarity(a[None], b[None]))
                    if a_norm > 0 and b_norm > 0 else None}
    finally:
        hook.remove()
        feature_box.clear()


def diagnose(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new diagnostic directory: {args.output_dir}')
    threshold = load_threshold(args.selection, args.checkpoint, args.root_path)
    device = torch.device(args.device)
    model, saved = load_model(args.checkpoint, device)
    module = model.swin_unet.e2_direction
    dataset = RoadSkeletonDataset(args.root_path, split=args.split, image_size=TILE_SIZE,
                                  source_patch_size=SOURCE_SIZE)
    images = dataset.image_files[:args.max_images or None]
    if not images:
        raise ValueError('No images to diagnose')
    args.output_dir.mkdir(parents=True)
    for name in ('direction_on', 'direction_off'):
        (args.output_dir / name).mkdir()
    model_hash = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            model_hash.update(chunk)
    stats = {key: Moment() for key in ('q_road', 'q_background', 'delta_road', 'delta_background')}
    angle_sum = angle_weight = direction_valid = relative_sum = 0.0
    n_windows = 0
    q_hist = {'road': np.zeros(10, np.int64), 'background': np.zeros(10, np.int64)}
    rows = {'direction_on': [], 'direction_off': []}
    paired = []
    gradients = []
    weight = blend_weight(TILE_SIZE, device)
    gradient_image_indices = set(np.linspace(
        0, len(images) - 1, min(args.gradient_samples, len(images)), dtype=int).tolist()
    ) if args.gradient_samples else set()
    window_rows = sliding_positions(SOURCE_SIZE, TILE_SIZE, STRIDE)
    gradient_position = (window_rows[len(window_rows) // 2], window_rows[len(window_rows) // 2])
    direction_cfg = {key: saved[key] for key in ('pca_radius', 'min_support', 'road_radius',
                                                 'min_coherence', 'lambda_encoder_direction')}
    for index, image_name in enumerate(images, 1):
        native_image, gt, skeleton = read_native(dataset, image_name)
        prepared = road_metrics.prepare_target(gt, {**road_metrics.DEFAULT_PROTOCOL,
            'image_size': SOURCE_SIZE})
        canvases = {name: torch.zeros((1, 1, SOURCE_SIZE, SOURCE_SIZE), device=device)
                    for name in rows}
        divisor = torch.zeros_like(canvases['direction_on'])
        for top in sliding_positions(SOURCE_SIZE, TILE_SIZE, STRIDE):
            for left in sliding_positions(SOURCE_SIZE, TILE_SIZE, STRIDE):
                crop = native_image[:, :, top:top + TILE_SIZE, left:left + TILE_SIZE].to(device)
                road = torch.from_numpy(gt[top:top + TILE_SIZE, left:left + TILE_SIZE].copy()).to(device)[None, None].float()
                skel = torch.from_numpy(skeleton[top:top + TILE_SIZE, left:left + TILE_SIZE].copy()).to(device)[None, None].float()
                captured = []

                def capture(_module, inputs, output):
                    feature, updated = inputs[0], output[0]
                    captured.append((feature, updated, output[1]))

                hook = module.register_forward_hook(capture)
                try:
                    with torch.no_grad():
                        on = model(crop)[0].float()
                finally:
                    hook.remove()
                if len(captured) != 1:
                    raise RuntimeError('E2 aggregation was not called exactly once')
                feature, updated, raw = captured.pop()
                with torch.no_grad():
                    q = module.gate_head(feature).sigmoid()
                    road_cells = F.adaptive_avg_pool2d(road, q.shape[-2:]) >= .5
                    delta = (updated - feature).abs().mean(1, keepdim=True)
                    stats['q_road'].add(q[road_cells])
                    stats['q_background'].add(q[~road_cells])
                    stats['delta_road'].add(delta[road_cells])
                    stats['delta_background'].add(delta[~road_cells])
                    for key, cells in (('road', road_cells), ('background', ~road_cells)):
                        counts = torch.histc(q[cells].float(), bins=10, min=0, max=1).long().cpu().numpy()
                        q_hist[key] += counts
                    relative_sum += float(delta.mean() / feature.abs().mean().clamp_min(1e-8))
                    target, direction_weight = direction_target(skel, road, q.shape[-2:], direction_cfg)
                    cosine = (F.normalize(raw.float(), dim=1, eps=1e-6) * target).sum(1, keepdim=True).clamp(-1, 1)
                    angle = .5 * torch.acos(cosine) * (180 / math.pi)
                    angle_sum += float((angle * direction_weight).sum())
                    angle_weight += float(direction_weight.sum())
                    direction_valid += int((direction_weight > 0).sum())
                    del feature, updated, raw, q, delta, target, direction_weight
                n_windows += 1
                if index - 1 in gradient_image_indices and (top, left) == gradient_position:
                    audit = gradient_audit(model, crop, road, skel, direction_cfg)
                    if audit['valid_direction_pixels']:
                        audit.update(image=image_name, top=top, left=left)
                        gradients.append(audit)
                # Forward hook replaces only E2's updated feature; all parameters stay unchanged.
                off_hook = module.register_forward_hook(lambda _module, inputs, output: (inputs[0], output[1]))
                try:
                    with torch.no_grad():
                        off = model(crop)[0].float()
                finally:
                    off_hook.remove()
                region = (slice(None), slice(None), slice(top, top + TILE_SIZE), slice(left, left + TILE_SIZE))
                canvases['direction_on'][region] += on * weight
                canvases['direction_off'][region] += off * weight
                divisor[region] += weight
                del crop, road, skel, on, off
        if bool((divisor <= 0).any()):
            raise RuntimeError('Uncovered native pixels')
        probabilities = {name: (canvas / divisor).sigmoid()[0, 0].cpu().numpy()
                         for name, canvas in canvases.items()}
        binary = {name: prob >= threshold for name, prob in probabilities.items()}
        for name, pred in binary.items():
            out = args.output_dir / name / (Path(image_name).stem + '_pred.png')
            if not cv2.imwrite(str(out), pred.astype(np.uint8) * 255):
                raise OSError(f'Could not save {out}')
            row = road_metrics.image_metrics(pred, prepared, {**road_metrics.DEFAULT_PROTOCOL,
                'image_size': SOURCE_SIZE})
            row['case_id'] = Path(image_name).stem
            rows[name].append(row)
        on, off = binary['direction_on'], binary['direction_off']
        onprob, offprob = probabilities['direction_on'], probabilities['direction_off']
        missing = skeleton & ~on
        paired.append({'case_id': Path(image_name).stem,
            'probability_abs_change_mean': float(np.abs(onprob - offprob).mean()),
            'probability_abs_change_on_on_missed_GT_skeleton': float(np.abs(onprob - offprob)[missing].mean())
                if missing.any() else None,
            'mask_flip_fraction': float(np.mean(on != off)),
            'GT_road_added_by_direction': int((on & ~off & gt).sum()),
            'background_added_by_direction': int((on & ~off & ~gt).sum()),
            'GT_road_removed_by_direction': int((off & ~on & gt).sum()),
            'background_removed_by_direction': int((off & ~on & ~gt).sum()),
            'missed_GT_skeleton_after_direction': int(missing.sum()),
            'missed_GT_skeleton_recovered_by_direction': int((skeleton & on & ~off).sum())})
        print(f'[{index}/{len(images)}] {image_name} paired native {SOURCE_SIZE} done', flush=True)
    summary = {name: road_metrics.summarize(items) for name, items in rows.items()}
    keys = ('iou', 'f1', 'cldice', 'break_rate', 'short_pred_components',
            'fragment_density_per_1000_px', 'apls_gt_to_pred_approx',
            'apls_pred_to_gt_approx', 'apls_bidirectional_approx')
    report = {'architecture': MODEL_NAME, 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': model_hash.hexdigest(), 'split': args.split,
        'selection': str(args.selection), 'threshold': threshold, 'images': len(images),
        'complete_split': len(images) == len(dataset.image_files),
        'protocol': {'native_size': SOURCE_SIZE, 'tile_size': TILE_SIZE, 'stride': STRIDE,
                     'merge': 'weighted logits then sigmoid', 'FP32': True,
                     'q_road_rule': 'GT road fraction in E2 cell >= 0.5',
                     'GT_direction': direction_cfg,
                     'direction_angle': 'weighted unsigned half acos of double-angle dot, degrees',
                     'APLS': 'unified raster approximation, not official APLS'},
        'module': {'gamma': float(module.gamma.detach()),
                   'projection_weight_norm': float(module.projection.weight.detach().float().norm()),
                   'projection_bias_norm': float(module.projection.bias.detach().float().norm()),
                   'direction_weight_norm': float(module.direction_head.weight.detach().float().norm()),
                   'q_weight_norm': float(module.gate_head.weight.detach().float().norm()),
                   'q_bias': float(module.gate_head.bias.detach().float().mean()),
                   'q_road': stats['q_road'].report(),
                   'q_background': stats['q_background'].report(),
                   'q_histogram_bins_0_to_1': {key: counts.tolist() for key, counts in q_hist.items()},
                   'delta_abs_road': stats['delta_road'].report(),
                   'delta_abs_background': stats['delta_background'].report(),
                   'delta_relative_abs_window_mean': relative_sum / n_windows,
                   'direction_weighted_angle_error_deg': angle_sum / angle_weight if angle_weight else None,
                   'direction_valid_E2_cells_counting_overlaps': direction_valid,
                   'direction_weight_sum_counting_overlaps': angle_weight},
        'paired_same_checkpoint_fixed_val_threshold': {'direction_on': summary['direction_on'],
            'direction_off': summary['direction_off'],
            'on_minus_off': {key: summary['direction_on'][key] - summary['direction_off'][key]
                             for key in keys},
            'limits': 'This is a frozen-checkpoint intervention. It cannot estimate retraining benefit.'},
        'surface_gradient_audit': gradients,
        'paired_pixel_changes': {key: sum(item[key] for item in paired) for key in
            ('GT_road_added_by_direction', 'background_added_by_direction',
             'GT_road_removed_by_direction', 'background_removed_by_direction',
             'missed_GT_skeleton_recovered_by_direction')},
        'missing_components': 'This architecture has no P2 gap candidate generator or repair head; Gap AP and candidate recall are inapplicable.'}
    log_path = args.log_path or (args.checkpoint.parent / 'e2_direction.jsonl')
    if log_path.is_file():
        report['training_log'] = summarize_log(log_path)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.output_dir / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)
    for name, items in rows.items():
        with (args.output_dir / (name + '_topology.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=items[0].keys())
            writer.writeheader()
            writer.writerows(items)
    print(json.dumps({'threshold': threshold, 'images': len(images),
                      'complete_split': report['complete_split'],
                      'module': report['module'],
                      'on_minus_off': report['paired_same_checkpoint_fixed_val_threshold']['on_minus_off'],
                      'gradient_audit': gradients}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    logs = sub.add_parser('logs', help='summarize existing training JSONL; no model or data needed')
    logs.add_argument('--log_path', type=Path, required=True)
    logs.add_argument('--output', type=Path)
    check = sub.add_parser('checkpoint', help='paired on/off inference and E2 diagnostics')
    check.add_argument('--checkpoint', type=Path, required=True)
    check.add_argument('--root_path', type=Path, required=True)
    check.add_argument('--selection', type=Path, required=True)
    check.add_argument('--split', choices=('val', 'test'), default='test')
    check.add_argument('--output_dir', type=Path, required=True)
    check.add_argument('--log_path', type=Path)
    check.add_argument('--device', default='cuda')
    check.add_argument('--max_images', type=int, default=0, help='0 runs the complete split')
    check.add_argument('--gradient_samples', type=int, default=1,
                       help='extra frozen-checkpoint surface vs D backward crops; 0 disables')
    args = parser.parse_args()
    if args.mode == 'logs':
        result = summarize_log(args.log_path)
        content = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.write_text(content, encoding='utf-8')
        print(content)
        return
    if args.max_images < 0 or args.gradient_samples < 0:
        parser.error('--max_images and --gradient_samples must be nonnegative')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu for a small diagnostic')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    diagnose(args)


if __name__ == '__main__':
    main()
