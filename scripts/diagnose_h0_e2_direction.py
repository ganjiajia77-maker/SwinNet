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
from scipy.stats import rankdata

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
                    'weighted_D_to_surface_grad_ratio': float(b_norm / a_norm)
                    if a_norm > 0 else None,
                    'gradient_cosine': float(F.cosine_similarity(a[None], b[None]))
                    if a_norm > 0 and b_norm > 0 else None}
    finally:
        hook.remove()
        feature_box.clear()


def run_variant(model, image, mode, seed):
    """Forward A/B/C/D while changing only E2's diagnostic-time aggregation."""
    module = model.swin_unet.e2_direction
    captured = []
    original_forward = module.forward

    def diagnostic_forward(feature):
        raw = module.direction_head(feature)
        gate = module.gate_head(feature).sigmoid()
        weights = module.radius_head(feature).softmax(1)
        sampled_raw = raw
        if mode == 'D':
            generator = torch.Generator(device='cpu').manual_seed(seed)
            permutation = torch.randperm(raw.shape[-2] * raw.shape[-1], generator=generator).to(raw.device)
            sampled_raw = raw.flatten(2)[:, :, permutation].view_as(raw)
        tangent = module.tangent(sampled_raw)
        message, valid = module.aggregate(feature, tangent, weights)
        projected = module.projection(message)
        effective_gate = torch.ones_like(gate) if mode == 'C' else gate
        delta = module.gamma * effective_gate * projected * valid.to(feature.dtype)
        if mode == 'B':
            delta = torch.zeros_like(delta)
        updated = feature + delta
        captured.append({'feature': feature.detach(), 'updated': updated.detach(),
                         'delta': delta.detach(), 'raw': raw.detach(), 'gate': gate.detach(),
                         'direction_used': sampled_raw.detach()})
        return updated, raw

    try:
        if mode == 'A':
            def capture(_module, inputs, output):
                feature, updated = inputs[0], output[0]
                q = _module.gate_head(feature).sigmoid()
                captured.append({'feature': feature.detach(), 'updated': updated.detach(),
                    'delta': (updated - feature).detach(), 'raw': output[1].detach(),
                    'gate': q.detach(), 'direction_used': output[1].detach()})
            hook = module.register_forward_hook(capture)
            try:
                outputs = model(image)
            finally:
                hook.remove()
        else:
            module.forward = diagnostic_forward
            try:
                outputs = model(image)
            finally:
                module.forward = original_forward
    finally:
        module.forward = original_forward
    if len(captured) != 1:
        raise RuntimeError(f'E2 diagnostic forward {mode} called {len(captured)} times')
    return outputs[0].float(), captured[0]


def feature_pair_scores(before, after, road, skeleton, probability, seed):
    """Sample local GT-defined continuous/gap/near-but-disconnected E2 pairs."""
    h, w = before.shape[-2:]
    road64 = F.adaptive_avg_pool2d(road, (h, w))[0, 0].detach().cpu().numpy()
    skeleton64 = F.adaptive_max_pool2d(skeleton, (h, w))[0, 0].detach().cpu().numpy() > .5
    probability64 = F.adaptive_avg_pool2d(probability, (h, w))[0, 0].detach().cpu().numpy()
    # Thin the downsampled centerline, then label local 8-connected strands.
    skeleton64 = road_metrics.skeletonize(skeleton64)
    n_components, component = cv2.connectedComponents(skeleton64.astype(np.uint8), connectivity=8)
    points = np.argwhere(skeleton64)
    if len(points) < 2:
        return {}
    if len(points) > 384:
        choose = np.linspace(0, len(points) - 1, 384, dtype=int)
        points = points[choose]
    from scipy.spatial import cKDTree
    candidate_pairs = sorted(cKDTree(points).query_pairs(r=8.0))
    if not candidate_pairs:
        return {}
    # GT support near the straight segment distinguishes a real low-response gap from
    # an arbitrary chord through background.
    support = cv2.dilate((road64 >= .10).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    before_grid = F.normalize(before[0].float(), dim=0, eps=1e-8).permute(1, 2, 0).detach().cpu().numpy()
    after_grid = F.normalize(after[0].float(), dim=0, eps=1e-8).permute(1, 2, 0).detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(candidate_pairs)
    grouped = {'continuous': [], 'difficult_gap': [], 'near_disconnected': []}
    for ia, ib in candidate_pairs:
        y1, x1 = map(int, points[ia]); y2, x2 = map(int, points[ib])
        distance = float(np.hypot(y2 - y1, x2 - x1))
        if distance < 2.0 or distance > 8.0:
            continue
        samples = np.linspace(0.0, 1.0, max(5, int(distance * 2) + 1))
        line_y = np.rint(y1 + samples * (y2 - y1)).astype(int).clip(0, h - 1)
        line_x = np.rint(x1 + samples * (x2 - x1)).astype(int).clip(0, w - 1)
        supported = float(support[line_y, line_x].mean()) >= .75
        same_component = component[y1, x1] == component[y2, x2]
        category = None
        if same_component and supported and distance <= 5.0:
            category = 'continuous'
        elif (same_component and supported and distance >= 3.0
              and probability64[y1, x1] >= .5 and probability64[y2, x2] >= .5
              and float(probability64[line_y, line_x].mean()) < .35):
            category = 'difficult_gap'
        elif not same_component and distance <= 6.0:
            category = 'near_disconnected'
        if category is None:
            continue
        grouped[category].append((ia, ib, y1, x1, y2, x2))
    result = {}
    for category, pairs in grouped.items():
        if len(pairs) > 256:
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, 256, dtype=int)]
        scores_before, scores_after = [], []
        for _, _, y1, x1, y2, x2 in pairs:
            scores_before.append(float(np.dot(before_grid[y1, x1], before_grid[y2, x2])))
            scores_after.append(float(np.dot(after_grid[y1, x1], after_grid[y2, x2])))
        result[category] = {'count': len(pairs),
            'cosine_before_mean': float(np.mean(scores_before)) if pairs else None,
            'cosine_after_mean': float(np.mean(scores_after)) if pairs else None,
            'cosine_delta_mean': float(np.mean(np.asarray(scores_after) - np.asarray(scores_before)))
                if pairs else None,
            'positive_cosine_fraction_before': float(np.mean(np.asarray(scores_before) > .5)) if pairs else None,
            'positive_cosine_fraction_after': float(np.mean(np.asarray(scores_after) > .5)) if pairs else None,
            '_scores_before': scores_before, '_scores_after': scores_after}
    return result


def auc_from_scores(positive, negative):
    if not positive or not negative:
        return None
    scores = np.asarray(positive + negative, dtype=np.float64)
    ranks = rankdata(scores, method='average')
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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
    modes = {'A': 'predicted_D_predicted_q', 'B': 'aggregation_off',
             'C': 'predicted_D_q_fixed_one', 'D': 'spatially_shuffled_D'}
    for name in modes:
        (args.output_dir / name).mkdir()
    model_hash = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            model_hash.update(chunk)
    stats = {key: Moment() for key in ('q_road', 'q_background', 'delta_road', 'delta_background')}
    angle_sum = angle_weight = direction_valid = relative_sum = 0.0
    n_windows = 0
    q_hist = {'road': np.zeros(10, np.int64), 'background': np.zeros(10, np.int64)}
    rows = {name: [] for name in modes}
    paired = []
    gradients = []
    pair_scores = {category: {'before': [], 'after': []} for category in
                   ('continuous', 'difficult_gap', 'near_disconnected')}
    weight = blend_weight(TILE_SIZE, device)
    gradient_image_indices = set(np.linspace(
        0, len(images) - 1, min(args.gradient_samples, len(images)), dtype=int).tolist()
    ) if args.gradient_samples else set()
    window_rows = sliding_positions(SOURCE_SIZE, TILE_SIZE, STRIDE)
    gradient_position = (window_rows[len(window_rows) // 2], window_rows[len(window_rows) // 2])
    feature_image_indices = set(np.linspace(
        0, len(images) - 1, min(args.feature_images, len(images)), dtype=int).tolist()
    ) if args.feature_images else set()
    direction_cfg = {key: saved[key] for key in ('pca_radius', 'min_support', 'road_radius',
                                                 'min_coherence', 'lambda_encoder_direction')}
    for index, image_name in enumerate(images, 1):
        native_image, gt, skeleton = read_native(dataset, image_name)
        prepared = road_metrics.prepare_target(gt, {**road_metrics.DEFAULT_PROTOCOL,
            'image_size': SOURCE_SIZE})
        canvases = {name: torch.zeros((1, 1, SOURCE_SIZE, SOURCE_SIZE), device=device)
                    for name in modes}
        divisor = torch.zeros_like(canvases['A'])
        for top in sliding_positions(SOURCE_SIZE, TILE_SIZE, STRIDE):
            for left in sliding_positions(SOURCE_SIZE, TILE_SIZE, STRIDE):
                crop = native_image[:, :, top:top + TILE_SIZE, left:left + TILE_SIZE].to(device)
                road = torch.from_numpy(gt[top:top + TILE_SIZE, left:left + TILE_SIZE].copy()).to(device)[None, None].float()
                skel = torch.from_numpy(skeleton[top:top + TILE_SIZE, left:left + TILE_SIZE].copy()).to(device)[None, None].float()
                logits, captures = {}, {}
                seed_base = int.from_bytes(hashlib.sha256(
                    f'{Path(image_name).stem}:{top}:{left}'.encode()).digest()[:4], 'little')
                for mode in modes:
                    with torch.no_grad():
                        logits[mode], captures[mode] = run_variant(model, crop, mode, seed_base)
                capture = captures['A']
                feature, updated, raw = capture['feature'], capture['updated'], capture['raw']
                q = capture['gate']
                with torch.no_grad():
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
                    if (index - 1 in feature_image_indices
                            and (top, left) == gradient_position):
                        pair_result = feature_pair_scores(
                            feature, updated, road, skel, torch.sigmoid(logits['A']), seed_base)
                        for category, values in pair_result.items():
                            pair_scores[category]['before'].extend(values.pop('_scores_before'))
                            pair_scores[category]['after'].extend(values.pop('_scores_after'))
                    del feature, updated, raw, q, delta, target, direction_weight
                n_windows += 1
                if index - 1 in gradient_image_indices and (top, left) == gradient_position:
                    audit = gradient_audit(model, crop, road, skel, direction_cfg)
                    if audit['valid_direction_pixels']:
                        audit.update(image=image_name, top=top, left=left)
                        gradients.append(audit)
                region = (slice(None), slice(None), slice(top, top + TILE_SIZE), slice(left, left + TILE_SIZE))
                for mode in modes:
                    canvases[mode][region] += logits[mode] * weight
                divisor[region] += weight
                del crop, road, skel, logits, captures
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
        row = {'case_id': Path(image_name).stem}
        for mode in ('B', 'C', 'D'):
            reference, variant = binary['A'], binary[mode]
            ref_prob, var_prob = probabilities['A'], probabilities[mode]
            tag = f'A_vs_{mode}'
            miss = skeleton & ~reference
            row.update({
                f'{tag}_probability_abs_change_mean': float(np.abs(ref_prob - var_prob).mean()),
                f'{tag}_prob_change_on_A_missed_GT_skeleton':
                    float(np.abs(ref_prob - var_prob)[miss].mean()) if miss.any() else None,
                f'{tag}_mask_flip_fraction': float(np.mean(reference != variant)),
                f'{tag}_GT_road_added': int((variant & ~reference & gt).sum()),
                f'{tag}_background_added': int((variant & ~reference & ~gt).sum()),
                f'{tag}_GT_road_removed': int((reference & ~variant & gt).sum()),
                f'{tag}_background_removed': int((reference & ~variant & ~gt).sum()),
            })
        row['A_missed_GT_skeleton'] = int((skeleton & ~binary['A']).sum())
        row['A_GT_skeleton_pixels_recovered_by_B'] = int((skeleton & binary['B'] & ~binary['A']).sum())
        row['A_GT_skeleton_pixels_recovered_by_C'] = int((skeleton & binary['C'] & ~binary['A']).sum())
        row['A_GT_skeleton_pixels_recovered_by_D'] = int((skeleton & binary['D'] & ~binary['A']).sum())
        paired.append(row)
        print(f'[{index}/{len(images)}] {image_name} A/B/C/D native {SOURCE_SIZE} done', flush=True)
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
        'paired_same_checkpoint_fixed_val_threshold': {
            'A': {'definition': modes['A'], 'metrics': summary['A']},
            'B': {'definition': modes['B'], 'metrics': summary['B'],
                  'A_minus_B': {key: summary['A'][key] - summary['B'][key] for key in keys}},
            'C': {'definition': modes['C'], 'metrics': summary['C'],
                  'A_minus_C': {key: summary['A'][key] - summary['C'][key] for key in keys}},
            'D': {'definition': modes['D'], 'metrics': summary['D'],
                  'A_minus_D': {key: summary['A'][key] - summary['D'][key] for key in keys}},
            'D_vs_A_is_primary': True,
            'limits': 'Frozen-checkpoint interventions measure dependence on forward behavior, not retraining benefit.'},
        'feature_pair_diagnostic': {
            'images_used': min(args.feature_images, len(images)),
            'pair_rules': {
                'continuous': 'downsampled GT skeleton cells in same 8-connected component; 2-5 E2 cells apart',
                'difficult_gap': 'same component; GT road supports the straight segment; endpoints A surface probability >=0.5; segment mean <0.35',
                'near_disconnected': 'different downsampled GT skeleton components within 6 E2 cells',
            },
            'metrics': {category: {'pairs': len(values['before']),
                'cosine_before_mean': float(np.mean(values['before'])) if values['before'] else None,
                'cosine_after_mean': float(np.mean(values['after'])) if values['after'] else None,
                'cosine_delta_mean': float(np.mean(np.asarray(values['after']) - np.asarray(values['before'])))
                    if values['before'] else None}
                for category, values in pair_scores.items()},
            'continuous_vs_near_disconnected_auc': {
                'before': auc_from_scores(pair_scores['continuous']['before'],
                                          pair_scores['near_disconnected']['before']),
                'after': auc_from_scores(pair_scores['continuous']['after'],
                                         pair_scores['near_disconnected']['after'])},
            'difficult_gap_vs_near_disconnected_auc': {
                'before': auc_from_scores(pair_scores['difficult_gap']['before'],
                                          pair_scores['near_disconnected']['before']),
                'after': auc_from_scores(pair_scores['difficult_gap']['after'],
                                         pair_scores['near_disconnected']['after'])},
            'limits': 'Pairs are local sampled probes under the declared rules; they are not a learned matcher or road-instance benchmark.'},
        'surface_gradient_audit': gradients,
        'paired_pixel_changes': {mode: {
            'GT_road_added': sum(item[f'A_vs_{mode}_GT_road_added'] for item in paired),
            'background_added': sum(item[f'A_vs_{mode}_background_added'] for item in paired),
            'GT_road_removed': sum(item[f'A_vs_{mode}_GT_road_removed'] for item in paired),
            'background_removed': sum(item[f'A_vs_{mode}_background_removed'] for item in paired),
            'A_missed_GT_skeleton_recovered': sum(item[f'A_GT_skeleton_pixels_recovered_by_{mode}'] for item in paired),
        } for mode in ('B', 'C', 'D')},
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
                      'paired_metrics': report['paired_same_checkpoint_fixed_val_threshold'],
                      'feature_pairs': report['feature_pair_diagnostic'],
                      'gradient_audit': gradients}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    logs = sub.add_parser('logs', help='summarize existing training JSONL; no model or data needed')
    logs.add_argument('--log_path', type=Path, required=True)
    logs.add_argument('--output', type=Path)
    check = sub.add_parser('checkpoint', help='paired A/B/C/D inference and E2 feature diagnostics')
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
    check.add_argument('--feature_images', type=int, default=8,
                       help='number of evenly spaced images used for E2 feature-pair probes; 0 disables')
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
    if args.max_images < 0 or args.gradient_samples < 0 or args.feature_images < 0:
        parser.error('--max_images, --gradient_samples, and --feature_images must be nonnegative')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu for a small diagnostic')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    diagnose(args)


if __name__ == '__main__':
    main()
