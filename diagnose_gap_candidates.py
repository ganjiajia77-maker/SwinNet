"""Frozen P2 candidate recall and Gap head audit; never selects a test threshold."""
import argparse
from collections import defaultdict
from collections import deque
from dataclasses import replace
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from config import _C
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from evaluation.road_comparison.road_metrics import skeletonize
from gap_candidates import endpoints
from gap_diagnostic_metrics import verified_gaps, gap_metrics
from local_gap_config import LocalGapConfig
from losses.gap_losses import label_edges
from mass_h0 import verify_checkpoint
from mass_protocol import sliding_positions
from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state
import networks.local_gap_connectivity as gap_module


def ranked_pairs(points, cfg, dual=False, high_points=None):
    """Replay the production distance, tangent, endpoint-degree, and edge-budget gates."""
    high_points = set() if high_points is None else high_points
    stages = {'endpoints': [], 'distance': [], 'direction': [], 'top3': [], 'top64': []}
    for i, (u, tu, pu) in enumerate(points):
        for j in range(i + 1, len(points)):
            v, tv, pv = points[j]
            if dual and u not in high_points and v not in high_points:
                continue
            edge = np.asarray([u, v], np.float32)
            stages['endpoints'].append(edge)
            delta = edge[1] - edge[0]
            distance = float(np.linalg.norm(delta))
            if not cfg.min_distance <= distance <= cfg.max_distance:
                continue
            stages['distance'].append(edge)
            unit = delta / distance
            cu, cv = float(tu @ unit), float(tv @ -unit)
            if min(cu, cv) < cfg.tangent_cosine:
                continue
            score = min(cu, cv) + .25 * (pu + pv) - .25 * distance / cfg.max_distance
            stages['direction'].append((-score, i, j, edge))
    ranked = sorted(stages['direction'], key=lambda p: (p[0], *points[p[1]][0], *points[p[2]][0]))
    stages['direction'] = [p[3] for p in ranked]
    degree = np.zeros(len(points), np.int32)
    for _, i, j, edge in ranked:
        if degree[i] >= cfg.per_endpoint or degree[j] >= cfg.per_endpoint:
            continue
        degree[i] += 1; degree[j] += 1
        stages['top3'].append(edge)
        if len(stages['top64']) < cfg.max_edges:
            stages['top64'].append(edge)
    return {key: np.asarray(value, np.float32).reshape(-1, 2, 2)
            for key, value in stages.items()}


def gap_covered(gap, edges, scale, snap_radius):
    """An edge must place its two endpoints near opposite verified GT flanks."""
    _, (y0, y1, x0, x1), _, anchors, pairs, _ = gap
    if not len(edges):
        return False
    native = (edges + .5) * scale - .5
    near = ((native[..., 0] >= y0 - snap_radius) & (native[..., 0] < y1 + snap_radius) &
            (native[..., 1] >= x0 - snap_radius) & (native[..., 1] < x1 + snap_radius)).any(axis=1)
    native = native[near]
    if not len(native):
        return False
    hits = []
    for anchor, _, _ in anchors:
        coords = np.argwhere(anchor) + [y0, x0]
        if not len(coords):
            hits.append(np.zeros((len(native), 2), bool))
        else:
            anchor_hits = []
            for start in range(0, len(native), 256):
                chunk = native[start:start+256]
                distance = np.linalg.norm(chunk[:, :, None, :] - coords[None, None, :, :], axis=-1)
                anchor_hits.append((distance <= snap_radius).any(axis=-1))
            hits.append(np.concatenate(anchor_hits, axis=0))
    for i, j in pairs:
        if np.any((hits[i][:, 0] & hits[j][:, 1]) | (hits[j][:, 0] & hits[i][:, 1])):
            return True
    return False


def average_precision(labels, scores):
    valid = np.asarray(labels) >= 0
    y = np.asarray(labels)[valid]
    s = np.asarray(scores)[valid]
    if not len(y) or not (y == 1).any():
        return None
    order = np.argsort(-s, kind='stable')
    positive = (y[order] == 1).astype(np.float64)
    return float((positive * (positive.cumsum() / np.arange(1, len(y)+1))).sum() / positive.sum())


def prediction_endpoints(probability, threshold, min_component=4):
    skeleton = skeletonize(probability >= threshold).astype(bool)
    count, components, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component
    skeleton &= keep[components]
    padded = np.pad(skeleton.astype(np.uint8), 1)
    h, w = skeleton.shape
    degree = sum(padded[1+dy:1+dy+h, 1+dx:1+dx+w]
                 for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1),
                                (1, 0), (1, -1), (0, -1), (-1, -1)))
    points = np.argwhere(skeleton & (degree == 1))
    _, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    return points, labels, skeleton


def endpoint_path_stats(skeleton, start, finish):
    h, w = skeleton.shape
    queue = deque([(tuple(start), 0)])
    seen = {tuple(start)}
    while queue:
        (y, x), distance = queue.popleft()
        if (y, x) == tuple(finish):
            return distance, False
        for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1),
                       (1, 0), (1, -1), (0, -1), (-1, -1)):
            point = (y+dy, x+dx)
            if (0 <= point[0] < h and 0 <= point[1] < w and skeleton[point]
                    and point not in seen):
                seen.add(point)
                queue.append((point, distance+1))
    return None, False


def classify_gap(gap, skeleton, endpoints_yx, component_labels, radius):
    """Classify whether output centerline endpoints land on each GT gap flank."""
    _, (y0, y1, x0, x1), _, anchors, pairs, _ = gap
    hits = []
    for anchor, _, _ in anchors:
        coords = np.argwhere(anchor) + [y0, x0]
        if len(coords) and len(endpoints_yx):
            distances = np.linalg.norm(endpoints_yx[:, None] - coords[None], axis=-1)
            hit_ids = np.flatnonzero((distances <= radius).any(axis=1))
        else:
            hit_ids = np.empty(0, dtype=np.int64)
        hits.append(hit_ids)
    matched_pairs = []
    for first, second in pairs:
        for a in hits[first]:
            for b in hits[second]:
                matched_pairs.append((int(a), int(b)))
    if matched_pairs:
        same_pairs = [(a, b) for a, b in matched_pairs
                      if component_labels[tuple(endpoints_yx[a])] ==
                      component_labels[tuple(endpoints_yx[b])]]
        if not same_pairs:
            category, path_ratio = 'both_flanks_disconnected', None
        else:
            path_ratios = []
            for a, b in same_pairs:
                distance, _ = endpoint_path_stats(skeleton, endpoints_yx[a], endpoints_yx[b])
                euclidean = np.linalg.norm(endpoints_yx[a] - endpoints_yx[b])
                if distance is not None:
                    path_ratios.append(distance / max(euclidean, 1e-6))
            path_ratio = max(path_ratios, default=1.)
            y0, y1, x0, x1 = gap[1]
            local = skeleton[max(0, y0-radius):min(skeleton.shape[0], y1+radius),
                             max(0, x0-radius):min(skeleton.shape[1], x1+radius)]
            padded = np.pad(local.astype(np.uint8), 1)
            hh, ww = local.shape
            degree = sum(padded[1+dy:1+dy+hh, 1+dx:1+dx+ww]
                         for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1),
                                        (1, 0), (1, -1), (0, -1), (-1, -1)))
            if np.any(local & (degree >= 3)):
                category = 'both_flanks_same_component_near_junction'
            elif path_ratio > 2.0:
                category = 'both_flanks_same_component_long_route'
            else:
                category = 'both_flanks_same_component'
        return category, [int(value.size) for value in hits], len(matched_pairs), path_ratio
    flank_count = sum(bool(value.size) for value in hits)
    if flank_count:
        return 'endpoint_on_one_or_more_flanks', [int(value.size) for value in hits], 0, None
    # Distinguish no nearby centerline endpoint from endpoints that miss GT flank matching.
    local = endpoints_yx[(endpoints_yx[:, 0] >= y0-radius) &
                         (endpoints_yx[:, 0] < y1+radius) &
                         (endpoints_yx[:, 1] >= x0-radius) &
                         (endpoints_yx[:, 1] < x1+radius)] if len(endpoints_yx) else endpoints_yx
    category = 'endpoint_near_gap_unmatched' if len(local) else 'no_endpoint_near_gap'
    return category, [int(value.size) for value in hits], 0, None


def oracle_proposals(gaps, feature_shape, cfg, skeleton, mask):
    """Build ideal GT-flank endpoints, then retain only positive local-path labels."""
    scale = np.asarray(skeleton.shape) / np.asarray(feature_shape)
    edges, geometry = [], []
    for _, (y0, y1, x0, x1), _, anchors, pairs, _ in gaps:
        choices = []
        for i, j in pairs:
            ci = anchors[i][1]  # OpenCV (x,y), local to the gap crop
            cj = anchors[j][1]
            u = np.asarray([ci[1] + y0, ci[0] + x0], np.float32)
            v = np.asarray([cj[1] + y0, cj[0] + x0], np.float32)
            uf = (u + .5) / scale - .5
            vf = (v + .5) / scale - .5
            distance = float(np.linalg.norm(vf-uf))
            if cfg.min_distance <= distance <= cfg.max_distance:
                choices.append((distance, uf, vf))
        if choices:
            _, uf, vf = min(choices, key=lambda item: item[0])
            edges.append([uf, vf])
    edge_array = np.asarray(edges, np.float32).reshape(-1, 2, 2)
    if not len(edge_array):
        return edge_array, np.empty((0, 8), np.float32)
    labels, _ = label_edges(skeleton, mask, edge_array, feature_shape, cfg)
    edge_array = edge_array[labels == 1]
    rows = []
    for edge in edge_array:
        delta = (edge[1] - edge[0])[::-1]
        distance = float(np.linalg.norm(delta))
        unit = delta / max(distance, 1e-6)
        rows.append([distance/cfg.max_distance, 1., 1., -1., unit[0], unit[1], 1., 1.])
    return edge_array[:cfg.max_edges], np.asarray(rows[:cfg.max_edges], np.float32).reshape(-1, 8)


def training_history(path):
    if not path.is_file():
        return {'available': False, 'note': 'No local_gap_audit.jsonl; epoch history unavailable'}
    epochs = defaultdict(lambda: {'batches': 0, 'empty_positive_batches': 0,
                                  'candidates': 0, 'positive': 0, 'negative': 0, 'ignored': 0})
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        slot = epochs[int(row['epoch'])]
        slot['batches'] += 1
        slot['empty_positive_batches'] += int(row.get('gap_positive', 0) == 0)
        for source, key in [('gap_candidates', 'candidates'), ('gap_positive', 'positive'),
                            ('gap_negative', 'negative'), ('gap_ignored', 'ignored')]:
            slot[key] += int(row.get(source, 0))
    return {'available': True, 'epochs': {str(k): dict(v, empty_positive_fraction=v['empty_positive_batches'] / v['batches'])
                                          for k, v in sorted(epochs.items()) if v['batches']},
            'scope': 'all logged training batches; epoch stats are not validation metrics'}


def gradient_audit(model, sample, cfg, device, top, left):
    """One frozen eval tile; compare actual weighted Gap BCE and surface BCE gradients."""
    captured = []
    hook = model.swin_unet.gap_connection.register_forward_hook(
        lambda _, inputs, outputs: captured.append((inputs, outputs[1])))
    try:
        image = sample['image'][None, :, top:top+512, left:left+512].to(device)
        surface = sample['mask'][None, :, top:top+512, left:left+512].to(device)
        skeleton = sample['skeleton'][None, :, top:top+512, left:left+512].numpy() > .5
        outputs = model(image)
        if len(captured) != 1:
            raise RuntimeError('Gradient audit expected one Gap invocation')
        inputs, record = captured[0]
        logits = record['gap_logits'][0]
        edges = record['gap_edges'][0].detach().cpu().numpy()
        labels, _ = label_edges(skeleton[0, 0], surface[0, 0].cpu().numpy() > .5,
                                edges, record['p2_logits'].shape[-2:], cfg)
        y = torch.as_tensor(labels, device=device)
        terms = [F.binary_cross_entropy_with_logits(logits[y == value],
                 y[y == value].float()) for value in (1, 0) if (y == value).any()]
        if not terms:
            return {'available': False, 'reason': 'No classifiable Gap edges in first tile'}
        gap_weighted = cfg.gap_weight * torch.stack(terms).mean()
        surface_bce = F.binary_cross_entropy_with_logits(outputs[0].float(), surface)
        shared = {'decoder_at_gap': inputs[0], 'H_at_gap': inputs[1],
                  'G3_structure': model.swin_unet.last_stage_features.get('G3')}
        targets = {**shared, **{f'gap_mlp_{n}': p for n, p in
                   model.swin_unet.gap_connection.edge_mlp.named_parameters()}}
        targets = {name: value for name, value in targets.items()
                   if value is not None and value.requires_grad}

        def norms(loss):
            grads = torch.autograd.grad(loss, list(targets.values()), retain_graph=True,
                                        allow_unused=True)
            return {name: float(grad.float().norm().detach().cpu()) if grad is not None else None
                    for name, grad in zip(targets, grads)}

        return {'available': True, 'gap_weighted_loss': float(gap_weighted.detach()),
                'surface_bce_loss': float(surface_bce.detach()),
                'positive': int((y == 1).sum()), 'negative': int((y == 0).sum()),
                'gap_weighted_grad_l2': norms(gap_weighted),
                'surface_bce_grad_l2': norms(surface_bce),
                'scope': 'First tile with a positive Gap label, frozen checkpoint in eval mode. '
                         'Surface reference is BCE only; '
                         'this is not the complete surface loss. Unused gradients are null.'}
    finally:
        hook.remove()


def load_model(checkpoint_path, device):
    info = verify_checkpoint(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = _C.clone()
    config.merge_from_file(str(Path(__file__).parent / 'configs/swin_tiny_patch4_window7_224_lite.yaml'))
    config.DATA.IMG_SIZE = 512
    model = SwinUnet(config, img_size=512, num_classes=1, return_skeleton=True,
                    structure_profile=info['args']['structure_profile'],
                    enable_highres_structure_stream=True, enable_h3_surface_fusion=True,
                    stage_skeleton_mode='direct', remove_stage2_pre_topology_source=True,
                    local_gap_config=info['args']['local_gap_config']).to(device).eval()
    load_topology_checkpoint_state(model, saved['model_state_dict'], saved['topology_attention_version'])
    return model, LocalGapConfig.parse(info['args']['local_gap_config'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max_images', type=int, default=0, help='0 scans all images')
    parser.add_argument('--max_tiles_per_image', type=int, default=0, help='0 scans all 512/256 tiles')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reference_threshold', type=float, default=.5)
    parser.add_argument('--gradient_audit', action='store_true')
    parser.add_argument('--gt_oracle', action='store_true',
                        help='Run extra GT-proposal forwards through Gap MLP and trained surface writeback')
    parser.add_argument('--surface_threshold', type=float, default=.45)
    args = parser.parse_args()
    if not 0 < args.reference_threshold < 1:
        parser.error('reference_threshold must be in (0,1)')
    if not 0 < args.surface_threshold < 1:
        parser.error('surface_threshold must be in (0,1)')
    if args.output_dir.exists():
        parser.error('Choose a new output_dir')
    model, cfg = load_model(args.model_path, args.device)
    dataset = RoadSkeletonDataset(args.root_path, args.split, image_size=None,
                                  source_patch_size=1500, return_full_image=True)
    args.output_dir.mkdir(parents=True)
    modes = ('p2_05', 'p2_03', 'dual_05_03', 'surface_512')
    stage_names = ('endpoints', 'distance', 'direction', 'top3', 'top64')
    totals = {mode: {stage: {'candidates': 0, 'covered_gaps': 0} for stage in stage_names}
              for mode in modes}
    unique_gap_coverage = {mode: {stage: set() for stage in stage_names} for mode in modes}
    gap_total = 0
    final_labels, final_scores = [], []
    no_positive_tiles = candidate_tiles = 0
    first_positive = None
    per_tile = []
    per_gap = []
    gap_categories = defaultdict(int)
    unique_gap_categories = defaultdict(int)
    seen_gap_keys = set()
    oracle_summary = {'tiles': 0, 'candidates': 0, 'positive_logits': 0,
                      'closed_gaps': 0, 'baseline_closed_gaps': 0,
                      'centerline_recovered_fraction_sum': 0., 'baseline_centerline_recovered_fraction_sum': 0.,
                      'oracle_added_gt_pixels': 0, 'oracle_added_background_pixels': 0,
                      'feature_delta_l2_sum': 0., 'feature_input_l2_sum': 0.,
                      'probability_abs_delta_sum': 0., 'probability_pixels': 0}
    native_writeback = {'tiles': 0, 'gamma': float(model.swin_unet.gap_connection.gamma.detach().cpu()),
                        'saved_writeback_scale': float(model.swin_unet.gap_connection.writeback_scale.detach().cpu()),
                        'delta_l2_sum': 0., 'decoder_l2_sum': 0., 'residual_abs_mean_sum': 0.}
    capture = []

    def hook(_, inputs, outputs):
        capture.append((inputs, outputs[0], outputs[1]))

    handle = model.swin_unet.gap_connection.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            count = len(dataset) if args.max_images == 0 else min(len(dataset), args.max_images)
            for index in range(count):
                sample = dataset[index]
                image = sample['image'][None].to(args.device)
                mask = sample['mask'][0].numpy() > .5
                skeleton = sample['skeleton'][0].numpy() > .5
                tile_index = 0
                for top in sliding_positions(1500, 512, 256):
                    for left in sliding_positions(1500, 512, 256):
                        if args.max_tiles_per_image and tile_index >= args.max_tiles_per_image:
                            break
                        tile_index += 1
                        capture.clear()
                        outputs = model(image[:, :, top:top+512, left:left+512])
                        if len(capture) != 1:
                            raise RuntimeError(f'Expected one Gap call, got {len(capture)}')
                        gap_inputs, gap_decoder_out, record = capture[0]
                        native_delta = gap_decoder_out.detach().float() - gap_inputs[0].detach().float()
                        native_writeback['tiles'] += 1
                        native_writeback['delta_l2_sum'] += float(native_delta.norm().cpu())
                        native_writeback['decoder_l2_sum'] += float(gap_inputs[0].detach().float().norm().cpu())
                        native_writeback['residual_abs_mean_sum'] += float(record['gap_residual_abs_mean'].cpu())
                        probability = record['p2_logits'][0, 0].sigmoid().cpu().numpy()
                        surface_probability = outputs[0][0, 0].sigmoid().cpu().numpy()
                        gt_mask = mask[top:top+512, left:left+512]
                        gt_skeleton = skeleton[top:top+512, left:left+512]
                        p2_mask = cv2.resize((probability >= args.reference_threshold).astype(np.uint8),
                                             (512, 512), interpolation=cv2.INTER_NEAREST)
                        gaps, _ = verified_gaps(gt_skeleton, p2_mask > 0, 2, 20, 3)
                        gap_total += len(gaps)
                        pred_endpoints, pred_components, pred_skeleton = prediction_endpoints(
                            surface_probability, args.surface_threshold)
                        for gap_id, gap in enumerate(gaps):
                            category, flank_endpoint_counts, matched_pairs, path_ratio = classify_gap(
                                gap, pred_skeleton, pred_endpoints, pred_components, cfg.gt_snap_radius)
                            gap_categories[category] += 1
                            _, (gy0, gy1, gx0, gx1), gap_region, _, _, _ = gap
                            gap_pixels = np.argwhere(gap_region)
                            global_gap_pixels = gap_pixels + [top+gy0, left+gx0]
                            first_pixel = global_gap_pixels.min(axis=0)
                            gap_key = (Path(sample['image_name']).stem,
                                       int(first_pixel[0]), int(first_pixel[1]))
                            if gap_key not in seen_gap_keys:
                                seen_gap_keys.add(gap_key)
                                unique_gap_categories[category] += 1
                            per_gap.append({'image': sample['image_name'], 'top': top, 'left': left,
                                            'gap_id': gap_id, 'pixels': gap[-1],
                                            'approx_unique_key': f'{gap_key[0]}:{gap_key[1]}:{gap_key[2]}',
                                            'endpoint_class': category,
                                            'matched_endpoints_per_flank': json.dumps(flank_endpoint_counts),
                                            'matched_flank_pairs': matched_pairs,
                                            'same_component_path_ratio': path_ratio})
                        scale = np.asarray(gt_mask.shape) / np.asarray(probability.shape)
                        high, _ = endpoints(probability, replace(cfg, threshold=.5))
                        high_set = {item[0] for item in high}
                        low, _ = endpoints(probability, replace(cfg, threshold=.3))
                        # Union is deterministic; high-confidence endpoints remain available.
                        dual = high + [item for item in low if item[0] not in high_set]
                        surface_cfg = replace(cfg, min_distance=cfg.min_distance*4,
                                              max_distance=cfg.max_distance*4,
                                              boundary_margin=cfg.boundary_margin*4,
                                              min_component=cfg.min_component*4,
                                              trace_steps=cfg.trace_steps*4,
                                              min_trace=cfg.min_trace*4)
                        surface_points, _ = endpoints(surface_probability, surface_cfg)
                        point_sets = {'p2_05': (high, False), 'p2_03': (low, False),
                                      'dual_05_03': (dual[:cfg.max_endpoints], True),
                                      'surface_512': (surface_points, False)}
                        tile_row = {'image': sample['image_name'], 'top': top, 'left': left,
                                    'verified_gt_gaps': len(gaps),
                                    'gap_gamma': native_writeback['gamma'],
                                    'gap_writeback_scale': native_writeback['saved_writeback_scale'],
                                    'gap_residual_abs_mean': float(record['gap_residual_abs_mean'].cpu()),
                                    'gap_delta_relative_l2': float(native_delta.norm().cpu()) /
                                        max(float(gap_inputs[0].detach().float().norm().cpu()), 1e-12)}
                        for mode, (points, is_dual) in point_sets.items():
                            candidate_cfg = surface_cfg if mode == 'surface_512' else cfg
                            candidate_scale = np.ones(2) if mode == 'surface_512' else scale
                            stages = ranked_pairs(points, candidate_cfg, is_dual, high_set)
                            for stage, edges in stages.items():
                                covered_flags = [gap_covered(gap, edges, candidate_scale, cfg.gt_snap_radius)
                                                 for gap in gaps]
                                covered = sum(covered_flags)
                                totals[mode][stage]['candidates'] += len(edges)
                                totals[mode][stage]['covered_gaps'] += covered
                                if any(covered_flags):
                                    unique_gap_coverage[mode][stage].add(gap_key)
                                tile_row[f'{mode}_{stage}_candidates'] = len(edges)
                                tile_row[f'{mode}_{stage}_covered'] = covered
                        edges = record['gap_edges'][0].cpu().numpy()
                        replay = ranked_pairs(high, cfg)['top64']
                        if not np.array_equal(edges, replay):
                            raise RuntimeError('Replay differs from production candidates')
                        labels, _ = label_edges(gt_skeleton, gt_mask, edges, probability.shape, cfg)
                        logits = record['gap_logits'][0].cpu().numpy()
                        final_labels.extend(labels.tolist()); final_scores.extend(logits.tolist())
                        candidate_tiles += 1
                        no_positive_tiles += int(not (labels == 1).any())
                        if first_positive is None and (labels == 1).any():
                            first_positive = (index, top, left)
                        tile_row.update(final_positive=int((labels == 1).sum()),
                                        final_negative=int((labels == 0).sum()),
                                        final_ignored=int((labels == -1).sum()))
                        per_tile.append(tile_row)
                        if args.gt_oracle:
                            oracle_edges, oracle_geometry = oracle_proposals(
                                gaps, probability.shape, cfg, gt_skeleton, gt_mask)
                            if len(oracle_edges):
                                original_generator = gap_module.generate_candidates
                                gap_module.generate_candidates = lambda _p, _c: [{
                                    'edges': oracle_edges, 'geometry': oracle_geometry,
                                    'endpoint_count': len(oracle_edges)*2,
                                    'retained_endpoints': len(oracle_edges)*2,
                                    'eligible_pairs': len(oracle_edges)}]
                                try:
                                    capture.clear()
                                    oracle_outputs = model(image[:, :, top:top+512, left:left+512])
                                finally:
                                    gap_module.generate_candidates = original_generator
                                oracle_inputs, oracle_decoder_out, oracle_record = capture[-1]
                                oracle_logits = oracle_record['gap_logits'][0].cpu().numpy()
                                oracle_surface_logits = oracle_outputs[0][0, 0].float().cpu().numpy()
                                base_surface_logits = outputs[0][0, 0].float().cpu().numpy()
                                base_mask = surface_probability >= args.surface_threshold
                                oracle_probability = 1 / (1 + np.exp(-np.clip(oracle_surface_logits, -80, 80)))
                                oracle_mask = oracle_probability >= args.surface_threshold
                                baseline_gap_rows = gap_metrics(gaps, base_mask, base_surface_logits,
                                                                base_surface_logits, 2)
                                oracle_gap_rows = gap_metrics(gaps, oracle_mask, base_surface_logits,
                                                              oracle_surface_logits, 2)
                                oracle_summary['tiles'] += 1
                                oracle_summary['candidates'] += len(oracle_edges)
                                oracle_summary['positive_logits'] += int((oracle_logits >= 0).sum())
                                oracle_summary['closed_gaps'] += sum(r['complete_closure'] for r in oracle_gap_rows)
                                oracle_summary['baseline_closed_gaps'] += sum(r['complete_closure'] for r in baseline_gap_rows)
                                oracle_summary['centerline_recovered_fraction_sum'] += sum(
                                    r['centerline_recovered_fraction'] for r in oracle_gap_rows)
                                oracle_summary['baseline_centerline_recovered_fraction_sum'] += sum(
                                    r['centerline_recovered_fraction'] for r in baseline_gap_rows)
                                oracle_summary['oracle_added_gt_pixels'] += int((oracle_mask & ~base_mask & gt_mask).sum())
                                oracle_summary['oracle_added_background_pixels'] += int(
                                    (oracle_mask & ~base_mask & ~gt_mask).sum())
                                delta = oracle_decoder_out.detach().float() - oracle_inputs[0].detach().float()
                                oracle_summary['feature_delta_l2_sum'] += float(delta.norm().cpu())
                                oracle_summary['feature_input_l2_sum'] += float(oracle_inputs[0].detach().float().norm().cpu())
                                oracle_summary['probability_abs_delta_sum'] += float(
                                    np.abs(oracle_probability-surface_probability).sum())
                                oracle_summary['probability_pixels'] += int(oracle_probability.size)
                                per_tile[-1].update(oracle_candidates=len(oracle_edges),
                                                    oracle_positive_logits=int((oracle_logits >= 0).sum()),
                                                    oracle_closed=sum(r['complete_closure'] for r in oracle_gap_rows),
                                                    baseline_closed=sum(r['complete_closure'] for r in baseline_gap_rows))
                    if args.max_tiles_per_image and tile_index >= args.max_tiles_per_image:
                        break
                print(f'[GAP-CANDIDATES] {index+1}/{count} images; tiles={candidate_tiles}; GT gaps={gap_total}', flush=True)
    finally:
        handle.remove()
    with (args.output_dir / 'per_tile.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=per_tile[0].keys())
        writer.writeheader(); writer.writerows(per_tile)
    if per_gap:
        with (args.output_dir / 'per_gap.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=per_gap[0].keys())
            writer.writeheader(); writer.writerows(per_gap)
    for mode in modes:
        for stage in stage_names:
            slot = totals[mode][stage]
            slot['recall'] = slot['covered_gaps'] / gap_total if gap_total else None
            slot['candidates_per_tile'] = slot['candidates'] / candidate_tiles
            slot['approx_unique_covered_gaps'] = len(unique_gap_coverage[mode][stage])
            slot['approx_unique_recall'] = (slot['approx_unique_covered_gaps'] / len(seen_gap_keys)
                                             if seen_gap_keys else None)
    labels = np.asarray(final_labels)
    score = np.asarray(final_scores)
    summary = {'checkpoint': args.model_path, 'split': args.split, 'images': count,
               'tiles': candidate_tiles, 'reference_p2_threshold': args.reference_threshold,
               'verified_gt_gaps': gap_total, 'modes': totals,
               'final_head': {'candidates': len(labels), 'positive': int((labels == 1).sum()),
                              'negative': int((labels == 0).sum()), 'ignored': int((labels == -1).sum()),
                              'no_positive_tile_fraction': no_positive_tiles / candidate_tiles,
                              'average_precision': average_precision(labels, score),
                              'positive_logit_quantiles': np.quantile(score[labels == 1], [.1, .5, .9]).tolist()
                              if (labels == 1).any() else None,
                              'negative_logit_quantiles': np.quantile(score[labels == 0], [.1, .5, .9]).tolist()
                              if (labels == 0).any() else None},
               'training_history': training_history(Path(args.model_path).parent / 'local_gap_audit.jsonl'),
               'gap_endpoint_classification': {'overlapping_tile_instances': dict(gap_categories),
                                               'approx_unique_gaps': dict(unique_gap_categories),
                                               'unique_count': len(seen_gap_keys)},
               'native_gap_writeback': dict(native_writeback,
                   mean_delta_relative_l2=native_writeback['delta_l2_sum'] /
                       max(native_writeback['decoder_l2_sum'], 1e-12),
                   mean_residual_abs_mean=native_writeback['residual_abs_mean_sum'] /
                       max(native_writeback['tiles'], 1)),
               'surface_threshold_for_endpoint_audit': args.surface_threshold,
               'definition': 'Fixed GT gaps are verified from P2 at the reference threshold. An edge covers a gap '
                             'when its endpoints match opposite GT flanks within gt_snap_radius; this is '
                             'candidate availability, not a successful surface repair. Tiles overlap. '
                             'Surface512 uses the same maximum endpoints and edges, with spatial distances '
                             'and trace length scaled by four.',
               'scope': 'Frozen checkpoint; candidate thresholds are diagnostic only; no test selection. '
                        'Tile gap counts include overlap duplicates; approx_unique_gaps deduplicates by image and '
                        'global first missing-skeleton pixel and is an approximate key.'}
    if args.gt_oracle:
        oracle_summary['mean_feature_delta_to_input_ratio'] = (
            oracle_summary['feature_delta_l2_sum'] / max(oracle_summary['feature_input_l2_sum'], 1e-12))
        oracle_summary['mean_probability_abs_delta'] = (
            oracle_summary['probability_abs_delta_sum'] / max(oracle_summary['probability_pixels'], 1))
        oracle_summary['oracle_logit_positive_fraction'] = (
            oracle_summary['positive_logits'] / max(oracle_summary['candidates'], 1))
        oracle_summary['scope'] = ('GT supplies geometrically positive local connections as proposals; they still pass '
                                   'the trained Gap MLP and learned Surface writeback. This is an optimistic '
                                   'diagnostic and uses overlapping tiles.')
        summary['gt_oracle'] = oracle_summary
    if args.gradient_audit:
        with torch.enable_grad():
            location = first_positive or (0, 0, 0)
            summary['gradient_audit'] = gradient_audit(model, dataset[location[0]], cfg,
                                                       args.device, *location[1:])
            summary['gradient_audit']['tile'] = {'image_index': location[0],
                                                 'top': location[1], 'left': location[2]}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
