"""Frozen P2 candidate recall and Gap head audit; never selects a test threshold."""
import argparse
from collections import defaultdict
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
from gap_diagnostic_metrics import verified_gaps
from local_gap_config import LocalGapConfig
from losses.gap_losses import label_edges
from mass_h0 import verify_checkpoint
from mass_protocol import sliding_positions
from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state


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
            distance = np.linalg.norm(native[:, :, None, :] - coords[None, None, :, :], axis=-1)
            hits.append((distance <= snap_radius).any(axis=-1))
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
    args = parser.parse_args()
    if not 0 < args.reference_threshold < 1:
        parser.error('reference_threshold must be in (0,1)')
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
    gap_total = 0
    final_labels, final_scores = [], []
    no_positive_tiles = candidate_tiles = 0
    first_positive = None
    per_tile = []
    capture = []

    def hook(_, inputs, outputs):
        capture.append(outputs[1])

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
                        record = capture[0]
                        probability = record['p2_logits'][0, 0].sigmoid().cpu().numpy()
                        surface_probability = outputs[0][0, 0].sigmoid().cpu().numpy()
                        gt_mask = mask[top:top+512, left:left+512]
                        gt_skeleton = skeleton[top:top+512, left:left+512]
                        p2_mask = cv2.resize((probability >= args.reference_threshold).astype(np.uint8),
                                             (512, 512), interpolation=cv2.INTER_NEAREST)
                        gaps, _ = verified_gaps(gt_skeleton, p2_mask > 0, 2, 20, 3)
                        gap_total += len(gaps)
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
                                    'verified_gt_gaps': len(gaps)}
                        for mode, (points, is_dual) in point_sets.items():
                            candidate_cfg = surface_cfg if mode == 'surface_512' else cfg
                            candidate_scale = np.ones(2) if mode == 'surface_512' else scale
                            stages = ranked_pairs(points, candidate_cfg, is_dual, high_set)
                            for stage, edges in stages.items():
                                covered = sum(gap_covered(gap, edges, candidate_scale, cfg.gt_snap_radius)
                                              for gap in gaps)
                                totals[mode][stage]['candidates'] += len(edges)
                                totals[mode][stage]['covered_gaps'] += covered
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
                    if args.max_tiles_per_image and tile_index >= args.max_tiles_per_image:
                        break
                print(f'[GAP-CANDIDATES] {index+1}/{count} images; tiles={candidate_tiles}; GT gaps={gap_total}', flush=True)
    finally:
        handle.remove()
    with (args.output_dir / 'per_tile.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=per_tile[0].keys())
        writer.writeheader(); writer.writerows(per_tile)
    for mode in modes:
        for stage in stage_names:
            slot = totals[mode][stage]
            slot['recall'] = slot['covered_gaps'] / gap_total if gap_total else None
            slot['candidates_per_tile'] = slot['candidates'] / candidate_tiles
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
               'definition': 'Fixed GT gaps are verified from P2 at the reference threshold. An edge covers a gap '
                             'when its endpoints match opposite GT flanks within gt_snap_radius; this is '
                             'candidate availability, not a successful surface repair. Tiles overlap. '
                             'Surface512 uses the same maximum endpoints and edges, with spatial distances '
                             'and trace length scaled by four.',
               'scope': 'Frozen checkpoint; candidate thresholds are diagnostic only; no test selection'}
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
