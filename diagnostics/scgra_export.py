"""Read-only anchor metrics and paired forward capture for SCGRA figures."""
from contextlib import contextmanager
import hashlib

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import torch


METHODS = ('structure_fps', 'probability_fps', 'random')


def seeded_rng(seed, case, top, left, repeat):
    value = f'{seed}|{case}|{top}|{left}|{repeat}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(value).digest()[:8], 'little'))


def grid_to_pixels(coords, grid_hw, tile_hw, offset=(0, 0)):
    """Nominal feature-cell centres, y/x order; never pixel endpoints."""
    coords = np.asarray(coords)
    return ((coords.astype(np.float64)+.5)*np.asarray(tile_hw)/np.asarray(grid_hw)
            +np.asarray(offset))


def point_metrics(coords, gt, labels, component_lengths, distance, tolerance, present=None):
    """One full image or window. Component coverage uses strict GT hits only."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    unique = np.unique(coords, axis=0)
    height, width = gt.shape
    if len(unique) and (np.any(unique < 0) or np.any(unique[:, 0] >= height) or np.any(unique[:, 1] >= width)):
        raise ValueError('Anchor coordinates outside the evaluated image/window')
    # Feature-cell centres in this protocol are integer (4*k+2). Floor is
    # explicitly defined for future noninteger centres and recorded in metadata.
    pixels = np.floor(unique).astype(np.int64)
    ids = labels[pixels[:, 0], pixels[:, 1]] if len(pixels) else np.empty(0, np.int64)
    if present is None:
        present = np.unique(labels)
        present = present[present > 0]
    covered = np.unique(ids[ids > 0])
    lengths = component_lengths[present].sum() if len(present) else 0
    exact = int((ids > 0).sum())
    tolerant = int((distance[pixels[:, 0], pixels[:, 1]] <= tolerance).sum()) if len(pixels) else 0
    nearest = cKDTree(unique).query(unique, k=2)[0][:, 1] if len(unique) >= 2 else np.empty(0)
    hull_area = float(cv2.contourArea(cv2.convexHull(unique[:, ::-1].astype(np.float32)))) if len(unique) >= 3 else 0.
    return dict(anchor_events=len(coords), unique_anchor_locations=len(unique),
        gt_component_count=len(present), covered_gt_components=len(covered),
        road_hit_count_exact=exact, road_hit_count_tolerant=tolerant,
        road_hit_rate_exact=exact/len(unique) if len(unique) else None,
        road_hit_rate_tolerant=tolerant/len(unique) if len(unique) else None,
        nearest_neighbor_mean_px=float(nearest.mean()) if len(nearest) else None,
        nearest_neighbor_min_px=float(nearest.min()) if len(nearest) else None,
        convex_hull_area_fraction=hull_area/(height*width),
        component_coverage= len(covered)/len(present) if len(present) else None,
        skeleton_length_weighted_component_coverage=float(component_lengths[covered].sum()/lengths) if lengths else None)


def prepare_gt(gt, skeletonize):
    _, labels = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    skeleton = skeletonize(gt)
    lengths = np.bincount(labels[skeleton].ravel(), minlength=int(labels.max())+1)
    distance = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
    return labels, lengths, distance, skeleton


@contextmanager
def trace_module(module):
    flags = {key: getattr(module, key, None) for key in
             ('capture_anchor_selection', 'capture_scgra_trace', 'last_anchor_selection',
              'last_scgra_trace', '_scgra_node_attention', '_scgra_trace_calls')}
    existed = {key: key in module.__dict__ for key in flags}
    module.capture_anchor_selection = True
    module.capture_scgra_trace = True
    module.last_anchor_selection = None
    module.last_scgra_trace = None
    module._scgra_trace_calls = 0
    try:
        yield
    finally:
        for key, value in flags.items():
            if existed[key]:
                setattr(module, key, value)
            elif key in module.__dict__:
                delattr(module, key)


@contextmanager
def zero_global_alpha(module):
    original = module.raw_alpha.detach().clone()
    with torch.no_grad():
        module.raw_alpha.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            module.raw_alpha.copy_(original)


def selection_controls(module, selection, trace, seed, case, top, left, repeats):
    """Prediction-only controls. GT is deliberately absent from this API."""
    valid = selection['selected_valid'][0].numpy().astype(bool)
    actual = selection['selected_coords'][0].numpy()[valid].copy()
    count = len(actual)
    road_prior = trace['road_prior'].to(device=module.raw_alpha.device, dtype=module.raw_alpha.dtype)
    saved_selection = module.last_anchor_selection
    try:
        # Reuse the real top-k/FPS implementation; match the actual valid K.
        coords, prob_valid, _, _ = module._extract_fps_anchors(road_prior.clamp(0, 1))
        probability_selection = module.last_anchor_selection
        probability = coords[0][prob_valid[0]].detach().cpu().numpy()[:count].copy()
    finally:
        module.last_anchor_selection = saved_selection
    if len(probability) != count:
        raise RuntimeError('Probability-only control could not match the actual anchor count')
    height, width = trace['feature_hw']
    random = []
    for repeat in range(repeats):
        indices = seeded_rng(seed, case, top, left, repeat).choice(height*width, count, replace=False)
        random.append(np.stack([indices//width, indices%width], axis=1))
    return actual, probability, random, probability_selection


@torch.inference_mode()
def paired_image(model, image, device, case, seed, repeats, on_window,
                 paired=True, verify_capture=False, tile=512, stride=256):
    """Normal ON predictions and optional alpha=0 OFF, same checkpoint/windows.

    Callback consumes detached CPU traces immediately; nonexported images do not
    retain spatial traces. No labels or GT are accepted by this forward API.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Expected CHW RGB image')
    image = image[None].to(device)
    height, width = image.shape[-2:]
    canvas = torch.zeros((1, 1, height, width), device=device)
    off_canvas = torch.zeros_like(canvas) if paired else None
    divisor = torch.zeros_like(canvas)
    ramp = (1-torch.linspace(-1., 1., tile, device=device).abs()).clamp_min(.1)
    weight = torch.outer(ramp, ramp)[None, None]
    from datasets.dataset_road_skeleton import RoadSkeletonDataset
    module = model.swin_unet.global_topology
    window_index, checked = 0, False
    for top in RoadSkeletonDataset.sliding_positions(height, tile, stride):
        for left in RoadSkeletonDataset.sliding_positions(width, tile, stride):
            rgb = image[:, :, top:top+tile, left:left+tile]
            normal = model(rgb)[0].clone() if verify_capture and not checked else None
            with trace_module(module):
                logits = model(rgb)[0]
                if (module.last_anchor_selection is None or module.last_scgra_trace is None
                        or module._scgra_trace_calls != 1):
                    raise RuntimeError('Expected one active Stage2 global trace per window')
                selection, trace = module.last_anchor_selection, module.last_scgra_trace
                if normal is not None:
                    if not torch.equal(normal, logits):
                        raise RuntimeError('Read-only capture changed normal surface logits')
                    checked = True
                controls = selection_controls(module, selection, trace, seed, case, top, left, repeats)
                on_window(window_index, top, left, tile, selection, trace, controls)
            if logits.shape != (1, 1, tile, tile) or not torch.isfinite(logits).all():
                raise ValueError('Invalid ON surface output')
            region = (slice(None), slice(None), slice(top, top+tile), slice(left, left+tile))
            canvas[region] += logits.float()*weight
            if paired:
                with zero_global_alpha(module):
                    off = model(rgb)[0]
                if off.shape != logits.shape or not torch.isfinite(off).all():
                    raise ValueError('Invalid OFF surface output')
                off_canvas[region] += off.float()*weight
            divisor[region] += weight
            window_index += 1
    if not bool((divisor > 0).all()):
        raise RuntimeError('Uncovered native pixels')
    on = (canvas/divisor.clamp_min(1e-8)).sigmoid()[0, 0].cpu().numpy()
    off = (off_canvas/divisor.clamp_min(1e-8)).sigmoid()[0, 0].cpu().numpy() if paired else None
    return on, off, window_index, checked
