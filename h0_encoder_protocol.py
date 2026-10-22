"""Native Massachusetts protocol shared by H0 training and inference."""
import math

import numpy as np
import torch


SOURCE_SIZE = 1500
TILE_SIZE = 512
STRIDE = 256
BASE_COMMIT = '599a410de8d341850e1810417307e4b3bbdfc82d'
MODEL_NAME = 'h0_encoder_road_skeleton_no_a1a2_v1'


def blend_weight(size, device):
    ramp = (1.0 - torch.linspace(-1.0, 1.0, size, device=device).abs()).clamp_min(0.1)
    return torch.outer(ramp, ramp).view(1, 1, size, size)


def sliding_positions(length, tile_size, stride):
    if length < tile_size or stride < 1 or stride > tile_size:
        raise ValueError('Invalid sliding-window size or stride')
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.no_grad()
def stitch_logits(model, image, tile_size=TILE_SIZE, stride=STRIDE):
    """One implementation for train-time validation, sweep, and test inference."""
    if image.shape != (1, 3, SOURCE_SIZE, SOURCE_SIZE) or tile_size != TILE_SIZE or stride != STRIDE:
        raise ValueError('Mass H0 stitching requires one native1500 RGB image and 512/256 windows')
    weight = blend_weight(tile_size, image.device)
    canvas = torch.zeros((1, 1, SOURCE_SIZE, SOURCE_SIZE), dtype=torch.float32, device=image.device)
    skeleton_canvas, divisor = torch.zeros_like(canvas), torch.zeros_like(canvas)
    has_skeleton = None
    for top in sliding_positions(SOURCE_SIZE, tile_size, stride):
        for left in sliding_positions(SOURCE_SIZE, tile_size, stride):
            outputs = model(image[:, :, top:top + tile_size, left:left + tile_size])
            surface = outputs[0] if isinstance(outputs, tuple) else outputs
            skeleton = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else None
            if surface.shape != (1, 1, tile_size, tile_size):
                raise ValueError(f'Unexpected surface shape: {surface.shape}')
            if has_skeleton is not None and has_skeleton != (skeleton is not None):
                raise ValueError('Skeleton output changed between windows')
            has_skeleton = skeleton is not None
            region = (slice(None), slice(None), slice(top, top + tile_size), slice(left, left + tile_size))
            canvas[region] += surface.float() * weight
            if skeleton is not None:
                if skeleton.shape != surface.shape:
                    raise ValueError('Skeleton and surface window shapes differ')
                skeleton_canvas[region] += skeleton.float() * weight
            divisor[region] += weight
    if bool((divisor <= 0).any()) or not bool(torch.isfinite(canvas).all()):
        raise ValueError('Sliding inference produced uncovered or nonfinite surface logits')
    result = canvas / divisor.clamp_min(1e-8)
    skeleton_result = skeleton_canvas / divisor.clamp_min(1e-8) if has_skeleton else None
    if skeleton_result is not None and not bool(torch.isfinite(skeleton_result).all()):
        raise ValueError('Sliding inference produced nonfinite skeleton logits')
    return result, skeleton_result


def crop_position(seed, image_index, crop_ordinal, epoch, height, width, size):
    rows, cols = height - size + 1, width - size + 1
    if rows < 1 or cols < 1:
        raise ValueError('Native image is smaller than the requested crop')
    total = rows * cols
    if total == 1:
        return 0, 0
    rng = np.random.RandomState((seed + image_index * 9176 + crop_ordinal * 101) % (2 ** 32))
    offset, step = int(rng.randint(total)), int(rng.randint(1, total))
    while math.gcd(step, total) != 1:
        step = (step + 1) % total or 1
    return divmod((offset + int(epoch) * step) % total, cols)


def validate_training_args(args):
    expected = dict(img_size=TILE_SIZE, source_patch_size=SOURCE_SIZE,
                    overlap_stride=STRIDE, random_crop_train=True,
                    random_crops_per_image=1, direct_resize_train=False,
                    max_epochs=80, batch_size=2, accumulation_steps=1,
                    amp_dtype='none', use_ema=False, trend_val_every_steps=0,
                    val_crop_list='', structure_profile='stage23_boundary_0626',
                    enable_highres_structure_stream=True, enable_global_topology=True,
                    enable_h3_surface_fusion=True, remove_stage2_pre_topology_source=True,
                    highres_structure_channels=64, highres_structure_fuse_stages='stage23',
                    highres_structure_fusion_mode='stage23', stage_skeleton_mode='direct')
    actual = vars(args) if hasattr(args, '__dict__') else args
    enabled = bool(actual.get('encoder_structure_exchange', False))
    expected.update(
        encoder_structure_exchange=enabled,
        encoder_road_aux_weight=0.10 if enabled else 0.0,
        encoder_skeleton_aux_weight=0.004 if enabled else 0.0,
        stage_skeleton_only_loss_factor=0.8 if enabled else 1.0,
        stage2_skeleton_weight=0.008,
        stage3_skeleton_weight=0.012,
    )
    mismatch = {key: (actual.get(key), value) for key, value in expected.items()
                if actual.get(key) != value}
    if mismatch:
        raise ValueError(f'Mass H0 protocol was overridden (actual, required): {mismatch}')
    return expected
