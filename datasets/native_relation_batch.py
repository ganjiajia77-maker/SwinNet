"""Pack the unchanged native_targets output for DataLoader collation."""
import hashlib

import numpy as np
import torch

from datasets.native_relation_targets import native_targets


TARGET_KEYS = ('connectivity', 'connectivity_valid', 'direction', 'direction_valid')


def pack_native_sample(mask, skeleton):
    """Use final CPU tensors; never reconstruct a skeleton or consume RNG."""
    if mask.device.type != 'cpu' or skeleton.device.type != 'cpu':
        raise ValueError('Dataset native targets require CPU tensors')
    if mask.ndim != 3 or mask.shape[0] != 1 or mask.shape != skeleton.shape:
        raise ValueError('Expected aligned [1,H,W] final road/skeleton tensors')
    row = native_targets(mask[0].detach().numpy() > .5,
                         skeleton[0].detach().numpy() > .5)
    targets = {key: torch.from_numpy(row[key]) for key in TARGET_KEYS}
    targets['direction_valid'] = targets['direction_valid'][None]
    stats = dict(
        valid_positive_edges=int(((row['connectivity'] > .5) & row['connectivity_valid']).sum()),
        valid_negative_edges=int(((row['connectivity'] < .5) & row['connectivity_valid']).sum()),
        ignored_edges=int((~row['connectivity_valid']).sum()),
        valid_direction_cells=int(row['direction_valid'].sum()),
        ambiguous_cells=int(row['ambiguous_cells'].sum()),
    )
    return targets, stats


def reduce_native_stats(stats):
    result = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            if value.device.type != 'cpu':
                raise ValueError('Keep native statistics on CPU to avoid GPU synchronization')
            result[key] = int(value.sum())
        else:
            result[key] = int(value)
    return result


def legacy_native_batch(mask, skeleton):
    """Reference loss-time path, also used by unmodified sliding validation."""
    from torch.utils.data._utils.collate import default_collate
    samples = [pack_native_sample(mask[i].detach().cpu(), skeleton[i].detach().cpu())
               for i in range(mask.shape[0])]
    targets, stats = default_collate(samples)
    return targets, reduce_native_stats(stats)


def transfer_native_targets(targets, mask):
    b, _, h, w = mask.shape
    if h % 4 or w % 4:
        raise ValueError('Native supervision requires dimensions divisible by four')
    shapes = {'connectivity': (b, 8, h//4, w//4),
              'connectivity_valid': (b, 8, h//4, w//4),
              'direction': (b, 2, h//4, w//4),
              'direction_valid': (b, 1, h//4, w//4)}
    for key, shape in shapes.items():
        expected_dtype = torch.bool if key.endswith('valid') else torch.float32
        if targets[key].shape != shape or targets[key].dtype != expected_dtype:
            raise ValueError(f'Invalid native target {key}: {targets[key].shape}, {targets[key].dtype}')
    return {key: targets[key].to(mask.device, non_blocking=True) for key in TARGET_KEYS}


def verify_native_batch(batch, mask, skeleton):
    """One explicit reference round-trip; fail on any differing element/statistic."""
    reference, stats = legacy_native_batch(mask, skeleton)
    candidate_stats = reduce_native_stats(batch['native_relation_stats'])
    if stats != candidate_stats:
        raise RuntimeError('Loader/reference native statistics differ')
    hashes = {}
    elements = 0
    for key in TARGET_KEYS:
        candidate = batch['native_relations'][key]
        if candidate.device.type != 'cpu' or not torch.equal(candidate, reference[key]):
            raise RuntimeError(f'Loader/reference native labels differ: {key}')
        hashes[key] = hashlib.sha256(candidate.contiguous().numpy().tobytes()).hexdigest()
        elements += candidate.numel()
    return dict(passed=True, elements_checked=elements, target_sha256=hashes, statistics=stats)
