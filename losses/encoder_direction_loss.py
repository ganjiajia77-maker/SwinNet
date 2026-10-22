"""PCA targets on the full-resolution augmented crop skeleton; one added loss."""
import math

import torch
from torch.nn import functional as F

DEFAULT_DIRECTION_CONFIG = dict(pca_radius=5, min_support=5, road_radius=3,
                                min_coherence=.1, lambda_encoder_direction=.1)


def validate_direction_config(config):
    result = dict(DEFAULT_DIRECTION_CONFIG)
    result.update(config or {})
    if set(result) != set(DEFAULT_DIRECTION_CONFIG):
        raise ValueError('Unknown encoder direction configuration')
    for key in ('pca_radius', 'min_support', 'road_radius'):
        if not isinstance(result[key], int) or result[key] < (0 if key == 'road_radius' else 1):
            raise ValueError('Invalid direction configuration: ' + key)
    if (not 0 < result['min_coherence'] <= 1 or not math.isfinite(result['lambda_encoder_direction'])
            or result['lambda_encoder_direction'] < 0):
        raise ValueError('Invalid encoder direction weight/coherence')
    return result


@torch.no_grad()
def direction_target(skeleton, road, size, config):
    """Vectorized local PCA using relative x/y moments; no eigensolver needed."""
    cfg = validate_direction_config(config)
    skel = (skeleton > .5).float()
    radius = cfg['pca_radius']
    axis = torch.arange(-radius, radius + 1, device=skel.device, dtype=skel.dtype)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    kernel = torch.stack((torch.ones_like(x), x, y, x*x, y*y, x*y))[:, None]
    moments = F.conv2d(skel, kernel, padding=radius)
    count = moments[:, :1]
    mean_x, mean_y = moments[:, 1:2] / count.clamp_min(1), moments[:, 2:3] / count.clamp_min(1)
    xx = (moments[:, 3:4] / count.clamp_min(1) - mean_x.square()).clamp_min(0)
    yy = (moments[:, 4:5] / count.clamp_min(1) - mean_y.square()).clamp_min(0)
    xy = moments[:, 5:6] / count.clamp_min(1) - mean_x * mean_y
    double = torch.cat((xx - yy, 2 * xy), 1)
    separation = double.norm(dim=1, keepdim=True)
    reliability = (separation / (xx + yy).clamp_min(1e-6)).clamp(0, 1)
    near = F.max_pool2d(skel, 2*cfg['road_radius']+1, stride=1, padding=cfg['road_radius']) > .5
    valid = (count >= cfg['min_support']) & near & (road > .5) & (separation > 1e-6)
    weight = reliability * valid
    double = F.normalize(double, dim=1, eps=1e-6)
    # Average double-angle VECTORS weighted by m*q_GT, never angles or thin labels.
    vector_sum = F.adaptive_avg_pool2d(double * weight, size)
    support = F.adaptive_avg_pool2d(weight, size)
    coherence = (vector_sum.norm(dim=1, keepdim=True) / support.clamp_min(1e-8)).clamp(0, 1)
    effective_weight = support * coherence * (coherence >= cfg['min_coherence'])
    target = F.normalize(vector_sum, dim=1, eps=1e-8)
    return target, effective_weight


def encoder_direction_loss(raw, skeleton, road, config):
    target, weight = direction_target(skeleton, road, raw.shape[-2:], config)
    cosine = (F.normalize(raw.float(), dim=1, eps=1e-6) * target).sum(1, keepdim=True)
    loss = ((1 - cosine.clamp(-1, 1)) * weight).sum() / weight.sum().clamp_min(1e-8)
    return loss, (weight > 0).sum().detach(), weight.sum().detach()
