"""Frozen H0 surface refinement without endpoint or gap candidates."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleFeatures(nn.Module):
    def __init__(self, in_channels, width=32):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(in_channels, width, 1, bias=False),
                                     nn.GroupNorm(8, width), nn.GELU())
        self.branches = nn.ModuleList(SeparableConv(width, d) for d in (1, 2, 4, 8))
        self.fuse = nn.Sequential(nn.Conv2d(width * 4, width, 1, bias=False),
                                  nn.GroupNorm(8, width), nn.GELU())

    def forward(self, x):
        x = self.project(x)
        return x + self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class Q3R3Repair(nn.Module):
    def __init__(self, base_model, eta=0.5, width=32, enable_stage2_local_propagation=False,
                 enable_gap_refinement=False, enable_q3_r3=True):
        super().__init__()
        if eta <= 0:
            raise ValueError('eta must be positive')
        self.base = base_model.requires_grad_(False)
        self.base.swin_unet.decoder_structure_blocks['2'].enable_local_propagation = bool(
            enable_stage2_local_propagation)
        self.base.swin_unet.enable_gap_refinement = bool(enable_gap_refinement)
        stage3 = self.base.swin_unet.decoder_structure_blocks['3']
        stage_channels = stage3.structure_branch[0].block[0].in_channels
        highres_channels = self.base.swin_unet.highres_structure_channels
        self.q_features = MultiScaleFeatures(stage_channels + highres_channels + 10, width)
        self.q_head = nn.Conv2d(width, 1, 1)
        self.r_features = MultiScaleFeatures(width + 2, width)
        self.r_head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.r_head.weight)
        nn.init.zeros_(self.r_head.bias)
        self.eta = float(eta)
        self.enable_q3_r3 = bool(enable_q3_r3)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def refine_with_q(self, z0, features, p0_q, q):
        if q.shape != p0_q.shape or q.shape[1] != 1:
            raise ValueError('Q and the aligned initial surface must have matching shapes')
        repair_features = self.r_features(torch.cat((features, q, p0_q), dim=1))
        delta = self.r_head(repair_features)
        gated_delta = (0.2 + 0.8 * q) * delta
        z1 = z0 + self.eta * F.interpolate(gated_delta, size=z0.shape[-2:],
                                           mode='bilinear', align_corners=False)
        return z1, delta

    def forward(self, image, return_aux=False):
        captured = []

        def stage3_hook(module, inputs, output):
            captured.append(output[0].detach())

        handle = self.base.swin_unet.decoder_structure_blocks['3'].register_forward_hook(stage3_hook)
        try:
            with torch.no_grad():
                base_outputs = self.base(image)
        finally:
            handle.remove()
        if len(captured) != 1 or not isinstance(base_outputs, tuple):
            raise RuntimeError('Expected exactly one Stage 3 structure call and tuple outputs')
        z0 = base_outputs[0].detach()
        if not self.enable_q3_r3:
            zero_q = z0.new_zeros((z0.shape[0], 1, *captured[0].shape[-2:]))
            return {'z0': z0, 'z1': z0, 'q_logits': zero_q, 'q': zero_q,
                    'delta': zero_q, 'stage2_s': zero_q,
                    'stage2_c': zero_q.expand(-1, 8, -1, -1)}
        structure = base_outputs[-1]
        stage2 = next((item for item in structure if item.get('stage') == 2
                       and item.get('skeleton') is not None), None)
        if stage2 is None or stage2.get('connectivity') is None:
            raise RuntimeError('Stage 2 S2/C2 outputs are missing')
        stage3_feature = captured[0]
        highres = self.base.swin_unet.last_stage_features.get('H2')
        if highres is None:
            raise RuntimeError('Updated high-resolution H feature is missing')
        size = highres.shape[-2:]

        def align(value):
            return F.interpolate(value.detach(), size=size, mode='bilinear',
                                 align_corners=False) if value.shape[-2:] != size else value.detach()

        p0 = z0.sigmoid()
        q_input = torch.cat((align(stage3_feature), align(highres), align(p0),
                             align(stage2['skeleton'].sigmoid()),
                             align(stage2['connectivity'].sigmoid())), dim=1)
        features = self.q_features(q_input)
        q_logits = self.q_head(features)
        q = q_logits.sigmoid()
        p0_q = align(p0)
        z1, delta = self.refine_with_q(z0, features, p0_q, q)
        result = {'z0': z0, 'z1': z1, 'q_logits': q_logits, 'q': q,
                  'delta': delta, 'stage2_s': stage2['skeleton'].detach(),
                  'stage2_c': stage2['connectivity'].detach()}
        if return_aux:
            result['q_features'] = features
            result['p0_q'] = p0_q
        return result


@torch.no_grad()
def make_q_target(skeleton, z0, output_size, min_segment_pixels=3, threshold=0.5):
    """Missed GT centerline is a training proxy, not a true gap oracle."""
    if skeleton.shape != z0.shape:
        raise ValueError('Skeleton and initial surface must share the training grid')
    nearby_p0 = F.max_pool2d(z0.sigmoid(), 5, stride=1, padding=2)
    missed = ((skeleton > 0.5) & (nearby_p0 < threshold)).cpu().numpy().astype(np.uint8)
    targets = []
    for binary in missed[:, 0]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        kept = np.zeros_like(binary)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= min_segment_pixels:
                kept[labels == index] = 1
        dilated = cv2.dilate(kept, np.ones((5, 5), np.uint8))
        targets.append(torch.from_numpy(dilated))
    target = torch.stack(targets, dim=0).to(device=z0.device, dtype=z0.dtype)[:, None]
    return F.adaptive_max_pool2d(target, output_size)


def dice_loss(logits, target):
    prob = logits.sigmoid()
    dims = (1, 2, 3)
    return (1 - (2 * (prob * target).sum(dims) + 1) /
            (prob.sum(dims) + target.sum(dims) + 1)).mean()


def repair_losses(result, mask, skeleton, lambda_q=0.1):
    z1 = result['z1']
    q_logits = result['q_logits']
    q_target = make_q_target(skeleton, result['z0'], q_logits.shape[-2:])
    bce = F.binary_cross_entropy_with_logits(z1, mask, reduction='none')
    pt = torch.exp(-bce)
    surface = (0.75 * (1 - pt).pow(2) * bce).mean() + 0.5 * dice_loss(z1, mask)
    q_bce = F.binary_cross_entropy_with_logits(q_logits, q_target, reduction='none')
    positive = q_target > 0.5
    negative = ~positive
    balanced = (q_bce[positive].mean() if positive.any() else q_bce.new_zeros(()))
    balanced = balanced * 0.5 + (q_bce[negative].mean() if negative.any()
                                  else q_bce.new_zeros(())) * 0.5
    q_loss = balanced + 0.5 * dice_loss(q_logits, q_target)
    total = surface + lambda_q * q_loss
    return total, {'surface': surface.detach(), 'q': q_loss.detach(),
                   'q_positive_fraction': q_target.mean().detach(),
                   'q_high_fraction': (result['q'] > 0.5).float().mean().detach()}
