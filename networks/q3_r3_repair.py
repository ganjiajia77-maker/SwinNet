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


class DirectionalFeatures(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.stripes = nn.ModuleList()
        for dilation in (1, 2):
            for kernel in ((1, 7), (7, 1)):
                padding = tuple(dilation * (size // 2) for size in kernel)
                self.stripes.append(nn.Sequential(
                    nn.Conv2d(width, width, kernel, padding=padding,
                              dilation=dilation, groups=width, bias=False),
                    nn.Conv2d(width, width, 1, bias=False),
                    nn.GroupNorm(8, width), nn.GELU(),
                ))
        self.fuse = nn.Sequential(nn.Conv2d(width * 5, width, 1, bias=False),
                                  nn.GroupNorm(8, width), nn.GELU())

    def forward(self, x):
        return x + self.fuse(torch.cat([x] + [branch(x) for branch in self.stripes], dim=1))


class Q3R3Repair(nn.Module):
    def __init__(self, base_model, eta_plus=0.5, eta_minus=0.5, width=32,
                 enable_stage2_local_propagation=False,
                 enable_gap_refinement=False, enable_q3_r3=True):
        super().__init__()
        if eta_plus <= 0 or eta_minus <= 0:
            raise ValueError('Both repair scales must be positive')
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
        self.r_directional = DirectionalFeatures(width)
        self.r_plus_head = nn.Conv2d(width, 1, 1)
        self.r_minus_head = nn.Conv2d(width, 1, 1)
        for head, bias in ((self.r_plus_head, -4.), (self.r_minus_head, -6.)):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, bias)
        self.eta_plus = float(eta_plus)
        self.eta_minus = float(eta_minus)
        self.positive_gate_floor = 0.2
        self.enable_q3_r3 = bool(enable_q3_r3)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, image, capture_diagnostics=False):
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
                    'delta': zero_q, 'delta_plus': zero_q, 'delta_minus': zero_q,
                    'stage2_s': zero_q,
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
        repair_features = self.r_features(torch.cat((features, q, align(p0)), dim=1))
        repair_features = self.r_directional(repair_features)
        delta_plus = F.softplus(self.r_plus_head(repair_features))
        delta_minus = F.softplus(self.r_minus_head(repair_features))
        delta = (self.eta_plus * (self.positive_gate_floor +
                 (1. - self.positive_gate_floor) * q) * delta_plus -
                 self.eta_minus * delta_minus)
        z1 = z0 + F.interpolate(delta, size=z0.shape[-2:],
                                mode='bilinear', align_corners=False)
        result = {'z0': z0, 'z1': z1, 'q_logits': q_logits, 'q': q,
                'delta': delta, 'delta_plus': delta_plus, 'delta_minus': delta_minus,
                'stage2_s': stage2['skeleton'].detach(),
                'stage2_c': stage2['connectivity'].detach()}
        if capture_diagnostics:
            result.update(stage2_d=(stage2.get('direction').detach()
                                    if stage2.get('direction') is not None else None),
                          highres_h=highres.detach(), stage3_h=captured[0])
        return result


@torch.no_grad()
def make_q_target(skeleton, z0, output_size, min_segment_pixels=3):
    """Missed GT centerline is a training proxy, not a true gap oracle."""
    if skeleton.shape != z0.shape:
        raise ValueError('Skeleton and initial surface must share the training grid')
    nearby_p0 = F.max_pool2d(z0.sigmoid(), 5, stride=1, padding=2)
    missed = ((skeleton > 0.5) & (nearby_p0 < 0.5)).cpu().numpy().astype(np.uint8)
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


@torch.no_grad()
def missed_segments(skeleton, z0, threshold=0.45, min_segment_pixels=3):
    if skeleton.shape != z0.shape:
        raise ValueError('Skeleton and initial surface must share the training grid')
    near_p0 = F.max_pool2d(z0.sigmoid(), 5, stride=1, padding=2)
    missing = ((skeleton > 0.5) & (near_p0 < threshold)).cpu().numpy().astype(np.uint8)
    segments = []
    for binary in missing[:, 0]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        regions = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= min_segment_pixels:
                ys, xs = np.nonzero(labels == index)
                regions.append((torch.as_tensor(ys, device=z0.device),
                                torch.as_tensor(xs, device=z0.device)))
        segments.append(regions)
    return segments


def recovery_losses(result, mask, skeleton, threshold=0.45, margin=0.2,
                    background_radius=8):
    segments = missed_segments(skeleton, result['z0'], threshold)
    target_logit = torch.logit(result['z1'].new_tensor(threshold)) + margin
    per_segment = []
    missing_mask = torch.zeros_like(mask, dtype=torch.bool)
    for batch_index, regions in enumerate(segments):
        for ys, xs in regions:
            missing_mask[batch_index, 0, ys, xs] = True
            deficit = F.relu(target_logit - result['z1'][batch_index, 0, ys, xs])
            hardest = deficit.topk(max(1, (deficit.numel() + 3) // 4)).values.mean()
            per_segment.append(0.5 * deficit.mean() + 0.5 * hardest)
    zero = result['z1'].sum() * 0.
    recover = torch.stack(per_segment).mean() if per_segment else zero
    radius = int(background_radius)
    near_missing = F.max_pool2d(missing_mask.float(), 2 * radius + 1,
                                stride=1, padding=radius) > 0
    background = near_missing & (mask < 0.5)
    background_loss = (F.binary_cross_entropy_with_logits(
        result['z1'][background], torch.zeros_like(result['z1'][background]))
        if background.any() else zero)
    return recover, background_loss, len(per_segment), int(missing_mask.sum().item())


def repair_losses(result, mask, skeleton, lambda_q=0.1, lambda_recover=0.05,
                  lambda_background=0.02, recovery_threshold=0.45, recovery_margin=0.2):
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
    recover, background, segment_count, missed_pixels = recovery_losses(
        result, mask, skeleton, threshold=recovery_threshold, margin=recovery_margin)
    total = (surface + lambda_q * q_loss + lambda_recover * recover +
             lambda_background * background)
    return total, {'surface': surface.detach(), 'q': q_loss.detach(),
                   'recover': recover.detach(), 'near_background': background.detach(),
                   'missing_segments': segment_count, 'missing_pixels': missed_pixels,
                   'q_positive_fraction': q_target.mean().detach(),
                   'q_high_fraction': (result['q'] > 0.5).float().mean().detach()}
