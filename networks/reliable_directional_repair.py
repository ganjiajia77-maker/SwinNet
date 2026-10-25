"""Frozen Q3 conditions with reliability-guided, local directional dynamics."""

import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


AXES = ((0, 1), (1, 0), (1, 1), (1, -1))
RADII = (1, 2, 4, 8, 16)
DIRECTIONS = tuple(offset for dy, dx in AXES for offset in ((-dy, -dx), (dy, dx)))


def compose_surface_logits(z0_full, p0_work, updated_work, eps=1e-5):
    if p0_work.shape != updated_work.shape:
        raise ValueError('The initial and updated working grids must match')
    delta = (torch.logit(updated_work.clamp(eps, 1. - eps)) -
             torch.logit(p0_work.clamp(eps, 1. - eps)))
    delta_full = F.interpolate(delta, size=z0_full.shape[-2:], mode='bilinear',
                               align_corners=False)
    return z0_full + delta_full, delta


def shifted(value, dy, dx):
    """Read at p+(dy,dx), returning zeros outside the image."""
    pad = max(abs(dy), abs(dx))
    height, width = value.shape[-2:]
    padded = F.pad(value, (pad, pad, pad, pad))
    return padded[..., pad + dy:pad + dy + height, pad + dx:pad + dx + width]


def directional_visual_sources(visual, reliable, weights, scale_logits):
    """Gather reliable visual context in W-selected directions and radii."""
    mass = torch.zeros_like(reliable)
    numerator = torch.zeros_like(visual)
    for axis_index, (dy, dx) in enumerate(AXES):
        scales = scale_logits[axis_index].softmax(dim=0)
        for radius_index, radius in enumerate(RADII):
            for side_index, sign in enumerate((-1, 1)):
                direction_index = axis_index * 2 + side_index
                source = (weights[:, direction_index:direction_index+1] *
                          scales[radius_index] *
                          shifted(reliable, sign * radius * dy, sign * radius * dx))
                mass = mass + source
                numerator = numerator + source * shifted(
                    visual, sign * radius * dy, sign * radius * dx)
    source_visual = numerator / mass.clamp_min(1e-6)
    coverage = mass.clamp(0., 1.)
    agreement = F.cosine_similarity(visual, source_visual, dim=1).unsqueeze(1)
    return source_visual, coverage, agreement.clamp(-1., 1.)


class VisualProjection(nn.Module):
    def __init__(self, in_channels, width):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(in_channels, width, 1, bias=False),
                                     nn.GroupNorm(8, width), nn.GELU())
        self.local = nn.Sequential(nn.Conv2d(width, width, 3, padding=1,
                                             groups=width, bias=False),
                                   nn.Conv2d(width, width, 1, bias=False),
                                   nn.GroupNorm(8, width), nn.GELU())

    def forward(self, highres, decoder):
        visual = self.project(torch.cat((highres, decoder), dim=1))
        return visual + self.local(visual)


class DirectionRelations(nn.Module):
    def __init__(self, width, bridge_route=False, conditional_transport=False,
                 transport_gate_mode='v6'):
        super().__init__()
        self.bridge_route = bridge_route
        self.conditional_transport = conditional_transport
        if transport_gate_mode not in ('v6', 'v7'):
            raise ValueError(f'Unsupported transport gate mode: {transport_gate_mode}')
        self.transport_gate_mode = transport_gate_mode
        self.semantic = nn.Conv2d(width, 1, 3, padding=1)
        self.tangent = nn.Conv2d(width, 4, 3, padding=1)
        self.pair = nn.Sequential(nn.Conv2d(16, 16, 1), nn.GELU(),
                                  nn.Conv2d(16, 3, 1))
        self.scale_logits = nn.Parameter(torch.tensor([0., -.1, -.2, -.3, -.4])
                                         .repeat(4, 1))
        self.direction_head = nn.Conv2d(width + 8, 9, 1)
        nn.init.zeros_(self.direction_head.weight)
        with torch.no_grad():
            self.direction_head.bias.fill_(-1.)
            self.direction_head.bias[0] = 1.

    def forward(self, visual, p0, reliable):
        batch, _, height, width = visual.shape
        normalized = F.normalize(visual, dim=1, eps=1e-6)
        semantic = self.semantic(visual).sigmoid()
        tangent = self.tangent(visual).sigmoid()
        road_width = F.avg_pool2d(p0, 3, stride=1, padding=1)
        ones = torch.ones_like(p0)
        direction_evidence, one_direction_evidence = [], []
        two_maps, one_maps = [], []
        for axis_index, (dy, dx) in enumerate(AXES):
            scores = []
            for radius in RADII:
                my, mx, py, px = -radius * dy, -radius * dx, radius * dy, radius * dx
                f_minus, f_plus = shifted(normalized, my, mx), shifted(normalized, py, px)
                r_minus, r_plus = shifted(reliable, my, mx), shifted(reliable, py, px)
                p_minus, p_plus = shifted(p0, my, mx), shifted(p0, py, px)
                v_minus, v_plus = shifted(ones, my, mx), shifted(ones, py, px)
                mid = max(radius // 2, 1)
                semantic_minus = shifted(semantic, -mid * dy, -mid * dx)
                semantic_plus = shifted(semantic, mid * dy, mid * dx)
                tangent_axis = tangent[:, axis_index:axis_index+1]
                tangent_minus = shifted(tangent_axis, my, mx)
                tangent_plus = shifted(tangent_axis, py, px)
                width_compatible = torch.exp(-2. * torch.abs(
                    shifted(road_width, my, mx) - shifted(road_width, py, px)))
                radius_feature = torch.full_like(p0, radius / RADII[-1])
                if self.bridge_route:
                    corridor_match = .5 * (
                        (f_minus * shifted(normalized, -mid * dy, -mid * dx))
                        .sum(1, keepdim=True) +
                        (f_plus * shifted(normalized, mid * dy, mid * dx))
                        .sum(1, keepdim=True))
                    evidence = torch.cat((
                        (f_minus * f_plus).sum(1, keepdim=True),
                        (normalized * f_minus).sum(1, keepdim=True),
                        (normalized * f_plus).sum(1, keepdim=True),
                        corridor_match, r_minus, r_plus,
                        .5 * (p_minus + p_plus), semantic,
                        semantic_minus, semantic_plus,
                        tangent_minus, tangent_plus,
                        tangent_minus * tangent_plus, width_compatible,
                        v_minus * v_plus, radius_feature), dim=1)
                else:
                    evidence = torch.cat((
                        (f_minus * f_plus).sum(1, keepdim=True),
                        (normalized * f_minus).sum(1, keepdim=True),
                        (normalized * f_plus).sum(1, keepdim=True),
                        r_minus, r_plus, p_minus, p_plus, semantic,
                        semantic_minus, semantic_plus,
                        tangent_minus, tangent_plus, width_compatible,
                        v_minus, v_plus, radius_feature), dim=1)
                relation = self.pair(evidence).sigmoid()
                if self.conditional_transport:
                    two = (relation[:, 0:1] * torch.minimum(r_minus, r_plus) *
                           v_minus * v_plus)
                    one_scale = .2
                elif self.bridge_route:
                    tangent_support = .5 + .5 * tangent_minus * tangent_plus
                    corridor_support = .5 + .5 * torch.minimum(
                        semantic, torch.minimum(semantic_minus, semantic_plus))
                    two = (relation[:, 0:1] * torch.minimum(r_minus, r_plus) *
                           tangent_support * corridor_support * width_compatible *
                           v_minus * v_plus)
                    one_scale = .2
                else:
                    two = relation[:, 0:1] * r_minus * r_plus * v_minus * v_plus
                    one_scale = .25
                one_minus = (one_scale * relation[:, 1:2] * r_minus * v_minus *
                             (1. - r_plus * v_plus))
                one_plus = (one_scale * relation[:, 2:3] * r_plus * v_plus *
                            (1. - r_minus * v_minus))
                scores.append(torch.cat((two, one_minus, one_plus), dim=1))
            scale_weight = self.scale_logits[axis_index].softmax(dim=0)
            combined = sum(weight * score for weight, score in zip(scale_weight, scores))
            two = combined[:, 0:1]
            one_minus, one_plus = combined[:, 1:2], combined[:, 2:3]
            two_maps.append(two)
            one_maps.append(one_minus + one_plus)
            direction_evidence.extend((two + one_minus, two + one_plus))
            one_direction_evidence.extend((one_minus, one_plus))
        directional = torch.cat(direction_evidence, dim=1)
        logits = self.direction_head(torch.cat((visual, directional), dim=1))
        if self.conditional_transport:
            direction_logits = logits[:, 1:] + torch.log(directional + .05)
            weights = direction_logits.softmax(dim=1)
            if self.transport_gate_mode == 'v7':
                # Reuse learned multi-radius C_bridge and one-side relations.
                # The relation head already reads endpoint and corridor visual features.
                bridge_by_direction = torch.cat(
                    [value.repeat_interleave(2, dim=1) for value in two_maps], dim=1
                )
                one_by_direction = torch.cat(one_direction_evidence, dim=1)
                path_evidence = bridge_by_direction + one_by_direction
                source_evidence = (weights * path_evidence).sum(dim=1, keepdim=True)
            else:
                source_evidence = directional.amax(dim=1, keepdim=True)
            transport_gate = torch.sigmoid(
                logits[:, :1] + torch.log(source_evidence.clamp_min(1e-6)))
            return weights, torch.cat(two_maps, dim=1), torch.cat(one_maps, dim=1), transport_gate
        logits = torch.cat((logits[:, :1], logits[:, 1:] +
                            torch.log(directional + .05)), dim=1)
        weights = logits.softmax(dim=1)[:, 1:]
        return weights, torch.cat(two_maps, dim=1), torch.cat(one_maps, dim=1), None


class DirectionalDynamics(nn.Module):
    def __init__(self, width, steps=3):
        super().__init__()
        self.steps = steps
        self.rates = nn.Sequential(nn.Conv2d(width + 7, width, 1), nn.GELU(),
                                   nn.Conv2d(width, 4, 1))
        nn.init.zeros_(self.rates[-1].weight)
        with torch.no_grad():
            self.rates[-1].bias.copy_(torch.tensor([2.5, 0., -1., 1.]))

    def forward(self, visual, p0, q, reliable, proposal, accept, weights,
                capture_diagnostics=False, transport_gate=None,
                freeze_generated_sources=False):
        state = p0
        states, masses, availabilities = [state], [], []
        diagnostics = {key: [] for key in (
            'alpha', 'beta', 'gamma', 'stay', 'm',
            'transport_delta', 'proposal_delta', 'anchor_delta')}
        for _ in range(self.steps):
            if freeze_generated_sources:
                support = reliable
            elif transport_gate is not None:
                gained = ((state - p0).clamp_min(0.) /
                          (1. - p0).clamp_min(1e-5)).clamp_max(1.)
                support = reliable + (1. - reliable) * gained
            else:
                support = reliable + (1. - reliable) * accept * state
            mass = torch.zeros_like(state)
            numerator = torch.zeros_like(state)
            for index, (dy, dx) in enumerate(DIRECTIONS):
                source = weights[:, index:index+1] * shifted(support, dy, dx)
                mass = mass + source
                numerator = numerator + source * shifted(state, dy, dx)
            available = mass.clamp(0., 1.)
            propagated = torch.where(mass > 1e-6, numerator / mass.clamp_min(1e-6),
                                     state)
            coefficients = self.rates(torch.cat((visual, p0, q, reliable, proposal,
                                                 accept, state, available), dim=1)).softmax(dim=1)
            transport = (coefficients[:, 1:2] * transport_gate
                         if transport_gate is not None else
                         accept * coefficients[:, 1:2] * available)
            propose = .2 * accept * coefficients[:, 2:3]
            anchor = coefficients[:, 3:4]
            stay = 1. - transport - propose - anchor
            if capture_diagnostics:
                for key, value in (('alpha', transport), ('beta', propose),
                                   ('gamma', anchor), ('stay', stay),
                                   ('m', propagated),
                                   ('transport_delta', transport * (propagated - state)),
                                   ('proposal_delta', propose * (proposal - state)),
                                   ('anchor_delta', anchor * (p0 - state))):
                    diagnostics[key].append(value)
            state = stay * state + transport * propagated + propose * proposal + anchor * p0
            states.append(state)
            masses.append(mass)
            availabilities.append(available)
        if capture_diagnostics:
            return states, masses, availabilities, {
                key: torch.stack(values, dim=1) for key, values in diagnostics.items()}
        return states, masses, availabilities


class ReliableDirectionalRepair(nn.Module):
    def __init__(self, frozen_q3, width=32, steps=3, bridge_route=True,
                 conditional_transport=True, transport_gate_mode='v6'):
        super().__init__()
        if conditional_transport and not bridge_route:
            raise ValueError('Conditional transport requires bridge evidence')
        self.frozen = frozen_q3.requires_grad_(False)
        self.frozen.q_only = True
        self.frozen.eval()
        stage3 = self.frozen.base.swin_unet.decoder_structure_blocks['3']
        stage_channels = stage3.structure_branch[0].block[0].in_channels
        highres_channels = self.frozen.base.swin_unet.highres_structure_channels
        self.visual = VisualProjection(stage_channels + highres_channels, width)
        self.reliable_head = nn.Sequential(nn.Conv2d(width + 1, width, 3, padding=1),
                                           nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.reliable_head[-1].weight)
        nn.init.constant_(self.reliable_head[-1].bias, 1.5)
        self.relations = DirectionRelations(
            width, bridge_route=bridge_route,
            conditional_transport=conditional_transport,
            transport_gate_mode=transport_gate_mode)
        proposal_channels = (2 * width + 7) if bridge_route else (width + 7)
        self.proposal_head = nn.Sequential(nn.Conv2d(proposal_channels, width, 3, padding=1),
                                           nn.GELU(), nn.Conv2d(width, 2 if bridge_route else 1, 1))
        nn.init.zeros_(self.proposal_head[-1].weight)
        with torch.no_grad():
            if bridge_route:
                self.proposal_head[-1].bias.copy_(torch.tensor([-6., -7.]))
            else:
                self.proposal_head[-1].bias.zero_()
        self.accept_head = nn.Sequential(nn.Conv2d(proposal_channels + 1, width, 3, padding=1),
                                         nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.accept_head[-1].weight)
        nn.init.constant_(self.accept_head[-1].bias, -1.)
        self.dynamics = DirectionalDynamics(width, steps)
        self.width = width
        self.steps = steps
        self.accept_floor = 0.
        self.legacy_route = False
        self.visual_transport_route = True
        self.bridge_route = bridge_route
        self.conditional_transport = conditional_transport
        self.transport_gate_mode = transport_gate_mode

    def set_accept_floor(self, value):
        if not 0. <= value < 1.:
            raise ValueError('The training accept floor must be in [0, 1)')
        self.accept_floor = float(value)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, image, capture_diagnostics=False, weights_override=None,
                freeze_generated_sources=False, reliable_override=None):
        with torch.no_grad():
            old = self.frozen(image, capture_diagnostics=True)
        z0, q = old['z0'].detach(), old['q'].detach()
        highres, decoder = old['highres_h'].detach(), old['stage3_h'].detach()
        size = highres.shape[-2:]
        align = lambda value: F.interpolate(value, size=size, mode='bilinear',
                                             align_corners=False)
        decoder = align(decoder) if decoder.shape[-2:] != size else decoder
        visual = self.visual(highres, decoder)
        p0_full = z0.sigmoid()
        p0 = align(p0_full).detach()
        q = align(q) if q.shape[-2:] != size else q
        reliable = p0 * self.reliable_head(torch.cat((visual, p0), dim=1)).sigmoid()
        if reliable_override is not None:
            if reliable_override.shape != reliable.shape:
                raise ValueError('The diagnostic R override must match the work grid')
            reliable = reliable_override
        weights, two, one, transport_gate = self.relations(visual, p0, reliable)
        if weights_override is not None:
            if weights_override.shape != weights.shape:
                raise ValueError('The diagnostic W override must retain eight directions')
            weights = weights_override
        two_mean, one_mean = two.mean(1, keepdim=True), one.mean(1, keepdim=True)
        weight_sum = weights.sum(1, keepdim=True)
        bilateral = (weights * two.repeat_interleave(2, dim=1)).sum(1, keepdim=True) / (
            weight_sum + 1e-6)
        bridge = (weights * two.repeat_interleave(2, dim=1)).sum(1, keepdim=True)
        one_side = (weights * one.repeat_interleave(2, dim=1)).sum(1, keepdim=True)
        connection_support = (bridge + .25 * one_side).clamp(0., 1.)
        if self.visual_transport_route or capture_diagnostics:
            source_visual, visual_coverage, visual_agreement = directional_visual_sources(
                visual, reliable, weights, self.relations.scale_logits)
        else:
            source_visual = visual
            visual_coverage = torch.zeros_like(p0)
            visual_agreement = torch.zeros_like(p0)
        visual_input = (visual + visual_coverage * (source_visual - visual)
                        if self.visual_transport_route else visual)
        continuation = (bilateral * visual_coverage *
                        (.5 + .5 * visual_agreement.clamp_min(0.)))
        proposal_evidence = continuation if self.visual_transport_route else bilateral
        directional_neighbor = sum(
            weights[:, index:index+1] * shifted(p0, dy, dx)
            for index, (dy, dx) in enumerate(DIRECTIONS)) / (weight_sum + 1e-6)
        p0_logit = torch.logit(p0.clamp(1e-5, 1. - 1e-5))
        if self.bridge_route:
            if self.conditional_transport:
                local_support = .05 * q * self.relations.semantic(visual).sigmoid()
                positive_support = (connection_support + local_support).clamp(0., 1.)
                transported_for_heads = source_visual
            else:
                positive_support = connection_support
                transported_for_heads = visual_coverage * source_visual
            proposal_features = torch.cat((
                visual, transported_for_heads,
                p0, q, reliable, bridge, one_side, positive_support,
                visual_coverage), dim=1)
            proposal_raw = self.proposal_head(proposal_features)
            delta_plus = 8. * proposal_raw[:, :1].sigmoid()
            delta_minus = 4. * proposal_raw[:, 1:2].sigmoid()
            proposal = (p0_logit + positive_support * delta_plus - delta_minus).sigmoid()
            accept_features = torch.cat((proposal_features, proposal - p0), dim=1)
            relation_balance = (bridge - one_side) / (bridge + one_side + .05)
            continuation = connection_support
        else:
            proposal_features = torch.cat((visual_input, p0, q, reliable, two_mean,
                                            one_mean, weight_sum if self.legacy_route else bilateral,
                                            two.amax(1, keepdim=True)), dim=1)
            proposal_delta = self.proposal_head(proposal_features)
            if not self.legacy_route:
                candidate = p0 + proposal_evidence * (directional_neighbor - p0)
                proposal_delta = proposal_delta + (
                    torch.logit(candidate.clamp(1e-5, 1. - 1e-5)) - p0_logit)
            proposal = (p0_logit + proposal_delta).sigmoid()
            accept_features = torch.cat((proposal_features, proposal), dim=1)
            relation_balance = (proposal_evidence - one_mean) / (
                proposal_evidence + one_mean + .05)
            delta_plus = torch.zeros_like(p0)
            delta_minus = torch.zeros_like(p0)
        accept_logit = self.accept_head(accept_features)
        if not self.legacy_route:
            accept_logit = accept_logit + relation_balance
        accept_raw = accept_logit.sigmoid()
        if self.conditional_transport:
            positive_fraction = torch.sigmoid(10. * (proposal - p0))
            acceptance_cap = .1 + .9 * positive_support
            accept_raw = accept_raw * (1. - positive_fraction * (1. - acceptance_cap))
        floor = self.accept_floor if self.training else 0.
        accept = floor + (1. - floor) * accept_raw
        dynamics_output = self.dynamics(
            visual, p0, q, reliable, proposal, accept, weights,
            capture_diagnostics=capture_diagnostics,
            transport_gate=transport_gate,
            freeze_generated_sources=freeze_generated_sources)
        states, masses, available = dynamics_output[:3]
        z1, delta_logit = compose_surface_logits(z0, p0, states[-1])
        p1 = z1.sigmoid()
        result = {'z0': z0, 'z1': z1, 'p0': p0_full, 'p1': p1,
                  'q': q, 'r': reliable, 'w': weights, 'b': proposal,
                  'g_transport': (transport_gate if transport_gate is not None else
                                  1. - weight_sum),
                  'a': accept, 'a_raw': accept_raw, 'e_two': two, 'e_one': one,
                  'proposal_raw': (proposal_raw if self.bridge_route else
                                   proposal_delta),
                  'u0': states[0], 'u1': states[1], 'u2': states[2], 'u3': states[3]}
        if capture_diagnostics:
            result.update(source_mass=torch.stack(masses, dim=1),
                          source_available=torch.stack(available, dim=1),
                          delta_logit=delta_logit, visual=visual,
                          directional_bilateral=bilateral,
                          directional_neighbor=directional_neighbor,
                          directional_visual_coverage=visual_coverage,
                          directional_visual_agreement=visual_agreement,
                          directional_continuation=continuation,
                          c_bridge=bridge, c_one_side=one_side,
                          c_support=connection_support,
                          g_positive=(positive_support if self.bridge_route else
                                      torch.zeros_like(p0)),
                          delta_z_plus=delta_plus, delta_z_minus=delta_minus,
                          transported_visual=source_visual,
                          relation_balance=relation_balance,
                          **dynamics_output[3])
        return result

@torch.no_grad()
def missed_skeleton_segments(skeleton, p0, threshold=.45, min_pixels=3):
    """Find continuous GT centerline pieces missing from the frozen surface."""
    nearby = F.max_pool2d(p0.detach(), 5, stride=1, padding=2)
    missing = ((skeleton > .5) & (nearby < threshold)).cpu().numpy()[:, 0]
    result = []
    for image in missing:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            image.astype(np.uint8), connectivity=8)
        pieces = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= min_pixels:
                pieces.append(np.nonzero(labels == index))
        result.append(pieces)
    return result


def dynamic_losses(result, mask, skeleton, lambda_bg=1., lambda_missing=2., lambda_cont=.2,
                   temperature=15., max_paths_per_image=128,
                   missing_threshold=.45, return_live_terms=False):
    z1, p1, p0, q = result['z1'], result['p1'], result['p0'], result['q']
    if mask.shape != z1.shape or skeleton.shape != z1.shape:
        raise ValueError('Surface, mask and GT skeleton must share the final grid')
    q_full = F.interpolate(q.detach(), size=z1.shape[-2:], mode='bilinear',
                           align_corners=False)
    missed_center = (skeleton > .5) & (
        F.max_pool2d(p0.detach(), 5, stride=1, padding=2) < missing_threshold)
    missed_band = F.max_pool2d(missed_center.float(), 5, stride=1, padding=2)
    weights = (1. + lambda_bg * (1. - mask) * q_full +
               lambda_missing * mask * missed_band)
    bce = F.binary_cross_entropy_with_logits(z1, mask, reduction='none')
    pt = torch.exp(-bce)
    focal = (.75 * (1. - pt).square() * bce * weights).sum() / weights.sum().clamp_min(1.)
    dice = 1. - (2. * (p1 * mask).sum((1, 2, 3)) + 1.) / (
        p1.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1.)
    segmentation = focal + .5 * dice.mean()
    tolerant = F.max_pool2d(p1, 3, stride=1, padding=1)
    paths = missed_skeleton_segments(skeleton, p0, threshold=missing_threshold)
    image_losses, soft_min_values = [], []
    for batch_index, image_paths in enumerate(paths):
        # Select the hardest missing pieces using only the frozen prediction.
        p0_image = p0[batch_index, 0].detach().cpu().numpy()
        ranked = sorted(image_paths,
                        key=lambda piece: float(p0_image[piece].mean()))
        segment_losses = []
        for ys, xs in ranked[:max_paths_per_image]:
            y = torch.as_tensor(ys, device=z1.device)
            x = torch.as_tensor(xs, device=z1.device)
            values = tolerant[batch_index, 0, y, x]
            soft_min = -(torch.logsumexp(-temperature * values, dim=0) -
                         math.log(values.numel())) / temperature
            segment_losses.append(.05 * F.softplus((.75 - soft_min) / .05))
            soft_min_values.append(soft_min.detach())
        if segment_losses:
            image_losses.append(torch.stack(segment_losses).mean())
    continuity = (torch.stack(image_losses).mean() if image_losses
                  else z1.sum() * 0.)
    total = segmentation + lambda_cont * continuity
    soft_values = (torch.stack(soft_min_values).float().cpu().numpy()
                   if soft_min_values else np.empty(0, dtype=np.float32))
    terms = {'seg': segmentation.detach(), 'cont': continuity.detach(),
             'focal': focal.detach(), 'dice': dice.mean().detach(),
             'missing_weighted_pixels': int(((mask > .5) & (missed_band > 0)).sum()),
             'hard_background_weighted_pixels': int(((mask < .5) & (q_full > .5)).sum()),
             'missing_segments': sum(map(len, paths)),
             'paths': len(soft_min_values),
             'path_soft_min_mean': float(soft_values.mean()) if soft_values.size else None,
             'path_soft_min_p10': float(np.percentile(soft_values, 10)) if soft_values.size else None,
             'path_soft_min_p50': float(np.median(soft_values)) if soft_values.size else None,
             'path_soft_min_p90': float(np.percentile(soft_values, 90)) if soft_values.size else None}
    if return_live_terms:
        terms['cont_live'] = continuity
    return total, terms
