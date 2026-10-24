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


def shifted(value, dy, dx):
    """Read at p+(dy,dx), returning zeros outside the image."""
    pad = max(abs(dy), abs(dx))
    height, width = value.shape[-2:]
    padded = F.pad(value, (pad, pad, pad, pad))
    return padded[..., pad + dy:pad + dy + height, pad + dx:pad + dx + width]


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
    def __init__(self, width):
        super().__init__()
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
        direction_evidence, two_maps, one_maps = [], [], []
        for axis_index, (dy, dx) in enumerate(AXES):
            scores = []
            for radius in RADII:
                my, mx, py, px = -radius * dy, -radius * dx, radius * dy, radius * dx
                f_minus, f_plus = shifted(normalized, my, mx), shifted(normalized, py, px)
                r_minus, r_plus = shifted(reliable, my, mx), shifted(reliable, py, px)
                p_minus, p_plus = shifted(p0, my, mx), shifted(p0, py, px)
                v_minus, v_plus = shifted(ones, my, mx), shifted(ones, py, px)
                mid = max(radius // 2, 1)
                evidence = torch.cat((
                    (f_minus * f_plus).sum(1, keepdim=True),
                    (normalized * f_minus).sum(1, keepdim=True),
                    (normalized * f_plus).sum(1, keepdim=True),
                    r_minus, r_plus, p_minus, p_plus, semantic,
                    shifted(semantic, -mid * dy, -mid * dx),
                    shifted(semantic, mid * dy, mid * dx),
                    shifted(tangent[:, axis_index:axis_index+1], my, mx),
                    shifted(tangent[:, axis_index:axis_index+1], py, px),
                    torch.exp(-2. * torch.abs(shifted(road_width, my, mx) -
                                                 shifted(road_width, py, px))),
                    v_minus, v_plus,
                    torch.full_like(p0, radius / RADII[-1])), dim=1)
                relation = self.pair(evidence).sigmoid()
                two = relation[:, 0:1] * r_minus * r_plus * v_minus * v_plus
                one_minus = (.25 * relation[:, 1:2] * r_minus * v_minus *
                             (1. - r_plus * v_plus))
                one_plus = (.25 * relation[:, 2:3] * r_plus * v_plus *
                            (1. - r_minus * v_minus))
                scores.append(torch.cat((two, one_minus, one_plus), dim=1))
            scale_weight = self.scale_logits[axis_index].softmax(dim=0)
            combined = sum(weight * score for weight, score in zip(scale_weight, scores))
            two = combined[:, 0:1]
            one_minus, one_plus = combined[:, 1:2], combined[:, 2:3]
            two_maps.append(two)
            one_maps.append(one_minus + one_plus)
            direction_evidence.extend((two + one_minus, two + one_plus))
        directional = torch.cat(direction_evidence, dim=1)
        logits = self.direction_head(torch.cat((visual, directional), dim=1))
        logits = torch.cat((logits[:, :1], logits[:, 1:] +
                            torch.log(directional + .05)), dim=1)
        weights = logits.softmax(dim=1)[:, 1:]
        return weights, torch.cat(two_maps, dim=1), torch.cat(one_maps, dim=1)


class DirectionalDynamics(nn.Module):
    def __init__(self, width, steps=3):
        super().__init__()
        self.steps = steps
        self.rates = nn.Sequential(nn.Conv2d(width + 7, width, 1), nn.GELU(),
                                   nn.Conv2d(width, 4, 1))
        nn.init.zeros_(self.rates[-1].weight)
        with torch.no_grad():
            self.rates[-1].bias.copy_(torch.tensor([2.5, 0., -1., 1.]))

    def forward(self, visual, p0, q, reliable, proposal, accept, weights):
        state = p0
        states, masses, availabilities = [state], [], []
        for _ in range(self.steps):
            support = reliable + (1. - reliable) * accept * state
            mass = torch.zeros_like(state)
            numerator = torch.zeros_like(state)
            for index, (dy, dx) in enumerate(DIRECTIONS):
                source = weights[:, index:index+1] * shifted(support, dy, dx)
                mass = mass + source
                numerator = numerator + source * shifted(state, dy, dx)
            available = mass.clamp(0., 1.)
            propagated = numerator / mass.clamp_min(1e-6)
            coefficients = self.rates(torch.cat((visual, p0, q, reliable, proposal,
                                                 accept, state, available), dim=1)).softmax(dim=1)
            transport = accept * coefficients[:, 1:2] * available
            propose = .2 * accept * coefficients[:, 2:3]
            anchor = coefficients[:, 3:4]
            stay = 1. - transport - propose - anchor
            state = stay * state + transport * propagated + propose * proposal + anchor * p0
            states.append(state)
            masses.append(mass)
            availabilities.append(available)
        return states, masses, availabilities


class ReliableDirectionalRepair(nn.Module):
    def __init__(self, frozen_q3, width=32, steps=3):
        super().__init__()
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
        self.relations = DirectionRelations(width)
        self.proposal_head = nn.Sequential(nn.Conv2d(width + 7, width, 3, padding=1),
                                           nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.proposal_head[-1].weight)
        nn.init.zeros_(self.proposal_head[-1].bias)
        self.accept_head = nn.Sequential(nn.Conv2d(width + 8, width, 3, padding=1),
                                         nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.accept_head[-1].weight)
        nn.init.constant_(self.accept_head[-1].bias, -2.)
        self.dynamics = DirectionalDynamics(width, steps)
        self.detail_projection = nn.Conv2d(width, 8, 1)
        self.detail_head = nn.Sequential(nn.Conv2d(11, 8, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.detail_head[-1].weight)
        nn.init.zeros_(self.detail_head[-1].bias)
        self.width = width
        self.steps = steps

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, image, capture_diagnostics=False, diagnostic_override=None):
        diagnostic_override = diagnostic_override or {}
        unknown = set(diagnostic_override) - {'internal_a_one', 'final_a_one',
                                               'proposal_shift', 'rotate_w'}
        if unknown:
            raise ValueError(f'Unknown diagnostic override: {sorted(unknown)}')
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
        weights, two, one = self.relations(visual, p0, reliable)
        two_mean, one_mean = two.mean(1, keepdim=True), one.mean(1, keepdim=True)
        weight_sum = weights.sum(1, keepdim=True)
        proposal_features = torch.cat((visual, p0, q, reliable, two_mean,
                                        one_mean, weight_sum, two.amax(1, keepdim=True)), dim=1)
        proposal_delta = self.proposal_head(proposal_features)
        proposal = (torch.logit(p0.clamp(1e-5, 1. - 1e-5)) + proposal_delta).sigmoid()
        accept_features = torch.cat((proposal_features, proposal), dim=1)
        accept = self.accept_head(accept_features).sigmoid()
        if diagnostic_override.get('proposal_shift'):
            proposal = (p0 + float(diagnostic_override['proposal_shift'])).clamp(0., 1.)
        if diagnostic_override.get('rotate_w'):
            weights = weights.roll(shifts=1, dims=1)
        if diagnostic_override.get('internal_a_one'):
            accept = torch.ones_like(accept)
        states, masses, available = self.dynamics(
            visual, p0, q, reliable, proposal, accept, weights)
        full_size = z0.shape[-2:]
        up = lambda value: F.interpolate(value, size=full_size, mode='bilinear',
                                         align_corners=False)
        u3_full, accept_full = up(states[-1]), up(accept)
        detail = self.detail_head(torch.cat((up(self.detail_projection(visual)),
                                            u3_full, accept_full, p0_full), dim=1))
        detail = .5 * torch.tanh(detail)
        refined = (torch.logit(u3_full.clamp(1e-5, 1. - 1e-5)) + detail).sigmoid()
        if diagnostic_override.get('final_a_one'):
            accept_full = torch.ones_like(accept_full)
        p1 = (1. - accept_full) * p0_full + accept_full * refined
        z1 = torch.logit(p1.clamp(1e-6, 1. - 1e-6))
        result = {'z0': z0, 'z1': z1, 'p0': p0_full, 'p1': p1,
                  'q': q, 'r': reliable, 'w': weights, 'b': proposal,
                  'a': accept, 'e_two': two, 'e_one': one,
                  'u0': states[0], 'u1': states[1], 'u2': states[2], 'u3': states[3]}
        if capture_diagnostics:
            rates, transport, propose, stay = [], [], [], []
            for state, availability in zip(states[:-1], available):
                coefficient = self.dynamics.rates(torch.cat((
                    visual, p0, q, reliable, proposal, accept, state,
                    availability), dim=1)).softmax(dim=1)
                step_transport = accept * coefficient[:, 1:2] * availability
                step_propose = .2 * accept * coefficient[:, 2:3]
                rates.append(coefficient)
                transport.append(step_transport)
                propose.append(step_propose)
                stay.append(1. - step_transport - step_propose - coefficient[:, 3:4])
            result.update(source_mass=torch.stack(masses, dim=1),
                          source_available=torch.stack(available, dim=1),
                          rate_coefficients=torch.stack(rates, dim=1),
                          transport=torch.stack(transport, dim=1),
                          propose=torch.stack(propose, dim=1),
                          stay=torch.stack(stay, dim=1),
                          detail=detail, visual=visual, p0_work=p0,
                          accept_full=accept_full, t_full=refined)
        return result

def gt_branch_segments(skeleton, min_pixels=4):
    """Split GT centerlines at junctions into connected branch proxies."""
    result = []
    for image in skeleton[:, 0].detach().cpu().numpy() > .5:
        degree = (cv2.filter2D(image.astype(np.uint8), cv2.CV_16S,
                               np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
                  - image.astype(np.int16))
        branches = (image & (degree <= 2)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(branches, connectivity=8)
        paths = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= min_pixels:
                paths.append(np.nonzero(labels == index))
        result.append(paths)
    return result


def dynamic_losses(result, mask, skeleton, lambda_bg=1., lambda_cont=.2,
                   temperature=15., max_paths_per_image=128,
                   return_live_terms=False):
    z1, p1, p0, q = result['z1'], result['p1'], result['p0'], result['q']
    if mask.shape != z1.shape or skeleton.shape != z1.shape:
        raise ValueError('Surface, mask and GT skeleton must share the final grid')
    q_full = F.interpolate(q.detach(), size=z1.shape[-2:], mode='bilinear',
                           align_corners=False)
    weights = 1. + lambda_bg * (1. - mask) * q_full
    bce = F.binary_cross_entropy_with_logits(z1, mask, reduction='none')
    pt = torch.exp(-bce)
    focal = (.75 * (1. - pt).square() * bce * weights).sum() / weights.sum().clamp_min(1.)
    dice = 1. - (2. * (p1 * mask).sum((1, 2, 3)) + 1.) / (
        p1.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1.)
    segmentation = focal + .5 * dice.mean()
    tolerant = F.max_pool2d(p1, 3, stride=1, padding=1)
    paths = gt_branch_segments(skeleton)
    path_losses, path_weights = [], []
    for batch_index, image_paths in enumerate(paths):
        difficulty = []
        for ys, xs in image_paths:
            y = torch.as_tensor(ys, device=z1.device)
            x = torch.as_tensor(xs, device=z1.device)
            weight = (.1 + (1. - p0[batch_index, 0, y, x].detach()).mean())
            difficulty.append((weight, y, x))
        difficulty.sort(key=lambda item: float(item[0]), reverse=True)
        for weight, y, x in difficulty[:max_paths_per_image]:
            values = tolerant[batch_index, 0, y, x]
            soft_min = -(torch.logsumexp(-temperature * values, dim=0) -
                         math.log(values.numel())) / temperature
            path_losses.append(.05 * F.softplus((.75 - soft_min) / .05))
            path_weights.append(weight)
    if path_losses:
        joined_weights = torch.stack(path_weights)
        continuity = (torch.stack(path_losses) * joined_weights).sum() / joined_weights.sum()
    else:
        continuity = z1.sum() * 0.
    total = segmentation + lambda_cont * continuity
    terms = {'seg': segmentation.detach(), 'cont': continuity.detach(),
             'focal': focal.detach(), 'dice': dice.mean().detach(),
             'paths': len(path_losses)}
    if return_live_terms:
        terms.update(seg_live=segmentation, cont_live=continuity)
    return total, terms
