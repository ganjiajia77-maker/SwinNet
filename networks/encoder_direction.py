"""E2 undirected, symmetric feature aggregation. No GT is used in forward."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class E2DirectionAggregation(nn.Module):
    def __init__(self, channels=192):
        super().__init__()
        self.direction_head = nn.Conv2d(channels, 2, 1)
        self.gate_head = nn.Conv2d(channels, 1, 1)
        self.radius_head = nn.Conv2d(channels, 2, 1)
        self.projection = nn.Conv2d(channels, channels, 1)
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(.05 / .95)))
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.radius_head.weight)
        nn.init.zeros_(self.radius_head.bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.last_stats = {}

    @property
    def gamma(self):
        return self.raw_gamma.sigmoid()

    @staticmethod
    def tangent(raw):
        # A constant fallback axis makes atan2 and its gradients finite near zero.
        valid = raw.square().sum(1, keepdim=True) >= 1e-8
        a = torch.where(valid, raw[:, :1], torch.ones_like(raw[:, :1]))
        b = torch.where(valid, raw[:, 1:2], torch.zeros_like(raw[:, 1:2]))
        angle = .5 * torch.atan2(b, a)
        return torch.cat((angle.cos(), angle.sin()), 1)

    @staticmethod
    def aggregate(feature, tangent, weights):
        batch, _, height, width = feature.shape
        y, x = torch.meshgrid(torch.arange(height, device=feature.device, dtype=feature.dtype),
                              torch.arange(width, device=feature.device, dtype=feature.dtype),
                              indexing='ij')
        base = torch.stack((x, y), -1)[None]
        tangent = tangent.permute(0, 2, 3, 1)
        messages, validities = [], []
        for radius in (1, 2):
            plus, minus = base + radius * tangent, base - radius * tangent
            valid = ((plus[..., 0] >= 0) & (plus[..., 0] <= width - 1)
                     & (minus[..., 0] >= 0) & (minus[..., 0] <= width - 1)
                     & (plus[..., 1] >= 0) & (plus[..., 1] <= height - 1)
                     & (minus[..., 1] >= 0) & (minus[..., 1] <= height - 1))[:, None]
            def sample(coords):
                # Pixel centers under align_corners=False, including singleton axes.
                grid = torch.stack((2 * (coords[..., 0] + .5) / width - 1,
                                    2 * (coords[..., 1] + .5) / height - 1), -1)
                return F.grid_sample(feature, grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False)
            messages.append(.5 * (sample(plus) + sample(minus)) - feature)
            validities.append(valid)
        valid = torch.cat(validities, 1)
        masked = weights * valid.to(weights.dtype)
        masked = masked / masked.sum(1, keepdim=True).clamp_min(1e-8)
        message = sum(masked[:, i:i+1] * messages[i] for i in range(2))
        return message, valid.any(1, keepdim=True)

    def forward(self, feature):
        raw = self.direction_head(feature)
        gate = self.gate_head(feature).sigmoid()
        weights = self.radius_head(feature).softmax(1)
        message, valid = self.aggregate(feature, self.tangent(raw), weights)
        # Mask AFTER projection as well: a learned bias must not modify invalid pixels.
        delta = self.gamma * gate * self.projection(message) * valid.to(feature.dtype)
        with torch.no_grad():
            self.last_stats = {
                'q_mean': gate.mean().detach(), 'q_min': gate.amin().detach(),
                'q_max': gate.amax().detach(), 'q_std': gate.std(unbiased=False).detach(),
                'gamma': self.gamma.detach(), 'delta_abs_mean': delta.abs().mean().detach(),
                'delta_relative_abs': (delta.abs().mean() / feature.abs().mean().clamp_min(1e-8)).detach(),
            }
        return feature + delta, raw
