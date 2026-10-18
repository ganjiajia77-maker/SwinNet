"""Finite-step, symmetric eight-neighbor diffusion on the Stage 3 grid."""
import torch
from torch import nn
from torch.nn import functional as F
from topology_direction_constants import (CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE,
                                         connectivity_double_angle_basis)

ARCHITECTURE_ID = 'h0_s2_skeleton_s3_directional_diffusion_v1'
MODES = ('identity', 'semantic', 'semantic_directional')


def neighbor(x, dy, dx):
    """Read x[p+(dy,dx)], zero outside the image; never wrap."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0)))
    return padded[..., max(dy, 0):max(dy, 0)+h, max(dx, 0):max(dx, 0)+w]


class SemanticDirectionalDiffusion(nn.Module):
    def __init__(self, channels=96, structure_channels=96, highres_channels=64,
                 latent_channels=32, steps=8, eta=0.1, beta=0.5,
                 mode='semantic_directional'):
        super().__init__()
        if mode not in MODES or steps < 1 or not 0 < eta <= 1 or beta < 0:
            raise ValueError('Invalid diffusion settings')
        self.mode, self.steps, self.eta, self.beta = mode, int(steps), float(eta), float(beta)
        self.value_proj = nn.Conv2d(channels, latent_channels, 1)
        self.guide_g_proj = nn.Conv2d(structure_channels, 16, 1)
        self.guide_h_proj = nn.Conv2d(highres_channels, 16, 1)
        self.guide_head = nn.Sequential(nn.Conv2d(latent_channels+32, 32, 3, padding=1),
                                        nn.GELU(), nn.Conv2d(32, 8, 1))
        self.out_proj = nn.Conv2d(latent_channels, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.register_buffer('axis_basis', connectivity_double_angle_basis().view(1, 8, 2, 1, 1))
        self.capture_diagnostics = False
        self.last_diagnostics = None

    def edge_weights(self, logits, direction, beta=None):
        if logits.shape[1] != 8 or direction.shape != (logits.shape[0], 2, *logits.shape[-2:]):
            raise ValueError('Expected ell[B,8,H,W] and D[B,2,H,W]')
        beta = (0.0 if self.mode == 'semantic' else self.beta) if beta is None else float(beta)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            unit = F.normalize(direction.detach().float(), dim=1, eps=1e-6)
            preference = (unit[:, None] * self.axis_basis.float()).sum(2)
            ones = torch.ones_like(logits[:, :1])
            edges = []
            for d, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
                opposite = CONNECTIVITY_OPPOSITE[d]
                pair_l = .5 * (logits[:, d:d+1] + neighbor(logits[:, opposite:opposite+1], dy, dx))
                pair_b = .5 * (preference[:, d:d+1] + neighbor(preference[:, opposite:opposite+1], dy, dx))
                edges.append(torch.sigmoid(pair_l + beta * pair_b) * neighbor(ones, dy, dx))
            return torch.cat(edges, dim=1)

    def propagate(self, z0, edges):
        with torch.autocast(device_type=z0.device.type, enabled=False):
            z0, edges = z0.float(), edges.float()
            denominator = 1.0 + edges.sum(1, keepdim=True)
            z = z0
            for _ in range(self.steps):
                numerator = z
                for d, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
                    numerator = numerator + edges[:, d:d+1] * neighbor(z, dy, dx)
                z = (1.0-self.eta) * (numerator / denominator) + self.eta*z0
            return z

    def forward(self, x, g, h, direction):
        if not (x.shape[0] == g.shape[0] == h.shape[0] == direction.shape[0] and
                x.shape[-2:] == g.shape[-2:] == h.shape[-2:] == direction.shape[-2:]):
            raise ValueError('X3/G3/H2/D3 must share batch and spatial dimensions')
        if self.mode == 'identity':
            return x
        z0 = self.value_proj(x)
        logits = self.guide_head(torch.cat((z0, self.guide_g_proj(g), self.guide_h_proj(h)), 1))
        edges = self.edge_weights(logits, direction)
        zk = self.propagate(z0, edges)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = zk.float() - z0.float()
        result = x + self.out_proj(delta.to(z0.dtype)).to(x.dtype)
        if self.capture_diagnostics:
            self.last_diagnostics = {'edges': edges.detach(), 'delta_z': delta.detach()}
        return result
