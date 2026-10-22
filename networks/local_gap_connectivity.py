"""One-hop sender-based aggregation and ordered sparse gap classification."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from local_gap_config import LocalGapConfig
from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE
from gap_candidates import generate_candidates


def neighbor_at(x, dy, dx):
    """Output[p] = input[p+(dy,dx)], zero outside; never wraps."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (1, 1, 1, 1))
    return padded[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


class LocalConnectivityPropagation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, feature, skeleton, connectivity, valid=None):
        valid = torch.ones_like(skeleton) if valid is None else valid
        numerator, denominator = feature, torch.ones_like(skeleton)
        for d, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            reverse = CONNECTIVITY_OPPOSITE[d]
            weight = neighbor_at(valid * skeleton * connectivity[:, reverse:reverse+1], dy, dx)
            weight = weight * valid
            numerator = numerator + weight * neighbor_at(feature, dy, dx)
            denominator = denominator + weight
        difference = numerator / denominator - feature
        return feature + self.beta * self.projection(difference)


def sample_at(feature, xy):
    """xy: [edges,lanes,samples,2] in feature pixel-center coordinates."""
    _, _, h, w = feature.shape
    scale = feature.new_tensor([w, h])
    grid = 2 * (xy + 0.5) / scale - 1
    n, lanes, samples, _ = grid.shape
    grid = grid.reshape(1, n * lanes, samples, 2)
    sampled = F.grid_sample(feature, grid, align_corners=False, padding_mode='border')
    return sampled[0].reshape(feature.shape[1], n, lanes, samples).permute(1, 2, 3, 0)


def path_coordinates(edges_yx, samples, side_offset):
    edges = edges_yx.flip(-1)  # yx -> xy
    vector = edges[:, 1] - edges[:, 0]
    unit = F.normalize(vector, dim=-1, eps=1e-6)
    normal = torch.stack([-unit[:, 1], unit[:, 0]], dim=-1)
    t = torch.linspace(0, 1, samples, device=edges.device, dtype=edges.dtype)
    center = edges[:, :1] + vector[:, None] * t[None, :, None]
    offsets = edges.new_tensor([-side_offset, 0, side_offset])
    return center[:, None] + offsets[None, :, None, None] * normal[:, None, None]


def rasterize_evidence(edges_yx, probabilities, height, width, radius):
    """Compact cosine bands, with confidence-independent overlap divisor."""
    rows, columns, values, kernels = [], [], [], []
    # Locations are discrete proposals; probabilities retain their autograd graph.
    for edge, probability in zip(edges_yx, probabilities):
        lo = (edge.min(0).values - radius).floor().long().clamp_min(0)
        hi = (edge.max(0).values + radius).ceil().long()
        hi = torch.minimum(hi, hi.new_tensor([height - 1, width - 1]))
        y0, x0 = lo.tolist()
        y1, x1 = hi.tolist()
        yy, xx = torch.meshgrid(torch.arange(y0, y1 + 1, device=edge.device),
                                torch.arange(x0, x1 + 1, device=edge.device), indexing='ij')
        points = torch.stack([yy, xx], -1).to(edge.dtype)
        vector = edge[1] - edge[0]
        along = ((points - edge[0]) * vector).sum(-1) / vector.square().sum().clamp_min(1e-6)
        closest = edge[0] + along.clamp(0, 1)[..., None] * vector
        distance = torch.linalg.vector_norm(points - closest, dim=-1)
        kernel = 0.5 * (1 + torch.cos(torch.pi * (distance / radius).clamp(0, 1)))
        rows.append(yy.reshape(-1))
        columns.append(xx.reshape(-1))
        kernels.append(kernel.reshape(-1))
        values.append(kernel.reshape(-1) * probability)
    if not values:
        return probabilities.new_zeros((1, 1, height, width))
    index = torch.cat(rows) * width + torch.cat(columns)
    value = probabilities.new_zeros(height * width).scatter_add(0, index, torch.cat(values))
    divisor = probabilities.new_zeros(height * width).scatter_add(0, index, torch.cat(kernels))
    return (value / divisor.clamp_min(1)).reshape(1, 1, height, width)


class GapConnectivity(nn.Module):
    def __init__(self, channels=96, highres_channels=64, config=None):
        super().__init__()
        self.config = LocalGapConfig.parse(config)
        self.decoder_projection = nn.Conv2d(channels, 32, 1)
        self.highres_projection = nn.Conv2d(highres_channels, 16, 1)
        self.edge_mlp = nn.Sequential(nn.Linear(3 * self.config.samples * 49 + 96 + 2 + 8, 128),
                                      nn.GELU(), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
        self.correction = nn.Sequential(nn.Conv2d(channels + highres_channels + 1, 32, 1),
                                         nn.GELU(), nn.Conv2d(32, channels, 3, padding=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(()))
        # Persist schedule progress. Evaluation of a warmup checkpoint must not enable an untrained path.
        self.register_buffer('writeback_scale', torch.zeros(()))

    def set_epoch(self, epoch):
        cfg = self.config
        self.writeback_scale.fill_(max(0., min(1., (epoch - cfg.writeback_start + 1) / cfg.writeback_ramp)))

    def forward(self, decoder, highres, p2_logits, s2_probability):
        cfg = self.config
        p2 = p2_logits.sigmoid()
        proposals = generate_candidates(p2.detach().float().cpu().numpy()[:, 0], cfg)
        h0 = highres if cfg.use_h0 else torch.zeros_like(highres)
        projected = torch.cat([self.decoder_projection(decoder), self.highres_projection(h0)], 1)
        # Detached probabilities remain evidence, rather than learning to game the classifier's inputs.
        all_features = torch.cat([projected, p2.detach()], 1)
        maps, logits_list, edges_list, geometry_list = [], [], [], []
        for b, proposal in enumerate(proposals):
            edges = decoder.new_tensor(proposal['edges']).reshape(-1, 2, 2)
            geometry = decoder.new_tensor(proposal['geometry']).reshape(-1, 8)
            edges_list.append(edges)
            geometry_list.append(geometry)
            if not len(edges):
                logits_list.append(decoder.new_empty(0))
                maps.append(decoder.new_zeros((1, 1, *decoder.shape[-2:])))
                continue
            xy = path_coordinates(edges, cfg.samples, cfg.side_offset)
            ordered = sample_at(all_features[b:b+1], xy)
            endpoint_xy = edges.flip(-1)[:, None]
            endpoints = sample_at(projected[b:b+1], endpoint_xy).flatten(1)
            s2 = sample_at(s2_probability[b:b+1].detach(), endpoint_xy).flatten(1)
            inputs = torch.cat([ordered.flatten(1), endpoints, s2, geometry], -1)
            reverse_geometry = geometry.clone()
            reverse_geometry[:, 1:3] = geometry[:, [2, 1]]
            reverse_geometry[:, 4:6] *= -1
            reverse = torch.cat([ordered.flip((1, 2)).flatten(1),
                                 endpoints.reshape(-1, 2, 48).flip(1).flatten(1),
                                 s2.flip(1), reverse_geometry], -1)
            logits = (self.edge_mlp(inputs) + self.edge_mlp(reverse)).squeeze(-1) * 0.5
            logits_list.append(logits)
            maps.append(rasterize_evidence(edges, logits.sigmoid(), *decoder.shape[-2:], cfg.band_radius))
        evidence = torch.cat(maps, 0)
        delta = self.gamma * self.writeback_scale * evidence * self.correction(torch.cat([decoder, h0, evidence], 1))
        record = {'stage': 'gap_connection', 'gap_logits': logits_list, 'gap_edges': edges_list,
                  'gap_geometry': geometry_list, 'gap_proposals': proposals, 'gap_config': cfg.to_dict(),
                  'p2_logits': p2_logits, 'gap_evidence': evidence,
                  'gap_residual_abs_mean': delta.detach().abs().mean(),
                  'gap_writeback_scale': self.writeback_scale.detach().clone()}
        return decoder + delta, record
