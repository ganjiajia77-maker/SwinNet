"""Supervised local reachability between a bounded set of predicted anchors."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .keypoint_global_topology import KeypointGuidedGlobalTopology


class SupervisedAnchorTopology(nn.Module):
    def __init__(self, channels, struct_channels, max_nodes=32, heads=4,
                 alpha_max=0.05, enabled=False, connectivity_channels=8,
                 neighbours=4, max_distance=64.0, samples=8, corridor_offset=2.0,
                 hidden=64, prior_warmup_epochs=10, prior_ramp_epochs=5):
        super().__init__()
        if max_nodes < 2 or neighbours < 1 or neighbours >= max_nodes:
            raise ValueError('Require 1 <= neighbours < max_nodes and max_nodes >= 2')
        if hidden % heads or samples < 2 or max_distance <= 0 or corridor_offset < 0:
            raise ValueError('Invalid anchor topology dimensions/range')
        self.channels, self.struct_channels = channels, struct_channels
        self.max_nodes, self.heads, self.hidden = max_nodes, heads, hidden
        self.alpha_max, self.enable_global_topology = alpha_max, enabled
        self.neighbours, self.max_distance = neighbours, max_distance
        self.samples, self.corridor_offset = samples, corridor_offset
        self.prior_warmup_epochs = prior_warmup_epochs
        self.prior_ramp_epochs = prior_ramp_epochs
        self.register_buffer('prior_strength', torch.tensor(0.0))
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        evidence_channels = channels + struct_channels + connectivity_channels + 2
        self.evidence_projection = nn.Linear(evidence_channels, hidden)
        self.node_projection = nn.Linear(hidden + 2, hidden)
        self.path_encoder = nn.Sequential(nn.Conv1d(hidden * 3, hidden, 3, padding=1),
                                          nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1),
                                          nn.GELU())
        self.connection_head = nn.Sequential(
            nn.Linear(hidden * (2 + samples) + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.node_update = nn.Linear(hidden, hidden)
        self.position_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, heads))
        self.write_gate = nn.Linear(hidden, 1)
        self.write_projection = nn.Linear(hidden, channels)
        self.capture_diagnostics = False
        self.last_diagnostics = None
        self.last_output = None

    @property
    def alpha_global(self):
        return self.alpha_max * torch.tanh(self.raw_alpha)

    def set_epoch(self, epoch):
        fraction = (int(epoch) + 1 - self.prior_warmup_epochs) / max(self.prior_ramp_epochs, 1)
        self.prior_strength.fill_(max(0.0, min(1.0, fraction)))

    _extract_fps_anchors = KeypointGuidedGlobalTopology._extract_fps_anchors
    _minmax_normalize_map = staticmethod(KeypointGuidedGlobalTopology._minmax_normalize_map)

    @staticmethod
    def _sample(feature, positions, reference_hw):
        shape = positions.shape[1:-1]
        height, width = reference_hw
        xy = positions[..., [1, 0]].float()
        scale = xy.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        grid = (2.0 * xy / scale - 1.0).reshape(positions.shape[0], -1, 1, 2)
        sampled = F.grid_sample(feature.float(), grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2).reshape(positions.shape[0], *shape, feature.shape[1])

    @staticmethod
    def _gather_nodes(nodes, indices):
        batch = torch.arange(nodes.shape[0], device=nodes.device)[:, None, None]
        return nodes[batch, indices]

    def _candidates(self, coords, valid):
        distance = torch.cdist(coords.float(), coords.float())
        allowed = valid[:, :, None] & valid[:, None, :] & (distance > 0)
        distance = distance.masked_fill(~allowed, float('inf'))
        lengths, neighbours = distance.topk(self.neighbours, dim=-1, largest=False)
        source = torch.arange(self.max_nodes, device=coords.device)[None, :, None].expand_as(neighbours)
        # Canonical orientation makes reverse edges use identical evidence/logits.
        pairs = torch.stack((torch.minimum(source, neighbours), torch.maximum(source, neighbours)), -1)
        edge_valid = torch.isfinite(lengths) & (lengths <= self.max_distance)
        return pairs, edge_valid

    def _corridors(self, coords, pairs):
        a = self._gather_nodes(coords, pairs[..., 0]).float()
        b = self._gather_nodes(coords, pairs[..., 1]).float()
        delta = b - a
        length = delta.norm(dim=-1, keepdim=True)
        normal = torch.stack((-delta[..., 1], delta[..., 0]), -1) / length.clamp_min(1e-6)
        t = torch.linspace(0, 1, self.samples, device=coords.device)
        centre = a[..., None, :] + delta[..., None, :] * t[None, None, None, :, None]
        offsets = delta.new_tensor([-self.corridor_offset, 0, self.corridor_offset])
        positions = centre[..., None, :, :] + normal[..., None, None, :] * offsets[None, None, None, :, None, None]
        geometry = torch.cat((delta / self.max_distance, length / self.max_distance), -1)
        return positions, geometry

    def _node_attention(self, nodes, coords, pairs, valid, logits):
        batch, count, hidden = nodes.shape
        pair_keys = pairs[..., 0] * count + pairs[..., 1]
        # scatter_reduce keeps one undirected probability if both directed candidates exist.
        edge_prob = torch.sigmoid(logits)
        dense_prob = nodes.new_zeros(batch, count * count)
        dense_prob.scatter_reduce_(1, pair_keys.flatten(1),
                                   (edge_prob * valid).flatten(1), reduce='amax', include_self=True)
        dense_prob = dense_prob.view(batch, count, count)
        dense_prob = torch.maximum(dense_prob, dense_prob.transpose(1, 2))
        self_mask = torch.eye(count, device=nodes.device, dtype=torch.bool)[None]
        allowed = (dense_prob > 0) | self_mask
        prior = torch.where(self_mask, torch.ones_like(dense_prob), dense_prob)
        qkv = self.qkv(nodes).view(batch, count, 3, self.heads, hidden // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        relative = (coords[:, :, None] - coords[:, None, :]).float() / self.max_distance
        geometry = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
        bias = self.position_bias(geometry).permute(0, 3, 1, 2)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(hidden // self.heads)
        attention = attention + bias + self.prior_strength * prior.clamp_min(1e-6).log()[:, None]
        attention = attention.masked_fill(~allowed[:, None], float('-inf')).softmax(-1)
        update = (attention @ v).transpose(1, 2).reshape(batch, count, hidden)
        return self.node_update(update) - nodes

    def _writeback(self, feature, positions, values, confidence):
        batch, channels, height, width = feature.shape
        positions = positions.reshape(batch, -1, 2)
        values = values.reshape(batch, -1, channels)
        confidence = confidence.reshape(batch, -1, 1)
        y, x = positions[..., 0], positions[..., 1]
        y0, x0 = y.floor(), x.floor()
        numerator = feature.new_zeros(batch, channels, height * width)
        denominator = feature.new_zeros(batch, 1, height * width)
        # Bilinear splatting retains the original sample coordinates. Normalization
        # excludes learned confidence so a single low-confidence edge stays weak.
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            yi, xi = y0 + dy, x0 + dx
            inside = (yi >= 0) & (yi < height) & (xi >= 0) & (xi < width)
            weight = (1 - (y - yi).abs()) * (1 - (x - xi).abs()) * inside
            index = (yi.clamp(0, height - 1).long() * width + xi.clamp(0, width - 1).long())[:, None]
            weighted = values * confidence * weight[..., None]
            numerator.scatter_add_(2, index.expand(-1, channels, -1), weighted.transpose(1, 2))
            denominator.scatter_add_(2, index, weight[:, None])
        residual = (numerator / denominator.clamp_min(1)).view_as(feature)
        return feature + self.alpha_global * residual, residual

    def forward_feature_anchors(self, feature, z_struct, surface_prob,
                                connectivity_feature=None, skeleton_prob=None):
        self.last_output = None
        if not self.enable_global_topology:
            return feature
        height, width = feature.shape[-2:]
        structure_score = self._minmax_normalize_map(z_struct.detach().float().norm(dim=1, keepdim=True))
        score = F.interpolate(structure_score, size=(height, width), mode='bilinear', align_corners=False)
        coords, node_valid, _, _ = self._extract_fps_anchors(score * surface_prob.detach().float())
        pairs, edge_valid = self._candidates(coords, node_valid)
        positions, geometry = self._corridors(coords, pairs)
        if connectivity_feature is None:
            connectivity_feature = feature.new_zeros(feature.shape[0], 8, height, width)
        if skeleton_prob is None:
            skeleton_prob = structure_score
        maps = (feature, z_struct, connectivity_feature, skeleton_prob, surface_prob.detach())
        node_evidence = torch.cat([self._sample(item, coords, (height, width)) for item in maps], -1)
        projected = self.evidence_projection(node_evidence)
        xy = coords.float() / coords.new_tensor([max(height - 1, 1), max(width - 1, 1)])
        nodes = self.node_projection(torch.cat((projected, xy), -1))
        path_evidence = torch.cat([self._sample(item, positions, (height, width)) for item in maps], -1)
        path = self.evidence_projection(path_evidence)
        batch, count, neighbours, lanes, samples, hidden = path.shape
        ordered = path.permute(0, 1, 2, 3, 5, 4).reshape(-1, lanes * hidden, samples)
        encoded = self.path_encoder(ordered).reshape(batch, count, neighbours, hidden * samples)
        a = self._gather_nodes(nodes, pairs[..., 0])
        b = self._gather_nodes(nodes, pairs[..., 1])
        edge_logits = self.connection_head(torch.cat((a + b, (a - b).abs(), encoded, geometry), -1)).squeeze(-1)
        update = self._node_attention(nodes, coords, pairs, edge_valid, edge_logits)
        relation_update = (self._gather_nodes(update, pairs[..., 0]) +
                           self._gather_nodes(update, pairs[..., 1])) * 0.5
        values = torch.tanh(self.write_projection(relation_update))[..., None, None, :].expand(-1, -1, -1, lanes, samples, -1)
        gates = torch.sigmoid(self.write_gate(path)).squeeze(-1)
        # Invalid duplicate/dummy edges must not dilute a valid edge's writeback.
        confidence = (torch.sigmoid(edge_logits)[..., None, None] * gates *
                      edge_valid[..., None, None] * self.prior_strength)
        positions = torch.where(edge_valid[..., None, None, None], positions,
                                torch.full_like(positions, -2.0))
        output, residual = self._writeback(feature, positions, values, confidence)
        self.last_output = dict(stage='global_anchor_topology', edge_logits=edge_logits,
                                edge_pairs=pairs, edge_valid=edge_valid, anchor_coords=coords,
                                connection_probability=(torch.sigmoid(edge_logits).detach() * edge_valid).sum() / edge_valid.sum().clamp_min(1),
                                candidate_edges=edge_valid.sum().detach(),
                                writeback_abs_mean=(self.alpha_global.detach() * residual.detach()).abs().mean(),
                                prior_strength=self.prior_strength.detach().clone())
        return output
