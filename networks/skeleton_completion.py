"""Differentiable skeleton completion with multi-scale candidate edges."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOCAL_OFFSETS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _shift(x, dy, dx):
    _, _, height, width = x.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return padded[:, :, y0:y0 + height, x0:x0 + width]


def build_candidate_offsets(radii=(1, 2, 4, 8)):
    offsets = []
    seen = set()
    axis = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for radius in radii:
        for dy, dx in axis:
            step = (dy * radius, dx * radius)
            if step not in seen:
                seen.add(step)
                offsets.append(step)
    return tuple(offsets)


def direction_axis_from_logits(direction_logits):
    direction = F.normalize(direction_logits, dim=1, eps=1e-6)
    cos_theta = direction[:, 0:1]
    sin_theta = direction[:, 1:2]
    cos2theta = cos_theta * cos_theta - sin_theta * sin_theta
    sin2theta = 2.0 * cos_theta * sin_theta
    return torch.cat([cos2theta, sin2theta], dim=1)


def undirected_direction_alignment(cos2sin2, dy, dx):
    phi = math.atan2(dy, dx)
    cos2phi = math.cos(2.0 * phi)
    sin2phi = math.sin(2.0 * phi)
    cos2theta = cos2sin2[:, 0:1]
    sin2theta = cos2sin2[:, 1:2]
    align_i = (1.0 + cos2theta * cos2phi + sin2theta * sin2phi) * 0.5
    back_phi = phi + math.pi
    cos2phi_back = math.cos(2.0 * back_phi)
    sin2phi_back = math.sin(2.0 * back_phi)
    return align_i.clamp(0.0, 1.0), cos2phi_back, sin2phi_back


class SkeletonCompletionBranch(nn.Module):
    """Score multi-scale candidate edges and render soft bridge paths onto skeleton logits."""

    def __init__(
        self,
        embed_channels=16,
        path_samples=5,
        dist_tau=4.0,
        node_tau=0.15,
        bridge_lambda_init=0.0,
        radii=(1, 2, 4, 8),
    ):
        super().__init__()
        self.embed_channels = int(embed_channels)
        self.path_samples = int(path_samples)
        self.dist_tau = float(dist_tau)
        self.node_tau = float(node_tau)
        self.candidate_offsets = build_candidate_offsets(radii)
        self.lambda_bridge = nn.Parameter(torch.tensor(float(bridge_lambda_init)))
        self.edge_mlp = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def _sample_line_mean(self, field, dy, dx):
        if self.path_samples <= 1:
            return _shift(field, dy, dx)
        total = torch.zeros_like(field)
        count = 0
        for step in range(1, self.path_samples):
            sy = int(round(dy * step / self.path_samples))
            sx = int(round(dx * step / self.path_samples))
            if sy == 0 and sx == 0:
                continue
            total = total + _shift(field, sy, sx)
            count += 1
        if count == 0:
            return _shift(field, dy, dx)
        return total / float(count)

    def _appearance_similarity(self, embedding, dy, dx):
        neighbor = _shift(embedding, dy, dx)
        return (
            (embedding * neighbor).sum(dim=1, keepdim=True).add(1.0).mul(0.5)
        ).clamp(0.0, 1.0)

    def _thin_support(self, skeleton_prob, dy, dx):
        center = self._sample_line_mean(skeleton_prob, dy, dx)
        if abs(dy) == 0 or abs(dx) == 0:
            side_a = _shift(skeleton_prob, 0, 1 if dx >= 0 else -1)
            side_b = _shift(skeleton_prob, 1 if dy >= 0 else -1, 0)
        else:
            side_a = _shift(skeleton_prob, dy, 0)
            side_b = _shift(skeleton_prob, 0, dx)
        side = 0.5 * (side_a + side_b)
        return (center - side).clamp_min(0.0)

    def forward(
        self,
        skeleton_logits_0,
        direction_logits,
        embedding,
        quality,
        connectivity_logits=None,
        road_prob=None,
    ):
        s0_prob = torch.sigmoid(skeleton_logits_0)
        cos2sin2 = direction_axis_from_logits(direction_logits)
        embed_norm = F.normalize(embedding, dim=1, eps=1e-6)
        node_mask = torch.sigmoid(
            (s0_prob - self.node_tau) / 0.05
        ) * quality

        edge_logits = []
        edge_probs = []
        path_delta = torch.zeros_like(skeleton_logits_0)

        for dy, dx in self.candidate_offsets:
            dist = math.sqrt(float(dy * dy + dx * dx))
            neighbor_s0 = _shift(s0_prob, dy, dx)
            neighbor_q = _shift(quality, dy, dx)
            neighbor_cos2sin2 = _shift(cos2sin2, dy, dx)

            align_i, cos2phi_back, sin2phi_back = undirected_direction_alignment(
                cos2sin2,
                dy,
                dx,
            )
            align_j = (
                1.0
                + neighbor_cos2sin2[:, 0:1] * cos2phi_back
                + neighbor_cos2sin2[:, 1:2] * sin2phi_back
            ).mul(0.5).clamp(0.0, 1.0)
            g_dir = torch.sqrt(align_i * align_j).clamp(0.0, 1.0)

            g_app = self._appearance_similarity(embed_norm, dy, dx)
            path_field = road_prob if road_prob is not None else s0_prob
            g_path = self._sample_line_mean(path_field, dy, dx)
            g_thin = self._thin_support(s0_prob, dy, dx)
            g_dist = torch.full_like(s0_prob, math.exp(-dist / self.dist_tau))

            edge_features = torch.cat(
                [
                    s0_prob,
                    neighbor_s0,
                    g_dir,
                    g_app,
                    g_path,
                    g_thin,
                    g_dist,
                    quality,
                    neighbor_q,
                ],
                dim=1,
            )
            edge_logit = self.edge_mlp(
                edge_features.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            edge_prob = torch.sigmoid(edge_logit)
            endpoint_mask = node_mask * neighbor_q
            edge_prob = edge_prob * endpoint_mask

            edge_logits.append(edge_logit)
            edge_probs.append(edge_prob)

            bridge = g_path * edge_prob * (1.0 - s0_prob)
            path_delta = path_delta + bridge

        if connectivity_logits is not None and len(edge_probs) >= 8:
            local_conn = torch.sigmoid(connectivity_logits[:, :8])
            for idx in range(8):
                path_delta = path_delta + (
                    0.25
                    * edge_probs[idx]
                    * local_conn[:, idx:idx + 1]
                    * (1.0 - s0_prob)
                )

        connected_logits = skeleton_logits_0 + self.lambda_bridge * path_delta
        aux = {
            "s0_logits": skeleton_logits_0,
            "connected_logits": connected_logits,
            "path_delta": path_delta,
            "edge_logits": torch.stack(edge_logits, dim=1),
            "edge_probs": torch.stack(edge_probs, dim=1),
            "direction_logits": direction_logits,
            "direction_axis": cos2sin2,
            "embedding": embed_norm,
            "quality": quality,
        }
        return connected_logits, aux
