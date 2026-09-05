import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton_guided_head import CONNECTIVITY_DIRECTIONS


class KeypointGuidedGlobalTopology(nn.Module):
    """Sparse topology attention driven by detached dense C8 predictions."""

    def __init__(
        self,
        channels,
        max_nodes=32,
        heads=4,
        reach_hops=12,
        nms_radius=2,
        skeleton_threshold=0.5,
        connectivity_threshold=0.25,
        bend_angle_threshold=45.0,
        alpha_max=0.05,
        enabled=False,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = int(channels)
        self.max_nodes = int(max_nodes)
        self.heads = int(heads)
        self.reach_hops = int(reach_hops)
        self.nms_radius = int(nms_radius)
        self.skeleton_threshold = float(skeleton_threshold)
        self.connectivity_threshold = float(connectivity_threshold)
        self.bend_angle_threshold = float(bend_angle_threshold)
        self.alpha_max = float(alpha_max)
        self.enable_global_topology = bool(enabled)
        self.node_type_embedding = nn.Embedding(3, 8)
        self.node_projection = nn.Linear(channels + 2 + 8 + 2, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.output_projection = nn.Linear(channels, channels)
        self.grid_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.capture_diagnostics = False
        self.last_diagnostics = None

    @property
    def alpha_global(self):
        return self.alpha_max * torch.tanh(self.raw_alpha)

    @staticmethod
    def _shift_from_neighbor(x, dy, dx):
        height, width = x.shape[-2:]
        pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
        padded = F.pad(x, pad)
        y0 = max(-dy, 0)
        x0 = max(-dx, 0)
        return padded[..., y0:y0 + height, x0:x0 + width]

    def build_symmetric_connectivity(self, connectivity_prob):
        values = []
        directions = tuple(CONNECTIVITY_DIRECTIONS)
        for index, (dy, dx) in enumerate(directions):
            opposite = directions.index((-dy, -dx))
            neighbor = self._shift_from_neighbor(
                connectivity_prob[:, opposite:opposite + 1], dy, dx
            )
            values.append(torch.sqrt((connectivity_prob[:, index:index + 1] * neighbor).clamp_min(0.0)))
        return torch.cat(values, dim=1)

    def _direction_alignment(self, direction, coords_i, coords_j):
        delta = coords_j.float() - coords_i.float()
        length = delta.norm(dim=-1, keepdim=True).clamp_min(1.0)
        unit = delta / length
        direction_flat = direction.flatten(2).transpose(1, 2)
        width = direction.shape[-1]
        index_i = (coords_i[..., 0] * width + coords_i[..., 1]).reshape(coords_i.shape[0], -1)
        index_j = (coords_j[..., 0] * width + coords_j[..., 1]).reshape(coords_j.shape[0], -1)
        dir_i = torch.gather(direction_flat, 1, index_i.unsqueeze(-1).expand(-1, -1, 2)).reshape(*coords_i.shape[:-1], 2)
        dir_j = torch.gather(direction_flat, 1, index_j.unsqueeze(-1).expand(-1, -1, 2)).reshape(*coords_j.shape[:-1], 2)
        phi_cos = unit[..., 1]
        phi_sin = unit[..., 0]
        edge_cos2 = phi_cos.square() - phi_sin.square()
        edge_sin2 = 2.0 * phi_cos * phi_sin
        align_i = ((1.0 + dir_i[..., 0:1] * edge_cos2.unsqueeze(-1) + dir_i[..., 1:2] * edge_sin2.unsqueeze(-1)) / 2.0).clamp_min(0.0).sqrt()
        reverse_cos2 = edge_cos2
        reverse_sin2 = edge_sin2
        align_j = ((1.0 + dir_j[..., 0:1] * reverse_cos2.unsqueeze(-1) + dir_j[..., 1:2] * reverse_sin2.unsqueeze(-1)) / 2.0).clamp_min(0.0).sqrt()
        return 0.5 * (align_i.squeeze(-1) + align_j.squeeze(-1))

    @torch.no_grad()
    def extract_keypoints(self, skeleton_prob, symmetric_connectivity, direction):
        batch, _, height, width = skeleton_prob.shape
        degree = (symmetric_connectivity >= self.connectivity_threshold).sum(dim=1)
        direction_values = symmetric_connectivity
        top2 = direction_values.topk(k=min(2, direction_values.shape[1]), dim=1).indices
        vectors = torch.tensor(CONNECTIVITY_DIRECTIONS, device=skeleton_prob.device, dtype=torch.float32)
        top2_vectors = vectors[top2.permute(0, 2, 3, 1)]
        first = top2_vectors[..., 0, :]
        second = top2_vectors[..., 1, :]
        dot = (first * second).sum(dim=-1).abs()
        bend = (degree == 2) & (dot < math.cos(math.radians(self.bend_angle_threshold)))
        endpoint = degree == 1
        junction = degree >= 3
        scores = torch.stack(
            [skeleton_prob[:, 0] * endpoint.float(), skeleton_prob[:, 0] * junction.float(), skeleton_prob[:, 0] * bend.float()],
            dim=1,
        )
        scores = scores * (skeleton_prob[:, 0:1] >= self.skeleton_threshold).float()
        pooled = F.max_pool2d(scores.reshape(batch * 3, 1, height, width), 2 * self.nms_radius + 1, 1, self.nms_radius).reshape(batch, 3, height, width)
        candidates = scores * (scores >= pooled).float()
        flat_scores = candidates.reshape(batch, -1)
        count = min(self.max_nodes, flat_scores.shape[1])
        values, indices = flat_scores.topk(count, dim=1)
        valid = values > 0
        node_types = indices // (height * width)
        flat = indices % (height * width)
        ys = flat // width
        xs = flat % width
        coords = torch.stack([ys, xs], dim=-1)
        return coords, node_types, valid, values

    @torch.no_grad()
    def build_multisource_reachability(self, seeds, symmetric_connectivity):
        batch, nodes, height, width = seeds.shape
        reach = seeds.clone()
        for _ in range(self.reach_hops):
            candidates = [reach]
            for index, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
                candidates.append(self._shift_from_neighbor(reach, dy, dx) * symmetric_connectivity[:, index:index + 1])
            reach = torch.stack(candidates, dim=0).amax(dim=0)
        return reach.clamp(0.0, 1.0)

    @torch.no_grad()
    def build_node_adjacency(self, reach, coords, valid, direction):
        batch, nodes, height, width = reach.shape
        flat_reach = reach.flatten(2)
        target_index = (coords[..., 0] * width + coords[..., 1]).unsqueeze(1).expand(batch, nodes, nodes)
        adjacency = torch.gather(flat_reach, 2, target_index)
        adjacency = torch.minimum(adjacency, adjacency.transpose(1, 2))
        alignment = self._direction_alignment(direction, coords.unsqueeze(2), coords.unsqueeze(1))
        adjacency = adjacency * alignment
        valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2)
        adjacency = adjacency * valid_pair.float()
        eye = torch.eye(nodes, device=reach.device).unsqueeze(0)
        return adjacency * (1.0 - eye)

    def _sample_features(self, feature, coords):
        batch, channels, height, width = feature.shape
        flat = feature.flatten(2).transpose(1, 2)
        index = (coords[..., 0] * width + coords[..., 1]).unsqueeze(-1).expand(-1, -1, channels)
        return torch.gather(flat, 1, index)

    @staticmethod
    def _minmax_normalize_map(score):
        flat = score.flatten(1)
        low = flat.amin(dim=1).view(-1, 1, 1, 1)
        high = flat.amax(dim=1).view(-1, 1, 1, 1)
        return (score - low) / (high - low).clamp_min(1e-6)

    @torch.no_grad()
    def _extract_fps_anchors(self, anchor_score):
        batch, _, height, width = anchor_score.shape
        flat_score = anchor_score.flatten(1)
        candidate_count = min(max(256, self.max_nodes), flat_score.shape[1])
        values, flat_indices = flat_score.topk(candidate_count, dim=1)
        ys = flat_indices // width
        xs = flat_indices % width
        candidate_coords = torch.stack([ys, xs], dim=-1)

        coords = torch.zeros(
            batch,
            self.max_nodes,
            2,
            device=anchor_score.device,
            dtype=torch.long,
        )
        scores = torch.zeros(
            batch,
            self.max_nodes,
            device=anchor_score.device,
            dtype=anchor_score.dtype,
        )
        valid = torch.zeros(
            batch,
            self.max_nodes,
            device=anchor_score.device,
            dtype=torch.bool,
        )

        for batch_index in range(batch):
            candidate_valid = values[batch_index] > 0
            num_candidates = int(candidate_valid.sum().item())
            if num_candidates == 0:
                continue
            sample_coords = candidate_coords[batch_index, :num_candidates]
            sample_scores = values[batch_index, :num_candidates]
            sample_coords_float = sample_coords.float()

            selected = [0]
            min_dist_sq = (
                (sample_coords_float - sample_coords_float[0:1]).square().sum(dim=1)
            )
            max_select = min(self.max_nodes, num_candidates)
            for _ in range(1, max_select):
                min_dist_sq[selected] = -1.0
                next_index = int(torch.argmax(min_dist_sq).item())
                if min_dist_sq[next_index] < 0:
                    break
                selected.append(next_index)
                dist_sq = (
                    (sample_coords_float - sample_coords_float[next_index:next_index + 1])
                    .square()
                    .sum(dim=1)
                )
                min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

            selected_tensor = torch.as_tensor(
                selected,
                device=anchor_score.device,
                dtype=torch.long,
            )
            count = selected_tensor.numel()
            coords[batch_index, :count] = sample_coords[selected_tensor]
            scores[batch_index, :count] = sample_scores[selected_tensor]
            valid[batch_index, :count] = True
        return coords, valid, scores, candidate_count

    def _cross_attention_from_tokens(self, feature, node_feature, valid):
        batch, channels, height, width = feature.shape
        grid_tokens = feature.flatten(2).transpose(1, 2)
        grid_qkv = self.qkv(grid_tokens).reshape(
            batch,
            height * width,
            3,
            self.heads,
            channels // self.heads,
        ).permute(2, 0, 3, 1, 4)
        node_qkv = self.qkv(node_feature).reshape(
            batch,
            self.max_nodes,
            3,
            self.heads,
            channels // self.heads,
        ).permute(2, 0, 3, 1, 4)

        query = grid_qkv[0]
        key = node_qkv[1]
        value = node_qkv[2]
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            channels // self.heads
        )
        logits = logits.masked_fill(
            ~valid[:, None, None, :],
            -torch.finfo(logits.dtype).max,
        )
        attention = torch.softmax(logits, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch,
            height * width,
            channels,
        )
        context = self.output_projection(context)
        context = context.transpose(1, 2).reshape(batch, channels, height, width)
        has_anchor = valid.any(dim=1).to(dtype=context.dtype).view(batch, 1, 1, 1)
        return context * has_anchor

    def forward_feature_anchors(self, feature, z_struct, surface_prob):
        batch, channels, height, width = feature.shape
        if not self.enable_global_topology or z_struct is None or surface_prob is None:
            return feature

        with torch.no_grad():
            z_score = torch.linalg.vector_norm(z_struct.detach().float(), dim=1, keepdim=True)
            z_score = self._minmax_normalize_map(z_score).to(dtype=feature.dtype)
            if z_score.shape[-2:] != (height, width):
                z_score = F.interpolate(
                    z_score,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            surface_gate = surface_prob.detach().to(dtype=feature.dtype)
            if surface_gate.shape[-2:] != (height, width):
                surface_gate = F.interpolate(
                    surface_gate,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            anchor_score = z_score * surface_gate.clamp(0.0, 1.0)
            coords, valid, scores, candidate_count = self._extract_fps_anchors(anchor_score)

        sampled_feature = self._sample_features(feature, coords)
        zeros_direction = sampled_feature.new_zeros(batch, self.max_nodes, 2)
        node_types = torch.zeros(
            batch,
            self.max_nodes,
            device=feature.device,
            dtype=torch.long,
        )
        coords_norm = coords.float() / feature.new_tensor(
            [max(height - 1, 1), max(width - 1, 1)]
        )
        node_input = torch.cat(
            [
                sampled_feature,
                zeros_direction,
                self.node_type_embedding(node_types),
                coords_norm,
            ],
            dim=-1,
        )
        node_feature = self.node_projection(node_input)
        context = self._cross_attention_from_tokens(feature, node_feature, valid)
        delta = self.grid_projection(context)
        output = feature + self.alpha_global * delta

        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "anchor_count": valid.sum(dim=1).float().detach(),
                    "candidate_count": feature.new_full(
                        (batch,),
                        float(candidate_count),
                    ).detach(),
                    "anchor_score_mean": scores.masked_fill(~valid, 0.0).sum(dim=1)
                    / valid.sum(dim=1).clamp_min(1).float(),
                    "anchor_score_max": scores.amax(dim=1).detach(),
                    "alpha_global": self.alpha_global.detach(),
                    "global_residual_relative_norm": (
                        torch.linalg.vector_norm(output - feature)
                        / (torch.linalg.vector_norm(feature) + 1e-6)
                    ).detach(),
                }
        return output

    def _node_attention(self, node_features, valid, adjacency):
        batch, nodes, channels = node_features.shape
        qkv = self.qkv(node_features).reshape(batch, nodes, 3, self.heads, channels // self.heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(channels // self.heads)
        global_reach = adjacency + 0.5 * torch.matmul(adjacency, adjacency) + 0.25 * torch.matmul(torch.matmul(adjacency, adjacency), adjacency)
        topology_bias = torch.log(global_reach.clamp_min(1e-4)).unsqueeze(1)
        logits = logits + topology_bias
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.finfo(logits.dtype).max)
        attention = torch.softmax(logits, dim=-1)
        if self.capture_diagnostics:
            with torch.no_grad():
                pair_attention = attention.mean(dim=1)
                eye = torch.eye(nodes, device=attention.device, dtype=torch.bool).unsqueeze(0)
                valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2) & ~eye
                same_pair = (adjacency > 1e-6) & valid_pair
                different_pair = (adjacency <= 1e-6) & valid_pair

                def masked_mean(values, mask):
                    count = mask.sum(dim=(1, 2))
                    total = (values * mask.float()).sum(dim=(1, 2))
                    mean = total / count.clamp_min(1).float()
                    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))

                self.last_attention_same_mean = masked_mean(pair_attention, same_pair).detach()
                self.last_attention_different_mean = masked_mean(pair_attention, different_pair).detach()
                self.last_attention_delta = (
                    self.last_attention_same_mean - self.last_attention_different_mean
                ).detach()
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, nodes, channels)
        attended = self.output_projection(attended)
        attended = attended * valid.unsqueeze(-1).float()
        return attended, global_reach, topology_bias

    def forward(self, feature, skeleton_prob, connectivity_prob, direction):
        if not self.training and not self.capture_diagnostics:
            capture = False
        else:
            capture = self.capture_diagnostics
        batch, channels, height, width = feature.shape
        if not self.enable_global_topology:
            return feature
        with torch.no_grad():
            symmetric = self.build_symmetric_connectivity(connectivity_prob.detach())
            coords, node_types, valid, scores = self.extract_keypoints(
                skeleton_prob.detach(), symmetric, direction.detach()
            )
            seeds = feature.new_zeros((batch, self.max_nodes, height, width))
            seed_flat = seeds.flatten(2)
            seed_index = (coords[..., 0] * width + coords[..., 1]).unsqueeze(-1)
            seed_flat.scatter_(2, seed_index, valid.unsqueeze(-1).float())
            reach = self.build_multisource_reachability(seed_flat.view_as(seeds), symmetric)
            adjacency = self.build_node_adjacency(reach, coords, valid, direction.detach())
        node_feature = self._sample_features(feature, coords)
        direction_node = self._sample_features(direction[:, :2], coords)
        coords_norm = coords.float() / feature.new_tensor([max(height - 1, 1), max(width - 1, 1)])
        node_input = torch.cat([node_feature, direction_node, self.node_type_embedding(node_types), coords_norm], dim=-1)
        node_feature = self.node_projection(node_input)
        attended, global_reach, topology_bias = self._node_attention(node_feature, valid, adjacency)
        weights = reach * skeleton_prob.detach()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        global_context = torch.einsum("bnhw,bnc->bchw", weights, attended).reshape(batch, channels, height, width)
        delta = self.grid_projection(global_context)
        output = feature + self.alpha_global * delta
        if capture:
            with torch.no_grad():
                active = valid.sum(dim=1).float()
                self.last_diagnostics = {
                    "node_count": active.detach(),
                    "endpoint_count": ((node_types == 0) & valid).sum(dim=1).detach(),
                    "junction_count": ((node_types == 1) & valid).sum(dim=1).detach(),
                    "bend_count": ((node_types == 2) & valid).sum(dim=1).detach(),
                    "active_node_pair_ratio": (adjacency > 0).float().mean(dim=(1, 2)).detach(),
                    "local_adjacency_mean": adjacency.mean(dim=(1, 2)).detach(),
                    "local_adjacency_max": adjacency.amax(dim=(1, 2)).detach(),
                    "global_reachability_mean": global_reach.mean(dim=(1, 2)).detach(),
                    "global_reachability_max": global_reach.amax(dim=(1, 2)).detach(),
                    "attention_topology_bias_mean": topology_bias.mean().detach(),
                    "attention_topology_bias_max": topology_bias.amax().detach(),
                    "alpha_global": self.alpha_global.detach(),
                    "global_residual_relative_norm": (torch.linalg.vector_norm(output - feature) / (torch.linalg.vector_norm(feature) + 1e-6)).detach(),
                    "node_feature_relative_change": (torch.linalg.vector_norm(attended - node_feature) / (torch.linalg.vector_norm(node_feature) + 1e-6)).detach(),
                    "attention_same_mean": self.last_attention_same_mean.detach(),
                    "attention_different_mean": self.last_attention_different_mean.detach(),
                    "attention_delta": self.last_attention_delta.detach(),
                }
        return output

