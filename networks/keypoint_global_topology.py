import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointGuidedGlobalTopology(nn.Module):
    """Sparse global residual using structure-selected anchors and fused topology tokens."""

    def __init__(
        self,
        channels,
        struct_channels,
        max_nodes=32,
        heads=4,
        alpha_max=0.05,
        enabled=False,
        connectivity_channels=8,
        direction_channels=2,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = int(channels)
        self.struct_channels = int(struct_channels)
        self.max_nodes = int(max_nodes)
        self.heads = int(heads)
        self.alpha_max = float(alpha_max)
        self.enable_global_topology = bool(enabled)
        self.connectivity_channels = int(connectivity_channels)
        self.direction_channels = int(direction_channels)
        self.node_type_embedding = nn.Embedding(1, 8)
        token_input_channels = (
            self.struct_channels
            + self.channels
            + self.connectivity_channels
            + self.direction_channels
            + 8
            + 2
        )
        self.node_projection = nn.Linear(token_input_channels, channels)
        relation_hidden = max(channels // 4, 32)
        self.token_relation_qkv = nn.Linear(channels, channels * 3)
        self.token_relation_projection = nn.Linear(channels, channels)
        self.token_relation_bias = nn.Sequential(
            nn.Linear(4, relation_hidden),
            nn.GELU(),
            nn.Linear(relation_hidden, heads),
        )
        self.token_relation_scale = nn.Parameter(torch.tensor(0.1))
        self.connectivity_topology_bias_scale = nn.Parameter(torch.tensor(0.5))
        self.direction_topology_bias_scale = nn.Parameter(torch.tensor(0.5))
        self.grid_q = nn.Linear(channels, channels)
        self.node_kv = nn.Linear(channels, channels * 2)
        self.output_projection = nn.Linear(channels, channels)
        self.grid_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.capture_diagnostics = False
        self.last_diagnostics = None

    @property
    def alpha_global(self):
        return self.alpha_max * torch.tanh(self.raw_alpha)

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

    def _sample_features_at_anchor_coords(self, feature, coords, anchor_hw):
        batch, channels, height, width = feature.shape
        anchor_height, anchor_width = anchor_hw
        y = coords[..., 0].float()
        x = coords[..., 1].float()
        if anchor_height != height:
            y = y * float(max(height - 1, 0)) / float(max(anchor_height - 1, 1))
        if anchor_width != width:
            x = x * float(max(width - 1, 0)) / float(max(anchor_width - 1, 1))
        y = y.round().long().clamp(0, max(height - 1, 0))
        x = x.round().long().clamp(0, max(width - 1, 0))

        flat = feature.flatten(2).transpose(1, 2)
        index = (y * width + x).unsqueeze(-1).expand(-1, -1, channels)
        return torch.gather(flat, 1, index)

    def _prepare_token_map(
        self,
        feature,
        batch,
        target_hw,
        expected_channels,
        dtype,
        device,
    ):
        target_height, target_width = target_hw
        if feature is None:
            return torch.zeros(
                batch,
                expected_channels,
                target_height,
                target_width,
                device=device,
                dtype=dtype,
            )
        feature = feature.to(device=device, dtype=dtype)
        if feature.shape[-2:] != target_hw:
            feature = F.interpolate(
                feature,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        channels = feature.shape[1]
        if channels == expected_channels:
            return feature
        if channels > expected_channels:
            return feature[:, :expected_channels]
        pad = feature.new_zeros(
            feature.shape[0],
            expected_channels - channels,
            target_height,
            target_width,
        )
        return torch.cat([feature, pad], dim=1)

    def _relative_topology_bias(self, coords, valid, anchor_hw):
        anchor_height, anchor_width = anchor_hw
        coords_float = coords.float()
        y = coords_float[..., 0] / float(max(anchor_height - 1, 1))
        x = coords_float[..., 1] / float(max(anchor_width - 1, 1))
        dy = y[:, :, None] - y[:, None, :]
        dx = x[:, :, None] - x[:, None, :]
        distance = torch.sqrt(dx.square() + dy.square() + 1e-6)
        inv_distance = 1.0 / (1.0 + distance)
        relation = torch.stack(
            [
                dx,
                dy,
                torch.log1p(distance),
                inv_distance,
            ],
            dim=-1,
        )
        bias = self.token_relation_bias(relation).permute(0, 3, 1, 2)
        valid_pair = valid[:, None, :, None] & valid[:, None, None, :]
        return bias.masked_fill(~valid_pair, 0.0)

    def _local_topology_bias(self, connectivity_feature, direction_feature, valid):
        batch, nodes, _ = connectivity_feature.shape
        bias = connectivity_feature.new_zeros(batch, self.heads, nodes, nodes)
        valid_pair = valid[:, None, :, None] & valid[:, None, None, :]
        if connectivity_feature is not None and connectivity_feature.numel() > 0:
            conn = F.normalize(connectivity_feature.float(), dim=-1, eps=1e-6)
            conn_similarity = torch.matmul(conn, conn.transpose(1, 2))
            bias = bias + self.connectivity_topology_bias_scale * conn_similarity[:, None]
        if direction_feature is not None and direction_feature.numel() > 0:
            direction = F.normalize(direction_feature.float(), dim=-1, eps=1e-6)
            direction_similarity = torch.matmul(direction, direction.transpose(1, 2))
            bias = bias + self.direction_topology_bias_scale * direction_similarity[:, None]
        return bias.to(dtype=connectivity_feature.dtype).masked_fill(~valid_pair, 0.0)

    @staticmethod
    def _masked_corr(x, y, mask):
        values = []
        for batch_index in range(x.shape[0]):
            mask_i = mask[batch_index]
            if mask_i.sum() < 2:
                continue
            x_i = x[batch_index][mask_i].float()
            y_i = y[batch_index][mask_i].float()
            x_i = x_i - x_i.mean()
            y_i = y_i - y_i.mean()
            denom = x_i.square().mean().sqrt() * y_i.square().mean().sqrt()
            if denom <= 1e-6:
                continue
            values.append((x_i * y_i).mean() / denom)
        if not values:
            return x.new_tensor(0.0)
        return torch.stack(values).mean()

    def _refine_tokens_with_relative_topology(
        self,
        node_feature,
        coords,
        valid,
        anchor_hw,
        connectivity_feature=None,
        direction_feature=None,
    ):
        batch, nodes, channels = node_feature.shape
        qkv = self.token_relation_qkv(node_feature).reshape(
            batch,
            nodes,
            3,
            self.heads,
            channels // self.heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            channels // self.heads
        )
        qk_logits = logits
        topology_bias = self._relative_topology_bias(coords, valid, anchor_hw)
        local_topology_bias = node_feature.new_zeros(batch, self.heads, nodes, nodes)
        if connectivity_feature is not None and direction_feature is not None:
            local_topology_bias = self._local_topology_bias(
                connectivity_feature,
                direction_feature,
                valid,
            )
        b_topology = self.token_relation_scale * topology_bias + local_topology_bias
        logits = logits + b_topology
        logits = logits.masked_fill(
            ~valid[:, None, None, :],
            -torch.finfo(logits.dtype).max,
        )
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch,
            nodes,
            channels,
        )
        attended = self.token_relation_projection(attended)
        refined = node_feature + attended
        relation_diagnostics = {}
        if self.capture_diagnostics:
            with torch.no_grad():
                valid_pair = valid[:, :, None] & valid[:, None, :]
                eye = torch.eye(nodes, device=valid.device, dtype=torch.bool).unsqueeze(0)
                valid_pair = valid_pair & (~eye)
                valid_pair_heads = valid_pair[:, None, :, :].expand_as(qk_logits)
                if valid_pair_heads.any():
                    qk_valid = qk_logits.detach()[valid_pair_heads]
                    bias_valid = b_topology.detach()[valid_pair_heads]
                    qk_std = qk_valid.float().std(unbiased=False)
                    btopo_std = bias_valid.float().std(unbiased=False)
                else:
                    qk_std = qk_logits.detach().float().sum() * 0.0
                    btopo_std = b_topology.detach().float().sum() * 0.0

                attention_mean = attention.detach().mean(dim=1)
                conn_corr = qk_std.new_tensor(0.0)
                dir_corr = qk_std.new_tensor(0.0)
                if connectivity_feature is not None and connectivity_feature.numel() > 0:
                    conn = F.normalize(connectivity_feature.detach().float(), dim=-1, eps=1e-6)
                    conn_similarity = torch.matmul(conn, conn.transpose(1, 2))
                    conn_corr = self._masked_corr(attention_mean, conn_similarity, valid_pair)
                if direction_feature is not None and direction_feature.numel() > 0:
                    direction = F.normalize(direction_feature.detach().float(), dim=-1, eps=1e-6)
                    direction_similarity = torch.matmul(direction, direction.transpose(1, 2))
                    dir_corr = self._masked_corr(attention_mean, direction_similarity, valid_pair)
                relation_diagnostics = {
                    "token_qk_std": qk_std.detach(),
                    "token_btopo_std": btopo_std.detach(),
                    "token_btopo_qk_std_ratio": (
                        btopo_std / qk_std.clamp_min(1e-6)
                    ).detach(),
                    "attention_conn_corr": conn_corr.detach(),
                    "attention_dir_corr": dir_corr.detach(),
                }
        return (
            refined * valid.unsqueeze(-1).to(dtype=refined.dtype),
            topology_bias,
            local_topology_bias,
            relation_diagnostics,
        )

    def _cross_attention_from_structure_tokens(self, feature, node_feature, valid):
        batch, channels, height, width = feature.shape
        grid_tokens = feature.flatten(2).transpose(1, 2)
        query = self.grid_q(grid_tokens).reshape(
            batch,
            height * width,
            self.heads,
            channels // self.heads,
        ).permute(0, 2, 1, 3)
        node_kv = self.node_kv(node_feature).reshape(
            batch,
            self.max_nodes,
            2,
            self.heads,
            channels // self.heads,
        ).permute(2, 0, 3, 1, 4)
        key, value = node_kv[0], node_kv[1]

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

    def forward_feature_anchors(
        self,
        feature,
        z_struct,
        surface_prob,
        connectivity_feature=None,
        direction_feature=None,
    ):
        batch, channels, height, width = feature.shape
        if not self.enable_global_topology or z_struct is None or surface_prob is None:
            return feature

        with torch.no_grad():
            z_score = torch.linalg.vector_norm(
                z_struct.detach().float(),
                dim=1,
                keepdim=True,
            )
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

        sampled_struct = self._sample_features_at_anchor_coords(
            z_struct.to(dtype=feature.dtype),
            coords,
            anchor_hw=(height, width),
        )
        sampled_feature = self._sample_features_at_anchor_coords(
            feature,
            coords,
            anchor_hw=(height, width),
        )
        connectivity_map = self._prepare_token_map(
            connectivity_feature,
            batch,
            (height, width),
            self.connectivity_channels,
            feature.dtype,
            feature.device,
        )
        direction_map = self._prepare_token_map(
            direction_feature,
            batch,
            (height, width),
            self.direction_channels,
            feature.dtype,
            feature.device,
        )
        sampled_connectivity = self._sample_features_at_anchor_coords(
            connectivity_map,
            coords,
            anchor_hw=(height, width),
        )
        sampled_direction = self._sample_features_at_anchor_coords(
            direction_map,
            coords,
            anchor_hw=(height, width),
        )
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
                sampled_struct,
                sampled_feature,
                sampled_connectivity,
                sampled_direction,
                self.node_type_embedding(node_types),
                coords_norm,
            ],
            dim=-1,
        )
        node_feature = self.node_projection(node_input)
        (
            node_feature,
            topology_bias,
            local_topology_bias,
            relation_diagnostics,
        ) = self._refine_tokens_with_relative_topology(
            node_feature,
            coords,
            valid,
            anchor_hw=(height, width),
            connectivity_feature=sampled_connectivity,
            direction_feature=sampled_direction,
        )
        context = self._cross_attention_from_structure_tokens(
            feature,
            node_feature,
            valid,
        )
        delta = self.grid_projection(context)
        delta = delta * surface_gate.clamp(0.0, 1.0)
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
                    "surface_gate_mean": surface_gate.mean(dim=(1, 2, 3)).detach(),
                    "surface_gate_max": surface_gate.amax(dim=(1, 2, 3)).detach(),
                    "token_relation_scale": self.token_relation_scale.detach(),
                    "token_relation_bias_abs_mean": topology_bias.abs().mean().detach(),
                    "connectivity_topology_bias_scale": (
                        self.connectivity_topology_bias_scale.detach()
                    ),
                    "direction_topology_bias_scale": (
                        self.direction_topology_bias_scale.detach()
                    ),
                    "local_topology_bias_abs_mean": (
                        local_topology_bias.abs().mean().detach()
                    ),
                    "token_qk_std": relation_diagnostics.get(
                        "token_qk_std",
                        feature.new_tensor(0.0),
                    ).detach(),
                    "token_btopo_std": relation_diagnostics.get(
                        "token_btopo_std",
                        feature.new_tensor(0.0),
                    ).detach(),
                    "token_btopo_qk_std_ratio": relation_diagnostics.get(
                        "token_btopo_qk_std_ratio",
                        feature.new_tensor(0.0),
                    ).detach(),
                    "attention_conn_corr": relation_diagnostics.get(
                        "attention_conn_corr",
                        feature.new_tensor(0.0),
                    ).detach(),
                    "attention_dir_corr": relation_diagnostics.get(
                        "attention_dir_corr",
                        feature.new_tensor(0.0),
                    ).detach(),
                    "connectivity_token_abs_mean": (
                        sampled_connectivity.abs().mean()
                    ).detach(),
                    "direction_token_abs_mean": sampled_direction.abs().mean().detach(),
                    "global_residual_relative_norm": (
                        torch.linalg.vector_norm(output - feature)
                        / (torch.linalg.vector_norm(feature) + 1e-6)
                    ).detach(),
                }
        return output
