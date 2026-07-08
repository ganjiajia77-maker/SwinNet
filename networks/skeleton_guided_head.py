import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SkeletonSpatialHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = ConvBNReLU(channels, channels, kernel_size=1, padding=0)
        self.context = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dir_h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.dir_v = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.dir_fuse = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.context(x)
        x = self.depthwise(x)
        x = self.dir_fuse(self.dir_h(x) + self.dir_v(x))
        return self.out(x)


class StageTopologyPredictor(nn.Module):
    def __init__(self, channels, connectivity_channels=8):
        super().__init__()
        self.structure_branch = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )
        self.skeleton_head = SkeletonSpatialHead(channels)
        self.connectivity_head = nn.Conv2d(
            channels,
            connectivity_channels,
            kernel_size=1,
        )

    def forward(self, x):
        structure_feat = self.structure_branch(x)
        return (
            self.skeleton_head(structure_feat),
            self.connectivity_head(structure_feat),
        )


class LightweightTopologyGate(nn.Module):
    def __init__(
        self,
        channels,
        gamma_max=0.05,
        gamma_init=0.005,
        trainable=True,
    ):
        super().__init__()
        self.gamma_max = float(gamma_max)
        gamma_ratio = float(gamma_init) / self.gamma_max
        if not 0.0 <= gamma_ratio < 1.0:
            raise ValueError("gamma_init must be in [0, gamma_max)")
        if trainable and gamma_ratio == 0.0:
            raise ValueError("trainable gamma_init must be greater than zero")
        self.register_buffer(
            "fixed_gamma",
            torch.tensor(float(gamma_init)),
        )
        raw_gamma_init = (
            torch.tensor(0.0)
            if gamma_ratio == 0.0
            else torch.logit(torch.tensor(gamma_ratio))
        )
        self.raw_gamma = nn.Parameter(
            raw_gamma_init,
            requires_grad=bool(trainable),
        )
        self.feature_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.capture_gate_diagnostics = False
        self.last_gate_diagnostics = None

    def effective_gamma(self):
        if not self.raw_gamma.requires_grad:
            return self.fixed_gamma
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    def forward(self, feature_tokens, skeleton_prob, connectivity_prob):
        batch, length, channels = feature_tokens.shape
        height, width = skeleton_prob.shape[-2:]
        if length != height * width:
            raise ValueError("Topology gate feature and topology sizes do not match.")

        feature = feature_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
        conn_strength = connectivity_prob.topk(
            k=min(2, connectivity_prob.shape[1]),
            dim=1,
        ).values.mean(dim=1, keepdim=True)
        topo_conf = skeleton_prob * conn_strength
        gated_residual = topo_conf * self.feature_proj(feature)
        gamma = self.effective_gamma()
        scaled_residual = gamma * gated_residual
        if self.capture_gate_diagnostics:
            with torch.no_grad():
                feature_norm = torch.linalg.vector_norm(feature)
                residual_norm = torch.linalg.vector_norm(scaled_residual)
                self.last_gate_diagnostics = {
                    "gamma_gate_eff": float(gamma.detach().cpu()),
                    "topo_conf_mean": float(topo_conf.mean().detach().cpu()),
                    "topo_conf_max": float(topo_conf.max().detach().cpu()),
                    "topo_conf_gt_0_1_ratio": float(
                        (topo_conf > 0.1).float().mean().detach().cpu()
                    ),
                    "topo_conf_gt_0_3_ratio": float(
                        (topo_conf > 0.3).float().mean().detach().cpu()
                    ),
                    "gamma_topo_conf_mean": float(
                        (gamma * topo_conf).mean().detach().cpu()
                    ),
                    "gamma_topo_conf_max": float(
                        (gamma * topo_conf).max().detach().cpu()
                    ),
                    "scaled_residual_norm": float(residual_norm.detach().cpu()),
                    "feature_norm": float(feature_norm.detach().cpu()),
                    "residual_feature_relative_norm": float(
                        (residual_norm / (feature_norm + 1e-6)).detach().cpu()
                    ),
                }
        output = feature + scaled_residual
        return output.flatten(2).transpose(1, 2).contiguous()


class FinalTopologyRepairAttention(nn.Module):
    def __init__(
        self,
        channels,
        window_size=8,
        tau=4.0,
        eta_max=0.05,
        eta_init=0.005,
    ):
        super().__init__()
        self.channels = channels
        self.window_size = int(window_size)
        self.tau = float(tau)
        self.eta_max = float(eta_max)
        eta_ratio = float(eta_init) / self.eta_max
        if not 0.0 <= eta_ratio < 1.0:
            raise ValueError("eta_init must be in [0, eta_max)")
        self.register_buffer("fixed_eta", torch.tensor(float(eta_init)))
        raw_eta_init = (
            torch.tensor(0.0)
            if eta_ratio == 0.0
            else torch.logit(torch.tensor(eta_ratio))
        )
        self.raw_eta = nn.Parameter(
            raw_eta_init,
            requires_grad=eta_init > 0.0,
        )
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.output_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        relative_size = (2 * self.window_size - 1) ** 2
        self.relative_position_bias = nn.Parameter(torch.zeros(relative_size))
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        self._register_topology_buffers()
        self.capture_diagnostics = False
        self.last_diagnostics = None

    def _register_topology_buffers(self):
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(self.window_size),
                torch.arange(self.window_size),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        delta = coords.unsqueeze(0) - coords.unsqueeze(1)
        dy = delta[..., 0]
        dx = delta[..., 1]

        direction = torch.zeros_like(dy, dtype=torch.long)
        direction[(dy > 0) & (dx == 0)] = 1
        direction[(dy == 0) & (dx < 0)] = 2
        direction[(dy == 0) & (dx > 0)] = 3
        direction[(dy < 0) & (dx < 0)] = 4
        direction[(dy < 0) & (dx > 0)] = 5
        direction[(dy > 0) & (dx < 0)] = 6
        direction[(dy > 0) & (dx > 0)] = 7
        opposite = torch.tensor([1, 0, 3, 2, 7, 6, 5, 4], dtype=torch.long)

        relative = delta.clone()
        relative[..., 0] += self.window_size - 1
        relative[..., 1] += self.window_size - 1
        relative[..., 0] *= 2 * self.window_size - 1
        relative_index = relative.sum(dim=-1)

        distance = torch.maximum(dy.abs(), dx.abs()).float()
        distance_decay = torch.exp(-distance / self.tau)
        distance_decay[distance == 0] = 0.0

        self.register_buffer(
            "direction_one_hot",
            F.one_hot(direction, num_classes=8).float(),
        )
        self.register_buffer(
            "opposite_direction_one_hot",
            F.one_hot(opposite[direction], num_classes=8).float(),
        )
        self.register_buffer("distance_decay", distance_decay)
        self.register_buffer("relative_position_index", relative_index)

    def effective_eta(self):
        if not self.raw_eta.requires_grad:
            return self.fixed_eta
        return self.eta_max * torch.sigmoid(self.raw_eta)

    def _partition(self, feature):
        batch, channels, height, width = feature.shape
        window = self.window_size
        if height % window != 0 or width % window != 0:
            raise ValueError("Final topology attention requires divisible spatial size.")
        return (
            feature.view(
                batch,
                channels,
                height // window,
                window,
                width // window,
                window,
            )
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, window * window, channels)
        )

    def _reverse(self, windows, batch, height, width):
        window = self.window_size
        channels = windows.shape[-1]
        return (
            windows.view(
                batch,
                height // window,
                width // window,
                window,
                window,
                channels,
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, height, width)
        )

    def _topology_term(self, skeleton_prob, connectivity_prob):
        skeleton = self._partition(skeleton_prob).squeeze(-1)
        connectivity = self._partition(connectivity_prob)
        conn_forward = torch.einsum(
            "bik,ijk->bij",
            connectivity,
            self.direction_one_hot,
        )
        conn_backward = torch.einsum(
            "bjk,ijk->bij",
            connectivity,
            self.opposite_direction_one_hot,
        )
        direction_consistency = 0.75 * conn_forward + 0.25 * conn_backward

        source = skeleton.unsqueeze(2)
        target = skeleton.unsqueeze(1)
        affinity = 0.5 * torch.sqrt(source * target + 1e-6)
        affinity = affinity + 0.5 * torch.maximum(source, target)
        weak_target = 1.0 - target

        conn_strength = connectivity_prob.topk(
            k=min(2, connectivity_prob.shape[1]),
            dim=1,
        ).values.mean(dim=1, keepdim=True)
        token_reliability = self._partition(
            skeleton_prob * conn_strength
        ).squeeze(-1)
        top_count = max(1, token_reliability.shape[1] // 4)
        top_mean = token_reliability.topk(
            k=top_count,
            dim=1,
        ).values.mean(dim=1, keepdim=True)
        window_reliability = 0.5 * top_mean + 0.5 * token_reliability.max(
            dim=1,
            keepdim=True,
        ).values

        topology = (
            affinity
            * direction_consistency
            * self.distance_decay.unsqueeze(0)
            * weak_target
        )
        topology = window_reliability.unsqueeze(2) * topology
        return topology, window_reliability, weak_target

    def forward(self, feature, skeleton_prob, connectivity_prob):
        batch, _, height, width = feature.shape
        qkv = self.qkv(feature)
        query, key, value = qkv.chunk(3, dim=1)
        query = self._partition(query)
        key = self._partition(key)
        value = self._partition(value)

        logits = torch.matmul(query, key.transpose(1, 2))
        logits = logits * (self.channels ** -0.5)
        relative_bias = self.relative_position_bias[
            self.relative_position_index
        ]
        logits = logits + relative_bias.unsqueeze(0)

        topology, reliability, weak_target = self._topology_term(
            skeleton_prob,
            connectivity_prob,
        )
        eta = self.effective_eta()
        topology_term = eta * topology
        base_attention = torch.softmax(logits, dim=-1)
        topology_attention = torch.softmax(logits + topology_term, dim=-1)
        delta_windows = torch.matmul(
            topology_attention - base_attention,
            value,
        )
        delta = self.output_proj(
            self._reverse(delta_windows, batch, height, width)
        )
        output = feature + delta

        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "eta_eff": float(eta.detach().cpu()),
                    "window_reliability_mean": float(
                        reliability.mean().detach().cpu()
                    ),
                    "window_reliability_max": float(
                        reliability.max().detach().cpu()
                    ),
                    "topology_mean": float(topology.mean().detach().cpu()),
                    "topology_max": float(topology.max().detach().cpu()),
                    "topology_term_mean": float(
                        topology_term.mean().detach().cpu()
                    ),
                    "topology_term_max": float(
                        topology_term.max().detach().cpu()
                    ),
                    "weak_target_mean": float(
                        weak_target.mean().detach().cpu()
                    ),
                    "delta_feature_relative_norm": float(
                        (
                            torch.linalg.vector_norm(delta)
                            / (torch.linalg.vector_norm(feature) + 1e-6)
                        ).detach().cpu()
                    ),
                }
        return output


class DirectionalValueAggregation(nn.Module):
    def __init__(
        self,
        channels,
        connectivity_channels=8,
        gamma_max=0.05,
        gamma_init=0.005,
        enabled=False,
    ):
        super().__init__()
        self.connectivity_channels = connectivity_channels
        self.enabled = bool(enabled)
        self.value_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.gamma_max = float(gamma_max)
        gamma_ratio = float(gamma_init) / self.gamma_max
        if not 0.0 < gamma_ratio < 1.0:
            raise ValueError("gamma_init must be between 0 and gamma_max")
        raw_gamma_init = torch.logit(torch.tensor(gamma_ratio))
        # Keep the state-dict key for checkpoint compatibility; this stores raw gamma.
        self.gamma = nn.Parameter(raw_gamma_init, requires_grad=self.enabled)

    @staticmethod
    def _shift_feature(x, dy, dx):
        _, _, height, width = x.shape
        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)
        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)
        padded = torch.nn.functional.pad(
            x,
            (pad_left, pad_right, pad_top, pad_bottom),
        )
        y0 = max(-dy, 0)
        x0 = max(-dx, 0)
        return padded[:, :, y0:y0 + height, x0:x0 + width]

    def effective_gamma(self):
        if not self.enabled:
            return self.gamma.new_zeros(())
        return self.gamma_max * torch.sigmoid(self.gamma)

    def forward(self, feature, connectivity_prob):
        if not self.enabled:
            return feature

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        value = self.value_proj(feature)
        propagated = torch.zeros_like(value)
        for index, (dy, dx) in enumerate(directions):
            propagated = propagated + connectivity_prob[:, index:index + 1] * (
                self._shift_feature(value, -dy, -dx)
            )
        propagated = propagated / float(self.connectivity_channels)
        return feature + self.effective_gamma() * propagated


class GlobalContextHead(nn.Module):
    """Bottleneck GAP context projector used by the 0626 stage3 structure gate."""

    def __init__(self, in_channels=768, hidden_channels=128, out_channels=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, context):
        return self.mlp(context)


STAGE3_GLOBAL_CONTEXT_CHANNELS = 32


class DecoderStructureRefinement(nn.Module):
    def __init__(
        self,
        channels,
        connectivity_channels=8,
        init_gamma1=0.0,
        init_gamma2=0.0,
        gamma_limit=None,
        context_channels=None,
        context_strength=0.03,
        enable_roadness_head=False,
    ):
        super().__init__()
        fusion_channels = max(channels // 2, 16)
        self.connectivity_channels = connectivity_channels
        self.gamma_limit = gamma_limit
        self.context_strength = float(context_strength)
        self.enable_roadness_head = bool(enable_roadness_head)

        self.structure_branch = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )
        self.skeleton_head = SkeletonSpatialHead(channels)
        self.connectivity_head = nn.Conv2d(channels, connectivity_channels, kernel_size=1)
        self.structure_gate = nn.Sequential(
            nn.Conv2d(channels + 2, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, 1, kernel_size=1),
        )
        if self.enable_roadness_head:
            self.stage_roadness_head = nn.Conv2d(channels, 1, kernel_size=1)
        else:
            self.stage_roadness_head = None
        if context_channels is not None:
            self.context_to_gate = nn.Conv2d(context_channels, 1, kernel_size=1)
            nn.init.zeros_(self.context_to_gate.weight)
            nn.init.zeros_(self.context_to_gate.bias)
        else:
            self.context_to_gate = None
        self.feature_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.raw_gamma1 = nn.Parameter(torch.tensor(float(init_gamma1)))
        self.raw_gamma2 = nn.Parameter(torch.tensor(float(init_gamma2)))
        self.capture_diagnostics = False
        self.last_diagnostics = None

    @property
    def gamma1(self):
        if self.gamma_limit is None:
            return self.raw_gamma1
        return float(self.gamma_limit) * torch.tanh(self.raw_gamma1)

    @property
    def gamma2(self):
        if self.gamma_limit is None:
            return self.raw_gamma2
        return float(self.gamma_limit) * torch.tanh(self.raw_gamma2)

    @staticmethod
    def _shift_feature(x, dy, dx):
        _, _, height, width = x.shape
        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)
        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)
        padded = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        y0 = max(-dy, 0)
        x0 = max(-dx, 0)
        return padded[:, :, y0:y0 + height, x0:x0 + width]

    def directional_propagation(self, feature, connectivity_prob):
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        propagated = torch.zeros_like(feature)
        for idx, (dy, dx) in enumerate(directions):
            shifted = self._shift_feature(feature, dy, dx)
            propagated = propagated + connectivity_prob[:, idx:idx + 1] * shifted
        return propagated / float(len(directions))

    def forward(self, x, global_context=None):
        structure_feat = self.structure_branch(x)
        skeleton_logits = self.skeleton_head(structure_feat)
        connectivity_logits = self.connectivity_head(structure_feat)

        skeleton_prob = torch.sigmoid(skeleton_logits)
        connectivity_prob = torch.sigmoid(connectivity_logits)
        topk = min(2, self.connectivity_channels)
        conn_strength = connectivity_prob.topk(k=topk, dim=1).values.mean(dim=1, keepdim=True)
        structure_gate_logits = self.structure_gate(
            torch.cat([structure_feat, skeleton_prob, conn_strength], dim=1)
        )
        if self.context_to_gate is not None and global_context is not None:
            context_bias = self.context_strength * torch.tanh(
                self.context_to_gate(global_context)
            )
            structure_gate_logits = structure_gate_logits + context_bias
        structure_gate = torch.sigmoid(structure_gate_logits)

        residual = structure_gate * self.feature_residual(x)
        gate_residual = self.gamma1 * residual
        refined = x + gate_residual
        directional = self.directional_propagation(refined, connectivity_prob)
        directional_residual = self.gamma2 * directional
        out = refined + directional_residual
        if self.capture_diagnostics:
            with torch.no_grad():
                feature_norm = torch.linalg.vector_norm(x)
                self.last_diagnostics = {
                    "gamma1": float(self.gamma1.detach().cpu()),
                    "gamma2": float(self.gamma2.detach().cpu()),
                    "gate_mean": float(structure_gate.mean().detach().cpu()),
                    "gate_max": float(structure_gate.max().detach().cpu()),
                    "gate_residual_relative_norm": float(
                        (
                            torch.linalg.vector_norm(gate_residual)
                            / (feature_norm + 1e-6)
                        ).detach().cpu()
                    ),
                    "directional_residual_relative_norm": float(
                        (
                            torch.linalg.vector_norm(directional_residual)
                            / (feature_norm + 1e-6)
                        ).detach().cpu()
                    ),
                    "total_residual_relative_norm": float(
                        (
                            torch.linalg.vector_norm(out - x)
                            / (feature_norm + 1e-6)
                        ).detach().cpu()
                    ),
                }
        roadness_logits = None
        if self.stage_roadness_head is not None:
            roadness_logits = self.stage_roadness_head(structure_feat)
        return out, skeleton_logits, connectivity_logits, structure_gate, roadness_logits


class SoftSkeletonGraphPropagation(nn.Module):
    """Dense soft 8-direction propagation on final surface features."""

    DIRECTIONS = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    def __init__(
        self,
        channels,
        lambda_init=0.05,
        lambda_max=0.20,
        edge_beta=0.7,
        near_pool=7,
    ):
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.edge_beta = float(edge_beta)
        self.near_pool = int(near_pool)
        ratio = float(lambda_init) / self.lambda_max
        if not 0.0 < ratio < 1.0:
            raise ValueError("lambda_init must be in (0, lambda_max)")
        self.raw_lambda = nn.Parameter(torch.logit(torch.tensor(ratio)))
        self.dir_convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=1, bias=False)
                for _ in range(len(self.DIRECTIONS))
            ]
        )
        for conv in self.dir_convs:
            nn.init.zeros_(conv.weight)
        self.eval_lambda_scale = 1.0
        self.eval_lambda_override = None
        self.capture_diagnostics = False
        self.last_diagnostics = None
        self.last_export = None

    def effective_lambda(self):
        return self.lambda_max * torch.sigmoid(self.raw_lambda)

    def set_eval_mode(self, use_soft_graph=True, lambda_scale=1.0):
        self.eval_lambda_scale = float(lambda_scale)
        return use_soft_graph

    def set_eval_lambda(self, lambda_value=None, lambda_scale=1.0):
        if lambda_value is None:
            self.eval_lambda_override = None
            self.eval_lambda_scale = float(lambda_scale)
        else:
            self.eval_lambda_override = float(lambda_value)
            self.eval_lambda_scale = 1.0

    def reset_dir_convs_to_identity(self):
        """Eval-only passthrough: use neighbor features without trained 1x1 convs."""
        for conv in self.dir_convs:
            with torch.no_grad():
                conv.weight.zero_()
                channels = min(conv.weight.shape[0], conv.weight.shape[1])
                for idx in range(channels):
                    conv.weight[idx, idx, 0, 0] = 1.0

    @staticmethod
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

    @staticmethod
    def build_anchor_map(surface_pre_logits, skeleton_logits, connectivity_logits):
        P = torch.sigmoid(surface_pre_logits).detach()
        S = torch.sigmoid(skeleton_logits).detach()
        C = torch.sigmoid(connectivity_logits).detach()
        topk = min(2, C.shape[1])
        conn_strength = C.topk(k=topk, dim=1).values.mean(dim=1, keepdim=True)
        topo_conf = S * conn_strength
        surface_anchor = torch.sigmoid((P - 0.60) / 0.08)
        H = 1.0 - (1.0 - topo_conf) * (1.0 - surface_anchor)
        return H, P

    @staticmethod
    def build_gap_map(P, H, near_pool=7):
        pad = near_pool // 2
        near_H = F.max_pool2d(
            H,
            kernel_size=near_pool,
            stride=1,
            padding=pad,
        )
        weak_surface = torch.sigmoid((P - 0.02) / 0.03) * torch.sigmoid(
            (0.85 - P) / 0.12
        )
        return near_H * weak_surface * (1.0 - H)

    def forward(
        self,
        feature,
        surface_pre_logits,
        skeleton_logits,
        connectivity_logits,
    ):
        H, P = self.build_anchor_map(
            surface_pre_logits,
            skeleton_logits,
            connectivity_logits,
        )
        G = self.build_gap_map(P, H, near_pool=self.near_pool)
        C = torch.sigmoid(connectivity_logits).detach()

        beta = self.edge_beta
        weighted_messages = feature.new_zeros(feature.shape)
        weights_sum = feature.new_zeros(feature.shape[0], 1, feature.shape[2], feature.shape[3])

        for d_idx, (dy, dx) in enumerate(self.DIRECTIONS):
            feature_neighbor = self._shift(feature, dy, dx)
            anchor_neighbor = self._shift(H, dy, dx)
            conn_neighbor = self._shift(C[:, d_idx:d_idx + 1], dy, dx)
            edge = beta * conn_neighbor + (1.0 - beta) * anchor_neighbor
            weight = anchor_neighbor * edge
            weighted_messages = weighted_messages + weight * self.dir_convs[d_idx](
                feature_neighbor
            )
            weights_sum = weights_sum + weight

        message = weighted_messages / (weights_sum + 1e-6)
        if self.eval_lambda_override is not None:
            lam = feature.new_tensor(float(self.eval_lambda_override))
        else:
            lam = self.effective_lambda() * float(self.eval_lambda_scale)
        residual = lam * G * message
        feature_out = feature + residual

        if self.capture_diagnostics:
            with torch.no_grad():
                feature_norm = torch.linalg.vector_norm(feature)
                message_norm = torch.linalg.vector_norm(message)
                residual_norm = torch.linalg.vector_norm(residual)
                post_graph_logits = surface_pre_logits  # filled by head when available
                self.last_diagnostics = {
                    "lambda_eff": float(lam.detach().cpu()),
                    "lambda_scale": float(self.eval_lambda_scale),
                    "message_over_feature_norm": float(
                        (message_norm / (feature_norm + 1e-6)).detach().cpu()
                    ),
                    "residual_over_feature_norm": float(
                        (residual_norm / (feature_norm + 1e-6)).detach().cpu()
                    ),
                    "G_mean": float(G.mean().detach().cpu()),
                    "G_max": float(G.max().detach().cpu()),
                    "G_gt_0_1_ratio": float(
                        (G > 0.1).float().mean().detach().cpu()
                    ),
                    "H_mean": float(H.mean().detach().cpu()),
                }
                self.last_export = {
                    "G": G.detach(),
                    "H": H.detach(),
                    "P": P.detach(),
                    "message": message.detach(),
                    "feature": feature.detach(),
                    "feature_out": feature_out.detach(),
                    "surface_pre_logits": surface_pre_logits.detach(),
                    "residual": residual.detach(),
                }

        return feature_out


class SkeletonGuidedHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        init_alpha=0.0,
        connectivity_channels=8,
        topology_eta_init=0.005,
        gap_rho_init=0.005,
        enable_final_structure=True,
        enable_graph_prop=False,
        graph_prop_lambda_init=0.05,
        graph_prop_lambda_max=0.20,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, 32)

        fusion_channels = max(hidden_channels // 2, 16)
        self.connectivity_channels = connectivity_channels
        self.enable_final_structure = bool(enable_final_structure)
        self.enable_graph_prop = bool(enable_graph_prop)

        self.shared_proj = ConvBNReLU(in_channels, hidden_channels, kernel_size=3, padding=1)

        self.surface_branch = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        self.structure_branch = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )
        self.skeleton_head = SkeletonSpatialHead(hidden_channels)
        self.connectivity_head = nn.Conv2d(
            hidden_channels,
            connectivity_channels,
            kernel_size=1,
        )
        self.final_topology_attention = FinalTopologyRepairAttention(
            channels=hidden_channels,
            window_size=8,
            tau=4.0,
            eta_max=0.05,
            eta_init=topology_eta_init,
        )
        self.structure_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels + 2, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.structure_residual = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.surface_refine = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        self.boundary_branch = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )
        self.boundary_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.boundary_residual = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )

        self.surface_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        if self.enable_graph_prop:
            self.graph_propagation = SoftSkeletonGraphPropagation(
                channels=hidden_channels,
                lambda_init=graph_prop_lambda_init,
                lambda_max=graph_prop_lambda_max,
            )
        else:
            self.graph_propagation = None
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.rho_gap_max = 0.05
        rho_ratio = float(gap_rho_init) / self.rho_gap_max
        if not 0.0 <= rho_ratio < 1.0:
            raise ValueError("gap_rho_init must be in [0, 0.05)")
        self.register_buffer("fixed_rho_gap", torch.tensor(float(gap_rho_init)))
        raw_rho_init = (
            torch.tensor(0.0)
            if rho_ratio == 0.0
            else torch.logit(torch.tensor(rho_ratio))
        )
        self.raw_rho_gap = nn.Parameter(
            raw_rho_init,
            requires_grad=gap_rho_init > 0.0,
        )
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.use_legacy_surface_residual = False
        self.capture_surface_diagnostics = False
        self.last_surface_diagnostics = None
        self.eval_use_soft_graph = True
        self.capture_graph_diagnostics = False
        self.last_graph_diagnostics = None

        self._init_weights()
        self.alpha.requires_grad_(False)

    def effective_rho_gap(self):
        if not self.raw_rho_gap.requires_grad:
            return self.fixed_rho_gap
        return self.rho_gap_max * torch.sigmoid(self.raw_rho_gap)

    @staticmethod
    def build_gap_mask(surface_pre_logits, skeleton_prob, connectivity_prob):
        surface_prob = torch.sigmoid(surface_pre_logits).detach()
        skeleton = skeleton_prob.detach()
        connectivity = connectivity_prob.detach()
        conn_strength = connectivity.topk(
            k=min(2, connectivity.shape[1]),
            dim=1,
        ).values.mean(dim=1, keepdim=True)
        topo_conf = skeleton * conn_strength
        near_structure = F.max_pool2d(
            topo_conf,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        weak_surface = torch.sigmoid(
            (surface_prob - 0.05) / 0.03
        ) * torch.sigmoid(
            (0.65 - surface_prob) / 0.08
        )
        return (
            near_structure
            * (1.0 - topo_conf)
            * weak_surface
        ).detach()

    def forward(
        self,
        x,
        stage_skeleton_logits=None,
        stage_connectivity_logits=None,
    ):
        feat = self.shared_proj(x)

        surface_feat = self.surface_branch(feat)

        if not self.enable_final_structure:
            guided_surface_feat = self.surface_refine(surface_feat)

            use_graph = (
                self.enable_graph_prop
                and self.graph_propagation is not None
                and self.eval_use_soft_graph
                and stage_skeleton_logits is not None
                and stage_connectivity_logits is not None
            )
            if use_graph:
                surface_pre_logits = self.surface_head(guided_surface_feat)
                if self.capture_graph_diagnostics:
                    self.graph_propagation.capture_diagnostics = True
                guided_surface_feat = self.graph_propagation(
                    guided_surface_feat,
                    surface_pre_logits,
                    stage_skeleton_logits,
                    stage_connectivity_logits,
                )
                if self.capture_graph_diagnostics:
                    self.graph_propagation.capture_diagnostics = False
                surface_post_graph_logits = self.surface_head(guided_surface_feat)
            else:
                surface_pre_logits = None
                surface_post_graph_logits = None

            boundary_feat = self.boundary_branch(guided_surface_feat)
            boundary_logits = self.boundary_head(boundary_feat)
            boundary_attn = torch.sigmoid(boundary_logits)
            boundary_correction = self.boundary_residual(guided_surface_feat)
            guided_surface_feat = (
                guided_surface_feat
                + self.beta * boundary_attn * boundary_correction
            )
            surface_logits = self.surface_head(guided_surface_feat)

            if self.capture_graph_diagnostics and use_graph:
                with torch.no_grad():
                    graph_logit_delta = surface_post_graph_logits - surface_pre_logits
                    self.last_graph_diagnostics = {
                        "use_soft_graph": True,
                        "eval_use_soft_graph": bool(self.eval_use_soft_graph),
                        "graph_logit_delta_mean": float(
                            graph_logit_delta.mean().detach().cpu()
                        ),
                        "graph_logit_delta_abs_mean": float(
                            graph_logit_delta.abs().mean().detach().cpu()
                        ),
                        "surface_logit_delta_relative": float(
                            (
                                torch.linalg.vector_norm(graph_logit_delta)
                                / (
                                    torch.linalg.vector_norm(surface_pre_logits)
                                    + 1e-6
                                )
                            ).detach().cpu()
                        ),
                    }
                    if (
                        self.graph_propagation is not None
                        and self.graph_propagation.last_diagnostics is not None
                    ):
                        self.last_graph_diagnostics.update(
                            self.graph_propagation.last_diagnostics
                        )
                    if (
                        self.graph_propagation is not None
                        and self.graph_propagation.last_export is not None
                    ):
                        self.last_graph_diagnostics["export"] = (
                            self.graph_propagation.last_export
                        )
                        self.last_graph_diagnostics["graph_logit_delta"] = (
                            graph_logit_delta.detach()
                        )
                        self.last_graph_diagnostics["surface_pre_logits"] = (
                            surface_pre_logits.detach()
                        )
                        self.last_graph_diagnostics["surface_post_graph_logits"] = (
                            surface_post_graph_logits.detach()
                        )

            return surface_logits, boundary_logits, None, None

        structure_feat = self.structure_branch(feat)

        # First predict a topology seed, then use it to refine only the
        # structure feature. Surface features never receive this attention
        # residual directly.
        seed_skeleton_logits = self.skeleton_head(structure_feat)
        seed_connectivity_logits = self.connectivity_head(structure_feat)
        structure_feat = self.final_topology_attention(
            structure_feat,
            torch.sigmoid(seed_skeleton_logits).detach(),
            torch.sigmoid(seed_connectivity_logits).detach(),
        )
        skeleton_logits = self.skeleton_head(structure_feat)
        connectivity_logits = self.connectivity_head(structure_feat)
        skeleton_prob = torch.sigmoid(skeleton_logits)
        connectivity_prob = torch.sigmoid(connectivity_logits)

        if self.use_legacy_surface_residual:
            topk = min(2, self.connectivity_channels)
            conn_strength = connectivity_prob.topk(
                k=topk,
                dim=1,
            ).values.mean(dim=1, keepdim=True)
            structure_attn = self.structure_fusion(
                torch.cat(
                    [structure_feat, skeleton_prob, conn_strength],
                    dim=1,
                )
            )
            structure_residual = self.structure_residual(
                torch.cat([surface_feat, structure_attn], dim=1)
            )
            guided_surface_feat = surface_feat + self.alpha * structure_residual
            guided_surface_feat = self.surface_refine(guided_surface_feat)
        else:
            base_surface_feat = self.surface_refine(surface_feat)
            surface_pre_logits = self.surface_head(base_surface_feat)
            gap_mask = self.build_gap_mask(
                surface_pre_logits,
                skeleton_prob,
                connectivity_prob,
            )
            topk = min(2, self.connectivity_channels)
            conn_strength = connectivity_prob.topk(
                k=topk,
                dim=1,
            ).values.mean(dim=1, keepdim=True)
            structure_attn = self.structure_fusion(
                torch.cat(
                    [structure_feat, skeleton_prob, conn_strength],
                    dim=1,
                )
            )
            structure_residual = self.structure_residual(
                torch.cat([base_surface_feat, structure_attn], dim=1)
            )
            gap_residual = (
                self.effective_rho_gap()
                * gap_mask
                * structure_residual
            )
            guided_surface_feat = base_surface_feat + gap_residual

        boundary_feat = self.boundary_branch(guided_surface_feat)
        boundary_logits = self.boundary_head(boundary_feat)
        boundary_attn = torch.sigmoid(boundary_logits)
        boundary_correction = self.boundary_residual(guided_surface_feat)
        boundary_residual = self.beta * boundary_attn * boundary_correction
        if self.capture_surface_diagnostics:
            with torch.no_grad():
                self.last_surface_diagnostics = {
                    "boundary_beta": float(self.beta.detach().cpu()),
                    "boundary_attention_mean": float(
                        boundary_attn.mean().detach().cpu()
                    ),
                    "boundary_attention_max": float(
                        boundary_attn.max().detach().cpu()
                    ),
                    "boundary_residual_relative_norm": float(
                        (
                            torch.linalg.vector_norm(boundary_residual)
                            / (
                                torch.linalg.vector_norm(guided_surface_feat)
                                + 1e-6
                            )
                        ).detach().cpu()
                    ),
                }
                if not self.use_legacy_surface_residual:
                    self.last_surface_diagnostics.update(
                        {
                            "rho_gap_eff": float(
                                self.effective_rho_gap().detach().cpu()
                            ),
                            "gap_mask_mean": float(
                                gap_mask.mean().detach().cpu()
                            ),
                            "gap_mask_max": float(
                                gap_mask.max().detach().cpu()
                            ),
                            "gap_mask_gt_0_1_ratio": float(
                                (gap_mask > 0.1)
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                            ),
                            "gap_mask_gt_0_3_ratio": float(
                                (gap_mask > 0.3)
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                            ),
                            "gap_residual_relative_norm": float(
                                (
                                    torch.linalg.vector_norm(gap_residual)
                                    / (
                                        torch.linalg.vector_norm(
                                            base_surface_feat
                                        )
                                        + 1e-6
                                    )
                                ).detach().cpu()
                            ),
                        }
                    )
        guided_surface_feat = guided_surface_feat + boundary_residual

        surface_logits = self.surface_head(guided_surface_feat)

        return (
            surface_logits,
            boundary_logits,
            skeleton_logits,
            connectivity_logits,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
