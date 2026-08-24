import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_gradient(x, ratio: float):
    return x.detach() + float(ratio) * (x - x.detach())


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


class ConnectivityContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dilate1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dilate2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        context = torch.cat([self.dilate1(x), self.dilate2(x)], dim=1)
        return x + self.fuse(context)


CONNECTIVITY_DIRECTIONS = (
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
)


class PairwiseConnectivityHead(nn.Module):
    def __init__(self, channels, connectivity_channels=8, hidden_channels=None):
        super().__init__()
        if connectivity_channels != len(CONNECTIVITY_DIRECTIONS):
            raise ValueError("PairwiseConnectivityHead expects 8 connectivity channels.")
        hidden_channels = hidden_channels or max(channels // 2, 16)
        self.connectivity_channels = connectivity_channels
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(2 * channels + 3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        basis = []
        for dy, dx in CONNECTIVITY_DIRECTIONS:
            theta = torch.atan2(torch.tensor(float(dy)), torch.tensor(float(dx)))
            basis.append([torch.cos(2.0 * theta), torch.sin(2.0 * theta)])
        self.register_buffer("axis_basis", torch.tensor(basis).float().view(1, 8, 2, 1, 1))

    @staticmethod
    def _shift_feature(x, dy, dx):
        _, _, height, width = x.shape
        pad_left = max(-dx, 0)
        pad_right = max(dx, 0)
        pad_top = max(-dy, 0)
        pad_bottom = max(dy, 0)
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        y0 = max(dy, 0)
        x0 = max(dx, 0)
        return padded[:, :, y0:y0 + height, x0:x0 + width]

    def direction_alignment(self, direction_logits):
        direction = F.normalize(direction_logits, dim=1, eps=1e-6)
        direction = direction.unsqueeze(1)
        return ((direction * self.axis_basis).sum(dim=2) + 1.0) * 0.5

    def forward(self, feature, direction_alignment=None, skeleton_prob=None):
        if direction_alignment is None:
            direction_alignment = feature.new_zeros(
                feature.shape[0],
                self.connectivity_channels,
                feature.shape[-2],
                feature.shape[-1],
            )
        if skeleton_prob is None:
            skeleton_prob = feature.new_zeros(
                feature.shape[0],
                1,
                feature.shape[-2],
                feature.shape[-1],
            )
        logits = []
        for idx, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            neighbor = self._shift_feature(feature, dy, dx)
            neighbor_skeleton = self._shift_feature(skeleton_prob, dy, dx)
            edge_feature = torch.cat(
                [
                    feature,
                    neighbor,
                    skeleton_prob,
                    neighbor_skeleton,
                    direction_alignment[:, idx:idx + 1],
                ],
                dim=1,
            )
            logits.append(self.edge_mlp(edge_feature))
        return torch.cat(logits, dim=1)


class StageTopologyPredictor(nn.Module):
    def __init__(self, channels, connectivity_channels=8):
        super().__init__()
        self.structure_branch = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )
        self.skeleton_head = SkeletonSpatialHead(channels)
        self.connectivity_context = ConnectivityContextBlock(channels)
        self.connectivity_head = PairwiseConnectivityHead(channels, connectivity_channels)

    def forward(self, x):
        structure_feat = self.structure_branch(x)
        skeleton_logits = self.skeleton_head(structure_feat)
        skeleton_prob = torch.sigmoid(skeleton_logits).detach()
        connectivity_feat = self.connectivity_context(structure_feat)
        return (
            skeleton_logits,
            self.connectivity_head(
                connectivity_feat,
                skeleton_prob=skeleton_prob,
            ),
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

        target_dy = -dy
        target_dx = -dx
        direction = torch.zeros_like(dy, dtype=torch.long)
        direction[(target_dy < 0) & (target_dx == 0)] = 0
        direction[(target_dy < 0) & (target_dx > 0)] = 1
        direction[(target_dy == 0) & (target_dx > 0)] = 2
        direction[(target_dy > 0) & (target_dx > 0)] = 3
        direction[(target_dy > 0) & (target_dx == 0)] = 4
        direction[(target_dy > 0) & (target_dx < 0)] = 5
        direction[(target_dy == 0) & (target_dx < 0)] = 6
        direction[(target_dy < 0) & (target_dx < 0)] = 7
        opposite = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], dtype=torch.long)

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
        pad_left = max(-dx, 0)
        pad_right = max(dx, 0)
        pad_top = max(-dy, 0)
        pad_bottom = max(dy, 0)
        padded = torch.nn.functional.pad(
            x,
            (pad_left, pad_right, pad_top, pad_bottom),
        )
        y0 = max(dy, 0)
        x0 = max(dx, 0)
        return padded[:, :, y0:y0 + height, x0:x0 + width]

    def effective_gamma(self):
        if not self.enabled:
            return self.gamma.new_zeros(())
        return self.gamma_max * torch.sigmoid(self.gamma)

    def forward(self, feature, connectivity_prob):
        if not self.enabled:
            return feature

        value = self.value_proj(feature)
        propagated = torch.zeros_like(value)
        for index, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            propagated = propagated + connectivity_prob[:, index:index + 1] * (
                self._shift_feature(value, dy, dx)
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


class PostRefineStructureInteraction(nn.Module):
    def __init__(self, surface_channels, structure_channels, structure_hidden=16):
        super().__init__()
        self.structure_proj = nn.Conv2d(
            structure_channels,
            structure_hidden,
            kernel_size=1,
            bias=False,
        )
        self.delta = nn.Sequential(
            nn.Conv2d(
                surface_channels + structure_hidden,
                surface_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(surface_channels),
            nn.GELU(),
            nn.Conv2d(surface_channels, surface_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def reset_output(self):
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, surface_feat, z_struct):
        if z_struct is None:
            return surface_feat
        z_struct = z_struct.detach()
        struct_feat = self.structure_proj(z_struct)
        struct_feat = F.interpolate(
            struct_feat,
            size=surface_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        delta = self.delta(torch.cat([surface_feat, struct_feat], dim=1))
        return surface_feat + delta


class DecoderStructureRefinement(nn.Module):
    def __init__(
        self,
        channels,
        connectivity_channels=8,
        init_gamma1=0.0,
        gamma_limit=None,
        context_channels=None,
        context_strength=0.03,
        enable_direct_feature_refinement=True,
        skeleton_gradient_ratio=0.5,
    ):
        super().__init__()
        fusion_channels = max(channels // 2, 16)
        self.connectivity_channels = connectivity_channels
        self.gamma_limit = gamma_limit
        self.context_strength = float(context_strength)
        self.enable_direct_feature_refinement = bool(enable_direct_feature_refinement)
        self.skeleton_gradient_ratio = float(skeleton_gradient_ratio)

        self.structure_branch = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )
        self.gate_branch = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )
        self.skeleton_head = SkeletonSpatialHead(channels)
        self.connectivity_context = ConnectivityContextBlock(channels)
        self.connectivity_head = PairwiseConnectivityHead(channels, connectivity_channels)
        self.direction_head = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        self.structure_gate = nn.Sequential(
            nn.Conv2d(channels + 2, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, 1, kernel_size=1),
        )
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
        self.capture_diagnostics = False
        self.last_diagnostics = None

    @property
    def gamma1(self):
        if self.gamma_limit is None:
            return self.raw_gamma1
        return float(self.gamma_limit) * torch.tanh(self.raw_gamma1)

    @staticmethod
    def _shift_feature(x, dy, dx):
        _, _, height, width = x.shape
        pad_left = max(-dx, 0)
        pad_right = max(dx, 0)
        pad_top = max(-dy, 0)
        pad_bottom = max(dy, 0)
        padded = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        y0 = max(dy, 0)
        x0 = max(dx, 0)
        return padded[:, :, y0:y0 + height, x0:x0 + width]

    def directional_propagation(self, feature, connectivity_prob):
        propagated = torch.zeros_like(feature)
        for idx, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            shifted = self._shift_feature(feature, dy, dx)
            propagated = propagated + connectivity_prob[:, idx:idx + 1] * shifted
        return propagated / float(len(CONNECTIVITY_DIRECTIONS))

    def forward(
        self,
        x,
        global_context=None,
        apply_feature_refinement=True,
        disable_skeleton_prediction=False,
    ):
        structure_input = scale_gradient(x, self.skeleton_gradient_ratio)
        structure_feat = self.structure_branch(structure_input)
        if disable_skeleton_prediction:
            skeleton_logits = None
            skeleton_prob = x.new_zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]))
        else:
            skeleton_logits = self.skeleton_head(structure_feat)
            skeleton_prob = torch.sigmoid(skeleton_logits)
        direction_logits = self.direction_head(structure_feat)
        direction_alignment = self.connectivity_head.direction_alignment(
            direction_logits
        ).detach()
        connectivity_feat = self.connectivity_context(structure_feat)
        connectivity_logits = self.connectivity_head(
            connectivity_feat,
            direction_alignment,
            skeleton_prob=skeleton_prob.detach(),
        )

        connectivity_prob = torch.sigmoid(connectivity_logits)
        topk = min(2, self.connectivity_channels)
        conn_strength = connectivity_prob.topk(k=topk, dim=1).values.mean(dim=1, keepdim=True)
        gate_feat = self.gate_branch(x)
        structure_gate_logits = self.structure_gate(
            torch.cat(
                [
                    gate_feat,
                    skeleton_prob.detach(),
                    conn_strength.detach(),
                ],
                dim=1,
            )
        )
        if self.context_to_gate is not None and global_context is not None:
            context_bias = self.context_strength * torch.tanh(
                self.context_to_gate(global_context)
            )
            structure_gate_logits = structure_gate_logits + context_bias
        structure_gate = torch.sigmoid(structure_gate_logits)

        if self.enable_direct_feature_refinement and apply_feature_refinement:
            residual = structure_gate * self.feature_residual(x)
            gate_residual = self.gamma1 * residual
            out = x + gate_residual
        else:
            gate_residual = torch.zeros_like(x)
            out = x
        if self.capture_diagnostics:
            with torch.no_grad():
                feature_norm = torch.linalg.vector_norm(x)
                self.last_diagnostics = {
                    "gamma1": float(self.gamma1.detach().cpu()),
                    "gate_mean": float(structure_gate.mean().detach().cpu()),
                    "gate_max": float(structure_gate.max().detach().cpu()),
                    "conn_strength_mean": float(
                        conn_strength.mean().detach().cpu()
                    ),
                    "gate_residual_relative_norm": float(
                        (
                            torch.linalg.vector_norm(gate_residual)
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
        return (
            out,
            skeleton_logits,
            connectivity_logits,
            direction_logits,
            structure_gate,
            None,
        )


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
        final_skeleton_gradient_ratio=0.0,
        enable_post_refine_structure_interaction=False,
        highres_structure_channels=64,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, 32)

        fusion_channels = max(hidden_channels // 2, 16)
        self.connectivity_channels = connectivity_channels
        self.enable_final_structure = bool(enable_final_structure)
        self.final_skeleton_gradient_ratio = float(final_skeleton_gradient_ratio)
        self.enable_post_refine_structure_interaction = bool(
            enable_post_refine_structure_interaction
        )

        self.surface_proj = ConvBNReLU(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.skeleton_proj = ConvBNReLU(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.surface_branch = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        self.structure_branch = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )
        self.skeleton_head = SkeletonSpatialHead(hidden_channels)
        self.detached_skeleton_refine = nn.Sequential(
            ConvBNReLU(hidden_channels + 1, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )
        self.detached_skeleton_head = SkeletonSpatialHead(hidden_channels)
        self.connectivity_context = ConnectivityContextBlock(hidden_channels)
        self.connectivity_head = PairwiseConnectivityHead(hidden_channels, connectivity_channels)
        self.structure_to_surface = nn.Sequential(
            ConvBNReLU(hidden_channels + connectivity_channels + 3, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.structure_to_surface_gamma = nn.Parameter(torch.tensor(0.0))
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
        self.post_refine_structure_interaction = PostRefineStructureInteraction(
            hidden_channels,
            highres_structure_channels,
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
        self.last_surface_pre_logits = None
        self.last_graph_delta = None
        self.last_delta_logit = None

        self._init_weights()
        self.post_refine_structure_interaction.reset_output()
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

    def _apply_post_refine_structure_interaction(self, guided_surface_feat, z_struct):
        if not self.enable_post_refine_structure_interaction:
            return guided_surface_feat
        return self.post_refine_structure_interaction(guided_surface_feat, z_struct)

    def forward(self, x, z_struct=None):
        surface_feat = self.surface_branch(self.surface_proj(x))
        skeleton_feat = self.skeleton_proj(x)

        if not self.enable_final_structure:
            final_structure_feat = self.structure_branch(skeleton_feat)
            final_skeleton_logits = self.skeleton_head(final_structure_feat)
            guided_surface_feat = self.surface_refine(surface_feat)
            guided_surface_feat = self._apply_post_refine_structure_interaction(
                guided_surface_feat,
                z_struct,
            )

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
            guided_surface_feat = (
                guided_surface_feat
                + boundary_residual
            )
            surface_pre_logits = self.surface_head(guided_surface_feat)
            self.last_surface_pre_logits = surface_pre_logits
            self.last_graph_delta = None
            self.last_delta_logit = None

            surface_logits = surface_pre_logits
            self.last_final_direction_logits = None
            return surface_logits, boundary_logits, final_skeleton_logits, None

        structure_feat = self.structure_branch(skeleton_feat)

        # First predict a topology seed, then use it to refine only the
        # structure feature. Surface features never receive this attention
        # residual directly.
        seed_skeleton_logits = self.skeleton_head(structure_feat)
        seed_skeleton_prob = torch.sigmoid(seed_skeleton_logits).detach()
        seed_connectivity_feat = self.connectivity_context(structure_feat)
        seed_connectivity_logits = self.connectivity_head(
            seed_connectivity_feat,
            skeleton_prob=seed_skeleton_prob,
        )
        structure_feat = self.final_topology_attention(
            structure_feat,
            seed_skeleton_prob,
            torch.sigmoid(seed_connectivity_logits).detach(),
        )
        skeleton_logits = self.skeleton_head(structure_feat)
        skeleton_prob = torch.sigmoid(skeleton_logits)
        connectivity_feat = self.connectivity_context(structure_feat)
        connectivity_logits = self.connectivity_head(
            connectivity_feat,
            skeleton_prob=skeleton_prob.detach(),
        )
        connectivity_prob = torch.sigmoid(connectivity_logits)

        guided_surface_feat = self.surface_refine(surface_feat)
        guided_surface_feat = self._apply_post_refine_structure_interaction(
            guided_surface_feat,
            z_struct,
        )

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
