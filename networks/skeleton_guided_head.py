import torch
import torch.nn as nn


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


class DirectionalValueAggregation(nn.Module):
    def __init__(self, channels, connectivity_channels=8, gamma_max=0.05):
        super().__init__()
        self.connectivity_channels = connectivity_channels
        self.value_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        # Keep the state-dict key for checkpoint compatibility; this stores raw gamma.
        self.gamma = nn.Parameter(torch.tensor(-8.0))
        self.gamma_max = float(gamma_max)

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
        return self.gamma_max * torch.sigmoid(self.gamma)

    def forward(self, feature, connectivity_prob):
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
    ):
        super().__init__()
        fusion_channels = max(channels // 2, 16)
        self.connectivity_channels = connectivity_channels
        self.gamma_limit = gamma_limit
        self.context_strength = float(context_strength)

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
        refined = x + self.gamma1 * residual
        directional = self.directional_propagation(refined, connectivity_prob)
        out = refined + self.gamma2 * directional
        return out, skeleton_logits, connectivity_logits, structure_gate


class SkeletonGuidedHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        init_alpha=0.0,
        connectivity_channels=8,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, 32)

        fusion_channels = max(hidden_channels // 2, 16)
        self.connectivity_channels = connectivity_channels

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
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.beta = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def forward(self, x):
        feat = self.shared_proj(x)

        surface_feat = self.surface_branch(feat)
        structure_feat = self.structure_branch(feat)
        skeleton_logits = self.skeleton_head(structure_feat)
        connectivity_logits = self.connectivity_head(structure_feat)

        skeleton_prob = torch.sigmoid(skeleton_logits)
        connectivity_prob = torch.sigmoid(connectivity_logits)
        topk = min(2, self.connectivity_channels)
        conn_strength = connectivity_prob.topk(k=topk, dim=1).values.mean(
            dim=1,
            keepdim=True,
        )
        structure_attn = self.structure_fusion(
            torch.cat([structure_feat, skeleton_prob, conn_strength], dim=1)
        )
        structure_residual = self.structure_residual(
            torch.cat([surface_feat, structure_attn], dim=1)
        )
        guided_surface_feat = surface_feat + self.alpha * structure_residual
        guided_surface_feat = self.surface_refine(guided_surface_feat)

        boundary_feat = self.boundary_branch(guided_surface_feat)
        boundary_logits = self.boundary_head(boundary_feat)
        boundary_attn = torch.sigmoid(boundary_logits)
        boundary_correction = self.boundary_residual(guided_surface_feat)
        guided_surface_feat = guided_surface_feat + self.beta * boundary_attn * boundary_correction

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
