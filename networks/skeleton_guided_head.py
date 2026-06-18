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

        self.surface_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

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
        conn_strength = connectivity_prob.topk(k=topk, dim=1).values.mean(dim=1, keepdim=True)
        structure_attn = self.structure_fusion(
            torch.cat([structure_feat, skeleton_prob, conn_strength], dim=1)
        )

        structure_residual = self.structure_residual(
            torch.cat([surface_feat, structure_attn], dim=1)
        )
        guided_surface_feat = surface_feat + self.alpha * structure_residual
        guided_surface_feat = self.surface_refine(guided_surface_feat)
        surface_logits = self.surface_head(guided_surface_feat)

        return surface_logits, skeleton_logits, connectivity_logits, structure_attn

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
