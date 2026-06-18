import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterDilatedConvBlock(nn.Module):
    """D-LinkNet style center block with cascaded dilated convolutions."""

    def __init__(self, channels, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=True)
            for d in dilations
        ])

    def forward(self, x):
        out = x
        fused = x
        for conv in self.convs:
            out = F.relu(conv(out), inplace=True)
            fused = fused + out
        return fused


class AttentionalFeatureFusion(nn.Module):
    """Two-input attentional feature fusion adapted from AFF."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 32)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        weight = self.sigmoid(self.local_att(xa) + self.global_att(xa))
        return 2.0 * x * weight + 2.0 * residual * (1.0 - weight)


class GlobalLocalContextFusion(nn.Module):
    """Fuse Swin bottleneck tokens with local dilated-conv context."""

    def __init__(self, channels, input_resolution, reduction=4):
        super().__init__()
        self.channels = channels
        self.input_resolution = input_resolution
        self.local_context = CenterDilatedConvBlock(channels)
        self.fusion = AttentionalFeatureFusion(channels, reduction=reduction)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "bottleneck token length does not match spatial resolution"
        assert C == self.channels, "bottleneck channel count does not match fusion module"

        global_map = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        local_map = self.local_context(global_map)
        fused = self.fusion(local_map, global_map)
        fused = self.out_proj(fused)
        return fused.permute(0, 2, 3, 1).contiguous().view(B, L, C)
