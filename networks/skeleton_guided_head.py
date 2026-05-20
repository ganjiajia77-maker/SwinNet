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
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SkeletonGuidedHead(nn.Module):
    """
    中心线引导道路面分割头。

    输入：
        x: 最终 decoder feature, [B, C, H, W]

    输出：
        surface_logits:  道路面预测, [B, 1, H, W]
        skeleton_logits: 中心线预测, [B, 1, H, W]
        skeleton_attn:   中心线注意力图, [B, 1, H, W]
    """

    def __init__(self, in_channels, hidden_channels=None, init_alpha=0.0):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, 32)

        self.shared_proj = ConvBNReLU(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1
        )

        # surface 分支
        self.surface_branch = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels)
        )

        # skeleton 分支
        self.skeleton_branch = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels)
        )

        # skeleton 输出头
        self.skeleton_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1
        )

        # 由 skeleton feature 生成 attention
        self.skeleton_attention = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 引导后的 surface 特征再融合一下
        self.surface_refine = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels)
        )

        # surface 输出头
        self.surface_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1
        )

        # 中心线引导强度，初始化为 0 最稳
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        self._init_weights()

    def forward(self, x):
        feat = self.shared_proj(x)

        surface_feat = self.surface_branch(feat)
        skeleton_feat = self.skeleton_branch(feat)

        skeleton_logits = self.skeleton_head(skeleton_feat)

        skeleton_attn = self.skeleton_attention(skeleton_feat)

        # 中心线引导 surface 特征
        guided_surface_feat = surface_feat * (1.0 + self.alpha * skeleton_attn)

        guided_surface_feat = self.surface_refine(guided_surface_feat)

        surface_logits = self.surface_head(guided_surface_feat)

        return surface_logits, skeleton_logits, skeleton_attn

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
