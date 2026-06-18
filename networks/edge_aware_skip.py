import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDepthwiseConv2d(nn.Conv2d):
    def __init__(self, channels, mask, bias=True):
        super().__init__(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=bias,
        )
        self.register_buffer("mask", mask.view(1, 1, 3, 3))

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EdgeAwareSkipEnhance(nn.Module):
    def __init__(self, channels, init_alpha=0.1):
        super().__init__()
        self.diag_45 = MaskedDepthwiseConv2d(
            channels,
            torch.tensor(
                [[0.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0]]
            ),
            bias=True,
        )
        self.diag_135 = MaskedDepthwiseConv2d(
            channels,
            torch.tensor(
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]
            ),
            bias=True,
        )
        self.horizontal = nn.Conv2d(
            channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels, bias=True
        )
        self.vertical = nn.Conv2d(
            channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels, bias=True
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        diag_45 = F.relu(self.diag_45(x), inplace=True)
        diag_135 = F.relu(self.diag_135(x), inplace=True)
        diagonal = 0.5 * (diag_45 + diag_135)
        horizontal = F.relu(self.horizontal(x), inplace=True)
        vertical = F.relu(self.vertical(x), inplace=True)
        edge_attn = self.fuse(torch.cat([diagonal, horizontal, vertical], dim=1))
        return x * (1.0 + self.alpha * edge_attn)
