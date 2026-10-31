import torch
import torch.nn as nn


class RoadAttentionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden_channels = max(in_channels // 2, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        attention = self.conv(x)
        return torch.sigmoid(attention)
