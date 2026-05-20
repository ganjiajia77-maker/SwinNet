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
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DeformableCrossAttention2D(nn.Module):
    """
    Pure PyTorch lightweight deformable cross-attention.

    deep feature:
        used to generate query, offsets and sampling weights.

    shallow feature:
        used as value feature to be sampled.

    Input:
        deep_feat:    [B, Cd, Hd, Wd]
        shallow_feat: [B, Cs, Hs, Ws]

    Output:
        guided_feat:  [B, C_out, Hs, Ws]
    """

    def __init__(
        self,
        deep_channels: int,
        shallow_channels: int,
        out_channels: int,
        num_heads: int = 4,
        num_points: int = 4,
        max_offset: float = 0.20
    ):
        super().__init__()

        assert out_channels % num_heads == 0, (
            f"out_channels={out_channels} must be divisible by num_heads={num_heads}"
        )

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = out_channels // num_heads
        self.max_offset = max_offset

        # Use deep feature as query source.
        self.query_proj = ConvBNReLU(
            deep_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )

        # Use shallow feature as value source.
        self.value_proj = ConvBNReLU(
            shallow_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )

        # Predict offsets from deep query.
        # Output: [B, num_heads * num_points * 2, H, W]
        self.offset_conv = nn.Conv2d(
            out_channels,
            num_heads * num_points * 2,
            kernel_size=3,
            padding=1
        )

        # Predict attention weights from deep query.
        # Output: [B, num_heads * num_points, H, W]
        self.attn_conv = nn.Conv2d(
            out_channels,
            num_heads * num_points,
            kernel_size=3,
            padding=1
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def _make_base_grid(self, batch_size, height, width, device, dtype):
        """
        Create normalized base grid for grid_sample.
        grid shape: [B, H, W, 2]
        value range: [-1, 1]
        """
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij"
        )

        grid = torch.stack([x, y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return grid

    def forward(self, deep_feat, shallow_feat):
        B, _, Hs, Ws = shallow_feat.shape

        # Align deep feature to shallow feature size.
        deep_up = F.interpolate(
            deep_feat,
            size=(Hs, Ws),
            mode="bilinear",
            align_corners=False
        )

        query = self.query_proj(deep_up)        # [B, C, Hs, Ws]
        value = self.value_proj(shallow_feat)   # [B, C, Hs, Ws]

        # Predict offsets.
        offsets = self.offset_conv(query)
        offsets = torch.tanh(offsets) * self.max_offset

        # [B, heads, points, 2, H, W] -> [B, heads, points, H, W, 2]
        offsets = offsets.view(
            B,
            self.num_heads,
            self.num_points,
            2,
            Hs,
            Ws
        ).permute(0, 1, 2, 4, 5, 3).contiguous()

        # Predict attention weights.
        attn = self.attn_conv(query)
        attn = attn.view(B, self.num_heads, self.num_points, Hs, Ws)
        attn = torch.softmax(attn, dim=2)

        # Split value by heads.
        value = value.view(B, self.num_heads, self.head_dim, Hs, Ws)

        base_grid = self._make_base_grid(
            batch_size=B,
            height=Hs,
            width=Ws,
            device=shallow_feat.device,
            dtype=shallow_feat.dtype
        )

        sampled_heads = []

        for h in range(self.num_heads):
            head_sum = 0.0

            value_h = value[:, h, :, :, :]  # [B, head_dim, H, W]

            for p in range(self.num_points):
                grid_hp = base_grid + offsets[:, h, p, :, :, :]  # [B, H, W, 2]

                sampled = F.grid_sample(
                    value_h,
                    grid_hp,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True
                )  # [B, head_dim, H, W]

                weight = attn[:, h, p, :, :].unsqueeze(1)  # [B, 1, H, W]
                head_sum = head_sum + sampled * weight

            sampled_heads.append(head_sum)

        guided_feat = torch.cat(sampled_heads, dim=1)  # [B, C_out, H, W]
        guided_feat = self.out_proj(guided_feat)

        return guided_feat

    def _init_weights(self):
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        nn.init.zeros_(self.attn_conv.weight)
        nn.init.zeros_(self.attn_conv.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m not in [self.offset_conv, self.attn_conv]:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class DCAFPNBlock(nn.Module):
    """
    DCA-FPN block.

    It refines shallow skip feature using deep semantic feature.

    Input:
        deep_feat:    [B, Cd, Hd, Wd]
        shallow_feat: [B, Cs, Hs, Ws]

    Output:
        refined_skip: [B, C_out, Hs, Ws]
    """

    def __init__(
        self,
        deep_channels: int,
        shallow_channels: int,
        out_channels: int,
        num_heads: int = 4,
        num_points: int = 4,
        max_offset: float = 0.20,
        init_alpha: float = 0.1
    ):
        super().__init__()

        self.deep_proj = ConvBNReLU(
            deep_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )

        self.shallow_proj = ConvBNReLU(
            shallow_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )

        self.dca = DeformableCrossAttention2D(
            deep_channels=deep_channels,
            shallow_channels=shallow_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            num_points=num_points,
            max_offset=max_offset
        )

        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels * 3, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Learnable residual strength.
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, deep_feat, shallow_feat):
        Hs, Ws = shallow_feat.shape[-2:]

        deep_up = F.interpolate(
            deep_feat,
            size=(Hs, Ws),
            mode="bilinear",
            align_corners=False
        )

        deep_p = self.deep_proj(deep_up)
        shallow_p = self.shallow_proj(shallow_feat)

        guided = self.dca(deep_feat, shallow_feat)

        fused = self.fuse(torch.cat([deep_p, shallow_p, guided], dim=1))

        # Keep shallow detail as residual base.
        refined_skip = shallow_p + self.alpha * fused

        return refined_skip


if __name__ == "__main__":
    # Example with Swin-like channel setting.
    deep = torch.randn(2, 768, 14, 14)
    shallow = torch.randn(2, 384, 28, 28)

    block = DCAFPNBlock(
        deep_channels=768,
        shallow_channels=384,
        out_channels=384,
        num_heads=4,
        num_points=4,
        max_offset=0.20
    )

    out = block(deep, shallow)
    print("deep:", deep.shape)
    print("shallow:", shallow.shape)
    print("out:", out.shape)
