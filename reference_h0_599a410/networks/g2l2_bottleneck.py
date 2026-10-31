import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LN(nn.Module):
    def __init__(self, c_channel, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln = nn.LayerNorm(c_channel)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        _, _, channels_per_head, _ = q.shape
        channels_per_head = max(channels_per_head, 1)

        mask1 = torch.zeros(b, self.num_heads, channels_per_head, channels_per_head, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, channels_per_head, channels_per_head, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, channels_per_head, channels_per_head, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, channels_per_head, channels_per_head, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=max(int(channels_per_head / 2), 1), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=max(int(channels_per_head * 2 / 3), 1), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=max(int(channels_per_head * 3 / 4), 1), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=max(int(channels_per_head * 4 / 5), 1), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = attn1 @ v
        out2 = attn2 @ v
        out3 = attn3 @ v
        out4 = attn4 @ v

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class LID2Conv(nn.Module):
    def __init__(self, dim, num_param, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class StripDynamicConv2d(nn.Module):
    def __init__(self, dim, num_param, padding=0, stride=1, reduction_ratio=4, dilation=1, num_groups=4, bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.K = num_param
        self.stride = (num_param, 1)
        self.dilation = dilation
        self.num_groups = num_groups
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, num_param, 1), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(num_param, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        scale = self.proj(self.pool(x)).reshape(b, self.num_groups, c, self.K, 1)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, 1)
        if self.dilation == 1:
            padding = (self.K // 2, 0)
        else:
            padding = (self.dilation, 0)
        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(b, self.num_groups, c), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None
        x = F.conv2d(x.reshape(1, -1, h, w), weight=weight, bias=bias, stride=self.stride, padding=padding, dilation=self.dilation, groups=b * c)
        return x.reshape(b, c, -1, w)


class SpatialMix2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        return self.net(x)


class G2L2Attention(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.norm1 = LN(dim)
        self.norm2 = LN(dim)
        self.norm3 = LN(dim)
        self.conv_weights = nn.Sequential(Attention(dim, 8))
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.self_attention = SpatialMix2D(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.conv2_l = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv1 = LID2Conv(dim, dim, 3)
        self.norm2_l = LN(dim)

    def forward(self, x):
        x_ln = self.norm1(x)
        _ = self.conv1(x)

        x = x.permute(0, 2, 3, 1)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2)
        z = z.permute(0, 3, 1, 2)
        z = self.conv_weights(z)

        x_m = self.conv2(x)
        x_m = self.self_attention(x_m)
        x_m = self.norm2(x_m)
        x_m = x_m * F.silu(z) + x_ln

        x_l = self.conv2_l(x)
        x_l = self.conv1(x_l)
        x_l = self.norm2_l(x_l)
        x_l = x_l * F.silu(z)

        x = self.mlp(self.norm3(x_m + x_l)) + x
        return x


class G2L2Bottleneck(nn.Module):
    """Token-space adapter around G2L2Attention for the Swin bottleneck."""

    def __init__(self, channels, input_resolution, mlp_ratio=4):
        super().__init__()
        self.channels = channels
        self.input_resolution = input_resolution
        self.block = G2L2Attention(channels, mlp_ratio=mlp_ratio)

    def forward(self, x):
        batch_size, length, channels = x.shape
        height, width = self.input_resolution
        assert length == height * width, "bottleneck token length does not match spatial resolution"
        assert channels == self.channels, "bottleneck channel count does not match fusion module"

        feature_map = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
        feature_map = self.block(feature_map)
        return feature_map.permute(0, 2, 3, 1).contiguous().view(batch_size, length, channels)