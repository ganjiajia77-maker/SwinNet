import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .dca_fpn_lite import DCAFPNLite
from .bottleneck_context_fusion import GlobalLocalContextFusion
from .g2l2_bottleneck import G2L2Bottleneck
from .keypoint_global_topology import KeypointGuidedGlobalTopology
from .road_attention_head import RoadAttentionHead
from .psi_directional import PSI_DIRECTIONS, psi_directional_descriptor
from losses.road_losses import build_connectivity_target
from .skeleton_guided_head import (
    DecoderStructureRefinement,
    GlobalContextHead,
    STAGE3_GLOBAL_CONTEXT_CHANNELS,
    SkeletonGuidedHead,
)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_nhwc_to_window(x, window_size):
    """Pad an NHWC feature map on its bottom/right edges to full windows."""
    H, W = x.shape[1:3]
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
    return x, H + pad_h, W + pad_w


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))) )
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid(relative_coords_h, relative_coords_w, indexing="ij"), dim=-1).unsqueeze(0)
        relative_coords_table[:, :, :, 0] /= max(window_size[0] - 1, 1)
        relative_coords_table[:, :, :, 1] /= max(window_size[1] - 1, 1)
        relative_coords_table = relative_coords_table * 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x,
        mask=None,
        road_attention_bias=None,
        structure_attention_bias=None,
    ):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        qk_logits = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        qk_logits = qk_logits * torch.clamp(self.logit_scale, max=torch.log(torch.tensor(100.0, device=x.device))).exp()
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias.permute(2, 0, 1).contiguous())
        attn = qk_logits + relative_position_bias.unsqueeze(0)
        if road_attention_bias is not None:
            if road_attention_bias.dim() == 3:
                road_attention_bias = road_attention_bias.unsqueeze(1)
            attn = attn + road_attention_bias
        if structure_attention_bias is not None:
            if structure_attention_bias.dim() == 3:
                structure_attention_bias = structure_attention_bias.unsqueeze(1)
            attn = attn + structure_attention_bias

        if mask is not None:
            if mask.shape[0] == B_:
                # Sparse routing may select a different number of windows per
                # image. In that case each selected window already carries its
                # own shifted-window mask.
                attn = attn + mask.unsqueeze(1)
            else:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


# ============== Token ↔ Feature Map 转换函数 ==============
def token_to_map(x, H, W):
    """
    将 token 格式转换为特征图格式
    
    Args:
        x: [B, L, C] token 格式
        H, W: 特征图的空间分辨率
    
    Returns:
        [B, C, H, W] 特征图格式
    """
    B, L, C = x.shape
    assert L == H * W, f"Token数量 {L} 不匹配 H×W={H}×{W}"
    x = x.view(B, H, W, C)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


def map_to_token(x):
    """
    将特征图格式转换为 token 格式
    
    Args:
        x: [B, C, H, W] 特征图格式
    
    Returns:
        [B, L, C] token 格式
    """
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(B, H * W, C)
    return x


def _largest_group_divisor(channels, candidates=(8, 4, 2, 1)):
    for groups in candidates:
        if channels % groups == 0:
            return groups
    return 1


class PrePatchStructureEncoder(nn.Module):
    def __init__(self, struct_channels):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(
                num_groups=_largest_group_divisor(16),
                num_channels=16,
            ),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_largest_group_divisor(32),
                num_channels=32,
            ),
            nn.GELU(),
        )
        self.project = nn.Conv2d(32, struct_channels, kernel_size=1, bias=False)
        bottleneck_channels = max(struct_channels // 4, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(struct_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_largest_group_divisor(bottleneck_channels),
                num_channels=bottleneck_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=1,
                groups=bottleneck_channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_largest_group_divisor(bottleneck_channels),
                num_channels=bottleneck_channels,
            ),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, struct_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_largest_group_divisor(struct_channels),
                num_channels=struct_channels,
            ),
        )
        self.act = nn.GELU()
        self._shape_logged = False
        self._init_weights()

    def forward(self, x, return_e128=False):
        input_shape = tuple(x.shape)
        e128 = self.down1(x)
        down1_shape = tuple(e128.shape)
        z_struct = self.down2(e128)
        down2_shape = tuple(z_struct.shape)
        z_struct = self.project(z_struct)
        project_shape = tuple(z_struct.shape)
        z_struct = self.act(self.refine(z_struct) + z_struct)
        if not self._shape_logged:
            print(
                "[PrePatch Lite Structure] input={} E128={} after_down2={} projected={} z_struct={}".format(
                    input_shape,
                    down1_shape,
                    down2_shape,
                    project_shape,
                    tuple(z_struct.shape),
                ),
                flush=True,
            )
            self._shape_logged = True
        return (e128, z_struct) if return_e128 else z_struct

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class CoarseRoadMaskHead(nn.Module):
    """Semantic plus optional PSI fusion head for the 64x64 coarse road map."""

    def __init__(self, semantic_channels, psi_channels=12, hidden_channels=32):
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.psi_proj = nn.Sequential(
            nn.Conv2d(psi_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.psi_reliability_gate = nn.Sequential(
            nn.Conv2d(hidden_channels + 16, 16, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels + 16, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.psi_reliability_gate[0].weight)
        nn.init.zeros_(self.psi_reliability_gate[0].bias)

    def forward(self, feature32, psi_descriptor=None, output_size=(64, 64)):
        semantic = self.semantic_proj(feature32)
        semantic = F.interpolate(
            semantic, size=output_size, mode="bilinear", align_corners=False
        )
        if psi_descriptor is None:
            psi_projected = semantic.new_zeros(
                semantic.shape[0], 16, *semantic.shape[-2:]
            )
            psi_gate = semantic.new_zeros(
                semantic.shape[0], 1, *semantic.shape[-2:]
            )
        else:
            psi_projected = self.psi_proj(psi_descriptor)
            psi_gate = self.psi_reliability_gate(
                torch.cat([semantic, psi_projected], dim=1)
            )
        fused = self.fuse(torch.cat([semantic, psi_gate * psi_projected], dim=1))
        return self.out(fused), psi_gate


class SparseStructureFallbackHead(nn.Module):
    """Small spatial fallback used where P64 does not select a window.

    The fallback deliberately contains no attention, MLP, or topology
    propagation. Its feature path starts as an identity so inactive locations
    remain useful to the following decoder stage.
    """

    def __init__(self, channels, connectivity_channels=8):
        super().__init__()
        self.feature_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.skeleton_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.connectivity_head = nn.Conv2d(channels, connectivity_channels, kernel_size=1)
        self.direction_head = nn.Conv2d(channels, 2, kernel_size=1)
        self.structure_gate = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.feature_projection.weight)

    def forward(self, feature_map):
        fallback_feature = feature_map + self.feature_projection(feature_map)
        connectivity_logits = self.connectivity_head(fallback_feature)
        connectivity_prob = torch.sigmoid(connectivity_logits)
        return {
            "feature": fallback_feature,
            "skeleton": self.skeleton_head(fallback_feature),
            "connectivity": connectivity_logits,
            "direction": self.direction_head(fallback_feature),
            "structure_gate": torch.sigmoid(self.structure_gate(fallback_feature)),
            "roadness": {
                "structure_feat": fallback_feature,
                "conn_strength": connectivity_prob.topk(
                    k=min(2, connectivity_prob.shape[1]), dim=1
                ).values.mean(dim=1, keepdim=True),
                "connectivity_prob": connectivity_prob,
            },
        }


class BottleneckCoarseRoadMaskHead(nn.Module):
    """8x8 semantic+PSI coarse road head used for decoder routing."""

    def __init__(self, semantic_channels, psi_channels=12, hidden_channels=32):
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.psi_proj = nn.Sequential(
            nn.Conv2d(psi_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(hidden_channels + 16, 16, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels + 16, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.reliability_gate[0].weight)
        nn.init.zeros_(self.reliability_gate[0].bias)

    def forward(self, bottleneck_feature, psi_descriptor=None):
        semantic = self.semantic_proj(bottleneck_feature)
        if psi_descriptor is None:
            psi_projected = semantic.new_zeros(
                semantic.shape[0], 16, *semantic.shape[-2:]
            )
            psi_gate = semantic.new_zeros(
                semantic.shape[0], 1, *semantic.shape[-2:]
            )
        else:
            psi_projected = self.psi_proj(psi_descriptor)
            psi_gate = self.reliability_gate(
                torch.cat([semantic, psi_projected], dim=1)
            )
        fused = self.fuse(torch.cat([semantic, psi_gate * psi_projected], dim=1))
        return self.out(fused), psi_gate


class HighResStructureFusion(nn.Module):
    def __init__(self, feature_channels, struct_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(struct_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=True),
        )
        self._init_weights()
        nn.init.constant_(self.delta[-1].weight, 0)
        nn.init.constant_(self.delta[-1].bias, 0)

    def forward(self, feature_map, z_struct):
        z = F.interpolate(
            z_struct,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z = self.project(z)
        return feature_map + self.delta(torch.cat([feature_map, z], dim=1))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class StructureSurfaceCorrectionHead(nn.Module):
    def __init__(self, struct_channels, hidden_channels=32):
        super().__init__()
        mid_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(struct_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_largest_group_divisor(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_largest_group_divisor(mid_channels), mid_channels),
            nn.GELU(),
        )
        self.out = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
        self._init_weights()
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_struct, target_hw):
        x = self.conv1(z_struct)
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.conv2(x)
        return self.out(x)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_road_bias=False,
                 use_decoder_structure_bias=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_road_bias = bool(use_road_bias)
        self.use_decoder_structure_bias = bool(use_decoder_structure_bias)
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if self.use_road_bias:
            self.road_bias_scale_a1 = nn.Parameter(torch.tensor(0.0))
            self.road_bias_scale_a2 = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("road_bias_scale_a1", None)
            self.register_parameter("road_bias_scale_a2", None)
        if self.use_decoder_structure_bias:
            self.decoder_connectivity_bias_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("decoder_connectivity_bias_scale", None)
        self._register_structure_bias_buffers()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Input sizes at successive Swin stages are not necessarily divisible by
        # window=7. Build the shift mask after runtime padding in forward().
        self.register_buffer("attn_mask", None)

    def _get_attention_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in h_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    @staticmethod
    def _pad_probability_map(x, H, W, Hp, Wp):
        if x is None:
            return None
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        if (Hp, Wp) != (H, W):
            x = F.pad(x, (0, Wp - W, 0, Hp - H))
        return x

    def _pad_road_prior(self, road_prior, H, W, Hp, Wp):
        if road_prior is None:
            return None
        if isinstance(road_prior, (tuple, list)):
            return tuple(self._pad_road_prior(item, H, W, Hp, Wp) for item in road_prior)
        return self._pad_probability_map(road_prior, H, W, Hp, Wp)

    def _register_structure_bias_buffers(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).view(-1, 2)
        delta = coords.unsqueeze(0) - coords.unsqueeze(1)
        dy, dx = delta[..., 0], delta[..., 1]
        target_dy, target_dx = -dy, -dx
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
        distance = torch.maximum(dy.abs(), dx.abs()).float()
        pair_axis = torch.stack(
            (torch.cos(2 * torch.atan2(target_dy.float(), target_dx.float())),
             torch.sin(2 * torch.atan2(target_dy.float(), target_dx.float()))), dim=-1
        )
        pair_axis[distance == 0] = 0
        self.register_buffer("topology_direction_one_hot", F.one_hot(direction, 8).float(), persistent=False)
        self.register_buffer("topology_opposite_direction_one_hot", F.one_hot(opposite[direction], 8).float(), persistent=False)
        self.register_buffer("topology_pair_distance", distance, persistent=False)
        self.register_buffer("topology_pair_unit_i_to_j", pair_axis, persistent=False)
        self.register_buffer("topology_pair_unit_j_to_i", pair_axis, persistent=False)

    def _road_prior_to_pair_bias(self, road_prior):
        H, W = road_prior.shape[-2:]
        if road_prior.dim() == 3:
            road_prior = road_prior.unsqueeze(1)
        if road_prior.shape[-2:] != (H, W):
            road_prior = F.interpolate(
                road_prior,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        road_prior = road_prior.permute(0, 2, 3, 1).contiguous()
        if self.shift_size > 0:
            road_prior = torch.roll(
                road_prior,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        road_windows = window_partition(
            road_prior,
            self.window_size,
        ).view(-1, self.window_size * self.window_size, 1)
        road_windows = road_windows.clamp(0.0, 1.0)
        return road_windows * road_windows.transpose(1, 2)

    def _build_road_attention_bias(self, road_prior):
        if road_prior is None or not self.use_road_bias:
            return None
        if isinstance(road_prior, (tuple, list)):
            road_priors = list(road_prior)
        else:
            road_priors = [road_prior]

        bias = None
        scales = (self.road_bias_scale_a1, self.road_bias_scale_a2)
        for prior, scale in zip(road_priors, scales):
            if prior is None:
                continue
            pair_bias = scale * self._road_prior_to_pair_bias(prior)
            bias = pair_bias if bias is None else bias + pair_bias
        return bias

    @staticmethod
    def _row_normalize_attention_graph(graph):
        return graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _build_decoder_structure_attention_bias(
        self,
        skeleton_prob,
        connectivity_prob,
        direction_prob=None,
    ):
        if connectivity_prob is None or not self.use_decoder_structure_bias:
            return None

        connectivity = connectivity_prob.detach().permute(0, 2, 3, 1).contiguous()
        direction = None
        if direction_prob is not None:
            direction = F.normalize(direction_prob.detach(), dim=1, eps=1e-6)
            direction = direction.permute(0, 2, 3, 1).contiguous()
        if self.shift_size > 0:
            shifts = (-self.shift_size, -self.shift_size)
            connectivity = torch.roll(connectivity, shifts=shifts, dims=(1, 2))
            if direction is not None:
                direction = torch.roll(direction, shifts=shifts, dims=(1, 2))

        connectivity_windows = window_partition(
            connectivity,
            self.window_size,
        ).view(-1, self.window_size * self.window_size, 8)
        if direction is not None:
            direction_windows = window_partition(
                direction,
                self.window_size,
            ).view(-1, self.window_size * self.window_size, 2)
            direction_forward = torch.einsum(
                "bic,ijc->bij",
                direction_windows,
                self.topology_pair_unit_i_to_j,
            ).abs()
            direction_backward = torch.einsum(
                "bjc,ijc->bij",
                direction_windows,
                self.topology_pair_unit_j_to_i,
            ).abs()
            direction_alignment = direction_forward * direction_backward
        else:
            direction_alignment = 1.0

        conn_forward = torch.einsum(
            "bik,ijk->bij",
            connectivity_windows,
            self.topology_direction_one_hot,
        )
        conn_backward = torch.einsum(
            "bjk,ijk->bij",
            connectivity_windows,
            self.topology_opposite_direction_one_hot,
        )
        one_hop_mask = (self.topology_pair_distance == 1).to(
            dtype=connectivity_windows.dtype
        )
        adjacency = (
            0.5
            * (conn_forward + conn_backward)
            * one_hop_mask.unsqueeze(0)
        )
        adjacency = self._row_normalize_attention_graph(adjacency.clamp_min(0.0))
        direction_soft_gate = 0.5 + 0.5 * direction_alignment
        directional_adjacency = (
            0.5
            * (conn_forward + conn_backward)
            * direction_soft_gate
            * one_hop_mask.unsqueeze(0)
        )
        directional_adjacency = self._row_normalize_attention_graph(
            directional_adjacency.clamp_min(0.0)
        )
        directional_adjacency_2 = self._row_normalize_attention_graph(
            torch.bmm(directional_adjacency, directional_adjacency)
        )
        directional_adjacency_3 = self._row_normalize_attention_graph(
            torch.bmm(directional_adjacency_2, directional_adjacency)
        )

        distance = self.topology_pair_distance.to(dtype=connectivity_windows.dtype)
        distance_decay = 1.0 / (1.0 + 0.2 * distance)
        connectivity_bias = (
            adjacency
            + 0.5 * directional_adjacency_2
            + 0.25 * directional_adjacency_3
        )
        connectivity_bias = connectivity_bias * distance_decay.unsqueeze(0)
        connectivity_bias = connectivity_bias.masked_fill(
            self.topology_pair_distance.unsqueeze(0) == 0,
            0.0,
        )
        connectivity_bias = self._row_normalize_attention_graph(
            connectivity_bias.clamp_min(0.0)
        )
        return self.decoder_connectivity_bias_scale * connectivity_bias

    def forward(
        self,
        x,
        road_prior=None,
        decoder_skeleton_prob=None,
        decoder_connectivity_prob=None,
        decoder_direction_prob=None,
    ):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        x, Hp, Wp = pad_nhwc_to_window(x, self.window_size)
        attention_mask = self._get_attention_mask(Hp, Wp, x.device)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        road_attention_bias = self._build_road_attention_bias(
            self._pad_road_prior(road_prior, H, W, Hp, Wp)
        )
        structure_attention_bias = self._build_decoder_structure_attention_bias(
            self._pad_probability_map(decoder_skeleton_prob, H, W, Hp, Wp),
            self._pad_probability_map(decoder_connectivity_prob, H, W, Hp, Wp),
            self._pad_probability_map(decoder_direction_prob, H, W, Hp, Wp),
        )
        attn_windows = self.attn(
            x_windows,
            mask=attention_mask,
            road_attention_bias=road_attention_bias,
            structure_attention_bias=structure_attention_bias,
        )
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :H, :W, :].contiguous().view(B, H * W, C)

        # FFN
        # Swin-V2 post-norm residual formulation.
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def forward_sparse_windows(
        self,
        x,
        candidate_probability,
        threshold=0.10,
        decoder_skeleton_prob=None,
        decoder_connectivity_prob=None,
        decoder_direction_prob=None,
    ):
        """Run the complete Swin block only for selected real Swin windows.

        The P64 probability map is first aligned to this block resolution and
        then reduced with the block's actual ``self.window_size``. Selected
        windows are gathered into one batch for attention and MLP, then
        scattered back once. Unselected windows remain exact identity paths.
        """
        height, width = self.input_resolution
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("Sparse window input length does not match resolution.")
        if candidate_probability is None:
            return self(
                x,
                decoder_skeleton_prob=decoder_skeleton_prob,
                decoder_connectivity_prob=decoder_connectivity_prob,
                decoder_direction_prob=decoder_direction_prob,
            ), {"active_windows": 0, "total_windows": 0, "active_token_mask": None}

        if candidate_probability.shape[-2:] != (height, width):
            candidate_probability = F.adaptive_max_pool2d(
                candidate_probability.float(), (height, width)
            )
        candidate = candidate_probability.float()
        window_size = int(self.window_size)
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        feature = x.view(batch, height, width, channels)
        feature, padded_h, padded_w = pad_nhwc_to_window(feature, self.window_size)
        candidate_map = F.pad(candidate, (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        if self.shift_size > 0:
            shifted_feature = torch.roll(
                feature,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
            shifted_candidate = torch.roll(
                candidate_map,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_feature = feature
            shifted_candidate = candidate_map

        windows_per_image = (padded_h // self.window_size) * (
            padded_w // self.window_size
        )
        feature_windows = window_partition(
            shifted_feature, self.window_size
        ).view(-1, self.window_size * self.window_size, channels)
        active_windows = window_partition(
            shifted_candidate, self.window_size
        ).view(-1, self.window_size * self.window_size).amax(dim=1) >= float(threshold)
        active_count = int(active_windows.sum().item())
        total_count = int(active_windows.numel())
        if active_count == 0:
            return x, {
                "active_windows": 0,
                "total_windows": total_count,
                "active_ratio": 0.0,
                "active_token_mask": x.new_zeros(batch, length, 1, dtype=torch.bool),
            }
        if active_count == total_count:
            return self(
                x,
                decoder_skeleton_prob=decoder_skeleton_prob,
                decoder_connectivity_prob=decoder_connectivity_prob,
                decoder_direction_prob=decoder_direction_prob,
            ), {
                "active_windows": total_count,
                "total_windows": total_count,
                "active_ratio": 1.0,
                "active_token_mask": x.new_ones(batch, length, 1, dtype=torch.bool),
            }

        selected_indices = active_windows.nonzero(as_tuple=False).flatten()
        selected_window_ids = selected_indices.remainder(windows_per_image)
        attention_mask = self._get_attention_mask(
            padded_h, padded_w, x.device
        )
        if attention_mask is not None:
            selected_attention_mask = attention_mask.index_select(
                0, selected_window_ids
            )
        else:
            selected_attention_mask = None

        structure_attention_bias = self._build_decoder_structure_attention_bias(
            self._pad_probability_map(decoder_skeleton_prob, height, width, padded_h, padded_w),
            self._pad_probability_map(decoder_connectivity_prob, height, width, padded_h, padded_w),
            self._pad_probability_map(decoder_direction_prob, height, width, padded_h, padded_w),
        )
        if structure_attention_bias is not None:
            structure_attention_bias = structure_attention_bias.index_select(
                0, selected_indices
            )

        selected_attention = self.attn(
            feature_windows.index_select(0, selected_indices),
            mask=selected_attention_mask,
            structure_attention_bias=structure_attention_bias,
        )

        selected_feature_windows = feature_windows.index_select(0, selected_indices)
        selected_feature_windows = selected_feature_windows + self.drop_path(
            self.norm1(selected_attention.to(dtype=selected_feature_windows.dtype))
        )
        selected_feature_windows = selected_feature_windows + self.drop_path(
            self.norm2(self.mlp(selected_feature_windows))
        )
        # Under BF16 autocast, LayerNorm/MLP can promote the gathered path to
        # FP32 while the full feature tensor remains BF16.  The scatter target
        # must have the same dtype as its source; cast only at the boundary so
        # the sparse computation keeps the autocast precision internally.
        selected_feature_windows = selected_feature_windows.to(
            dtype=feature_windows.dtype
        )

        # Scatter the updated windows into the original shifted window tensor;
        # no full-image norm/MLP or zero-filled intermediate is needed.
        updated_windows = feature_windows.clone()
        updated_windows.index_copy_(
            0,
            selected_indices,
            selected_feature_windows.to(dtype=feature_windows.dtype),
        )
        shifted_x = window_reverse(
            updated_windows.view(-1, self.window_size, self.window_size, channels),
            self.window_size,
            padded_h,
            padded_w,
        )

        selected_window_grid = active_windows.view(
            batch, padded_h // self.window_size, padded_w // self.window_size
        )
        selected_window_map = selected_window_grid.repeat_interleave(
            self.window_size, dim=1
        ).repeat_interleave(self.window_size, dim=2).unsqueeze(-1)
        if self.shift_size > 0:
            shifted_x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
            selected_window_map = torch.roll(
                selected_window_map,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        x = shifted_x[:, :height, :width, :].contiguous().view(
            batch, length, channels
        )
        selected_tokens = selected_window_map[:, :height, :width, :].contiguous().view(
            batch, length, 1
        )
        return x, {
            "active_windows": active_count,
            "total_windows": total_count,
            "active_ratio": (
                float(active_count) / float(total_count)
                if total_count > 0 else 0.0
            ),
            "active_token_mask": selected_tokens,
        }

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        input_resolution,
        dim,
        norm_layer=nn.LayerNorm,
        road_alpha_init=0.1,
        road_attention_merge_mode="residual",
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.linear_reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.mlp_reduction = nn.Sequential(
            nn.Linear(4 * dim, 2 * dim, bias=False),
            nn.GELU(),
            nn.Linear(2 * dim, 2 * dim, bias=False),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.road_alpha = nn.Parameter(torch.tensor(float(road_alpha_init)))
        self.road_attention_merge_mode = str(road_attention_merge_mode).lower()
        if self.road_attention_merge_mode not in {"residual", "merge_attention"}:
            raise ValueError(
                "road_attention_merge_mode must be residual or merge_attention"
            )
        self.norm = norm_layer(4 * dim)

    def compute_merge_score(self, feature, road_prior):
        feature_norm = F.normalize(feature, dim=-1, eps=1e-6)
        feature_score = torch.matmul(
            feature_norm,
            feature_norm.transpose(-1, -2),
        )
        road_score = road_prior * road_prior.transpose(-1, -2)
        return feature_score + self.road_alpha * road_score

    def _apply_road_merge_attention(self, patch_tokens, patch_road_prior):
        merge_score = self.compute_merge_score(patch_tokens, patch_road_prior)
        merge_weight = torch.softmax(merge_score, dim=-1)
        return torch.matmul(merge_weight, patch_tokens)

    def forward(self, x, road_attention=None):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        if road_attention is not None:
            if road_attention.dim() == 3:
                road_attention = road_attention.unsqueeze(1)
            assert road_attention.shape[-2:] == (H, W), (
                "road attention size must match patch merge input resolution"
            )
            road_attention = road_attention.permute(0, 2, 3, 1).contiguous()
            if self.road_attention_merge_mode == "residual":
                x = x * (1.0 + self.road_alpha * road_attention)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        if road_attention is not None and self.road_attention_merge_mode == "merge_attention":
            a0 = road_attention[:, 0::2, 0::2, :]
            a1 = road_attention[:, 1::2, 0::2, :]
            a2 = road_attention[:, 0::2, 1::2, :]
            a3 = road_attention[:, 1::2, 1::2, :]
            patch_tokens = torch.stack([x0, x1, x2, x3], dim=-2)
            patch_road_prior = torch.stack([a0, a1, a2, a3], dim=-2)
            patch_tokens = self._apply_road_merge_attention(
                patch_tokens,
                patch_road_prior,
            )
            x0, x1, x2, x3 = patch_tokens.unbind(dim=-2)
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x_linear = self.linear_reduction(x)
        x_mlp = self.mlp_reduction(x)
        x = x_linear + self.alpha * x_mlp

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += (H // 2) * (W // 2) * (4 * self.dim * 2 * self.dim + 2 * self.dim * 2 * self.dim)
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 road_attention_head=None, road_alpha_init=0.1, use_road_bias=False,
                 road_attention_modulates_downsample=True,
                 road_attention_merge_mode="residual"):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.road_attention_head = road_attention_head
        self.last_road_attention = None
        self.road_alpha_init = float(road_alpha_init)
        self.use_road_bias = bool(use_road_bias)
        self.road_attention_modulates_downsample = bool(road_attention_modulates_downsample)
        self.road_attention_merge_mode = str(road_attention_merge_mode).lower()
        self.last_pre_downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_road_bias=self.use_road_bias)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                road_alpha_init=self.road_alpha_init,
                road_attention_merge_mode=self.road_attention_merge_mode,
            )
        else:
            self.downsample = None

    def forward(self, x, road_prior=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, None, None, None, road_prior)
            else:
                x = blk(x, road_prior=road_prior)
        self.last_road_attention = None
        self.last_pre_downsample = x
        if self.downsample is not None:
            road_attention = None
            if self.road_attention_head is not None:
                H, W = self.input_resolution
                x_map = token_to_map(x, H, W)
                road_attention = self.road_attention_head(x_map)
                self.last_road_attention = road_attention
            x = self.downsample(
                x,
                road_attention=(
                    road_attention
                    if self.road_attention_modulates_downsample
                    else None
                ),
            )
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 use_decoder_structure_bias=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_decoder_structure_bias = bool(use_decoder_structure_bias)
        self.window_size = int(window_size)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_decoder_structure_bias=self.use_decoder_structure_bias)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None
        self.last_sparse_stats = {"active_windows": 0, "total_windows": 0}
        self.last_active_token_mask = None

    def forward(
        self,
        x,
        decoder_skeleton_prob=None,
        decoder_connectivity_prob=None,
        decoder_direction_prob=None,
        sparse_probability_map=None,
        sparse_window_compute=False,
        sparse_threshold=0.25,
    ):
        active_windows = 0
        total_windows = 0
        active_token_mask = None
        for blk in self.blocks:
            if sparse_window_compute and sparse_probability_map is not None:
                x, block_stats = blk.forward_sparse_windows(
                    x,
                    sparse_probability_map,
                    threshold=sparse_threshold,
                    decoder_skeleton_prob=decoder_skeleton_prob,
                    decoder_connectivity_prob=decoder_connectivity_prob,
                    decoder_direction_prob=decoder_direction_prob,
                )
                active_windows += block_stats["active_windows"]
                total_windows += block_stats["total_windows"]
                block_mask = block_stats.get("active_token_mask")
                if block_mask is not None:
                    active_token_mask = (
                        block_mask if active_token_mask is None
                        else active_token_mask | block_mask
                    )
            elif self.use_checkpoint:
                x = checkpoint.checkpoint(
                    lambda feature, dec_skeleton, dec_connectivity, dec_direction: blk(
                        feature,
                        decoder_skeleton_prob=dec_skeleton,
                        decoder_connectivity_prob=dec_connectivity,
                        decoder_direction_prob=dec_direction,
                    ),
                    x,
                    decoder_skeleton_prob,
                    decoder_connectivity_prob,
                    decoder_direction_prob,
                )
            else:
                x = blk(
                    x,
                    decoder_skeleton_prob=decoder_skeleton_prob,
                    decoder_connectivity_prob=decoder_connectivity_prob,
                    decoder_direction_prob=decoder_direction_prob,
                )
            if not (sparse_window_compute and sparse_probability_map is not None):
                height, width = blk.input_resolution
                padded_h = height + (self.window_size - height % self.window_size) % self.window_size
                padded_w = width + (self.window_size - width % self.window_size) % self.window_size
                total_windows += (padded_h // self.window_size) * (padded_w // self.window_size) * x.shape[0]
                active_windows += (padded_h // self.window_size) * (padded_w // self.window_size) * x.shape[0]
                active_token_mask = x.new_ones(
                    x.shape[0], x.shape[1], 1, dtype=torch.bool
                )
        self.last_sparse_stats = {
            "active_windows": int(active_windows),
            "total_windows": int(total_windows),
            "active_ratio": (
                float(active_windows) / float(total_windows)
                if total_windows > 0 else 0.0
            ),
        }
        if self.upsample is not None:
            if active_token_mask is not None:
                mask_map = active_token_mask.view(
                    x.shape[0], self.input_resolution[0], self.input_resolution[1], 1
                ).permute(0, 3, 1, 2).float()
                mask_map = F.interpolate(mask_map, scale_factor=2, mode="nearest")
                active_token_mask = mask_map.permute(0, 2, 3, 1).reshape(
                    x.shape[0], -1, 1
                ).bool()
            x = self.upsample(x)
        self.last_active_token_mask = active_token_mask
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Preserve official Swin names so patch_embed.proj weights load directly.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.edge_stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=7, stride=patch_size, padding=3, bias=False),
        )
        self.edge_scale = nn.Parameter(torch.tensor(0.01))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) + self.edge_scale * self.edge_stem(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        H, W = self.img_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * 4 * 4
        flops += H * W * self.in_chans * 32 * 3 * 3
        flops += H * W * 32 * 32 * 3 * 3
        flops += Ho * Wo * 32 * self.embed_dim * 7 * 7
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class TopologyAttentionScale(nn.Module):
    """Nonnegative topology coefficient without an extra attention block."""

    def __init__(
        self,
        alpha_max=0.20,
        alpha_init=0.02,
        trainable=False,
    ):
        super().__init__()
        self.alpha_max = float(alpha_max)
        alpha_ratio = float(alpha_init) / self.alpha_max
        if not 0.0 <= alpha_ratio < 1.0:
            raise ValueError("alpha_init must be in [0, alpha_max)")
        self.register_buffer(
            "fixed_alpha",
            torch.tensor(float(alpha_init)),
        )
        raw_alpha_init = (
            torch.tensor(0.0)
            if alpha_ratio == 0.0
            else torch.logit(torch.tensor(alpha_ratio))
        )
        self.topology_alpha = nn.Parameter(
            raw_alpha_init,
            requires_grad=bool(trainable),
        )

    def effective_topology_alpha(self):
        if not self.topology_alpha.requires_grad:
            return self.fixed_alpha
        return self.alpha_max * torch.sigmoid(self.topology_alpha)


class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", return_skeleton=False,
                 bottleneck_type="global_local", final_topology_eta_init=0.005,
                 final_gap_rho_init=0.005,
                 structure_profile="full",
                 stage2_skeleton_gradient_ratio=0.5,
                 stage3_skeleton_gradient_ratio=0.5,
                 stage3_gate_topology_gradient_ratio=0.0,
                 final_skeleton_gradient_ratio=0.0,
                 enable_highres_structure_stream=False,
                 highres_structure_channels=64,
                 highres_structure_fuse_stages="stage23",
                 highres_structure_fusion_mode="stage23",
                 enable_post_refine_structure_interaction=False,
                 enable_h3_surface_fusion=False,
                  enable_global_topology=False,
                  global_topology_max_nodes=32,
                  global_topology_heads=4,
                  global_topology_alpha_max=0.05,
                  stage_skeleton_mode="prior_residual",
                  enable_e128_stage_fusion=False,
                  enable_coarse_road_mask=False,
                  enable_psi_directional_descriptor=False,
                  sparse_window_compute=False,
                  stage2_window_threshold=0.25,
                  stage3_window_threshold=0.25,
                  coarse_candidate_window_size=8,
                  coarse_corridor_window_radius=0,
                  coarse_routing_mode="dense",
                  bottleneck_coarse_road_mask=False,
                  bottleneck_window_threshold=0.25,
                  bottleneck_route_warmup_epochs=0,
                  bottleneck_route_warmup_mode="dense",
                  coarse_route_warmup_epochs=5,
                  routing_warmup_epochs=10,
                  stage_skeleton_bias_init="zero",
                  stage_skeleton_positive_prior=0.05,
                  remove_stage2_pre_topology_source=False,
                  **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                drop_path_rate,
                num_classes,
            )
        )

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.return_skeleton = return_skeleton
        self.bottleneck_type = bottleneck_type.lower()
        self.structure_profile = structure_profile.lower()
        if self.structure_profile not in {
            "full",
            "stage23_boundary_0626",
            "stage23_boundary_0626_final_ske",
        }:
            raise ValueError(
                "structure_profile must be one of: full, stage23_boundary_0626, "
                "stage23_boundary_0626_final_ske"
            )
        self.use_stage3_global_context = (
            self.structure_profile in {
                "stage23_boundary_0626",
                "stage23_boundary_0626_final_ske",
            }
        )
        self.stage2_skeleton_gradient_ratio = float(stage2_skeleton_gradient_ratio)
        self.stage3_skeleton_gradient_ratio = float(stage3_skeleton_gradient_ratio)
        self.stage3_gate_topology_gradient_ratio = float(stage3_gate_topology_gradient_ratio)
        self.final_skeleton_gradient_ratio = float(final_skeleton_gradient_ratio)
        self.enable_highres_structure_stream = bool(enable_highres_structure_stream)
        self.highres_structure_channels = int(highres_structure_channels)
        self.enable_global_topology = bool(enable_global_topology)
        self.stage_skeleton_mode = str(stage_skeleton_mode).lower()
        if self.stage_skeleton_mode not in {"direct", "prior_residual"}:
            raise ValueError("stage_skeleton_mode must be direct or prior_residual")
        self.enable_e128_stage_fusion = bool(enable_e128_stage_fusion)
        self.enable_coarse_road_mask = bool(enable_coarse_road_mask)
        self.enable_psi_directional_descriptor = bool(
            enable_psi_directional_descriptor
        )
        self.coarse_routing_mode = str(coarse_routing_mode).lower()
        if self.coarse_routing_mode not in {
            "dense", "p64", "bottleneck", "bottleneck_no_psi"
        }:
            raise ValueError(
                "coarse_routing_mode must be dense, p64, bottleneck, or bottleneck_no_psi"
            )
        self.bottleneck_coarse_road_mask = bool(
            bottleneck_coarse_road_mask or self.coarse_routing_mode.startswith("bottleneck")
        )
        self.sparse_window_compute = bool(sparse_window_compute)
        if (
            self.sparse_window_compute
            and not self.enable_coarse_road_mask
            and not self.bottleneck_coarse_road_mask
        ):
            raise ValueError("Sparse window compute requires the coarse road mask head.")
        self.stage2_window_threshold = float(stage2_window_threshold)
        self.stage3_window_threshold = float(stage3_window_threshold)
        self.coarse_candidate_window_size = int(coarse_candidate_window_size)
        self.coarse_corridor_window_radius = int(coarse_corridor_window_radius)
        if (
            self.coarse_routing_mode == "dense"
            and sparse_window_compute
            and enable_coarse_road_mask
        ):
            self.coarse_routing_mode = "p64"
        self.bottleneck_window_threshold = float(bottleneck_window_threshold)
        self.bottleneck_route_warmup_epochs = max(
            0, int(bottleneck_route_warmup_epochs)
        )
        self.bottleneck_route_warmup_mode = str(bottleneck_route_warmup_mode).lower()
        if self.bottleneck_route_warmup_mode != "dense":
            raise ValueError("bottleneck_route_warmup_mode must be dense")
        self.coarse_route_warmup_epochs = max(0, int(coarse_route_warmup_epochs))
        self.routing_warmup_epochs = max(0, int(routing_warmup_epochs))
        self.routing_calibration_done = False
        self.current_epoch = 0
        self.stage_skeleton_bias_init = str(stage_skeleton_bias_init).lower()
        self.stage_skeleton_positive_prior = float(stage_skeleton_positive_prior)
        if self.stage_skeleton_bias_init not in {"zero", "prior"}:
            raise ValueError("stage_skeleton_bias_init must be zero or prior")
        self.remove_stage2_pre_topology_source = bool(
            remove_stage2_pre_topology_source
        )
        self.last_coarse_road_logits = None
        self.last_psi_reliability_gate = None
        self.last_sparse_window_stats = {}
        self.sparse_selection_probability_override = None
        self.last_stage_features = {}
        self.last_route_stats = {}
        self.last_e128 = None
        self.last_z_struct = None
        self.highres_structure_fuse_stages = str(highres_structure_fuse_stages).lower()
        self._highres_structure_shape_logged = False
        self.last_highres_z_struct = None
        self.last_highres_structure_skeleton = None
        if self.highres_structure_fuse_stages not in {"stage2", "stage3", "stage23"}:
            raise ValueError(
                "highres_structure_fuse_stages must be one of: stage2, stage3, stage23"
            )
        self.highres_structure_fusion_mode = str(highres_structure_fusion_mode).lower()
        if self.highres_structure_fusion_mode not in {
            "stage23",
            "final_correction",
            "stage23_final_correction",
            "post_refine_interaction",
            "none",
        }:
            raise ValueError(
                "highres_structure_fusion_mode must be one of: "
                "stage23, final_correction, stage23_final_correction, "
                "post_refine_interaction, none"
            )
        self.enable_post_refine_structure_interaction = bool(
            enable_post_refine_structure_interaction
            or self.highres_structure_fusion_mode == "post_refine_interaction"
        )
        self.enable_h3_surface_fusion = bool(enable_h3_surface_fusion)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        self.encoder_stage1_road_attention_head = RoadAttentionHead(embed_dim)
        self.encoder_stage2_road_attention_head = RoadAttentionHead(embed_dim * 2)
        print(
            "[INFO] Road priors: A1 channels={} and A2 channels={} -> "
            "Stage3 attention bias; A1/A2 -> residual PatchMerging priors".format(
                embed_dim,
                embed_dim * 2,
            )
        )
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               road_attention_head=(
                                   self.encoder_stage1_road_attention_head
                                   if i_layer == 0
                                   else self.encoder_stage2_road_attention_head
                                   if i_layer == 1
                                   else None
                               ),
                               road_alpha_init=(
                                   0.1
                               ),
                               use_road_bias=(i_layer == 2),
                               road_attention_modulates_downsample=(i_layer in (0, 1)),
                               road_attention_merge_mode=(
                                   "residual"
                               ))
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint,
                                use_decoder_structure_bias=(
                                    i_layer in (2, 3)
                                    and not (
                                        self.remove_stage2_pre_topology_source
                                        and i_layer == 2
                                    )
                                ))
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self.bottleneck_resolution = (
            patches_resolution[0] // (2 ** (self.num_layers - 1)),
            patches_resolution[1] // (2 ** (self.num_layers - 1)),
        )
        if self.bottleneck_type in {"global_local", "legacy_global_local", "original", "default"}:
            self.bottleneck_context_fusion = GlobalLocalContextFusion(
                channels=self.num_features,
                input_resolution=self.bottleneck_resolution,
                reduction=4,
            )
            print(
                "[INFO] GlobalLocalContextFusion bottleneck: final-layer global, "
                "resolution={}, channels={}".format(
                    self.bottleneck_resolution, self.num_features
                )
            )
        elif self.bottleneck_type in {"g2l2", "g2l2attention"}:
            self.bottleneck_context_fusion = G2L2Bottleneck(
                channels=self.num_features,
                input_resolution=self.bottleneck_resolution,
                mlp_ratio=4,
            )
            print(
                "[INFO] G2L2Attention bottleneck: resolution={}, channels={}".format(
                    self.bottleneck_resolution, self.num_features
                )
            )
        else:
            raise ValueError(f"Unsupported bottleneck_type: {bottleneck_type}")

        # ===== DCA-FPN skip refinement for decoder stages 2 and 3 =====
        self.bottleneck_swin_block = SwinTransformerBlock(
            dim=self.num_features,
            input_resolution=self.bottleneck_resolution,
            num_heads=num_heads[-1],
            window_size=window_size,
            shift_size=0,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[-1] if len(dpr) > 0 else 0.0,
            norm_layer=norm_layer,
        )
        print(
            "[INFO] Bottleneck SwinTransformerBlock after fusion: resolution={}, channels={}, heads={}".format(
                self.bottleneck_resolution, self.num_features, num_heads[-1]
            )
        )

        self.dca_blocks = nn.ModuleList()
        
        for skip_idx in range(2, self.num_layers):  # inx=2,3: Layer2 和 Layer1
            # 通道数: skip_idx=2 → 384 (Layer2), skip_idx=3 → 192 (Layer1)
            skip_channels = int(embed_dim * 2 ** (self.num_layers - 1 - skip_idx))
            
            # DCA-FPN-Lite 模块（轻量级可变形交叉注意）
            dca_block = DCAFPNLite(
                channels=skip_channels,
                num_heads=4,
                num_points=4,
                max_offset=0.20
            )
            self.dca_blocks.append(dca_block)
            
            layer_name = "Layer 2" if skip_idx == 2 else "Layer 1"
            print(f"[INFO] DCA-FPN-Lite {layer_name} (inx={skip_idx}): {skip_channels} channels")

        decoder_structure_channels = {2: embed_dim, 3: embed_dim}
        if self.use_stage3_global_context:
            self.global_context_head = GlobalContextHead(
                in_channels=self.num_features,
                hidden_channels=128,
                out_channels=STAGE3_GLOBAL_CONTEXT_CHANNELS,
            )
        else:
            self.global_context_head = None

        self.decoder_structure_blocks = nn.ModuleDict(
            {
                str(stage_index):
                DecoderStructureRefinement(
                    channels=channels,
                    context_channels=(
                        STAGE3_GLOBAL_CONTEXT_CHANNELS
                        if (
                            self.use_stage3_global_context
                            and stage_index == 3
                        )
                        else None
                    ),
                    enable_direct_feature_refinement=True,
                    skeleton_gradient_ratio=(
                        self.stage2_skeleton_gradient_ratio
                        if stage_index == 2
                        else self.stage3_skeleton_gradient_ratio
                        if stage_index == 3
                        else 0.0
                    ),
                    gate_topology_gradient_ratio=(
                        self.stage3_gate_topology_gradient_ratio
                        if stage_index == 3
                        else 0.0
                    ),
                    previous_structure_channels=(
                        channels if stage_index == 3 else None
                    ),
                )
                for stage_index, channels in decoder_structure_channels.items()
            }
        )
        self.sparse_structure_fallback_heads = nn.ModuleDict(
            {
                "2": SparseStructureFallbackHead(embed_dim),
                "3": SparseStructureFallbackHead(embed_dim),
                "stage2_topology_source": SparseStructureFallbackHead(embed_dim * 2),
            }
        )
        self.prepatch_structure_encoder = PrePatchStructureEncoder(
            struct_channels=self.highres_structure_channels,
        )
        self.coarse_road_mask_head = (
            CoarseRoadMaskHead(
                semantic_channels=embed_dim * 2,
                psi_channels=len(PSI_DIRECTIONS) + 4,
            )
            if self.enable_coarse_road_mask
            else None
        )
        self.bottleneck_coarse_road_mask_head = (
            BottleneckCoarseRoadMaskHead(
                semantic_channels=self.num_features,
                psi_channels=len(PSI_DIRECTIONS) + 4,
            )
            if self.bottleneck_coarse_road_mask
            else None
        )
        self.highres_structure_skeleton_head = nn.Conv2d(
            self.highres_structure_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        self.highres_structure_fusion = nn.ModuleDict(
            {
                "2": HighResStructureFusion(embed_dim, self.highres_structure_channels),
                "3": HighResStructureFusion(embed_dim, self.highres_structure_channels),
            }
        )
        self.structure_surface_correction_head = StructureSurfaceCorrectionHead(
            self.highres_structure_channels,
        )
        print(
            "[INFO] High-res structure stream: {}, source=prepatch, channels={}, fuse_stages={}, fusion_mode={}".format(
                "enabled" if self.enable_highres_structure_stream else "disabled",
                self.highres_structure_channels,
                self.highres_structure_fuse_stages,
                self.highres_structure_fusion_mode,
            )
        )
        if self.structure_profile in {
            "stage23_boundary_0626",
            "stage23_boundary_0626_final_ske",
        }:
            print(
                "[INFO] Decoder structure gates: stage2/stage3 only "
                "(0626 profile), channels={}".format(
                    decoder_structure_channels
                )
            )
            print(
                "[INFO] Stage2 direct topology feature residual enabled; "
                "stage2 structure gate can refine decoder features"
            )
            print(
                "[INFO] Stage2/3 skeleton prediction mode: {}".format(
                    "direct H2/H3 predictions"
                    if self.stage_skeleton_mode == "direct"
                    else "high-res prior plus decoder residual"
                )
            )
            print(
                "[INFO] Stage2 pre-topology source and attention bias: {}".format(
                    "removed"
                    if self.remove_stage2_pre_topology_source
                    else "enabled"
                )
            )
            print(
                "[INFO] Global context calibration: bottleneck GAP -> "
                "stage3 structure gate only (strength=0.03)"
            )
        else:
            print(
                "[INFO] Restored 0621 decoder structure gates: channels={}".format(
                    decoder_structure_channels
                )
            )
        self.stage2_topology_source = (
            None
            if self.remove_stage2_pre_topology_source
            else DecoderStructureRefinement(
                channels=embed_dim * 2,
                enable_direct_feature_refinement=False,
                skeleton_gradient_ratio=self.stage2_skeleton_gradient_ratio,
            )
        )
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            if self.return_skeleton:
                self.global_topology = KeypointGuidedGlobalTopology(
                    channels=embed_dim,
                    struct_channels=self.highres_structure_channels,
                    max_nodes=global_topology_max_nodes,
                    heads=global_topology_heads,
                    alpha_max=global_topology_alpha_max,
                    enabled=self.enable_global_topology,
                    connectivity_channels=8,
                )
                self.guided_head = SkeletonGuidedHead(
                    in_channels=embed_dim,
                    hidden_channels=max(embed_dim // 2, 32),
                    init_alpha=0.0,
                    topology_eta_init=final_topology_eta_init,
                    gap_rho_init=final_gap_rho_init,
                    enable_final_structure=(
                        self.structure_profile != "stage23_boundary_0626"
                        and self.structure_profile != "stage23_boundary_0626_final_ske"
                    ),
                    enable_final_skeleton_aux=(
                        self.structure_profile == "stage23_boundary_0626_final_ske"
                    ),
                    final_skeleton_gradient_ratio=self.final_skeleton_gradient_ratio,
                    enable_post_refine_structure_interaction=(
                        self.enable_post_refine_structure_interaction
                    ),
                    highres_structure_channels=self.highres_structure_channels,
                )
                if self.enable_h3_surface_fusion:
                    self.h3_surface_proj = nn.Conv2d(
                        decoder_structure_channels[3],
                        embed_dim,
                        kernel_size=1,
                        bias=True,
                    )
                    self.surface_h3_fusion = nn.Conv2d(
                        embed_dim * 2,
                        embed_dim,
                        kernel_size=1,
                        bias=True,
                    )
                    with torch.no_grad():
                        nn.init.zeros_(self.h3_surface_proj.bias)
                        nn.init.zeros_(self.surface_h3_fusion.bias)
                        self.surface_h3_fusion.weight.zero_()
                        self.surface_h3_fusion.weight[:, :embed_dim, 0, 0].copy_(
                            torch.eye(embed_dim)
                        )
                        self.surface_h3_fusion.weight[:, embed_dim:, 0, 0].normal_(
                            mean=0.0, std=0.01
                        )
                else:
                    self.h3_surface_proj = None
                    self.surface_h3_fusion = None
                if self.structure_profile == "stage23_boundary_0626":
                    print(
                        "[INFO] Final head: surface + boundary residual only "
                        "(0626 profile; final skeleton/connectivity removed)"
                    )
                elif self.structure_profile == "stage23_boundary_0626_final_ske":
                    print(
                        "[INFO] Final head: surface + boundary residual + "
                        "final skeleton auxiliary only "
                        "(0626 profile; final connectivity removed)"
                    )
                if self.enable_global_topology:
                    print(
                        "[INFO] Global topology residual: anchors=z_struct*surface, "
                        "tokens=[z_struct,decoder_feature,connectivity], "
                        "relation_bias=relative_xy_distance+connectivity"
                    )
            else:
                self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        if self.enable_highres_structure_stream:
            skeleton_bias = 0.0
            if self.stage_skeleton_bias_init == "prior":
                prior = min(max(self.stage_skeleton_positive_prior, 1e-5), 1.0 - 1e-5)
                skeleton_bias = math.log(prior / (1.0 - prior))
            for stage_index in (2, 3):
                nn.init.zeros_(self.decoder_structure_blocks[str(stage_index)].skeleton_head.out.weight)
                nn.init.constant_(
                    self.decoder_structure_blocks[str(stage_index)].skeleton_head.out.bias,
                    skeleton_bias,
                )
            if self.stage2_topology_source is not None:
                nn.init.zeros_(self.stage2_topology_source.skeleton_head.out.weight)
                nn.init.zeros_(self.stage2_topology_source.skeleton_head.out.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        road_attentions = []
        stage1_road_attention = None
        stage2_road_attention = None

        for i_layer, layer in enumerate(self.layers):
            x_downsample.append(x)
            x = layer(
                x,
                road_prior=(
                    (stage1_road_attention, stage2_road_attention)
                    if i_layer == 2
                    else None
                ),
            )
            if i_layer == 0 and layer.last_road_attention is not None:
                stage1_road_attention = layer.last_road_attention
                road_attentions.append(
                    {
                        "stage": "encoder_stage{}_road_attention".format(i_layer + 1),
                        "road_attention": stage1_road_attention,
                    }
                )
            if i_layer == 1 and layer.last_road_attention is not None:
                stage2_road_attention = layer.last_road_attention
                road_attentions.append(
                    {
                        "stage": "encoder_stage{}_road_attention".format(i_layer + 1),
                        "road_attention": stage2_road_attention,
                    }
                )

        x = self.bottleneck_context_fusion(x)
        x = self.bottleneck_swin_block(x)
        x = self.norm(x)  # B L C

        return x, x_downsample, road_attentions

    def _highres_structure_stage_enabled(self, stage):
        if not self.enable_highres_structure_stream:
            return False
        if self.highres_structure_fusion_mode not in {"stage23", "stage23_final_correction"}:
            return False
        if self.highres_structure_fuse_stages == "stage23":
            return stage in (2, 3)
        if self.highres_structure_fuse_stages == "stage2":
            return stage == 2
        if self.highres_structure_fuse_stages == "stage3":
            return stage == 3
        return False

    def _build_highres_structure_outputs(self, structure_input):
        if (
            structure_input is None
            or not self.enable_highres_structure_stream
            and not self.enable_e128_stage_fusion
        ):
            return None, None, None
        e128, z_struct = self.prepatch_structure_encoder(
            structure_input, return_e128=True
        )
        if not self._highres_structure_shape_logged:
            expected_stage1_shape = (
                structure_input.shape[0],
                self.highres_structure_channels,
                self.patches_resolution[0],
                self.patches_resolution[1],
            )
            print(
                "[HighRes Structure] source=prepatch expected_stage1_z_struct={} z_struct={}".format(
                    expected_stage1_shape,
                    tuple(z_struct.shape),
                ),
                flush=True,
            )
            self._highres_structure_shape_logged = True
        skeleton_logits = (
            self.highres_structure_skeleton_head(z_struct)
            if self.stage_skeleton_mode == "prior_residual"
            and self.enable_highres_structure_stream
            else None
        )
        self.last_highres_z_struct = z_struct
        self.last_e128 = e128
        self.last_z_struct = z_struct
        self.last_highres_structure_skeleton = skeleton_logits
        return z_struct, e128, skeleton_logits

    def _apply_highres_structure_fusion(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        z_struct_for_surface = z_struct
        feature_map = self.highres_structure_fusion[str(stage)](
            feature_map,
            z_struct_for_surface,
        )
        return map_to_token(feature_map)

    @staticmethod
    def _resize_route_probability(probability, target_hw):
        if probability is None:
            return None
        target_hw = tuple(int(v) for v in target_hw)
        if probability.shape[-2:] == target_hw:
            return probability
        # Route expansion must preserve any positive bottleneck cell.  Never
        # average a thin-road candidate away.
        if probability.shape[-2] <= target_hw[0] and probability.shape[-1] <= target_hw[1]:
            return F.interpolate(probability, size=target_hw, mode="nearest")
        return F.adaptive_max_pool2d(probability, target_hw)

    @staticmethod
    def _route_active_token_mask(probability, layer, threshold):
        if probability is None:
            return None
        height, width = layer.input_resolution
        window_size = int(layer.window_size)
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        candidate = F.pad(probability.float(), (0, pad_w, 0, pad_h))
        selected_tokens = torch.zeros_like(candidate, dtype=torch.bool)
        for block in layer.blocks:
            shift = int(block.shift_size)
            shifted = (
                torch.roll(candidate, shifts=(-shift, -shift), dims=(-2, -1))
                if shift else candidate
            )
            scores = F.max_pool2d(
                shifted, kernel_size=window_size, stride=window_size
            )
            selected = scores >= float(threshold)
            selected_map = selected.repeat_interleave(
                window_size, dim=-2
            ).repeat_interleave(window_size, dim=-1)
            if shift:
                selected_map = torch.roll(
                    selected_map, shifts=(shift, shift), dims=(-2, -1)
                )
            selected_tokens |= selected_map
        return selected_tokens[..., :height, :width]

    def _build_bottleneck_route(self, bottleneck_tokens, psi_image):
        if not self.bottleneck_coarse_road_mask or self.bottleneck_coarse_road_mask_head is None:
            return None, None
        height, width = self.bottleneck_resolution
        bottleneck_feature = token_to_map(bottleneck_tokens, height, width)
        psi_descriptor = None
        use_psi = (
            self.enable_psi_directional_descriptor
            and self.coarse_routing_mode != "bottleneck_no_psi"
            and psi_image is not None
        )
        if use_psi:
            psi_descriptor = psi_directional_descriptor(
                psi_image,
                max_extension=4,
                color_threshold=40.0,
                output_size=(height, width),
            )
        logits, psi_gate = self.bottleneck_coarse_road_mask_head(
            bottleneck_feature,
            psi_descriptor=psi_descriptor,
        )
        probability = torch.sigmoid(logits)
        self.last_stage_features["F8"] = bottleneck_feature
        self.last_stage_features["PSI8"] = psi_descriptor
        self.last_stage_features["P8_logits"] = logits
        self.last_stage_features["P8"] = probability
        self.last_stage_features["M8"] = probability >= self.bottleneck_window_threshold
        self.last_stage_features["P8_psi_gate"] = psi_gate
        self.last_coarse_road_logits = logits
        self.last_psi_reliability_gate = psi_gate
        self.last_route_stats["bottleneck"] = {
            "active_ratio": float(
                (probability >= self.bottleneck_window_threshold).float().mean().item()
            ),
            "active_windows": int(
                (probability >= self.bottleneck_window_threshold).sum().item()
            ),
            "total_windows": int(probability.numel()),
        }
        return probability, psi_gate

    def _apply_structure_surface_correction(self, outputs, z_struct, structure_outputs):
        if (
            not self.enable_highres_structure_stream
            or self.highres_structure_fusion_mode
            not in {"final_correction", "stage23_final_correction"}
            or z_struct is None
            or not isinstance(outputs, tuple)
        ):
            return outputs

        base_surface_logits = outputs[0]
        delta_surface_logits = self.structure_surface_correction_head(
            z_struct,
            base_surface_logits.shape[-2:],
        )
        final_surface_logits = base_surface_logits + delta_surface_logits
        if structure_outputs is not None:
            structure_outputs.append(
                {
                    "stage": "structure_surface_correction",
                    "structure_surface_base_logits": base_surface_logits.detach(),
                    "structure_surface_delta_logits": delta_surface_logits,
                }
            )
        return (final_surface_logits, *outputs[1:])

    def _decoder_structure_enabled(self, stage):
        return stage in (2, 3)

    @staticmethod
    def _placeholder_structure_outputs(feature_map):
        batch = feature_map.shape[0]
        height = feature_map.shape[-2]
        width = feature_map.shape[-1]
        device = feature_map.device
        dtype = feature_map.dtype
        skeleton_logits = torch.zeros(
            batch,
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        connectivity_logits = torch.zeros(
            batch,
            8,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        direction_logits = torch.zeros(
            batch,
            2,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        structure_gate = torch.zeros(
            batch,
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        return skeleton_logits, connectivity_logits, direction_logits, structure_gate, None

    def _build_stage3_global_context(self, bottleneck_tokens, target_hw):
        batch, length, channels = bottleneck_tokens.shape
        bottleneck_height, bottleneck_width = self.bottleneck_resolution
        if length != bottleneck_height * bottleneck_width:
            raise ValueError("Bottleneck token length does not match resolution.")
        context = bottleneck_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            bottleneck_height,
            bottleneck_width,
        )
        context = F.adaptive_avg_pool2d(context, output_size=1)
        if self.global_context_head is not None:
            context = self.global_context_head(context)
        context = F.interpolate(
            context,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return context

    @staticmethod
    def _partition_nchw_windows(feature_map, window_size):
        batch, channels, height, width = feature_map.shape
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        padded = F.pad(feature_map, (0, pad_w, 0, pad_h))
        windows = window_partition(
            padded.permute(0, 2, 3, 1).contiguous(), window_size
        ).permute(0, 3, 1, 2).contiguous()
        return windows, padded, height + pad_h, width + pad_w

    @staticmethod
    def _reverse_nchw_windows(windows, batch, height, width, window_size):
        return window_reverse(
            windows.permute(0, 2, 3, 1).contiguous(),
            window_size,
            height,
            width,
        ).permute(0, 3, 1, 2).contiguous()[:batch]

    def _run_sparse_decoder_structure_block(
        self,
        feature_map,
        stage,
        bottleneck_tokens,
        active_token_mask,
        apply_feature_refinement=True,
        disable_skeleton_prediction=False,
        skeleton_prior=None,
        previous_structure_feat=None,
        block_stage=None,
    ):
        """Run the existing H2/H3 structure module on selected 8x8 windows.

        Inactive locations use a zero-initialized 1x1 fallback head. Active
        windows are gathered in their original spatial order, processed by
        the existing structure block, and scattered back to the full map.
        """
        if active_token_mask is None:
            return self._run_decoder_structure_block(
                feature_map,
                stage,
                bottleneck_tokens,
                block_stage=block_stage,
                apply_feature_refinement=apply_feature_refinement,
                disable_skeleton_prediction=disable_skeleton_prediction,
                skeleton_prior=skeleton_prior,
                previous_structure_feat=previous_structure_feat,
            )

        height, width = feature_map.shape[-2:]
        mask = active_token_mask
        if mask.dim() == 3:
            mask = mask.view(mask.shape[0], height, width, 1).permute(0, 3, 1, 2)
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(mask.float(), size=(height, width), mode="nearest")
        mask = mask.bool()
        block_key = stage if block_stage is None else block_stage
        if bool(mask.all().item()):
            return self._run_decoder_structure_block(
                feature_map,
                stage,
                bottleneck_tokens,
                block_stage=block_key,
                apply_feature_refinement=apply_feature_refinement,
                disable_skeleton_prediction=disable_skeleton_prediction,
                skeleton_prior=skeleton_prior,
                previous_structure_feat=previous_structure_feat,
            )
        block = (
            self.stage2_topology_source
            if block_key == "stage2_topology_source"
            else self.decoder_structure_blocks[str(block_key)]
        )
        fallback = self.sparse_structure_fallback_heads[str(block_key)](feature_map)
        fallback_skeleton = fallback["skeleton"]
        if disable_skeleton_prediction:
            if skeleton_prior is None:
                fallback_skeleton = None
            else:
                prior = F.interpolate(
                    skeleton_prior,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                fallback_skeleton = prior + fallback_skeleton

        window_size = int(self.layers_up[stage].window_size)
        mask_windows, _, padded_h, padded_w = self._partition_nchw_windows(
            mask.float(), window_size
        )
        active_window_mask = mask_windows.amax(dim=(1, 2, 3)) > 0
        selected_indices = active_window_mask.nonzero(as_tuple=False).flatten()
        if selected_indices.numel() == 0:
            return (
                fallback["feature"], fallback_skeleton,
                fallback["connectivity"], fallback["direction"],
                fallback["structure_gate"], fallback["roadness"],
            )
        batch = feature_map.shape[0]

        feature_windows, _, _, _ = self._partition_nchw_windows(feature_map, window_size)

        def gather_windows(value):
            if value is None:
                return None
            if value.shape[-2:] != (height, width):
                value = F.interpolate(
                    value, size=(height, width), mode="bilinear", align_corners=False
                )
            windows, _, _, _ = self._partition_nchw_windows(value, window_size)
            return windows.index_select(0, selected_indices)

        selected_feature = feature_windows.index_select(0, selected_indices)
        selected_previous = gather_windows(previous_structure_feat)
        selected_prior = gather_windows(skeleton_prior)
        selected_context = None
        if self.use_stage3_global_context and stage == 3:
            context = self._build_stage3_global_context(
                bottleneck_tokens, (height, width)
            )
            selected_context = gather_windows(context)

        heavy = block(
            selected_feature,
            global_context=selected_context,
            apply_feature_refinement=apply_feature_refinement,
            disable_skeleton_prediction=disable_skeleton_prediction,
            skeleton_prior=selected_prior,
            previous_structure_feat=selected_previous,
        )

        def scatter_windows(selected_value, fallback_value):
            if selected_value is None:
                return fallback_value
            if fallback_value is None:
                return None
            fallback_windows, _, _, _ = self._partition_nchw_windows(
                fallback_value, window_size
            )
            padded_windows = fallback_windows.clone()
            padded_windows.index_copy_(0, selected_indices, selected_value)
            return self._reverse_nchw_windows(
                padded_windows, batch, padded_h, padded_w, window_size
            )[:, :, :height, :width]

        out = scatter_windows(heavy[0], fallback["feature"])
        skeleton = scatter_windows(heavy[1], fallback_skeleton)
        connectivity = scatter_windows(heavy[2], fallback["connectivity"])
        direction = scatter_windows(heavy[3], fallback["direction"])
        structure_gate = scatter_windows(heavy[4], fallback["structure_gate"])
        heavy_roadness = heavy[5] if isinstance(heavy[5], dict) else {}
        fallback_roadness = fallback["roadness"]
        roadness = {}
        for key, fallback_value in fallback_roadness.items():
            roadness[key] = scatter_windows(
                heavy_roadness.get(key), fallback_value
            )
        return out, skeleton, connectivity, direction, structure_gate, roadness

    def _run_decoder_structure_block(
        self,
        feature_map,
        stage,
        bottleneck_tokens,
        block_stage=None,
        apply_feature_refinement=True,
        disable_skeleton_prediction=False,
        skeleton_prior=None,
        previous_structure_feat=None,
        active_token_mask=None,
    ):
        if not self._decoder_structure_enabled(stage):
            return feature_map, *self._placeholder_structure_outputs(feature_map)

        block_stage = stage if block_stage is None else block_stage
        if (
            block_stage == "stage2_topology_source"
            and self.stage2_topology_source is not None
        ):
            block = self.stage2_topology_source
        else:
            block = self.decoder_structure_blocks[str(block_stage)]
        if active_token_mask is not None:
            return self._run_sparse_decoder_structure_block(
                feature_map,
                stage,
                bottleneck_tokens,
                active_token_mask=active_token_mask,
                apply_feature_refinement=apply_feature_refinement,
                disable_skeleton_prediction=disable_skeleton_prediction,
                skeleton_prior=skeleton_prior,
                previous_structure_feat=previous_structure_feat,
                block_stage=block_stage,
            )
        global_context = None
        if self.use_stage3_global_context and stage == 3:
            global_context = self._build_stage3_global_context(
                bottleneck_tokens,
                feature_map.shape[-2:],
            )
        return block(
            feature_map,
            global_context=global_context,
            apply_feature_refinement=apply_feature_refinement,
            disable_skeleton_prediction=disable_skeleton_prediction,
            skeleton_prior=skeleton_prior,
            previous_structure_feat=previous_structure_feat,
        )

    def _decoder_skeleton_disabled(self, stage):
        return (
            self.stage_skeleton_mode == "prior_residual"
            and self.enable_highres_structure_stream
            and stage in (2, 3)
        )

    @staticmethod
    def _append_structure_output(
        structure_outputs,
        stage,
        skeleton,
        connectivity,
        direction,
        structure_gate,
        roadness,
        refinement_step=None,
        stage_loss_scale=None,
    ):
        diagnostics = roadness if isinstance(roadness, dict) else None
        roadness = None if diagnostics is not None else roadness
        item = {
            "stage": stage,
            "connectivity": connectivity,
            "direction": direction,
            "structure_gate": structure_gate,
            "roadness": roadness,
        }
        if diagnostics:
            item.update(diagnostics)
        if skeleton is not None:
            item["skeleton"] = skeleton
        if refinement_step is not None:
            item["refinement_step"] = refinement_step
        if stage_loss_scale is not None:
            item["stage_loss_scale"] = stage_loss_scale
        structure_outputs.append(item)

    @staticmethod
    def _decoder_connectivity_used(connectivity_logits):
        return torch.sigmoid(connectivity_logits).detach()

    @staticmethod
    def _latest_local_topology_features(structure_outputs):
        if not structure_outputs:
            return None
        for item in reversed(structure_outputs):
            if not isinstance(item, dict):
                continue
            connectivity = item.get("connectivity")
            if connectivity is None:
                continue
            connectivity_feature = (
                torch.sigmoid(connectivity).detach()
                if connectivity is not None
                else None
            )
            return connectivity_feature
        return None

    # Decoder and skip connection with DCA-FPN.
    def forward_up_features(
        self,
        x,
        x_downsample,
        bottleneck_tokens=None,
        z_struct=None,
        e128=None,
        psi_image=None,
        highres_structure_skeleton=None,
    ):
        """
        Decoder with DCA-FPN-Lite refinement on stage 2 and stage 3 skips.
        
        Args:
            x: bottleneck feature [B, L, C]
            x_downsample: list of encoder skip features [[B, L, C], ...]
        
        Returns:
            decoder output [B, L, C]
        """
        structure_outputs = []
        stage2_structure_feat = None
        coarse_road_logits = None
        coarse_probability = None
        selection_probability = None
        self.last_sparse_window_stats = {}
        self.last_stage_features = {}
        self.last_route_stats = {}
        if bottleneck_tokens is None:
            bottleneck_tokens = x
        if self.coarse_routing_mode.startswith("bottleneck"):
            selection_probability, _ = self._build_bottleneck_route(
                bottleneck_tokens,
                psi_image,
            )
            if selection_probability is not None:
                structure_outputs.append({
                    "stage": "bottleneck_coarse_road",
                    "bottleneck_coarse_road_logits": self.last_stage_features["P8_logits"],
                    "psi_reliability_gate": self.last_stage_features["P8_psi_gate"],
                    "routing_threshold": self.bottleneck_window_threshold,
                })
                if (
                    self.current_epoch < self.coarse_route_warmup_epochs
                    and self.bottleneck_route_warmup_mode == "dense"
                ):
                    # Supervise the coarse mask during warmup, but do not let
                    # its early hard threshold change the decoder feature path.
                    selection_probability = None
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # Bottleneck layer, no skip connection
                pass
            elif inx == 1:
                # inx=1 (Layer 3 skip, 1/16 resolution): direct concatenation.
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
            else:
                # inx=2,3 (Layer 2,1 skip): apply DCA-FPN.
                skip = x_downsample[3 - inx]  # [B, L, C]
                
                # 计算当前层的空间分辨率
                H = self.patches_resolution[0] // (2 ** (3 - inx))
                W = self.patches_resolution[1] // (2 ** (3 - inx))
                
                # 1. Token → Feature Map 转换
                skip_map = token_to_map(skip, H, W)  # [B, C, H, W]
                x_map = token_to_map(x, H, W)  # [B, C, H, W]
                
                block_idx = inx - 2
                skip_refined = self.dca_blocks[block_idx](deep=x_map, shallow=skip_map)  # [B, C, H, W]
                
                # 4. Feature Map → Token 转换
                skip_refined = map_to_token(skip_refined)  # [B, L, C]
                
                # 5. Skip concatenation
                x = torch.cat([x, skip_refined], -1)
                x = self.concat_back_dim[inx](x)

            decoder_structure_gate_enabled = (
                self._decoder_structure_enabled(inx)
                and isinstance(layer_up, BasicLayer_up)
                and inx in (2, 3)
            )
            if decoder_structure_gate_enabled:
                decoder_skeleton_disabled = self._decoder_skeleton_disabled(inx)
                if inx == 3:
                    stage3_sparse_probability = (
                        self._resize_route_probability(
                            selection_probability,
                            layer_up.input_resolution,
                        )
                        if selection_probability is not None
                        else None
                    )
                    if (
                        stage3_sparse_probability is not None
                        and self.coarse_routing_mode != "dense"
                    ):
                        self.last_stage_features["M64_stage3"] = (
                            stage3_sparse_probability >= self.stage3_window_threshold
                        )
                    x = layer_up(
                        x,
                        sparse_probability_map=stage3_sparse_probability,
                        sparse_window_compute=(
                            self.sparse_window_compute
                            and self.coarse_routing_mode != "dense"
                        ),
                        sparse_threshold=(
                            self.bottleneck_window_threshold
                            if self.coarse_routing_mode.startswith("bottleneck")
                            else self.stage3_window_threshold
                        ),
                    )
                    self.last_sparse_window_stats["stage3"] = dict(
                        layer_up.last_sparse_stats
                    )
                    self.last_route_stats["stage3"] = dict(layer_up.last_sparse_stats)
                    output_scale = 2 ** max(2 - inx, 0)
                    output_height = self.patches_resolution[0] // output_scale
                    output_width = self.patches_resolution[1] // output_scale
                    x = self._apply_highres_structure_fusion(
                        x,
                        z_struct,
                        inx,
                        (output_height, output_width),
                    )
                    x_map = token_to_map(x, output_height, output_width)
                    (
                        x_map,
                        skeleton_i,
                        connectivity_i,
                        direction_i,
                        structure_gate_i,
                        roadness_i,
                    ) = self._run_decoder_structure_block(
                        x_map,
                        inx,
                        bottleneck_tokens,
                        apply_feature_refinement=True,
                        disable_skeleton_prediction=decoder_skeleton_disabled,
                        skeleton_prior=(
                            highres_structure_skeleton
                            if self.stage_skeleton_mode == "prior_residual"
                            else None
                        ),
                        previous_structure_feat=stage2_structure_feat,
                        active_token_mask=(
                            layer_up.last_active_token_mask
                            if self.sparse_window_compute
                            else None
                        ),
                    )
                    if isinstance(roadness_i, dict):
                        self.last_stage_features["H3"] = roadness_i.get(
                            "structure_feat"
                        )
                        roadness_i.pop("structure_feat", None)
                    x = map_to_token(x_map)
                    self._append_structure_output(
                        structure_outputs,
                        stage=inx,
                        skeleton=skeleton_i,
                        connectivity=connectivity_i,
                        direction=direction_i,
                        structure_gate=structure_gate_i,
                        roadness=roadness_i,
                        refinement_step=1,
                        stage_loss_scale=1.0,
                    )
                    stage2_structure_feat = None
                    continue
                if inx == 2 and self.remove_stage2_pre_topology_source:
                    stage2_sparse_probability = (
                        self._resize_route_probability(
                            selection_probability,
                            layer_up.input_resolution,
                        )
                        if selection_probability is not None
                        else None
                    )
                    if (
                        stage2_sparse_probability is not None
                        and self.coarse_routing_mode != "dense"
                    ):
                        self.last_stage_features["M64_stage2"] = (
                            stage2_sparse_probability >= self.stage2_window_threshold
                        )
                    x = layer_up(
                        x,
                        sparse_probability_map=stage2_sparse_probability,
                        sparse_window_compute=(
                            self.sparse_window_compute
                            and self.coarse_routing_mode != "dense"
                        ),
                        sparse_threshold=(
                            self.bottleneck_window_threshold
                            if self.coarse_routing_mode.startswith("bottleneck")
                            else self.stage2_window_threshold
                        ),
                    )
                    self.last_sparse_window_stats["stage2"] = dict(
                        layer_up.last_sparse_stats
                    )
                    self.last_route_stats["stage2"] = dict(layer_up.last_sparse_stats)
                    skeleton_0 = connectivity_0 = direction_0 = structure_gate_0 = roadness_0 = None
                else:
                    input_height, input_width = layer_up.input_resolution
                    x_map = token_to_map(x, input_height, input_width)
                    prehead_probability = (
                        self._resize_route_probability(
                            selection_probability, layer_up.input_resolution
                        )
                        if inx == 2 and selection_probability is not None
                        else None
                    )
                    prehead_active_mask = (
                        self._route_active_token_mask(
                            prehead_probability,
                            layer_up,
                            self.stage2_window_threshold,
                        )
                        if self.sparse_window_compute
                        and self.coarse_routing_mode != "dense"
                        and prehead_probability is not None
                        else None
                    )
                    (
                        _,
                        skeleton_0,
                        connectivity_0,
                        direction_0,
                        structure_gate_0,
                        roadness_0,
                    ) = self._run_decoder_structure_block(
                        x_map,
                        inx,
                        bottleneck_tokens,
                        block_stage="stage2_topology_source" if inx == 2 else inx,
                        apply_feature_refinement=False,
                        disable_skeleton_prediction=decoder_skeleton_disabled,
                        skeleton_prior=highres_structure_skeleton,
                        active_token_mask=prehead_active_mask,
                    )
                    if isinstance(roadness_0, dict):
                        roadness_0.pop("structure_feat", None)
                    if decoder_skeleton_disabled:
                        skeleton_used = None
                        connectivity_used = self._decoder_connectivity_used(connectivity_0)
                    else:
                        skeleton_used = torch.sigmoid(skeleton_0).detach()
                        connectivity_used = torch.sigmoid(connectivity_0).detach()
                    stage2_sparse_probability = (
                        self._resize_route_probability(
                            selection_probability,
                            layer_up.input_resolution,
                        )
                        if inx == 2 and selection_probability is not None
                        else None
                    )
                    if (
                        stage2_sparse_probability is not None
                        and self.coarse_routing_mode != "dense"
                    ):
                        self.last_stage_features["M64_stage2"] = (
                            stage2_sparse_probability >= self.stage2_window_threshold
                        )
                    x = layer_up(
                        x,
                        decoder_skeleton_prob=skeleton_used,
                        decoder_connectivity_prob=connectivity_used,
                        decoder_direction_prob=direction_0,
                        sparse_probability_map=stage2_sparse_probability,
                        sparse_window_compute=(
                            self.sparse_window_compute
                            and self.coarse_routing_mode != "dense"
                            and inx == 2
                        ),
                        sparse_threshold=(
                            self.bottleneck_window_threshold
                            if self.coarse_routing_mode.startswith("bottleneck")
                            else self.stage2_window_threshold
                        ),
                    )
                    if inx == 2:
                        self.last_sparse_window_stats["stage2"] = dict(
                            layer_up.last_sparse_stats
                        )
                        self.last_route_stats["stage2"] = dict(layer_up.last_sparse_stats)
                output_scale = 2 ** max(2 - inx, 0)
                output_height = self.patches_resolution[0] // output_scale
                output_width = self.patches_resolution[1] // output_scale
                x = self._apply_highres_structure_fusion(
                    x,
                    z_struct,
                    inx,
                    (output_height, output_width),
                )
                if skeleton_0 is not None or connectivity_0 is not None or direction_0 is not None:
                    self._append_structure_output(
                        structure_outputs,
                        stage=inx,
                        skeleton=skeleton_0,
                        connectivity=connectivity_0,
                        direction=direction_0,
                        structure_gate=structure_gate_0,
                        roadness=roadness_0,
                        refinement_step=0,
                        stage_loss_scale=0.5,
                    )

                x_map = token_to_map(x, output_height, output_width)
                (
                    x_map,
                    skeleton_i,
                    connectivity_i,
                    direction_i,
                    structure_gate_i,
                    roadness_i,
                ) = self._run_decoder_structure_block(
                    x_map,
                    inx,
                    bottleneck_tokens,
                    apply_feature_refinement=True,
                    disable_skeleton_prediction=decoder_skeleton_disabled,
                    skeleton_prior=(
                        highres_structure_skeleton
                        if self.stage_skeleton_mode == "prior_residual"
                        else None
                    ),
                    active_token_mask=(
                        layer_up.last_active_token_mask
                        if self.sparse_window_compute
                        else None
                    ),
                )
                if inx == 2 and isinstance(roadness_i, dict):
                    stage2_structure_feat = roadness_i.pop("structure_feat", None)
                    self.last_stage_features["H2"] = stage2_structure_feat
                x = map_to_token(x_map)
                self._append_structure_output(
                    structure_outputs,
                    stage=inx,
                    skeleton=skeleton_i,
                    connectivity=connectivity_i,
                    direction=direction_i,
                    structure_gate=structure_gate_i,
                    roadness=roadness_i,
                    refinement_step=1,
                    stage_loss_scale=1.0,
                )
                continue
            else:
                decoder_skeleton_disabled = self._decoder_skeleton_disabled(inx)
                route_probability = (
                    self._resize_route_probability(
                        selection_probability,
                        layer_up.input_resolution,
                    )
                    if selection_probability is not None
                    and isinstance(layer_up, BasicLayer_up)
                    else None
                )
                if route_probability is not None and self.coarse_routing_mode != "dense":
                    route_mask = route_probability >= self.bottleneck_window_threshold
                    self.last_stage_features[
                        "M16" if inx == 0 else "M32"
                    ] = route_mask
                    self.last_route_stats["inx0" if inx == 0 else "inx1"] = {
                        "active_ratio": float(route_mask.float().mean().item()),
                        "active_windows": int(route_mask.sum().item()),
                        "total_windows": int(route_mask.numel()),
                    }
                if isinstance(layer_up, BasicLayer_up):
                    x = layer_up(
                        x,
                        sparse_probability_map=route_probability,
                        sparse_window_compute=(
                            self.sparse_window_compute
                            and self.coarse_routing_mode != "dense"
                        ),
                        sparse_threshold=(
                            self.bottleneck_window_threshold
                            if self.coarse_routing_mode.startswith("bottleneck")
                            else self.stage2_window_threshold
                        ),
                    )
                else:
                    x = layer_up(x)
                if isinstance(layer_up, BasicLayer_up):
                    self.last_sparse_window_stats["inx1"] = dict(
                        layer_up.last_sparse_stats
                    )
                output_scale = 2 ** max(2 - inx, 0)
                output_height = self.patches_resolution[0] // output_scale
                output_width = self.patches_resolution[1] // output_scale
                x = self._apply_highres_structure_fusion(
                    x,
                    z_struct,
                    inx,
                    (output_height, output_width),
                )
                x_map = token_to_map(x, output_height, output_width)
                (
                    x_map,
                    skeleton_i,
                    connectivity_i,
                    direction_i,
                    structure_gate_i,
                    roadness_i,
                ) = self._run_decoder_structure_block(
                    x_map,
                    inx,
                    bottleneck_tokens,
                    disable_skeleton_prediction=decoder_skeleton_disabled,
                    skeleton_prior=highres_structure_skeleton,
                )
                x = map_to_token(x_map)
                if (
                    inx == 1
                    and self.enable_coarse_road_mask
                    and not self.coarse_routing_mode.startswith("bottleneck")
                ):
                    # inx=1 is outside the stage2/3 structure-gated branch.
                    # Emit P64 here so it is available before stage2 blocks.
                    feature32 = token_to_map(x, output_height, output_width)
                    self.last_stage_features["F32"] = feature32
                    psi_descriptor = None
                    if self.enable_psi_directional_descriptor and psi_image is not None:
                        psi_descriptor = psi_directional_descriptor(
                            psi_image,
                            max_extension=4,
                            color_threshold=40.0,
                            # P64 is twice the spatial resolution of the
                            # P32 semantic feature. Keep this dynamic so
                            # 256 and 512 inputs use matching PSI/semantic
                            # maps instead of relying on a fixed 64x64 size.
                            output_size=(
                                feature32.shape[-2] * 2,
                                feature32.shape[-1] * 2,
                            ),
                        )
                    coarse_road_logits, psi_gate = self.coarse_road_mask_head(
                        feature32,
                        psi_descriptor=psi_descriptor,
                        output_size=(
                            feature32.shape[-2] * 2,
                            feature32.shape[-1] * 2,
                        ),
                    )
                    coarse_probability = torch.sigmoid(coarse_road_logits)
                    selection_probability = (
                        self.sparse_selection_probability_override
                        if self.sparse_selection_probability_override is not None
                        else coarse_probability
                    )
                    route_warmup = max(
                        self.coarse_route_warmup_epochs,
                        self.routing_warmup_epochs,
                    )
                    if (
                        self.current_epoch < route_warmup
                        or not self.routing_calibration_done
                    ):
                        # Keep Stage 2/3 dense while P64 learns its road map.
                        selection_probability = None
                    self.last_coarse_road_logits = coarse_road_logits
                    self.last_psi_reliability_gate = psi_gate
                    self.last_stage_features["P64_logits"] = coarse_road_logits
                    self.last_stage_features["P64"] = coarse_probability
                    structure_outputs.append(
                        {
                            "stage": "coarse_road",
                            "coarse_road_logits": coarse_road_logits,
                            "psi_reliability_gate": psi_gate,
                        }
                    )
            if self._decoder_structure_enabled(inx):
                self._append_structure_output(
                    structure_outputs,
                    stage=inx,
                    skeleton=skeleton_i,
                    connectivity=connectivity_i,
                    direction=direction_i,
                    structure_gate=structure_gate_i,
                    roadness=roadness_i,
                )

        x = self.norm_up(x)  # B L C

        return x, structure_outputs

    def _surface_prior_for_global_topology(self, x, z_struct):
        prior_modules = (
            self.guided_head.surface_proj,
            self.guided_head.surface_branch,
            self.guided_head.surface_refine,
            self.guided_head.surface_head,
        )
        if getattr(
            self.guided_head,
            "enable_post_refine_structure_interaction",
            False,
        ):
            prior_modules = (
                *prior_modules,
                self.guided_head.post_refine_structure_interaction,
            )
        prior_training = [module.training for module in prior_modules]
        for module in prior_modules:
            module.eval()
        try:
            with torch.no_grad():
                surface_feat = self.guided_head.surface_branch(
                    self.guided_head.surface_proj(x)
                )
                surface_feat = self.guided_head.surface_refine(surface_feat)
                surface_feat = self.guided_head._apply_post_refine_structure_interaction(
                    surface_feat,
                    z_struct,
                )
                return torch.sigmoid(self.guided_head.surface_head(surface_feat))
        finally:
            for module, was_training in zip(prior_modules, prior_training):
                module.train(was_training)

    def up_x4(self, x, structure_outputs=None, z_struct=None):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            if self.return_skeleton:
                if self.enable_global_topology and z_struct is not None:
                    surface_prob = self._surface_prior_for_global_topology(
                        x,
                        z_struct,
                    )
                    connectivity_feature = self._latest_local_topology_features(
                        structure_outputs
                    )
                    x = self.global_topology.forward_feature_anchors(
                        x,
                        z_struct,
                        surface_prob,
                        connectivity_feature=connectivity_feature,
                    )
                if (
                    self.enable_h3_surface_fusion
                    and self.h3_surface_proj is not None
                    and self.surface_h3_fusion is not None
                ):
                    h3 = self.last_stage_features.get("H3")
                    if h3 is not None:
                        h3_surface = self.h3_surface_proj(h3)
                        h3_surface = F.interpolate(
                            h3_surface,
                            size=x.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        x = self.surface_h3_fusion(
                            torch.cat([x, h3_surface], dim=1)
                        )
                x = self.guided_head(x, z_struct=z_struct)
            else:
                x = self.output(x)

        return x

    def set_route_epoch(self, epoch):
        """Set the zero-based training epoch used by route warmup."""
        self.current_epoch = max(0, int(epoch))

    def set_routing_thresholds(self, stage2=None, stage3=None):
        if stage2 is not None:
            self.stage2_window_threshold = float(stage2)
        if stage3 is not None:
            self.stage3_window_threshold = float(stage3)

    def set_routing_calibration_done(self, value=True):
        self.routing_calibration_done = bool(value)

    def routing_state(self):
        return {
            "routing_warmup_epochs": int(self.routing_warmup_epochs),
            "current_epoch": int(self.current_epoch),
            "stage2_window_threshold": float(self.stage2_window_threshold),
            "stage3_window_threshold": float(self.stage3_window_threshold),
            "sparse_enabled": bool(self.sparse_window_compute),
            "calibration_done": bool(self.routing_calibration_done),
        }

    def forward(
        self,
        x,
    ):
        structure_input = x
        x, x_downsample, road_attentions = self.forward_features(x)
        z_struct, e128, highres_structure_skeleton = (
            self._build_highres_structure_outputs(structure_input)
        )
        x, structure_outputs = self.forward_up_features(
            x,
            x_downsample,
            bottleneck_tokens=x,
            z_struct=z_struct,
            e128=e128,
            psi_image=structure_input,
            highres_structure_skeleton=highres_structure_skeleton,
        )
        if self.return_skeleton and highres_structure_skeleton is not None:
            structure_outputs.append(
                {
                    "stage": "highres_structure",
                    "highres_structure_skeleton": highres_structure_skeleton,
                }
            )
        if self.return_skeleton and road_attentions:
            structure_outputs.extend(road_attentions)
        x = self.up_x4(
            x,
            structure_outputs=structure_outputs if self.return_skeleton else None,
            z_struct=z_struct,
        )
        if self.return_skeleton and isinstance(x, tuple):
            x = self._apply_structure_surface_correction(
                x,
                z_struct,
                structure_outputs,
            )
            x = (*x, structure_outputs)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
