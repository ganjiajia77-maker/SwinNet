import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .dca_fpn_lite import DCAFPNLite
from .msfe_block import MSFEBlock
from .bottleneck_context_fusion import GlobalLocalContextFusion
from .g2l2_bottleneck import G2L2Bottleneck
from .road_attention_head import RoadAttentionHead
from losses.road_losses import build_connectivity_target
from .skeleton_guided_head import (
    DecoderStructureRefinement,
    GlobalContextHead,
    STAGE3_GLOBAL_CONTEXT_CHANNELS,
    SkeletonGuidedHead,
)


class MoEFFNGating(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFNGating, self).__init__()
        self.gating_network = nn.Linear(dim, dim)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)) for _ in range(num_experts)])

    def forward(self, x):
        weights = self.gating_network(x)
        weights = torch.nn.functional.softmax(weights, dim=-1)
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)
        outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
        return outputs


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


TOPOLOGY_QK_RELATIVE_EPS = 1e-6
DEFAULT_STAGE_TOPOLOGY_RATIO = 0.08
DEFAULT_STAGE_TOPOLOGY_TOPO_CLIP = 4.0


class GapQueryFormulaConfig:
    """Eval-time overrides for gap-query topology bias (no new trainable params)."""

    __slots__ = (
        "topo_conf_threshold",
        "h_threshold",
        "h_key_threshold",
        "g_mode",
        "pool_kernel",
        "g_h_max",
        "directional_mode",
    )

    def __init__(
        self,
        topo_conf_threshold=0.0,
        h_threshold=0.0,
        h_key_threshold=0.0,
        g_mode="pool_gap",
        pool_kernel=7,
        g_h_max=1.0,
        directional_mode="max",
    ):
        self.topo_conf_threshold = float(topo_conf_threshold)
        self.h_threshold = float(h_threshold)
        self.h_key_threshold = float(h_key_threshold)
        self.g_mode = str(g_mode)
        self.pool_kernel = int(pool_kernel)
        self.g_h_max = float(g_h_max)
        self.directional_mode = str(directional_mode)

    def copy(self, **overrides):
        params = {slot: getattr(self, slot) for slot in self.__slots__}
        params.update(overrides)
        return GapQueryFormulaConfig(**params)

    def to_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}


DEFAULT_GAP_QUERY_FORMULA = GapQueryFormulaConfig()


def _mean_positive(values, eps=TOPOLOGY_QK_RELATIVE_EPS):
    positive = values[values > 0]
    if positive.numel() > 0:
        return positive.mean()
    return values.abs().mean().clamp_min(eps)


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
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x,
        mask=None,
        topology_bias=None,
        road_attention_bias=None,
        structure_attention_bias=None,
        topology_alpha=None,
        use_qk_relative_topology=True,
        topology_ratio=DEFAULT_STAGE_TOPOLOGY_RATIO,
        topo_clip=DEFAULT_STAGE_TOPOLOGY_TOPO_CLIP,
        eps=TOPOLOGY_QK_RELATIVE_EPS,
    ):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        qk_logits = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = qk_logits + relative_position_bias.unsqueeze(0)
        if getattr(self, "capture_logits", False):
            with torch.no_grad():
                self.last_qk_logits_abs_mean = float(qk_logits.abs().mean().detach().cpu())
                self.last_logits_before_topology = attn.detach()
        if topology_bias is not None:
            if use_qk_relative_topology:
                qk_scale = qk_logits.abs().mean().detach()
                topo_scale = _mean_positive(topology_bias, eps=eps).detach()
                t_unit = (topology_bias / (topo_scale + eps)).clamp(0.0, float(topo_clip))
                topology_term = float(topology_ratio) * qk_scale * t_unit.unsqueeze(1)
                active_pair_ratio = float((topology_bias > eps).float().mean().detach().cpu())
            else:
                if topology_alpha is None:
                    raise ValueError("topology_alpha is required for legacy pairwise topology bias")
                topology_term = topology_alpha * topology_bias.unsqueeze(1)
                qk_scale = qk_logits.abs().mean().detach()
                topo_scale = _mean_positive(topology_bias, eps=eps).detach()
                active_pair_ratio = float((topology_bias > eps).float().mean().detach().cpu())
            if getattr(self, "capture_logits", False):
                with torch.no_grad():
                    qk_mean = qk_logits.abs().mean()
                    term_mean = topology_term.abs().mean()
                    self.last_qk_scale = float(qk_scale.detach().cpu())
                    self.last_topo_scale = float(topo_scale.detach().cpu())
                    self.last_active_pair_ratio = active_pair_ratio
                    self.last_topology_term_over_qk_ratio = float(
                        (term_mean / (qk_mean + eps)).detach().cpu()
                    )
                    self.last_topology_term_abs_mean = float(term_mean.detach().cpu())
            attn = attn + topology_term
        if road_attention_bias is not None:
            if road_attention_bias.dim() == 3:
                road_attention_bias = road_attention_bias.unsqueeze(1)
            attn = attn + road_attention_bias
        if structure_attention_bias is not None:
            if structure_attention_bias.dim() == 3:
                structure_attention_bias = structure_attention_bias.unsqueeze(1)
            attn = attn + structure_attention_bias

        if mask is not None:
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
        self.capture_topology_diagnostics = False
        self.last_topology_diagnostics = None
        self.topology_bias_mode = "pairwise_skeleton"
        self.topology_ratio = DEFAULT_STAGE_TOPOLOGY_RATIO
        self.topology_topo_clip = DEFAULT_STAGE_TOPOLOGY_TOPO_CLIP
        if self.use_road_bias:
            self.road_bias_scale_a1 = nn.Parameter(torch.tensor(0.0))
            self.road_bias_scale_a2 = nn.Parameter(torch.tensor(0.0))
            self.road_bias_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("road_bias_scale_a1", None)
            self.register_parameter("road_bias_scale_a2", None)
            self.register_parameter("road_bias_scale", None)
        if self.use_decoder_structure_bias:
            self.decoder_skeleton_bias_scale = nn.Parameter(torch.tensor(0.0))
            self.decoder_connectivity_bias_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("decoder_skeleton_bias_scale", None)
            self.register_parameter("decoder_connectivity_bias_scale", None)
        self.gap_query_formula = GapQueryFormulaConfig()
        self._register_topology_buffers()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _register_topology_buffers(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"),
            dim=-1,
        ).view(-1, 2)
        delta = coords.unsqueeze(0) - coords.unsqueeze(1)
        dy = delta[..., 0]
        dx = delta[..., 1]

        direction = torch.zeros_like(dy, dtype=torch.long)
        direction[(dy > 0) & (dx == 0)] = 1
        direction[(dy == 0) & (dx < 0)] = 2
        direction[(dy == 0) & (dx > 0)] = 3
        direction[(dy < 0) & (dx < 0)] = 4
        direction[(dy < 0) & (dx > 0)] = 5
        direction[(dy > 0) & (dx < 0)] = 6
        direction[(dy > 0) & (dx > 0)] = 7

        opposite = torch.tensor([1, 0, 3, 2, 7, 6, 5, 4], dtype=torch.long)
        distance = torch.maximum(dy.abs(), dx.abs()).float()
        distance_decay = torch.exp(-distance / 4.0)
        distance_decay[distance == 0] = 0.0
        pair_norm = distance.clamp_min(1.0)
        pair_unit_i_to_j = torch.stack(
            (
                -dx.float() / pair_norm,
                -dy.float() / pair_norm,
            ),
            dim=-1,
        )
        pair_unit_i_to_j[distance == 0] = 0.0

        self.register_buffer(
            "topology_direction_one_hot",
            F.one_hot(direction, num_classes=8).float(),
            persistent=False,
        )
        self.register_buffer(
            "topology_opposite_direction_one_hot",
            F.one_hot(opposite[direction], num_classes=8).float(),
            persistent=False,
        )
        self.register_buffer(
            "topology_distance_decay",
            distance_decay,
            persistent=False,
        )
        self.register_buffer(
            "topology_pair_distance",
            distance,
            persistent=False,
        )
        self.register_buffer(
            "topology_pair_unit_i_to_j",
            pair_unit_i_to_j,
            persistent=False,
        )
        self.register_buffer(
            "topology_pair_unit_j_to_i",
            -pair_unit_i_to_j,
            persistent=False,
        )

    def _topology_bias_pairwise_skeleton(self, skeleton_windows, connectivity_windows):
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
        skeleton_pair = skeleton_windows * skeleton_windows.transpose(1, 2)
        return (
            skeleton_pair
            * 0.5
            * (conn_forward + conn_backward)
            * self.topology_distance_decay.unsqueeze(0)
        )

    @staticmethod
    def _compute_gap_query_maps(
        skeleton,
        connectivity,
        roadness=None,
        formula=None,
    ):
        formula = formula or DEFAULT_GAP_QUERY_FORMULA
        if formula.g_mode == "roadness_gap":
            if roadness is None:
                raise ValueError("roadness_gap requires roadness logits")
            topk = min(2, connectivity.shape[-1])
            conn_strength = connectivity.topk(k=topk, dim=-1).values.mean(
                dim=-1,
                keepdim=True,
            )
            topo_conf = skeleton * conn_strength
            H_topo = topo_conf * torch.sigmoid((topo_conf - 0.15) / 0.05)
            H_nchw = H_topo.permute(0, 3, 1, 2).contiguous()
            near_H = F.max_pool2d(
                H_nchw,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            R = torch.sigmoid(roadness).permute(0, 3, 1, 2).contiguous()
            weak_road = torch.sigmoid((R - 0.03) / 0.03) * torch.sigmoid(
                (0.85 - R) / 0.12
            )
            lowH = torch.sigmoid((0.35 - H_nchw) / 0.05)
            G_nchw = near_H * lowH * weak_road
            return H_nchw, G_nchw

        topk = min(2, connectivity.shape[-1])
        conn_strength = connectivity.topk(k=topk, dim=-1).values.mean(
            dim=-1,
            keepdim=True,
        )
        topo_conf = skeleton * conn_strength
        if formula.topo_conf_threshold > 0:
            topo_conf = topo_conf * (
                topo_conf >= formula.topo_conf_threshold
            ).float()
        if roadness is None:
            H = topo_conf
        else:
            H = 1.0 - (1.0 - topo_conf) * (1.0 - roadness)
        if formula.h_threshold > 0:
            H = H * (H >= formula.h_threshold).float()
        H_nchw = H.permute(0, 3, 1, 2).contiguous()
        kernel = max(1, int(formula.pool_kernel))
        padding = kernel // 2
        pooled = F.max_pool2d(
            H_nchw,
            kernel_size=kernel,
            stride=1,
            padding=padding,
        )
        if formula.g_mode == "pool_gap":
            G_nchw = pooled * (1.0 - H_nchw)
        elif formula.g_mode == "local_gap":
            G_nchw = (1.0 - H_nchw) * (pooled - H_nchw).clamp(min=0.0)
        elif formula.g_mode == "relu_gap":
            G_nchw = (pooled - H_nchw).clamp(min=0.0)
        elif formula.g_mode == "pool_gap_lowH":
            G_nchw = (
                pooled
                * (1.0 - H_nchw)
                * (H_nchw < formula.g_h_max).float()
            )
        else:
            raise ValueError(f"Unknown gap-query g_mode: {formula.g_mode}")
        return H_nchw, G_nchw

    def _topology_bias_gap_query(
        self,
        H_windows,
        G_windows,
        connectivity_windows,
        formula=None,
    ):
        formula = formula or self.gap_query_formula
        H = H_windows.squeeze(-1)
        G = G_windows.squeeze(-1)
        if formula.h_key_threshold > 0:
            H = H * (H >= formula.h_key_threshold).float()
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
        if formula.directional_mode == "mean":
            directional = 0.5 * (conn_forward + conn_backward)
        else:
            directional = torch.maximum(conn_forward, conn_backward)
        decay = self.topology_distance_decay.unsqueeze(0)
        return G.unsqueeze(-1) * H.unsqueeze(-2) * directional * decay

    def _build_topology_bias(
        self,
        skeleton,
        connectivity,
        roadness=None,
    ):
        skeleton_windows = window_partition(
            skeleton,
            self.window_size,
        ).view(-1, self.window_size * self.window_size, 1)
        connectivity_windows = window_partition(
            connectivity,
            self.window_size,
        ).view(-1, self.window_size * self.window_size, 8)
        if self.topology_bias_mode == "gap_query":
            H_map, G_map = self._compute_gap_query_maps(
                skeleton,
                connectivity,
                roadness=roadness,
                formula=self.gap_query_formula,
            )
            H_windows = window_partition(
                H_map.permute(0, 2, 3, 1).contiguous(),
                self.window_size,
            ).view(-1, self.window_size * self.window_size, 1)
            G_windows = window_partition(
                G_map.permute(0, 2, 3, 1).contiguous(),
                self.window_size,
            ).view(-1, self.window_size * self.window_size, 1)
            return self._topology_bias_gap_query(
                H_windows,
                G_windows,
                connectivity_windows,
                formula=self.gap_query_formula,
            )
        return self._topology_bias_pairwise_skeleton(
            skeleton_windows,
            connectivity_windows,
        )

    def _road_prior_to_pair_bias(self, road_prior):
        H, W = self.input_resolution
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
        skeleton_prob=None,
        connectivity_prob=None,
        topology_alpha=None,
        roadness_prob=None,
        road_prior=None,
        decoder_skeleton_prob=None,
        decoder_connectivity_prob=None,
        decoder_direction_prob=None,
    ):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        topology_bias = None
        use_qk_relative = self.topology_bias_mode == "gap_query"
        if skeleton_prob is not None or connectivity_prob is not None:
            if skeleton_prob is None or connectivity_prob is None:
                raise ValueError(
                    "skeleton_prob and connectivity_prob must be provided together"
                )
            if not use_qk_relative and topology_alpha is None:
                raise ValueError(
                    "topology_alpha is required for pairwise_skeleton topology bias"
                )
            skeleton = torch.sigmoid(skeleton_prob).detach().permute(
                0, 2, 3, 1
            ).contiguous()
            connectivity = torch.sigmoid(connectivity_prob).detach().permute(
                0, 2, 3, 1
            ).contiguous()
            roadness = None
            if roadness_prob is not None:
                roadness = roadness_prob.permute(0, 2, 3, 1).contiguous()
            if self.shift_size > 0:
                shifts = (-self.shift_size, -self.shift_size)
                skeleton = torch.roll(skeleton, shifts=shifts, dims=(1, 2))
                connectivity = torch.roll(
                    connectivity,
                    shifts=shifts,
                    dims=(1, 2),
                )
                if roadness is not None:
                    roadness = torch.roll(roadness, shifts=shifts, dims=(1, 2))
            topology_bias = self._build_topology_bias(
                skeleton,
                connectivity,
                roadness=roadness,
            )

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.capture_topology_diagnostics and topology_bias is not None:
            self.attn.capture_logits = True
        road_attention_bias = self._build_road_attention_bias(road_prior)
        structure_attention_bias = self._build_decoder_structure_attention_bias(
            decoder_skeleton_prob,
            decoder_connectivity_prob,
            decoder_direction_prob,
        )
        attn_windows = self.attn(
            x_windows,
            mask=self.attn_mask,
            topology_bias=topology_bias,
            road_attention_bias=road_attention_bias,
            structure_attention_bias=structure_attention_bias,
            topology_alpha=topology_alpha,
            use_qk_relative_topology=use_qk_relative,
            topology_ratio=self.topology_ratio,
            topo_clip=self.topology_topo_clip,
        )
        if self.capture_topology_diagnostics and topology_bias is not None:
            self.attn.capture_logits = False
        if self.capture_topology_diagnostics and topology_bias is not None:
            with torch.no_grad():
                if use_qk_relative:
                    qk_scale = float(getattr(self.attn, "last_qk_scale", 0.0))
                    topo_scale = float(getattr(self.attn, "last_topo_scale", 0.0))
                    active_pair_ratio = float(
                        getattr(self.attn, "last_active_pair_ratio", 0.0)
                    )
                    topology_term = (
                        float(self.topology_ratio)
                        * qk_scale
                        * (topology_bias / (topo_scale + TOPOLOGY_QK_RELATIVE_EPS))
                        .clamp(0.0, float(self.topology_topo_clip))
                    )
                else:
                    qk_scale = float(
                        getattr(self.attn, "last_qk_logits_abs_mean", 0.0)
                    )
                    topo_scale = float(topology_bias[topology_bias > 0].mean().item()) if (
                        topology_bias > 0
                    ).any() else float(topology_bias.abs().mean().item())
                    active_pair_ratio = float(
                        (topology_bias > TOPOLOGY_QK_RELATIVE_EPS).float().mean().item()
                    )
                    topology_term = topology_alpha * topology_bias
                baseline_windows = self.attn(
                    x_windows,
                    mask=self.attn_mask,
                    topology_bias=topology_bias,
                    topology_alpha=(
                        topology_alpha.new_zeros(())
                        if topology_alpha is not None
                        else None
                    ),
                    use_qk_relative_topology=use_qk_relative,
                    topology_ratio=0.0 if use_qk_relative else self.topology_ratio,
                    topo_clip=self.topology_topo_clip,
                )
                attn_diff = (attn_windows - baseline_windows).abs().mean()
                attn_rel = attn_diff / (baseline_windows.abs().mean() + 1e-6)

                def _attn_to_tokens(attn_out):
                    attn_out = attn_out.view(
                        -1,
                        self.window_size,
                        self.window_size,
                        C,
                    )
                    shifted = window_reverse(
                        attn_out,
                        self.window_size,
                        H,
                        W,
                    )
                    if self.shift_size > 0:
                        shifted = torch.roll(
                            shifted,
                            shifts=(self.shift_size, self.shift_size),
                            dims=(1, 2),
                        )
                    return shifted.view(B, H * W, C)

                tokens_with = shortcut + self.drop_path(_attn_to_tokens(attn_windows))
                tokens_baseline = shortcut + self.drop_path(
                    _attn_to_tokens(baseline_windows)
                )
                stage_with = tokens_with + self.drop_path(
                    self.mlp(self.norm2(tokens_with))
                )
                stage_baseline = tokens_baseline + self.drop_path(
                    self.mlp(self.norm2(tokens_baseline))
                )
                stage_delta_rel = (
                    (stage_with - stage_baseline).abs().mean()
                    / (stage_baseline.abs().mean() + 1e-6)
                )
                conn_strength = connectivity_prob.topk(
                    k=min(2, connectivity_prob.shape[1]),
                    dim=1,
                ).values.mean(dim=1)
                self.last_topology_diagnostics = {
                    "topology_bias_mode": self.topology_bias_mode,
                    "topology_ratio": float(self.topology_ratio),
                    "alpha_eff": float(topology_alpha.detach().cpu()) if topology_alpha is not None else 0.0,
                    "qk_scale": qk_scale,
                    "topo_scale": topo_scale,
                    "active_pair_ratio": active_pair_ratio,
                    "topology_bias_mean": float(topology_bias.mean().detach().cpu()),
                    "topology_bias_max": float(topology_bias.max().detach().cpu()),
                    "topology_bias_min": float(topology_bias.min().detach().cpu()),
                    "topology_term_mean": float(topology_term.mean().detach().cpu()),
                    "topology_term_max": float(topology_term.max().detach().cpu()),
                    "topology_term_min": float(topology_term.min().detach().cpu()),
                    "qk_logits_abs_mean": float(
                        getattr(
                            self.attn,
                            "last_qk_logits_abs_mean",
                            0.0,
                        )
                    ),
                    "topology_term_abs_mean": float(
                        getattr(
                            self.attn,
                            "last_topology_term_abs_mean",
                            0.0,
                        )
                    ),
                    "topology_term_over_qk_ratio": float(
                        getattr(
                            self.attn,
                            "last_topology_term_over_qk_ratio",
                            0.0,
                        )
                    ),
                    "attention_output_diff": float(attn_diff.detach().cpu()),
                    "attention_relative_diff": float(attn_rel.detach().cpu()),
                    "stage_output_delta_relative_norm": float(
                        stage_delta_rel.detach().cpu()
                    ),
                    "skeleton_prob_mean": float(skeleton_prob.mean().detach().cpu()),
                    "skeleton_prob_max": float(skeleton_prob.max().detach().cpu()),
                    "skeleton_prob_min": float(skeleton_prob.min().detach().cpu()),
                    "conn_strength_mean": float(conn_strength.mean().detach().cpu()),
                    "conn_strength_max": float(conn_strength.max().detach().cpu()),
                    "conn_strength_min": float(conn_strength.min().detach().cpu()),
                }

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

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

    def forward(
        self,
        x,
        skeleton_prob=None,
        connectivity_prob=None,
        topology_alpha=None,
        roadness_prob=None,
        topology_gate=None,
        decoder_skeleton_prob=None,
        decoder_connectivity_prob=None,
        decoder_direction_prob=None,
    ):
        for blk in self.blocks:
            if self.use_checkpoint:
                if skeleton_prob is None:
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
                    x = checkpoint.checkpoint(
                        lambda feature, skeleton, connectivity, alpha, roadness, dec_skeleton, dec_connectivity, dec_direction: blk(
                            feature,
                            skeleton,
                            connectivity,
                            alpha,
                            roadness,
                            decoder_skeleton_prob=dec_skeleton,
                            decoder_connectivity_prob=dec_connectivity,
                            decoder_direction_prob=dec_direction,
                        ),
                        x,
                        skeleton_prob,
                        connectivity_prob,
                        topology_alpha,
                        roadness_prob,
                        decoder_skeleton_prob,
                        decoder_connectivity_prob,
                        decoder_direction_prob,
                    )
            else:
                x = blk(
                    x,
                    skeleton_prob,
                    connectivity_prob,
                    topology_alpha,
                    roadness_prob,
                    decoder_skeleton_prob=decoder_skeleton_prob,
                    decoder_connectivity_prob=decoder_connectivity_prob,
                    decoder_direction_prob=decoder_direction_prob,
                )
        if topology_gate is not None:
            if skeleton_prob is None or connectivity_prob is None:
                raise ValueError("Topology gate requires skeleton/connectivity probabilities.")
            x = topology_gate(
                x,
                skeleton_prob,
                connectivity_prob,
            )
        if self.upsample is not None:
            x = self.upsample(x)
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

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=7, stride=patch_size, padding=3),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        H, W = self.img_size
        flops = H * W * self.embed_dim * self.in_chans * 3 * 3
        flops += H * W * self.embed_dim * self.embed_dim * 3 * 3
        flops += Ho * Wo * self.embed_dim * self.embed_dim * 7 * 7
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


class _UnusedLegacyTopologyAwareSwinBlock(nn.Module):
    """Legacy extra-block implementation; retained only for checkpoint archaeology."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        topology_alpha_max=0.20,
        topology_alpha_init=0.02,
        topology_alpha_trainable=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = min(window_size, min(input_resolution))
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.topology_alpha_max = float(topology_alpha_max)
        alpha_ratio = float(topology_alpha_init) / self.topology_alpha_max
        if not 0.0 < alpha_ratio < 1.0:
            raise ValueError("topology_alpha_init must be between 0 and topology_alpha_max")
        raw_alpha_init = torch.logit(torch.tensor(alpha_ratio))
        # Keep the state-dict key for checkpoint compatibility; this stores raw alpha.
        self.topology_alpha = nn.Parameter(
            raw_alpha_init,
            requires_grad=bool(topology_alpha_trainable),
        )
        self.capture_topology_diagnostics = False
        self.last_topology_diagnostics = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

        self._register_topology_buffers()
        self._register_attention_mask()

    def _register_topology_buffers(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"),
            dim=-1,
        ).view(-1, 2)
        delta = coords.unsqueeze(0) - coords.unsqueeze(1)
        dy = delta[..., 0]
        dx = delta[..., 1]

        direction = torch.zeros_like(dy, dtype=torch.long)
        direction[(dy > 0) & (dx == 0)] = 1
        direction[(dy == 0) & (dx < 0)] = 2
        direction[(dy == 0) & (dx > 0)] = 3
        direction[(dy < 0) & (dx < 0)] = 4
        direction[(dy < 0) & (dx > 0)] = 5
        direction[(dy > 0) & (dx < 0)] = 6
        direction[(dy > 0) & (dx > 0)] = 7

        opposite = torch.tensor([1, 0, 3, 2, 7, 6, 5, 4], dtype=torch.long)
        direction_one_hot = F.one_hot(direction, num_classes=8).float()
        opposite_one_hot = F.one_hot(opposite[direction], num_classes=8).float()
        distance = torch.maximum(dy.abs(), dx.abs()).float()
        distance_decay = 1.0 / (1.0 + distance)
        distance_decay[distance == 0] = 0.0

        self.register_buffer("direction_one_hot", direction_one_hot)
        self.register_buffer("opposite_direction_one_hot", opposite_one_hot)
        self.register_buffer("topology_distance_decay", distance_decay)

    def _register_attention_mask(self):
        if self.shift_size == 0:
            self.register_buffer("attn_mask", None)
            return

        height, width = self.input_resolution
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size).view(
            -1,
            self.window_size * self.window_size,
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0,
            float(-100.0),
        ).masked_fill(attn_mask == 0, 0.0)
        self.register_buffer("attn_mask", attn_mask)

    def _topology_bias(self, skeleton_windows, connectivity_windows):
        conn_forward = torch.einsum(
            "bik,ijk->bij",
            connectivity_windows,
            self.direction_one_hot,
        )
        conn_backward = torch.einsum(
            "bjk,ijk->bij",
            connectivity_windows,
            self.opposite_direction_one_hot,
        )
        skeleton_pair = skeleton_windows * skeleton_windows.transpose(1, 2)
        return (
            skeleton_pair
            * 0.5
            * (conn_forward + conn_backward)
            * self.topology_distance_decay.unsqueeze(0)
        )

    def effective_topology_alpha(self):
        return self.topology_alpha_max * torch.sigmoid(self.topology_alpha)

    def _prepare_topology_windows(self, x, skeleton, connectivity):
        if self.shift_size > 0:
            shifts = (-self.shift_size, -self.shift_size)
            x = torch.roll(x, shifts=shifts, dims=(1, 2))
            skeleton = torch.roll(skeleton, shifts=shifts, dims=(1, 2))
            connectivity = torch.roll(connectivity, shifts=shifts, dims=(1, 2))

        channels = x.shape[-1]
        x_windows = window_partition(x, self.window_size).view(
            -1,
            self.window_size * self.window_size,
            channels,
        )
        skeleton_windows = window_partition(skeleton, self.window_size).view(
            -1,
            self.window_size * self.window_size,
            1,
        )
        connectivity_windows = window_partition(connectivity, self.window_size).view(
            -1,
            self.window_size * self.window_size,
            8,
        )
        return x_windows, skeleton_windows, connectivity_windows

    def forward(self, x, skeleton_prob, connectivity_prob):
        height, width = self.input_resolution
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("topology-aware block input feature has wrong size")

        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        skeleton = skeleton_prob.permute(0, 2, 3, 1).contiguous()
        connectivity = connectivity_prob.permute(0, 2, 3, 1).contiguous()

        x_windows, skeleton_windows, connectivity_windows = (
            self._prepare_topology_windows(
                x,
                skeleton,
                connectivity,
            )
        )
        topology_bias = self._topology_bias(
            skeleton_windows,
            connectivity_windows,
        )
        alpha_eff = self.effective_topology_alpha()
        attended = self.attn(
            x_windows,
            mask=self.attn_mask,
            topology_bias=topology_bias,
            topology_alpha=alpha_eff,
        )
        if self.capture_topology_diagnostics:
            with torch.no_grad():
                topology_term = alpha_eff * topology_bias
                baseline_attended = self.attn(
                    x_windows,
                    mask=self.attn_mask,
                    topology_bias=topology_bias,
                    topology_alpha=alpha_eff.new_zeros(()),
                )
                output_diff = (attended - baseline_attended).abs().mean()
                relative_diff = output_diff / (
                    baseline_attended.abs().mean() + 1e-6
                )
                conn_strength = connectivity_prob.topk(
                    k=min(2, connectivity_prob.shape[1]),
                    dim=1,
                ).values.mean(dim=1)
                self.last_topology_diagnostics = {
                    "alpha_eff": float(alpha_eff.detach().cpu()),
                    "topology_bias_mean": float(topology_bias.mean().detach().cpu()),
                    "topology_bias_max": float(topology_bias.max().detach().cpu()),
                    "topology_bias_min": float(topology_bias.min().detach().cpu()),
                    "topology_term_mean": float(topology_term.mean().detach().cpu()),
                    "topology_term_max": float(topology_term.max().detach().cpu()),
                    "topology_term_min": float(topology_term.min().detach().cpu()),
                    "attention_output_diff": float(output_diff.detach().cpu()),
                    "attention_relative_diff": float(relative_diff.detach().cpu()),
                    "skeleton_prob_mean": float(skeleton_prob.mean().detach().cpu()),
                    "skeleton_prob_max": float(skeleton_prob.max().detach().cpu()),
                    "skeleton_prob_min": float(skeleton_prob.min().detach().cpu()),
                    "conn_strength_mean": float(conn_strength.mean().detach().cpu()),
                    "conn_strength_max": float(conn_strength.max().detach().cpu()),
                    "conn_strength_min": float(conn_strength.min().detach().cpu()),
                }
        attended = attended.view(
            -1,
            self.window_size,
            self.window_size,
            channels,
        )
        x = window_reverse(attended, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )

        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", return_skeleton=False,
                 bottleneck_type="global_local", final_topology_eta_init=0.005,
                 final_gap_rho_init=0.005,
                 stage_topology_stages="none",
                 stage_topology_alpha_max=1.0,
                 stage_topology_alpha_init=0.1,
                 stage_topology_bias_mode="pairwise_skeleton",
                 stage_topology_ratio=DEFAULT_STAGE_TOPOLOGY_RATIO,
                 stage_topology_topo_clip=DEFAULT_STAGE_TOPOLOGY_TOPO_CLIP,
                 structure_profile="full",
                 enable_final_graph_prop=False,
                 **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

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
        self.stage_topology_stages = stage_topology_stages.lower()
        if self.stage_topology_stages not in {"none", "stage3", "stage23"}:
            raise ValueError(
                "stage_topology_stages must be one of: none, stage3, stage23"
            )
        self.stage_topology_bias_mode = stage_topology_bias_mode.lower()
        if self.stage_topology_bias_mode not in {"pairwise_skeleton", "gap_query"}:
            raise ValueError(
                "stage_topology_bias_mode must be pairwise_skeleton or gap_query"
            )
        self.stage_topology_ratio = float(stage_topology_ratio)
        self.stage_topology_topo_clip = float(stage_topology_topo_clip)
        self.structure_profile = structure_profile.lower()
        if self.structure_profile not in {"full", "stage23_boundary_0626"}:
            raise ValueError(
                "structure_profile must be one of: full, stage23_boundary_0626"
            )
        self.use_stage3_global_context = (
            self.structure_profile == "stage23_boundary_0626"
        )
        self.enable_final_graph_prop = bool(enable_final_graph_prop)

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
            "Stage3 attention bias with lambda_init=(0,0); "
            "A1/A2 -> residual PatchMerging priors".format(
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
                                         use_decoder_structure_bias=(i_layer in (2, 3)))
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

        # ===== 仅在 Layer 1、2 上添加 MSFE + DCA-FPN 模块 (inx=2,3) =====
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

        self.msce_blocks = nn.ModuleList()
        self.dca_blocks = nn.ModuleList()
        
        for skip_idx in range(2, self.num_layers):  # inx=2,3: Layer2 和 Layer1
            # 通道数: skip_idx=2 → 384 (Layer2), skip_idx=3 → 192 (Layer1)
            skip_channels = int(embed_dim * 2 ** (self.num_layers - 1 - skip_idx))
            
            # MSFE 模块（多尺度特征增强）
            msce_block = MSFEBlock(channel=skip_channels)
            self.msce_blocks.append(msce_block)
            
            # DCA-FPN-Lite 模块（轻量级可变形交叉注意）
            dca_block = DCAFPNLite(
                channels=skip_channels,
                num_heads=4,
                num_points=4,
                max_offset=0.20
            )
            self.dca_blocks.append(dca_block)
            
            layer_name = "Layer 2" if skip_idx == 2 else "Layer 1"
            print(f"[INFO] MSFE Block {layer_name} (inx={skip_idx}): {skip_channels} channels")
            print(f"[INFO] DCA-FPN-Lite {layer_name} (inx={skip_idx}): {skip_channels} channels")

        decoder_structure_channels = (
            embed_dim * 4,
            embed_dim * 2,
            embed_dim,
            embed_dim,
        )
        if self.use_stage3_global_context:
            self.global_context_head = GlobalContextHead(
                in_channels=self.num_features,
                hidden_channels=128,
                out_channels=STAGE3_GLOBAL_CONTEXT_CHANNELS,
            )
        else:
            self.global_context_head = None

        self.decoder_structure_blocks = nn.ModuleList(
            [
                DecoderStructureRefinement(
                    channels=channels,
                    connectivity_channels=(24 if stage_index == 3 else 8),
                    enable_roadness_head=(
                        stage_index == 3
                        and self.structure_profile != "stage23_boundary_0626"
                    ),
                    context_channels=(
                        STAGE3_GLOBAL_CONTEXT_CHANNELS
                        if (
                            self.use_stage3_global_context
                            and stage_index == 3
                        )
                        else None
                    ),
                    enable_direct_feature_refinement=True,
                    enable_directional_feature_refinement=False,
                )
                for stage_index, channels in enumerate(decoder_structure_channels)
            ]
        )
        if self.structure_profile == "stage23_boundary_0626":
            print(
                "[INFO] Decoder structure gates: stage2/stage3 only "
                "(0626 profile), channels={}".format(
                    decoder_structure_channels
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
        self.stage2_topology_source = DecoderStructureRefinement(
            channels=embed_dim * 2,
            enable_direct_feature_refinement=False,
        )
        self.stage_topology_scales = nn.ModuleDict(
            {
                str(stage): TopologyAttentionScale(
                    alpha_max=stage_topology_alpha_max,
                    alpha_init=stage_topology_alpha_init,
                    trainable=True,
                )
                for stage in (2, 3)
            }
        )
        print(
            "[INFO] Native decoder topology attention: stages={}, "
            "bias_mode={}, ratio={}, topo_clip={}, "
            "alpha_init={}, alpha_max={}".format(
                self.stage_topology_stages,
                self.stage_topology_bias_mode,
                self.stage_topology_ratio,
                self.stage_topology_topo_clip,
                stage_topology_alpha_init,
                stage_topology_alpha_max,
            )
        )
        self._apply_stage_topology_config()

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            if self.return_skeleton:
                self.guided_head = SkeletonGuidedHead(
                    in_channels=embed_dim,
                    hidden_channels=max(embed_dim // 2, 32),
                    init_alpha=0.0,
                    topology_eta_init=final_topology_eta_init,
                    gap_rho_init=final_gap_rho_init,
                    enable_final_structure=(
                        self.structure_profile != "stage23_boundary_0626"
                    ),
                    enable_graph_prop=self.enable_final_graph_prop,
                )
                if self.structure_profile == "stage23_boundary_0626":
                    print(
                        "[INFO] Final skeleton: shared-feature S0 + r2 bridge -> S1; "
                        "only ReLU(S1-S0) guides final surface"
                    )
                    if self.enable_final_graph_prop:
                        print(
                            "[INFO] Final soft skeleton graph propagation: enabled "
                            "(stage2/3 priors -> surface feature, lambda_init=0.05, "
                            "lambda_max=0.20)"
                        )
            else:
                self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

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

    def _stage_topology_enabled(self, stage):
        if self.stage_topology_stages == "stage23":
            return stage in (2, 3)
        if self.stage_topology_stages == "stage3":
            return stage == 3
        return False

    def _decoder_structure_enabled(self, stage):
        if self.structure_profile == "stage23_boundary_0626":
            return stage in (2, 3)
        return True

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

    @staticmethod
    def _fuse_stage_structure_to_fullres(
        structure_outputs,
        target_hw,
        stages=(2, 3),
    ):
        skeletons = []
        connectivities = []
        stage_set = set(stages)
        for item in structure_outputs:
            if item["stage"] not in stage_set:
                continue
            skeletons.append(
                F.interpolate(
                    item["skeleton"],
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            connectivities.append(
                F.interpolate(
                    item["connectivity"],
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        if not skeletons:
            raise ValueError(
                "Graph propagation requires stage structure outputs "
                f"for stages {tuple(stages)}."
            )
        skeleton_logits = torch.stack(skeletons, dim=0).max(dim=0).values
        connectivity_logits = torch.stack(connectivities, dim=0).max(dim=0).values
        return skeleton_logits, connectivity_logits

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

    def _run_decoder_structure_block(
        self,
        feature_map,
        stage,
        bottleneck_tokens,
        block_stage=None,
        apply_feature_refinement=True,
    ):
        if not self._decoder_structure_enabled(stage):
            return feature_map, *self._placeholder_structure_outputs(feature_map)

        block_stage = stage if block_stage is None else block_stage
        global_context = None
        if self.use_stage3_global_context and stage == 3:
            global_context = self._build_stage3_global_context(
                bottleneck_tokens,
                feature_map.shape[-2:],
            )
        return self.decoder_structure_blocks[block_stage](
            feature_map,
            global_context=global_context,
            apply_feature_refinement=apply_feature_refinement,
        )

    def _apply_stage_topology_config(self):
        for inx, layer_up in enumerate(self.layers_up):
            if not isinstance(layer_up, BasicLayer_up):
                continue
            if not self._stage_topology_enabled(inx):
                continue
            use_gap_query = (
                self.stage_topology_bias_mode == "gap_query" and inx == 3
            )
            for blk in layer_up.blocks:
                if use_gap_query:
                    blk.topology_bias_mode = "gap_query"
                    blk.topology_ratio = self.stage_topology_ratio
                    blk.topology_topo_clip = self.stage_topology_topo_clip
                    blk.gap_query_formula = GapQueryFormulaConfig(
                        g_mode="roadness_gap",
                        pool_kernel=5,
                    )
                else:
                    blk.topology_bias_mode = "pairwise_skeleton"

    @staticmethod
    def _mix_teacher_topology(
        predicted_skeleton,
        predicted_connectivity,
        gt_skeleton,
        teacher_forcing_ratio,
    ):
        predicted_skeleton = torch.sigmoid(predicted_skeleton).detach()
        predicted_connectivity = torch.sigmoid(predicted_connectivity).detach()[:, :8]
        ratio = float(teacher_forcing_ratio)
        if ratio <= 0.0:
            return predicted_skeleton, predicted_connectivity
        if gt_skeleton is None:
            raise ValueError(
                "gt_skeleton is required when teacher_forcing_ratio > 0"
            )

        gt_stage = F.interpolate(
            (gt_skeleton > 0.5).to(dtype=predicted_skeleton.dtype),
            size=predicted_skeleton.shape[-2:],
            mode="nearest",
        )
        gt_connectivity = build_connectivity_target(
            gt_stage,
            erode_kernel_size=1,
        ).to(
            device=predicted_connectivity.device,
            dtype=predicted_connectivity.dtype,
        )
        skeleton_used = ratio * gt_stage + (1.0 - ratio) * predicted_skeleton
        connectivity_used = (
            ratio * gt_connectivity
            + (1.0 - ratio) * predicted_connectivity
        )
        return skeleton_used.detach(), connectivity_used.detach()

    # Decoder and skip connection with DCA-FPN and MSFE.
    def forward_up_features(
        self,
        x,
        x_downsample,
        bottleneck_tokens=None,
        gt_skeleton=None,
        topology_alpha_scale=1.0,
        teacher_forcing_ratio=0.0,
    ):
        """
        Decoder with enhanced skip connection fusion using MSFE + DCA-FPN-Lite
        仅在 Layer 1、2 (inx=2,3) 上应用 MSFE + DCA-FPN
        
        Args:
            x: bottleneck feature [B, L, C]
            x_downsample: list of encoder skip features [[B, L, C], ...]
        
        Returns:
            decoder output [B, L, C]
        """
        structure_outputs = []
        if bottleneck_tokens is None:
            bottleneck_tokens = x
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # Bottleneck layer, no skip connection
                pass
            elif inx == 1:
                # inx=1 (Layer 3 skip, 1/16分辨率): 直接 concat，不加 MSFE + DCA
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
            else:
                # inx=2,3 (Layer 2,1 skip): 应用 MSFE + DCA-FPN
                skip = x_downsample[3 - inx]  # [B, L, C]
                
                # 计算当前层的空间分辨率
                H = self.patches_resolution[0] // (2 ** (3 - inx))
                W = self.patches_resolution[1] // (2 ** (3 - inx))
                
                # 1. Token → Feature Map 转换
                skip_map = token_to_map(skip, H, W)  # [B, C, H, W]
                x_map = token_to_map(x, H, W)  # [B, C, H, W]
                
                # 2. MSFE 增强 skip feature
                # msce_blocks[0] for inx=2 (Layer2), msce_blocks[1] for inx=3 (Layer1)
                block_idx = inx - 2
                skip_msce = self.msce_blocks[block_idx](skip_map)  # [B, C, H, W]
                
                # 3. DCA-FPN 用 decoder feature 精化 skip
                skip_refined = self.dca_blocks[block_idx](deep=x_map, shallow=skip_msce)  # [B, C, H, W]
                
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
            topology_enabled = self._stage_topology_enabled(inx)
            if decoder_structure_gate_enabled:
                input_height, input_width = layer_up.input_resolution
                x_map = token_to_map(x, input_height, input_width)
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
                    block_stage=1 if inx == 2 else inx,
                    apply_feature_refinement=False,
                )
                skeleton_used, connectivity_used = self._mix_teacher_topology(
                    skeleton_0,
                    connectivity_0,
                    gt_skeleton,
                    teacher_forcing_ratio,
                )
                x = layer_up(
                    x,
                    decoder_skeleton_prob=skeleton_used,
                    decoder_connectivity_prob=connectivity_used,
                    decoder_direction_prob=direction_0,
                )
                structure_outputs.append(
                    {
                        "stage": inx,
                        "refinement_step": 0,
                        "stage_loss_scale": 0.5,
                        "skeleton": skeleton_0,
                        "connectivity": connectivity_0,
                        "direction": direction_0,
                        "structure_gate": structure_gate_0,
                        "roadness": roadness_0,
                    }
                )

                output_scale = 2 ** max(2 - inx, 0)
                output_height = self.patches_resolution[0] // output_scale
                output_width = self.patches_resolution[1] // output_scale
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
                )
                x = map_to_token(x_map)
                structure_outputs.append(
                    {
                        "stage": inx,
                        "refinement_step": 1,
                        "stage_loss_scale": 1.0,
                        "skeleton": skeleton_i,
                        "connectivity": connectivity_i,
                        "direction": direction_i,
                        "structure_gate": structure_gate_i,
                        "roadness": roadness_i,
                    }
                )
                continue
            elif topology_enabled:
                topology_source = (
                    self.stage2_topology_source
                    if inx == 2
                    else self.decoder_structure_blocks[inx]
                )
                input_height, input_width = layer_up.input_resolution
                x_map = token_to_map(x, input_height, input_width)
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
                )
                x = map_to_token(x_map)
                skeleton_used, connectivity_used = self._mix_teacher_topology(
                    skeleton_i,
                    connectivity_i,
                    gt_skeleton,
                    teacher_forcing_ratio,
                )
                use_gap_query = (
                    self.stage_topology_bias_mode == "gap_query" and inx == 3
                )
                topology_alpha = None
                if not use_gap_query:
                    topology_alpha = (
                        self.stage_topology_scales[str(inx)]
                        .effective_topology_alpha()
                        * float(topology_alpha_scale)
                    )
                roadness_prob = roadness_i if use_gap_query else None
                x = layer_up(
                    x,
                    skeleton_prob=skeleton_used,
                    connectivity_prob=connectivity_used,
                    topology_alpha=topology_alpha,
                    roadness_prob=roadness_prob,
                )
            else:
                x = layer_up(x)
                output_scale = 2 ** max(2 - inx, 0)
                output_height = self.patches_resolution[0] // output_scale
                output_width = self.patches_resolution[1] // output_scale
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
                )
                x = map_to_token(x_map)
            structure_outputs.append(
                {
                    "stage": inx,
                    "skeleton": skeleton_i,
                    "connectivity": connectivity_i,
                    "direction": direction_i,
                    "structure_gate": structure_gate_i,
                    "roadness": roadness_i,
                }
            )

        x = self.norm_up(x)  # B L C

        return x, structure_outputs

    def up_x4(self, x, structure_outputs=None):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            if self.return_skeleton:
                stage_skeleton_logits = None
                stage_connectivity_logits = None
                if self.enable_final_graph_prop and structure_outputs:
                    target_hw = (x.shape[2], x.shape[3])
                    (
                        stage_skeleton_logits,
                        stage_connectivity_logits,
                    ) = self._fuse_stage_structure_to_fullres(
                        structure_outputs,
                        target_hw,
                        stages=(2, 3),
                    )
                x = self.guided_head(
                    x,
                    stage_skeleton_logits=stage_skeleton_logits,
                    stage_connectivity_logits=stage_connectivity_logits,
                )
            else:
                x = self.output(x)

        return x

    def forward(
        self,
        x,
        gt_skeleton=None,
        topology_alpha_scale=1.0,
        teacher_forcing_ratio=0.0,
    ):
        x, x_downsample, road_attentions = self.forward_features(x)
        x, structure_outputs = self.forward_up_features(
            x,
            x_downsample,
            bottleneck_tokens=x,
            gt_skeleton=gt_skeleton,
            topology_alpha_scale=topology_alpha_scale,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        if self.return_skeleton and road_attentions:
            structure_outputs.extend(road_attentions)
        x = self.up_x4(
            x,
            structure_outputs=structure_outputs if self.return_skeleton else None,
        )
        if self.return_skeleton and isinstance(x, tuple):
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
