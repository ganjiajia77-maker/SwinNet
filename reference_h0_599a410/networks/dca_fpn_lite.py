import torch
import torch.nn as nn
import torch.nn.functional as F


class DCAFPNLite(nn.Module):
    """
    轻量级 Deformable Cross-Attention FPN
    用于动态精化 skip connection
    """
    def __init__(self, channels, num_heads=4, num_points=4, max_offset=0.2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.max_offset = max_offset
        
        # 1x1 Conv 对齐通道
        self.align_deep = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.align_shallow = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
        # 从 deep 预测采样偏移 [K, 2]
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_points * 2, kernel_size=1)
        )
        
        # 从 deep 预测采样权重 [K]
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_points, kernel_size=1)
        )
        
        # 融合采样结果
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_points, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 输出融合层
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
        # 残差强度
        self.register_parameter('beta', nn.Parameter(torch.tensor(0.1)))

    def forward(self, deep, shallow):
        """
        Args:
            deep: decoder feature [B, C, H, W] - 语义强
            shallow: MSCE-enhanced skip feature [B, C, H, W] - 细节强
        
        Returns:
            refined skip [B, C, H, W]
        """
        B, C, H, W = deep.shape
        device = deep.device
        
        # 对齐通道
        Q = self.align_deep(deep)  # [B, C, H, W]
        V = self.align_shallow(shallow)  # [B, C, H, W]
        
        # 预测采样偏移 [B, K*2, H, W]
        offsets = self.offset_conv(Q)  # [B, K*2, H, W]
        offsets = offsets * self.max_offset  # 限制偏移范围
        
        # 预测采样权重 [B, K, H, W] → 经 softmax 归一化
        weights = self.weight_conv(Q)  # [B, K, H, W]
        weights = torch.softmax(weights, dim=1)  # [B, K, H, W]
        
        # 构造网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, self.num_points, -1, -1, -1)  # [B, K, H, W, 2]
        
        # 添加偏移
        offsets_reshaped = offsets.view(B, self.num_points, 2, H, W).permute(0, 1, 3, 4, 2)  # [B, K, H, W, 2]
        sampled_grid = grid + offsets_reshaped  # [B, K, H, W, 2]
        
        # 从 shallow 中采样
        sampled_features = []
        for k in range(self.num_points):
            feat_k = F.grid_sample(
                V,
                sampled_grid[:, k, :, :, :],  # [B, H, W, 2]
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )  # [B, C, H, W]
            sampled_features.append(feat_k)
        
        # 按权重融合
        fused = torch.zeros_like(V)  # [B, C, H, W]
        for k in range(self.num_points):
            weight_k = weights[:, k:k+1, :, :]  # [B, 1, H, W]
            fused = fused + weight_k * sampled_features[k]  # [B, C, H, W]
        
        # 融合输出
        fused = self.fusion_conv(
            torch.cat(sampled_features, dim=1)  # [B, C*K, H, W]
        )  # [B, C, H, W]
        
        # 最终输出
        out = self.out_conv(fused)  # [B, C, H, W]
        
        # 残差加回原 shallow
        refined = shallow + self.beta * out
        
        return refined
