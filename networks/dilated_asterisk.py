import torch
import torch.nn as nn


class DilatedAsterisk(nn.Module):
    """
    改进的 Asterisk 模块 - 简化版本
    
    四个分支：
    1. 水平条带卷积: (1, 5) - 捕捉横向特征
    2. 垂直条带卷积: (5, 1) - 捕捉纵向特征
    3. 普通局部卷积: 3×3 dilation=1 - 细节特征
    4. 空洞卷积: 3×3 dilation=2 - 中等感受野
    
    特点：
    - 去掉了dilation=3分支，简化模型
    - 可学习的残差权重alpha（初始化为0.1）
    - 输入输出尺寸完全一致 [B, C, H, W]
    - 残差连接: out = x + alpha * fused
    """
    
    def __init__(self, in_channels, out_channels, alpha_init=0.1):
        super(DilatedAsterisk, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Branch 1: 水平条带卷积 (1, 5)
        self.h_strip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2), bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 垂直条带卷积 (5, 1)
        self.v_strip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0), bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 普通卷积 kernel=3, dilation=1
        self.dilated_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 空洞卷积 kernel=3, dilation=2
        self.dilated_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合层：4个分支 concat -> 1x1 conv -> 3x3 conv
        self.fusion_1x1 = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=True)
        self.fusion_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 输出投影
        if out_channels != in_channels:
            self.output_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.output_proj = None
        
        # 可学习的残差权重 - 初始化为alpha_init
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        
        self._init_weight()
    
    def forward(self, x):
        """前向传播"""
        # 4个并行分支
        h_out = self.h_strip(x)      # [B, C, H, W]
        v_out = self.v_strip(x)      # [B, C, H, W]
        d1_out = self.dilated_1(x)   # [B, C, H, W]
        d2_out = self.dilated_2(x)   # [B, C, H, W]
        
        # 拼接4个分支
        concat = torch.cat([h_out, v_out, d1_out, d2_out], dim=1)  # [B, 4*C, H, W]
        
        # 融合
        fused = self.fusion_1x1(concat)  # [B, C, H, W]
        fused = self.fusion_3x3(fused)   # [B, C, H, W]
        
        # 输出投影
        if self.output_proj is not None:
            fused = self.output_proj(fused)  # [B, out_C, H, W]
        
        # 残差连接：out = x + alpha * fused
        out = x + self.alpha * fused
        
        return out
    
    def _init_weight(self):
        """Kaiming 权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DilatedAsteriskWithDirections(DilatedAsterisk):
    """
    保持与 DilatedAsterisk 相同的实现
    （为了向后兼容性）
    """
    pass


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DilatedAsterisk 模块测试")
    print("="*80)
    
    # 测试 1: 768通道 (用于Swin-UNet中间层)
    print("\n[Test 1] 768通道 (Swin-UNet中间层)")
    print("-" * 80)
    model1 = DilatedAsterisk(in_channels=768, out_channels=768)
    x = torch.randn(2, 768, 32, 32)
    y = model1(x)
    print(f"输入:   {x.shape}")
    print(f"输出:   {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model1.parameters()) / 1e6:.2f}M")
    assert y.shape == x.shape, "❌ 形状不匹配"
    print("✓ 通过: 形状一致")
    
    # 测试 2: 2通道 (二分类输出)
    print("\n[Test 2] 2通道 (二分类输出)")
    print("-" * 80)
    model2 = DilatedAsterisk(in_channels=2, out_channels=2)
    x_bin = torch.randn(2, 2, 224, 224)
    y_bin = model2(x_bin)
    print(f"输入:   {x_bin.shape}")
    print(f"输出:   {y_bin.shape}")
    print(f"参数量: {sum(p.numel() for p in model2.parameters()) / 1e6:.3f}M")
    assert y_bin.shape == x_bin.shape, "❌ 形状不匹配"
    print("✓ 通过: 形状一致")
    
    # 测试 3: 梯度流
    print("\n[Test 3] 梯度流测试")
    print("-" * 80)
    model3 = DilatedAsterisk(in_channels=64, out_channels=64)
    x_grad = torch.randn(1, 64, 16, 16, requires_grad=True)
    y_grad = model3(x_grad)
    loss = y_grad.sum()
    loss.backward()
    print(f"输入梯度存在: {x_grad.grad is not None}")
    print(f"梯度形状:    {x_grad.grad.shape if x_grad.grad is not None else 'N/A'}")
    assert x_grad.grad is not None, "❌ 梯度流中断"
    assert x_grad.grad.shape == x_grad.shape, "❌ 梯度形状不对"
    print("✓ 通过: 梯度流正常")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)
    print("\n使用说明:")
    print("  model = DilatedAsterisk(in_channels=768, out_channels=768)")
    print("  output = model(input)  # [B, 768, H, W] → [B, 768, H, W]")

