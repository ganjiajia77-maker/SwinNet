"""
简化的 DilatedAsterisk 模块
移除复杂的 transform 逻辑，使用纯多方向空洞卷积实现
"""

import torch
import torch.nn as nn


class DilatedAsterisk(nn.Module):
    """
    基础版本：多尺度空洞卷积
    
    在 decoder 特征上进行空间增强，使用 3 个不同的感受野
    - kernel=3, dilation=2 → RF=5 (局部细节)
    - kernel=5, dilation=2 → RF=9 (中等上下文)
    - kernel=5, dilation=3 → RF=13 (全局语义)
    """
    
    def __init__(self, in_channels, out_channels, kernel_sizes=None):
        super(DilatedAsterisk, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 默认的 kernel 和 dilation 配置
        if kernel_sizes is None:
            kernel_sizes = [
                {'kernel': 3, 'dilation': 2},
                {'kernel': 5, 'dilation': 2},
                {'kernel': 5, 'dilation': 3},
            ]
        self.kernel_sizes = kernel_sizes
        
        # 中间通道数
        mid_channels = max(in_channels // 4, 16)
        
        # 1x1 卷积降维
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn_in = nn.BatchNorm2d(mid_channels)
        
        # 水平卷积分支（kernel_sizes 对应的卷积）
        self.h_branches = nn.ModuleList()
        for ks_info in kernel_sizes:
            kernel = ks_info['kernel']
            dilation = ks_info['dilation']
            padding = dilation * (kernel - 1) // 2
            self.h_branches.append(
                nn.Conv2d(mid_channels, mid_channels, 
                         kernel_size=(1, kernel),
                         padding=(0, padding),
                         dilation=(1, dilation),
                         bias=False)
            )
        
        # 垂直卷积分支
        self.v_branches = nn.ModuleList()
        for ks_info in kernel_sizes:
            kernel = ks_info['kernel']
            dilation = ks_info['dilation']
            padding = dilation * (kernel - 1) // 2
            self.v_branches.append(
                nn.Conv2d(mid_channels, mid_channels,
                         kernel_size=(kernel, 1),
                         padding=(padding, 0),
                         dilation=(dilation, 1),
                         bias=False)
            )
        
        # 特征聚合
        num_branches = len(kernel_sizes) * 2  # 水平 + 垂直
        self.bn_mid = nn.BatchNorm2d(mid_channels * num_branches)
        self.relu_mid = nn.ReLU(inplace=True)
        
        # 输出层
        self.conv_out_1 = nn.Conv2d(mid_channels * num_branches, mid_channels,
                                    kernel_size=3, padding=1)
        self.bn_out_1 = nn.BatchNorm2d(mid_channels)
        self.relu_out_1 = nn.ReLU(inplace=True)
        
        self.conv_out_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        
        self._init_weight()
    
    def forward(self, x):
        """前向传播"""
        # 降维
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = torch.relu(out)
        
        # 多分支
        branch_outs = []
        
        # 水平分支
        for h_branch in self.h_branches:
            branch_outs.append(h_branch(out))
        
        # 垂直分支
        for v_branch in self.v_branches:
            branch_outs.append(v_branch(out))
        
        # 拼接所有分支
        out = torch.cat(branch_outs, dim=1)
        out = self.bn_mid(out)
        out = self.relu_mid(out)
        
        # 升维
        out = self.conv_out_1(out)
        out = self.bn_out_1(out)
        out = self.relu_out_1(out)
        
        out = self.conv_out_2(out)
        
        # 残差连接
        out = out + x
        
        return out
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DilatedAsteriskWithDirections(nn.Module):
    """
    增强版本：支持 4 个方向（水平、垂直、两条对角线）
    
    相比基础版本，增加了对角线卷积分支，能更好地捕捉斜向特征
    """
    
    def __init__(self, in_channels, out_channels, kernel_sizes=None):
        super(DilatedAsteriskWithDirections, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 默认配置
        if kernel_sizes is None:
            kernel_sizes = [
                {'kernel': 3, 'dilation': 2},
                {'kernel': 5, 'dilation': 2},
                {'kernel': 5, 'dilation': 3},
            ]
        self.kernel_sizes = kernel_sizes
        
        # 中间通道数
        mid_channels = max(in_channels // 4, 16)
        
        # 1x1 卷积降维
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn_in = nn.BatchNorm2d(mid_channels)
        self.relu_in = nn.ReLU(inplace=True)
        
        # 水平分支
        self.h_branches = nn.ModuleList()
        for ks_info in kernel_sizes:
            kernel = ks_info['kernel']
            dilation = ks_info['dilation']
            padding = dilation * (kernel - 1) // 2
            self.h_branches.append(
                nn.Conv2d(mid_channels, mid_channels,
                         kernel_size=(1, kernel),
                         padding=(0, padding),
                         dilation=(1, dilation),
                         bias=False)
            )
        
        # 垂直分支
        self.v_branches = nn.ModuleList()
        for ks_info in kernel_sizes:
            kernel = ks_info['kernel']
            dilation = ks_info['dilation']
            padding = dilation * (kernel - 1) // 2
            self.v_branches.append(
                nn.Conv2d(mid_channels, mid_channels,
                         kernel_size=(kernel, 1),
                         padding=(padding, 0),
                         dilation=(dilation, 1),
                         bias=False)
            )
        
        # 45° 对角线分支（先水平后垂直）
        self.d45_branches = nn.ModuleList()
        for ks_info in kernel_sizes:
            kernel = ks_info['kernel']
            dilation = ks_info['dilation']
            padding = dilation * (kernel - 1) // 2
            self.d45_branches.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels,
                             kernel_size=(1, kernel),
                             padding=(0, padding),
                             dilation=(1, dilation),
                             bias=False),
                    nn.Conv2d(mid_channels, mid_channels,
                             kernel_size=(kernel, 1),
                             padding=(padding, 0),
                             dilation=(dilation, 1),
                             bias=False)
                )
            )
        
        # 135° 对角线分支（先垂直后水平）
        self.d135_branches = nn.ModuleList()
        for ks_info in kernel_sizes:
            kernel = ks_info['kernel']
            dilation = ks_info['dilation']
            padding = dilation * (kernel - 1) // 2
            self.d135_branches.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels,
                             kernel_size=(kernel, 1),
                             padding=(padding, 0),
                             dilation=(dilation, 1),
                             bias=False),
                    nn.Conv2d(mid_channels, mid_channels,
                             kernel_size=(1, kernel),
                             padding=(0, padding),
                             dilation=(1, dilation),
                             bias=False)
                )
            )
        
        # 特征聚合
        num_branches = len(kernel_sizes) * 4  # 4 个方向
        self.bn_mid = nn.BatchNorm2d(mid_channels * num_branches)
        self.relu_mid = nn.ReLU(inplace=True)
        
        # 输出层
        self.conv_out_1 = nn.Conv2d(mid_channels * num_branches, mid_channels,
                                    kernel_size=3, padding=1)
        self.bn_out_1 = nn.BatchNorm2d(mid_channels)
        self.relu_out_1 = nn.ReLU(inplace=True)
        
        self.conv_out_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        
        self._init_weight()
    
    def forward(self, x):
        """前向传播"""
        # 降维
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu_in(out)
        
        branch_outs = []
        
        # 水平分支
        for h_branch in self.h_branches:
            branch_outs.append(h_branch(out))
        
        # 垂直分支
        for v_branch in self.v_branches:
            branch_outs.append(v_branch(out))
        
        # 45° 对角线分支
        for d45_branch in self.d45_branches:
            branch_outs.append(d45_branch(out))
        
        # 135° 对角线分支
        for d135_branch in self.d135_branches:
            branch_outs.append(d135_branch(out))
        
        # 拼接所有分支
        out = torch.cat(branch_outs, dim=1)
        out = self.bn_mid(out)
        out = self.relu_mid(out)
        
        # 升维
        out = self.conv_out_1(out)
        out = self.bn_out_1(out)
        out = self.relu_out_1(out)
        
        out = self.conv_out_2(out)
        
        # 残差连接
        out = out + x
        
        return out
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DilatedAsterisk 模块测试")
    print("="*80)
    
    # 测试 1: 基础版本
    print("\n[1] DilatedAsterisk (基础版本)")
    print("-" * 80)
    model1 = DilatedAsterisk(in_channels=768, out_channels=768)
    x = torch.randn(2, 768, 32, 32)
    y1 = model1(x)
    print(f"输入:  {x.shape}")
    print(f"输出:  {y1.shape}")
    print(f"参数量: {sum(p.numel() for p in model1.parameters()) / 1e6:.2f}M")
    assert y1.shape == x.shape, "形状不匹配"
    print("✓ 通过")
    
    # 测试 2: 方向版本
    print("\n[2] DilatedAsteriskWithDirections (方向版本)")
    print("-" * 80)
    model2 = DilatedAsteriskWithDirections(in_channels=768, out_channels=768)
    y2 = model2(x)
    print(f"输入:  {x.shape}")
    print(f"输出:  {y2.shape}")
    print(f"参数量: {sum(p.numel() for p in model2.parameters()) / 1e6:.2f}M")
    assert y2.shape == x.shape, "形状不匹配"
    print("✓ 通过")
    
    # 测试 3: 二分类
    print("\n[3] 用于二分类 (num_classes=2)")
    print("-" * 80)
    model3 = DilatedAsteriskWithDirections(in_channels=2, out_channels=2)
    x_bin = torch.randn(2, 2, 32, 32)
    y3 = model3(x_bin)
    print(f"输入:  {x_bin.shape}")
    print(f"输出:  {y3.shape}")
    print(f"参数量: {sum(p.numel() for p in model3.parameters()) / 1e6:.2f}M")
    assert y3.shape == x_bin.shape, "形状不匹配"
    print("✓ 通过")
    
    # 测试 4: 梯度流
    print("\n[4] 梯度流测试")
    print("-" * 80)
    model4 = DilatedAsteriskWithDirections(in_channels=2, out_channels=2)
    x_grad = torch.randn(2, 2, 32, 32, requires_grad=True)
    y_grad = model4(x_grad)
    loss = y_grad.sum()
    loss.backward()
    print(f"输入梯度存在: {x_grad.grad is not None}")
    print(f"梯度形状: {x_grad.grad.shape if x_grad.grad is not None else 'N/A'}")
    assert x_grad.grad is not None, "梯度流中断"
    print("✓ 通过")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)
