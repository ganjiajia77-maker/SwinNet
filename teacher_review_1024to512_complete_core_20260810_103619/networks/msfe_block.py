"""
Multi-Scale Feature Enhancement Block (MSFE)
Based on MSCE from DARENet - designed for skip connections
Enhances encoder features with multi-scale dilated convolutions and ASPP pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class ASPPPoolingH(nn.Module):
    """ASPP pooling along height dimension"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingH, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Global average pooling along height
        B, C, H, W = x.shape
        # Pool over H dimension: (B, C, H, W) -> (B, C, 1, W)
        pool_out = self.pool(x)  # (B, C, 1, W)
        # Upsample back to original height
        pool_out = F.interpolate(pool_out, size=(H, W), mode='bilinear', align_corners=False)
        return self.conv(pool_out)


class ASPPPoolingW(nn.Module):
    """ASPP pooling along width dimension"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingW, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Global average pooling along width
        B, C, H, W = x.shape
        # Pool over W dimension: (B, C, H, W) -> (B, C, H, 1)
        pool_out = self.pool(x)  # (B, C, H, 1)
        # Upsample back to original width
        pool_out = F.interpolate(pool_out, size=(H, W), mode='bilinear', align_corners=False)
        return self.conv(pool_out)


class MSFEBlock(nn.Module):
    """
    Multi-Scale Feature Enhancement Block
    Enhances skip connection features with:
    - Multi-scale dilated convolutions (square kernels)
    - Multi-scale vertical/horizontal dilated convolutions  
    - ASPP pooling (height and width)
    - Learnable feature fusion with gamma
    
    Args:
        channel: Input/output channel dimension
    """
    def __init__(self, channel):
        super(MSFEBlock, self).__init__()
        
        # ===== Square kernel dilated convolutions (dilation rates: 1,2,4,8) =====
        self.dilate11 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate22 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate33 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate44 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        
        # ===== Vertical (3×1) dilated convolutions =====
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=1, padding=(1, 0))
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=2, padding=(2, 0))
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=4, padding=(4, 0))
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=8, padding=(8, 0))
        
        # ===== Horizontal (1×3) dilated convolutions =====
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=1, padding=(0, 1))
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=2, padding=(0, 2))
        self.dilate7 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=4, padding=(0, 4))
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=8, padding=(0, 8))
        
        # ===== Feature fusion convolutions =====
        self.dconv = nn.Conv2d(channel * 5, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        
        # ===== ASPP pooling modules =====
        self.ASPPH = ASPPPoolingH(in_channels=channel, out_channels=channel)
        self.ASPPW = ASPPPoolingW(in_channels=channel, out_channels=channel)
        
        # ===== Learnable feature fusion weight (gamma) =====
        # Start with 0.1 instead of 0 to ensure early-stage contribution
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
        # ===== Initialize biases to zero =====
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass with multi-scale feature enhancement
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            out: Enhanced output tensor (B, C, H, W)
        """
        # ===== Branch 1: Square kernel dilated convolutions =====
        dilate11_out = nonlinearity(self.dilate11(x))
        dilate21_out = nonlinearity(self.dilate22(dilate11_out))
        dilate31_out = nonlinearity(self.dilate33(dilate21_out))
        dilate41_out = nonlinearity(self.dilate44(dilate31_out))
        dilate1_out = self.conv1(dilate11_out + dilate21_out + dilate31_out + dilate41_out)

        # ===== Branch 2: Vertical (3×1) dilated convolutions =====
        dilate12_out = nonlinearity(self.dilate1(x))
        dilate22_out = nonlinearity(self.dilate2(dilate12_out))
        dilate32_out = nonlinearity(self.dilate3(dilate22_out))
        dilate42_out = nonlinearity(self.dilate4(dilate32_out))
        dilate2_out = self.conv2(dilate12_out + dilate22_out + dilate32_out + dilate42_out)

        # ===== Branch 3: Horizontal (1×3) dilated convolutions =====
        dilate13_out = nonlinearity(self.dilate5(x))
        dilate23_out = nonlinearity(self.dilate6(dilate13_out))
        dilate33_out = nonlinearity(self.dilate7(dilate23_out))
        dilate43_out = nonlinearity(self.dilate8(dilate33_out))
        dilate3_out = self.conv3(dilate13_out + dilate23_out + dilate33_out + dilate43_out)

        # ===== Branch 4 & 5: ASPP pooling (height and width) =====
        dilateH_out = self.ASPPH(x)
        dilateW_out = self.ASPPW(x)

        # ===== Concatenate all branches =====
        outsum = torch.cat([dilate1_out, dilate2_out, dilate3_out, dilateH_out, dilateW_out], dim=1)
        
        # ===== Fuse concatenated features =====
        out = self.dconv(outsum)
        
        # ===== Residual connection with learnable gamma =====
        # gamma controls how much of MSFE contribution to add (0 = no effect, 1 = full effect)
        out = self.gamma * out + x * (1 - self.gamma)
        
        return out
