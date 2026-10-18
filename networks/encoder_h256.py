"""Native H256 feature exchange and final surface residual."""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE_ID = 'h0_encoder_h256_s2_skeleton_s3_diffusion_v1'


class EncoderHExchange(nn.Module):
    def __init__(self, encoder_channels, highres_channels=64):
        super().__init__()
        self.refine = nn.Sequential(nn.Conv2d(highres_channels, highres_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(highres_channels), nn.GELU())
        self.to_h = nn.Sequential(nn.Conv2d(encoder_channels, highres_channels, 1, bias=False),
                                  nn.BatchNorm2d(highres_channels))
        self.to_e = nn.Sequential(nn.Conv2d(highres_channels, encoder_channels, 1, bias=False),
                                  nn.BatchNorm2d(encoder_channels))
        self.theta_eh = nn.Parameter(torch.tensor(math.log(.05/.95)))
        self.theta_he = nn.Parameter(torch.tensor(math.log(.05/.95)))

    def forward(self, e, h):
        hbar = h + self.refine(h)
        to_h = F.interpolate(self.to_h(e), size=h.shape[-2:], mode='bilinear', align_corners=False)
        to_e = self.to_e(F.adaptive_avg_pool2d(hbar, e.shape[-2:]))
        return e + self.theta_he.sigmoid()*to_e, hbar + self.theta_eh.sigmoid()*to_h


class HSurfaceResidual(nn.Module):
    def __init__(self, surface_channels, highres_channels=64):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(highres_channels, 16, 1, bias=False),
                                     nn.GroupNorm(8, 16), nn.GELU())
        self.out = nn.Conv2d(16, surface_channels, 1)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, h):
        p = self.project(h)
        p = F.interpolate(p, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x + self.out(p)
