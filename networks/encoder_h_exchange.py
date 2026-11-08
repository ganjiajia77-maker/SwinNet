"""Three pre-merging encoder exchanges; carry one H state into the decoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EncoderHExchange(nn.Module):
    def __init__(self, encoder_channels, h_channels=64, initial_strength=.05):
        super().__init__()
        if not 0 < initial_strength < 1:
            raise ValueError('Encoder H initial strength must lie strictly between 0 and 1')
        self.to_h=nn.Sequential(nn.Conv2d(encoder_channels,h_channels,1,bias=False),
                                nn.GroupNorm(8,h_channels))
        self.to_e=nn.Sequential(nn.Conv2d(h_channels,encoder_channels,1,bias=False),
                                nn.GroupNorm(8,encoder_channels))
        initial=math.log(initial_strength/(1-initial_strength))
        self.theta_eh=nn.Parameter(torch.tensor(initial))
        self.theta_he=nn.Parameter(torch.tensor(initial))
        self.last_stats={}

    def forward(self, tokens, h, resolution):
        height,width=resolution
        b,n,c=tokens.shape
        if n!=height*width:raise ValueError('Encoder token/grid mismatch')
        e=tokens.transpose(1,2).reshape(b,c,height,width)
        # Both updates read the same old E/H; no sequential circular feedback.
        dh=self.theta_eh.sigmoid()*F.interpolate(self.to_h(e),size=h.shape[-2:],
                                                mode='bilinear',align_corners=False)
        de=self.theta_he.sigmoid()*self.to_e(F.adaptive_avg_pool2d(h,resolution))
        h_next=h+dh
        e_next=e+de
        self.last_stats=dict(H_delta_abs_mean=dh.detach().abs().mean(),
                             E_delta_abs_mean=de.detach().abs().mean(),
                             strength_E_to_H=self.theta_eh.detach().sigmoid(),
                             strength_H_to_E=self.theta_he.detach().sigmoid())
        return e_next.flatten(2).transpose(1,2).contiguous(),h_next


def install_encoder_h_exchange(model, initial_strength=.05):
    core=model.swin_unet
    if not (getattr(core,'native_relation_enabled',False) and
            getattr(core,'encoder_road_priors_removed',False) and hasattr(core,'h_pixel_reassembly')):
        raise ValueError('Encoder-H experiment requires the c77fc30 no-A pixel-reassembly backbone')
    if hasattr(core,'encoder_h_exchanges'):raise ValueError('Encoder H exchanges already installed')
    with torch.random.fork_rng(devices=[]):
        # Separate deterministic initialization without consuming the shared stream.
        torch.random.default_generator.manual_seed((torch.initial_seed()+104729) % (2**63-1))
        modules=nn.ModuleList([EncoderHExchange(core.embed_dim*(2**i),64,initial_strength)
                               for i in range(3)])
    core.encoder_h_exchanges=modules.to(next(core.parameters()).device)
    core.last_encoder_features={}
    return model
