"""Original H0: write E into H before merging, read H phases after merging."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EncoderHPostMergePhase(nn.Module):
    def __init__(self, encoder_channels, phase_factor, h_channels=64, initial_strength=.05):
        super().__init__()
        if not 0 < initial_strength < 1 or phase_factor not in (2, 4, 8):
            raise ValueError('Expected a positive strength <1 and phase factor 2/4/8')
        self.phase_factor = phase_factor
        self.to_h = nn.Sequential(nn.Conv2d(encoder_channels, h_channels, 1, bias=False),
                                  nn.GroupNorm(8, h_channels))
        self.to_e = nn.Sequential(nn.Conv2d(h_channels*phase_factor**2, 2*encoder_channels, 1, bias=False),
                                  nn.GroupNorm(8, 2*encoder_channels))
        initial = math.log(initial_strength/(1-initial_strength))
        self.theta_eh = nn.Parameter(torch.tensor(initial))
        self.theta_he = nn.Parameter(torch.tensor(initial))
        self.last_stats = {}

    def before_merge(self, tokens, h, resolution):
        height, width = resolution
        b, n, c = tokens.shape
        if n != height*width:
            raise ValueError('Pre-merge token/grid mismatch')
        e = tokens.transpose(1, 2).reshape(b, c, height, width)
        update = self.theta_eh.sigmoid()*F.interpolate(self.to_h(e), size=h.shape[-2:],
                                                       mode='bilinear', align_corners=False)
        self.last_stats = dict(H_delta_abs_mean=update.detach().abs().mean(),
                               strength_E_to_H=self.theta_eh.detach().sigmoid())
        return h+update

    def after_merge(self, tokens, h, resolution):
        height, width = resolution
        if h.shape[-2:] != (height*self.phase_factor, width*self.phase_factor):
            raise ValueError('H phases are not aligned to the post-merge token grid')
        phases = F.pixel_unshuffle(h, self.phase_factor)
        update = self.theta_he.sigmoid()*self.to_e(phases)
        if tokens.shape != (h.shape[0], height*width, update.shape[1]):
            raise ValueError('Post-merge token/channel mismatch')
        self.last_stats.update(E_delta_abs_mean=update.detach().abs().mean(),
            strength_H_to_E=self.theta_he.detach().sigmoid(), phase_factor=self.phase_factor)
        return tokens+update.flatten(2).transpose(1, 2)


def remove_encoder_a1a2(core):
    before = set(core.state_dict())
    for index in (0, 1):
        layer = core.layers[index]
        layer.road_attention_head = None
        layer.road_attention_modulates_downsample = False
        layer.last_road_attention = None
        layer.downsample.road_alpha = None
    del core.encoder_stage1_road_attention_head
    del core.encoder_stage2_road_attention_head
    core.layers[2].use_road_bias = False
    for block in core.layers[2].blocks:
        block.use_road_bias = False
        block.road_bias_scale_a1 = None
        block.road_bias_scale_a2 = None
    return sorted(before-set(core.state_dict()))


def install_encoder_h_postmerge_phase(model, initial_strength=.05):
    core = model.swin_unet
    if hasattr(core, 'encoder_h_postmerge_phases'):
        raise ValueError('Encoder-H post-merge phases already installed')
    if not (core.enable_highres_structure_stream and core.highres_structure_channels == 64):
        raise ValueError('Expected the original full H0 with a 64-channel H stream')
    if getattr(core, 'native_relation_enabled', False) or hasattr(core, 'h_pixel_reassembly'):
        raise ValueError('This experiment is based on original 599a410 H0, not native B')
    removed = remove_encoder_a1a2(core)
    with torch.random.fork_rng(devices=[]):
        # CPU-only seed; do not reset CUDA RNG when installing after .to(cuda).
        torch.random.default_generator.manual_seed((torch.initial_seed()+104729) % (2**63-1))
        modules = nn.ModuleList([EncoderHPostMergePhase(core.embed_dim*2**i, 2**(i+1),
                                   initial_strength=initial_strength) for i in range(3)])
    core.encoder_h_postmerge_phases = modules.to(next(core.parameters()).device)
    core.last_encoder_features = {}
    core.encoder_a1a2_removed_keys = ['swin_unet.'+key for key in removed]
    print('[ENCODER PHASE] A1/A2 heads, first-two merge modulation and Stage3 A bias removed; '
          'H64 stays high resolution; E->H before merge, phase H->E after merge.', flush=True)
    return model
