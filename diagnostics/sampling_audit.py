"""Small frozen-checkpoint controls for final expansion and A1/A2 usage."""
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from diagnostics.stage_feature_readout import token_map


class UpsampleCapture:
    def __init__(self, model):
        self.values={};core=model.swin_unet
        def parent(m,args,result):self.values['parent']=token_map(result).detach().clone()
        def raw(m,args):self.values['raw']=token_map(args[0]).detach().clone()
        def post(m,args,result):self.values['post']=token_map(result).detach().clone()
        self.handles=[core.norm_up.register_forward_hook(parent),core.up.norm.register_forward_pre_hook(raw),
                      core.up.register_forward_hook(post)]

    def close(self):
        for h in self.handles:h.remove()
        self.values.clear()


class PhaseReadout(nn.Module):
    """Independent linear weights for each of the sixteen subpixel phases."""
    def __init__(self, shared, scale=4):
        super().__init__();self.scale=scale
        self.register_buffer('mean',shared.mean.clone());self.register_buffer('std',shared.std.clone())
        self.weight=nn.Parameter(shared.head.weight.detach().reshape(1,-1).repeat(scale*scale,1))
        self.bias=nn.Parameter(shared.head.bias.detach().repeat(scale*scale))

    def forward(self,x):
        x=(x-self.mean)/self.std;b,c,h,w=x.shape;s=self.scale
        x=F.pixel_unshuffle(x,s).reshape(b,c,s*s,h//s,w//s)
        z=torch.einsum('bcphw,pc->bphw',x,self.weight)+self.bias[None,:,None,None]
        return F.pixel_shuffle(z,s)


def expansion_matrix_report(up):
    w=up.expand.weight.detach().double().cpu()
    singular=torch.linalg.svdvals(w)
    tolerance=float(singular[0])*max(w.shape)*torch.finfo(torch.float32).eps
    return dict(input_channels=w.shape[1],expanded_channels=w.shape[0],
                rank_at_float32_tolerance=int((singular>tolerance).sum()),
                smallest_singular=float(singular[-1]),largest_singular=float(singular[0]),
                condition_number=float(singular[0]/singular[-1]),rank_tolerance=tolerance,
                full_column_rank=bool((singular>tolerance).sum()==w.shape[1]))


@torch.no_grad()
def reconstruction_report(up,captured):
    parent,raw,post=(captured[k] for k in ('parent','raw','post'))
    s=up.dim_scale;b,c,h,w=parent.shape
    expanded=F.pixel_unshuffle(raw,s).reshape(b,c,s*s,h,w).permute(0,2,1,3,4).reshape(b,s*s*c,h,w)
    inverse=torch.linalg.pinv(up.expand.weight.detach().double()).to(device=parent.device,dtype=parent.dtype)
    recovered=torch.einsum('co,bohw->bchw',inverse,expanded)
    # LN reconstruction uses the actual saved mean/std as diagnostic side info.
    # These statistics are not assumed available to an ordinary readout.
    mean=raw.mean(1,keepdim=True);std=(raw.var(1,unbiased=False,keepdim=True)+up.norm.eps).sqrt()
    gamma=up.norm.weight.detach()[None,:,None,None];beta=up.norm.bias.detach()[None,:,None,None]
    result=dict(pre_layernorm_parent_reconstruction_mean_abs=float((recovered-parent).abs().mean()),
                pre_layernorm_parent_reconstruction_max_abs=float((recovered-parent).abs().max()),
                layernorm_gamma_min_abs=float(gamma.abs().min()))
    if gamma.abs().min()>1e-8:
        reconstructed_raw=(post-beta)/gamma*std+mean
        result['layernorm_reconstruction_with_saved_stats_max_abs']=float((raw-reconstructed_raw).abs().max())
    return result


def prior_parameters(core):
    merge=[core.layers[i].downsample.road_alpha for i in (0,1)]
    bias=[]
    for block in core.layers[2].blocks:
        bias.extend([block.road_bias_scale_a1,block.road_bias_scale_a2])
    return merge,bias


@contextmanager
def prior_intervention(core,mode):
    merge,bias=prior_parameters(core)
    selected={'merge12_off':merge,'stage3_bias_off':bias,'both_off':merge+bias,'normal':[]}[mode]
    saved=[p.detach().clone() for p in selected]
    try:
        with torch.no_grad():
            for p in selected:p.zero_()
        yield
    finally:
        with torch.no_grad():
            for p,v in zip(selected,saved):p.copy_(v)
