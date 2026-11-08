"""Pixel-specific H residual reconstruction with a detached native C prior.

The coarse grid is indexed by floor(pixel / 4). Border sources are masked.
Only scalar nine-way logits are stacked; key/value sources are gathered one
at a time. Row-chunk checkpointing bounds retained attention intermediates.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE


class HPixelReassembly(nn.Module):
    def __init__(self, query_channels=96, h_channels=64, g_channels=96,
                 latent_channels=16, relation_floor=.1, prior_mode='predicted',
                 chunk_rows=32, checkpoint_chunks=True):
        super().__init__()
        if not 0 < relation_floor <= 1 or prior_mode not in ('predicted', 'uniform'):
            raise ValueError('Expected positive relation floor <=1 and predicted/uniform prior')
        if latent_channels < 1 or chunk_rows < 1:
            raise ValueError('Latent channels and chunk rows must be positive')
        self.query = nn.Conv2d(query_channels, latent_channels, 1, bias=False)
        self.key = nn.Conv2d(h_channels+g_channels, latent_channels, 1, bias=False)
        self.value = nn.Conv2d(h_channels, latent_channels, 1, bias=False)
        self.position = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        self.out = nn.Conv2d(latent_channels, query_channels, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.relation_floor = float(relation_floor)
        self.prior_mode = prior_mode
        self.chunk_rows = int(chunk_rows)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.last_stats = {}

    @staticmethod
    def _shift(x, dy, dx):
        h, w = x.shape[-2:]
        padded = F.pad(x, (max(-dx,0), max(dx,0), max(-dy,0), max(dy,0)))
        return padded[..., max(dy,0):max(dy,0)+h, max(dx,0):max(dx,0)+w]

    def log_priors(self, c_logits):
        # This new route never backpropagates into C logits. Existing routes do.
        with torch.no_grad():
            c = c_logits.detach().sigmoid()
            priors = [torch.zeros_like(c[:, :1])]
            for d, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
                reverse = self._shift(c[:, CONNECTIVITY_OPPOSITE[d]:CONNECTIVITY_OPPOSITE[d]+1],dy,dx)
                r = self.relation_floor+(1-self.relation_floor)*torch.sqrt(c[:,d:d+1]*reverse)
                priors.append(r.log() if self.prior_mode == 'predicted' else torch.zeros_like(r))
            return torch.cat(priors, 1)

    def _chunk(self, query, keys, values, priors, top, coarse_hw, return_weights=False):
        b, channels, rows, width = query.shape
        h, w = coarse_hw
        yy = torch.arange(top, top+rows, device=query.device)[:,None].expand(rows,width)
        xx = torch.arange(width, device=query.device)[None,:].expand(rows,width)
        py, px = yy//4, xx//4
        fy = (yy%4+.5)/4-.5
        fx = (xx%4+.5)/4-.5
        parent = (py*w+px).flatten()
        sources = [(0,0)]+list(CONNECTIVITY_DIRECTIONS)
        flat_keys, flat_values = keys.flatten(2), values.flatten(2)
        reference = flat_values.index_select(2,parent).reshape(b,channels,rows,width)
        flat_priors = priors.flatten(2)
        scores = []
        for i, (dy,dx) in enumerate(sources):
            sy,sx = py+dy,px+dx
            valid = (sy>=0)&(sy<h)&(sx>=0)&(sx<w)
            index = (sy.clamp(0,h-1)*w+sx.clamp(0,w-1)).flatten()
            key = flat_keys.index_select(2,index).reshape(b,channels,rows,width)
            geometry = torch.stack((fx,fy,torch.full_like(fx,dx),torch.full_like(fy,dy)),-1)
            positional = self.position(geometry).permute(2,0,1)[None]
            score = (query*key).sum(1,keepdim=True)/math.sqrt(channels)+positional
            score = score+flat_priors[:,i:i+1].index_select(2,parent).reshape(b,1,rows,width)
            scores.append(score.masked_fill(~valid[None,None],-torch.inf))
        weights = torch.softmax(torch.cat(scores,1),1)
        delta = torch.zeros_like(query)
        for i, (dy,dx) in enumerate(sources[1:],1):
            index = ((py+dy).clamp(0,h-1)*w+(px+dx).clamp(0,w-1)).flatten()
            neighbour = flat_values.index_select(2,index).reshape(b,channels,rows,width)
            delta = delta+weights[:,i:i+1]*(neighbour-reference)
        with torch.no_grad():
            stats = torch.stack((weights[:,0].sum(),
                                 -(weights*weights.clamp_min(1e-12).log()).sum()))
        if return_weights:
            return delta, stats, weights
        return delta, stats

    def forward(self, x, h, g, c_logits, return_weights=False):
        b, _, height, width = x.shape
        ch, cw = h.shape[-2:]
        if (height,width)!=(4*ch,4*cw) or g.shape[-2:]!=(ch,cw) or c_logits.shape!=(b,8,ch,cw):
            raise ValueError('Expected native pixels at 4x the aligned H/G/C grid')
        query = self.query(x)
        keys = self.key(torch.cat((h,g),1))
        values = self.value(h)
        priors = self.log_priors(c_logits)
        chunks=[]; stats=[]; weights=[]
        for top in range(0,height,self.chunk_rows):
            q = query[:,:,top:top+self.chunk_rows]
            # Bind top: checkpoint backward must not capture the final loop value.
            def run(q,k,v,r,origin=top):
                return self._chunk(q,k,v,r,origin,(ch,cw),return_weights)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled() and not return_weights:
                result = checkpoint(run,q,keys,values,priors,use_reentrant=False,preserve_rng_state=False)
            else:
                result = run(q,keys,values,priors)
            chunks.append(result[0]); stats.append(result[1])
            if return_weights:
                weights.append(result[2])
        residual = self.out(torch.cat(chunks,2))
        total_stats = torch.stack(stats).sum(0)/(b*height*width)
        absolute = residual.detach().abs()
        self.last_stats = dict(self_weight_mean=total_stats[0].detach(),
                               entropy_mean=total_stats[1].detach(),
                               residual_abs_mean=absolute.mean(),
                               residual_abs_max=absolute.amax())
        if return_weights:
            return residual, torch.cat(weights,2)
        return residual


def install_h_pixel_reassembly(model, **kwargs):
    """Call after original native-B installation; preserve the shared RNG state."""
    core = model.swin_unet
    if not getattr(core,'native_relation_enabled',False):
        raise ValueError('Pixel reconstruction requires the original native B')
    if hasattr(core,'h_pixel_reassembly'):
        raise ValueError('Pixel reconstruction is already installed')
    with torch.random.fork_rng(devices=[]):
        module = HPixelReassembly(query_channels=core.embed_dim,**kwargs)
    core.h_pixel_reassembly = module.to(next(core.parameters()).device)
    return model


def install_reassembly_from_args(model, saved):
    if saved.get('h_pixel_reassembly',False):
        install_h_pixel_reassembly(model,
            latent_channels=saved.get('h_reassembly_channels',16),
            relation_floor=saved.get('h_reassembly_floor',.1),
            prior_mode=saved.get('h_reassembly_prior','predicted'),
            chunk_rows=saved.get('h_reassembly_chunk_rows',32),
            checkpoint_chunks=saved.get('h_reassembly_checkpoint',True))
    if saved.get('no_encoder_a1a2',False):
        from .remove_encoder_road_priors import remove_encoder_road_priors
        remove_encoder_road_priors(model)
    return model
