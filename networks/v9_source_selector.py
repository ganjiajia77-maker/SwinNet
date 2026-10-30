"""Eight actual neighbours plus stay; original V9 files/checkpoint hashes unchanged."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from networks.reliable_directional_repair import DIRECTIONS, shifted

VERSION = 'v9_source_relation_selector_v1'


def source_support(state, p0, reliable):
    gained = ((state-p0).clamp_min(0.)/(1.-p0).clamp_min(1e-5)).clamp_max(1.)
    return reliable+(1.-reliable)*gained


class SourceRelationSelector(nn.Module):
    """Shared per-edge scorer; no mixing of eight edges before scoring."""
    def __init__(self, width=32, hidden=32):
        super().__init__(); self.width=width
        # Endpoint/difference/product, existing directional pair, local context,
        # midpoint and left/right flanks: nine width-channel feature blocks.
        self.edge = nn.Sequential(nn.Conv2d(9*width+16,hidden,1),nn.GELU(),
                                  nn.Conv2d(hidden,16,1),nn.GELU(),nn.Conv2d(16,1,1))
        self.stay = nn.Sequential(nn.Conv2d(width+10,16,1),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.normal_(self.edge[-1].weight,std=1e-3); nn.init.zeros_(self.edge[-1].bias)
        nn.init.zeros_(self.stay[-1].weight)
        nn.init.constant_(self.stay[-1].bias,math.log(72.)) # .9 stay at zero edge logits

    def forward(self, context, state, support, available, rate):
        feature=context['feature']; p0=context['p0']; q=context['q']; r=context['r']; w=context['w']
        logits=[]; sources=[]; valid=[]; supports=[]
        ones=torch.ones_like(state)
        for d,(dy,dx) in enumerate(DIRECTIONS):
            other=shifted(feature,dy,dx); u_other=shifted(state,dy,dx)
            source=shifted(support,dy,dx); supports.append(source)
            # These neighbour/midpoint/flank features refer to the SAME short edge
            # used for state transport, rather than a different multi-radius source.
            py,px=-dx,dy
            midpoint=.5*(feature+other)
            left=.5*(shifted(feature,py,px)+shifted(feature,dy+py,dx+px))
            right=.5*(shifted(feature,-py,-px)+shifted(feature,dy-py,dx-px))
            scalar=torch.cat((state,u_other,u_other-state,r,shifted(r,dy,dx),support,source,
                w[:,d:d+1],p0,shifted(p0,dy,dx),q,shifted(q,dy,dx),
                torch.full_like(state,float(dy)),torch.full_like(state,float(dx)),rate,available),1)
            value=torch.cat((feature,other,other-feature,other*feature,
                context['pair'][:,d],context['local'],midpoint,left,right,scalar),1)
            logits.append(self.edge(value)); sources.append(u_other)
            valid.append(shifted(ones,dy,dx)>.5)
        edge_logits=torch.cat(logits,1); valid=torch.cat(valid,1)
        stack=torch.cat(supports,1)
        entropy=-(w.clamp_min(1e-8)*w.clamp_min(1e-8).log()).sum(1,keepdim=True)
        stay=self.stay(torch.cat((feature,state,p0,q,r,available,rate,
            stack.mean(1,keepdim=True),stack.amax(1,keepdim=True),entropy,edge_logits.amax(1,keepdim=True)),1))
        choice=torch.cat((stay,edge_logits.masked_fill(~valid,-torch.inf)),1).softmax(1)
        return {'edge_logits':edge_logits,'stay_logit':stay,'choice':choice,
                'source_states':torch.cat(sources,1),'valid':valid}


class SourceSelectedRepair(nn.Module):
    """Reuse exact original forward and replace only its dynamics output via a hook.

    Scalar mode delegates untouched. In source mode, the original scalar loop is
    computed but discarded; this extra small-loop cost avoids a core refactor.
    No original G is multiplied into the new transport, even when it is computed
    to preserve the legacy reference and existing directional feature interfaces.
    """
    def __init__(self, original, hidden=32):
        super().__init__(); self.original=original; self.selector=SourceRelationSelector(32,hidden)
        self.mode='source'; self.training_phase='relation'; self._context=None; self._states=None
        self.set_phase('relation')

    def set_phase(self, phase):
        if phase not in ('relation','repair','joint'): raise ValueError('Unknown selector phase')
        self.training_phase=phase; self.original.requires_grad_(False); self.selector.requires_grad_(True)
        if phase=='relation': self.selector.stay.requires_grad_(False)
        if phase=='joint':
            for name in ('visual','reliable_head','relations','proposal_head','accept_head'):
                getattr(self.original,name).requires_grad_(True)
            self.original.dynamics.rates.requires_grad_(True)
            # Reuse feature projections/pair context, but not unused scalar G heads.
            for name in ('highres_project','decoder_project','local','context','pair'):
                getattr(self.original.gate_relation,name).requires_grad_(True)
        self.original.frozen.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.original.eval() # frozen BN/dropout stats stay fixed, also during joint
        return self

    def set_accept_floor(self, value): self.original.set_accept_floor(value)

    def _context_hook(self, unused, inputs, output):
        highres,decoder,visual,p0,q,r,w,two,one,mode=inputs
        diagnostic=output[2]
        self._context={'feature':torch.cat((diagnostic['highres_projected'],diagnostic['decoder_projected']),1),
            'pair':diagnostic['per_direction_pair_raw'],'local':diagnostic['center'],
            'p0':p0,'q':q,'r':r,'w':w}

    def _source_dynamics(self, visual,p0,q,r,b,a,w):
        state=p0; states=[state]; masses=[]; available_rows=[]
        logs={key:[] for key in ('alpha','beta','gamma','stay','m','transport_delta','proposal_delta',
            'anchor_delta','g_step','state_gate_delta','transport_rate','source_choice','source_edge_logits',
            'source_stay_logit','source_states','source_valid','source_support','legacy_m','source_transport_delta')}
        for _ in range(3):
            support=source_support(state,p0,r)
            mass=torch.zeros_like(state); numerator=torch.zeros_like(state)
            for d,(dy,dx) in enumerate(DIRECTIONS):
                source=w[:,d:d+1]*shifted(support,dy,dx)
                mass=mass+source; numerator=numerator+source*shifted(state,dy,dx)
            available=mass.clamp(0.,1.)
            original_m=torch.where(mass>1e-6,numerator/mass.clamp_min(1e-6),state)
            rates=self.original.dynamics.rates(torch.cat((visual,p0,q,r,b,a,state,available),1)).softmax(1)
            scored=self.selector(self._context,state,support,available,rates[:,1:2])
            choice=scored['choice']; leave=choice[:,1:].sum(1,keepdim=True)
            edge_transport=rates[:,1:2]*choice[:,1:]*(scored['source_states']-state)
            transport=edge_transport.sum(1,keepdim=True)
            selected_mean=torch.where(leave>1e-8,(choice[:,1:]*scored['source_states']).sum(1,keepdim=True)/leave.clamp_min(1e-8),state)
            beta=.2*a*rates[:,2:3]; gamma=rates[:,3:4]
            proposal=beta*(b-state); anchor=gamma*(p0-state)
            new_state=state+transport+proposal+anchor
            # Bounds follow from the convex update; no extra clamp hides a bad budget.
            if not bool(torch.isfinite(new_state).all()) or bool(((new_state < -1e-6)|(new_state > 1.+1e-6)).any()):
                raise RuntimeError('Source selector produced an invalid probability state')
            values={'alpha':rates[:,1:2]*leave,'beta':beta,'gamma':gamma,
                'stay':1.-rates[:,1:2]*leave-beta-gamma,'m':selected_mean,'legacy_m':original_m,
                'transport_delta':transport,'proposal_delta':proposal,'anchor_delta':anchor,
                'g_step':leave,'state_gate_delta':torch.zeros_like(state),'transport_rate':rates[:,1:2],
                'source_choice':choice,'source_edge_logits':scored['edge_logits'],
                'source_stay_logit':scored['stay_logit'],'source_states':scored['source_states'],
                'source_valid':scored['valid'],'source_support':support,'source_transport_delta':edge_transport}
            for key,value in values.items(): logs[key].append(value)
            state=new_state; states.append(state); masses.append(mass); available_rows.append(available)
        return states,masses,available_rows,{key:torch.stack(values,1) for key,values in logs.items()}

    def _dynamics_hook(self, unused, inputs, output):
        if self.mode=='scalar': self._states=output[0]; return None
        replacement=self._source_dynamics(*inputs[:7]); self._states=replacement[0]
        return replacement

    def forward(self,image,capture_diagnostics=False,mode=None,frozen_override=None):
        requested=mode or self.mode
        if requested not in ('scalar','source'): raise ValueError('Expected scalar or source mode')
        previous=self.mode; self.mode=requested; self._context=None; self._states=None
        handles=[self.original.gate_relation.register_forward_hook(self._context_hook),
                 self.original.dynamics.register_forward_hook(self._dynamics_hook)]
        try: result=self.original(image,capture_diagnostics=True,frozen_override=frozen_override)
        finally:
            for handle in handles: handle.remove()
            self.mode=previous
        result['states']=self._states; result['selector_context']=self._context
        result['transport_mode']=requested; result['original_scalar_G_unused']=(requested=='source')
        if requested=='source':
            result['legacy_scalar_G']=result['g_transport']; result['g_transport']=result['g_step'][:,0]
            result['m_note']='m is the post-choice conditional source mean for diagnostics, not an input to the selector; legacy_m retains the old mixed target. Actual transport sums eight source_transport_delta terms without scalar G.'
        return result


def relation_loss(selector, context, reference, labels, selected):
    """Train only shared relation scoring on detached original V9 states."""
    terms=[]; scored=[]; count=0
    for k in range(3):
        state=reference['states'][k].detach()
        result=selector(context,state,source_support(state,context['p0'],context['r']),
            reference['source_available'][:,k].detach(),reference['transport_rate'][:,k].detach())
        take=selected[:,k]; target=labels[:,k]; losses=[]
        for value in (0,1):
            mask=take&(target==value)
            if bool(mask.any()):
                losses.append(F.binary_cross_entropy_with_logits(result['edge_logits'][mask],target[mask].float()))
                count+=int(mask.sum())
        terms.append(torch.stack(losses).mean() if losses else result['edge_logits'].sum()*0.)
        scored.append(result)
    return torch.stack(terms).mean(),scored,count
