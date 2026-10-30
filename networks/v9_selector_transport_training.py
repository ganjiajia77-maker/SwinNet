"""Joint optimisation utilities; reuse the source-selector forward unchanged."""
import torch
from torch import nn
from torch.nn import functional as F

from networks.v9_source_relation_labels import native_relations,eligible_relations,match_relations

VERSION='v9_source_selector_actual_transport_joint_v1'


class SplitDirectionHead(nn.Module):
    """Keep old scalar-G row OUT of AdamW; train only the eight W rows."""
    def __init__(self,head):
        super().__init__()
        if not isinstance(head,nn.Conv2d) or head.out_channels!=9 or head.kernel_size!=(1,1):
            raise ValueError('Expected original V9 nine-channel direction head')
        self.scalar=nn.Conv2d(head.in_channels,1,1,bias=head.bias is not None).to(head.weight)
        self.directions=nn.Conv2d(head.in_channels,8,1,bias=head.bias is not None).to(head.weight)
        with torch.no_grad():
            self.scalar.weight.copy_(head.weight[:1]); self.directions.weight.copy_(head.weight[1:])
            if head.bias is not None:
                self.scalar.bias.copy_(head.bias[:1]); self.directions.bias.copy_(head.bias[1:])
        self.scalar.requires_grad_(False)

    def forward(self,value): return torch.cat((self.scalar(value),self.directions(value)),1)


def canonical_state(module):
    state=dict(module.state_dict())
    prefix='direction_head.'
    if prefix+'scalar.weight' in state:
        for key in ('weight','bias'):
            if prefix+'scalar.'+key in state:
                state[prefix+key]=torch.cat((state.pop(prefix+'scalar.'+key),state.pop(prefix+'directions.'+key)),0)
    return state


def load_canonical(module,state):
    state=dict(state)
    if isinstance(getattr(module,'direction_head',None),SplitDirectionHead):
        for key in ('weight','bias'):
            if 'direction_head.'+key in state:
                value=state.pop('direction_head.'+key)
                state['direction_head.scalar.'+key]=value[:1]
                state['direction_head.directions.'+key]=value[1:]
    module.load_state_dict(state,strict=True)


def configure_joint(model,rates,selector_lr):
    head=model.original.relations.direction_head
    if not isinstance(head,SplitDirectionHead): model.original.relations.direction_head=SplitDirectionHead(head)
    model.set_phase('joint')
    model.original.relations.direction_head.scalar.requires_grad_(False)
    groups=[{'name':'selector','params':list(model.selector.parameters()),'lr':selector_lr}]
    for name in ('visual','reliable_head','relations','proposal_head','accept_head'):
        parameters=[p for p in getattr(model.original,name).parameters() if p.requires_grad]
        groups.append({'name':name,'params':parameters,'lr':rates[name]})
    groups.append({'name':'rates','params':list(model.original.dynamics.rates.parameters()),'lr':rates['rates']})
    parameters=[p for p in model.original.gate_relation.parameters() if p.requires_grad]
    groups.append({'name':'gate_feature_projections_and_pairs','params':parameters,'lr':rates['gate_relation']})
    actual=[id(p) for group in groups for p in group['params']]
    if len(actual)!=len(set(actual)) or set(actual)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Incorrect joint optimizer parameter coverage')
    excluded=list(model.original.frozen.parameters())+list(model.original.state_gate.parameters())+list(model.original.gate_relation.delta.parameters())+list(model.original.relations.direction_head.scalar.parameters())
    if any(id(p) in actual for p in excluded): raise RuntimeError('Unused scalar G or BASE/Q3 entered optimizer')
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    report={group['name']:{'parameters':sum(p.numel() for p in group['params']),'lr':group['lr']} for group in groups}
    return optimizer,report


def live_relation_loss(result,mask,skeleton,config,seed):
    """Labels/matching detach; loss reads LIVE logits from the NEW state trajectory.

    Same native label definition, interest rule, matched classes and equal-step
    normalisation as the validated relation phase. No second scalar V9 forward.
    """
    native=native_relations(mask,skeleton,result['r'].shape[-2:],config)
    rows=eligible_relations(result,native,config)
    selected,counts=match_relations(rows,config,seed)
    logits=result['source_edge_logits']; terms=[]; per_step=[]
    for k in range(3):
        losses=[]; positives=negatives=0
        for label in (0,1):
            take=selected[:,k]&(rows['labels'][:,k]==label)
            count=int(take.sum())
            if label: positives=count
            else: negatives=count
            if count:
                losses.append(F.binary_cross_entropy_with_logits(logits[:,k][take],rows['labels'][:,k][take].float()))
        terms.append(torch.stack(losses).mean() if losses else logits[:,k].sum()*0.)
        per_step.append({'step':k+1,'positive':positives,'negative':negatives})
    return torch.stack(terms).mean(),{'matching':counts,'steps':per_step,'native':native,'rows':rows,'selected':selected}


def gradient_norm(loss,parameters):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    terms=[g.detach().double().square().sum() for g in gradients if g is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.
