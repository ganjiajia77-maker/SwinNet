"""Detached local transport-benefit labels; live pre-sigmoid gate supervision."""
from dataclasses import dataclass,asdict
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

BENEFIT_VERSION='v9_detached_transport_benefit_gate_v1'


def restore_model(model,path,identity,expected_steps=None):
    """New checkpoints reject legacy loaders that would silently assume3 steps."""
    import run_reliable_v9_mass as v9
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('benefit_version') is None:
        if expected_steps not in (None,3): raise ValueError('Original V9 is a trained three-step checkpoint')
        return v9.restore_checkpoint(model,path,identity,'D')
    if payload.get('version')!=BENEFIT_VERSION or payload.get('benefit_version')!=BENEFIT_VERSION or payload.get('identity')!=identity:
        raise ValueError('Gate-benefit checkpoint version or identity differs')
    steps=payload.get('benefit_config',{}).get('steps')
    if steps not in (3,6) or expected_steps is not None and steps!=expected_steps:
        raise ValueError('Trained dynamics step count differs')
    if payload.get('stage')!='D' or set(payload.get('modules',{}))!=set(v9.MODULE_NAMES):
        raise ValueError('Incomplete gate-benefit model checkpoint')
    for name in v9.MODULE_NAMES: getattr(model,name).load_state_dict(payload['modules'][name],strict=True)
    model.gate_relation.load_state_dict(payload['gate_relation'],strict=True)
    model.state_gate.load_state_dict(payload['state_gate'],strict=True)
    return payload


@dataclass(frozen=True)
class BenefitConfig:
    min_action: float = 1e-5
    gain_epsilon: float = 1e-7
    road_fraction: float = .75
    background_fraction: float = .05
    max_pairs_per_sign: int = 256
    v_cuts: tuple = (.0001,.0005,.001,.005,.01,.025,.05,.1,.2,.4,.6,.8)

    def validate(self):
        if self.min_action<=0 or self.gain_epsilon<=0 or self.max_pairs_per_sign<1:
            raise ValueError('Positive action/gain thresholds and pair budget required')
        if not 0<=self.background_fraction<self.road_fraction<=1:
            raise ValueError('Invalid unambiguous GT fractions')
        return self


def forward_trace(model,image,steps=3,frozen_override=None):
    """Original model forward. Hook records existing states; no GT argument."""
    if steps not in (3,6,12): raise ValueError('Supported step counts are 3/6/12')
    saved=model.dynamics.steps; captured=[]
    handle=model.dynamics.register_forward_hook(lambda module,args,output:captured.append(output[0]))
    model.dynamics.steps=steps
    try: result=model(image,capture_diagnostics=True,frozen_override=frozen_override)
    finally: handle.remove(); model.dynamics.steps=saved
    if len(captured)!=1 or len(captured[0])!=steps+1: raise RuntimeError('Dynamics trace is incomplete')
    result['states']=captured[0]
    result['gate_logits']=(result['g_v7_logit']+result['delta_g_logit'])[:,None]+result['state_gate_delta']
    if not torch.equal(result['gate_logits'].sigmoid(),result['g_step']):
        raise RuntimeError('Pre-sigmoid logit does not reproduce actual G')
    if not bool(torch.isfinite(result['gate_logits']).all()): raise RuntimeError('Nonfinite gate logits')
    # The original core's u3 key is still the third state; P1 already uses the last state.
    result['last_state']=captured[0][-1]
    return result


@torch.no_grad()
def benefit_labels(result,gt,config=BenefitConfig(),ownership=None):
    config.validate(); states=result['states']; steps=len(states)-1
    size=states[0].shape[-2:]
    fraction=F.interpolate(gt.detach().float(),size=size,mode='area')
    clear=(fraction>=config.road_fraction)|(fraction<=config.background_fraction)
    owned=torch.ones_like(clear,dtype=torch.bool)
    if ownership is not None:
        if ownership.shape!=clear.shape: raise ValueError('Ownership mask shape differs')
        owned=ownership.bool()
    unambiguous=clear&owned
    target=(fraction>=config.road_fraction).float()
    values={key:[] for key in ('off','on','action','gain','actual_gain','valid','target','fraction','source','g','mixed','tiny','tie')}
    for k in range(steps):
        u=states[k].detach()
        off=u+result['proposal_delta'][:,k].detach()+result['anchor_delta'][:,k].detach()
        action=result['transport_rate'][:,k].detach()*(result['m'][:,k].detach()-u)
        on=off+action
        loss=lambda p:F.binary_cross_entropy(p.clamp(1e-6,1.-1e-6),target,reduction='none')
        gain=loss(off)-loss(on)
        valid=unambiguous&(action.abs()>=config.min_action)&(gain.abs()>config.gain_epsilon)
        actual_gain=loss(off)-loss(off+result['g_step'][:,k].detach()*action)
        for key,value in (('off',off),('on',on),('action',action),('gain',gain),('actual_gain',actual_gain),
            ('valid',valid),('target',(gain>0).float()),('fraction',fraction),
            ('mixed',owned&~clear),('tiny',unambiguous&(action.abs()<config.min_action)),
            ('tie',unambiguous&(action.abs()>=config.min_action)&(gain.abs()<=config.gain_epsilon)),
            ('source',result['source_available'][:,k].detach()),('g',result['g_step'][:,k].detach())):
            values[key].append(value)
    return {key:torch.stack(value,1).detach() for key,value in values.items()}


@torch.no_grad()
def match_labels(labels,config=BenefitConfig(),seed=1234):
    valid=labels['valid'].cpu().numpy()[:,:,0]; target=labels['target'].cpu().numpy()[:,:,0]
    action=labels['action'].cpu().numpy()[:,:,0]; source=labels['source'].cpu().numpy()[:,:,0]
    selected=np.zeros_like(valid); rng=np.random.default_rng(seed); differences=[]; counts=Counter()
    for b in range(valid.shape[0]):
        for k in range(valid.shape[1]):
            for sign in (-1,1):
                candidates=valid[b,k] & (action[b,k]*sign>0)
                positive=candidates & (target[b,k]>0); negative=candidates & (target[b,k]<.5)
                counts['allow_before']+=int(positive.sum()); counts['reject_before']+=int(negative.sum())
                if not positive.any() or not negative.any():
                    counts['groups_missing_one_class']+=1; continue
                # log2 bins constrain action magnitudes to the same factor-of-two band.
                magnitude=np.abs(action[b,k])
                abin=np.clip(np.floor(np.log2(np.maximum(magnitude,config.min_action)/config.min_action)),0,30).astype(int)
                vbin=np.digitize(source[b,k],config.v_cuts)
                key=abin*(len(config.v_cuts)+1)+vbin
                common=np.intersect1d(np.unique(key[positive]),np.unique(key[negative])); budget=config.max_pairs_per_sign
                for stratum in rng.permutation(common):
                    p=np.flatnonzero(positive.ravel()&(key.ravel()==stratum))
                    n=np.flatnonzero(negative.ravel()&(key.ravel()==stratum))
                    amount=min(len(p),len(n),budget)
                    if not amount: break
                    p=rng.choice(p,amount,replace=False); n=rng.choice(n,amount,replace=False)
                    selected[b,k].flat[p]=True; selected[b,k].flat[n]=True; budget-=amount
                    for pp,nn in zip(p,n):
                        differences.append((abs(float(magnitude.flat[pp]-magnitude.flat[nn])),
                                            abs(float(source[b,k].flat[pp]-source[b,k].flat[nn]))))
                    counts['pairs']+=amount; counts['pairs_positive_action' if sign>0 else 'pairs_negative_action']+=amount
    for key in ('pairs','pairs_positive_action','pairs_negative_action','allow_before','reject_before'):
        counts.setdefault(key,0)
    counts['matched_allow']=counts['pairs']; counts['matched_reject']=counts['pairs']
    counts['unmatched_valid']=int(valid.sum())-int(selected.sum())
    counts['matching_action_abs_difference_mean']=float(np.mean([row[0] for row in differences])) if differences else None
    counts['matching_V_abs_difference_mean']=float(np.mean([row[1] for row in differences])) if differences else None
    return torch.as_tensor(selected[:,:,None],device=labels['valid'].device),dict(counts)


def gate_benefit_loss(result,gt,config=BenefitConfig(),seed=1234,ownership=None):
    labels=benefit_labels(result,gt,config,ownership)
    selected,matching=match_labels(labels,config,seed)
    logits=result['gate_logits']; target=labels['target']; terms=[]; by_step=[]
    if logits.shape!=target.shape: raise ValueError('Gate logit and label shapes differ')
    for k in range(logits.shape[1]):
        pos=selected[:,k]&(target[:,k]>.5); neg=selected[:,k]&(target[:,k]<.5)
        losses=[]
        if bool(pos.any()): losses.append(F.binary_cross_entropy_with_logits(logits[:,k][pos],torch.ones_like(logits[:,k][pos])))
        if bool(neg.any()): losses.append(F.binary_cross_entropy_with_logits(logits[:,k][neg],torch.zeros_like(logits[:,k][neg])))
        # No fallback to unmatched single-class examples. The empty step is explicit zero.
        terms.append(torch.stack(losses).mean() if losses else logits[:,k].sum()*0.)
        by_step.append({'step':k+1,'allow':int(pos.sum()),'reject':int(neg.sum()),'empty':not bool(losses)})
    return torch.stack(terms).mean(),labels,selected,{'matching':matching,'steps':by_step}


def _hist_auc(positive,negative):
    pos=int(positive.sum()); neg=int(negative.sum())
    if not pos or not neg: return {'roc_auc':None,'pr_auc':None}
    tp=positive[::-1].cumsum(); fp=negative[::-1].cumsum(); tpr=tp/pos; fpr=fp/neg
    precision=tp/np.maximum(tp+fp,1)
    return {'roc_auc':float(np.trapz(np.r_[0.,tpr],np.r_[0.,fpr])),
            'pr_auc':float(np.sum(np.diff(np.r_[0.,tpr])*precision))}


def _exact_auc(positive,negative):
    if not len(positive) or not len(negative): return {'roc_auc':None,'pr_auc':None}
    scores=np.r_[positive,negative]; labels=np.r_[np.ones(len(positive),np.int8),np.zeros(len(negative),np.int8)]
    order=np.argsort(-scores,kind='stable'); sorted_scores=scores[order]
    ends=np.r_[np.nonzero(np.diff(sorted_scores))[0],len(scores)-1]
    tp=labels[order].cumsum()[ends]; fp=ends+1-tp; recall=tp/len(positive); fpr=fp/len(negative)
    precision=tp/(tp+fp)
    return {'roc_auc':float(np.trapz(np.r_[0.,recall],np.r_[0.,fpr])),
            'pr_auc':float(np.sum(np.diff(np.r_[0.,recall])*precision))}


class BenefitMetrics:
    """Streaming natural/matched on-policy decision and action-effect statistics."""
    def __init__(self,steps):
        self.steps=steps; self.data=[Counter() for _ in range(steps)]
        self.hist=np.zeros((steps,2,2,1000),np.int64)
        self.matched_samples={False:[],True:[]}

    def add(self,labels,selected,result):
        arrays={key:value.detach().float().cpu().numpy()[:,:,0] for key,value in labels.items()}
        matched=selected.detach().cpu().numpy()[:,:,0]
        for k in range(self.steps):
            count=self.data[k]; valid=arrays['valid'][:,k]>.5; target=arrays['target'][:,k]>.5; gate=arrays['g'][:,k]
            actual=arrays['actual_gain'][:,k]; action=arrays['action'][:,k]
            count['valid']+=int(valid.sum()); count['masked_out']+=int(valid.size-valid.sum())
            for reason in ('mixed','tiny','tie'): count['ignored_'+reason]+=int(arrays[reason][:,k].sum())
            for population,mask in enumerate((valid,matched[:,k])):
                prefix='natural' if population==0 else 'matched'
                for label,name in ((True,'allow'),(False,'reject')):
                    selected_class=mask&(target==label); vals=gate[selected_class]
                    self.hist[k,population,int(label)]+=np.histogram(vals,bins=1000,range=(0.,1.))[0]
                    count[prefix+'_'+name]+=len(vals); count[prefix+'_'+name+'_G_sum']+=float(vals.sum())
                    count[prefix+'_'+name+'_accepted']+=int((vals>=.5).sum())
                    if population==1 and len(vals): self.matched_samples[label].append(vals.astype(np.float32,copy=True))
            count['harmful_actual_loss_sum']+=float(np.maximum(-actual[valid],0.).sum())
            count['beneficial_actual_gain_sum']+=float(np.maximum(actual[valid],0.).sum())
            for sign,name in ((1,'positive_action'),(-1,'negative_action')):
                for label,labelname in ((True,'allow'),(False,'reject')):
                    take=valid&(action*sign>0)&(target==label)
                    count[name+'_'+labelname]+=int(take.sum())
            for region,take in (('road',valid&(arrays['fraction'][:,k]>=.75)),
                                ('background',valid&(arrays['fraction'][:,k]<=.05))):
                count[region+'_pixels']+=int(take.sum())
                count[region+'_transport_sum']+=float((gate*action)[take].sum())
                count[region+'_transport_abs_sum']+=float(np.abs(gate*action)[take].sum())
                count[region+'_V_sum']+=float(arrays['source'][:,k][take].sum())
            state=result['states'][k].detach(); p0=result['states'][0].detach(); reliable=result['r'].detach()
            gained=((state-p0).clamp_min(0.)/(1.-p0).clamp_min(1e-5)).clamp_max(1.)
            added_source=((1.-reliable)*gained)[:,0].cpu().numpy()
            bg=valid&(arrays['fraction'][:,k]<=.05)
            count['background_generated_source_sum']+=float(added_source[bg].sum())

    def report(self):
        rows=[]; global_counts=Counter()
        for k,counter in enumerate(self.data):
            row={'step':k+1,**dict(counter)}; n=max(counter['valid'],1)
            row['harmful_actual_loss_per_valid_action']=counter['harmful_actual_loss_sum']/n
            row['beneficial_actual_gain_per_valid_action']=counter['beneficial_actual_gain_sum']/n
            for population,prefix in enumerate(('natural','matched')):
                row[prefix+'_auc_hist_approx']=_hist_auc(self.hist[k,population,1],self.hist[k,population,0])
                for label in ('allow','reject'):
                    count=counter[prefix+'_'+label]
                    row[prefix+'_'+label+'_G_mean']=counter[prefix+'_'+label+'_G_sum']/count if count else None
                    row[prefix+'_'+label+'_acceptance_rate']=counter[prefix+'_'+label+'_accepted']/count if count else None
            rows.append(row); global_counts.update(counter)
        n=max(global_counts['valid'],1)
        def acceptance(label):
            total=global_counts['matched_'+label]
            return global_counts['matched_'+label+'_accepted']/total if total else None
        exact=_exact_auc(np.concatenate(self.matched_samples[True]) if self.matched_samples[True] else np.empty(0),
                         np.concatenate(self.matched_samples[False]) if self.matched_samples[False] else np.empty(0))
        return {'steps':rows,'valid':int(global_counts['valid']),'matched_auc_exact':exact,
            'matched_allow':int(global_counts['matched_allow']),'matched_reject':int(global_counts['matched_reject']),
            'matched_helpful_acceptance_rate':acceptance('allow'),'matched_harmful_acceptance_rate':acceptance('reject'),
            'harmful_actual_loss_per_valid_action':global_counts['harmful_actual_loss_sum']/n,
            'beneficial_actual_gain_per_valid_action':global_counts['beneficial_actual_gain_sum']/n,
            'limitation':'On-policy action populations change with the model. Matching controls sign/V/action bands within image and step; this is not a fixed-input gate-only causal comparison. Hist-AUC resolution is .001.'}
