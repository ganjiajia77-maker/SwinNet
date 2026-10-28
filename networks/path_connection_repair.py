"""A symmetric path classifier and bounded, differentiable nonlocal repair."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from networks.connection_paths import PathConfig, generate_paths, soft_corridor
from networks.reliable_directional_repair import DIRECTIONS, shifted, compose_surface_logits


STAT_NAMES=('p0_mean','p0_min','p0_max','q_mean','q_min','q_max','r_mean','r_min','r_max',
            'w_axis_mean','w_axis_min','source_p0_mean','source_p0_diff','source_r_mean','source_r_diff',
            'length_work','length_native','curvature','width_work','source_distance_work','endpoint_alignment',
            'endpoint_parallel','extension')


def sample_points(feature,points):
    h,w=feature.shape[-2:]
    points=torch.as_tensor(np.asarray(points).copy(),device=feature.device,dtype=feature.dtype)
    grid=torch.stack((2.*(points[:,1]+.5)/w-1.,2.*(points[:,0]+.5)/h-1.),dim=-1)
    return F.grid_sample(feature,grid[None,None],mode='bilinear',padding_mode='border',align_corners=False)[0,:,0]


def reliable_pair_confidence(values):
    product=values[0]*values[1]
    return torch.where(product>0,product.clamp_min(1e-12).sqrt(),torch.zeros_like(product))


def candidate_features(result,context,paths):
    """Endpoint exchange and path reversal preserve the descriptor."""
    center=result['gate_relation_diagnostics']['center']
    feature=torch.cat((center,context),dim=1)
    relation=result['gate_relation_diagnostics']['relation_embedding']
    rows=[]
    for b,items in enumerate(paths):
        vectors=[]
        for path in items:
            points=path['path_points']; endpoints=np.stack((path['endpoint_a'],path['endpoint_b']))
            end=sample_points(feature[b:b+1],endpoints)
            along=sample_points(feature[b:b+1],points).mean(1)
            diffs=np.gradient(points.astype(float),axis=0)
            norm=np.maximum(np.linalg.norm(diffs,axis=1,keepdims=True),1e-6); tangent=diffs/norm
            normals=np.stack((-tangent[:,1],tangent[:,0]),axis=1)*max(path['width_work'],1.)
            left=sample_points(feature[b:b+1],points+normals).mean(1)
            right=sample_points(feature[b:b+1],points-normals).mean(1)
            semantic=torch.cat(((end[:,0]+end[:,1])*.5,(end[:,0]-end[:,1]).abs(),end[:,0]*end[:,1],
                                along,(left+right)*.5,(left-right).abs(),
                                sample_points(relation[b:b+1],points).mean(1)))
            numeric=[]
            for key in ('u0','q','r'):
                values=sample_points(result[key][b:b+1],points)[0]
                numeric.extend((values.mean(),values.min(),values.max()))
            weights=sample_points(result['w'][b:b+1],points)
            dirs=np.asarray(DIRECTIONS,float); dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
            index=np.argmax(tangent@dirs.T,axis=1)
            ids=torch.as_tensor(index,device=weights.device,dtype=torch.long)
            positions=torch.arange(len(points),device=weights.device)
            axial=weights[ids,positions]+weights[ids^1,positions]
            numeric.extend((axial.mean(),axial.min()))
            for key in ('u0','r'):
                source=sample_points(result[key][b:b+1],endpoints)[0]
                numeric.extend((source.mean(),(source[0]-source[1]).abs()))
            ta,tb=path['endpoint_direction']; delta=endpoints[1]-endpoints[0]
            delta=delta/max(float(np.linalg.norm(delta)),1e-6)
            align=[abs(float(np.dot(t,delta))) for t in (ta,tb) if t is not None]
            geometry=(path['path_length_work']/40.,path['path_length_native']/160.,
                      float(np.abs(np.diff(tangent,axis=0)).sum())/max(len(points)-1,1),
                      path['width_work']/3.,float(np.linalg.norm(endpoints[1]-endpoints[0]))/40.,
                      float(np.mean(align)) if align else 0.,abs(float(np.dot(ta,tb))) if ta is not None and tb is not None else 0.,
                      float(path['kind']=='extension'))
            numeric.extend(feature.new_tensor(value) for value in geometry)
            vectors.append(torch.cat((semantic,torch.stack(numeric))))
        rows.append(torch.stack(vectors) if vectors else feature.new_empty((0,13*center.shape[1]+len(STAT_NAMES))))
    return rows


class PathRelationHead(nn.Module):
    def __init__(self,input_features,hidden=64):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_features),nn.Linear(input_features,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.constant_(self.net[-1].bias,-2.)

    def forward(self,features): return self.net(features).squeeze(-1)


def replay_with_suppression(base,result,negative,rho):
    """Same local dynamics; only alpha receives the candidate-local factor."""
    state=result['u0']; p0=state; states=[state]; terms=[]
    for _ in range(base.dynamics.steps):
        gained=((state-p0).clamp_min(0.)/(1.-p0).clamp_min(1e-5)).clamp_max(1.)
        support=result['r']+(1.-result['r'])*gained
        mass=torch.zeros_like(state); numerator=torch.zeros_like(state)
        for d,(dy,dx) in enumerate(DIRECTIONS):
            source=result['w'][:,d:d+1]*shifted(support,dy,dx)
            mass=mass+source; numerator=numerator+source*shifted(state,dy,dx)
        available=mass.clamp(0.,1.)
        m=torch.where(mass>1e-6,numerator/mass.clamp_min(1e-6),state)
        rates=base.dynamics.rates(torch.cat((result['visual'],p0,result['q'],result['r'],result['b'],result['a'],state,available),1)).softmax(1)
        state_delta=base.state_gate(torch.cat((state,m-state,available,rates[:,1:2],result['b']-state,p0-state),1))
        g=torch.sigmoid(result['g_v7_logit']+result['delta_g_logit']+state_delta)
        alpha=rates[:,1:2]*g*(1.-rho*negative)
        beta=.2*result['a']*rates[:,2:3]; gamma=rates[:,3:4]
        terms.append({'transport':alpha*(m-state),'proposal':beta*(result['b']-state),'anchor':gamma*(p0-state)})
        state=(1.-alpha-beta-gamma)*state+alpha*m+beta*result['b']+gamma*p0
        states.append(state)
    return state,states,terms


class PathConnectionRepair(nn.Module):
    def __init__(self,base,config=PathConfig(),hidden=64):
        super().__init__(); self.base=base; self.config=config.validate()
        width=base.gate_relation.local[0].out_channels
        self.path_head=PathRelationHead(13*width+len(STAT_NAMES),hidden)
        self.long_gain=nn.Parameter(torch.tensor(0.))
        self.long_ramp=0.; self.suppress=False; self.rho=.5; self.negative_threshold=.1; self.joint=False
        self.base.requires_grad_(False); self.base.eval()

    def train(self,mode=True):
        super().train(mode)
        if not self.joint: self.base.eval()
        self.base.frozen.eval()
        return self

    def configure(self,phase,lr=1e-4):
        if phase not in ('B','C','D','E','F'): raise ValueError('Unknown path experiment phase')
        self.zero_grad(set_to_none=True)
        self.requires_grad_(False); self.path_head.requires_grad_(phase!='E')
        self.long_gain.requires_grad_(phase in ('C','D','F')); self.joint=phase=='F'
        if self.joint:
            for name in ('reliable_head','relations','gate_relation','state_gate'):
                getattr(self.base,name).requires_grad_(True)
            self.base.dynamics.rates.requires_grad_(True)
        self.base.frozen.requires_grad_(False)
        self.suppress=phase in ('E','F'); self.long_ramp=0. if phase=='B' else 1.
        groups=[]
        for name,parameters,rate in (
            ('path_head',list(self.path_head.parameters()),lr),('long_gain',[self.long_gain],lr),
            ('local_dynamics',[p for p in self.base.parameters() if p.requires_grad],lr*.1)):
            selected=[p for p in parameters if p.requires_grad]
            if selected: groups.append({'params':selected,'lr':rate,'name':name})
        actual=[id(p) for group in groups for p in group['params']]
        if len(actual)!=len(set(actual)) or set(actual)!={id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError('Trainable parameter coverage differs from optimizer groups')
        self.train(phase!='E')
        return groups

    def forward(self,image=None,capture_diagnostics=False,paths_override=None,base_override=None):
        result=self.base(image,capture_diagnostics=True) if base_override is None else base_override
        center=result['gate_relation_diagnostics']['center']
        context=self.base.gate_relation.context(center)
        if paths_override is None:
            paths=[]; generation=[]
            for i in range(center.shape[0]):
                arr=lambda key:result[key][i].detach().float().cpu().numpy()
                semantic=torch.cat((center[i],context[i]),0).detach().float().cpu().numpy()
                items,counts=generate_paths(arr('u0')[0],arr('r')[0],arr('w'),arr('q')[0],semantic,result['p0'].shape[-2:],self.config)
                paths.append(items); generation.append(counts)
        else: paths=paths_override; generation=[]
        features=candidate_features(result,context,paths)
        logits=[self.path_head(v) for v in features]
        positive=[]; negative=[]
        for b,items in enumerate(paths):
            long_map=result['u3'][b:b+1]*0.; neg_map=long_map.clone()
            for i,path in enumerate(items):
                support=torch.as_tensor(soft_corridor(path['path_points'],long_map.shape[-2:],path['width_work']),device=long_map.device,dtype=long_map.dtype)[None,None]
                endpoint=np.stack((path['endpoint_a'],path['endpoint_b']))
                source_r=sample_points(result['r'][b:b+1],endpoint)[0]
                source_p=sample_points(result['u0'][b:b+1],endpoint)[0]
                psrc=(source_r*source_p).sum()/(source_r.sum()+1e-6)
                confidence=reliable_pair_confidence(source_r)
                score=logits[b][i].sigmoid()
                candidate=score*confidence*support*(psrc-result['u3'][b:b+1]).relu()
                long_map=torch.maximum(long_map,candidate)
                neg_score=((self.negative_threshold-score)/self.negative_threshold).clamp(0.,1.)
                neg_map=torch.maximum(neg_map,neg_score*support)
            positive.append(long_map); negative.append(neg_map)
        long_map=torch.cat(positive); neg_map=torch.cat(negative)
        local=result['u3']; suppressed_terms=[]
        if self.suppress and self.long_ramp>0.:
            local,_,suppressed_terms=replay_with_suppression(self.base,result,neg_map,self.rho)
            # Re-evaluate source-to-state differences after the controlled local intervention.
            positive=[]
            for b,items in enumerate(paths):
                value=local[b:b+1]*0.
                for i,path in enumerate(items):
                    end=np.stack((path['endpoint_a'],path['endpoint_b']))
                    sr=sample_points(result['r'][b:b+1],end)[0]; sp=sample_points(result['u0'][b:b+1],end)[0]
                    psrc=(sr*sp).sum()/(sr.sum()+1e-6); confidence=reliable_pair_confidence(sr)
                    support=torch.as_tensor(soft_corridor(path['path_points'],value.shape[-2:],path['width_work']),device=value.device,dtype=value.dtype)[None,None]
                    value=torch.maximum(value,logits[b][i].sigmoid()*confidence*support*(psrc-local[b:b+1]).relu())
                positive.append(value)
            long_map=torch.cat(positive)
        strength=self.long_gain.sigmoid()*self.long_ramp
        final=local+strength*long_map
        if self.long_ramp==0.:
            z1=result['z1']; delta=result['delta_logit']
        else: z1,delta=compose_surface_logits(result['z0'],result['u0'],final)
        out={**result,'z1':z1,'p1':z1.sigmoid(),'u_final':final,'long_delta':strength*long_map,
             'negative_candidate_map':neg_map,'path_logits':logits,'path_features':features,'paths':paths,
             'path_generation':generation,'lambda_long':strength,'suppressed_terms':suppressed_terms,
             'delta_logit':delta,'v9_p1':result['p1'],'v9_z1':result['z1']}
        if not bool(torch.isfinite(final).all()) or bool((final<0).any()) or bool((final>1.+1e-6).any()):
            raise RuntimeError('Nonlocal state violates finite probability bounds')
        return out
