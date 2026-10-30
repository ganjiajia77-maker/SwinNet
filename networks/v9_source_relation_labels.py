"""Native-coordinate short-corridor GT relations, not per-pixel BCE gain labels."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from networks.reliable_directional_repair import DIRECTIONS, shifted


@dataclass(frozen=True)
class RelationLabels:
    endpoint_tolerance: float=1.
    corridor_radius: float=1.25
    junction_radius: int=2
    interest_threshold: float=.25
    max_pairs_per_direction: int=64


def _unique_endpoint(road,cy,cx,tolerance):
    """Ignore tolerance disks that snap to multiple disconnected native roads."""
    radius=int(np.ceil(tolerance)); side=2*radius+1
    patch=road[:,:,cy-radius:cy+radius+1,cx-radius:cx+radius+1]
    coord=torch.arange(-radius,radius+1,device=road.device)
    yy,xx=torch.meshgrid(coord,coord,indexing='ij')
    patch=patch&((yy.square()+xx.square())<=tolerance**2)[None,None]
    flat=patch.flatten(1); seed=torch.zeros_like(flat)
    seed.scatter_(1,flat.int().argmax(1,keepdim=True),True); reach=seed.reshape_as(patch)&patch
    for _ in range(side*side): reach=(F.max_pool2d(reach.float(),3,stride=1,padding=1)>.5)&patch
    return ~((patch&~reach).flatten(1).any(1))


@torch.no_grad()
def native_relations(mask,skeleton,work_shape,config=RelationLabels()):
    """Check native GT reachability ONLY inside a bounded p->q short corridor.

    Pixels of GT are never dilated to fill a background gap. Endpoint disks
    provide tolerance; local flood fill is restricted to actual native road GT.
    Junctions, near misses, unsupported endpoints and crop boundaries are ignored.
    No remote GT connected-component identity is used.
    """
    mask=mask.detach().float(); skeleton=skeleton.detach().float()
    b,_,height,width=mask.shape; h,w=work_shape
    if skeleton.shape!=mask.shape: raise ValueError('Native skeleton/mask shape differs')
    sy,sx=height/h,width/w
    if abs(sy-round(sy))>1e-6 or abs(sx-round(sx))>1e-6:
        raise ValueError('Native label mapping requires integer work-grid strides')
    sy,sx=int(round(sy)),int(round(sx)); oy,ox=sy//2,sx//2
    radius=max(sy,sx)+int(np.ceil(config.endpoint_tolerance))+1; side=2*radius+1
    patches=F.unfold(mask[:,:,oy:,ox:],kernel_size=side,padding=radius,stride=(sy,sx))
    if patches.shape[-1]!=h*w: raise ValueError('Native patch/work-grid alignment differs')
    road=patches.transpose(1,2).reshape(b*h*w,1,side,side)>.5
    coordinate=torch.arange(-radius,radius+1,device=mask.device,dtype=torch.float32)
    yy,xx=torch.meshgrid(coordinate,coordinate,indexing='ij')
    start_disk=(yy.square()+xx.square())<=config.endpoint_tolerance**2
    start=(road&start_disk[None,None]); endpoint_p=start.flatten(1).any(1)
    unique_p=_unique_endpoint(road,radius,radius,config.endpoint_tolerance)
    clear_p=~(road&((yy.square()+xx.square())<=(config.endpoint_tolerance+1.)**2)[None,None]).flatten(1).any(1)
    degrees=F.conv2d((skeleton>.5).float(),torch.ones((1,1,3,3),device=mask.device),padding=1)-(skeleton>.5).float()
    junction=(skeleton>.5)&(degrees>=3)
    jr=config.junction_radius
    junction=F.max_pool2d(junction.float(),2*jr+1,stride=1,padding=jr)>.5
    y_native=oy+torch.arange(h,device=mask.device)*sy; x_native=ox+torch.arange(w,device=mask.device)*sx
    j_center=junction[:,:,y_native[:,None],x_native[None,:]]
    in_bounds=((y_native[:,None]>=radius)&(y_native[:,None]<height-radius)&
               (x_native[None,:]>=radius)&(x_native[None,:]<width-radius))[None,None].expand(b,-1,-1,-1)
    outputs=[]; types=[]
    for dy,dx in DIRECTIONS:
        qy,qx=dy*sy,dx*sx
        length2=float(qy*qy+qx*qx)
        t=((yy*qy+xx*qx)/length2).clamp(0.,1.)
        corridor=((yy-t*qy).square()+(xx-t*qx).square())<=config.corridor_radius**2
        end_disk=((yy-qy).square()+(xx-qx).square())<=config.endpoint_tolerance**2
        end=(road&end_disk[None,None]); endpoint_q=end.flatten(1).any(1)
        unique_q=_unique_endpoint(road,radius+qy,radius+qx,config.endpoint_tolerance)
        clear_q=~(road&(((yy-qy).square()+(xx-qx).square())<=(config.endpoint_tolerance+1.)**2)[None,None]).flatten(1).any(1)
        active=torch.nonzero(endpoint_p&endpoint_q,as_tuple=True)[0]
        connected=torch.zeros_like(endpoint_p)
        # A bounded local search; even a distant network connection cannot make
        # a lateral shortcut a positive relation.
        # Empty/background endpoint pairs cannot be positive; skip their flood
        # fill rather than run it across every background work cell.
        if active.numel():
            allowed=road[active]&corridor[None,None]; reach=start[active]&allowed
            for _ in range(int(np.ceil(np.sqrt(length2)))*2+3):
                reach=(F.max_pool2d(reach.float(),3,stride=1,padding=1)>.5)&allowed
            connected[active]=(reach&end[active]).flatten(1).any(1)
        # A curved native path can leave the narrow edge corridor. Ignore these
        # modest local detours instead of treating them as hard shortcut negatives.
        uncertain_path=torch.zeros_like(endpoint_p)
        unresolved=torch.nonzero(endpoint_p&endpoint_q&~connected,as_tuple=True)[0]
        if unresolved.numel():
            wider=((yy-t*qy).square()+(xx-t*qx).square())<=max(2.5,2*config.corridor_radius)**2
            allowed=road[unresolved]&wider[None,None]; reach=start[unresolved]&allowed
            for _ in range(int(np.ceil(np.sqrt(length2)))*2+3):
                reach=(F.max_pool2d(reach.float(),3,stride=1,padding=1)>.5)&allowed
            uncertain_path[unresolved]=(reach&end[unresolved]).flatten(1).any(1)
        positive=endpoint_p&endpoint_q&connected
        negative=~connected&~uncertain_path&((endpoint_p&endpoint_q)|clear_p|clear_q)
        label=torch.full((b*h*w,),-1,dtype=torch.int8,device=mask.device)
        label[positive]=1; label[negative]=0
        label=label.reshape(b,1,h,w)
        ambiguous=j_center|shifted(j_center.float(),dy,dx).bool()|~in_bounds
        valid_source=shifted(torch.ones_like(j_center,dtype=torch.float32),dy,dx)>.5
        unique=(unique_p&unique_q).reshape(b,1,h,w)
        label[ambiguous|~valid_source|~unique]=-1
        category=torch.zeros((b*h*w,),dtype=torch.int8,device=mask.device)
        category[negative&endpoint_p&endpoint_q]=1 # local blocked road pair
        category[negative&(endpoint_p^endpoint_q)]=2 # extension to/from background
        category[negative&~endpoint_p&~endpoint_q]=3 # background response
        outputs.append(label); types.append(category.reshape(b,1,h,w))
    return {'labels':torch.cat(outputs,1),'negative_type':torch.cat(types,1),
        'native_stride':(sy,sx),'endpoint_tolerance':config.endpoint_tolerance,
        'definition':'Native GT local short-corridor reachability proxy; uncertain=-1. Not formal graph labels.'}


@torch.no_grad()
def eligible_relations(reference,native,config=RelationLabels(),ownership=None):
    """No end-point proposal step; consider fixed grid neighbours with model evidence."""
    context=reference['selector_context']; p0=context['p0']; q=context['q']; r=context['r']
    base=native['labels']; labels=base[:,None].expand(-1,3,-1,-1,-1).clone(); valid=[]; action=[]; source=[]
    for k in range(3):
        state=reference['states'][k].detach()
        gained=((state-p0).clamp_min(0.)/(1.-p0).clamp_min(1e-5)).clamp_max(1.)
        support=r+(1.-r)*gained
        selected=[]; actions=[]; sources=[]
        for d,(dy,dx) in enumerate(DIRECTIONS):
            uq=shifted(state,dy,dx); rq=shifted(support,dy,dx)
            interest=torch.maximum(torch.maximum(state,uq),torch.maximum(p0,shifted(p0,dy,dx)))>=config.interest_threshold
            interest=interest|(q>=.5)|(r>=config.interest_threshold)|(rq>=config.interest_threshold)
            if ownership is not None: interest=interest&ownership.bool()
            selected.append(interest&(base[:,d:d+1]>=0))
            actions.append(reference['transport_rate'][:,k].detach()*(uq-state)); sources.append(rq)
        valid.append(torch.cat(selected,1)); action.append(torch.cat(actions,1)); source.append(torch.cat(sources,1))
    return {'labels':labels.detach(),'valid':torch.stack(valid,1).detach(),
            'action':torch.stack(action,1).detach(),'source':torch.stack(source,1).detach()}


@torch.no_grad()
def match_relations(rows,config=RelationLabels(),seed=1234):
    """Match within image/step/direction and action sign, then magnitude/source bins."""
    labels=rows['labels'].cpu().numpy(); valid=rows['valid'].cpu().numpy()
    action=rows['action'].cpu().numpy(); source=rows['source'].cpu().numpy()
    result=np.zeros_like(valid); rng=np.random.default_rng(seed); pairs=0
    magnitude=np.abs(action)
    abin=np.clip(np.floor(np.log2(np.maximum(magnitude,1e-6)/1e-6)),0,30).astype(int)
    vbin=np.digitize(source,(.001,.005,.01,.025,.05,.1,.2,.4,.6,.8))
    bins=abin*11+vbin
    for b in range(valid.shape[0]):
        for k in range(3):
            for d in range(8):
                for sign in (-1,0,1):
                    sign_mask=np.abs(action[b,k,d])<1e-6 if sign==0 else action[b,k,d]*sign>=1e-6
                    p=valid[b,k,d]&sign_mask&(labels[b,k,d]==1)
                    n=valid[b,k,d]&sign_mask&(labels[b,k,d]==0)
                    keys=bins[b,k,d]; budget=config.max_pairs_per_direction
                    for value in rng.permutation(np.intersect1d(np.unique(keys[p]),np.unique(keys[n]))):
                        pp=np.flatnonzero(p.ravel()&(keys.ravel()==value)); nn=np.flatnonzero(n.ravel()&(keys.ravel()==value))
                        amount=min(len(pp),len(nn),budget)
                        if amount==0: break
                        pp=rng.choice(pp,amount,False); nn=rng.choice(nn,amount,False)
                        result[b,k,d].flat[pp]=True; result[b,k,d].flat[nn]=True
                        pairs+=amount; budget-=amount
    return torch.as_tensor(result,device=rows['valid'].device),{'matched_pairs':pairs,
        'natural_positive':int(((labels==1)&valid).sum()),'natural_negative':int(((labels==0)&valid).sum())}
