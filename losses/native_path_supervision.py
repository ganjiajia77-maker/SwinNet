"""Detached maximin search with gradient-bearing GT-supervised path logits."""
import time
from collections import Counter
import cv2
import numpy as np
import torch
from losses.cross_scale_path_loss import widest_path,selected_vertex_loss,selected_edge_loss
from losses.path_pair_sampling import sample_pairs
from topology_direction_constants import CONNECTIVITY_DIRECTIONS,CONNECTIVITY_OPPOSITE


def _cells(path):
    out=[]
    for y,x in path:
        c=(y//4,x//4)
        if not out or out[-1]!=c:out.append(c)
    return out


def _path_score_vertices(probs,path):
    return min(float(probs[y,x]) for y,x in path)


def _path_score_edges(probs,path):
    lookup={tuple(d):i for i,d in enumerate(CONNECTIVITY_DIRECTIONS)};values=[]
    for (y,x),(yy,xx) in zip(path,path[1:]):
        d=lookup[(yy-y,xx-x)];values.append(min(probs[d,y,x],probs[CONNECTIVITY_OPPOSITE[d],yy,xx]))
    return float(min(values)) if values else 0.


def _roi_arrays(box,shape):
    y0,y1,x0,x1=box
    a=np.zeros(shape,bool);a[y0:y1,x0:x1]=True
    b=np.zeros((shape[0]//4,shape[1]//4),bool);b[y0//4:(y1+3)//4,x0//4:(x1+3)//4]=True
    return a,b


def _weighted(items,mode,zero):
    if not items:return zero,zero
    if any(x[1] is None for x in items):
        # Unrepresentable C paths must not suppress independently valid
        # surface negatives. Exclude them from C and conflict weighting.
        paired=[x for x in items if x[1] is not None]
        unpaired=[x[0] for x in items if x[1] is None]
        ps,pc=_weighted(paired,mode,zero)
        if paired and unpaired:
            ps=(ps*len(paired)+torch.stack(unpaired).sum())/len(items)
        elif unpaired:ps=torch.stack(unpaired).mean()
        return ps,pc
    sp=torch.stack([x[0] for x in items]);cp=torch.stack([x[1] for x in items])
    if mode=='surface':return sp.mean(),cp.mean()
    errors=np.asarray([[x[2],x[3]] for x in items],dtype=np.float64)
    # Positive bounded weights; normalize each branch within each class so
    # direct supervision budgets are unchanged in the joint ablation.
    diff=np.clip(errors[:,0]-errors[:,1],-1,1)
    ws=1.+.5*diff;wc=1.-.5*diff
    ws=ws/ws.mean();wc=wc/wc.mean()
    return (sp*torch.as_tensor(ws,device=sp.device,dtype=sp.dtype)).mean(),(cp*torch.as_tensor(wc,device=cp.device,dtype=cp.dtype)).mean()


def path_supervision(surface_logits,c_logits,surface_gt,skeleton_gt,native_relations,mode,seed):
    if mode not in ('surface','joint'):raise ValueError('Invalid path training mode')
    if surface_logits.shape[-2:]!=(512,512) or c_logits.shape[-2:]!=(128,128):raise ValueError('Expected 512/128 grids')
    start=time.perf_counter();counts=Counter();examples={'positive':[],'negative':[]}
    probs=surface_logits.detach().sigmoid().cpu().numpy()[:,0];edges=c_logits.detach().sigmoid().cpu().numpy()
    roads=surface_gt.detach().cpu().numpy()[:,0]>.5;skels=skeleton_gt.detach().cpu().numpy()[:,0]>.5
    gt_c=native_relations['connectivity'].detach().cpu().numpy()>.5
    valid=native_relations['connectivity_valid'].detach().cpu().numpy().astype(bool)
    # The target generator already produced the ambiguity mask through its valid edges.
    ambiguity=~valid.any(axis=1)
    for i in range(len(roads)):
        rng=np.random.default_rng(int(seed)+i)
        positives,negatives,stats=sample_pairs(roads[i],skels[i],ambiguity[i],rng)
        counts.update(stats)
        for kind,pairs in (('positive',positives),('negative',negatives)):
          for pair in pairs:
            a,b=pair['a'],pair['b'];allowed,allowed_c=_roi_arrays(pair['roi'],roads[i].shape)
            if kind=='positive':
                # Restrict surface to a GT-supported corridor around the sampled curved path.
                reference=np.zeros_like(roads[i],np.uint8)
                yy,xx=zip(*pair['path']);reference[list(yy),list(xx)]=1
                corridor=cv2.dilate(reference,np.ones((17,17),np.uint8))>0
                allowed &= roads[i]&corridor
                allowed_c &= corridor.reshape(128,4,128,4).any(axis=(1,3))
                edge_ok=valid[i]&gt_c[i]
            else:edge_ok=valid[i]
            spath=widest_path(probs[i],a,b,allowed=allowed)
            ca,cb=(a[0]//4,a[1]//4),(b[0]//4,b[1]//4)
            cpath=widest_path(np.ones((128,128)),ca,cb,allowed=allowed_c,edge_scores=edges[i],edge_valid=edge_ok)
            if spath is None:counts[kind+'_surface_unreachable']+=1;continue
            if any(ambiguity[i,y//4,x//4] for y,x in spath):
                counts[kind+'_path_ambiguous']+=1;continue
            if cpath is None or any(ambiguity[i,y,x] for y,x in cpath):
                if kind=='negative' and any(not roads[i,y,x] for y,x in spath):
                    ls=selected_vertex_loss(surface_logits[i,0],spath,0.,~roads[i])
                    examples[kind].append((ls,None,_path_score_vertices(probs[i],spath),None))
                    counts['negative_surface_only']+=1;counts['negative_used']+=1
                else:counts[kind+'_C_unreachable_or_ambiguous']+=1
                continue
            supervise_s=roads[i] if kind=='positive' else ~roads[i]
            supervise_c=(valid[i]&gt_c[i]) if kind=='positive' else (valid[i]&~gt_c[i])
            if kind=='negative':
                if not any(supervise_s[y,x] for y,x in spath):counts['negative_no_background']+=1;continue
                # At least one eligible negative edge is needed for matched C budget.
                lookup={tuple(d):j for j,d in enumerate(CONNECTIVITY_DIRECTIONS)}
                edge_list=[(lookup[(yy-y,xx-x)],y,x) for (y,x),(yy,xx) in zip(cpath,cpath[1:])]
                if not any(supervise_c[d,y,x] for d,y,x in edge_list):
                    ls=selected_vertex_loss(surface_logits[i,0],spath,0.,supervise_s)
                    examples[kind].append((ls,None,_path_score_vertices(probs[i],spath),None))
                    counts['negative_surface_only']+=1;counts['negative_used']+=1
                    continue
            target=1. if kind=='positive' else 0.
            ls=selected_vertex_loss(surface_logits[i,0],spath,target,supervise_s)
            lc=selected_edge_loss(c_logits[i],cpath,target,supervise_c)
            ss=_path_score_vertices(probs[i],spath);cs=_path_score_edges(edges[i],cpath)
            se=1.-ss if kind=='positive' else ss;ce=1.-cs if kind=='positive' else cs
            examples[kind].append((ls,lc,se,ce));counts[kind+'_used']+=1
            counts[kind+'_both_wrong']+=int(se>=.5 and ce>=.5)
            counts[kind+'_C_better']+=int(ce<se)
            counts[kind+'_surface_better']+=int(se<ce)
            counts[kind+'_surface_connected_05']+=int(ss>=.5)
            counts[kind+'_C_connected_05']+=int(cs>=.5)
    zero=surface_logits.sum()*0+c_logits.sum()*0
    s_losses=[];c_losses=[]
    for kind in ('positive','negative'):
        if examples[kind]:
            s,c=_weighted(examples[kind],mode,zero);s_losses.append(s);c_losses.append(c)
    counts['seconds']=time.perf_counter()-start
    return (torch.stack(s_losses).mean() if s_losses else zero,
            torch.stack(c_losses).mean() if c_losses else zero,dict(counts))
