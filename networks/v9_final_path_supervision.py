"""Train-only labels for prediction-generated paths; losses act on final P1."""
import math
from collections import Counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from evaluation.road_comparison.road_metrics import skeleton_graph
from networks.connection_paths import PathConfig, generate_paths, label_paths


def _native_line(path, shape):
    scale=np.asarray((path['scale_y'],path['scale_x']))
    coords=np.rint((path['path_points']+.5)*scale-.5).astype(np.int32)
    canvas=np.zeros(shape,np.uint8)
    cv2.polylines(canvas,[coords[:,::-1].reshape(-1,1,2)],False,1,1)
    y,x=np.nonzero(canvas)
    return np.column_stack((y,x))


def _backtrack(start,end,predecessors):
    points=[int(end)]; current=int(end)
    while current!=start and current>=0 and len(points)<=len(predecessors):
        current=int(predecessors[current]); points.append(current)
    return points[::-1] if current==start else None


def _matched_pairs(items,max_pairs,seed):
    """One-to-one, same-image matching on predicted length/source/P0/Q/distance."""
    rng=np.random.default_rng(seed)
    positives=[i for i,item in enumerate(items) if item['label']==1]
    negatives=[i for i,item in enumerate(items) if item['label']==0]
    rng.shuffle(positives); available=set(negatives); matched=[]
    for i in positives:
        a=items[i]; choices=[]
        for j in available:
            b=items[j]
            if a['kind']!=b['kind']: continue
            length_gap=abs(math.log(max(a['length_native'],1.)/max(b['length_native'],1.)))
            source_gap=abs(a['source']-b['source']); endpoint_gap=abs(a['endpoint_p0']-b['endpoint_p0'])
            distance_gap=abs(a['separation_native']-b['separation_native'])
            q_gap=abs(a['path_q']-b['path_q'])
            if (length_gap>.5 or source_gap>.25 or endpoint_gap>.25 or q_gap>.3 or
                    distance_gap>max(16.,.5*a['separation_native'])): continue
            choices.append((length_gap+source_gap+endpoint_gap+q_gap+distance_gap/max(a['separation_native'],16.),j))
        if choices:
            _,j=min(choices); available.remove(j); matched.append((i,j))
        if len(matched)>=max_pairs: break
    return matched


@torch.no_grad()
def predicted_candidates(model,result,mask,skeleton,reference_threshold=.45,config=PathConfig(),
                         max_candidates=128,max_rank_pairs=64,seed=1234):
    """Proposal generation sees P0/R/W/H/H3/Q only; GT enters after proposal creation."""
    result_items=[]; result_pairs=[]; stats=Counter(); bins=Counter()
    centers=result['gate_relation_diagnostics']['center'].detach()
    contexts=model.gate_relation.context(centers).detach()
    for b in range(mask.shape[0]):
        def array(key): return result[key][b].detach().float().cpu().numpy()
        p0=array('p0')[0]; q=array('q')[0]; r=array('r')[0]; weights=array('w')
        work_p0=array('u0')[0]; feature=torch.cat((centers[b],contexts[b]),0).float().cpu().numpy()
        proposals,counts=generate_paths(work_p0,r,weights,q,feature,p0.shape,config)
        for key,value in counts.items(): stats['generator_'+key]+=int(value)
        gt=mask[b,0].detach().cpu().numpy()>.5; sk=skeleton[b,0].detach().cpu().numpy()>.5
        labels,gaps=label_paths(proposals,gt,sk,p0,reference_threshold,config)
        nodes,graph=skeleton_graph(sk); tree=cKDTree(nodes) if len(nodes) else None
        predecessors={}; chosen=[]
        missing=sk & ~(cv2.dilate((p0>=reference_threshold).astype(np.uint8),np.ones((5,5),np.uint8))>0)
        for path,label in zip(proposals,labels):
            stats['label_'+str(label['label'])]+=1; stats['subtype_'+str(label['subtype'])]+=1
            if label['label']<0: continue
            line=_native_line(path,gt.shape)
            if len(line)<2: stats['ignored_short']+=1; continue
            points=line
            if label['label']==1:
                if tree is None: stats['ignored_unaligned_positive']+=1; continue
                scale=np.asarray((path['scale_y'],path['scale_x']))
                endpoints=np.rint((np.stack((path['endpoint_a'],path['endpoint_b']))+.5)*scale-.5).astype(int)
                _,indices=tree.query(endpoints); start,end=map(int,indices)
                if start==end: stats['ignored_unaligned_positive']+=1; continue
                if start not in predecessors:
                    predecessors[start]=dijkstra(graph,indices=start,directed=False,return_predecessors=True,
                        limit=max(20.,config.max_path_work*max(scale)*2.))
                dist,prev=predecessors[start]; ids=_backtrack(start,end,prev)
                if ids is None or not np.isfinite(dist[end]): stats['ignored_unaligned_positive']+=1; continue
                points=nodes[np.asarray(ids,dtype=int)]
                # Both directions of geometric agreement are checked. No GT path is used for inference.
                raster=np.ones(gt.shape,np.uint8); raster[line[:,0],line[:,1]]=0
                distance=cv2.distanceTransform(raster,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                if ((distance[points[:,0],points[:,1]]<=3.).mean()<.9 or
                        not missing[points[:,0],points[:,1]].any()):
                    stats['ignored_unaligned_positive']+=1; continue
                background=np.empty((0,2),dtype=int)
            else:
                # Positive endpoints are not forced to background: require GT background
                # and stay away from both candidate ends by two work-grid cells.
                scale=np.asarray((path['scale_y'],path['scale_x']))
                ends=(np.stack((path['endpoint_a'],path['endpoint_b']))+.5)*scale-.5
                distance=np.linalg.norm(line[:,None,:]-ends[None,:,:],axis=2).min(axis=1)
                background=line[(~gt[line[:,0],line[:,1]]) & (distance>=2.*float(np.mean(scale)))]
                if len(background)<3: stats['ignored_negative_without_confirmed_background']+=1; continue
            work=np.asarray(path['path_points']); q_mean=float(q[work[:,0],work[:,1]].mean())
            separation=float(np.linalg.norm((path['endpoint_a']-path['endpoint_b'])*
                                             np.asarray((path['scale_y'],path['scale_x']))))
            chosen.append({'label':int(label['label']),'subtype':label['subtype'],
                'kind':path['kind'],'points':points.astype(np.int32),'background':background.astype(np.int32),
                'length_native':float(path['path_length_native']),'length_work':float(path['path_length_work']),
                'source':float(np.mean(path['source_confidence'])),'endpoint_p0':float(np.mean(path['endpoint_p0'])),
                'separation_native':separation,'path_q':q_mean,
                'gt_missing_overlap':int(missing[points[:,0],points[:,1]].sum())})
        # Avoid one image's numerous candidates setting the loss scale.
        positive=[item for item in chosen if item['label']==1]
        negative=[item for item in chosen if item['label']==0]
        rng=np.random.default_rng(seed+b); rng.shuffle(positive); rng.shuffle(negative)
        pos_cap=min(len(positive),max_candidates//2)
        neg_cap=min(len(negative),max_candidates-pos_cap)
        chosen=positive[:pos_cap]+negative[:neg_cap]
        matches=_matched_pairs(chosen,max_rank_pairs,seed+b)
        stats['selected_positive']+=pos_cap; stats['selected_negative']+=neg_cap; stats['matched_pairs']+=len(matches)
        for item in chosen:
            length=item['length_native']; group='3-7' if length<8 else '8-15' if length<16 else '16-31' if length<32 else '32+'
            bins[f'{group}_'+('positive' if item['label']==1 else 'negative')]+=1
        for gap in gaps:
            if not gap['boundary_excluded']:
                bins[gap['length_bin']+'_gaps']+=1
                bins[gap['length_bin']+'_pair_path_full_coverage']+=int(gap['full_coverage'])
        result_items.append(chosen); result_pairs.append(matches)
    return result_items,result_pairs,dict(stats),dict(bins)


def soft_min_score(values,tau=.1,eps=1e-6):
    if tau<=0 or values.numel()==0: raise ValueError('soft-min requires tau>0 and a nonempty path')
    score=-tau*(torch.logsumexp(-values/tau,dim=0)-math.log(values.numel()))
    return score.clamp(eps,1.-eps)


def connection_loss(p1,items,pairs,tau=.1,margin=.1,temperature=.1):
    """Return live positive, negative, background and matched ranking losses."""
    if temperature<=0 or margin<0: raise ValueError('Invalid ranking parameters')
    positive=[]; negative=[]; background=[]; ranking=[]; scores=[]; counts=Counter()
    for b,candidates in enumerate(items):
        image_scores=[]
        for item in candidates:
            points=torch.as_tensor(item['points'],device=p1.device,dtype=torch.long)
            value=p1[b,0,points[:,0],points[:,1]]
            score=soft_min_score(value,tau)
            image_scores.append(score)
            if item['label']==1:
                positive.append(-torch.log(score)); counts['positive_gt_overlap_pixels']+=item['gt_missing_overlap']
            elif item['label']==0:
                negative.append(-torch.log1p(-score))
                bg=torch.as_tensor(item['background'],device=p1.device,dtype=torch.long)
                values=p1[b,0,bg[:,0],bg[:,1]].clamp(1e-6,1.-1e-6)
                background.append(-torch.log1p(-values).mean()); counts['confirmed_background_pixels']+=len(bg)
            else: raise ValueError('Uncertain candidates must not reach loss')
        for pos,neg in pairs[b]:
            if candidates[pos]['label']!=1 or candidates[neg]['label']!=0: raise ValueError('Invalid ranking pair')
            ranking.append(F.softplus((margin-image_scores[pos]+image_scores[neg])/temperature))
        scores.append(image_scores)
    zero=p1.sum()*0.
    mean=lambda entries: torch.stack(entries).mean() if entries else zero
    terms={'pos':mean(positive),'neg':mean(negative),'bg':mean(background),'rank':mean(ranking)}
    counts.update(positive=len(positive),negative=len(negative),matched_pairs=len(ranking))
    for key in ('positive_gt_overlap_pixels','confirmed_background_pixels'): counts.setdefault(key,0)
    return terms,dict(counts)


def weighted_connection(terms,weights):
    return sum((float(weights[key])*terms[key] for key in ('pos','neg','bg','rank')),terms['pos']*0.)
