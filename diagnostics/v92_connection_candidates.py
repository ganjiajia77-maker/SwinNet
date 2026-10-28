"""Prediction-generated path proposals; GT labels are applied only afterwards."""
import hashlib
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from evaluation.road_comparison.road_metrics import skeletonize, skeleton_graph
from networks.reliable_directional_repair import DIRECTIONS


FEATURE_NAMES=('length','endpoint_p0_a','endpoint_p0_b','endpoint_r_a','endpoint_r_b',
               'line_p0','line_q','direction_w','tangent_alignment','type_extension')


def cap_pairs_by_length(proposals, max_pairs):
    """Round-robin sorted prediction-only pair proposals across length bins."""
    by_length=defaultdict(list)
    for proposal in proposals:
        by_length[int(np.digitize(proposal[-1],[4,8,16]))].append(proposal)
    for items in by_length.values():
        items.sort(key=lambda item:(item[1],item[2]))
    selected=[]; cursor=0
    while len(selected)<max_pairs and any(cursor<len(items) for items in by_length.values()):
        for bin_id in sorted(by_length):
            if cursor<len(by_length[bin_id]):
                selected.append(by_length[bin_id][cursor])
                if len(selected)>=max_pairs: break
        cursor+=1
    return selected


def segment(a,b):
    length=float(np.linalg.norm(np.asarray(a)-np.asarray(b)))
    points=np.rint(np.linspace(a,b,max(2,int(np.ceil(length*2))+1))).astype(int)
    return np.unique(points,axis=0),length


def tangent(skeleton, p):
    y,x=p; h,w=skeleton.shape; y0,y1=max(0,y-3),min(h,y+4); x0,x1=max(0,x-3),min(w,x+4)
    pts=np.argwhere(skeleton[y0:y1,x0:x1])+[y0,x0]
    if len(pts)<2: return None
    delta=np.asarray(p)-pts.mean(axis=0); norm=np.linalg.norm(delta)
    return delta/norm if norm>1e-8 else None


def generate(p0,r,w,q,threshold,max_endpoints=96,max_pairs=1200):
    """Proposals from model outputs only. GT is intentionally absent."""
    mask=p0>=threshold; sk=skeletonize(mask); h,width=mask.shape
    count=cv2.filter2D(sk.astype(np.uint8),-1,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)-sk.astype(np.uint8)
    points=np.argwhere(sk & (count==1)); total=int(len(points))
    if len(points)>max_endpoints:
        order=np.argsort(-p0[points[:,0],points[:,1]],kind='stable')[:max_endpoints]; points=points[order]
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    tangents=[tangent(sk,pt) for pt in points]
    proposals=[]
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            a,b=points[i],points[j]; length=float(np.linalg.norm(a-b))
            if length<2 or length>32 or labels[tuple(a)]==labels[tuple(b)]: continue
            proposals.append(('pair',i,j,a,b,length))
    pair_count=len(proposals)
    # Keep the cap from silently discarding long-gap pairs: select round-robin
    # across length bins after sorting within each bin by endpoint indices.
    proposals=cap_pairs_by_length(proposals,max_pairs)
    pair_used=len(proposals)
    for i,a in enumerate(points):
        t=tangents[i]
        if t is None: continue
        for length in (4,8,16):
            b=np.rint(a+t*length).astype(int)
            if not (0<=b[0]<h and 0<=b[1]<width) or mask[tuple(b)]: continue
            proposals.append(('extension',i,None,a,b,float(np.linalg.norm(a-b))))
    out=[]
    for kind,i,j,a,b,length in proposals:
        coords,_=segment(a,b); ys,xs=coords[:,0],coords[:,1]
        direction=(b-a)/max(length,1e-8)
        dots=[float(np.dot(direction,np.asarray(d)/np.linalg.norm(d))) for d in DIRECTIONS]
        axis=int(np.argmax(dots)); opposite=axis^1
        alignment=abs(float(np.dot(tangents[i],direction))) if tangents[i] is not None else 0.
        if j is not None and tangents[j] is not None:
            alignment=(alignment+abs(float(np.dot(tangents[j],direction))))/2.
        opposite_tangent=(abs(float(np.dot(tangents[i],tangents[j])))
                          if j is not None and tangents[i] is not None and tangents[j] is not None else None)
        comp_size=int(stats[labels[tuple(a)],cv2.CC_STAT_AREA])
        features=[length,float(p0[tuple(a)]),float(p0[tuple(b)]),float(r[tuple(a)]),float(r[tuple(b)]),
                  float(p0[ys,xs].mean()),float(q[ys,xs].mean()),
                  float((w[axis,ys,xs]+w[opposite,ys,xs]).mean()),alignment,float(kind=='extension')]
        out.append({'kind':kind,'a_y':int(a[0]),'a_x':int(a[1]),'b_y':int(b[0]),'b_x':int(b[1]),
                    'length_work':length,'direction_channel':axis,'endpoint_tangent_alignment':alignment,
                    'endpoint_tangent_similarity':opposite_tangent,'source_component_area_work':comp_size,
                    'endpoint_p0_a':features[1],'endpoint_p0_b':features[2],
                    'endpoint_r_a':features[3],'endpoint_r_b':features[4],
                    'features':features,'coords':coords})
    return out, {'threshold':threshold,'endpoints_before_cap':total,'endpoints_used':len(points),
                 'pairs_before_cap':pair_count,'pairs_used':pair_used,
                 'pair_cap_policy':'round-robin length bins [0,4], (4,8], (8,16], (16,32]',
                 'extension_proposals':sum(x['kind']=='extension' for x in out)}


def label(proposals,gt_mask,p0,p0_threshold,gap_work):
    """Label using GT after proposal generation. Unknown cases stay unlabeled."""
    h,w=p0.shape; gt=cv2.resize(gt_mask.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)>0
    gt_sk=skeletonize(gt); node_points=np.argwhere(gt_sk)
    if len(node_points):
        node_points,graph=skeleton_graph(gt_sk); tree=cKDTree(node_points)
    else: tree=graph=None
    distance_to_gt=cv2.distanceTransform((~gt_sk).astype(np.uint8),cv2.DIST_L2,3) if len(node_points) else np.full((h,w),1e6)
    cache={}; best={gap:0. for gap in gap_work}
    for c in proposals:
        a=(c['a_y'],c['a_x']); b=(c['b_y'],c['b_x']); coords=c['coords']; ys,xs=coords[:,0],coords[:,1]
        c['gt_line_background_fraction']=float((~gt[ys,xs]).mean())
        c['gt_line_near_skeleton_fraction']=float((distance_to_gt[ys,xs]<=1.5).mean())
        c['gt_label']='uncertain'; c['gt_reason']='endpoint_not_snapped'; c['subtype']='uncertain'; c['gt_graph_distance']=None; gt_path=None
        if tree is not None:
            dist,ids=tree.query(np.asarray([a,b]),k=1)
            c['gt_snap_distance_a']=float(dist[0]); c['gt_snap_distance_b']=float(dist[1])
            if c['kind']=='pair' and max(dist)<=2.5:
                start,end=int(ids[0]),int(ids[1])
                if start not in cache: cache[start]=dijkstra(graph,indices=start,directed=False,return_predecessors=True,limit=80.)
                distances,previous=cache[start]; path_dist=float(distances[end])
                c['gt_graph_distance']=path_dist if np.isfinite(path_dist) else None
                if np.isfinite(path_dist) and path_dist<=max(3.,1.8*c['length_work']):
                    sequence=[end]; cur=end
                    while cur!=start and cur>=0 and len(sequence)<=len(node_points):
                        cur=int(previous[cur]); sequence.append(cur)
                    if cur==start:
                        gt_path=node_points[np.asarray(sequence,dtype=int)]
                        missing_fraction=float((p0[gt_path[:,0],gt_path[:,1]]<p0_threshold).mean())
                        if missing_fraction>0:
                            c.update(gt_label='positive',gt_reason='short_gt_skeleton_path_with_missing_p0',subtype='true_missing_connection')
                        else: c['gt_reason']='gt_path_without_p0_missing'
                elif c['gt_line_background_fraction']>=.2:
                    parallel=(c['endpoint_tangent_similarity'] is not None and c['endpoint_tangent_similarity']>=.75
                              and c['endpoint_tangent_alignment']<.5)
                    c.update(gt_label='negative',gt_reason='gt_disconnected_or_long_detour',
                             subtype='parallel_gap_proxy' if parallel else 'lateral_bridge_proxy')
            elif c['kind']=='extension' and dist[0]<=2.5:
                if c['gt_line_near_skeleton_fraction']>=.75 and (p0[ys,xs]<p0_threshold).any():
                    c.update(gt_label='positive',gt_reason='extension_near_gt_skeleton',subtype='true_extension_proxy'); gt_path=coords
                elif c['gt_line_background_fraction']>=.7 and c['gt_line_near_skeleton_fraction']<.3:
                    c.update(gt_label='negative',gt_reason='extension_far_from_gt',subtype='isolated_extension_proxy' if c['source_component_area_work']<20 else 'false_extension_proxy')
            elif c['gt_line_background_fraction']>=.9:
                c['subtype']='linear_background_uncertain'; c['gt_reason']='unsnapped_sources_not_certified'
        else:
            c['gt_snap_distance_a']=None; c['gt_snap_distance_b']=None
        # Pair-path coverage must not be inflated by one-sided extensions.
        if c['gt_label']=='positive' and c['kind']=='pair' and gt_path is not None:
            path_set=set(map(tuple,gt_path))
            expanded={(y+dy,x+dx) for y,x in path_set for dy in (-1,0,1) for dx in (-1,0,1)}
            for gap,points in gap_work.items():
                if points: best[gap]=max(best[gap],len(points & expanded)/len(points))
        c.pop('coords')
    return proposals,best


def aucs(y,scores,weights=None):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float)
    if len(y)==0 or y.min()==y.max(): return {'n':len(y),'roc_auc':None,'pr_auc':None,'reason':'both labels required'}
    weights=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    order=np.argsort(-scores,kind='stable'); y=y[order]; scores=scores[order]; weights=weights[order]
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    tp=np.cumsum(weights*y)[ends]; fp=np.cumsum(weights*(1-y))[ends]
    positives=int(y.sum()); negatives=len(y)-positives; pos_weight=float((weights*y).sum()); neg_weight=float((weights*(1-y)).sum())
    recall=tp/pos_weight; fpr=fp/neg_weight; precision=tp/(tp+fp)
    roc=float(np.trapz(np.r_[0,recall],np.r_[0,fpr])); ap=float(np.sum(np.diff(np.r_[0,recall])*precision))
    return {'n':len(y),'positive':positives,'negative':negatives,'sample_positive_fraction':positives/len(y),
            'weighted_positive_fraction':pos_weight/(pos_weight+neg_weight),'roc_auc':roc,'pr_auc':ap}


def split_name(case,seed=1234):
    return 'heldout' if int(hashlib.sha256((str(seed)+case).encode()).hexdigest()[:8],16)%5==0 else 'train'


def classifier_audit(rows,seed=1234,population=None):
    labeled=[r for r in rows if r['gt_label'] in ('positive','negative')]
    train=[r for r in labeled if split_name(r['case'],seed)=='train']; test=[r for r in labeled if split_name(r['case'],seed)=='heldout']
    split={'train_images':len({r['case'] for r in train}),'heldout_images':len({r['case'] for r in test}),
           'train_positive':sum(r['gt_label']=='positive' for r in train),'train_negative':sum(r['gt_label']=='negative' for r in train),
           'heldout_positive':sum(r['gt_label']=='positive' for r in test),'heldout_negative':sum(r['gt_label']=='negative' for r in test)}
    if min(split['train_positive'],split['train_negative'],split['heldout_positive'],split['heldout_negative'])==0:
        return {'split':split,'available':False,'population_counts':population,
                'reason':'image-disjoint train and holdout each require both labels'}
    x=np.asarray([r['features'] for r in train],float); y=np.asarray([r['gt_label']=='positive' for r in train],float)
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-8]=1.; x=(x-mean)/scale
    x=np.column_stack((np.ones(len(x)),x)); weights=np.where(y>0,.5/y.sum(),.5/(1-y).sum()); coef=np.zeros(x.shape[1])
    for _ in range(600):
        z=np.clip(x@coef,-30,30); prob=1/(1+np.exp(-z)); grad=x.T@((prob-y)*weights); grad[1:]+=1e-3*coef[1:]; coef-=.2*grad
    xt=(np.asarray([r['features'] for r in test],float)-mean)/scale
    scores=1/(1+np.exp(-np.clip(np.column_stack((np.ones(len(xt)),xt))@coef,-30,30)))
    yt=np.asarray([r['gt_label']=='positive' for r in test],int)
    if population:
        pos=population.get('heldout_positive',0); neg=population.get('heldout_negative',0)
        eval_weights=np.where(yt>0,pos/max(int(yt.sum()),1),neg/max(int((1-yt).sum()),1))
        natural=aucs(yt,scores,eval_weights)
    else: natural=aucs(yt,scores)
    rng=np.random.default_rng(seed); grouped=defaultdict(lambda:[[],[]])
    for i,row in enumerate(test):
        bucket=(row['threshold'],int(np.digitize(row['length_work'],[4,8,16,32])),row['kind'])
        grouped[bucket][yt[i]].append(i)
    matched=[]
    for negative,positive in grouped.values():
        n=min(len(negative),len(positive))
        if n: matched.extend(rng.choice(negative,n,replace=False)); matched.extend(rng.choice(positive,n,replace=False))
    return {'split':split,'available':True,'feature_names':list(FEATURE_NAMES),
            'natural_holdout':natural,'matched_holdout':aucs(yt[matched],scores[matched]),
            'population_counts':population,
            'matching':'equal classes within same threshold, length bin and candidate type',
            'limitations':'Image-disjoint training-set holdout, not val/test model quality; uncertain labels excluded. Natural PR-AUC uses class-prevalence weighting if reservoir-sampled; proposals and subtypes are proxies.'}
