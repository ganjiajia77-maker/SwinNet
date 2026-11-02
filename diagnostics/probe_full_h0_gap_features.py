"""Frozen B feature probe: low-P0 missed GT skeleton vs nearby hard background.

Candidates are generated from prediction only on the native 4-pixel grid.
GT is used later for proxy labels and uncertainty exclusions, never as input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from diagnostics.audit_full_h0_saved_predictions import digest, read_json, write_csv
from diagnostics.diagnose_full_h0_optimistic_reachability import (
    atomic_json, atomic_npz, rebuild_model, signature, verify_recorded_code,
)
from run_q3_r3_mass import full_dataset
from run_slim_end_to_end_mass import seed_all

P0_THRESHOLD = .45
MATCH_TOLERANCE = np.array([.05, 8., .15, 16.], dtype=np.float32)
DIM_BASE, DIM_Q, DIM_H, DIM_H3 = 4, 1, 16, 16
GROUPS = {
    'base': np.arange(4),
    'base_q': np.arange(5),
    'base_q_h': np.arange(21),
    'base_q_h_h3': np.arange(37),
    'base_q_relation': np.r_[np.arange(5), np.arange(37, 311)],
    'all': np.arange(311),
}


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256((str(seed)+'|'+name).encode()).digest()[:4], 'little')


def component_distances(pred, limit=64, min_area=8):
    """Exact nearest two distinct predicted components, capped beyond limit."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    first = np.full(pred.shape, limit+1., np.float32)
    second = first.copy()
    first_id=np.zeros(pred.shape,np.int32);second_id=first_id.copy()
    h, w = pred.shape
    for k in range(1, n):
        x, y, width, height, area = map(int, stats[k])
        if area < min_area: continue
        x0, x1 = max(0,x-limit), min(w,x+width+limit)
        y0, y1 = max(0,y-limit), min(h,y+height+limit)
        dist = cv2.distanceTransform((labels[y0:y1,x0:x1] != k).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        a, b = first[y0:y1,x0:x1], second[y0:y1,x0:x1]
        ai,bi=first_id[y0:y1,x0:x1],second_id[y0:y1,x0:x1]
        old_a=a.copy();old_ai=ai.copy();better=dist<old_a
        bi[:]=np.where(better,old_ai,np.where(dist<b,k,bi))
        b[:]=np.where(better,old_a,np.minimum(b,dist))
        ai[:]=np.where(better,k,ai);a[:]=np.minimum(old_a,dist)
    return first, second, first_id, second_id, labels


def bridge_added_proxy(pred, after):
    """Prediction-only added components touching two distinct old components."""
    _, old=cv2.connectedComponents(pred.astype(np.uint8),connectivity=8)
    n, labels, stats, _=cv2.connectedComponentsWithStats((after&~pred).astype(np.uint8),connectivity=8)
    result=np.zeros(pred.shape,bool)
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<3: continue
        x0,x1=max(0,x-2),min(pred.shape[1],x+w+2);y0,y1=max(0,y-2),min(pred.shape[0],y+h+2)
        region=labels[y0:y1,x0:x1]==k
        near=cv2.dilate(region.astype(np.uint8),np.ones((5,5),np.uint8))>0
        old_ids=np.unique(old[y0:y1,x0:x1][near]);old_ids=old_ids[old_ids>0]
        if len(old_ids)>=2: result[y0:y1,x0:x1] |= region
    return result


def make_candidates(p0, p1, gt, skeleton, reference_ids=None, max_pos_per_component=8, max_neg_per_type=300, seed=1):
    """Prediction-defined candidate grid; native GT only assigns proxy labels."""
    if p0.shape != (512,512) or p1.shape != p0.shape or gt.shape != p0.shape:
        raise ValueError('Expected one native 512 crop')
    pred = p0 >= P0_THRESHOLD
    d1, d2, first_id, second_id, pred_labels = component_distances(pred)
    density = cv2.boxFilter(pred.astype(np.float32), -1, (33,33), normalize=True)
    yy, xx = np.meshgrid(np.arange(2,512,4), np.arange(2,512,4), indexing='ij')
    low = p0[yy,xx] < P0_THRESHOLD
    # Fixed inference-only candidate rule, chosen before validation.
    eligible = low & (d1[yy,xx] >= 2.) & (d1[yy,xx] <= 64.)
    to_gt = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Exclude ambiguous junction neighbourhoods; retain endpoints and one-sided gaps.
    degrees = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3,3),np.uint8)) - skeleton.astype(np.uint8)
    junction = cv2.dilate((skeleton & (degrees >= 3)).astype(np.uint8), np.ones((9,9),np.uint8)) > 0
    uncovered = skeleton & (cv2.distanceTransform((~pred).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) > 2.) & ~junction
    count, missing_ids, stats, _ = cv2.connectedComponentsWithStats(uncovered.astype(np.uint8),connectivity=8)
    grid_ids = np.zeros((128,128),np.int32); collision = np.zeros_like(grid_ids,dtype=bool)
    ys, xs = np.nonzero(uncovered)
    for y, x in zip(ys.tolist(),xs.tolist()):
        gy,gx=y//4,x//4; label=int(missing_ids[y,x]); old=grid_ids[gy,gx]
        if old and old != label: collision[gy,gx]=True
        else: grid_ids[gy,gx]=label
    component_sizes=stats[:,cv2.CC_STAT_AREA]
    positive = eligible & (grid_ids > 0) & (component_sizes[grid_ids]>=3) & ~collision
    negative = eligible & (to_gt[yy,xx] > 3.) & ~junction[yy,xx] & ~positive
    negative_kind = np.where(d2[yy,xx] <= 64., 1, 2).astype(np.int8)
    # These are prediction/GT proxies, not verified semantic parallel-road labels.
    added_fp = (p1[yy,xx] >= P0_THRESHOLD) & (p0[yy,xx] < P0_THRESHOLD)
    negative_kind[negative & added_fp] = 3
    bridge=bridge_added_proxy(pred,p1>=P0_THRESHOLD)
    negative_kind[negative & bridge[yy,xx]] = 4
    rng=np.random.default_rng(seed); chosen=[]; sample_weights=[]
    for comp in np.unique(grid_ids[positive]):
        positions=np.flatnonzero((positive & (grid_ids==comp)).ravel())
        take=min(len(positions),max_pos_per_component)
        selected=rng.choice(positions,size=take,replace=False)
        chosen.extend(selected.tolist());sample_weights.extend([len(positions)/take]*take)
    for kind in (1,2,3,4):
        positions=np.flatnonzero((negative & (negative_kind==kind)).ravel())
        take=min(len(positions),max_neg_per_type)
        if take:
            selected=rng.choice(positions,size=take,replace=False)
            chosen.extend(selected.tolist());sample_weights.extend([len(positions)/take]*take)
    chosen=np.asarray(chosen,dtype=np.int64)
    gy,gx=np.divmod(chosen,128); y,x=yy[gy,gx],xx[gy,gx]
    labels=positive[gy,gx].astype(np.int8)
    components=grid_ids[gy,gx]
    lengths=np.array([int(stats[c,cv2.CC_STAT_AREA]) if c else 0 for c in components],np.int32)
    # Near each missing component's extrema is only a spatial flank proxy.
    flank=np.zeros(len(chosen),np.int8)
    for comp in np.unique(components[components>0]):
        sy,sx=np.nonzero(missing_ids==comp)
        selected=np.flatnonzero(components==comp)
        if len(sy):
            coords=np.column_stack((sy,sx)).astype(np.float32)
            axis=np.linalg.svd(coords-coords.mean(0),full_matrices=False)[2][0] if len(coords)>1 else np.array([1.,0.])
            proj=coords@axis; cproj=np.column_stack((y[selected],x[selected]))@axis
            flank[selected]=((cproj<=proj.min()+8)|(cproj>=proj.max()-8)).astype(np.int8)
    ids=np.zeros(len(chosen),np.int32) if reference_ids is None else reference_ids[y,x].astype(np.int32)
    # Endpoints and corridor come only from the two predicted components;
    # GT is used afterwards to report how much of that polyline it supports.
    end_a=np.full((len(chosen),2),-1,np.int32);end_b=end_a.copy()
    comp_a=first_id[y,x];comp_b=second_id[y,x]
    for comp in np.unique(np.r_[comp_a,comp_b]):
        if comp==0:continue
        positions=np.column_stack(np.nonzero(pred_labels==comp))
        tree=cKDTree(positions)
        for numbers,output in ((np.flatnonzero(comp_a==comp),end_a),(np.flatnonzero(comp_b==comp),end_b)):
            if len(numbers):
                _,idx=tree.query(np.column_stack((y[numbers],x[numbers])),k=1)
                output[numbers]=positions[idx]
    corridor_gt=np.full(len(chosen),np.nan,np.float32);corridor_length=np.zeros(len(chosen),np.float32)
    for i in np.flatnonzero((comp_a>0)&(comp_b>0)):
        a=end_a[i];b=end_b[i];center=np.array([y[i],x[i]])
        pieces=[]
        for left,right in ((a,center),(center,b)):
            span=float(np.linalg.norm(right-left));corridor_length[i]+=span
            n=max(2,int(np.ceil(span))+1)
            pieces.append(np.rint(np.linspace(left,right,n)).astype(np.int32))
        points=np.unique(np.concatenate(pieces,axis=0),axis=0)
        corridor_gt[i]=float(gt[points[:,0],points[:,1]].mean())
    return {'gy':gy,'gx':gx,'y':y,'x':x,'label':labels,'weight':np.asarray(sample_weights,np.float32),
            'component':components,'length':lengths,'flank_proxy':flank,'fixed_gap_id':ids,
            'corridor_gt_fraction':corridor_gt,'corridor_length':corridor_length,
            'negative_kind':np.where(labels,0,negative_kind[gy,gx]),
            'nuisance':np.stack((p0[y,x],d1[y,x],density[y,x],d2[y,x]),1).astype(np.float32),
            'counts':{'prediction_candidates':int(eligible.sum()),'positive_candidates':int(positive.sum()),
                      'negative_candidates':int(negative.sum()),'uncertain_candidates':int((eligible & ~positive & ~negative).sum()),
                      'junction_excluded':int((eligible & junction[yy,xx]).sum()),
                      'grid_collisions':int((eligible & collision).sum()),'missing_components':int(count-1),
                      'missing_components_at_least_3_pixels':int((component_sizes[1:]>=3).sum())}}


def match_image(labels, nuisance, tolerance=MATCH_TOLERANCE, seed=1):
    """One-to-one same-image matches, all four predeclared calipers enforced."""
    positives=np.flatnonzero(labels==1); negatives=np.flatnonzero(labels==0)
    if not len(positives) or not len(negatives): return np.empty((0,2),np.int64)
    rng=np.random.default_rng(seed); positives=rng.permutation(positives)
    scaled=nuisance/tolerance
    tree=cKDTree(scaled[negatives]); used=set(); pairs=[]
    for p in positives:
        # Distinct-component presence must agree; 65 is the censored value.
        candidates=tree.query_ball_point(scaled[p],2.00001)
        candidates=[negatives[i] for i in candidates if negatives[i] not in used and
                    (nuisance[p,3] <= 64)==(nuisance[negatives[i],3] <= 64) and
                    bool((np.abs(nuisance[p]-nuisance[negatives[i]])<=tolerance+1e-6).all())]
        if candidates:
            q=min(candidates,key=lambda z:(float(np.square(scaled[p]-scaled[z]).sum()),int(z)))
            used.add(q); pairs.append((int(p),int(q)))
    return np.asarray(pairs,dtype=np.int64).reshape(-1,2)


def features_from_result(result, sample):
    gy,gx=sample['gy'],sample['gx']; ctx=result['selector_context']
    def pick(tensor):
        return tensor[0].detach().cpu().numpy()[:,gy,gx].T
    projected=pick(ctx['feature']);h,w=ctx['feature'].shape[-2:]
    pair=ctx['pair'][0].detach().cpu().numpy().reshape(256,h,w)[:,gy,gx].T
    q=pick(result['q']); r=pick(result['r']); w=pick(result['w']); choice=pick(result['source_choice'][:,0])
    relation=np.concatenate((r,w,pair,choice),1)
    x=np.concatenate((sample['nuisance'],q,projected[:,:16],projected[:,16:],relation),1).astype(np.float32)
    if x.shape[1]!=311 or not np.isfinite(x).all(): raise ValueError(f'Unexpected model feature dimensions: {x.shape}')
    return x


def subset_sample(sample, indices):
    return {k:v[indices] for k,v in sample.items() if isinstance(v,np.ndarray)}


def tile_positions():
    """512/256 windows with deterministic, nonoverlapping ownership cores."""
    intervals=((0,0,384),(256,384,640),(512,640,896),(768,896,1134),(988,1134,1500))
    return [(top,left,y0,y1,x0,x1) for top,y0,y1 in intervals for left,x0,x1 in intervals]


@torch.inference_mode()
def extract_image(model, sample, split, seed, reference=None):
    image=sample['image']; mask=sample['mask'][0].numpy()>.5; skeleton=sample['skeleton'][0].numpy()>.5
    name=str(sample['case_name']); n=mask.shape[0]
    if (n,mask.shape[1])!=(1500,1500): raise ValueError('Expected original 1500x1500 images')
    if split=='val': positions=tile_positions()
    else:
        rng=np.random.default_rng(stable_seed(seed,name)); top=int(rng.integers(0,989));left=int(rng.integers(0,989))
        positions=[(top,left,top,top+512,left,left+512)]
    ref_map=np.zeros((1500,1500),np.int32)
    if reference:
        for gap in reference['gaps']:
            ref_map.reshape(-1)[np.asarray(gap['pixels'],np.int64)] = int(gap['id'])
        # Native gap ID is a proximity index; it is never used as an input feature.
        ref_map=cv2.dilate(ref_map.astype(np.float32),np.ones((5,5),np.uint8)).astype(np.int32)
    chunks=[];counts=[];p0_map=np.zeros((1500,1500),np.float16) if split=='val' else np.empty((0,0),np.float16)
    for i,(top,left,y0,y1,x0,x1) in enumerate(positions):
        result=model(image[None,:,top:top+512,left:left+512].to(next(model.parameters()).device))
        p0=result['p0'][0,0].detach().cpu().numpy();p1=result['p1'][0,0].detach().cpu().numpy()
        if split=='val': p0_map[y0:y1,x0:x1]=p0[y0-top:y1-top,x0-left:x1-left]
        cand=make_candidates(p0,p1,mask[top:top+512,left:left+512],skeleton[top:top+512,left:left+512],
                             ref_map[top:top+512,left:left+512],seed=stable_seed(seed,name+':'+str(i)))
        # Generate on the entire tile before applying the predetermined spatial core.
        keep=(cand['y']+top>=y0)&(cand['y']+top<y1)&(cand['x']+left>=x0)&(cand['x']+left<x1)
        selected=np.flatnonzero(keep); x=features_from_result(result,cand)[selected]
        item=subset_sample(cand,selected);item['features']=x
        item['native_y']=item['y']+top;item['native_x']=item['x']+left
        item['tile_index']=np.full(len(selected),i,np.int16)
        chunks.append(item);counts.append({'tile':i,'counts_before_core':cand['counts'],'selected_in_core':int(len(selected))})
        if split=='val' and ((i+1)%5==0 or i+1==len(positions)):
            print(f'[VAL TILE] {name} {i+1}/{len(positions)}',flush=True)
    keys=chunks[0].keys()
    merged={key:np.concatenate([c[key] for c in chunks],axis=0) for key in keys}
    merged['counts_json']=np.asarray(json.dumps(counts))
    merged['input_sha256']=np.asarray(signature(sample))
    merged['p0_map']=p0_map
    return merged


def image_digest(sample):
    return hashlib.sha256(sample['image'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def binary_metrics(y, score, weights=None):
    y=np.asarray(y,dtype=np.int8);score=np.asarray(score,dtype=np.float64)
    weights=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=np.float64)
    pos=float(weights[y==1].sum());neg=float(weights[y==0].sum())
    result={'n':len(y),'positive':int((y==1).sum()),'negative':int((y==0).sum()),
            'weighted_positive_fraction':pos/(pos+neg) if pos+neg else None,
            'roc_auc':None,'pr_auc':None,'fpr_at_80_recall':None,'fpr_at_90_recall':None}
    if not pos or not neg: return result
    order=np.argsort(-score,kind='mergesort'); yy=y[order];ss=score[order];ww=weights[order]
    ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
    tp=np.cumsum(ww*(yy==1))[ends];fp=np.cumsum(ww*(yy==0))[ends]
    recall=tp/pos;fpr=fp/neg;precision=tp/(tp+fp)
    result['pr_auc']=float((np.diff(np.r_[0.,recall])*precision).sum())
    r=np.r_[0.,recall];f=np.r_[0.,fpr]
    result['roc_auc']=float((np.diff(f)*(r[1:]+r[:-1])*.5).sum())
    for target in (.8,.9):
        choices=np.flatnonzero(recall>=target)
        result[f'fpr_at_{int(100*target)}_recall']=float(fpr[choices[0]]) if len(choices) else None
    return result


def threshold_for_recall(y, score, target):
    y=np.asarray(y);score=np.asarray(score)
    if not (y==1).any(): return None
    positives=np.sort(score[y==1])[::-1]
    return float(positives[max(0,min(len(positives)-1,int(np.ceil(target*len(positives)))-1))])


def evaluate_threshold(y,score,threshold,weights=None):
    if threshold is None: return {'tpr':None,'fpr':None}
    weights=np.ones(len(y)) if weights is None else weights
    pos=weights[y==1].sum();neg=weights[y==0].sum()
    return {'tpr':float(weights[(y==1)&(score>=threshold)].sum()/pos) if pos else None,
            'fpr':float(weights[(y==0)&(score>=threshold)].sum()/neg) if neg else None}


def fit_probe(train_x,train_y,hold_x,hold_y,kind,seed):
    if len(np.unique(train_y))<2 or len(np.unique(hold_y))<2:
        raise ValueError('Both labels are required in train fit and train-image holdout')
    mean=train_x.mean(0);std=np.maximum(train_x.std(0),1e-4)
    a=torch.from_numpy(np.clip((train_x-mean)/std,-10,10).astype(np.float32))
    b=torch.from_numpy(np.clip((hold_x-mean)/std,-10,10).astype(np.float32))
    ya=torch.from_numpy(train_y.astype(np.float32)).reshape(-1,1)
    yb=torch.from_numpy(hold_y.astype(np.float32)).reshape(-1,1)
    torch.manual_seed(seed)
    model=nn.Linear(a.shape[1],1) if kind=='linear' else nn.Sequential(nn.Linear(a.shape[1],32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.001)
    best=None;best_loss=float('inf');best_step=0
    for step in range(1,201):
        model.train();optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(a),ya)
        loss.backward();optimizer.step()
        if step%10==0:
            model.eval()
            with torch.no_grad(): value=float(nn.functional.binary_cross_entropy_with_logits(model(b),yb))
            if value<best_loss:
                best_loss=value;best_step=step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best);model.eval()
    def predict(x):
        data=torch.from_numpy(np.clip((x-mean)/std,-10,10).astype(np.float32))
        with torch.no_grad(): return model(data).numpy().reshape(-1)
    return predict,{'best_step':best_step,'holdout_bce':best_loss}


def smd_report(y, nuisance, weights):
    rows=[]
    for i,name in enumerate(('p0','distance_to_road','road_density','distance_to_second_component')):
        values=nuisance[:,i];means=[];variances=[]
        for label in (1,0):
            take=y==label;w=weights[take];v=values[take]
            if not len(v) or not w.sum(): means.append(None);variances.append(None);continue
            m=np.average(v,weights=w);means.append(float(m));variances.append(float(np.average((v-m)**2,weights=w)))
        denom=np.sqrt((variances[0]+variances[1])/2) if None not in variances else None
        smd=(means[0]-means[1])/denom if denom and denom>1e-12 else (0. if denom is not None and means[0]==means[1] else None)
        rows.append({'field':name,'positive_mean':means[0],'negative_mean':means[1],'smd':float(smd) if smd is not None else None})
    return rows


def combine(records):
    keys=('features','label','weight','nuisance','native_y','native_x','component','length','flank_proxy',
          'fixed_gap_id','negative_kind','tile_index','corridor_gt_fraction','corridor_length')
    result={key:np.concatenate([r[key] for r in records],axis=0) for key in keys}
    result['case']=np.concatenate([np.full(len(r['label']),r['case'],dtype='U128') for r in records])
    result['split']=np.concatenate([np.full(len(r['label']),r['split'],dtype='U16') for r in records])
    matched=[];match_rows=[];offset=0
    for r in records:
        pairs=match_image(r['label'],r['nuisance'],seed=stable_seed(1234,r['case']))
        matched.extend((pairs+offset).reshape(-1).tolist());offset+=len(r['label'])
        match_rows.append({'case':r['case'],'split':r['split'],'positive':int((r['label']==1).sum()),
                           'negative':int((r['label']==0).sum()),'matched_pairs':len(pairs),
                           'unmatched_positive':int((r['label']==1).sum()-len(pairs)),
                           'unmatched_negative':int((r['label']==0).sum()-len(pairs))})
    result['matched_indices']=np.asarray(matched,np.int64)
    return result,match_rows


def bootstrap_by_image(y,score,weight,case,seed=5,repetitions=100):
    names=np.unique(case);rng=np.random.default_rng(seed);vals={'roc_auc':[],'pr_auc':[],'fpr_at_90_recall':[]}
    if len(names)<2: return {key:None for key in vals}
    indices={name:np.flatnonzero(case==name) for name in names}
    for _ in range(repetitions):
        picked=rng.choice(names,size=len(names),replace=True)
        ix=np.concatenate([indices[name] for name in picked])
        m=binary_metrics(y[ix],score[ix],weight[ix])
        for key in vals:
            if m[key] is not None: vals[key].append(m[key])
    return {key:([float(x) for x in np.quantile(v,[.025,.975])] if len(v)>=10 else None) for key,v in vals.items()}


def paired_delta_ci(y,a,b,weight,case,repetitions=100):
    names=np.unique(case);rng=np.random.default_rng(321);indices={name:np.flatnonzero(case==name) for name in names}
    values={key:[] for key in ('roc_auc','pr_auc','fpr_at_80_recall','fpr_at_90_recall')}
    if len(names)<2:return {key:None for key in values}
    for _ in range(repetitions):
        ix=np.concatenate([indices[name] for name in rng.choice(names,size=len(names),replace=True)])
        one=binary_metrics(y[ix],a[ix],weight[ix]);two=binary_metrics(y[ix],b[ix],weight[ix])
        for key in values:
            if one[key] is not None and two[key] is not None:values[key].append(one[key]-two[key])
    return {key:([float(v) for v in np.quantile(rows,[.025,.975])] if len(rows)>=10 else None) for key,rows in values.items()}


def evaluate(data, output, reps):
    x=data['features'];y=data['label'];split=data['split'];case=data['case'];weight=data['weight']
    matched=np.zeros(len(y),bool);matched[data['matched_indices']]=True
    masks={'fit':(split=='fit')&matched,'hold':(split=='hold')&matched,
           'val_natural':split=='val','val_matched':(split=='val')&matched}
    if not masks['fit'].any() or not masks['hold'].any():
        raise ValueError('Insufficient within-image positive/negative matches in training fit or training holdout')
    if not masks['val_matched'].any(): raise ValueError('No matched validation samples; report matching failure rather than infer a result')
    results=[];per_image=[];subgroups=[];score_rows=[];predictions={};gap_detection=[]
    for kind in ('linear','mlp'):
        for group,indices in GROUPS.items():
            print(f'[PROBE] {kind} {group}',flush=True)
            predict,training=fit_probe(x[masks['fit']][:,indices],y[masks['fit']],
                                       x[masks['hold']][:,indices],y[masks['hold']],kind,101)
            scores=predict(x[:,indices]);predictions[(kind,group)]=scores
            held=scores[masks['hold']];held_y=y[masks['hold']]
            thresholds={t:threshold_for_recall(held_y,held,t) for t in (.8,.9)}
            for population in ('val_natural','val_matched'):
                m=masks[population];w=weight[m] if population=='val_natural' else np.ones(int(m.sum()))
                metrics=binary_metrics(y[m],scores[m],w)
                ci=bootstrap_by_image(y[m],scores[m],w,case[m],repetitions=reps)
                row={'classifier':kind,'feature_group':group,'population':population,**training,**metrics,
                     'ci95_image_bootstrap':json.dumps(ci),
                     'holdout_threshold_80':thresholds[.8],'holdout_threshold_90':thresholds[.9],
                     'val_at_holdout_80':json.dumps(evaluate_threshold(y[m],scores[m],thresholds[.8],w)),
                     'val_at_holdout_90':json.dumps(evaluate_threshold(y[m],scores[m],thresholds[.9],w))}
                results.append(row)
                for image in np.unique(case[m]):
                    take=m&(case==image);wi=weight[take] if population=='val_natural' else np.ones(int(take.sum()))
                    per_image.append({'classifier':kind,'feature_group':group,'population':population,'case':str(image),
                                      **binary_metrics(y[take],scores[take],wi)})
                for label,subset in [('pos_3_7',(y==1)&(data['length']>=3)&(data['length']<=7)),
                                     ('pos_8_15',(y==1)&(data['length']>=8)&(data['length']<=15)),
                                     ('pos_16_31',(y==1)&(data['length']>=16)&(data['length']<=31)),
                                     ('pos_32_plus',(y==1)&(data['length']>=32)),
                                     ('pos_centre_proxy',(y==1)&(data['flank_proxy']==0)),
                                     ('pos_flank_proxy',(y==1)&(data['flank_proxy']==1)),
                                     ('neg_two_components_proxy',(y==0)&(data['nuisance'][:,3]<=64)),
                                     ('neg_new_fp_proxy',(y==0)&np.isin(data['negative_kind'],[3,4])),
                                     ('neg_bridge_proxy',(y==0)&(data['negative_kind']==4))]:
                    take=m&subset
                    wsub=weight[take] if population=='val_natural' else np.ones(int(take.sum()))
                    rate=float(np.average((scores[take]>=thresholds[.9]).astype(float),weights=wsub)) if take.any() and thresholds[.9] is not None else None
                    subgroups.append({'classifier':kind,'feature_group':group,'population':population,'subgroup':label,
                                      'n':int(take.sum()),'accepted_at_train_holdout_90':rate})
                # Max score gives an 'at least one sampled location' detection
                # proxy per fixed gap when available, else per tile component.
                positive=np.flatnonzero(m&(y==1));grouped={}
                for i in positive:
                    gid=int(data['fixed_gap_id'][i])
                    key=(str(case[i]),'fixed',gid) if gid>0 else (str(case[i]),int(data['tile_index'][i]),int(data['component'][i]))
                    grouped.setdefault(key,[]).append(int(i))
                max_scores=np.array([max(scores[j] for j in positions) for positions in grouped.values()])
                negatives=np.flatnonzero(m&(y==0));negweights=weight[negatives] if population=='val_natural' else np.ones(len(negatives))
                gap90=None;neg_fpr=None
                if len(max_scores) and len(negatives):
                    gap90=threshold_for_recall(np.ones(len(max_scores),np.int8),max_scores,.9)
                    neg_fpr=float(np.average(scores[negatives]>=gap90,weights=negweights))
                gap_detection.append({'classifier':kind,'feature_group':group,'population':population,
                    'gap_units':len(grouped),'gap_units_with_fixed_id':sum(key[1]=='fixed' for key in grouped),
                    'at_least_one_at_train_holdout_90':float((max_scores>=thresholds[.9]).mean()) if len(max_scores) and thresholds[.9] is not None else None,
                    'val_threshold_at_90_gap_proxy_recall':gap90,'negative_fpr_at_90_gap_proxy_recall':neg_fpr,
                    'definition':'max sampled pixel score per nearby fixed gap ID else tile missing-skeleton component; not path closure'})
    # Fixed-selector acceptance is an association baseline, not a trained probe.
    scores=1-x[:,302]
    for population in ('val_natural','val_matched'):
        m=masks[population];w=weight[m] if population=='val_natural' else np.ones(int(m.sum()))
        results.append({'classifier':'existing_selector','feature_group':'1_minus_stay','population':population,
                        'best_step':None,'holdout_bce':None,**binary_metrics(y[m],scores[m],w),
                        'ci95_image_bootstrap':json.dumps(bootstrap_by_image(y[m],scores[m],w,case[m],repetitions=reps)),
                        'holdout_threshold_80':None,'holdout_threshold_90':None,'val_at_holdout_80':None,'val_at_holdout_90':None})
    # Outputs use the exact same candidate indices for all feature groups.
    for i in np.flatnonzero(split=='val'):
        row={'case':str(case[i]),'y':int(data['native_y'][i]),'x':int(data['native_x'][i]),
             'label':int(y[i]),'matched':bool(matched[i]),'sampling_weight':float(weight[i]),
             'missing_length_native':int(data['length'][i]),'flank_proxy':int(data['flank_proxy'][i]),
             'fixed_gap_id_nearby':int(data['fixed_gap_id'][i]),'negative_kind':int(data['negative_kind'][i]),
             'corridor_gt_fraction':float(data['corridor_gt_fraction'][i]) if np.isfinite(data['corridor_gt_fraction'][i]) else None,
             'corridor_length_native':float(data['corridor_length'][i]),
             'p0':float(data['nuisance'][i,0]),'distance':float(data['nuisance'][i,1]),
             'density':float(data['nuisance'][i,2]),'second_component_distance_censored':float(data['nuisance'][i,3]),
             'selector_leave':float(1-x[i,302])}
        for key,values in predictions.items():row['_'.join(key)]=float(values[i])
        score_rows.append(row)
    return results,per_image,subgroups,score_rows,predictions,masks,gap_detection


def save_rows(path,rows):
    write_csv(path,rows,list(rows[0]) if rows else ['empty'])


def save_matching_diagnostics(data, records, output):
    matched=np.zeros(len(data['label']),bool);matched[data['matched_indices']]=True
    rows=[]
    for split in ('fit','hold','val'):
        for population in ('natural_sample_weighted','matched'):
            take=(data['split']==split)&(matched if population=='matched' else True)
            weight=np.ones(int(take.sum())) if population=='matched' else data['weight'][take]
            for row in smd_report(data['label'][take],data['nuisance'][take],weight):
                rows.append({'split':split,'population':population,'samples':int(take.sum()),**row})
    save_rows(output/'matching_smd.csv',rows)
    atomic_json(output/'candidate_extraction_counts.json',{
        r['split']+':'+r['case']:json.loads(str(r['counts_json'])) for r in records})
    return rows


def make_visualizations(dataset, records, data, scores, masks, output, limit=8):
    folder=output/'cases_visual';folder.mkdir(exist_ok=True)
    val=np.flatnonzero(masks['val_natural']);neg=val[data['label'][val]==0]
    ranked=neg[np.argsort(-scores[neg])];by_case={r['case']:r for r in records if r['split']=='val'}
    index_by_name={Path(name).stem.replace('_sat',''):i for i,name in enumerate(dataset.image_files)}
    chosen=[];seen=set()
    for ix in ranked:
        name=str(data['case'][ix])
        if name in seen: continue
        chosen.append(int(ix));seen.add(name)
        if len(chosen)>=limit:break
    rows=[]
    for ix in chosen:
        name=str(data['case'][ix]);sample=dataset[index_by_name[name]]
        image=sample['image'].numpy().transpose(1,2,0)
        image=np.clip((image*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8)
        gt=sample['mask'][0].numpy()>.5;p0=by_case[name]['p0_map'].astype(np.float32)
        y=int(data['native_y'][ix]);x=int(data['native_x'][ix]);radius=64
        y0,y1=max(0,y-radius),min(1500,y+radius);x0,x1=max(0,x-radius),min(1500,x+radius)
        rgb=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_RGB2BGR)
        overlay=rgb.copy();overlay[gt[y0:y1,x0:x1]]=(0,255,0)
        overlay=cv2.addWeighted(rgb,.6,overlay,.4,0)
        heat=cv2.applyColorMap(np.clip(p0[y0:y1,x0:x1]*255,0,255).astype(np.uint8),cv2.COLORMAP_JET)
        for panel in (rgb,overlay,heat): cv2.circle(panel,(x-x0,y-y0),4,(0,0,255),2)
        title=f'{name} x={x} y={y} score={scores[ix]:.3f}'
        for panel,label in ((rgb,'RGB'),(overlay,'GT'),(heat,'P0')):
            cv2.putText(panel,label,(3,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        figure=np.concatenate((rgb,overlay,heat),axis=1)
        cv2.putText(figure,title,(4,figure.shape[0]-5),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1,cv2.LINE_AA)
        path=folder/(name+'_'+str(y)+'_'+str(x)+'.png');cv2.imwrite(str(path),figure)
        rows.append({'case':name,'x':x,'y':y,'score':float(scores[ix]),'path':str(path),
                     'matched':bool(masks['val_matched'][ix]),'negative_kind':int(data['negative_kind'][ix])})
    save_rows(output/'visualizations.csv',rows)


def write_report(output,data,records,match_rows,results,per_image,subgroups,score_rows,identity,predictions,masks,gap_detection):
    save_rows(output/'probe_results.csv',results);save_rows(output/'per_image_metrics.csv',per_image)
    save_rows(output/'subgroup_metrics.csv',subgroups);save_rows(output/'val_candidate_scores.csv',score_rows)
    save_rows(output/'gap_detection_proxy.csv',gap_detection)
    save_rows(output/'matching_by_image.csv',match_rows)
    match_index=np.zeros(len(data['label']),bool);match_index[data['matched_indices']]=True
    smd=save_matching_diagnostics(data,records,output)
    val_smd=[r['smd'] for r in smd if r['split']=='val' and r['population']=='matched' and r['smd'] is not None]
    max_val_smd=max((abs(x) for x in val_smd),default=None)
    coverage={}
    for split in ('fit','hold','val'):
        take=data['split']==split;matched=take&match_index
        positives=int((take&(data['label']==1)).sum());negatives=int((take&(data['label']==0)).sum())
        coverage[split]={'images':len(set(data['case'][take])),'sampled_positive':positives,'sampled_negative':negatives,
                         'matched_pairs':int((matched&(data['label']==1)).sum()),
                         'unmatched_positive_fraction':float(1-(matched&(data['label']==1)).sum()/positives) if positives else None,
                         'unmatched_negative_fraction':float(1-(matched&(data['label']==0)).sum()/negatives) if negatives else None}
    val_fixed={(str(r['case']),int(r['fixed_gap_id_nearby'])) for r in score_rows if r['label']==1 and r['fixed_gap_id_nearby']>0}
    corridor=[]
    for label in (1,0):
        for population in ('val_natural','val_matched'):
            take=(data['split']=='val')&(data['label']==label)&np.isfinite(data['corridor_gt_fraction'])
            if population=='val_matched':take &= match_index
            values=data['corridor_gt_fraction'][take]
            corridor.append({'label':label,'population':population,'n':len(values),
                             'mean_gt_fraction':float(values.mean()) if len(values) else None,
                             'fraction_gt_coverage_ge_0_8':float((values>=.8).mean()) if len(values) else None,
                             'fraction_gt_coverage_le_0_2':float((values<=.2).mean()) if len(values) else None})
    save_rows(output/'corridor_proxy_summary.csv',corridor)
    delta=[]
    for r in results:
        if r['classifier'] not in ('linear','mlp'):continue
        base=next(v for v in results if v['classifier']==r['classifier'] and v['population']==r['population'] and v['feature_group']=='base')
        row={'classifier':r['classifier'],'feature_group':r['feature_group'],'population':r['population']}
        take=masks[r['population']];w=data['weight'][take] if r['population']=='val_natural' else np.ones(int(take.sum()))
        ci=paired_delta_ci(data['label'][take],predictions[(r['classifier'],r['feature_group'])][take],
                           predictions[(r['classifier'],'base')][take],w,data['case'][take],identity['bootstrap'])
        for key in ('roc_auc','pr_auc','fpr_at_80_recall','fpr_at_90_recall'):
            row[key+'_delta_vs_base']=(r[key]-base[key]) if r[key] is not None and base[key] is not None else None
            row[key+'_paired_image_bootstrap_ci95']=json.dumps(ci[key])
        delta.append(row)
    save_rows(output/'increment_vs_base.csv',delta)
    lines=['# Frozen B low-P0 feature probe','',
           f'Training images sampled: {identity["train_images_used"]}; validation images: {identity["val_images"]}.',
           'Validation uses 25 native 512/256 windows with disjoint ownership cores; train uses one seeded crop per image.',
           'The tile-owned probabilities are not the original weighted-logit stitched maps.',
           'Probes fit matched training images, choose iteration and thresholds on image-disjoint train holdout, then evaluate val.',
           f'Matched validation max absolute SMD: {max_val_smd}; values above 0.1 indicate residual imbalance.',
           f'Fixed reference gap IDs near at least one sampled positive: {len(val_fixed)} / {identity["fixed_gaps"]}; proximity is not verified path identity.',
           f'Sample/matching coverage: `{json.dumps(coverage)}`','','| Classifier | Features | Population | N | Positive fraction | ROC-AUC | PR-AUC | FPR at 90% val recall |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(value):return 'n/a' if value is None else f'{value:.4f}'
    for r in results:
        lines.append(f'| {r["classifier"]} | {r["feature_group"]} | {r["population"]} | {r["n"]} | {fmt(r["weighted_positive_fraction"])} | {fmt(r["roc_auc"])} | {fmt(r["pr_auc"])} | {fmt(r["fpr_at_90_recall"])} |')
    lines += ['', 'FPR at fixed validation recall is descriptive, not a chosen deployment threshold. '
                  'Also see train-holdout thresholds and achieved val TPR/FPR in probe_results.csv.',
              'Natural-population metrics use inverse inclusion weights for the prediction-defined candidate grid after GT uncertainty exclusions.',
              'A two-predicted-component negative is only a spatial proxy; it is not a verified parallel-road connection.',
              'Predicted components smaller than 8 native pixels are excluded from both distance features; distances beyond 64 pixels are censored at 65.',
              'negative_kind: 1 two components, 2 other near-road background, 3 new FP proxy, 4 new bridge proxy (priority order).',
              'corridor_proxy_summary.csv evaluates GT coverage of a prediction-only nearest-component/center polyline, including endpoints; this is not GT path connectivity.',
              'The existing selector score is 1 minus step-one stay weight; it is not a correctness probability.',
              'Uncertain GT/road-edge/junction cells are excluded; P0 and distance matching are based on predictions only.',
              'Tile candidates, gap lengths and flank labels are proxies; closure and path-level connection accuracy are not measured.',
              'gap_detection_proxy.csv reports at-least-one sampled location per gap unit, not complete-gap coverage or closure.',
              'Per-image and image-bootstrap statistics avoid treating adjacent pixels as independent images.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(output/'summary.json',{'complete':True,'identity':identity,'sample_coverage':coverage,
                                     'fixed_gap_ids_near_sampled_positives':len(val_fixed),
                                     'selection_uses_validation':False,'results':results,
                                     'limits':lines[-7:]})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair_dir',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--b_run_name',default='B_transport_0p01_proposal_0p001')
    p.add_argument('--root_path',type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--train_images',type=int,default=128,help='0 uses every train image; each contributes one seeded 512 crop')
    p.add_argument('--seed',type=int,default=1234);p.add_argument('--bootstrap',type=int,default=100)
    p.add_argument('--expected_images',type=int,default=14);p.add_argument('--expected_gaps',type=int,default=198)
    args=p.parse_args()
    if args.train_images<0 or args.bootstrap<0: p.error('Negative counts are invalid')
    if Path(args.b_run_name).name!=args.b_run_name or not args.b_run_name.startswith('B'): p.error('Invalid B run')
    pair=args.pair_dir.resolve();run=pair/args.b_run_name;checkpoint_path=run/'best.pth'
    reference_path=pair/'reference/fixed_gaps_and_anchors.json';manifest=read_json(reference_path)
    if manifest['count']!=args.expected_gaps or len(manifest['images'])!=args.expected_images:
        raise ValueError('Fixed validation reference does not match expected counts')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False);config=checkpoint['config']
    if config['reference_sha256']!=digest(reference_path):raise ValueError('Checkpoint/reference fingerprint mismatch')
    verify_recorded_code(config);root=(args.root_path or Path(config['root_path'])).resolve()
    train=full_dataset(root,'train');val=full_dataset(root,'val')
    def case_names(dataset):return [Path(name).stem.replace('_sat','') for name in dataset.image_files]
    train_names=case_names(train);val_names=case_names(val)
    if len(set(train_names))!=len(train_names) or len(set(val_names))!=len(val_names) or set(train_names)&set(val_names):
        raise ValueError('Duplicate or overlapping train/validation image IDs')
    if len(val)!=args.expected_images or set(val_names)!=set(manifest['images']):raise ValueError('Validation image set mismatch')
    rng=np.random.default_rng(args.seed);count=min(len(train),args.train_images or len(train))
    if count<5:raise ValueError('At least five training images are required')
    chosen=np.sort(rng.choice(len(train),size=count,replace=False));permutation=rng.permutation(chosen)
    hold=set(int(i) for i in permutation[:max(1,int(round(count*.2)))])
    splits={train_names[int(i)]:('hold' if int(i) in hold else 'fit') for i in chosen}
    identity={'version':'frozen_h0_low_p0_probe_v1','checkpoint_sha256':digest(checkpoint_path),
              'reference_sha256':digest(reference_path),'root_path':str(root),'script_sha256':digest(Path(__file__)),
              'seed':args.seed,'train_images_used':count,'val_images':len(val),'fixed_gaps':manifest['count'],
              'train_image_splits':splits,'val_image_names':val_names,'device':torch.device(args.device).type,
              'torch_version':str(torch.__version__),'bootstrap':args.bootstrap,'p0_threshold':P0_THRESHOLD,
              'match_calipers':MATCH_TOLERANCE.tolist(),'train_crops':'one seeded 512 crop per training image',
              'validation_tiles':'512/256, nonoverlapping ownership cores, features from one tile forward',
              'feature_groups':{key:value.tolist() for key,value in GROUPS.items()}}
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True);identity_path=output/'identity.json'
    if identity_path.exists() and read_json(identity_path)!=identity:raise ValueError('Output already belongs to a different audit configuration')
    if not identity_path.exists() and any(output.iterdir()):raise ValueError('Choose an empty output directory')
    atomic_json(identity_path,identity);cache=output/'cases';cache.mkdir(exist_ok=True)
    seed_all(args.seed);model=rebuild_model(checkpoint,torch.device(args.device));model.requires_grad_(False);del checkpoint
    records=[];seen_image_hashes={}
    work=[(train,int(i),splits[train_names[int(i)]]) for i in chosen]+[(val,i,'val') for i in range(len(val))]
    for ordinal,(dataset,index,split) in enumerate(work,1):
        sample=dataset[index];name=str(sample['case_name']);path=cache/(split+'_'+name+'.npz')
        if split=='val':
            gt_path=pair/'reference'/(name+'.npz')
            if not gt_path.is_file():gt_path=run/'eval/val'/(name+'.npz')
            with np.load(gt_path) as f:
                if not np.array_equal(f['gt'].astype(bool),sample['mask'][0].numpy()>.5):raise ValueError('Fixed reference GT mismatch: '+name)
        if path.is_file():
            with np.load(path) as f:item={k:f[k].copy() for k in f.files}
            if str(item['input_sha256'])!=signature(sample):raise ValueError('Cached image/GT changed: '+name)
            print(f'[CACHE] {ordinal}/{len(work)} {split} {name}',flush=True)
        else:
            print(f'[EXTRACT] {ordinal}/{len(work)} {split} {name}',flush=True)
            item=extract_image(model,sample,split,args.seed,manifest['images'].get(name) if split=='val' else None)
            atomic_npz(path,item)
        img_hash=image_digest(sample)
        if img_hash in seen_image_hashes and seen_image_hashes[img_hash][1]!=split:
            raise ValueError(f'Identical image content occurs across splits: {seen_image_hashes[img_hash][0]} and {name}')
        seen_image_hashes[img_hash]=(name,split)
        item['case']=name;item['split']=split;records.append(item)
    data,match_rows=combine(records)
    save_rows(output/'matching_by_image.csv',match_rows)
    save_matching_diagnostics(data,records,output)
    try:
        results,per_image,subgroups,score_rows,predictions,masks,gap_detection=evaluate(data,output,args.bootstrap)
    except ValueError as error:
        atomic_json(output/'summary.json',{'complete':False,'reason':str(error),'identity':identity,'matching_by_image':match_rows})
        (output/'REPORT.md').write_text('# Probe unavailable\n\n'+str(error)+'\nSee matching_by_image.csv.\n',encoding='utf-8')
        print('[INCOMPLETE] '+str(error),flush=True);return
    write_report(output,data,records,match_rows,results,per_image,subgroups,score_rows,identity,predictions,masks,gap_detection)
    make_visualizations(val,records,data,predictions[('mlp','all')],masks,output)
    print('[DONE] '+str(output/'REPORT.md'),flush=True)


if __name__=='__main__':main()
