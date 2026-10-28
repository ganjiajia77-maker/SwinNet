"""Shared cache, classification and val-only operating-point protocol."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnostics.v91_counterfactual import evaluate, write_json
from diagnostics.diagnose_v9_transport import matched_threshold
from diagnostics.v92_connection_candidates import split_name
from mass_protocol import blend_weight,sliding_positions
from networks.path_connection_losses import attach_labels
from networks.path_connection_repair import sample_points


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8'); temp.replace(path)


def atomic_npz(path,**values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as handle: np.savez_compressed(handle,**values)
    temp.replace(path)


def binary_metrics(labels,scores):
    labels=np.asarray(labels,dtype=np.int64); scores=np.asarray(scores,dtype=float)
    keep=(labels>=0)&np.isfinite(scores); labels=labels[keep]; scores=scores[keep]
    pos=int(labels.sum()); neg=len(labels)-pos
    out={'n':len(labels),'positive':pos,'negative':neg,'positive_fraction':pos/max(len(labels),1),
         'roc_auc':None,'pr_auc':None,'fpr_at_recall_80':None,'fpr_at_recall_90':None}
    if min(pos,neg)==0: return out
    order=np.argsort(-scores,kind='stable'); y=labels[order]; score=scores[order]
    # Group ties: a constant classifier has ROC-AUC .5 and AP equal to prevalence.
    ends=np.r_[np.nonzero(np.diff(score))[0],len(score)-1]
    tp=np.cumsum(y)[ends]; fp=(ends+1)-tp
    recall=tp/pos; fpr=fp/neg; precision=tp/(tp+fp)
    out['pr_auc']=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    out['roc_auc']=float(np.trapz(np.r_[0.,recall],np.r_[0.,fpr]))
    for target in (.8,.9):
        index=min(int(np.searchsorted(recall,target)),len(recall)-1)
        out[f'fpr_at_recall_{int(target*100)}']=float(fpr[index])
    return out


def strata_for(path):
    return '|'.join(map(str,(path['kind'],int(np.digitize(path['path_length_native'],[8,16,32,64,128])),
        int(np.digitize(np.mean(path['source_confidence']),[.1,.3,.6])),
        int(np.digitize(np.mean(path['endpoint_p0']),[.3,.5,.75])),
        int(np.digitize(np.linalg.norm(path['endpoint_a']-path['endpoint_b']),[4,8,16,32])))))


def matched_indices(labels,strata,seed):
    groups={}; rng=np.random.default_rng(seed)
    for i,(label,key) in enumerate(zip(labels,strata)):
        if label<0: continue
        groups.setdefault(str(key),[[],[]])[int(label)].append(i)
    selected=[]
    for neg,pos in groups.values():
        n=min(len(neg),len(pos))
        if n: selected.extend(rng.choice(neg,n,replace=False)); selected.extend(rng.choice(pos,n,replace=False))
    return np.asarray(selected,dtype=np.int64)


def cache_tile(model,result,mask,skeleton,case,split,key,output,reference,origin=(0,0)):
    path=output/(key+'.npz')
    labels,gaps=attach_labels(result,mask,skeleton,reference,model.config)
    items=result['paths'][0]; rows=labels[0]
    metadata={'case':case,'split':split,'partition':split_name(case,getattr(model,'partition_seed',1234)),
              'tile_origin_native':list(map(int,origin)),'work_shape':list(result['u0'].shape[-2:]),
              'native_crop_shape':list(mask.shape[-2:]),'gaps':gaps[0],
              'generation':result['path_generation'][0] if result['path_generation'] else {}}
    offsets=np.cumsum([0]+[len(item['path_points']) for item in items])
    old_g=result['g_step'].mean(1)
    gate_scores=[float(sample_points(old_g,item['path_points']).mean().detach()) for item in items]
    atomic_npz(path,features=result['path_features'][0].detach().cpu().numpy().astype(np.float32),
        baseline_gate_scores=np.asarray(gate_scores,np.float32),
        labels=np.asarray([row['label'] for row in rows],np.int8),
        strata=np.asarray([strata_for(item) for item in items],dtype='U160'),
        subtypes=np.asarray([row['subtype'] for row in rows],dtype='U64'),
        lengths_native=np.asarray([item['path_length_native'] for item in items],np.float32),
        lengths_work=np.asarray([item['path_length_work'] for item in items],np.float32),
        deviation_native=np.asarray([row['gt_path_deviation_native'] for row in rows],np.float32),
        source_confidence=np.asarray([item['source_confidence'] for item in items],np.float32).reshape(-1,2),
        endpoint_p0=np.asarray([item['endpoint_p0'] for item in items],np.float32).reshape(-1,2),
        distance_work=np.asarray([np.linalg.norm(item['endpoint_a']-item['endpoint_b']) for item in items],np.float32),
        endpoint_a_work=np.asarray([item['endpoint_a'] for item in items],np.int32).reshape(-1,2),
        endpoint_b_work=np.asarray([item['endpoint_b'] for item in items],np.int32).reshape(-1,2),
        path_offsets=offsets,path_points_work=np.concatenate([item['path_points'] for item in items]) if items else np.empty((0,2),np.int32),
        width_work=np.asarray([item['width_work'] for item in items],np.float32),
        width_native=np.asarray([item['width_native'] for item in items],np.float32),
        meta=np.asarray(json.dumps(metadata)))
    return path


@torch.no_grad()
def infer_cases(model,dataset,device,output=None,cache_features=None,reference=.45,variant='model'):
    """Full native-image weighted-logit sliding inference; atomic per-image resume."""
    cases=[]; model.eval(); timing=[]
    for index in range(len(dataset)):
        sample=dataset[index]; case=str(sample['case_name']); cachekey=hashlib.sha256(case.encode()).hexdigest()[:16]
        target=Path(output)/(cachekey+'.npz') if output else None
        height,width=sample['mask'].shape[-2:]
        features_complete=cache_features is None or all((Path(cache_features)/(cachekey+f'_{y}_{x}.npz')).is_file()
            for y in sliding_positions(height,512,256) for x in sliding_positions(width,512,256))
        if target and target.is_file() and features_complete:
            with np.load(target,allow_pickle=False) as data:
                cases.append({'case':case,'gt':data['gt'].astype(bool),'p0':data['p0'],variant:data['p1']})
                timing.append(float(data['seconds']))
            continue
        image=sample['image'][None].to(device); gt=sample['mask'][0].numpy()>.5; h,w=gt.shape
        canvas=torch.zeros((2,1,h,w),device=device); divisor=torch.zeros((1,1,h,w),device=device)
        weight=blend_weight(512,device); start=time.perf_counter()
        for y in sliding_positions(h,512,256):
            for x in sliding_positions(w,512,256):
                result=model(image[:,:,y:y+512,x:x+512],capture_diagnostics=True)
                region=(slice(None),slice(None),slice(y,y+512),slice(x,x+512))
                canvas[region]+=torch.cat((result['z0'],result['z1']),0).float()*weight
                divisor[region]+=weight
                if cache_features is not None:
                    cache=Path(cache_features)/(cachekey+f'_{y}_{x}.npz')
                    if not cache.is_file():
                        cache_tile(model,result,sample['mask'][None,:,y:y+512,x:x+512].to(device),
                            sample['skeleton'][None,:,y:y+512,x:x+512].to(device),case,'val',cache.stem,Path(cache_features),reference,(y,x))
        if not bool((divisor>0).all()): raise RuntimeError('Uncovered sliding pixels')
        if torch.cuda.is_available() and device.type=='cuda': torch.cuda.synchronize(device)
        seconds=time.perf_counter()-start; timing.append(seconds)
        # Match V9.1/V9.2's exact conversion for fixed P0-mask fingerprints.
        logits=(canvas/divisor).cpu().numpy()[:,0]
        probability=1./(1.+np.exp(-np.clip(logits,-80.,80.)))
        if not np.isfinite(probability).all(): raise RuntimeError('Nonfinite final probabilities')
        if target: atomic_npz(target,p0=probability[0],p1=probability[1],gt=gt,seconds=np.asarray(seconds))
        cases.append({'case':case,'gt':gt,'p0':probability[0],variant:probability[1]})
        print(f'[VAL INFERENCE] {variant} {index+1}/{len(dataset)} {case} {seconds:.1f}s',flush=True)
    return cases,{'images':len(cases),'seconds_total':float(sum(timing)),'seconds_per_image':float(np.mean(timing)),
                  'peak_cuda_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None}


def count_curve(cases,field,thresholds):
    bins=10000; ph=np.zeros(bins+1,np.int64); nh=ph.copy()
    for case in cases:
        values=np.minimum((case[field]*bins).astype(int),bins)
        ph+=np.bincount(values[case['gt']],minlength=bins+1)
        nh+=np.bincount(values[~case['gt']],minlength=bins+1)
    tp=ph[::-1].cumsum()[::-1]; fp=nh[::-1].cumsum()[::-1]; total=ph.sum()
    reports={}
    for threshold in thresholds:
        index=int(round(threshold*bins)); a=int(tp[index]); b=int(fp[index]); c=int(total-a)
        reports[float(threshold)]={'tp':a,'fp':b,'fn':c,'iou':a/max(a+b+c,1),'f1':2*a/max(2*a+b+c,1),
             'precision':a/max(a+b,1),'recall':a/max(a+c,1)}
    return reports


def own_threshold(cases,field):
    curve=count_curve(cases,field,np.round(np.arange(.05,.951,.01),2))
    threshold=max(curve,key=lambda value:(curve[value]['iou'],-abs(value-.5)))
    return threshold,curve


def compare_cases(cases,names,reference,expected_gaps,output,reference_file=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    chosen={}; curves={}
    for name in names:
        chosen[name],curves[name]=own_threshold(cases,name)
    threshold=chosen['A_original']; original=curves['A_original'][threshold]
    points={}
    for name in names:
        points[name]={'own_val_threshold':chosen[name],'same_threshold':threshold,
          'fixed_reference_threshold':reference,
          'matched_recall':matched_threshold(cases,name,original['tp'],'tp',10000)[0],
          'matched_foreground_area':matched_threshold(cases,name,original['tp']+original['fp'],'area',10000)[0]}
    reports=evaluate(cases,points,reference,expected_gaps,output,modes=names)
    if reference_file:
        expected=json.loads(Path(reference_file).read_text()); actual=json.loads((output/'reference_set.json').read_text())
        if actual!=expected: raise ValueError('P0 reference fingerprint differs from the fixed V9.1/V9.2 reference')
    atomic_json(output/'comparison.json',{'selected_on':'val','selection_rule':'max global pixel IoU on val, thresholds .05:.01:.95; all variants use the same rule',
        'proxy_warning':'APLS is approximate; gap closure, bridge and candidate labels are proxies, not official graph topology metrics.',
        'points':points,'curves':curves,'reports':reports,'fixed_p0_threshold':reference})
    lines=['# Path connection comparison (validation only)','',
        '| model | work point | threshold | IoU | clDice | GT→Pred APLS | Pred→GT APLS | bidirectional APLS | closed | bridge | short components |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        for point,row in reports[name].items():
            m=row['metrics']; p=row['paired']
            lines.append(f'| {name} | {point} | {row["threshold"]:.4f} | {m["iou"]:.4f} | {m["cldice"]:.4f} | '
                f'{m["apls_gt_to_pred_approx"]:.4f} | {m["apls_pred_to_gt_approx"]:.4f} | {m["apls_bidirectional_approx"]:.4f} | '
                f'{p["closed"]}/{p["eligible"]} | {p["background_bridge_proxy"]} | {m["short_pred_components"]:.2f} |')
    lines+=['','Reference gaps stay fixed. See per-image CSV and JSON for length bins, matching residuals and added TP/FP.',
        'All thresholds and model choices use val only. Global-IoU selection is re-applied to the original V9 for comparable work points; it differs from the older weighted topology selection score.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return reports


def visuals(cases,name,threshold,output,limit=4):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for case in cases[:limit]:
        p0=case['p0']; p1=case[name]; gt=case['gt']; binary=p1>=threshold
        base=cv2.applyColorMap(np.uint8(np.clip(p0,0,1)*255),cv2.COLORMAP_VIRIDIS)
        final=cv2.applyColorMap(np.uint8(np.clip(p1,0,1)*255),cv2.COLORMAP_VIRIDIS)
        delta=cv2.applyColorMap(np.uint8(np.clip((p1-p0)*4+.5,0,1)*255),cv2.COLORMAP_TURBO)
        added=np.zeros((*gt.shape,3),np.uint8); added[gt]=[80,80,80]
        added[binary&~(p0>=threshold)&gt]=[0,255,0]; added[binary&~(p0>=threshold)&~gt]=[0,0,255]
        cv2.imwrite(str(output/(case['case']+'.png')),np.concatenate((base,final,delta,added),1))
