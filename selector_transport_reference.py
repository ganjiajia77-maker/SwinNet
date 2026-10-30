"""Freeze original P0 gap pixels AND anchors once; all models read the same list."""
import hashlib

import cv2
import numpy as np

from evaluation.road_comparison.road_metrics import prepare_target,DEFAULT_PROTOCOL

PROTOCOL=dict(DEFAULT_PROTOCOL,image_size=1500)


def path_length_bin(length): return '3-7' if length<8 else '8-15' if length<16 else '16-31' if length<32 else '32+'


def prepared(case):
    if '_prepared' not in case: case['_prepared']=prepare_target(case['gt'],PROTOCOL)
    return case['_prepared']


def reference_digest(cases):
    digest=hashlib.sha256()
    for case in cases:
        gt=case['gt']; p0=case['p0']>=.45; skel=prepared(case)['skeleton']
        digest.update(case['case'].encode()+b'\0'); digest.update(np.asarray(gt.shape,dtype=np.int64).tobytes())
        for mask in (p0,gt,skel): digest.update(np.packbits(mask,axis=None).tobytes())
    return digest.hexdigest()


def freeze_gaps(cases,old_reference):
    digest=reference_digest(cases)
    if digest!=old_reference['sha256'] or len(cases)!=old_reference['images']:
        raise ValueError('P0/GT/skeleton fingerprint differs from the original diagnostic reference')
    images={}; total=0
    for case in cases:
        p0=case['p0']>=.45; gt=case['gt']; h,w=gt.shape
        skeleton=prepared(case)['skeleton']
        covered=cv2.dilate(p0.astype(np.uint8),np.ones((5,5),np.uint8))>0
        n,labels,stats,_=cv2.connectedComponentsWithStats((skeleton&~covered).astype(np.uint8),connectivity=8)
        _,old_labels=cv2.connectedComponents(p0.astype(np.uint8),connectivity=8)
        gaps=[]
        for index in range(1,n):
            length=int(stats[index,cv2.CC_STAT_AREA])
            if length<3: continue
            x,y,width,height=map(int,stats[index,:4])
            x0,x1=max(x-7,0),min(x+width+7,w); y0,y1=max(y-7,0),min(y+height+7,h)
            region=labels[y0:y1,x0:x1]==index
            flank=cv2.dilate(region.astype(np.uint8),np.ones((15,15),np.uint8))>0
            old=old_labels[y0:y1,x0:x1]; ground=gt[y0:y1,x0:x1]
            components=np.unique(old[flank&ground]); components=components[components>0]
            if len(components)<2: continue
            def indices(mask):
                yy,xx=np.nonzero(mask); return ((yy+y0)*w+xx+x0).tolist()
            anchors=[indices(flank&(old==component)&ground) for component in components]
            gaps.append({'id':index,'length':length,'length_bin':path_length_bin(length),
                         'pixels':indices(region),'anchors':anchors})
        total+=len(gaps); images[case['case']]={'shape':[h,w],'gaps':gaps}
    if total!=old_reference['eligible_reference_gaps']:
        raise ValueError('Frozen gap/anchor list count differs from the old reference')
    return {'version':'original_v9_frozen_gaps_and_anchors_v1','sha256':digest,'images':images,
            'count':total,'p0_reference_threshold':.45,
            'origin':'Materialized once from the original fixed P0/GT fingerprint and unchanged V9 gap/anchor definition. Never regenerated per model.'}


def closed_gaps(pred,entry):
    if list(pred.shape)!=entry['shape']: raise ValueError('Prediction shape differs from frozen gap coordinates')
    _,components=cv2.connectedComponents(pred.astype(np.uint8),connectivity=8)
    prediction=pred.reshape(-1); components=components.reshape(-1)
    bins={name:{'eligible':0,'closed':0} for name in ('3-7','8-15','16-31','32+')}
    ids=[]
    for gap in entry['gaps']:
        row=bins[gap['length_bin']]; row['eligible']+=1
        if prediction[gap['pixels']].mean()<.8: continue
        connected=[set(np.unique(components[anchor]))-{0} for anchor in gap['anchors']]
        if connected and set.intersection(*connected): row['closed']+=1; ids.append(gap['id'])
    return bins,ids


def bridge_proxy(p0,pred,gt):
    _,old=cv2.connectedComponents(p0.astype(np.uint8),connectivity=8)
    background=pred&~p0&~(cv2.dilate(gt.astype(np.uint8),np.ones((7,7),np.uint8))>0)
    n,labels,stats,_=cv2.connectedComponentsWithStats(background.astype(np.uint8),connectivity=8)
    result=0
    for index in range(1,n):
        if stats[index,cv2.CC_STAT_AREA]<3: continue
        x,y,width,height=map(int,stats[index,:4])
        x0,x1=max(x-2,0),min(x+width+2,gt.shape[1]); y0,y1=max(y-2,0),min(y+height+2,gt.shape[0])
        near=cv2.dilate((labels[y0:y1,x0:x1]==index).astype(np.uint8),np.ones((5,5),np.uint8))>0
        result+=int((np.unique(old[y0:y1,x0:x1][near])>0).sum()>=2)
    return result


def exact_recall_threshold(cases,name,target_tp):
    positive=np.concatenate([case[name][case['gt']] for case in cases])
    if not 0<=target_tp<=len(positive): raise ValueError('Invalid Recall target')
    if target_tp==0: return 1.000001,0
    threshold=np.partition(positive,len(positive)-target_tp)[len(positive)-target_tp]
    candidates=(float(threshold),float(np.nextafter(np.float32(threshold),np.float32(np.inf))))
    values=[(value,int((positive>=value).sum())) for value in candidates]
    value,count=min(values,key=lambda row:(abs(row[1]-target_tp),-row[0]))
    return value,count-target_tp
