"""Uncertain-aware relation matching and deduplicated path supervision."""
from collections import defaultdict
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from networks.connection_paths import label_paths, soft_corridor
from networks.reliable_directional_repair import dynamic_losses


def matching_pairs(paths,labels,seed=1234):
    """Match length, R source quality, endpoint P0 and separation; GT only labels."""
    groups=defaultdict(lambda:[[],[]]); rng=np.random.default_rng(seed)
    for i,(path,label) in enumerate(zip(paths,labels)):
        if label['label']<0: continue
        key=(path['kind'],int(np.digitize(path['path_length_native'],[8,16,32,64,128])),
             int(np.digitize(np.mean(path['source_confidence']),[.1,.3,.6])),
             int(np.digitize(np.mean(path['endpoint_p0']),[.3,.5,.75])),
             int(np.digitize(np.linalg.norm(path['endpoint_a']-path['endpoint_b']),[4,8,16,32])))
        groups[key][label['label']].append(i)
    selected=[]
    for negatives,positives in groups.values():
        n=min(len(negatives),len(positives))
        if n:
            selected.extend(rng.choice(negatives,n,replace=False).tolist())
            selected.extend(rng.choice(positives,n,replace=False).tolist())
    return selected


def attach_labels(result,mask,skeleton,reference_threshold,config):
    labels=[]; gaps=[]
    for b,paths in enumerate(result['paths']):
        label,coverage=label_paths(paths,mask[b,0].detach().cpu().numpy()>.5,
            skeleton[b,0].detach().cpu().numpy()>.5,result['p0'][b,0].detach().cpu().numpy(),reference_threshold,config)
        labels.append(label); gaps.append(coverage)
    return labels,gaps


def relation_loss(result,labels,seed=1234):
    terms=[]; total=matched=positive=negative=uncertain=0
    for b,(paths,rows,logits) in enumerate(zip(result['paths'],labels,result['path_logits'])):
        indices=matching_pairs(paths,rows,seed+b)
        positive+=sum(row['label']==1 for row in rows); negative+=sum(row['label']==0 for row in rows)
        uncertain+=sum(row['label']<0 for row in rows); total+=len(rows)
        if indices:
            target=logits.new_tensor([rows[i]['label'] for i in indices])
            terms.append(F.binary_cross_entropy_with_logits(logits[indices],target)); matched+=len(indices)
    zero=sum((logits.sum()*0. for logits in result['path_logits']),result['z1'].sum()*0.)
    return (torch.stack(terms).mean() if terms else zero),{'candidates':total,'positive':positive,'negative':negative,
        'uncertain':uncertain,'matched_candidates':matched}


def path_losses(result,mask,skeleton,labels,source,phase,seed=1234,
                lambda_relation=1.,lambda_path=.2,lambda_neg=.1):
    classification,counts=relation_loss(result,labels,seed)
    baseline,terms=dynamic_losses(result,mask,skeleton,lambda_bg=source['lambda_bg'],
        lambda_missing=source['lambda_missing'],lambda_cont=source['lambda_cont'],
        missing_threshold=source['missing_threshold'],return_live_terms=True)
    if phase=='B':
        return lambda_relation*classification,{**counts,'loss':float(classification.detach()),
            'relation':float(classification.detach()),'seg':float(terms['seg']),'cont':float(terms['cont']),
            'path':0.,'hardneg':0.,'path_supervised_pixels':0,'removed_duplicate_cont_pixels':0}
    zero=result['z1'].sum()*0.; recover=zero; hardneg=zero; excluded=torch.zeros_like(mask)
    bg=torch.zeros_like(mask)
    if phase in ('D','E','F'):
        for b,(paths,rows) in enumerate(zip(result['paths'],labels)):
            for path,row in zip(paths,rows):
                if row['label']<0: continue
                corridor=soft_corridor(path['path_points'],result['u0'].shape[-2:],path['width_work'])
                full=F.interpolate(torch.as_tensor(corridor,device=mask.device,dtype=mask.dtype)[None,None],size=mask.shape[-2:],mode='bilinear',align_corners=False)[0,0]>.05
                if row['label']==1: excluded[b,0]=torch.maximum(excluded[b,0],full.float())
                else: bg[b,0]=torch.maximum(bg[b,0],(full & (mask[b,0]<.5)).float())
        missing=(skeleton>.5)&(F.max_pool2d(result['p0'].detach(),5,1,2)<source['missing_threshold'])
        excluded=excluded*missing.float()
        # Only remove pixels that will actually receive the path objective below.
        for b in range(mask.shape[0]):
            n,components,stats,_=cv2.connectedComponentsWithStats(excluded[b,0].detach().cpu().numpy().astype(np.uint8),connectivity=8)
            keep=np.zeros(components.shape,bool)
            for i in range(1,n):
                if stats[i,cv2.CC_STAT_AREA]>=3: keep|=components==i
            excluded[b,0]=torch.as_tensor(keep,device=mask.device,dtype=mask.dtype)
        # Preserve the original segmentation weighting, partition only L_cont's pixels.
        _,remaining=dynamic_losses(result,mask,skeleton*(1.-excluded),lambda_bg=source['lambda_bg'],
            lambda_missing=source['lambda_missing'],lambda_cont=source['lambda_cont'],
            missing_threshold=source['missing_threshold'],return_live_terms=True)
        baseline=baseline-source['lambda_cont']*terms['cont_live']+source['lambda_cont']*remaining['cont_live']
        terms['cont']=remaining['cont']; tolerant=F.max_pool2d(result['p1'],3,1,1)
        pieces=[]
        for b in range(mask.shape[0]):
            n,components=cv2.connectedComponents(excluded[b,0].detach().cpu().numpy().astype(np.uint8),connectivity=8)
            sample=[]
            for i in range(1,n):
                yy,xx=np.nonzero(components==i)
                if len(yy)<3: continue
                values=tolerant[b,0,torch.as_tensor(yy,device=mask.device),torch.as_tensor(xx,device=mask.device)]
                softmin=-(torch.logsumexp(-15.*values,0)-math.log(len(yy)))/15.
                sample.append(.05*F.softplus((.75-softmin)/.05))
            if sample: pieces.append(torch.stack(sample).mean())
        if pieces: recover=torch.stack(pieces).mean()
        if bool(bg.any()): hardneg=(F.softplus(result['z1'])*bg).sum()/bg.sum()
    total=baseline+lambda_relation*classification+lambda_path*recover+lambda_neg*hardneg
    return total,{**counts,'loss':float(total.detach()),'relation':float(classification.detach()),
        'seg':float(terms['seg']),'cont':float(terms['cont']),'path':float(recover.detach()),'hardneg':float(hardneg.detach()),
        'path_supervised_pixels':int(excluded.sum()),'removed_duplicate_cont_pixels':int(excluded.sum()),
        'hardneg_pixels':int(bg.sum())}
