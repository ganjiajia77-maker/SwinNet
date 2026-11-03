"""Cached training-reference components and deterministic per-image crops."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from mass_protocol import crop_position
from run_q3_r3_mass import full_dataset


def seed_for(seed,epoch,index,domain):
    return int.from_bytes(hashlib.sha256(f'{seed}|{epoch}|{index}|{domain}'.encode()).digest()[:4],'little')


def packed_components(mask,min_size=1):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    sizes=stats[:,cv2.CC_STAT_AREA];valid=np.flatnonzero(sizes>=min_size);valid=valid[valid>0]
    flat=labels.reshape(-1);points=np.flatnonzero(np.isin(flat,valid)).astype(np.int32)
    order=np.argsort(flat[points],kind='stable');points=points[order];ids=flat[points]
    starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(points)].astype(np.int32) if len(points) else np.array([0],np.int32)
    component_ids=ids[starts[:-1]] if len(points) else np.empty(0,np.int32)
    return points,starts,component_ids,labels


def reference_regions(p0,b,p1,gt,skeleton,threshold=.45):
    pred=p0>=threshold
    covered=cv2.dilate(pred.astype(np.uint8),np.ones((5,5),np.uint8))>0
    missed=skeleton&~covered
    mp,mo,mi,_=packed_components(missed,3)
    dist_gt=cv2.distanceTransform((~gt).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    if pred.any():dist_pred=cv2.distanceTransform((~pred).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    else:dist_pred=np.full(gt.shape,np.inf,np.float32)
    hard=(~gt)&(dist_gt>3.)&(dist_pred<=64.)&(np.maximum.reduce([p0,b,p1])>=threshold)
    hp,ho,hi,labels=packed_components(hard,1)
    added=(p1>=threshold)&~pred&~gt
    # A prediction-only bridge proxy: a new component touches two distinct
    # old components. GT confirms its background pixels, not its road topology.
    _,old=cv2.connectedComponents(pred.astype(np.uint8),connectivity=8)
    count,new,new_stats,_=cv2.connectedComponentsWithStats(added.astype(np.uint8),connectivity=8)
    bridge=np.zeros(gt.shape,bool)
    for k in range(1,count):
        x,y,w,h,area=map(int,new_stats[k])
        if area<3:continue
        x0,x1=max(0,x-2),min(gt.shape[1],x+w+2);y0,y1=max(0,y-2),min(gt.shape[0],y+h+2)
        region=new[y0:y1,x0:x1]==k
        near=cv2.dilate(region.astype(np.uint8),np.ones((5,5),np.uint8))>0
        ids=np.unique(old[y0:y1,x0:x1][near]);ids=ids[ids>0]
        if len(ids)>=2:bridge[y0:y1,x0:x1]|=region
    priority=np.unique(labels[hard&(added|bridge)]);priority=priority[priority>0]
    return {'missing_points':mp,'missing_offsets':mo,'missing_ids':mi.astype(np.int32),
            'hard_points':hp,'hard_offsets':ho,'hard_ids':hi.astype(np.int32),
            'priority_ids':priority.astype(np.int32),
            'added_fp_points':np.flatnonzero(hard&added).astype(np.int32),
            'bridge_proxy_points':np.flatnonzero(hard&bridge).astype(np.int32),
            'near_background_points':np.flatnonzero((~gt)&(dist_gt>3.)&(dist_pred<=64.)).astype(np.int32),
            'reference_counts':np.asarray([len(mi),len(hi),int(hard.sum()),int((hard&added).sum()),int((hard&bridge).sum())],np.int64)}


def choose_crop(reference,mode,seed,epoch,index,height=1500,width=1500,margin=64,probabilities=(.5,.25,.25)):
    rng=np.random.default_rng(seed_for(seed,epoch,index,'sampling'))
    draw=float(rng.random());requested=('random' if mode=='random' or draw<probabilities[0]
        else 'missing' if draw<probabilities[0]+probabilities[1] else 'hard')
    actual=requested;component=-1;boundary_failed=False;priority=False
    if requested=='random':
        top,left=crop_position(seed,index,0,epoch,height,width,512)
    else:
        ids=reference[requested+'_ids'];offsets=reference[requested+'_offsets'];points=reference[requested+'_points']
        if not len(ids):
            actual='random';top,left=crop_position(seed,index,0,epoch,height,width,512)
        else:
            available=np.arange(len(ids))
            if requested=='hard':
                preferred=np.flatnonzero(np.isin(ids,reference['priority_ids']))
                if len(preferred) and rng.random()<.7:available=preferred;priority=True
            chosen=int(rng.choice(available));component=int(ids[chosen])
            pos=int(rng.choice(points[int(offsets[chosen]):int(offsets[chosen+1])]))
            y,x=divmod(pos,width)
            def start(value,limit):
                nonlocal boundary_failed
                low=max(0,value-(511-margin));high=min(limit-512,value-margin)
                if low>high:
                    boundary_failed=True
                    return max(0,min(limit-512,value-256))
                return int(rng.integers(low,high+1))
            top,left=start(y,height),start(x,width)
    return {'top':int(top),'left':int(left),'requested':requested,'actual':actual,
            'fallback':requested!=actual,'component':component,'boundary_failed':boundary_failed,'priority':priority}


class CandidateCropDataset(Dataset):
    def __init__(self,root,cache_dir,mode,seed,probabilities=(.5,.25,.25)):
        if mode not in ('random','hard'):raise ValueError('Unknown sampling mode')
        self.full=full_dataset(root,'train');self.cache_dir=Path(cache_dir);self.mode=mode;self.seed=int(seed);self.epoch=0
        self.probabilities=tuple(probabilities)
        self.available_components={'missing':0,'hard':0}
        for image_name in self.full.image_files:
            name=Path(image_name).stem.replace('_sat','')
            with np.load(self.cache_dir/(name+'.npz')) as saved:
                self.available_components['missing']+=len(saved['missing_ids'])
                self.available_components['hard']+=len(saved['hard_ids'])
    def __len__(self):return len(self.full)
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __getitem__(self,index):
        sample=self.full[index];name=str(sample['case_name'])
        with np.load(self.cache_dir/(name+'.npz')) as saved:
            ref={k:saved[k].copy() for k in ('missing_points','missing_offsets','missing_ids','hard_points','hard_offsets','hard_ids','priority_ids')}
            expected=str(saved['input_sha256'])
        from diagnostics.diagnose_full_h0_optimistic_reachability import signature
        if signature(sample)!=expected:raise ValueError('Training image/GT differs from cached reference: '+name)
        choice=choose_crop(ref,self.mode,self.seed,self.epoch,index,probabilities=self.probabilities)
        top,left=choice['top'],choice['left'];region=(slice(top,top+512),slice(left,left+512))
        # Reconstruct RGB from the dataset's ImageNet normalization before crop augmentation.
        rgb=sample['image'].numpy().transpose(1,2,0)
        rgb=np.clip(np.rint((rgb*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255),0,255).astype(np.uint8)
        image=rgb[region].copy();mask=sample['mask'][0].numpy()[region].copy();skeleton=sample['skeleton'][0].numpy()[region].copy()
        rng=np.random.RandomState(seed_for(self.seed,self.epoch,index,'augmentation'))
        if rng.rand()<.5:image=np.flip(image,1);mask=np.flip(mask,1);skeleton=np.flip(skeleton,1)
        if rng.rand()<.5:image=np.flip(image,0);mask=np.flip(mask,0);skeleton=np.flip(skeleton,0)
        rotations=int(rng.randint(0,4))
        if rotations:image=np.rot90(image,rotations);mask=np.rot90(mask,rotations);skeleton=np.rot90(skeleton,rotations)
        image=RoadSkeletonDataset._augment_color_degrade(image.copy(),rng=rng).astype(np.float32)/255.
        image=(image-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return {'case_name':name,'image':torch.from_numpy(image.transpose(2,0,1).copy()),
                'mask':torch.from_numpy(mask[None].copy()).float(),'skeleton':torch.from_numpy(skeleton[None].copy()).float(),
                'requested':choice['requested'],'actual':choice['actual'],'fallback':choice['fallback'],
                'boundary_failed':choice['boundary_failed'],'priority':choice['priority'],'component':choice['component']}
