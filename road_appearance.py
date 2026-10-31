"""Training-only GT-directed RGB shadow/texture; no model/label mutation."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import time

import cv2
import numpy as np

VERSION='road_appearance_v1'


@dataclass(frozen=True)
class AppearanceConfig:
    enabled: bool=False
    clean_probability: float=.5
    shadow_probability: float=.25
    texture_probability: float=.25
    road_probability: float=.8
    max_attempts: int=10
    donor_attempts: int=10
    exclusion_radius: int=8
    boundary_margin: int=16
    tangent_radius: int=10
    tangent_min_points: int=6
    tangent_min_fraction: float=.85
    angle_jitter_degrees: float=15.
    center_jitter_pixels: float=2.
    visible_context_pixels: float=8.
    background_distance: tuple=(6.,64.)
    background_max_road_fraction: float=.05
    donor_max_road_fraction: float=.01
    donor_min_spatial_std: float=2.
    shadow_length: tuple=(16.,64.)
    shadow_width_factor: tuple=(1.5,3.)
    shadow_width: tuple=(16.,96.)
    shadow_transition: tuple=(2.,4.)
    shadow_alpha: tuple=(.35,.75)
    texture_length: tuple=(8.,32.)
    texture_width_factor: tuple=(1.5,2.5)
    texture_width: tuple=(12.,64.)
    texture_transition: tuple=(1.,2.)
    along_road_axis_is_long_side: bool=True

    def __post_init__(self):
        if abs(self.clean_probability+self.shadow_probability+self.texture_probability-1)>1e-9:
            raise ValueError('Exclusive appearance probabilities must sum to1')
        if not 0<=self.road_probability<=1 or not 1<=self.max_attempts<=10 or not 1<=self.donor_attempts<=10:
            raise ValueError('Invalid probability / retry budget')


def independent_rng(seed,epoch,sample_id,crop_ordinal=0,salt='training'):
    text=f'{VERSION}|{seed}|{epoch}|{sample_id}|{crop_ordinal}|{salt}'
    value=int.from_bytes(hashlib.blake2b(text.encode(),digest_size=8).digest(),'little')
    return np.random.Generator(np.random.PCG64(value))


def array_sha(value):
    return hashlib.sha256(str(value.shape).encode()+value.dtype.str.encode()+np.ascontiguousarray(value).tobytes()).hexdigest()


def tangent_at(skeleton,y,x,config):
    radius=config.tangent_radius
    y0,y1=max(0,y-radius),min(skeleton.shape[0],y+radius+1)
    x0,x1=max(0,x-radius),min(skeleton.shape[1],x+radius+1)
    yy,xx=np.nonzero(skeleton[y0:y1,x0:x1])
    if len(xx)<config.tangent_min_points: return None
    points=np.stack((xx+x0,yy+y0),1).astype(np.float64)
    eigenvalues,eigenvectors=np.linalg.eigh(np.cov(points,rowvar=False))
    fraction=float(eigenvalues[-1]/max(float(eigenvalues.sum()),1e-8))
    if fraction<config.tangent_min_fraction: return None
    vector=eigenvectors[:,-1]
    return float(math.atan2(vector[1],vector[0])),fraction


def rectangle_mask(shape,cx,cy,length,width,angle,transition):
    """Interior is exactly1; transition is inside an RGB-independent rectangle."""
    yy,xx=np.indices(shape,dtype=np.float32);dx=xx-cx;dy=yy-cy
    u=dx*math.cos(angle)+dy*math.sin(angle)
    v=-dx*math.sin(angle)+dy*math.cos(angle)
    signed=np.minimum(length*.5-np.abs(u),width*.5-np.abs(v))
    return np.clip(signed/max(transition,1e-6),0,1).astype(np.float32)


def road_disk_visible(road,skeleton,x,y,radius,mask):
    x0,x1=max(0,int(x-radius)),min(road.shape[1],int(x+radius)+2)
    y0,y1=max(0,int(y-radius)),min(road.shape[0],int(y+radius)+2)
    if x0>=x1 or y0>=y1: return False
    yy,xx=np.mgrid[y0:y1,x0:x1];disk=(xx-x)**2+(yy-y)**2<=radius**2
    return bool((disk&road[y0:y1,x0:x1]&(mask[y0:y1,x0:x1]==0)).any() and (disk&skeleton[y0:y1,x0:x1]&(mask[y0:y1,x0:x1]==0)).any())


def safe_points(road,skeleton,config):
    degree=cv2.filter2D(skeleton.astype(np.int16),-1,np.ones((3,3),np.int16))-skeleton
    critical=skeleton&((degree<=1)|(degree>=3))
    radius=config.exclusion_radius
    exclusion=cv2.dilate(critical.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
    valid=skeleton&road&~exclusion
    margin=config.boundary_margin
    valid[:margin]=False;valid[-margin:]=False;valid[:,:margin]=False;valid[:,-margin:]=False
    return np.argwhere(valid)


def donor_patch(image,road,bounds,rng,config):
    x0,y0,x1,y1=bounds;height=y1-y0;width=x1-x0
    integral=cv2.integral(road.astype(np.uint8));reasons=Counter()
    for attempt in range(1,config.donor_attempts+1):
        x=int(rng.integers(0,image.shape[1]-width+1));y=int(rng.integers(0,image.shape[0]-height+1))
        if x<x1 and x+width>x0 and y<y1 and y+height>y0:
            reasons['donor_overlaps_target']+=1;continue
        fraction=float((integral[y+height,x+width]-integral[y,x+width]-integral[y+height,x]+integral[y,x])/(height*width))
        if fraction>config.donor_max_road_fraction:
            reasons['donor_contains_road']+=1;continue
        patch=image[y:y+height,x:x+width]
        if float(patch.astype(np.float32).std((0,1)).mean())<config.donor_min_spatial_std:
            reasons['donor_no_texture']+=1;continue
        return patch.copy(),{'donor_bbox':[x,y,x+width,y+height],'donor_road_fraction':fraction,'donor_attempts':attempt,'donor_rejections':dict(reasons)}
    return None,{'donor_attempts':config.donor_attempts,'donor_rejections':dict(reasons)}


def apply_appearance(image,road,skeleton,rng,config=AppearanceConfig(),force=None,debug=False):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3: raise ValueError('Expected pre-normalization uint8 RGB')
    if road.shape!=image.shape[:2] or skeleton.shape!=road.shape: raise ValueError('RGB/road/skeleton must share final crop coordinates')
    start=time.perf_counter();road=road.astype(bool,copy=False);skeleton=skeleton.astype(bool,copy=False)
    metadata={'version':VERSION,'enabled':config.enabled,'requested_type':'clean','requested_target':'none','success':False,
        'skip_reason':'disabled' if not config.enabled else 'not_requested','placement_attempts':0,'donor_attempts':0,'rejections':{}}
    def finish(result,mask=None):
        metadata['augmentation_seconds']=time.perf_counter()-start
        return result,metadata,(np.zeros(road.shape,np.float32) if mask is None else mask) if debug else None
    if not config.enabled and force is None: return finish(image)
    if force is None:
        draw=float(rng.random())
        if draw<config.clean_probability: return finish(image)
        kind='shadow' if draw<config.clean_probability+config.shadow_probability else 'texture'
        target='road' if rng.random()<config.road_probability else 'background'
    else:
        kind,target=force
        if kind not in ('shadow','texture') or target not in ('road','background'): raise ValueError('Unknown preview-only forced case')
    metadata.update(requested_type=kind,requested_target=target,skip_reason='no_reliable_position')
    points=safe_points(road,skeleton,config)
    if not len(points): metadata['skip_reason']='no_regular_skeleton';return finish(image)
    distances=cv2.distanceTransform(road.astype(np.uint8),cv2.DIST_L2,5)
    backgrounds=None;nearest=None
    if target=='background':
        from scipy.ndimage import distance_transform_edt
        distance,nearest=distance_transform_edt(~skeleton,return_indices=True)
        # Road distance excludes the road width itself; nearest skeleton gives
        # a reliable tangent and road-width source, not a model input.
        road_distance=distance_transform_edt(~road)
        minimum,maximum=config.background_distance
        valid=(~road)&(road_distance>=minimum)&(road_distance<=maximum)
        reliable=np.zeros(road.shape,bool);reliable[points[:,0],points[:,1]]=True
        valid &= reliable[nearest[0],nearest[1]]
        backgrounds=np.argwhere(valid)
        if not len(backgrounds): metadata['skip_reason']='no_near_road_background';return finish(image)
    rejections=Counter();selected=None
    for attempt in range(1,config.max_attempts+1):
        metadata['placement_attempts']=attempt
        if target=='road':
            sy,sx=map(int,points[int(rng.integers(len(points)))]);cy,cx=float(sy),float(sx)
        else:
            by,bx=map(int,backgrounds[int(rng.integers(len(backgrounds)))]);cy,cx=float(by),float(bx)
            sy,sx=int(nearest[0,by,bx]),int(nearest[1,by,bx])
        estimated=tangent_at(skeleton,sy,sx,config)
        if estimated is None: rejections['unreliable_tangent']+=1;continue
        angle,reliability=estimated
        angle+=math.radians(float(rng.uniform(-config.angle_jitter_degrees,config.angle_jitter_degrees)))
        cx+=float(rng.uniform(-config.center_jitter_pixels,config.center_jitter_pixels))
        cy+=float(rng.uniform(-config.center_jitter_pixels,config.center_jitter_pixels))
        road_width=max(1.,2.*float(distances[sy,sx]))
        length=float(rng.uniform(*getattr(config,kind+'_length')))
        width=float(np.clip(road_width*rng.uniform(*getattr(config,kind+'_width_factor')),*getattr(config,kind+'_width')))
        if config.along_road_axis_is_long_side and width>length:
            rejections['width_exceeds_along_road_length']+=1;continue
        transition=float(rng.uniform(*getattr(config,kind+'_transition')))
        ex=abs(math.cos(angle))*length*.5+abs(math.sin(angle))*width*.5
        ey=abs(math.sin(angle))*length*.5+abs(math.cos(angle))*width*.5
        margin=config.visible_context_pixels
        if cx-ex<margin or cx+ex>=image.shape[1]-margin or cy-ey<margin or cy+ey>=image.shape[0]-margin:
            rejections['rectangle_near_crop_boundary']+=1;continue
        mask=rectangle_mask(road.shape,cx,cy,length,width,angle,transition);support=mask>0
        if not support.any(): rejections['empty_rectangle']+=1;continue
        fraction=float(road[support].mean())
        if target=='background':
            if fraction>config.background_max_road_fraction:
                rejections['background_rectangle_contains_road']+=1;continue
        else:
            context=length*.5+margin;radius=max(2.,min(road_width*.5,6.))
            ends=[(cx+sign*context*math.cos(angle),cy+sign*context*math.sin(angle)) for sign in (-1,1)]
            if not all(road_disk_visible(road,skeleton,x,y,radius,mask) for x,y in ends):
                rejections['visible_road_context_missing']+=1;continue
        yy,xx=np.nonzero(support);bounds=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
        selected=(mask,bounds);metadata.update(center_xy=[cx,cy],source_skeleton_xy=[sx,sy],estimated_road_width=road_width,
            length=length,width=width,angle_degrees=math.degrees(angle),tangent_reliability=reliability,
            transition=transition,area_pixels=int(support.sum()),area_fraction=float(support.mean()),target_road_fraction=fraction,
            full_strength_fraction=float((mask>=1.-1e-6).sum()/max(support.sum(),1)))
        break
    metadata['rejections']=dict(rejections)
    if selected is None: return finish(image)
    mask,bounds=selected
    if kind=='shadow':
        alpha=float(rng.uniform(*config.shadow_alpha));metadata['alpha']=alpha
        result=np.rint(image.astype(np.float32)*(1.-mask[...,None]*(1.-alpha))).clip(0,255).astype(np.uint8)
    else:
        donor,info=donor_patch(image,road,bounds,rng,config);metadata.update(info)
        if donor is None: metadata['skip_reason']='no_nonroad_texture_donor';return finish(image)
        x0,y0,x1,y1=bounds;blend=mask[y0:y1,x0:x1,None]
        result=image.copy()
        result[y0:y1,x0:x1]=np.rint((1.-blend)*image[y0:y1,x0:x1]+blend*donor).clip(0,255).astype(np.uint8)
    if not np.any(result!=image): metadata['skip_reason']='no_rgb_change';return finish(image)
    metadata.update(success=True,skip_reason='')
    return finish(result,mask)


class AppearanceStats:
    def __init__(self): self.counts=Counter();self.measurements=[]
    def add(self,record):
        kind,target=record['requested_type'],record['requested_target'];prefix=kind+':'+target
        self.counts['tiles']+=1;self.counts[prefix+':requested']+=1
        if kind!='clean':
            self.counts['perturbation_requested']+=1
            self.counts[prefix+(':success' if record['success'] else ':skipped')]+=1
            self.counts['success']+=int(record['success'])
        if kind!='clean' and not record['success']: self.counts['skip:'+record['skip_reason']]+=1
        for reason,amount in record.get('rejections',{}).items(): self.counts['placement_reject:'+reason]+=amount
        for reason,amount in record.get('donor_rejections',{}).items(): self.counts['donor_reject:'+reason]+=amount
        self.measurements.append(record)
    def report(self):
        values={}
        for field in ('length','width','area_fraction','alpha','augmentation_seconds','read_seconds','dataset_seconds'):
            data=[r[field] for r in self.measurements if field in r and (r['success'] or field.endswith('seconds'))]
            if data: values[field]={'mean':float(np.mean(data)),'p10':float(np.quantile(data,.1)),'p50':float(np.quantile(data,.5)),'p90':float(np.quantile(data,.9))}
        proportions={}
        for target in ('road','background'):
            for status in ('requested','success'):
                proportions[target+'_'+status+'_fraction_all_tiles']=sum(self.counts[kind+':'+target+':'+status] for kind in ('shadow','texture'))/max(self.counts['tiles'],1)
        return {'counts':dict(self.counts),**proportions,'successful_fraction_all_tiles':self.counts['success']/max(self.counts['tiles'],1),
            'successful_fraction_requested':self.counts['success']/max(self.counts['perturbation_requested'],1),'measurements':values}
