"""Paired crop-first pipeline. Appearance RNG never consumes base augmentation RNG."""
import json
import time

import cv2
import numpy as np
import torch

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from mass_protocol import crop_position
from road_appearance import AppearanceConfig,apply_appearance,independent_rng,array_sha


class RoadAppearanceDataset(RoadSkeletonDataset):
    def __init__(self,root,seed=1234,config=AppearanceConfig(),augment=True):
        super().__init__(str(root),'train',image_size=512,source_patch_size=1500,
            random_crop_train=True,random_crops_per_image=1,random_crop_seed=seed,augment=augment)
        self.appearance_config=config

    def sample(self,index,force=None,debug=False,preview_salt='training'):
        started=time.perf_counter();image_index=index//self.random_crops_per_image;ordinal=index%self.random_crops_per_image
        name=self.image_files[image_index]
        from pathlib import Path
        rgb=cv2.imread(str(Path(self.image_dir)/name),cv2.IMREAD_COLOR)
        road=cv2.imread(str(Path(self.mask_dir)/self.label_files[image_index]),cv2.IMREAD_GRAYSCALE)
        decode_seconds=time.perf_counter()-started
        if rgb is None or road is None or rgb.shape[:2]!=(1500,1500) or road.shape!=(1500,1500): raise ValueError('Expected native1500 training RGB and GT')
        values=np.unique(road)
        if not np.isin(values,(0,1,255)).all() or (1 in values and 255 in values): raise ValueError('Expected binary training road GT')
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB);road=(road>0).astype(np.uint8)*255
        top,left=crop_position(self.random_crop_seed,image_index,ordinal,self.epoch,1500,1500,512)
        rgb=rgb[top:top+512,left:left+512].copy();road=road[top:top+512,left:left+512].copy()
        rng=np.random.RandomState((self.random_crop_seed+self.epoch*1000003+image_index*9176+ordinal*101)%(2**32))
        geometry=None
        if self.augment:
            probe=np.random.RandomState();probe.set_state(rng.get_state())
            geometry=[bool(probe.rand()<.5),bool(probe.rand()<.5),int(probe.randint(0,4))]
            rgb,road=self._augment_geometry(rgb,road,rng=rng)
            rgb=self._augment_color_degrade(rgb,rng=rng)
        road=(road>127).astype(np.float32)
        skeleton=(self._skeletonize_binary(road)>127).astype(np.float32)
        dilated=(self._dilate_skeleton(skeleton*255,1)>127).astype(np.float32)
        before_sha=array_sha(rgb);label_sha=array_sha(road);skeleton_sha=array_sha(skeleton)
        before=rgb.copy() if debug else None
        local_rng=independent_rng(self.random_crop_seed,self.epoch,name,ordinal,preview_salt)
        config=self.appearance_config if self.augment or force is not None else AppearanceConfig(enabled=False)
        rgb,metadata,mask=apply_appearance(rgb,road,skeleton,local_rng,config,force=force,debug=debug)
        if array_sha(road)!=label_sha or array_sha(skeleton)!=skeleton_sha: raise RuntimeError('RGB-only augmentation modified labels')
        metadata.update(image_name=name,sample_id=image_index,crop_ordinal=ordinal,epoch=self.epoch,tile_top=top,tile_left=left,
            base_geometry=geometry,before_local_sha256=before_sha,road_sha256=label_sha,skeleton_sha256=skeleton_sha,
            read_seconds=decode_seconds,dataset_seconds=time.perf_counter()-started)
        image=(rgb.astype(np.float32)/255.-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        sample={'image':torch.from_numpy(image.transpose(2,0,1).copy()).float(),
            'mask':torch.from_numpy(road[None].copy()).float(),'skeleton':torch.from_numpy(skeleton[None].copy()).float(),
            'skeleton_dilate':torch.from_numpy(dilated[None].copy()).float(),'image_name':name,
            'case_name':name.rsplit('.',1)[0].replace('_sat',''),'tile_top':top,'tile_left':left,
            'appearance_metadata':json.dumps(metadata)}
        if debug: return sample,{'before':before,'after':rgb,'road':road,'skeleton':skeleton,'mask':mask,'metadata':metadata}
        return sample

    def __getitem__(self,index): return self.sample(index)
