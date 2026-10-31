"""24 forced TRAIN-only QC examples; these do not estimate natural proportions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from road_appearance import AppearanceConfig,AppearanceStats,array_sha
from road_appearance_dataset import RoadAppearanceDataset


def save_rgb(path,rgb):
    if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)): raise OSError(str(path))


def export_examples(root,output,seed=1234,epoch=1,max_samples=1108):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=AppearanceConfig(enabled=True);dataset=RoadAppearanceDataset(root,seed,config);dataset.set_epoch(epoch)
    required=[('shadow','road',8),('texture','road',8),('shadow','background',4),('texture','background',4)]
    stats=AppearanceStats();records=[];cursor=0
    for kind,target,amount in required:
        success=0
        while success<amount and cursor<max_samples:
            index=cursor%len(dataset);salt=f'preview:{kind}:{target}:{cursor}'
            sample,debug=dataset.sample(index,force=(kind,target),debug=True,preview_salt=salt);cursor+=1
            info=debug['metadata'];stats.add(info)
            if not info['success']: continue
            if 'mask' in sample and sample['image'].shape[0]!=3: raise RuntimeError('Model RGB gained a mask channel')
            if 'appearance_mask' in sample or 'tangent' in sample or 'sampling_class' in sample: raise RuntimeError('Debug information entered model data')
            success+=1;stem=f'{len(records)+1:02d}_{target}_{kind}'
            before=debug['before'];after=debug['after'];gt=np.repeat((debug['road']*255).astype(np.uint8)[...,None],3,2)
            gt[debug['skeleton']>.5]=[255,220,0]
            mask=np.repeat(np.rint(debug['mask']*255).astype(np.uint8)[...,None],3,2)
            panels=[]
            for rgb,title in ((before,'Before LOCAL perturbation'),(after,'After LOCAL perturbation'),(gt,'GT unchanged / skeleton yellow'),(mask,'Debug mask ONLY - never model input')):
                bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);panel=cv2.copyMakeBorder(bgr,30,0,0,0,cv2.BORDER_CONSTANT,value=20)
                cv2.putText(panel,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);panels.append(panel)
            cv2.imwrite(str(output/(stem+'_comparison.jpg')),np.concatenate(panels,1))
            save_rgb(output/(stem+'_before.png'),before);save_rgb(output/(stem+'_after.png'),after)
            cv2.imwrite(str(output/(stem+'_gt.png')),(debug['road']*255).astype(np.uint8))
            cv2.imwrite(str(output/(stem+'_mask.png')),np.rint(debug['mask']*255).astype(np.uint8))
            info.update(preview_file=stem+'_comparison.jpg',dataset_index=index,preview_salt=salt,
                actual_model_tensor_shape=list(sample['image'].shape),target_unchanged=True,
                mask_is_debug_only=True,force=[kind,target])
            records.append(info)
        if success!=amount:
            (output/'preview_failed.json').write_text(json.dumps({'required':required,'obtained':len(records),'attempts':cursor,'statistics':stats.report()},indent=2),encoding='utf-8')
            raise RuntimeError(f'Cannot collect {amount} {target}/{kind} examples within total preview budget{max_samples}')
    report={'passed':True,'split':'train','root_path':str(Path(root).resolve()),'epoch':epoch,'seed':seed,'samples':records,'config':asdict(config),
        'counts':{'road_shadow':8,'road_texture':8,'background_shadow':4,'background_texture':4},'statistics':stats.report(),
        'limits':'FORCED preview groups, not actual training proportions. Before RGB already includes the unchanged geometry/color functions.'}
    (output/'preview_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    items=''.join(f'<figure><figcaption>{r["preview_file"]} · {r["image_name"]} · crop ({r["tile_left"]},{r["tile_top"]})</figcaption><a href="{r["preview_file"]}" target="_blank"><img loading="lazy" style="width:100%" src="{r["preview_file"]}"></a></figure>' for r in records)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Training-only appearance QC</title><h1>24组训练增强检查</h1><p>8道路阴影、8道路纹理、8背景扰动。强制展示，不代表自然训练比例。仅RGB进入模型；遮罩只用于检查。</p>'+items,encoding='utf-8')
    print('[PREVIEW PASSED] '+str(output/'index.html'),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root_path',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True);parser.add_argument('--seed',type=int,default=1234);parser.add_argument('--epoch',type=int,default=1)
    args=parser.parse_args();export_examples(args.root_path,args.output_dir,args.seed,args.epoch)
