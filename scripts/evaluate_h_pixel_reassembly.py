"""Native1500 validation sweep, fixed-threshold test, optional shared six metrics."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_native_h0_pair import load_model, infer, digest
from datasets.dataset_road_skeleton import RoadSkeletonDataset


def dump(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(tp,fp,fn):
    return dict(iou=tp/max(tp+fp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
                precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))


def save(path,mask):
    if not cv2.imwrite(str(path),mask.astype(np.uint8)*255):raise IOError(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--root_path',type=Path,default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--command',choices=['all','infer','topology','show'],default='all')
    p.add_argument('--topo_tool',type=Path,default=Path('/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();out=a.output_dir
    if a.command in ('all','infer'):
        if not a.checkpoint or not a.checkpoint.is_file():p.error('Specify an existing checkpoint')
        if out.exists():p.error('Use a new inference output directory; topology/show reuse an existing one')
        out.mkdir(parents=True)
        device=torch.device(a.device)
        model,config=load_model(a.checkpoint,device,True)
        if not config.get('h_pixel_reassembly'):raise ValueError('Expected H pixel reassembly checkpoint')
        model.requires_grad_(False)
        name='pixel_'+config['h_reassembly_prior']
        if config.get('no_encoder_a1a2',False):name+='_no_a1a2'
        thresholds=np.round(np.arange(.05,.951,.01),2).astype(np.float32)
        counts=np.zeros((len(thresholds),3),np.int64)
        probabilities=out/'val_probabilities';probabilities.mkdir()
        cases=[]
        with torch.inference_mode():
            data=RoadSkeletonDataset(str(a.root_path),'val',image_size=None,source_patch_size=1500,return_full_image=True)
            if len(data)!=14:raise ValueError('Expected 14 validation images')
            for i in range(len(data)):
                sample=data[i];prob,gt,_=infer(model,sample,device,False);case=sample['case_name']
                np.save(probabilities/(case+'.npy'),prob.astype(np.float32));cases.append(case)
                for j,t in enumerate(thresholds):
                    pred=prob>=t
                    counts[j]+=(int((pred & gt).sum()),int((pred & ~gt).sum()),int((~pred & gt).sum()))
                print(f'[VAL] {i+1}/{len(data)} {case}',flush=True)
        curves=[dict(threshold=float(t),tp=int(c[0]),fp=int(c[1]),fn=int(c[2]),**metrics(*map(int,c)))
                for t,c in zip(thresholds,counts)]
        table(out/'val_threshold_scores.csv',curves)
        selected=max(curves,key=lambda r:(r['iou'],-r['threshold']))
        t=selected['threshold'];sha=digest(a.checkpoint)
        dump(out/'val_selection.json',dict(selected_on='val',criterion='global_iou',checkpoint_sha256=sha,**selected))
        val_masks=out/'val_masks';test_masks=out/'test_masks';val_masks.mkdir();test_masks.mkdir()
        for case in cases:save(val_masks/(case+'.png'),np.load(probabilities/(case+'.npy'))>=t)
        rows=[]
        with torch.inference_mode():
            data=RoadSkeletonDataset(str(a.root_path),'test',image_size=None,source_patch_size=1500,return_full_image=True)
            if len(data)!=49:raise ValueError('Expected 49 test images')
            for i in range(len(data)):
                sample=data[i];prob,gt,_=infer(model,sample,device,False);case=sample['case_name'];pred=prob>=t
                save(test_masks/(case+'.png'),pred)
                tp,fp,fn=int((pred & gt).sum()),int((pred & ~gt).sum()),int((~pred & gt).sum())
                rows.append(dict(case=case,threshold=t,tp=tp,fp=fp,fn=fn,**metrics(tp,fp,fn)))
                print(f'[TEST] {i+1}/{len(data)} {case}',flush=True)
        table(out/'test_per_image.csv',rows)
        total=[sum(r[k] for r in rows) for k in ('tp','fp','fn')]
        table(out/'test_segmentation.csv',[dict(model=name,threshold=t,**metrics(*total))])
        dump(out/'manifest.json',dict(data_root=str(a.root_path.resolve()),protocol=dict(image_size=1500),models=[dict(
            name=name,provenance=dict(checkpoint=str(a.checkpoint.resolve()),sha256=sha,protocol='native1500 FP32 512/256 weighted logits; no TTA'),
            validation_selection=dict(selected_on='val',criterion='global_iou',threshold=t,evidence=str((out/'val_selection.json').resolve())),
            predictions=dict(val=[dict(path=str(val_masks.resolve()),threshold=t)],test=[dict(path=str(test_masks.resolve()),threshold=t)]))]))
        print('[INFERENCE DONE]',out,flush=True)
    if a.command in ('all','topology'):
        if not a.topo_tool.is_file():raise FileNotFoundError(a.topo_tool)
        subprocess.run([sys.executable,'-u',str(a.topo_tool),'--manifest',str(out/'manifest.json'),
                        '--split','test','--import_val_selection','--output_dir',str(out/'topology'),
                        '--workers',str(a.workers),'--progress_every','1'],check=True)
        print('[ALL DONE]',out,flush=True)
    if a.command=='show':
        for filename in ('val_threshold_scores.csv','test_segmentation.csv','topology/test_comparison.csv'):
            path=out/filename;print('\n========== '+filename+' ==========')
            if not path.exists():print('Not generated:',path);continue
            with path.open(newline='',encoding='utf-8-sig') as f:
                reader=csv.DictReader(f)
                for row in reader:print(' | '.join(f'{k}={v}' for k,v in row.items()))


if __name__=='__main__':main()
