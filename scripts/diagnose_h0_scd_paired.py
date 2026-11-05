"""Paired, validation-only H0 S/C/D audit on the same native labels and tiles."""
import argparse
from collections import defaultdict
import csv
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from datasets.native_relation_targets import native_targets
from mass_protocol import blend_weight, sliding_positions
from selector_transport_reference import closed_gaps
from topology_direction_constants import CONNECTIVITY_DIRECTIONS
from evaluate_native_h0_pair import digest, load_model, owner


def original(path, reference_root, device):
    sys.path.insert(0, str(reference_root.parent))
    from reference_h0_599a410.networks.vision_transformer import SwinUnet
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    defaults = json.loads((reference_root / 'argument_defaults.json').read_text())
    saved = ckpt['args']
    if not isinstance(saved, dict): saved = vars(saved)
    args = dict(defaults, **saved)
    opts = SimpleNamespace(cfg=str(ROOT/'configs/swin_tiny_patch4_window7_224_lite.yaml'),
        opts=None, batch_size=2, img_size=512, zip=False, cache_mode=None, resume=None,
        accumulation_steps=1, use_checkpoint=False, amp_opt_level=None, tag=None,
        eval=False, throughput=False)
    config = get_config(opts)
    kwargs = {k: args[k] for k in inspect.signature(SwinUnet).parameters
              if k in args and k not in ('config','img_size','num_classes','return_skeleton')}
    model = SwinUnet(config, img_size=512, num_classes=1, return_skeleton=True, **kwargs)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model.to(device).eval()


def pool(x):
    return x.reshape(128, 4, 128, 4).any(axis=(1,3))


def add_hist(store, name, values, select):
    x = values[select]
    if x.size:
        store[name] += np.histogram(x, bins=1000, range=(0,1))[0]


def summary_hist(pos, neg, recall):
    if not pos.sum() or not neg.sum(): return None
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1])
    k = min(int(np.searchsorted(tp, recall*pos.sum())), 999)
    return {'positive_count': int(pos.sum()), 'negative_count': int(neg.sum()),
            'target_recall': recall, 'achieved_recall': float(tp[k]/pos.sum()),
            'negative_acceptance': float(fp[k]/neg.sum()), 'threshold_approx': (999-k)/1000}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True); p.add_argument('--reference-code',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--threshold',type=float,default=.45)
    p.add_argument('--device',default='cuda'); args=p.parse_args()
    record=json.loads((args.reference_dir/'complete.json').read_text())
    manifest=json.loads((args.reference_dir/'fixed_gaps_and_anchors.json').read_text())
    if digest(args.base)!=record['h0_sha256']: raise ValueError('H0 SHA differs from fixed gap reference')
    if manifest['count']!=198: raise ValueError('Expected 198 fixed gaps')
    args.out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    base=original(args.base,args.reference_code,device)
    native,_=load_model(args.native,device,True)
    data=RoadSkeletonDataset(str(args.data_root),'val',image_size=None,source_patch_size=1500,augment=False,return_full_image=True)
    if len(data)!=14: raise ValueError('Expected 14 validation images')
    counts=defaultdict(float); hist=defaultdict(lambda:np.zeros(1000,dtype=np.int64)); gaps=[]
    for i in range(len(data)):
        sample=data[i]; name=sample['case_name']; entry=manifest['images'][name]
        image=sample['image'][None].to(device); gt=sample['mask'][0].numpy()>.5
        skeleton=sample['skeleton'][0].numpy()>.5
        gap=np.zeros((1500,1500),bool)
        for g in entry['gaps']: gap.ravel()[np.asarray(g['pixels'],dtype=np.int64)]=True
        road_near=cv2.dilate(gt.astype(np.uint8),np.ones((33,33),np.uint8))>0
        canv=[torch.zeros((1,1,1500,1500),device=device) for _ in range(2)]
        div=torch.zeros_like(canv[0]); blend=blend_weight(512,device)
        for top in sliding_positions(1500,512,256):
          for left in sliding_positions(1500,512,256):
            patch=image[:,:,top:top+512,left:left+512]
            outputs=[base(patch),native(patch)]
            for j,o in enumerate(outputs): canv[j][:,:,top:top+512,left:left+512]+=o[0].float()*blend
            div[:,:,top:top+512,left:left+512]+=blend
            tile=gt[top:top+512,left:left+512]; sk=skeleton[top:top+512,left:left+512]
            labels=native_targets(tile,sk); core=owner(top,left); positive=labels['connectivity']>.5
            valid=labels['connectivity_valid'] & core[None,:,:]
            occupied=pool(tile); padded=np.pad(occupied,1)
            neighbour=np.stack([padded[1+dy:129+dy,1+dx:129+dx] for dy,dx in CONNECTIVITY_DIRECTIONS])
            gap_cells=pool(gap[top:top+512,left:left+512]); sk_cells=pool(sk)
            near_cells=pool(road_near[top:top+512,left:left+512])
            groups={'gap':gap_cells&core,'ordinary_skeleton':sk_cells&~gap_cells&core,
                    'near_background':near_cells&~occupied&core}
            edgegroups={'positive':positive,'all_negative':~positive,
                        'road_road_negative':(~positive)&occupied[None,:,:]&neighbour,
                        'gap_positive':positive&gap_cells[None,:,:]}
            for model_name,out in zip(('H0','B'),outputs):
              stages={x.get('stage'):x for x in out[4]}
              for stage in (2,3):
                row=stages[stage]; prefix=f'{model_name}/S{stage}'
                s=row['skeleton'][0,0].sigmoid().cpu().numpy()
                for group,mask in groups.items():
                    key=f'{prefix}/{group}'; counts[key+'/sum']+=float(s[mask].sum());counts[key+'/n']+=int(mask.sum())
                    add_hist(hist,key,s,mask)
                c=row['connectivity'][0].sigmoid().cpu().numpy()
                for group,mask in edgegroups.items(): add_hist(hist,f'{model_name}/C{stage}/{group}',c,valid&mask)
                d=row['direction'][0].cpu().numpy(); norm=np.linalg.norm(d,axis=0)
                cos=(d*labels['direction']).sum(axis=0)/np.maximum(norm,1e-6)
                angle=.5*np.degrees(np.arccos(np.clip(cos,-1,1))); dv=labels['direction_valid']&core
                for group,mask in {'all':dv,'gap':dv&gap_cells}.items():
                    key=f'{model_name}/D{stage}/{group}'; counts[key+'/angle_sum']+=float(angle[mask].sum());counts[key+'/n']+=int(mask.sum())
        probs=[(canvas/div).sigmoid()[0,0].cpu().numpy() for canvas in canv]
        closed=[set(closed_gaps(prob>=args.threshold,entry)[1]) for prob in probs]
        for gaprow in entry['gaps']:
            gid=gaprow['id']; gaps.append({'case':name,'gap_id':gid,'length':gaprow['length'],
                'length_bin':gaprow['length_bin'],'H0_closed':gid in closed[0], 'B_closed':gid in closed[1]})
        print(f'[VAL] {i+1}/{len(data)} {name}',flush=True)
    summary={'H0_sha256':digest(args.base),'B_sha256':digest(args.native),'gap_count':len(gaps),
        'surface_threshold':args.threshold,'notes':['Same 4x4 native labels and owned tiles for both models.',
        'S3 heads differ: old G3 versus new H2; fixed gaps and anchor closures are proxies.',
        'C negative class includes valid background; road_road_negative reports difficult subset.',
        'S gap cells are fixed reference gap pixels mapped to work cells; not whole-gap closure.',
        'Confidence histograms use 0.001 bins; threshold matching is approximate.'], 'S':{},'C':{},'D':{}}
    for model in ('H0','B'):
      for stage in (2,3):
        for group in ('gap','ordinary_skeleton','near_background'):
            key=f'{model}/S{stage}/{group}'; n=counts[key+'/n']; h=hist[key]; cum=np.cumsum(h)
            summary['S'][key]={'count':int(n),'mean':counts[key+'/sum']/n if n else None,
                'p10':int(np.searchsorted(cum,.1*n))/1000 if n else None,
                'median':int(np.searchsorted(cum,.5*n))/1000 if n else None,
                'p90':int(np.searchsorted(cum,.9*n))/1000 if n else None}
        for neg in ('all_negative','road_road_negative'):
            key=f'{model}/C{stage}/{neg}'; pos=hist[f'{model}/C{stage}/positive']; negs=hist[key]
            summary['C'][key]={str(r):summary_hist(pos,negs,r) for r in (.5,.8,.9)}
        for group in ('all','gap'):
            key=f'{model}/D{stage}/{group}'; n=counts[key+'/n']
            summary['D'][key]={'count':int(n),'mean_angle_deg':counts[key+'/angle_sum']/n if n else None}
    summary['closures']={'H0':sum(x['H0_closed'] for x in gaps),'B':sum(x['B_closed'] for x in gaps),
        'new':sum(not x['H0_closed'] and x['B_closed'] for x in gaps),
        'lost':sum(x['H0_closed'] and not x['B_closed'] for x in gaps)}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (args.out/'fixed_gap_changes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(gaps[0]));writer.writeheader();writer.writerows(gaps)
    print('[DONE]',args.out,flush=True)

if __name__=='__main__': main()
