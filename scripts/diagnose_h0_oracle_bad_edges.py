"""Validation-only GT oracle: return confirmed bad local edge weights to stay."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from datasets.native_relation_targets import native_targets
from mass_protocol import blend_weight,sliding_positions
from networks.native_h0_relations import shifted
from selector_transport_reference import closed_gaps
from topology_direction_constants import CONNECTIVITY_DIRECTIONS
from evaluate_native_h0_pair import digest,load_model


def difficult_negative(mask,skel):
    labels=native_targets(mask,skel)
    occupied=mask.reshape(128,4,128,4).any(axis=(1,3))
    pad=np.pad(occupied,1)
    neighbours=np.stack([pad[1+dy:129+dy,1+dx:129+dx] for dy,dx in CONNECTIVITY_DIRECTIONS])
    return labels['connectivity_valid'] & (labels['connectivity']<.5) & occupied[None] & neighbours


class OracleHook:
    def __init__(self):
        self.mask=None;self.count=0;self.weight_sum=0.;self.max_weight_error=0.;self.calls=0

    def __call__(self,module,inputs,output):
        if self.mask is None:return output
        if len(output)!=6 or not isinstance(output[5],dict):raise ValueError('Unexpected Stage3 output')
        info=dict(output[5]);h=info['h2_before_relation'];choice=info['relation_weights']
        mask=self.mask.to(device=h.device,dtype=torch.bool)
        if mask.shape != choice[:,1:].shape:raise ValueError('Bad edge mask shape')
        if module.out.bias is not None:raise ValueError('Expected bias-free H projection')
        removed=choice[:,1:]*mask.to(choice.dtype)
        newchoice=torch.cat((choice[:,:1]+removed.sum(1,keepdim=True),choice[:,1:]-removed),1)
        error=(newchoice.sum(1)-choice.sum(1)).abs().max()
        self.max_weight_error=max(self.max_weight_error,float(error.cpu()))
        original_mixture=torch.zeros_like(h);oracle_mixture=torch.zeros_like(h)
        for j,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            difference=shifted(h,dy,dx)-h
            original_mixture+=choice[:,j+1:j+2]*difference
            oracle_mixture+=newchoice[:,j+1:j+2]*difference
        check=module.gamma*module.out(original_mixture)
        if not torch.allclose(check,info['local_delta'],atol=2e-6,rtol=1e-4):
            raise ValueError('Original local update does not match source-weight formula')
        delta=module.gamma*module.out(oracle_mixture)
        info['highres_skeleton_feat']=h+delta
        info['local_delta']=delta
        info['relation_weights']=newchoice
        module.last_choice=newchoice
        self.count+=int(mask.sum().cpu());self.weight_sum+=float(removed.sum().cpu());self.calls+=1
        return (*output[:5],info)


def pix(pred,gt):
    return {'tp':int((pred&gt).sum()),'fp':int((pred&~gt).sum()),'fn':int((~pred&gt).sum())}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--paired-summary',type=Path,required=True)
    p.add_argument('--gap-manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--threshold',type=float,required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if not 0<=args.threshold<=1:raise ValueError('Invalid threshold')
    sha=digest(args.checkpoint);audit=json.loads(args.paired_summary.read_text(encoding='utf-8'))
    if sha!=audit['B_sha256']:raise ValueError('Checkpoint differs from paired S/C/D audit')
    manifest=json.loads(args.gap_manifest.read_text(encoding='utf-8'))
    if manifest['count']!=198:raise ValueError('Expected fixed 198-gap manifest')
    device=torch.device(args.device);model,_=load_model(args.checkpoint,device,True)
    ds=RoadSkeletonDataset(str(args.data_root),'val',image_size=None,source_patch_size=1500,augment=False,return_full_image=True)
    if len(ds)!=14:raise ValueError('Expected 14 validation images')
    args.out.mkdir(parents=True,exist_ok=True)
    hook=OracleHook();handle=model.swin_unet.decoder_structure_blocks['3'].register_forward_hook(hook)
    rows=[];gaprows=[];weight=blend_weight(512,device)
    try:
      for i in range(len(ds)):
        sample=ds[i];name=sample['case_name'];entry=manifest['images'][name]
        img=sample['image'][None].to(device);gt=sample['mask'][0].numpy()>.5;sk=sample['skeleton'][0].numpy()>.5
        if gt.shape!=(1500,1500):raise ValueError('Expected native 1500x1500 GT')
        canv=[torch.zeros((1,1,1500,1500),device=device) for _ in range(2)];den=torch.zeros_like(canv[0])
        start_count=hook.count;start_weight=hook.weight_sum
        for y in sliding_positions(1500,512,256):
          for x in sliding_positions(1500,512,256):
            image=img[:,:,y:y+512,x:x+512]
            hook.mask=None;base=model(image)[0].float()
            mask=difficult_negative(gt[y:y+512,x:x+512],sk[y:y+512,x:x+512])
            hook.mask=torch.from_numpy(mask[None]);before=hook.calls
            oracle=model(image)[0].float();hook.mask=None
            if hook.calls!=before+1:raise ValueError('Oracle hook not called exactly once')
            if not mask.any() and not torch.allclose(base,oracle,atol=1e-6,rtol=1e-6):
                raise ValueError('No-mask intervention changed output')
            for canvas,logits in zip(canv,(base,oracle)):canvas[:,:,y:y+512,x:x+512]+=logits*weight
            den[:,:,y:y+512,x:x+512]+=weight
        probs=[(c/den).sigmoid()[0,0].cpu().numpy() for c in canv]
        base,oracle=[p>=args.threshold for p in probs]
        a=pix(base,gt);b=pix(oracle,gt)
        ids0=set(closed_gaps(base,entry)[1]);ids1=set(closed_gaps(oracle,entry)[1])
        rows.append({'case':name,'baseline_tp':a['tp'],'baseline_fp':a['fp'],'baseline_fn':a['fn'],
            'oracle_tp':b['tp'],'oracle_fp':b['fp'],'oracle_fn':b['fn'],
            'new_fp':int((oracle&~base&~gt).sum()),'removed_fp':int((base&~oracle&~gt).sum()),
            'new_tp':int((oracle&~base&gt).sum()),'lost_tp':int((base&~oracle&gt).sum()),
            'masked_edges_tile_count':hook.count-start_count,'transferred_weight_sum':hook.weight_sum-start_weight,
            'mean_abs_probability_change':float(np.abs(probs[1]-probs[0]).mean()),
            'baseline_closed':len(ids0),'oracle_closed':len(ids1)})
        for g in entry['gaps']:
            gid=g['id'];gaprows.append({'case':name,'gap_id':gid,'length':g['length'],
                'length_bin':g['length_bin'],'baseline_closed':gid in ids0,'oracle_closed':gid in ids1,
                'newly_closed':gid not in ids0 and gid in ids1,'lost_closed':gid in ids0 and gid not in ids1})
        print(f'[VAL] {i+1}/{len(ds)} {name}',flush=True)
    finally:handle.remove()
    if len(gaprows)!=198:raise ValueError('Validation gaps do not match manifest')
    for filename,records in (('per_image.csv',rows),('fixed_gap_changes.csv',gaprows)):
        with (args.out/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    totals={k:sum(r[k] for r in rows) for k in ('baseline_tp','baseline_fp','baseline_fn','oracle_tp','oracle_fp','oracle_fn','new_fp','removed_fp','new_tp','lost_tp','masked_edges_tile_count','transferred_weight_sum')}
    totals['baseline_closed']=sum(r['baseline_closed'] for r in gaprows)
    totals['oracle_closed']=sum(r['oracle_closed'] for r in gaprows)
    totals['newly_closed']=sum(r['newly_closed'] for r in gaprows)
    totals['lost_closed']=sum(r['lost_closed'] for r in gaprows)
    report={'checkpoint_sha256':sha,'threshold':args.threshold,'gap_manifest':str(args.gap_manifest),
        'summary':totals,'choice_mass_max_abs_error':hook.max_weight_error,'oracle_tile_calls':hook.calls,
        'notes':['GT is used only by this oracle intervention, not a deployable model.',
        'No weight renormalization: masked source weights go to stay; other source weights remain unchanged.',
        'Difficult negative is a native-GT local adjacency proxy, not verified road topology or bridge label.',
        'Each overlapping inference tile is intervened independently; edge counts are tile counts, not unique image edges.',
        'Gap closure is the fixed reference proxy using 80% coverage and globally connected anchors.',
        'Same threshold on the stitched surface; false positives and false negatives use GT masks.']}
    (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[DONE]',args.out,flush=True)

if __name__=='__main__':main()
