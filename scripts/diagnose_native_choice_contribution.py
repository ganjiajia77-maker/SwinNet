"""Measure actual B relation weights and per-edge H residual on frozen val tiles."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from datasets.native_relation_targets import native_targets
from mass_protocol import sliding_positions
from networks.native_h0_relations import shifted
from topology_direction_constants import CONNECTIVITY_DIRECTIONS
from evaluate_native_h0_pair import digest,load_model,owner


def pool(x):
    return x.reshape(128,4,128,4).any(axis=(1,3))


def describe(values):
    if not values: return {'count':0}
    a=np.concatenate(values)
    if not a.size:return {'count':0}
    return {'count':int(a.size),'mean':float(a.mean(dtype=np.float64)),
            'median':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),
            'p99':float(np.quantile(a,.99))}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--paired-summary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args(); prior=json.loads(args.paired_summary.read_text(encoding='utf-8'))
    ckpt_sha=digest(args.checkpoint)
    if ckpt_sha!=prior['B_sha256']:raise ValueError('Checkpoint differs from prior paired audit')
    cutoff_info=prior['C']['B/C3/road_road_negative']['0.8']
    if cutoff_info is None:raise ValueError('Paired audit has no positive or difficult negative examples')
    cutoff=float(cutoff_info['threshold_approx'])
    device=torch.device(args.device);model,_=load_model(args.checkpoint,device,True)
    block=model.swin_unet.decoder_structure_blocks['3']
    if block.out.bias is not None:raise ValueError('Per-edge decomposition requires bias-free output projection')
    dataset=RoadSkeletonDataset(str(args.data_root),'val',image_size=None,source_patch_size=1500,augment=False,return_full_image=True)
    if len(dataset)!=14:raise ValueError('Expected the same 14 validation images')
    args.out.mkdir(parents=True,exist_ok=True)
    groups=('positive','road_road_negative')
    vals={g:{k:[] for k in ('c','weight','contribution_norm','aligned_component')} for g in groups}
    image_rows=[]; max_residual=0.; unit_error=0.
    for idx in range(len(dataset)):
        sample=dataset[idx]; image=sample['image'][None].to(device)
        gt=sample['mask'][0].numpy()>.5;sk=sample['skeleton'][0].numpy()>.5
        sums={g:{'count':0,'c_sum':0.,'weight_sum':0.,'norm_sum':0.,'rejected':0,'rejected_weight_sum':0.} for g in groups}
        for top in sliding_positions(1500,512,256):
          for left in sliding_positions(1500,512,256):
            output=model(image[:,:,top:top+512,left:left+512]); rows=[x for x in output[4] if x.get('stage')==3]
            if len(rows)!=1:raise ValueError('Expected one Stage3 output')
            row=rows[0];h=row['h2_before_relation'];choice=row['relation_weights'];delta=row['local_delta']
            if choice.shape[1]!=9:raise ValueError('Expected one stay and eight source weights')
            unit_error=max(unit_error,float((choice.sum(1)-1).abs().max().cpu()))
            label=native_targets(gt[top:top+512,left:left+512],sk[top:top+512,left:left+512])
            core=owner(top,left);valid=label['connectivity_valid']&core[None,:,:]
            positive=label['connectivity']>.5; occupied=pool(gt[top:top+512,left:left+512]);pad=np.pad(occupied,1)
            neighbour=np.stack([pad[1+dy:129+dy,1+dx:129+dx] for dy,dx in CONNECTIVITY_DIRECTIONS])
            masks={'positive':valid&positive,'road_road_negative':valid&~positive&occupied[None,:,:]&neighbour}
            c=row['connectivity'][0].sigmoid().cpu().numpy();reconstructed=torch.zeros_like(delta)
            total_norm=delta.square().sum(1).sqrt().clamp_min(1e-12)
            for d,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS):
                weight=choice[:,d+1:d+2]
                contribution=block.gamma*block.out(shifted(h,dy,dx)-h)*weight
                reconstructed+=contribution
                norm=contribution.square().sum(1).sqrt()[0].cpu().numpy()
                alignment=((contribution*delta).sum(1)/total_norm)[0].cpu().numpy()
                w=weight[0,0].cpu().numpy()
                for group in groups:
                    mask=masks[group][d];n=int(mask.sum())
                    if not n:continue
                    cv=c[d][mask];wv=w[mask];nv=norm[mask];av=alignment[mask]
                    for key,v in (('c',cv),('weight',wv),('contribution_norm',nv),('aligned_component',av)):
                        vals[group][key].append(v.copy())
                    s=sums[group];s['count']+=n;s['c_sum']+=float(cv.sum());s['weight_sum']+=float(wv.sum())
                    s['norm_sum']+=float(nv.sum());rejected=cv<cutoff;s['rejected']+=int(rejected.sum())
                    s['rejected_weight_sum']+=float(wv[rejected].sum())
            max_residual=max(max_residual,float((reconstructed-delta).abs().max().cpu()))
        for group,s in sums.items():
            n=s['count'];rn=s['rejected']
            image_rows.append({'case':sample['case_name'],'group':group,'count':n,
                'mean_C':s['c_sum']/n if n else None,'mean_weight':s['weight_sum']/n if n else None,
                'mean_contribution_norm':s['norm_sum']/n if n else None,
                'C_rejected_count':rn,'C_rejected_mean_weight':s['rejected_weight_sum']/rn if rn else None})
        print(f'[VAL] {idx+1}/{len(dataset)} {sample["case_name"]}',flush=True)
    report={'B_sha256':ckpt_sha,'source_paired_summary':str(args.paired_summary),
        'C_threshold':cutoff,'C_threshold_origin':'B C3 threshold at approximately 80% positive recall from paired validation audit',
        'choice_sum_max_abs_error':unit_error,'per_edge_reconstruction_max_abs_error':max_residual,
        'groups':{},'notes':['The prediction forward does not receive GT; masks are used for evaluation only.',
        'Per-edge contribution = gamma * weight * output_projection(H_neighbor-H_center); L2 over 64 H channels.',
        'Aligned component is signed projection of edge contribution onto total local H update; cancellation is possible.',
        'C is an input to the selector, so weight and C are not independent decisions.',
        'Correct means positive local GT adjacency; difficult negatives have road in both cells but no valid native cross-boundary connection.',
        'Statistics cover the same tile ownership and GT labels as the paired audit; no inference intervention or causal proof.',
        'Values are edge-weighted; per-image results are provided and the threshold was set on this validation set.']}
    for group in groups:
        report['groups'][group]={k:describe(v) for k,v in vals[group].items()}
        c=np.concatenate(vals[group]['c']) if vals[group]['c'] else np.array([])
        w=np.concatenate(vals[group]['weight']) if vals[group]['weight'] else np.array([])
        n=np.concatenate(vals[group]['contribution_norm']) if vals[group]['contribution_norm'] else np.array([])
        rejected=c<cutoff
        report['groups'][group]['C_rejected']={'count':int(rejected.sum()),
            'mean_weight':float(w[rejected].mean()) if rejected.any() else None,
            'mean_contribution_norm':float(n[rejected].mean()) if rejected.any() else None}
    for group,key in (('positive','positive_count'),('road_road_negative','negative_count')):
        actual=report['groups'][group]['c']['count']
        if actual!=cutoff_info[key]:
            raise ValueError(f'Paired audit sample count mismatch for {group}: {actual} vs {cutoff_info[key]}')
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    with (args.out/'per_image.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(image_rows[0]));writer.writeheader();writer.writerows(image_rows)
    print('[DONE]',args.out,flush=True)

if __name__=='__main__':main()
