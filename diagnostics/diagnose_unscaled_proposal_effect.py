"""Read-only attribution of the trained unscaled-proposal B on val images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.audit_full_h0_saved_predictions import component_data,digest,gap_state,read_json,write_csv
from diagnostics.diagnose_full_h0_optimistic_reachability import (atomic_json,rebuild_source,
    signature,validate_saved_prediction,verify_recorded_code)
from mass_protocol import blend_weight,sliding_positions
from networks.full_h0_joint import VERSION,build_h0
from networks.full_h0_proposal_adapt import ProposalAdaptedJoint
from run_q3_r3_mass import full_dataset
from run_slim_end_to_end_mass import seed_all

REGIONS=('fixed_reference_gap','current_missed_GT_skeleton','road_edge_GT',
         'road_interior_GT','near_road_background_GT','Q_high_background_GT',
         'new_FP_GT')
FIELDS=('P0','B','B_minus_P0','U','B_minus_U','A','rB','proposal_coefficient',
        'proposal_delta','transport_delta','anchor_delta','net_delta',
        'proposal_delta_abs','transport_delta_abs','anchor_delta_abs','net_delta_abs',
        'positive_proposal','opposing_transport_anchor','positive_proposal_canceled',
        'positive_support','delta_plus','delta_minus')
def _hist_edges(bound):
    half=np.geomspace(1e-9,bound,300)
    return np.r_[-half[::-1],0.,half]
HIST_EDGES={1.:_hist_edges(1.),4.:_hist_edges(4.),8.:_hist_edges(8.)}


def verify_adaptation_code(config):
    info=config.get('proposal_adaptation',{})
    if info.get('initialization_mode')!='same_reconstructed_pretraining_state_as_old_B':
        raise ValueError('Expected the same-initialization unscaled proposal experiment')
    hashes=info.get('code_sha256',{})
    if not hashes:raise ValueError('Missing unscaled training-code fingerprints')
    for name,expected in hashes.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or digest(path)!=expected:
            raise ValueError('Unscaled training code differs: '+name)


def build_unscaled(checkpoint,device,size=512):
    if checkpoint.get('version')!=VERSION or checkpoint.get('config',{}).get('variant')!='B':
        raise ValueError('Expected complete full-H0 B checkpoint')
    state=checkpoint['model'];source,q=rebuild_source(state)
    base,_=build_h0(checkpoint['config']['h0_args'],size=size)
    model=ProposalAdaptedJoint(base,q,source)
    model.load_state_dict(state,strict=True)
    model.proposal_mode='unscaled'
    return model.to(device).eval()


class Stats:
    """Exact sums/fractions and approximate quantiles via fixed histogram bins."""
    def __init__(self):
        self.n=0;self.sums={k:0. for k in FIELDS};self.hist={k:np.zeros(600,np.int64) for k in FIELDS}
        self.counts={'B_above_U_by_0_1':0,'B_above_P0_by_0_1':0,'positive_proposal':0,
                     'negative_proposal':0,'positive_proposal_over_1e_4':0,'positive_proposal_half_canceled':0,
                     'delta_plus_ge_7_9':0,'delta_minus_below_0_1':0}
    @staticmethod
    def edges(key):
        bound=8. if key=='delta_plus' else 4. if key=='delta_minus' else 1.
        return HIST_EDGES[bound]
    def add(self,values,mask):
        n=int(mask.sum())
        if not n:return
        self.n+=n
        for key in FIELDS:
            v=values[key][mask].astype(np.float64)
            if not np.isfinite(v).all():raise ValueError('Nonfinite diagnostic '+key)
            self.sums[key]+=float(v.sum())
            hist=np.histogram(v,bins=self.edges(key))[0]
            if int(hist.sum())!=n:raise ValueError('Diagnostic outside expected range: '+key)
            self.hist[key]+=hist
        self.counts['B_above_U_by_0_1']+=int((mask&(values['B_minus_U']>.1)).sum())
        self.counts['B_above_P0_by_0_1']+=int((mask&(values['B_minus_P0']>.1)).sum())
        positive=mask&(values['proposal_delta']>0)
        self.counts['positive_proposal']+=int(positive.sum())
        self.counts['negative_proposal']+=int((mask&(values['proposal_delta']<0)).sum())
        self.counts['delta_plus_ge_7_9']+=int((mask&(values['delta_plus']>=7.9)).sum())
        self.counts['delta_minus_below_0_1']+=int((mask&(values['delta_minus']<.1)).sum())
        self.counts['positive_proposal_over_1e_4']+=int((mask&(values['proposal_delta']>1e-4)).sum())
        self.counts['positive_proposal_half_canceled']+=int((positive&(
            values['positive_proposal_canceled']>=.5*values['positive_proposal'])).sum())
    def result(self):
        row={'cells':self.n}
        for key in FIELDS:
            row[key+'_mean']=self.sums[key]/self.n if self.n else None
            hist=self.hist[key];edges=self.edges(key)
            for fraction in (.1,.5,.9):
                if self.n:
                    index=min(int(np.searchsorted(np.cumsum(hist),fraction*self.n,side='left')),len(hist)-1)
                    value=float((edges[index]+edges[index+1])*.5)
                else:value=None
                row[f'{key}_p{int(fraction*100)}_approx']=value
        for key,value in self.counts.items():row[key+'_fraction']=value/self.n if self.n else None
        row['positive_proposal_half_canceled_fraction_of_positive']=(
            self.counts['positive_proposal_half_canceled']/self.counts['positive_proposal']
            if self.counts['positive_proposal'] else None)
        row['positive_proposal_canceled_amount_ratio']=(
            self.sums['positive_proposal_canceled']/self.sums['positive_proposal']
            if self.sums['positive_proposal']>0 else None)
        return row
    def add_stats(self,other):
        self.n+=other.n
        for key in FIELDS:self.sums[key]+=other.sums[key];self.hist[key]+=other.hist[key]
        for key in self.counts:self.counts[key]+=other.counts[key]
    def dump(self):
        return {'n':self.n,'sums':self.sums,'counts':self.counts,
                'hist':{key:value.tolist() for key,value in self.hist.items()}}
    @classmethod
    def load(cls,data):
        value=cls();value.n=data['n'];value.sums=data['sums'];value.counts=data['counts']
        value.hist={key:np.asarray(array,np.int64) for key,array in data['hist'].items()}
        return value


def pool_max(array):
    return F.max_pool2d(torch.from_numpy(array.astype(np.float32))[None,None],4,4)[0,0].numpy()>0


def work_regions(gt,skel,p0,p1,q,reference_mask,owner,threshold):
    """Native GT -> work cells; overlapping semantic regions are intentional."""
    road=gt.astype(np.uint8);background=(~gt).astype(np.uint8)
    inner=cv2.distanceTransform(road,cv2.DIST_L2,5)
    outer=cv2.distanceTransform(background,cv2.DIST_L2,5)
    predicted=p0>=.45
    near_p0=cv2.distanceTransform((~predicted).astype(np.uint8),cv2.DIST_L2,5)
    missed=skel & (near_p0>2.)
    b_fraction=F.avg_pool2d(torch.from_numpy(background.astype(np.float32))[None,None],4,4)[0,0].numpy()
    road_fraction=1.-b_fraction
    hard_background=(b_fraction>=.95)&pool_max(outer>3.)
    near=hard_background&pool_max(near_p0<=32.)
    newly_added_fp=(p0<threshold)&(p1>=threshold)&~gt
    return {
        'fixed_reference_gap':pool_max(reference_mask)&owner,
        'current_missed_GT_skeleton':pool_max(missed)&owner,
        'road_edge_GT':(road_fraction>=.5)&pool_max((inner>0)&(inner<=3))&owner,
        'road_interior_GT':(road_fraction>=.95)&~pool_max((inner>0)&(inner<=3))&owner,
        'near_road_background_GT':near&owner,
        'Q_high_background_GT':hard_background&(q>=.5)&owner,
        'new_FP_GT':hard_background&pool_max(newly_added_fp)&owner,
    }


def fields_for_step(result,rates,step):
    p0=result['u0'];u=result['states'][step];b=result['b'];a=result['a']
    proposal=result['proposal_delta'][:,step];transport=result['transport_delta'][:,step];anchor=result['anchor_delta'][:,step]
    torch.testing.assert_close(result['beta'][:,step],.2*a*rates[step][:,2:3],rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(proposal,result['beta'][:,step]*(b-u),rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(result['states'][step+1],u+proposal+transport+anchor,rtol=1e-5,atol=1e-6)
    pos=proposal.clamp_min(0);opposing=(-(transport+anchor)).clamp_min(0)
    tensors={'P0':p0,'B':b,'B_minus_P0':b-p0,'U':u,'B_minus_U':b-u,'A':a,
             'rB':rates[step][:,2:3],'proposal_coefficient':result['beta'][:,step],
             'proposal_delta':proposal,'transport_delta':transport,'anchor_delta':anchor,
             'net_delta':proposal+transport+anchor,'proposal_delta_abs':proposal.abs(),
             'transport_delta_abs':transport.abs(),'anchor_delta_abs':anchor.abs(),
             'net_delta_abs':(proposal+transport+anchor).abs(),'positive_proposal':pos,
             'opposing_transport_anchor':opposing,'positive_proposal_canceled':torch.minimum(pos,opposing),
             'positive_support':result['positive_support'],'delta_plus':result['delta_z_plus'],
             'delta_minus':result['delta_z_minus']}
    block=torch.cat([tensors[k] for k in FIELDS],1)[0].detach().cpu().numpy()
    return {k:block[i] for i,k in enumerate(FIELDS)}


def fixed_gap_map(entry,shape):
    result=np.zeros(shape,np.int32);flat=result.reshape(-1)
    for gap in entry['gaps']:
        indices=np.asarray(gap['pixels'],np.int64)
        if np.any(flat[indices]!=0):raise ValueError('Fixed gap pixels overlap')
        flat[indices]=int(gap['id'])
    return result


def owner_mask(top,left,h=1500,w=1500):
    # Same 512/256 owner cores as the earlier feature probe: disjoint in native coordinates.
    mapping={0:(0,384),256:(384,640),512:(640,896),768:(896,1134),988:(1134,1500)}
    if top not in mapping or left not in mapping or (h,w)!=(1500,1500):raise ValueError('Unexpected tile layout')
    y0,y1=mapping[top];x0,x1=mapping[left]
    y=top+np.arange(128)*4+2;x=left+np.arange(128)*4+2
    return (y[:,None]>=y0)&(y[:,None]<y1)&(x[None,:]>=x0)&(x[None,:]<x1)


@torch.inference_mode()
def inspect_case(model,sample,entry,threshold,device):
    image=sample['image'][None].to(device);gt=sample['mask'][0].numpy()>.5;skel=sample['skeleton'][0].numpy()>.5
    if gt.shape!=(1500,1500):raise ValueError('Expected 1500x1500 validation images')
    ref=fixed_gap_map(entry,gt.shape);weight=blend_weight(512,device)
    canvas=torch.zeros((2,1,1500,1500),dtype=torch.float32,device=device)
    divisor=torch.zeros((1,1,1500,1500),dtype=torch.float32,device=device)
    stats={(region,step):Stats() for region in REGIONS for step in range(3)}
    per_gap={};ambiguous_cells=0;positions=[(y,x) for y in sliding_positions(1500,512,256) for x in sliding_positions(1500,512,256)]
    for index,(top,left) in enumerate(positions):
        rates=[]
        handle=model.original.dynamics.rates.register_forward_hook(lambda module,inputs,output:rates.append(output.softmax(1)))
        try:result=model(image[:,:,top:top+512,left:left+512])
        finally:handle.remove()
        if len(rates)!=3:raise RuntimeError('Expected exactly three live dynamics rate calls')
        canvas[:,:,top:top+512,left:left+512]+=torch.cat((result['z0'],result['z1']),0).float()*weight
        divisor[:,:,top:top+512,left:left+512]+=weight
        p0=result['p0'][0,0].cpu().numpy();p1=result['p1'][0,0].cpu().numpy();q=result['q'][0,0].cpu().numpy()
        reference_crop=ref[top:top+512,left:left+512];owner=owner_mask(top,left)
        region=work_regions(gt[top:top+512,left:left+512],skel[top:top+512,left:left+512],p0,p1,q,
                            reference_crop>0,owner,threshold)
        # Map native missing skeleton pixels to work cells while preserving the
        # fixed gap ID; multiple native pixels within one cell are counted once.
        yy,xx=np.nonzero(reference_crop);cell_ids={}
        for y,x in zip(yy.tolist(),xx.tolist()):
            gy,gx=y//4,x//4
            if owner[gy,gx]:cell_ids.setdefault((gy,gx),set()).add(int(reference_crop[y,x]))
        gap_cells={}
        for cell,ids in cell_ids.items():
            if len(ids)>1:ambiguous_cells+=1;continue
            gap_cells.setdefault(next(iter(ids)),set()).add(cell)
        for step in range(3):
            values=fields_for_step(result,rates,step)
            for name in REGIONS:stats[(name,step)].add(values,region[name])
            for gap,cells in gap_cells.items():
                mask=np.zeros((128,128),bool)
                coords=np.asarray(list(cells),np.int64);mask[coords[:,0],coords[:,1]]=True
                per_gap.setdefault((gap,step),Stats()).add(values,mask)
        if index==0 or (index+1)%5==0 or index+1==len(positions):
            print(f'[TILE] {sample["case_name"]} {index+1}/{len(positions)}',flush=True)
    predictions=(canvas/divisor).sigmoid().cpu().numpy()[:,0]
    if not np.isfinite(predictions).all():raise ValueError('Nonfinite stitched predictions')
    threshold_mask=predictions[1]>=threshold;before=predictions[0]>=threshold
    added=threshold_mask&~before;removed=before&~threshold_mask
    road=gt.astype(np.uint8);inner=cv2.distanceTransform(road,cv2.DIST_L2,5)
    edge=gt&(inner<=3);interior=gt&(inner>3)
    change={'pixels':gt.size,'ambiguous_gap_cells_excluded':ambiguous_cells,
            'p1_minus_p0_mean':float((predictions[1]-predictions[0]).mean()),
            'p1_minus_p0_abs_mean':float(np.abs(predictions[1]-predictions[0]).mean()),
            'added_tp':int((added&gt).sum()),'added_fp':int((added&~gt).sum()),
            'lost_tp':int((removed&gt).sum()),'removed_fp':int((removed&~gt).sum()),
            'added_gt_edge':int((added&edge).sum()),'added_gt_interior':int((added&interior).sum()),
            'added_gt_background':int((added&~gt).sum())}
    return predictions,stats,per_gap,change


def rows_for_case(case,stats,per_gap,entry,predictions,threshold):
    region_rows=[];gap_rows=[]
    for (region,step),value in stats.items():
        region_rows.append({'case':case,'region':region,'step':step+1,**value.result()})
    masks=[predictions[i]>=threshold for i in (0,1)]
    labels=[component_data(mask)[1] for mask in masks]
    for gap in entry['gaps']:
        initial=gap_state(masks[0],labels[0],gap,tuple(entry['shape']))
        final=gap_state(masks[1],labels[1],gap,tuple(entry['shape']))
        for step in range(3):
            value=per_gap.get((int(gap['id']),step),Stats())
            gap_rows.append({'case':case,'gap_id':int(gap['id']),'length_bin':gap.get('length_bin','unknown'),
                'step':step+1,'p0_coverage':initial['coverage'],'p1_coverage':final['coverage'],
                'p0_local_closed':initial['local_closed_proxy'],'p1_local_closed':final['local_closed_proxy'],
                'newly_closed':bool(final['local_closed_proxy'] and not initial['local_closed_proxy']),
                'lost_closure':bool(initial['local_closed_proxy'] and not final['local_closed_proxy']),**value.result()})
    return region_rows,gap_rows


def write_outputs(output,identity,items,complete):
    global_stats={(name,step):Stats() for name in REGIONS for step in range(3)}
    images=[];per_image=[];gaps=[];checks=[]
    for item in items:
        images.append({'case':item['case'],**item['change']})
        per_image.extend(item['region_rows']);gaps.extend(item['gap_rows']);checks.append(item['saved_comparison'])
        for key,value in item['region_stats'].items():
            name,step=key.rsplit('|',1);global_stats[(name,int(step))].add_stats(Stats.load(value))
    global_rows=[{'region':name,'step':step+1,**value.result()} for (name,step),value in global_stats.items()]
    def csvfile(name,rows,defaults):write_csv(output/name,rows,list(rows[0]) if rows else defaults)
    csvfile('region_summary.csv',global_rows,['region','step'])
    csvfile('per_image_regions.csv',per_image,['case','region','step'])
    csvfile('fixed_gap_steps.csv',gaps,['case','gap_id','step'])
    csvfile('full_resolution_changes.csv',images,['case'])
    summary={'complete':complete,'identity':identity,'processed_images':len(items),
             'saved_comparisons_available':sum(v['available'] for v in checks),
             'regions':global_rows,'full_resolution_changes':images,
             'limits':['Validation only; checkpoint weights and model forward are unchanged.',
                       'Work-grid statistics use each 512 tile once at disjoint ownership cores; they are not stitched probabilities.',
                       'Current GT missed skeleton uses each tile P0 at 0.45 and 2px coverage tolerance; fixed gaps use the earlier frozen H0 reference.',
                       'A near-road GT background or high-Q background is a proxy; neither is a manually verified false connection.',
                       'new_FP work cells contain at least one native GT-background pixel newly added between P0 and P1 at the selected threshold; it is not a bridge label.',
                       'Cancellation is min(positive proposal, negative net transport+anchor) per cell; this describes coincident signed changes and is not a counterfactual.',
                       'Region masks overlap and work-grid quantiles are approximate log-spaced histograms; counts and means are exact for included work cells.',
                       'Fixed gap closure remains the existing 80%-coverage plus local-anchor proxy, not verified true connectivity.',
                       'A small observed effect cannot by itself isolate why A, rB or B has that value.']}
    atomic_json(output/'summary.json',summary)
    lines=['# Unscaled proposal: fixed-checkpoint dynamics diagnosis','',
           f'Complete: {complete}; processed {len(items)}/{identity["expected_images"]} images.',
           f'Checkpoint epoch: {identity["checkpoint_epoch"]}; selected threshold: {identity["threshold"]}.',
           f'Saved P0/P1 comparisons available for {summary["saved_comparisons_available"]}/{len(items)} images.', '',
           '| Region | Step | Cells | P0 | B | U | B−P0 | B−U | A | rB | 0.2 A rB | Proposal Δ | Transport Δ | Anchor Δ | Net Δ |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'n/a' if x is None else f'{x:.7g}'
    for r in global_rows:
        lines.append(f'| {r["region"]} | {r["step"]} | {r["cells"]} | {fmt(r["P0_mean"])} | {fmt(r["B_mean"])} | {fmt(r["U_mean"])} | {fmt(r["B_minus_P0_mean"])} | {fmt(r["B_minus_U_mean"])} | {fmt(r["A_mean"])} | {fmt(r["rB_mean"])} | {fmt(r["proposal_coefficient_mean"])} | {fmt(r["proposal_delta_mean"])} | {fmt(r["transport_delta_mean"])} | {fmt(r["anchor_delta_mean"])} | {fmt(r["net_delta_mean"])} |')
    lines+=['','See region_summary.csv for quantiles, positive-proposal fractions and cancellation ratios.',
            'See fixed_gap_steps.csv for each reference gap and full_resolution_changes.csv for GT edge/interior/background changes.','','## Limits','']+['- '+line for line in summary['limits']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair_dir',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--run_name',default='B_unscaled_same_initial_20e');p.add_argument('--root_path',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--threshold',type=float)
    p.add_argument('--expected_images',type=int,default=14);p.add_argument('--expected_gaps',type=int,default=198)
    args=p.parse_args()
    if Path(args.run_name).name!=args.run_name or not args.run_name.startswith('B_'):p.error('Invalid run name')
    pair=args.pair_dir.resolve();run=pair/args.run_name;ckpt_path=run/'best.pth'
    manifest_path=pair/'reference/fixed_gaps_and_anchors.json';manifest=read_json(manifest_path)
    if manifest['count']!=args.expected_gaps or len(manifest['images'])!=args.expected_images:
        raise ValueError('Fixed reference counts do not match expected validation')
    ckpt_sha=digest(ckpt_path);checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=False);config=checkpoint['config']
    reference_sha=digest(manifest_path)
    if config['reference_sha256']!=reference_sha:raise ValueError('Checkpoint/reference mismatch')
    verify_recorded_code(config);verify_adaptation_code(config)
    selection=read_json(run/'eval/val_selection.json')
    if selection['checkpoint_sha256']!=ckpt_sha or selection['reference_sha256']!=reference_sha:
        raise ValueError('Saved validation selection does not match checkpoint or fixed reference')
    threshold=float(selection['threshold'] if args.threshold is None else args.threshold)
    if not 0<threshold<1:p.error('threshold must be between 0 and 1')
    root=(args.root_path or Path(config['root_path'])).resolve()
    identity={'version':'unscaled_proposal_effect_v1','checkpoint_sha256':ckpt_sha,'checkpoint_epoch':checkpoint['epoch'],
              'reference_sha256':reference_sha,'root_path':str(root),'run_name':args.run_name,
              'threshold':threshold,'threshold_source':'saved validation selection' if args.threshold is None else 'explicit argument',
              'expected_images':args.expected_images,'expected_gaps':args.expected_gaps,
              'diagnostic_code_sha256':digest(Path(__file__)),'device_type':torch.device(args.device).type,
              'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),'opencv_version':str(cv2.__version__)}
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True);id_path=output/'identity.json'
    if id_path.is_file() and read_json(id_path)!=identity:raise ValueError('Existing output belongs to another diagnostic')
    if not id_path.is_file() and any(output.iterdir()):raise ValueError('Choose an empty output directory')
    atomic_json(id_path,identity);cache=output/'cases';cache.mkdir(exist_ok=True)
    seed_all(int(config['seed']));model=build_unscaled(checkpoint,torch.device(args.device));del checkpoint
    dataset=full_dataset(root,'val')
    if len(dataset)!=args.expected_images:raise ValueError('Unexpected validation image count')
    items=[];seen=set()
    for index in range(len(dataset)):
        sample=dataset[index];name=str(sample['case_name']);gt=sample['mask'][0].numpy()>.5
        if name in seen or name not in manifest['images']:raise ValueError('Validation case differs from fixed reference')
        seen.add(name);entry=manifest['images'][name]
        if list(gt.shape)!=entry['shape']:raise ValueError('Validation shape differs from fixed reference')
        ref_path=pair/'reference'/(name+'.npz')
        if not ref_path.is_file():ref_path=run/'eval/val'/(name+'.npz')
        with np.load(ref_path) as f:
            if not np.array_equal(f['gt'].astype(bool),gt):raise ValueError('GT differs from fixed reference: '+name)
        path=cache/(name+'.json');sample_sha=signature(sample)
        if path.is_file():
            item=read_json(path)
            if item['input_sha256']!=sample_sha:raise ValueError('Cached case input changed: '+name)
            print(f'[CACHE] {index+1}/{len(dataset)} {name}',flush=True)
        else:
            print(f'[CASE] {index+1}/{len(dataset)} {name}',flush=True)
            predictions,stats,gaps,change=inspect_case(model,sample,entry,threshold,torch.device(args.device))
            saved=validate_saved_prediction(pair,args.run_name,name,gt,{'p0':predictions[0],'p1':predictions[1]})
            region_rows,gap_rows=rows_for_case(name,stats,gaps,entry,predictions,threshold)
            item={'case':name,'input_sha256':sample_sha,'change':change,'saved_comparison':saved,
                  'region_stats':{name+'|'+str(step):value.dump() for (name,step),value in stats.items()},
                  'region_rows':region_rows,'gap_rows':gap_rows}
            atomic_json(path,item)
        items.append(item);write_outputs(output,identity,items,False)
    if seen!=set(manifest['images']):raise ValueError('Validation image set differs from reference')
    if sum(len(item['gap_rows']) for item in items)!=args.expected_gaps*3:
        raise ValueError('Fixed gap count mismatch')
    write_outputs(output,identity,items,True)
    print('[DONE] '+str(output/'REPORT.md'),flush=True)


if __name__=='__main__':main()
