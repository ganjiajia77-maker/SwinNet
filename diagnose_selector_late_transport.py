"""Val-only fixed-checkpoint late-transport contrasts. No training/optimizer."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import run_reliable_v9_mass as v9
from run_v9_gate_benefit import atomic_json
from networks.reliable_directional_repair import compose_surface_logits
from networks.v9_source_selector import SourceSelectedRepair
from networks.v9_source_relation_labels import RelationLabels,native_relations
from networks.v9_selector_transport_training import VERSION,SplitDirectionHead,load_canonical
from selector_transport_provenance import read_json,check_code_hashes
from selector_transport_reference import prepared,closed_gaps,bridge_proxy
from mass_protocol import blend_weight,sliding_positions
from diagnostics.v91_counterfactual import write_csv

MODES=('full_three','stop_late','oracle_late')


def transfer_to_stay(choice,blocked):
    if blocked.shape!=choice[:,1:].shape: raise ValueError('Blocked-source mask shape differs from eight choices')
    sources=choice[:,1:].masked_fill(blocked,0.)
    removed=(choice[:,1:]-sources).sum(1,keepdim=True)
    return torch.cat((choice[:,:1]+removed,sources),1),removed


class LateIntervention:
    def __init__(self,model,mode,labels=None):
        if mode not in MODES: raise ValueError('Unknown late-transport mode')
        if mode=='oracle_late' and labels is None: raise ValueError('Oracle diagnostics require native GT relation labels')
        self.model=model; self.mode=mode; self.labels=labels; self.records=[]

    def hook(self,unused,inputs,output):
        k=len(self.records); raw=output['choice']; used=raw; removed=torch.zeros_like(raw[:,:1])
        if k>=1 and self.mode!='full_three':
            blocked=(torch.ones_like(output['valid']) if self.mode=='stop_late' else
                     (self.labels==0)&output['valid'])
            used,removed=transfer_to_stay(raw,blocked)
        torch.testing.assert_close(used.sum(1),torch.ones_like(used[:,0]),atol=2e-6,rtol=1e-6)
        state=inputs[1]; rate=inputs[4]
        removed_update=rate*(raw[:,1:]-used[:,1:])*(output['source_states']-state)
        self.records.append({'raw_choice':raw.detach(),'actual_choice':used.detach(),
            'removed_mass':removed.detach(),'removed_positive':removed_update.clamp_min(0.).sum(1,keepdim=True).detach(),
            'removed_negative_abs':(-removed_update).clamp_min(0.).sum(1,keepdim=True).detach()})
        return dict(output,choice=used)

    def __enter__(self):
        if self.model.training: raise ValueError('Read-only contrast requires eval mode')
        self.handle=self.model.selector.register_forward_hook(self.hook); return self

    def __exit__(self,kind,value,traceback):
        self.handle.remove()
        if kind is None and len(self.records)!=3: raise RuntimeError('Expected exactly three selector calls')


@torch.no_grad()
def tile_contrasts(model,image,mask,skeleton,labels_config):
    frozen=model.original.frozen(image,capture_diagnostics=True)
    with LateIntervention(model,'full_three') as normal:
        baseline=model(None,mode='source',frozen_override=frozen)
    native=native_relations(mask,skeleton,baseline['r'].shape[-2:],labels_config)
    variants={'full_three':baseline}; records={'full_three':normal.records}
    for mode in MODES[1:]:
        with LateIntervention(model,mode,native['labels']) as intervention:
            result=model(None,mode='source',frozen_override=frozen)
        if not torch.equal(result['u1'],baseline['u1']): raise RuntimeError('Intervention changed the first-step state')
        if not torch.equal(result['source_choice'][:,0],baseline['source_choice'][:,0]): raise RuntimeError('First-step choice changed')
        if mode=='stop_late' and bool((result['transport_delta'][:,1:]!=0).any()):
            raise RuntimeError('Late stop failed to zero ONLY transport')
        for k in range(3):
            state=result['states'][k]
            torch.testing.assert_close(result['proposal_delta'][:,k],result['beta'][:,k]*(result['b']-state))
            torch.testing.assert_close(result['anchor_delta'][:,k],result['gamma'][:,k]*(result['u0']-state))
        variants[mode]=result; records[mode]=intervention.records
    return variants,records,native['labels']


def load_current(args):
    path=args.checkpoint or args.run_dir/'best.pth'
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version')!=VERSION: raise ValueError(f'Expected joint actual-transport checkpoint {VERSION}; got {state.get("version")}')
    config=state['config']; check_code_hashes(config)
    expected_identity=state['identity']
    original,_,identity=v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(expected_identity['q3']['path']),device=args.device))
    if identity!=expected_identity: raise ValueError('Frozen BASE/Q3/core identity differs from current checkpoint')
    hidden=state['selector']['edge.0.weight'].shape[0]
    model=SourceSelectedRepair(original,hidden).to(args.device)
    model.original.relations.direction_head=SplitDirectionHead(model.original.relations.direction_head)
    if set(state['modules'])!=set(v9.MODULE_NAMES): raise ValueError('Incomplete current V9 modules')
    for name in v9.MODULE_NAMES: load_canonical(getattr(model.original,name),state['modules'][name])
    model.original.gate_relation.load_state_dict(state['gate_relation'],strict=True)
    model.original.state_gate.load_state_dict(state['state_gate'],strict=True)
    model.selector.load_state_dict(state['selector'],strict=True)
    model.requires_grad_(False).eval()
    manifest=read_json(args.run_dir/'fixed_gaps_and_anchors.json')
    binding=read_json(args.run_dir/'fixed_reference_binding.json')
    if binding['manifest_sha256']!=v9.digest(args.run_dir/'fixed_gaps_and_anchors.json'):
        raise ValueError('Fixed gap/anchor list has changed')
    if manifest['count']!=206 or manifest['p0_reference_threshold']!=.45: raise ValueError('Wrong fixed reference protocol')
    if args.threshold is None:
        completed=read_json(args.run_dir/'training_complete.json')
        if completed['epochs']!=config['epochs'] or abs(completed['best_score']-state['score'])>1e-8:
            raise ValueError('Current checkpoint is not the completed selected best; supply an explicit fixed threshold')
        if v9.digest(path)!=v9.digest(args.run_dir/'best.pth'):
            raise ValueError('Different checkpoint requires an explicit common threshold')
        comparison=read_json(args.run_dir/'comparison.json')
        threshold=float(comparison['reports']['S_selector']['own_val_threshold']['threshold'])
        threshold_origin=str(args.run_dir/'comparison.json')+' : S_selector own_val_threshold'
    else: threshold=float(args.threshold); threshold_origin='explicit common diagnostic threshold; no threshold search'
    return model,state,path,manifest,threshold,threshold_origin


def probability_maps(height,width,device):
    return torch.zeros((1+3*3,1,height,width),device=device),torch.zeros((1,1,height,width),device=device)


def trace_file(path,variants,records,labels,origin):
    values={'native_tile_origin':np.asarray(origin),'oracle_relation_labels':labels[0].cpu().numpy()}
    for mode,result in variants.items():
        values[mode+'_U0_U3']=torch.stack(result['states'])[:,0,0].cpu().numpy()
        for name in ('transport_delta','proposal_delta','anchor_delta','transport_rate','source_available','source_choice'):
            values[mode+'_'+name]=result[name][0].cpu().numpy()
        values[mode+'_raw_choice']=torch.stack([row['raw_choice'][0] for row in records[mode]]).cpu().numpy()
    np.savez_compressed(path,**values)


def case_metrics(case,predictions,threshold,entry):
    gt=case['gt']; p0=case['p0']>=.45; previous0=case['p0']>=threshold; rows=[]
    initial_bins,initial_ids=closed_gaps(previous0,entry)
    def row(mode,step,pred,previous,old_ids):
        tp=int((pred & gt).sum()); fp=int((pred & ~gt).sum()); fn=int((~pred & gt).sum())
        bins,ids=closed_gaps(pred,entry)
        return {'case':case['case'],'mode':mode,'step':step,'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,
            'iou':tp/max(tp+fp+fn,1),'bridge_proxy':bridge_proxy(p0,pred,gt),'closed':len(ids),
            'reference_gaps':len(entry['gaps']),'new_fp_vs_previous':int((pred&~previous&~gt).sum()),
            'removed_fp_vs_previous':int((~pred&previous&~gt).sum()),
            'recovered_gt_vs_previous':int((pred & ~previous & gt).sum()),
            'new_fn_vs_previous':int((~pred & previous & gt).sum()),
            'newly_closed':len(set(ids)-set(old_ids)),'lost_closures':len(set(old_ids)-set(ids)),
            'closed_ids':ids,'gap_by_length':bins}
    rows.append(row('P0',0,previous0,previous0,initial_ids))
    for mode in MODES:
        previous=previous0; old_ids=initial_ids
        for k in range(3):
            pred=predictions[mode][k]>=threshold
            value=row(mode,k+1,pred,previous,old_ids); rows.append(value)
            previous=pred; old_ids=value['closed_ids']
    return rows


def error_montage(path,gt,predictions,threshold):
    panels=[]
    for mode in MODES:
        images=[]
        for k in range(3):
            pred=predictions[mode][k]>=threshold; colour=np.zeros((*gt.shape,3),np.uint8)
            colour[pred & gt]=(0,150,0); colour[pred & ~gt]=(0,0,255); colour[~pred & gt]=(255,0,0)
            colour=cv2.resize(colour,(512,512),interpolation=cv2.INTER_NEAREST)
            cv2.putText(colour,f'{mode} step{k+1}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            images.append(colour)
        panels.append(np.concatenate(images,1))
    cv2.imwrite(str(path),np.concatenate(panels,0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir',type=Path,required=True); parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path); parser.add_argument('--threshold',type=float)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--trace_images',type=int,default=4)
    args=parser.parse_args(); args.device=torch.device(args.device)
    if args.threshold is not None and not 0<args.threshold<1: parser.error('Threshold must be in(0,1)')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model,state,checkpoint,manifest,threshold,threshold_origin=load_current(args)
    expected={'version':'selector_late_transport_readonly_v1','checkpoint_sha256':v9.digest(checkpoint),
        'checkpoint_epoch':state['epoch'],'threshold':threshold,'threshold_origin':threshold_origin,
        'reference_sha256':manifest['sha256'],'manifest_sha256':v9.digest(args.run_dir/'fixed_gaps_and_anchors.json'),
        'native_labels':state['config']['labels'],'GT_oracle':'oracle_late only, step2/3, native labels==0; unknown=-1 unchanged',
        'code_sha256':v9.digest(Path(__file__)),
        'native_label_code_sha256':v9.digest(Path(__file__).parent/'networks/v9_source_relation_labels.py'),
        'source_selector_code_sha256':v9.digest(Path(__file__).parent/'networks/v9_source_selector.py')}
    marker=args.output_dir/'identity.json'
    if marker.is_file() and read_json(marker)!=expected: raise ValueError('Diagnostic output identity changed; use a new output directory')
    atomic_json(marker,expected)
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(Path(state['config']['root_path']),'val')
    if len(dataset)!=14: raise ValueError('Expected the same14 validation images')
    labels_config=RelationLabels(**state['config']['labels']); per_image=[]; removed=[]; fingerprint=hashlib.sha256()
    for index in range(len(dataset)):
        sample=dataset[index]; name=str(sample['case_name']); gt=sample['mask'][0].numpy()>.5; height,width=gt.shape
        if name not in manifest['images']: raise ValueError('Validation case differs from frozen reference')
        prefix=f'{index:02d}'; file=args.output_dir/(prefix+'_steps.npz'); report_file=args.output_dir/(prefix+'_metrics.json')
        input_digest=hashlib.sha256(sample['image'].contiguous().numpy().tobytes()+sample['mask'].contiguous().numpy().tobytes()).hexdigest()
        if file.is_file() and report_file.is_file():
            cache=read_json(report_file)
            if cache['case']!=name or cache['input_sha256']!=input_digest: raise ValueError('Cached validation inputs changed')
            with np.load(file,allow_pickle=False) as cache_arrays: p0=cache_arrays['p0']
            per_image.extend(cache['metrics']); removed.extend(cache['removed'])
        else:
            image=sample['image'][None].to(args.device); canvas,divisor=probability_maps(height,width,args.device)
            weight=blend_weight(512,args.device); own=np.zeros((height,width),bool); reductions=[]
            gap_mask=np.zeros(height*width,bool)
            for gap in manifest['images'][name]['gaps']: gap_mask[gap['pixels']]=True
            gap_mask=gap_mask.reshape(height,width)
            positions=[(y,x) for y in sliding_positions(height,512,256) for x in sliding_positions(width,512,256)]
            trace_position=max(positions,key=lambda pos:int(gap_mask[pos[0]:pos[0]+512,pos[1]:pos[1]+512].sum()))
            for y in sliding_positions(height,512,256):
                for x in sliding_positions(width,512,256):
                    mask=sample['mask'][None,:,y:y+512,x:x+512].to(args.device)
                    skeleton=sample['skeleton'][None,:,y:y+512,x:x+512].to(args.device)
                    variants,records,labels=tile_contrasts(model,image[:,:,y:y+512,x:x+512],mask,skeleton,labels_config)
                    logits=[variants['full_three']['z0']]
                    h,w=labels.shape[-2:]; yy=np.minimum(((np.arange(h)+.5)*512/h).astype(int),511); xx=np.minimum(((np.arange(w)+.5)*512/w).astype(int),511)
                    ownership=(~own[y:y+512,x:x+512])[yy[:,None],xx[None,:]]; own[y:y+512,x:x+512]=True
                    ownership=torch.as_tensor(ownership,device=args.device)[None,None]
                    for mode in MODES:
                        result=variants[mode]
                        for k in range(1,4):
                            logits.append(result['z1'] if k==3 else compose_surface_logits(result['z0'],result['u0'],result['states'][k])[0])
                            record=records[mode][k-1]
                            reductions.append({'case':name,'mode':mode,'step':k,'pixels':int(ownership.sum()),
                                **{field:float(record[field][ownership].sum()) for field in ('removed_mass','removed_positive','removed_negative_abs')}})
                    canvas[:,:,y:y+512,x:x+512]+=torch.cat(logits,0).float()*weight; divisor[:,:,y:y+512,x:x+512]+=weight
                    if index<args.trace_images and (y,x)==trace_position:
                        trace_file(args.output_dir/(prefix+'_tile_states.npz'),variants,records,labels,(y,x))
            logits=(canvas/divisor).cpu().numpy()[:,0]; probability=1./(1.+np.exp(-np.clip(logits,-80.,80.)))
            if not np.isfinite(probability).all(): raise RuntimeError('Nonfinite step predictions')
            p0=probability[0]; predictions={mode:probability[1+3*d:1+3*(d+1)] for d,mode in enumerate(MODES)}
            case={'case':name,'gt':gt,'p0':p0}
            metrics=case_metrics(case,predictions,threshold,manifest['images'][name])
            arrays={'p0':p0,'gt':gt,'step1':predictions['full_three'][0]}
            for mode in MODES: arrays[mode+'_step2']=predictions[mode][1]; arrays[mode+'_step3']=predictions[mode][2]
            with file.with_suffix('.tmp').open('wb') as handle: np.savez_compressed(handle,**arrays)
            file.with_suffix('.tmp').replace(file)
            atomic_json(report_file,{'case':name,'input_sha256':input_digest,'metrics':metrics,'removed':reductions})
            per_image.extend(metrics); removed.extend(reductions)
            if index<args.trace_images: error_montage(args.output_dir/(prefix+'_FP_FN.png'),gt,predictions,threshold)
        fingerprint.update(name.encode()+b'\0'); fingerprint.update(np.asarray(gt.shape,dtype=np.int64).tobytes())
        for mask in (p0>=.45,gt,prepared({'gt':gt})['skeleton']): fingerprint.update(np.packbits(mask,axis=None).tobytes())
        print(f'[LATE TRANSPORT VAL] {index+1}/14 {name}',flush=True)
    if fingerprint.hexdigest()!=manifest['sha256']: raise ValueError('P0/GT/skeleton fingerprint differs from the frozen reference')
    totals={}
    for row in per_image:
        target=totals.setdefault((row['mode'],row['step']),Counter()); target['images']+=1
        for key in ('tp','fp','fn','bridge_proxy','closed','reference_gaps','new_fp_vs_previous','removed_fp_vs_previous','recovered_gt_vs_previous','new_fn_vs_previous','newly_closed','lost_closures'):
            target[key]+=row[key]
    summary=[{'mode':mode,'step':step,'threshold':threshold,**dict(value),
        'iou':value['tp']/max(value['tp']+value['fp']+value['fn'],1),
        'oracle_GT_used':mode=='oracle_late'} for (mode,step),value in totals.items()]
    write_csv(args.output_dir/'step_comparison.csv',summary)
    write_csv(args.output_dir/'per_image_steps.csv',[{key:value for key,value in row.items() if key not in ('closed_ids','gap_by_length')} for row in per_image])
    reductions={}
    for row in removed:
        item=reductions.setdefault((row['mode'],row['step']),Counter())
        for key in ('pixels','removed_mass','removed_positive','removed_negative_abs'): item[key]+=row[key]
    write_csv(args.output_dir/'removed_transport.csv',[{'mode':mode,'step':step,**dict(value),
        'removed_mass_mean':value['removed_mass']/max(value['pixels'],1)} for (mode,step),value in reductions.items()])
    atomic_json(args.output_dir/'summary.json',{'identity':expected,'metrics':summary,'training':False,
        'limits':'oracle_late is GT-assisted native-relationship proxy diagnosis, not deployable performance or a rigorous global upper bound. Proposal/anchor and future state/rates are recomputed normally.'})
    lines=['# Fixed-checkpoint late transport contrasts','',f'Common threshold: {threshold}; {threshold_origin}.',
        'Same original frozen gaps/anchors. oracle_late uses GT only at steps2/3 and is NOT deployable performance.','',
        '| Mode | Step | FP | FN | IoU | Bridge proxy | Closed | New FP vs prior step | New closures | Lost closures |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["mode"]} | {row["step"]} | {row["fp"]} | {row["fn"]} | {row["iou"]:.6f} | {row["bridge_proxy"]} | {row["closed"]}/{row["reference_gaps"]} | {row["new_fp_vs_previous"]} | {row["newly_closed"]} | {row["lost_closures"]} |')
    lines+=['','stop_late zeros transport only; proposal/anchor remain active. Oracle transfers rejected source mass to stay without renormalising retained sources. '
        'Step1 equality was checked on every tile. Subsequent rates/supports/choices use each contrast\'s own updated states.',
        'Full native step probabilities: *_steps.npz (shared step1, mode_step2/3). Representative raw U0-U3/weights/update terms: *_tile_states.npz. '
        'FP/FN montage: red=FP, blue=FN, green=TP. All closure/bridge numbers remain proxy metrics.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if v9.digest(checkpoint)!=expected['checkpoint_sha256']: raise RuntimeError('Checkpoint changed during diagnostics')
    print(f'[DONE] {args.output_dir / "REPORT.md"}',flush=True)


if __name__=='__main__':
    with torch.no_grad(): main()
