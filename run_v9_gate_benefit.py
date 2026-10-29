"""Paired original-V9 continuations with detached on/off transport labels."""
import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_reliable_v9_mass as v9
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_transport_benefit import BenefitConfig,forward_trace,gate_benefit_loss,BenefitMetrics,BENEFIT_VERSION,restore_model

VERSION=BENEFIT_VERSION


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8'); temp.replace(path)


def load_original(args):
    source=args.run_dir/'stage_D/best.pth'; selected=args.run_dir/'direct_val_selection.json'
    if not source.is_file() or not selected.is_file(): raise FileNotFoundError('Missing original V9 checkpoint or val selection')
    payload=torch.load(source,map_location='cpu',weights_only=False); selection=json.loads(selected.read_text())
    if payload['identity'].get('initialization')!='direct_random_downstream': raise ValueError('Expected direct_random_downstream original V9')
    if payload['identity']!=selection['identity'] or selection.get('selected_on')!='val' or Path(selection['checkpoint']).resolve()!=source.resolve():
        raise ValueError('Original V9 checkpoint and val selection differ')
    config=payload['training_config']
    if Path(config['root_path']).resolve()!=args.root_path.resolve() or config['seed']!=args.seed or config['batch_size']!=args.batch_size:
        raise ValueError('Preserve V9 dataset, seed and batch size')
    if config['reference_threshold']!=.45 or selection['report']['paired']['eligible_p0_gaps']!=206:
        raise ValueError('Requires the fixed .45 P0 / 206 validation reference gaps')
    model,source_losses,identity=v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(payload['identity']['q3']['path']),device=args.device))
    v9.restore_checkpoint(model,source,identity,'D'); model.v9_training_config=dict(config)
    v9.LR.update(payload['learning_rates'])
    for key in ('lambda_bg','lambda_missing','lambda_cont','missing_threshold'): source_losses[key]=config[key]
    if float(source_losses['lambda_cont'])!=.2: raise ValueError('This experiment requires the original .2 L_cont coefficient')
    return model,source_losses,identity,payload,source,selection


def settings(args,identity,source,steps,variant):
    benefit=BenefitConfig(min_action=args.min_action,gain_epsilon=args.gain_epsilon,max_pairs_per_sign=args.max_pairs_per_sign)
    config={'version':VERSION,'variant':variant,'steps':steps,'epochs':args.epochs,'seed':args.seed,
        'batch_size':args.batch_size,'num_workers':args.num_workers,'val_interval':args.val_interval,
        'lambda_G':args.lambda_gate,'labels':asdict(benefit),'identity':identity,
        'original_checkpoint':str(source.resolve()),'original_sha256':v9.digest(source),
        'root_path':str(args.root_path.resolve()),'learning_rates':dict(v9.LR),
        'selection':'original V9 weighted score at .60 for checkpoints; weighted score over val .35:.01:.75 for thresholds',
        'code_sha256':{name:v9.digest(Path(__file__).parent/name) for name in (
            'run_v9_gate_benefit.py','gate_benefit_protocol.py','networks/v9_transport_benefit.py')},
        'protocol':'native1500->random512 one crop/image/epoch, FP32, no EMA, physical batch2 accumulation1; BASE/Q3 frozen'}
    return json.loads(json.dumps(config)),benefit


def save(path,model,identity,epoch,best,optimizer,config,record=None):
    payload={'version':VERSION,'identity':identity,'stage':'D','epoch':epoch,'score':best,
        'modules':{name:getattr(model,name).state_dict() for name in v9.MODULE_NAMES},
        'gate_relation':model.gate_relation.state_dict(),'state_gate':model.state_gate.state_dict(),
        'training_config':model.v9_training_config,'learning_rates':dict(v9.LR),
        'benefit_version':VERSION,'benefit_config':config,'epoch_record':record,'torch_rng':torch.get_rng_state(),
        'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if optimizer is not None: payload['optimizer']=optimizer.state_dict()
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
    torch.save(payload,temp); temp.replace(path)


def make_dataset(args,augment):
    return RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=augment)


def gradient_comparison(base,weighted_gate,model,segmentation=None):
    groups={name:tuple(v9.module(model,name).parameters()) for name in v9.STAGE_MODULES['D']}
    params=[p for group in groups.values() for p in group]
    def measure(loss):
        gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        offset=0; result={}; total=0.
        for name,group in groups.items():
            pieces=[value.detach().double().square().sum() for value in gradients[offset:offset+len(group)] if value is not None]
            squared=float(torch.stack(pieces).sum()) if pieces else 0.
            result[name]=squared**.5; total+=squared; offset+=len(group)
        return {'total_l2':total**.5,'modules':result}
    left=measure(base); right=measure(weighted_gate)
    ratio=right['total_l2']/max(left['total_l2'],1e-12)
    seg=measure(segmentation) if segmentation is not None else None
    seg_ratio=right['total_l2']/max(seg['total_l2'],1e-12) if seg is not None else None
    return {'base_loss':left,'segmentation_loss':seg,'weighted_lambda_G_loss':right,'gate_to_base_l2_ratio':ratio,
            'gate_to_segmentation_l2_ratio':seg_ratio,'review_gate_dominance':ratio>1. or seg_ratio is not None and seg_ratio>1.,
            'note':'Diagnostic gradient norms, not automatic reweighting.'}


def smoke(args,model,loss_config,identity,source):
    snapshot={name:{key:value.detach().clone() for key,value in getattr(model,name).state_dict().items()}
              for name in (*v9.MODULE_NAMES,'gate_relation','state_gate')}
    rng=torch.get_rng_state(); np_rng=np.random.get_state(); cuda_rng=torch.cuda.get_rng_state_all() if args.device.type=='cuda' else None
    ds=make_dataset(args,False); ds.set_epoch(1)
    batch=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))
    optimizer=v9.configure(model,'D'); model.train(); model.set_accept_floor(0.)
    if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
    try:
        optimizer.zero_grad(set_to_none=True)
        image,mask,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
        result=forward_trace(model,image,3)
        # No GT is passed to the forward helper; label construction cannot change states.
        before=[state.detach().clone() for state in result['states']]
        gate,labels,selected,report=gate_benefit_loss(result,mask,loss_config,args.seed)
        if any(not torch.equal(a,b) for a,b in zip(before,result['states'])): raise RuntimeError('GT labels changed model states')
        base,base_terms=dynamic_losses(result,mask,skeleton,**args.source_losses,return_live_terms=True)
        total=base+args.lambda_gate*gate
        if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite smoke loss')
        norms=gradient_comparison(base,args.lambda_gate*gate,model,base-.2*base_terms['cont_live']) if int(selected.sum()) else None
        total.backward()
        if any(p.grad is not None for p in model.frozen.parameters()): raise RuntimeError('Frozen BASE/Q3 received gradients')
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite smoke gradient')
        optimizer.step()
        metrics=BenefitMetrics(3); metrics.add(labels,selected,result)
        output={'passed':True,'identity':identity,'original_sha256':v9.digest(source),
            'GT_did_not_change_states':True,'loss':float(total.detach()),'gate_loss':float(gate.detach()),
            'matching':report,'gradient_comparison':norms,'decisions':metrics.report(),
            'physical_batch':len(image),'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None,
            'warning':'If the real batch has no matched labels, gradient evidence is unavailable; each training epoch is monitored and an empty gate epoch fails explicitly.'}
    finally:
        for name,state in snapshot.items(): getattr(model,name).load_state_dict(state,strict=True)
        model.zero_grad(set_to_none=True); model.eval(); torch.set_rng_state(rng); np.random.set_state(np_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
    atomic_json(args.output_dir/'smoke.json',output)
    print('[SMOKE] '+json.dumps({'passed':True,'matched_allow':output['decisions']['matched_allow'],
        'weighted_gate_to_seg_l2_ratio':norms['gate_to_segmentation_l2_ratio'] if norms else None,
        'review_gate_dominance':norms['review_gate_dominance'] if norms else None,
        'peak_cuda_mib':output['peak_cuda_mib'],'details':str(args.output_dir/'smoke.json')}),flush=True)
    return output


def train_variant(args,model,source,identity,variant,steps):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    from gate_benefit_protocol import StepModel
    # Fresh original V9 state and optimizer for each equal-budget continuation.
    v9.restore_checkpoint(model,source,identity,'D'); model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    config,benefit=settings(args,identity,source,steps,variant)
    out=args.output_dir/f'{variant}_{steps}step'; out.mkdir(parents=True,exist_ok=True)
    marker=out/'config.json'
    if marker.is_file() and json.loads(marker.read_text())!=config: raise ValueError(f'Run configuration changed: {out}')
    atomic_json(marker,config)
    optimizer=v9.configure(model,'D'); begin=1; best=-float('inf')
    if (out/'last.pth').is_file():
        state=restore_model(model,out/'last.pth',identity,steps)
        if state.get('benefit_version')!=VERSION or state.get('benefit_config')!=config: raise ValueError('Wrong resume checkpoint/configuration')
        optimizer.load_state_dict(state['optimizer']); torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        begin=state['epoch']+1; best=state['score']
        if state.get('epoch_record') is not None: store_epoch(out,state['epoch_record'])
    ds=make_dataset(args,True)
    for epoch in range(begin,args.epochs+1):
        ds.set_epoch(epoch)
        # An independent generator makes the ordering/worker seeds invariant to loss computations.
        generator=torch.Generator().manual_seed(args.seed+epoch*1000003)
        loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
            pin_memory=args.device.type=='cuda',generator=generator)
        model.train(); model.set_accept_floor(0.)
        totals=Counter(); matching=Counter(); decisions=BenefitMetrics(steps); gradients=None
        order_digest=hashlib.sha256(); image_digest=hashlib.sha256(); batches=0
        for index,batch in enumerate(loader):
            for case in batch['case_name']: order_digest.update(str(case).encode()+b'\0')
            image_digest.update(batch['image'].contiguous().numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            image,mask,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
            result=forward_trace(model,image,steps)
            base,terms=dynamic_losses(result,mask,skeleton,**args.source_losses,return_live_terms=True)
            gate,labels,selected,report=gate_benefit_loss(result,mask,benefit,args.seed+epoch*100000+index)
            total=base+(args.lambda_gate*gate if variant=='gate' else 0.)
            if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite training loss')
            if gradients is None and int(selected.sum()):
                gradients=gradient_comparison(base,args.lambda_gate*gate,model,base-.2*terms['cont_live'])
                gradients['gate_loss_applied_to_optimizer']=variant=='gate'
            total.backward()
            if any(p.grad is not None for p in model.frozen.parameters()): raise RuntimeError('Frozen BASE/Q3 received gradients')
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite training gradient')
            optimizer.step(); decisions.add(labels,selected,result); batches+=1
            for key,value in report['matching'].items():
                if value is not None:
                    if key.endswith('_mean'): matching[key+'_weighted_sum']+=value*report['matching']['pairs']
                    else: matching[key]+=value
            totals['total']+=float(total.detach()); totals['seg']+=float(terms['seg']); totals['cont']+=float(terms['cont']); totals['gate']+=float(gate.detach())
            if index%25==0: print(f'[GATE BENEFIT TRAIN] {variant} steps={steps} epoch={epoch}/{args.epochs} batch={index+1}/{len(loader)} matched_pairs={report["matching"]["pairs"]} loss={float(total.detach()):.5f}',flush=True)
        for key in ('matching_action_abs_difference_mean','matching_V_abs_difference_mean'):
            matching[key]=matching[key+'_weighted_sum']/matching['pairs'] if matching['pairs'] else None
        record={'epoch':epoch,'variant':variant,'steps':steps,'batches':batches,
            'losses':{key:value/max(batches,1) for key,value in totals.items()},
            'case_order_sha256':order_digest.hexdigest(),'augmented_images_sha256':image_digest.hexdigest(),
            'decisions':decisions.report(),'matching_counts':dict(matching),'gradient_comparison_first_matched_batch':gradients}
        if variant=='gate' and record['decisions']['matched_allow']==0:
            atomic_json(out/'failed_no_matched_labels.json',record)
            save(out/'failed_no_matched_labels.pth',model,identity,epoch,best,optimizer,config)
            raise RuntimeError('No matched gate labels in this epoch. Diagnostic failure saved; epoch is not marked successful.')
        if epoch%args.val_interval==0 or epoch==args.epochs:
            values,_,_,_=evaluate_thresholds(StepModel(model,steps),full_dataset(args.root_path,'val'),args.device,(.6,),reference_threshold=.45)
            report=values[.6]; score=float(report['selection_score']); record['val_score_at_0_60']=score
            atomic_json(out/f'val_epoch_{epoch:03d}.json',report)
            if score>best:
                best=score; save(out/'best.pth',model,identity,epoch,best,None,config)
            print(f'[VAL] {variant} {steps} steps epoch={epoch} score={score:.6f}',flush=True)
        save(out/'last.pth',model,identity,epoch,best,optimizer,config,record)
        store_epoch(out,record)
        # Do not print hundreds of step-distribution fields every epoch.
        print('[EPOCH] '+json.dumps({key:record[key] for key in ('epoch','variant','steps','losses','case_order_sha256')}),flush=True)
    atomic_json(out/'training_complete.json',{'epochs':args.epochs,'best_score':best,'variant':variant,'steps':steps})


def store_epoch(out,record):
    path=Path(out)/'epochs.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    rows=[row for row in rows if row['epoch']!=record['epoch']]; rows.append(record); rows.sort(key=lambda row:row['epoch'])
    temp=path.with_suffix('.tmp'); temp.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf-8'); temp.replace(path)


def verify_pair(args):
    paths=[args.output_dir/name/'epochs.jsonl' for name in ('control_3step','gate_3step')]
    rows=[{entry['epoch']:entry for line in path.read_text().splitlines() if line.strip() for entry in [json.loads(line)]} for path in paths]
    for epoch in range(1,args.epochs+1):
        if epoch not in rows[0] or epoch not in rows[1]: raise ValueError('Missing paired training epoch')
        for key in ('case_order_sha256','augmented_images_sha256'):
            if rows[0][epoch][key]!=rows[1][epoch][key]: raise ValueError(f'Paired {key} differs at epoch {epoch}')
    atomic_json(args.output_dir/'paired_data_verified.json',{'passed':True,'epochs':args.epochs,
        'verified':['same source checkpoint','same source learning rates','full case order hash','full augmented image tensor hash'],
        'note':'Both runs use the original stage-D trainable range and fresh same-config AdamW.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','smoke','control','gate','compare','steps'))
    p.add_argument('--run_dir',type=Path,required=True); p.add_argument('--root_path',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True); p.add_argument('--reference_set',type=Path)
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch_size',type=int,default=2); p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=1234); p.add_argument('--val_interval',type=int,default=5)
    p.add_argument('--lambda_gate',type=float,default=.1); p.add_argument('--min_action',type=float,default=1e-5)
    p.add_argument('--gain_epsilon',type=float,default=1e-7); p.add_argument('--max_pairs_per_sign',type=int,default=256)
    p.add_argument('--auto_train6',action='store_true')
    args=p.parse_args(); args.device=torch.device(args.device)
    if args.batch_size!=2 or min(args.epochs,args.val_interval,args.max_pairs_per_sign)<1 or args.num_workers<0 or args.lambda_gate<0:
        p.error('Requires batch2 and positive budgets/epochs')
    if args.command in ('all','compare','steps') and (args.reference_set is None or not args.reference_set.is_file()): p.error('Requires the fixed 206-gap reference JSON')
    args.output_dir.mkdir(parents=True,exist_ok=True); torch.manual_seed(args.seed); np.random.seed(args.seed)
    model,source_losses,identity,_,source,selection=load_original(args); args.source_losses=source_losses
    common,benefit=settings(args,identity,source,3,'paired')
    manifest=args.output_dir/'experiment.json'
    if manifest.is_file() and json.loads(manifest.read_text())!=common: raise ValueError('Experiment configuration changed')
    atomic_json(manifest,common)
    if args.command=='smoke' or args.command=='all' and not (args.output_dir/'smoke.json').is_file(): smoke(args,model,benefit,identity,source)
    if args.command in ('all','control'): train_variant(args,model,source,identity,'control',3)
    if args.command in ('all','gate'): train_variant(args,model,source,identity,'gate',3)
    if args.command in ('all','compare','steps'):
        verify_pair(args)
        from gate_benefit_protocol import run_comparison,run_step_diagnostics
        report=run_comparison(args,model,identity,source,selection,benefit)
        if args.command in ('all','steps'): run_step_diagnostics(args,model,identity,source,selection,benefit,report,train_variant)
    if v9.digest(source)!=common['original_sha256']: raise RuntimeError('Original V9 checkpoint changed')


if __name__=='__main__': main()
