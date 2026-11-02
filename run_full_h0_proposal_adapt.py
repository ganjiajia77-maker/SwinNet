"""Continue the trained full-H0 B checkpoint with an unscaled proposal logit.

Original H0 supervision, source relation/segmentation/continuity losses,
optimizer scope and 3-step dynamics are retained. This is a 20-epoch
continuation from B best, not a same-initialization causal ablation.
"""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diagnostics.diagnose_full_h0_optimistic_reachability import rebuild_source,verify_recorded_code
from networks.full_h0_joint import VERSION,build_h0,original_criterion,parameter_groups
from networks.full_h0_proposal_adapt import ProposalAdaptedJoint
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_selector_transport_training import gradient_norm
from run_full_h0_joint import (batch_loss,check_reusable_A,norms,restore,save,training_dataset,
                               validation,validation_repair_activity,evaluate)
from run_slim_end_to_end_mass import digest,schedule,seed_all
from run_v9_gate_benefit import atomic_json
from run_v9_selector_transport import point_metrics

ROOT=Path(__file__).resolve().parent


def build_model(checkpoint,device,size=512):
    if checkpoint.get('version')!=VERSION or checkpoint['config']['variant']!='B':
        raise ValueError('Expected the trained full-H0 B checkpoint')
    state=checkpoint['model'];source,q=rebuild_source(state)
    base,_=build_h0(checkpoint['config']['h0_args'],size=size)
    model=ProposalAdaptedJoint(base,q,source)
    model.load_state_dict(state,strict=True)
    return model.to(device)


def region_metrics(result,batch):
    """Training-only GT diagnostics; no GT enters model.forward."""
    p0=result['u0'].detach();s=result['positive_support'].detach()
    plus=result['delta_z_plus'].detach();minus=result['delta_z_minus'].detach()
    logits=torch.logit(p0.clamp(1e-5,1.-1e-5))
    old=(logits+s*plus-minus).sigmoid();ceiling=(logits+8*s).sigmoid()
    target=F.interpolate(batch['mask'],size=p0.shape[-2:],mode='nearest')
    skel=F.adaptive_max_pool2d(batch['skeleton'],p0.shape[-2:])
    regions={'missed_skeleton_proxy':(skel>.5)&(p0<.45),
             'q_high_background_proxy':(target<.5)&(p0<.45)&(result['q'].detach()>.5)}
    out={}
    for name,mask in regions.items():
        count=int(mask.sum());row={'pixels':count}
        for key,value in (('p0',p0),('support',s),('b_original_formula',old),('b_unscaled',result['b'].detach()),
                          ('old_best_possible_ignore_minus',ceiling),('delta_plus',plus),('delta_minus',minus)):
            row[key+'_sum']=float(value[mask].sum()) if count else 0.
        row['old_ceiling_below_045']=int((mask&(ceiling<.45)).sum())
        row['b_unscaled_above_045']=int((mask&(result['b'].detach()>=.45)).sum())
        out[name]=row
    return out


def smoke(args,model,criterion):
    dataset=training_dataset(args,False);dataset.set_epoch(1)
    batch=next(iter(DataLoader(dataset,batch_size=2,num_workers=0)))
    batch={k:(v.to(args.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    buffers={name:v.clone() for name,v in model.named_buffers()}
    model.eval()
    with torch.no_grad():
        model.proposal_mode='original';old=model(batch['image'])
        model.proposal_mode='unscaled';new=model(batch['image'])
        torch.testing.assert_close(old['p0'],new['p0'],rtol=0,atol=0)
        p=new['u0'];z=torch.logit(p.clamp(1e-5,1.-1e-5))
        torch.testing.assert_close(new['b'],(z+new['delta_z_plus']-new['delta_z_minus']).sigmoid())
        torch.testing.assert_close(old['b'],(z+new['positive_support']*new['delta_z_plus']-new['delta_z_minus']).sigmoid())
        if bool((new['b']+1e-6<old['b']).any()):raise RuntimeError('Unscaled proposal must not fall below original at same features')
        before=old['p1']>=.45;after=new['p1']>=.45;gt=batch['mask']>.5
        initial={'proposal_mean_abs':float((new['b']-old['b']).abs().mean()),
                 'proposal_max_abs':float((new['b']-old['b']).abs().max()),
                 'p1_mean_abs':float((new['p1']-old['p1']).abs().mean()),
                 'p1_max_abs':float((new['p1']-old['p1']).abs().max()),
                 'added_tp':int((after&~before&gt).sum()),'lost_tp':int((before&~after&gt).sum()),
                 'added_fp':int((after&~before&~gt).sum()),'removed_fp':int((before&~after&~gt).sum()),
                 'region_sums':region_metrics(new,batch)}
    del old,new
    seed_all(args.seed);model.train();model.zero_grad(set_to_none=True)
    if args.device.type=='cuda':torch.cuda.reset_peak_memory_stats(args.device)
    start=time.perf_counter();result=model(batch['image'])
    total,terms=batch_loss(args,model,criterion,result,batch,args.seed)
    seg,_=dynamic_losses(result,batch['mask'],batch['skeleton'],lambda_cont=0.)
    groups={'H0':tuple(p for p in model.base.parameters() if p.requires_grad),
            'Q3':tuple(model.original.upstream.q_features.parameters())+tuple(model.original.upstream.q_head.parameters()),
            'proposal':tuple(model.original.proposal_head.parameters()),'selector':tuple(model.selector.parameters()),
            'rates':tuple(model.original.dynamics.rates.parameters())}
    gradients={name:gradient_norm(seg,params) for name,params in groups.items()}
    if not bool(torch.isfinite(total)) or not all(np.isfinite(v) for v in gradients.values()):raise RuntimeError('Nonfinite smoke loss/gradient')
    if gradients['proposal']<=0:raise RuntimeError('Final segmentation has no proposal-head gradient')
    total.backward();grad=norms(model)
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError('Nonfinite smoke gradients')
    with torch.no_grad():
        for name,value in model.named_buffers():value.copy_(buffers[name])
    model.zero_grad(set_to_none=True);seed_all(args.seed)
    report={'passed':True,'mode':'unscaled','initial_comparison_same_checkpoint':initial,
            'total_loss':float(total.detach()),'final_segmentation_gradients':gradients,'total_gradient_norms':grad,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'seconds':time.perf_counter()-start,
            'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None,
            'GT_in_forward':False}
    atomic_json(args.output_dir/'smoke.json',report);print('[SMOKE] '+json.dumps(report),flush=True)


def train(args,model,criterion,optimizer):
    start=1;best=-float('inf');last=args.output_dir/'last.pth'
    if last.is_file():
        state=restore(args,model,last,optimizer);start=state['epoch']+1;best=state['score']
        if state.get('record'):atomic_json(args.output_dir/f'epoch_{state["epoch"]:03d}.json',state['record'])
    dataset=training_dataset(args)
    for epoch in range(start,args.epochs+1):
        dataset.set_epoch(epoch);schedule(optimizer,epoch-1,args.epochs)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed+epoch*1000003),pin_memory=args.device.type=='cuda')
        model.train();order=hashlib.sha256();images=hashlib.sha256();sums=Counter();regions={};first_grads=None
        for index,batch in enumerate(loader):
            for case in batch['case_name']:order.update(str(case).encode()+b'\0')
            images.update(batch['image'].contiguous().numpy().tobytes())
            torch.manual_seed(args.seed+epoch*1000003+index)
            batch={k:(v.to(args.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True);result=model(batch['image'])
            total,terms=batch_loss(args,model,criterion,result,batch,args.seed+epoch*100000+index)
            if not bool(torch.isfinite(total)):raise RuntimeError('Nonfinite training loss')
            audit=region_metrics(result,batch)
            for name,values in audit.items():
                regions.setdefault(name,Counter()).update(values)
            total.backward()
            if first_grads is None:first_grads=norms(model)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(grad)):raise RuntimeError('Nonfinite training gradient')
            optimizer.step();sums['total']+=float(total.detach());sums['batches']+=1
            for key,value in terms.items():sums[key]+=float(value)
            if index%25==0:
                print(f'[TRAIN] epoch={epoch}/{args.epochs} batch={index+1}/{len(loader)} loss={float(total.detach()):.6f} '
                      f'transport={float(terms["transport_abs_mean"]):.6g} proposal={float(terms["proposal_abs_mean"]):.6g} '
                      f'P1-P0={float(terms["P1_minus_P0_abs_mean"]):.6g}',flush=True)
        record={'epoch':epoch,'case_order_sha256':order.hexdigest(),'augmented_images_sha256':images.hexdigest(),
                'losses':{k:v/sums['batches'] for k,v in sums.items() if k!='batches'},
                'gradients':first_grads,'proposal_region_sums':{k:dict(v) for k,v in regions.items()}}
        control=json.loads((args.pair_dir/'A'/f'epoch_{epoch:03d}.json').read_text())
        for key in ('case_order_sha256','augmented_images_sha256'):
            if control[key]!=record[key]:raise ValueError(f'Training input differs from prior A at epoch {epoch}: {key}')
        record['reused_A_input_hashes_match']=True
        if epoch%5==0 or epoch==args.epochs:
            cases=validation(args,model);report=point_metrics(cases,'p1',.6,args.manifest)
            record['val_score_at_0_60']=report['selection_score']
            record['validation_repair_activity']=validation_repair_activity(cases)
            atomic_json(args.output_dir/f'val_epoch_{epoch:03d}.json',report)
            if report['selection_score']>best:
                best=report['selection_score'];save(args,model,None,args.output_dir/'best.pth',epoch,best)
            print(f'[VAL] epoch={epoch} score={best:.6f}',flush=True)
        atomic_json(args.output_dir/f'epoch_{epoch:03d}.json',record);save(args,model,optimizer,last,epoch,best,record)
    atomic_json(args.output_dir/'training_complete.json',{'epochs':args.epochs,'best_score':best})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','smoke','train','val','compare'))
    p.add_argument('--pair_dir',type=Path,required=True)
    p.add_argument('--source_run',default='B_transport_0p01_proposal_0p001')
    p.add_argument('--run_name',default='B_unscaled_proposal_20e')
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20)
    args=p.parse_args();args.device=torch.device(args.device);args.variant='B';args.b_run_name=args.run_name
    for name in (args.source_run,args.run_name):
        if Path(name).name!=name or not name.startswith('B') or name in ('B','B_'):
            p.error('Run names must be B-prefixed single directory names')
    if args.source_run==args.run_name:p.error('Use a separate output run')
    args.pair_dir=args.pair_dir.resolve();args.output_dir=args.pair_dir/args.run_name
    if args.command=='compare':
        from compare_full_h0_proposal_adapt import compare
        compare(args.pair_dir,args.run_name,args.source_run);return
    source=args.pair_dir/args.source_run;source_path=source/'best.pth'
    checkpoint=torch.load(source_path,map_location='cpu',weights_only=False)
    previous=checkpoint['config'];verify_recorded_code(previous)
    if checkpoint.get('version')!=VERSION or previous['variant']!='B':raise ValueError('Wrong source B checkpoint')
    if args.epochs!=20 or previous['epochs']!=20:raise ValueError('This comparison requires the completed 20-epoch A and B runs')
    manifest_path=args.pair_dir/'reference/fixed_gaps_and_anchors.json'
    if digest(manifest_path)!=previous['reference_sha256']:raise ValueError('Fixed reference differs from source B')
    old_selection=json.loads((source/'eval/val_selection.json').read_text())
    if old_selection['checkpoint_sha256']!=digest(source_path) or old_selection['reference_sha256']!=previous['reference_sha256']:
        raise ValueError('Source B best checkpoint and validation selection do not match')
    args.manifest=json.loads(manifest_path.read_text());args.root_path=Path(previous['root_path'])
    args.seed=int(previous['seed']);args.num_workers=int(previous['num_workers']);args.h0_lr=float(previous['h0_lr'])
    args.config=copy.deepcopy(previous)
    args.config['run_directory']=args.run_name
    args.config['rate_initialization']='strictly inherited from source B best checkpoint; no rate reset'
    args.config['proposal_adaptation']={'version':'unscaled_positive_logit_v1','source_B_checkpoint':str(source_path.resolve()),
        'source_B_sha256':digest(source_path),'source_B_epoch':checkpoint['epoch'],'additional_epochs':20,
        'formula':'sigmoid(logit(P0) + DeltaZ_plus - DeltaZ_minus)',
        'positive_support':'retained as proposal/accept input; removed only as positive-logit multiplier',
        'training_scope':'same H0/Q3/selector/repair parameter groups and original losses as B',
        'comparison_limit':'starts from old B best; not equal cumulative training budget or pure ablation',
        'code_sha256':{str(path.relative_to(ROOT)):digest(path) for path in
            (ROOT/'run_full_h0_proposal_adapt.py',ROOT/'networks/full_h0_proposal_adapt.py',ROOT/'compare_full_h0_proposal_adapt.py')}}
    check_reusable_A(args.pair_dir,args.config)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    marker=args.output_dir/'experiment.json'
    if marker.is_file() and json.loads(marker.read_text())!=args.config:raise ValueError('Existing experiment configuration differs')
    atomic_json(marker,args.config)
    seed_all(args.seed);model=build_model(checkpoint,args.device);del checkpoint;gc.collect()
    criterion=original_criterion(previous['h0_args'],args.device)
    rates={k:v['lr'] for k,v in previous['source_training'].get('optimizer',{}).items()}
    optimizer=parameter_groups(model,args.h0_lr,repair_rates=rates)
    atomic_json(args.output_dir/'parameter_groups.json',{g['name']:{'parameters':sum(p.numel() for p in g['params']),
        'initial_lr':g['initial_lr'],'minimum_lr':g['minimum_lr']} for g in optimizer.param_groups})
    if args.command in ('all','train','smoke') and not (args.output_dir/'last.pth').is_file():smoke(args,model,criterion)
    if args.command=='smoke':return
    if args.command in ('all','train'):train(args,model,criterion,optimizer)
    if args.command in ('all','val'):evaluate(args,model)
    if args.command=='all':
        from compare_full_h0_proposal_adapt import compare
        compare(args.pair_dir,args.run_name,args.source_run)
    print('[DONE] '+str(args.output_dir),flush=True)


if __name__=='__main__':main()
