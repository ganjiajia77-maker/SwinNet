"""Paired FULL H0 continuation and live source-selected repair, no L_choice."""
import argparse
import gc
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from networks.full_h0_joint import (VERSION,INITIAL_RATE_BIAS,build_h0,original_criterion,h0_loss,
    FullH0Control,FullH0Joint,parameter_groups)
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_source_relation_labels import RelationLabels
from networks.v9_selector_transport_training import VERSION as SOURCE_VERSION,live_relation_loss,gradient_norm
from run_slim_end_to_end_mass import (digest,seed_all,schedule,atomic_checkpoint,
    infer,fixed_reference,area_threshold)
from run_v9_selector_transport import point_metrics
from run_v9_gate_benefit import atomic_json
from selector_transport_reference import exact_recall_threshold
from selector_transport_provenance import SELECTION_RULE,check_code_hashes
from diagnostics.v91_counterfactual import write_csv
from datasets.dataset_road_skeleton import RoadSkeletonDataset

DEFAULT_H0='/home/gjj/SwinNet-h0-mass/model_out/mass_h0_599a410_random512_fp32_80e_bs2_20261011_113054/best.pth'
DEFAULT_REPAIR='/home/gjj/SwinNet-roadbias/model_out/h0_v9_source_selector_20261030_110206/actual_transport_20261030_181042'


def inspect_source(run):
    """Bind actual best weights to completion AND the reported 0.67 work point."""
    run=Path(run); path=run/'best.pth'
    source=torch.load(path,map_location='cpu',weights_only=False)
    if source.get('version')!=SOURCE_VERSION: raise ValueError('Not the joint actual-transport S_selector checkpoint')
    comparison=json.loads((run/'comparison.json').read_text())
    report=comparison['reports']['S_selector']['own_val_threshold']
    complete=json.loads((run/'training_complete.json').read_text())
    if abs(report['threshold']-.67)>1e-8: raise ValueError('Expected the specified S_selector own_val_threshold=0.67')
    if abs(complete['best_score']-source['score'])>1e-8 or complete['epochs']!=source['config']['epochs']:
        raise ValueError('S_selector best.pth and completion record differ')
    cfg=json.loads((run/'experiment.json').read_text())
    if cfg!=source['config']: raise ValueError('S_selector configuration and checkpoint differ')
    check_code_hashes(cfg)
    if Path(cfg['identity']['q3']['path'])!=Path(source['identity']['q3']['path']): raise ValueError('Q3 identity mismatch')
    q_path=Path(source['identity']['q3']['path'])
    if digest(q_path)!=source['identity']['q3']['sha256']: raise ValueError('Q3 weights differ from trained S_selector')
    q=torch.load(q_path,map_location='cpu',weights_only=False)
    if q.get('repair_version')!='dual_recovery_v1': raise ValueError('Expected trained dual-recovery Q3')
    return source,q,{'checkpoint':str(path.resolve()),'sha256':digest(path),'epoch':source['epoch'],
        'checkpoint_score_at_0_60':source['score'],'own_val_threshold':report['threshold'],
        'q3':source['identity']['q3'],'original_local_gap_base_not_loaded':source['identity']['base'],
        'unused_Q3_R3_heads':'Only q_features/q_head are used; old Q3 R3 residual heads are not part of source transport.',
        'unused_source_state_gate':'Retained in source file; not used in nine-choice update.'}


def checkpoint_config(args,h0,saved,repair_meta=None,repair_config=None):
    result={'version':VERSION,'variant':args.variant,'h0_checkpoint':str(args.h0_checkpoint.resolve()),
        'h0_sha256':digest(args.h0_checkpoint),'h0_epoch':h0['epoch'],'h0_args':saved,
        'repair':repair_meta,'source_training':repair_config,'epochs':args.epochs,'seed':args.seed,
        'root_path':str(args.root_path.resolve()),'batch_size':2,'num_workers':args.num_workers,
        'val_interval':5,'fp32':True,'EMA':False,'accumulation':1,'h0_lr':args.h0_lr,
        'rate_initial_bias':(INITIAL_RATE_BIAS if args.variant=='B' else -8.),
        'rate_initialization':('final rates output weight=0; bias=[2.5,-13,-13,1]; all rate parameters trainable'
                               if args.variant=='B' else 'not applicable'),
        'lambda_relation':(.1 if repair_config is None else repair_config['lambda_relation']),
        'lambda_choice':0.,'lambda_G':0.,'appearance_augmentation':False,
        'loss':'A: original H0 loss; B: same H0 loss + original S_selector Lseg + .2Lcont + lambda_relation*Lrelation; Q trained via final loss, no new Q loss',
        'selection_rule':SELECTION_RULE,'checkpoint_threshold':.60,'steps':3,'neighbours':8,
        'reference':'initial full H0 P0@0.45, shared and frozen BEFORE either group trains; not old local-gap 206 set',
        'reference_sha256':digest(args.pair_dir/'reference/fixed_gaps_and_anchors.json'),
        'code_sha256':{str(p.relative_to(Path(__file__).parent)):digest(p) for p in sorted(
            list((Path(__file__).parent/'reference_h0_599a410').rglob('*.py'))+
            [Path(__file__),Path(__file__).parent/'networks/full_h0_joint.py'])}}
    if args.variant=='B' and getattr(args,'transport_rate_init',None) is not None:
        result.update(rate_initial_bias=None,rate_initialization='explicit_softmax_budgets',
                      initial_transport_rate=args.transport_rate_init,
                      initial_proposal_rate=args.proposal_rate_init,
                      run_directory=getattr(args,'b_run_name','B'))
    return result


def check_reusable_A(pair,config):
    """Read-only verification of the completed control and fixed references."""
    folder=Path(pair)/'A'
    old=json.loads((folder/'experiment.json').read_text())
    complete=json.loads((folder/'training_complete.json').read_text())
    selection=json.loads((folder/'eval/val_selection.json').read_text())
    if complete['epochs']!=config['epochs'] or old['variant']!='A':
        raise ValueError('Existing A is not the completed same-budget control')
    for key in ('h0_sha256','epochs','seed','root_path','batch_size','num_workers',
                'val_interval','h0_lr','selection_rule','reference_sha256'):
        if old[key]!=config[key]: raise ValueError('Cannot reuse A: '+key+' differs')
    for key,value in old.get('code_sha256',{}).items():
        if key.startswith('reference_h0_599a410/') and config['code_sha256'].get(key)!=value:
            raise ValueError('Original H0 implementation differs from A: '+key)
    if selection['checkpoint_sha256']!=digest(folder/'best.pth'):
        raise ValueError('A evaluation no longer corresponds to its best checkpoint')
    for epoch in range(1,config['epochs']+1):
        record=json.loads((folder/f'epoch_{epoch:03d}.json').read_text())
        if not all(key in record for key in ('case_order_sha256','augmented_images_sha256')):
            raise ValueError('A lacks input reproducibility records')
    if len(list((folder/'eval/val').glob('*.npz')))!=14:
        raise ValueError('A must have saved predictions for all 14 validation images')
    print(f'[REUSE A] completed {config["epochs"]} epochs; checkpoint/reference/config verified; no A training',flush=True)


def load_h0(path,device):
    h0=torch.load(path,map_location='cpu',weights_only=False)
    if h0.get('experiment_protocol')!='h0-mass-native1500-random512-fp32-80e-bs2-v1':
        raise ValueError('Expected the original full H0 Massachusetts FP32/noEMA checkpoint')
    if h0.get('topology_attention_version')!='stage-topology-roadness-gap-v1': raise ValueError('Wrong H0 architecture')
    if h0['args'].get('use_ema') or h0.get('ema_state_dict') is not None: raise ValueError('Expected raw noEMA H0')
    base,saved=build_h0(h0['args'])
    base.load_state_dict(h0['model_state_dict'],strict=True)
    print(f'[H0 STRICT] tensors={len(base.state_dict())} missing=0 unexpected=0; full H/global ON',flush=True)
    return base.to(device),h0,saved


def reference(args):
    folder=args.pair_dir/'reference'; marker=folder/'complete.json'
    if marker.is_file():
        saved=json.loads(marker.read_text())
        if saved['h0_sha256']!=digest(args.h0_checkpoint) or saved['root_path']!=str(args.root_path.resolve()):
            raise ValueError('Shared reference belongs to another H0/dataset')
        if saved['manifest_sha256']!=digest(folder/'fixed_gaps_and_anchors.json'): raise ValueError('Reference manifest changed')
        contract=json.loads((args.pair_dir/'pair_contract.json').read_text())
        for key,value in (('epochs',args.epochs),('seed',args.seed),('num_workers',args.num_workers),('h0_lr',args.h0_lr)):
            if contract[key]!=value: raise ValueError('A/B settings differ: '+key)
        return
    base,_,_=load_h0(args.h0_checkpoint,args.device)
    model=FullH0Control(base).eval()
    cases=infer(args,model,'val',False)
    manifest=fixed_reference(cases)
    manifest['version']=VERSION
    manifest['origin']='Initial FULL H0 before either A/B training; fixed shared gaps/anchors for this pair, not the local-gap 206 list'
    folder.mkdir(parents=True,exist_ok=True)
    for case in cases: np.savez_compressed(folder/(case['case']+'.npz'),p0=case['p0'],gt=case['gt'])
    atomic_json(folder/'fixed_gaps_and_anchors.json',manifest)
    atomic_json(args.pair_dir/'pair_contract.json',{'h0_sha256':digest(args.h0_checkpoint),
        'epochs':args.epochs,'seed':args.seed,'root_path':str(args.root_path.resolve()),'batch_size':2,
        'num_workers':args.num_workers,'val_interval':5,'h0_lr':args.h0_lr,'selection_rule':SELECTION_RULE,
        'reference_sha256':digest(folder/'fixed_gaps_and_anchors.json')})
    atomic_json(marker,{'h0_sha256':digest(args.h0_checkpoint),'root_path':str(args.root_path.resolve()),
        'manifest_sha256':digest(folder/'fixed_gaps_and_anchors.json'),'gaps':manifest['count'],'images':len(cases)})
    print(f'[REFERENCE] fixed {manifest["count"]} gaps on {len(cases)} val images',flush=True)


def training_dataset(args,augment=True):
    return RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=augment)


def batch_loss(args,model,criterion,result,batch,seed):
    original,old_terms=h0_loss(criterion,result['h0_outputs'],batch)
    terms={'h0':original.detach()}; total=original
    if args.variant=='B':
        cfg=args.config['source_training']
        extra,values=dynamic_losses(result,batch['mask'],batch['skeleton'],**cfg['source_losses'])
        relation,info=live_relation_loss(result,batch['mask'],batch['skeleton'],RelationLabels(**cfg['labels']),seed)
        total=total+extra+args.config['lambda_relation']*relation
        terms.update(seg=values['seg'],cont=values['cont'],relation=relation.detach(),matched_pairs=info['matching']['matched_pairs'])
        terms.update(transport_rate_mean=result['transport_rate'].detach().mean(),
                     proposal_coefficient_mean=result['beta'].detach().mean(),
                     P1_minus_P0_abs_mean=(result['p1']-result['p0']).detach().abs().mean())
        terms.update(transport_abs_mean=result['transport_delta'].detach().abs().mean(),
                     transport_signed_mean=result['transport_delta'].detach().mean(),
                     proposal_abs_mean=result['proposal_delta'].detach().abs().mean(),
                     anchor_abs_mean=result['anchor_delta'].detach().abs().mean())
        old,new=result['p0'].detach()>=.45,result['p1'].detach()>=.45
        gt=batch['mask']>.5
        terms.update(repair_added_tp=((new&~old)&gt).sum(),repair_lost_tp=((old&~new)&gt).sum(),
                     repair_added_fp=((new&~old)&~gt).sum(),repair_removed_fp=((old&~new)&~gt).sum(),
                     P1_P0_flip_fraction=(new!=old).float().mean())
    return total,terms


def norms(model):
    values={}
    for name,param in model.named_parameters():
        if param.grad is None: continue
        if name.startswith(('base.','original.upstream.base.')):
            group='H0_encoder' if any(token in name for token in ('.layers.','.patch_embed.')) else 'H0_decoder_structure_global'
        elif name.startswith('original.upstream.'): group='Q3'
        elif name.startswith('selector.'): group='selector'
        else: group=name.split('.')[1]
        values.setdefault(group,[]).append(param.grad.detach().double().square().sum())
    return {k:float(torch.stack(v).sum().sqrt()) for k,v in values.items()}


def smoke(args,model,criterion):
    dataset=training_dataset(args,False); dataset.set_epoch(1)
    batch=next(iter(DataLoader(dataset,batch_size=2,num_workers=0)))
    batch={k:(v.to(args.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    saved_buffers={k:v.clone() for k,v in model.named_buffers()}
    seed_all(args.seed); model.train(); model.zero_grad(set_to_none=True)
    if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
    started=time.perf_counter(); result=model(batch['image'],capture_diagnostics=True)
    total,terms=batch_loss(args,model,criterion,result,batch,args.seed)
    extra={}
    if args.variant=='B':
        # FINAL segmentation alone, not the original H0 or relation losses.
        seg,_=dynamic_losses(result,batch['mask'],batch['skeleton'],lambda_cont=0.)
        groups={'H0':tuple(p for p in model.base.parameters() if p.requires_grad),
            'Q3':tuple(model.original.upstream.q_features.parameters())+tuple(model.original.upstream.q_head.parameters()),
            'selector':tuple(model.selector.parameters()),
            'rates_output':tuple(model.original.dynamics.rates[-1].parameters())}
        extra['final_segmentation_gradients']={k:gradient_norm(seg,v) for k,v in groups.items()}
        if any(v<=0 for v in extra['final_segmentation_gradients'].values()): raise RuntimeError('Final repair gradient is blocked')
        repair_groups={'R':model.original.reliable_head,'W_relations':model.original.relations,
                       'B':model.original.proposal_head,'A':model.original.accept_head,
                       'relation_features':model.original.gate_relation}
        extra['repair_segmentation_gradients']={name:gradient_norm(seg,tuple(p for p in module.parameters() if p.requires_grad))
                                                for name,module in repair_groups.items()}
        for k in range(3):
            a=result['source_choice'][:,k]; u=result['states'][k]
            expected=result['transport_rate'][:,k]*(a[:,1:]*(result['source_states'][:,k]-u)).sum(1,keepdim=True)
            torch.testing.assert_close(result['transport_delta'][:,k],expected)
            torch.testing.assert_close(result['states'][k+1],u+expected+result['proposal_delta'][:,k]+result['anchor_delta'][:,k])
            if bool((a[:,1:][~result['source_valid'][:,k]]!=0).any()): raise RuntimeError('Invalid source chosen')
    if not torch.isfinite(total): raise RuntimeError('Nonfinite loss')
    total.backward(); gradients=norms(model)
    if args.variant=='B' and any(p.grad is not None for p in model.original.gate_relation.delta.parameters()):
        raise RuntimeError('Unused scalar gate received gradients')
    report={'passed':True,'batch_size':2,'image_shape':list(batch['image'].shape),'loss':float(total.detach()),
        'gradients':gradients,'seconds':time.perf_counter()-started,
        'feature_shapes':result.get('feature_shapes',{}),
        'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'H0_parameters':sum(p.numel() for p in model.base.parameters()),'GT_in_model_forward':False,**extra}
    model.eval(); model.zero_grad(set_to_none=True)
    with torch.no_grad():
        # Restore BN before comparing to the shared original eval identity.
        for name,value in model.named_buffers(): value.copy_(saved_buffers[name])
        initial=model(batch['image']); delta=(initial['p1']-initial['p0']).abs()
        report['initial_P1_minus_P0']={'mean_abs':float(delta.mean()),'max_abs':float(delta.max()),
            'changed_binary_at_0_45':int(((initial['p1']>=.45)!=(initial['p0']>=.45)).sum())}
        if args.variant=='B':
            report['initial_dynamics']={
                'transport_rate_max':float(initial['transport_rate'].max()),
                'proposal_coefficient_max':float(initial['beta'].max()),
                'work_update_max_abs':float((initial['u3']-initial['u0']).abs().max()),
                'work_logit_delta_max_abs':float(initial['delta_logit'].abs().max())}
            explicit=args.config.get('rate_initialization')=='explicit_softmax_budgets'
            report['startup_policy']='explicit measurable budgets' if explicit else 'legacy near-identity budgets'
            if explicit:
                expected=args.config['initial_transport_rate']
                torch.testing.assert_close(initial['transport_rate'],torch.full_like(initial['transport_rate'],expected),rtol=1e-5,atol=1e-8)
                output=model.original.dynamics.rates[-1]
                actual=output.bias.softmax(0)
                if abs(float(actual[2])-args.config['initial_proposal_rate'])>1e-7:
                    raise RuntimeError('Initial proposal budget differs from requested value')
                report['initial_transport_abs_mean']=float(initial['transport_delta'].abs().mean())
                if not bool(torch.isfinite(initial['p1']).all()) or report['initial_transport_abs_mean']<=0 or float(delta.mean())<=0:
                    raise RuntimeError('Explicit-budget repair has no measurable finite forward action')
            if not explicit and (float(delta.mean())>.001 or float(delta.max())>.02):
                report['passed']=False
                atomic_json(args.output_dir/'smoke_failed.json',report)
                raise RuntimeError(f'Initial repair not close enough to identity: {report["initial_P1_minus_P0"]}')
            bias=model.selector.stay[-1].bias; before=bias.clone()
            try:
                bias.add_(3.); altered=model(batch['image'])
                report['selector_perturbation_P1_mean_abs']=float((altered['p1']-initial['p1']).abs().mean())
            finally: bias.copy_(before)
            report['initial_selector_perturbation_P1_mean_abs']=report['selector_perturbation_P1_mean_abs']
            report['selector_probe_uses_initial_rates']=True
            if report['selector_perturbation_P1_mean_abs']<=0:
                # The deliberately tiny initial budget can round to identical
                # FP32 states on flat predictions. Separately test the routing
                # with a measurable budget; restore ALL probe parameters.
                output=model.original.dynamics.rates[-1]
                output_weight=output.weight.clone(); output_bias=output.bias.clone()
                try:
                    output.weight.zero_()
                    output.bias.copy_(output.bias.new_tensor([2.5,-2.,-3.,1.]))
                    baseline=model(batch['image'])
                    bias.add_(3.); altered=model(batch['image'])
                    report['selector_probe_uses_initial_rates']=False
                    report['selector_probe_rate_bias']=[2.5,-2.,-3.,1.]
                    report['selector_perturbation_P1_mean_abs']=float((altered['p1']-baseline['p1']).abs().mean())
                finally:
                    output.weight.copy_(output_weight); output.bias.copy_(output_bias); bias.copy_(before)
            if report['selector_perturbation_P1_mean_abs']<=0:
                raise RuntimeError('Selector does not change final Surface even with a controlled nonzero budget')
    atomic_json(args.output_dir/'smoke.json',report)
    seed_all(args.seed)
    print('[SMOKE] '+json.dumps(report),flush=True)


def save(args,model,optimizer,path,epoch,score,record=None):
    value={'version':VERSION,'config':args.config,'model':model.state_dict(),'epoch':epoch,'score':score,'record':record}
    if optimizer is not None:
        value.update(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    atomic_checkpoint(path,value)


def restore(args,model,path,optimizer=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version')!=VERSION or state['config']!=args.config: raise ValueError('Checkpoint identity/config mismatch')
    model.load_state_dict(state['model'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def validation(args,model,save_predictions=False):
    cases=infer(args,model,'val',False)
    for case in cases:
        with np.load(args.pair_dir/'reference'/(case['case']+'.npz')) as old:
            if not np.array_equal(case['gt'],old['gt']): raise ValueError('Validation target changed')
            case['current_p0']=case['p0'];case['p0']=old['p0'].copy()
        if save_predictions:
            folder=args.output_dir/'eval/val';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/(case['case']+'.npz'),**{k:case[k] for k in ('gt','p0','current_p0','p1')})
    return cases


def validation_repair_activity(cases,threshold=.6):
    """Observe current P0/P1 differences without changing checkpoint selection."""
    count=sum(case['p1'].size for case in cases)
    result={'threshold':threshold,'pixels':count,'P1_minus_current_P0_abs_mean':0.,
            'P1_minus_current_P0_abs_max':0.,'flipped_pixels':0,'added_tp':0,'lost_tp':0,'added_fp':0,'removed_fp':0}
    for case in cases:
        before,after=case['current_p0']>=threshold,case['p1']>=threshold
        gt=case['gt']>.5
        delta=np.abs(case['p1']-case['current_p0'])
        result['P1_minus_current_P0_abs_mean']+=float(delta.sum(dtype=np.float64))/count
        result['P1_minus_current_P0_abs_max']=max(result['P1_minus_current_P0_abs_max'],float(delta.max()))
        result['flipped_pixels']+=int((before!=after).sum())
        result['added_tp']+=int((after&~before&gt).sum());result['lost_tp']+=int((before&~after&gt).sum())
        result['added_fp']+=int((after&~before&~gt).sum());result['removed_fp']+=int((before&~after&~gt).sum())
    return result


def train(args,model,criterion,optimizer):
    start=1; best=-float('inf'); last=args.output_dir/'last.pth'
    if last.is_file():
        state=restore(args,model,last,optimizer); start=state['epoch']+1;best=state['score']
        if state.get('record'): atomic_json(args.output_dir/f'epoch_{state["epoch"]:03d}.json',state['record'])
    dataset=training_dataset(args)
    for epoch in range(start,args.epochs+1):
        dataset.set_epoch(epoch);schedule(optimizer,epoch-1,args.epochs)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed+epoch*1000003),pin_memory=args.device.type=='cuda')
        model.train();order=hashlib.sha256();images=hashlib.sha256();sums=Counter(); first_grads=None
        for index,batch in enumerate(loader):
            for case in batch['case_name']: order.update(str(case).encode()+b'\0')
            images.update(batch['image'].contiguous().numpy().tobytes())
            # Same per-batch H0 dropout/stochastic-depth stream in A and B.
            torch.manual_seed(args.seed+epoch*1000003+index)
            batch={k:(v.to(args.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True);result=model(batch['image'])
            total,terms=batch_loss(args,model,criterion,result,batch,args.seed+epoch*100000+index)
            if not torch.isfinite(total): raise RuntimeError('Nonfinite training loss')
            total.backward()
            if first_grads is None: first_grads=norms(model)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            sums['total']+=float(total.detach());sums['batches']+=1
            for key,value in terms.items(): sums[key]+=float(value)
            if index%25==0:
                activity=(f' rate={float(terms["transport_rate_mean"]):.6g} transport={float(terms["transport_abs_mean"]):.6g}'
                          f' proposal={float(terms["proposal_abs_mean"]):.6g} P1-P0={float(terms["P1_minus_P0_abs_mean"]):.6g}'
                          if args.variant=='B' else '')
                print(f'[TRAIN {args.variant}] epoch={epoch}/{args.epochs} batch={index+1}/{len(loader)} loss={float(total.detach()):.6f}'+activity,flush=True)
        record={'epoch':epoch,'case_order_sha256':order.hexdigest(),'augmented_images_sha256':images.hexdigest(),
            'losses':{k:v/sums['batches'] for k,v in sums.items() if k!='batches'},'gradients':first_grads}
        if args.variant=='B':
            record['participation']={'transport_observed':record['losses']['transport_abs_mean']>0,
                                     'surface_correction_observed':record['losses']['P1_minus_P0_abs_mean']>0,
                                     'training_monitor_threshold':.45}
        if args.variant=='B' and getattr(args,'b_run_name','B')!='B':
            control=json.loads((args.pair_dir/'A'/f'epoch_{epoch:03d}.json').read_text())
            for key in ('case_order_sha256','augmented_images_sha256'):
                if control[key]!=record[key]: raise ValueError(f'A/B input data differ at epoch {epoch}: '+key)
            record['reused_A_input_hashes_match']=True
        if epoch%5==0 or epoch==args.epochs:
            cases=validation(args,model);report=point_metrics(cases,'p1',.6,args.manifest)
            record['val_score_at_0_60']=report['selection_score']
            if args.variant=='B':
                record['validation_repair_activity']=validation_repair_activity(cases)
            atomic_json(args.output_dir/f'val_epoch_{epoch:03d}.json',report)
            if report['selection_score']>best:
                best=report['selection_score'];save(args,model,None,args.output_dir/'best.pth',epoch,best)
            print(f'[VAL {args.variant}] epoch={epoch} score={report["selection_score"]:.6f}',flush=True)
        atomic_json(args.output_dir/f'epoch_{epoch:03d}.json',record)
        save(args,model,optimizer,last,epoch,best,record)
    atomic_json(args.output_dir/'training_complete.json',{'epochs':args.epochs,'best_score':best})


def evaluate(args,model):
    restore(args,model,args.output_dir/'best.pth')
    cases=validation(args,model,True);curve=[]
    for threshold in np.round(np.arange(.35,.751,.01),2):
        row=point_metrics(cases,'p1',float(threshold),args.manifest);curve.append(row)
        print(f'[SWEEP {args.variant}] {threshold:.2f}',flush=True)
    atomic_json(args.output_dir/'eval/val/threshold_curve.json',curve)
    best=max(curve,key=lambda row:row['selection_score'])
    atomic_json(args.output_dir/'eval/val_selection.json',{'selected_on':'val','checkpoint_sha256':digest(args.output_dir/'best.pth'),
        'threshold':best['threshold'],'report':best,'selection_rule':SELECTION_RULE,'reference_sha256':args.config['reference_sha256']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','all','train','val','smoke'))
    parser.add_argument('--variant',choices=('A','B'),required=True)
    parser.add_argument('--pair_dir',type=Path,required=True)
    parser.add_argument('--h0_checkpoint',type=Path,default=Path(DEFAULT_H0))
    parser.add_argument('--repair_run',type=Path,default=Path(DEFAULT_REPAIR))
    parser.add_argument('--root_path',type=Path,default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    parser.add_argument('--device',default='cuda');parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--num_workers',type=int,default=4);parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--h0_lr',type=float,default=5e-5)
    parser.add_argument('--b_run_name',default='B',help='Separate B output directory; existing A is reused')
    parser.add_argument('--transport_rate_init',type=float)
    parser.add_argument('--proposal_rate_init',type=float)
    args=parser.parse_args();args.device=torch.device(args.device)
    if args.epochs<1: parser.error('epochs must be positive')
    if not re.fullmatch(r'B(?:_[A-Za-z0-9_.-]+)?',args.b_run_name): parser.error('Invalid B directory name')
    if (args.transport_rate_init is None)!=(args.proposal_rate_init is None): parser.error('Provide both initial budgets')
    if args.transport_rate_init is not None and not (0<args.transport_rate_init<1 and 0<args.proposal_rate_init<1 and args.transport_rate_init+args.proposal_rate_init<1):
        parser.error('Initial budgets must be positive and sum to less than one')
    args.output_dir=args.pair_dir/(args.variant if args.variant=='A' else args.b_run_name)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed)
    if args.command=='prepare':reference(args);return
    if not (args.pair_dir/'reference/complete.json').is_file(): raise ValueError('Run shared prepare first')
    args.manifest=json.loads((args.pair_dir/'reference/fixed_gaps_and_anchors.json').read_text())
    base,h0,saved=load_h0(args.h0_checkpoint,args.device)
    source_cfg=None;meta=None
    if args.variant=='A':model=FullH0Control(base)
    else:
        source,q,meta=inspect_source(args.repair_run);source_cfg=source['config']
        model=FullH0Joint(base,q,source,transport_rate_init=args.transport_rate_init,
                          proposal_rate_init=args.proposal_rate_init).to(args.device)
        del source,q
    criterion=original_criterion(saved,args.device)
    args.config=checkpoint_config(args,h0,saved,meta,source_cfg)
    del h0;gc.collect()
    contract={k:args.config[k] for k in ('h0_sha256','epochs','seed','root_path','batch_size','num_workers','val_interval','h0_lr','selection_rule','reference_sha256')}
    # Created once by shared prepare under the shell's pair-level file lock.
    shared=args.pair_dir/'pair_contract.json'
    if not shared.is_file() or json.loads(shared.read_text())!=contract: raise ValueError('A/B training settings differ')
    if args.variant=='B' and args.b_run_name!='B': check_reusable_A(args.pair_dir,args.config)
    marker=args.output_dir/'experiment.json'
    if marker.is_file() and json.loads(marker.read_text())!=args.config: raise ValueError('Experiment configuration changed')
    atomic_json(marker,args.config)
    rates={k:v['lr'] for k,v in (source_cfg or {}).get('optimizer',{}).items()}
    optimizer=parameter_groups(model,args.h0_lr,repair_rates=rates)
    atomic_json(args.output_dir/'parameter_groups.json',{group['name']:{'parameters':sum(p.numel() for p in group['params']),
        'initial_lr':group['initial_lr'],'minimum_lr':group['minimum_lr']} for group in optimizer.param_groups})
    if args.command in ('all','train','smoke') and not (args.output_dir/'last.pth').is_file(): smoke(args,model,criterion)
    if args.command=='smoke':return
    if args.command in ('all','train'):train(args,model,criterion,optimizer)
    if args.command in ('all','val'):evaluate(args,model)
    print('[DONE] '+str(args.output_dir),flush=True)


if __name__=='__main__':main()
