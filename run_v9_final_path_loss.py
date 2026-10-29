"""Continue V9 with train-only candidate labels and final-surface path losses."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_reliable_v9_mass as v9
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.connection_paths import PathConfig
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_final_path_supervision import predicted_candidates,connection_loss,weighted_connection
from path_connection_protocol import atomic_json,infer_cases,compare_cases,visuals

VERSION='v9_final_surface_path_loss_v1'


def load_original(args):
    source=args.v9_run/'stage_D/best.pth'; selection=args.v9_run/'direct_val_selection.json'
    if not source.is_file() or not selection.is_file(): raise FileNotFoundError('Missing original V9 best.pth or direct val selection')
    original=torch.load(source,map_location='cpu',weights_only=False); chosen=json.loads(selection.read_text())
    if original['identity'].get('initialization')!='direct_random_downstream': raise ValueError('Expected the direct_random_downstream V9')
    if original['identity']!=chosen['identity'] or chosen.get('selected_on')!='val' or Path(chosen['checkpoint']).resolve()!=source.resolve():
        raise ValueError('Original checkpoint and validation selection do not match')
    config=original['training_config']; reference=float(config['reference_threshold'])
    if Path(config['root_path']).resolve()!=args.root_path.resolve() or args.seed!=config['seed'] or args.batch_size!=config['batch_size']:
        raise ValueError('Preserve original dataset root, seed, and physical batch size')
    if reference!=.45 or chosen['report']['paired']['eligible_p0_gaps']!=206:
        raise ValueError('Expected fixed P0 .45 and the original 206 validation reference gaps')
    model,baseline,identity=v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(original['identity']['q3']['path']),device=args.device))
    v9.restore_checkpoint(model,source,identity,'D')
    model.v9_training_config=dict(config); v9.LR.update(original['learning_rates'])
    for key in ('lambda_bg','lambda_missing','lambda_cont','missing_threshold'): baseline[key]=config[key]
    return model,baseline,identity,original,source,chosen


def run_config(args,identity,source):
    candidate=PathConfig(max_endpoints=args.max_endpoints,max_pairs=args.max_pairs,max_paths=args.max_paths)
    config={'version':VERSION,'original_v9':str(source.resolve()),'original_v9_sha256':v9.digest(source),
        'upstream_identity':identity,'root_path':str(args.root_path.resolve()),'candidate':candidate.__dict__,
        'epochs':args.epochs,'batch_size':args.batch_size,'seed':args.seed,'num_workers':args.num_workers,
        'validation_interval':args.val_interval,'weights':{'pos':args.lambda_pos,'neg':args.lambda_neg,
        'bg':args.lambda_bg_path,'rank':args.lambda_rank},'tau':args.tau,'margin':args.margin,
        'rank_temperature':args.rank_temperature,'max_rank_pairs':args.max_rank_pairs,
        'parameters':'R/W/G/rates/B/A and existing visual/gate modules use original V9 stage-D learning rates; BASE/Q3 frozen',
        'protocol':'native1500, one random512 crop per image per epoch, physical batch2, accumulation1, fp32, no EMA',
        'note':'GT is used only for training path labels/sampling. L_cont is preserved; overlap with positive path pixels is recorded. A frozen-original comparison is not a matched-continuation ablation.',
        'code_sha256':{name:v9.digest(Path(__file__).parent/name) for name in ('run_v9_final_path_loss.py',
            'networks/v9_final_path_supervision.py','networks/connection_paths.py')}}
    return json.loads(json.dumps(config)),candidate


def save(path,model,identity,epoch,best,optimizer,config):
    payload={'version':v9.VERSION,'identity':identity,'stage':'D','epoch':epoch,'score':best,
        'modules':{name:getattr(model,name).state_dict() for name in v9.MODULE_NAMES},
        'gate_relation':model.gate_relation.state_dict(),'state_gate':model.state_gate.state_dict(),
        'training_config':model.v9_training_config,'learning_rates':dict(v9.LR),
        'path_loss_version':VERSION,'path_loss_config':config,'torch_rng':torch.get_rng_state(),
        'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if optimizer is not None: payload['optimizer']=optimizer.state_dict()
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
    torch.save(payload,temp); temp.replace(path)


def dataset(args,augment):
    return RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=augment)


def measure_gradients(loss,model):
    values={}
    if not loss.requires_grad: return {name:0. for name in v9.STAGE_MODULES['D']}
    names=v9.STAGE_MODULES['D']; params=[tuple(v9.module(model,name).parameters()) for name in names]
    flattened=[parameter for group in params for parameter in group]
    grads=torch.autograd.grad(loss,flattened,retain_graph=True,allow_unused=True)
    offset=0
    for name,group in zip(names,params):
        pieces=[gradient.detach().float().square().sum() for gradient in grads[offset:offset+len(group)] if gradient is not None]
        values[name]=float(torch.stack(pieces).sum().sqrt()) if pieces else 0.; offset+=len(group)
    return values


def batch_losses(model,batch,baseline,args,candidate,seed,diagnose_gradient=False):
    image,mask,skeleton=(batch[name].to(args.device) for name in ('image','mask','skeleton'))
    result=model(image,capture_diagnostics=True)
    base,base_terms=dynamic_losses(result,mask,skeleton,lambda_bg=baseline['lambda_bg'],
        lambda_missing=baseline['lambda_missing'],lambda_cont=baseline['lambda_cont'],
        missing_threshold=baseline['missing_threshold'],return_live_terms=True)
    paths,pairs,candidate_stats,lengths=predicted_candidates(model,result,mask,skeleton,
        baseline['missing_threshold'],candidate,args.max_paths,args.max_rank_pairs,seed)
    terms,counts=connection_loss(result['p1'],paths,pairs,args.tau,args.margin,args.rank_temperature)
    weights={'pos':args.lambda_pos,'neg':args.lambda_neg,'bg':args.lambda_bg_path,'rank':args.lambda_rank}
    conn=weighted_connection(terms,weights); total=base+conn
    if not bool(torch.isfinite(total)) or not bool(torch.isfinite(result['p1']).all()): raise RuntimeError('Nonfinite final P1 or loss')
    needs=({'total':True,'pos':True,'neg':True,'bg':True,'rank':True} if diagnose_gradient is True
           else diagnose_gradient if isinstance(diagnose_gradient,dict) else {})
    grad=measure_gradients(conn,model) if needs.get('total') and (counts['positive'] or counts['negative']) else None
    active={'pos':counts['positive'],'neg':counts['negative'],'bg':counts['negative'],'rank':counts['matched_pairs']}
    term_gradients={key:measure_gradients(float(weights[key])*terms[key],model)
                    if needs.get(key) and active[key] and weights[key]>0 else None
                    for key in ('pos','neg','bg','rank')}
    return total,{'total':float(total.detach()),'seg':float(base_terms['seg']),'cont':float(base_terms['cont']),
        **{'connection_'+key:float(value.detach()) for key,value in terms.items()},
        'connection_weighted':float(conn.detach()),'candidate_counts':candidate_stats,
        'connection_counts':counts,'length_counts':lengths,'connection_gradient_norms':grad,
        'connection_term_gradient_norms':term_gradients}


def smoke(args,model,baseline,identity,candidate,config=None):
    ds=dataset(args,False); ds.set_epoch(1)
    # Save the complete trainable state and RNG so this real-batch probe never changes training initialization.
    original={name:{key:value.detach().clone() for key,value in getattr(model,name).state_dict().items()}
              for name in (*v9.MODULE_NAMES,'gate_relation','state_gate')}
    torch_rng=torch.get_rng_state(); np_rng=np.random.get_state()
    cuda_rng=torch.cuda.get_rng_state_all() if args.device.type=='cuda' else None
    start=time.perf_counter(); optimizer=v9.configure(model,'D'); model.train(); model.set_accept_floor(0.)
    if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
    try:
        batch=next(iter(DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)))
        optimizer.zero_grad(set_to_none=True)
        loss,report=batch_losses(model,batch,baseline,args,candidate,args.seed,True)
        loss.backward()
        if any(p.grad is not None for p in model.frozen.parameters()): raise RuntimeError('BASE/Q3 unexpectedly received gradients')
        norms={name:v9.grad_norm(v9.module(model,name)) for name in v9.STAGE_MODULES['D']}
        if any(not np.isfinite(x) for x in norms.values()): raise RuntimeError('Nonfinite model gradient')
        optimizer.step()
        report.update({'passed':True,'path_loss_config':config,'identity':identity,
            'physical_batch':len(batch['image']),'total_gradient_norms':norms,
            'seconds':time.perf_counter()-start,'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None,
            'note':'One real training batch; if no labeled candidates the connection gradient is unavailable and later batches are monitored.'})
    finally:
        for name,state in original.items(): getattr(model,name).load_state_dict(state,strict=True)
        model.zero_grad(set_to_none=True); model.eval(); torch.set_rng_state(torch_rng); np.random.set_state(np_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
    atomic_json(args.output_dir/'smoke.json',report); print('[SMOKE] '+json.dumps(report),flush=True)


def train(args,model,baseline,identity,source,config,candidate):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    output=args.output_dir; output.mkdir(parents=True,exist_ok=True)
    saved_config=output/'config.json'
    if saved_config.is_file() and json.loads(saved_config.read_text())!=config:
        raise ValueError('Existing run has different loss, dataset, source checkpoint or code')
    atomic_json(saved_config,config)
    smoke_path=output/'smoke.json'
    probe=json.loads(smoke_path.read_text()) if smoke_path.is_file() else {}
    if not probe.get('passed') or probe.get('path_loss_config')!=config or probe.get('identity')!=identity:
        smoke(args,model,baseline,identity,candidate,config)
    ds=dataset(args,True); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,
        num_workers=args.num_workers,pin_memory=args.device.type=='cuda')
    optimizer=v9.configure(model,'D'); start=1; best=-float('inf'); last=output/'last.pth'
    if last.is_file():
        state=v9.restore_checkpoint(model,last,identity,'D')
        if state.get('path_loss_version')!=VERSION or state.get('path_loss_config')!=config or 'optimizer' not in state:
            raise ValueError('Resume checkpoint does not match this path-loss run')
        optimizer.load_state_dict(state['optimizer']); torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['epoch']+1; best=state['score']; print(f'[RESUME] epoch {start}/{args.epochs}',flush=True)
    for epoch in range(start,args.epochs+1):
        model.train(); model.set_accept_floor(0.); ds.set_epoch(epoch)
        sums=Counter(); counts=Counter(); bins=Counter(); gradients=None
        term_gradients={key:None for key in ('pos','neg','bg','rank')}; used=0
        for step,batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            needs={'total':gradients is None,**{key:value is None for key,value in term_gradients.items()}}
            loss,report=batch_losses(model,batch,baseline,args,candidate,args.seed+epoch*100000+step,needs)
            if report['connection_gradient_norms'] is not None and gradients is None: gradients=report['connection_gradient_norms']
            for key,value in report['connection_term_gradient_norms'].items():
                if value is not None and term_gradients[key] is None: term_gradients[key]=value
            loss.backward()
            if any(p.grad is not None for p in model.frozen.parameters()): raise RuntimeError('Frozen BASE/Q3 received gradients')
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()): raise RuntimeError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            optimizer.step(); used+=1
            for key in ('total','seg','cont','connection_pos','connection_neg','connection_bg','connection_rank','connection_weighted'): sums[key]+=report[key]
            counts.update(report['candidate_counts']); counts.update({'used_'+k:v for k,v in report['connection_counts'].items()})
            bins.update(report['length_counts'])
            if step%25==0: print(f'[PATH LOSS TRAIN] epoch={epoch}/{args.epochs} batch={step+1}/{len(loader)} '
                f'positive={report["connection_counts"]["positive"]} negative={report["connection_counts"]["negative"]} '
                f'paired={report["connection_counts"]["matched_pairs"]} total={report["total"]:.5f}',flush=True)
        record={'epoch':epoch,'steps':used,'loss_mean':{key:value/max(used,1) for key,value in sums.items()},
            'counts':dict(counts),'length_counts':dict(bins),'connection_gradient_norms_first_labeled_batch':gradients,
            'connection_term_gradient_norms_first_eligible_batch':term_gradients,
            'loss_overlap_note':'Positive GT-path samples overlapping the original missing skeleton are counted; original L_cont is unchanged.'}
        if epoch%args.val_interval==0 or epoch==args.epochs:
            reports,_,_,_=evaluate_thresholds(model,full_dataset(args.root_path,'val'),args.device,(.6,),reference_threshold=.45)
            val=reports[.6]; score=float(val['selection_score']); record['val_score_at_0_60']=score
            atomic_json(output/f'val_epoch_{epoch:03d}.json',val)
            if score>best:
                best=score; save(output/'best.pth',model,identity,epoch,best,None,config)
            print(f'[VAL] epoch={epoch} score_at_0.60={score:.6f}',flush=True)
        save(last,model,identity,epoch,best,optimizer,config)
        with (output/'epochs.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
        print('[EPOCH] '+json.dumps(record,allow_nan=False),flush=True)
    if v9.digest(source)!=config['original_v9_sha256']: raise RuntimeError('Original V9 checkpoint changed')
    atomic_json(output/'training_complete.json',{'epochs':args.epochs,'best_score_at_0_60':best,
        'best_checkpoint':str((output/'best.pth').resolve()),'selection_split':'val'})


def compare(args,model,identity,source):
    from run_q3_r3_mass import full_dataset
    best=args.output_dir/'best.pth'
    if not best.is_file(): raise FileNotFoundError('Training must produce best.pth before validation')
    dataset_val=full_dataset(args.root_path,'val')
    if len(dataset_val)!=14: raise ValueError('Expected 14 original validation images')
    cases=[]
    for name,checkpoint in (('A_original',source),('P1_path',best)):
        state=v9.restore_checkpoint(model,checkpoint,identity,'D')
        if name=='P1_path' and state.get('path_loss_version')!=VERSION: raise ValueError('Wrong path-loss checkpoint')
        model.eval(); cache=args.output_dir/'predictions_val'/name; marker=cache/'checkpoint.json'
        expected={'sha256':v9.digest(checkpoint),'identity':identity}
        if marker.is_file() and json.loads(marker.read_text())!=expected: raise ValueError(f'Prediction cache uses another checkpoint: {cache}')
        atomic_json(marker,expected)
        values,timing=infer_cases(model,dataset_val,args.device,output=cache,variant=name)
        atomic_json(args.output_dir/f'{name}_inference_cost.json',timing)
        if name=='A_original': cases=[{**entry,'P0_surface':entry['p0']} for entry in values]
        else:
            for original,repaired in zip(cases,values):
                if original['case']!=repaired['case'] or not np.array_equal(original['p0'],repaired['p0']):
                    raise ValueError('The original and path-loss models do not share the same P0')
                original[name]=repaired[name]
    output=args.output_dir/'comparison_val'
    reports=compare_cases(cases,['A_original','P0_surface','P1_path'],.45,206,output,args.reference_set)
    visuals(cases,'P1_path',reports['P1_path']['own_val_threshold']['threshold'],args.output_dir/'visuals_val')
    print(f'[VALIDATION DONE] {output / "REPORT.md"}',flush=True)


def test(args,model,identity,source):
    from run_q3_r3_mass import full_dataset
    from diagnostics.v91_counterfactual import evaluate
    selection=args.output_dir/'comparison_val/comparison.json'
    if not selection.is_file(): raise FileNotFoundError('Run validation first to freeze test thresholds')
    frozen=json.loads(selection.read_text()); output=args.output_dir/'test_final'; output.mkdir(parents=True,exist_ok=True)
    points=frozen['points']; chosen={name:points[name] for name in ('A_original','P0_surface','P1_path')}
    atomic_json(output/'test_selection.json',{'selected_on':'val','points':chosen,'rule':frozen['selection_rule']})
    test_dataset=full_dataset(args.root_path,'test'); cases=[]
    for name,checkpoint in (('A_original',source),('P1_path',args.output_dir/'best.pth')):
        v9.restore_checkpoint(model,checkpoint,identity,'D'); model.eval()
        values,_=infer_cases(model,test_dataset,args.device,output=args.output_dir/'predictions_test'/name,variant=name)
        if name=='A_original': cases=[{**entry,'P0_surface':entry['p0']} for entry in values]
        else:
            for original,repaired in zip(cases,values):
                if original['case']!=repaired['case'] or not np.array_equal(original['p0'],repaired['p0']): raise ValueError('Test P0 mismatch')
                original[name]=repaired[name]
    reports=evaluate(cases,chosen,.45,None,output,modes=('A_original','P0_surface','P1_path'))
    atomic_json(output/'test_metrics.json',{'selected_on':'val','points':chosen,'reports':reports,
        'metric_note':'APLS is approximate; closure and background bridges are proxies.'})
    print(f'[TEST DONE] {output / "test_metrics.json"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','smoke','train','compare','test'))
    p.add_argument('--root_path',type=Path,required=True); p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--v9_run',type=Path,required=True); p.add_argument('--reference_set',type=Path)
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch_size',type=int,default=2); p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=1234); p.add_argument('--val_interval',type=int,default=5)
    p.add_argument('--max_endpoints',type=int,default=32); p.add_argument('--max_pairs',type=int,default=48)
    p.add_argument('--max_paths',type=int,default=128); p.add_argument('--max_rank_pairs',type=int,default=64)
    p.add_argument('--lambda_pos',type=float,default=.1); p.add_argument('--lambda_neg',type=float,default=.1)
    p.add_argument('--lambda_bg_path',type=float,default=.05); p.add_argument('--lambda_rank',type=float,default=.1)
    p.add_argument('--tau',type=float,default=.1); p.add_argument('--margin',type=float,default=.1); p.add_argument('--rank_temperature',type=float,default=.1)
    args=p.parse_args(); args.device=torch.device(args.device)
    if args.batch_size!=2 or min(args.epochs,args.val_interval,args.max_endpoints,args.max_pairs,args.max_paths,args.max_rank_pairs)<1 or args.num_workers<0:
        p.error('Requires physical batch2 and positive budgets/epochs')
    if args.tau<=0 or args.rank_temperature<=0 or min(args.lambda_pos,args.lambda_neg,args.lambda_bg_path,args.lambda_rank,args.margin)<0:
        p.error('Invalid path loss settings')
    if args.command in ('all','compare','test') and (args.reference_set is None or not args.reference_set.is_file()):
        p.error('Need the existing fixed 206-gap validation reference_set.json')
    args.output_dir.mkdir(parents=True,exist_ok=True); torch.manual_seed(args.seed); np.random.seed(args.seed)
    model,baseline,identity,_,source,_=load_original(args); config,candidate=run_config(args,identity,source)
    if args.command=='smoke': smoke(args,model,baseline,identity,candidate,config)
    if args.command in ('all','train'): train(args,model,baseline,identity,source,config,candidate)
    if args.command in ('all','compare'): compare(args,model,identity,source)
    if args.command=='test': test(args,model,identity,source)


if __name__=='__main__': main()
