"""Resumable staged candidate-relation and bounded nonlocal repair experiment."""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

import run_reliable_v9_mass as v9
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnostics.v92_connection_candidates import split_name
from networks.connection_paths import PathConfig
from networks.path_connection_repair import PathConnectionRepair
from networks.path_connection_losses import attach_labels,path_losses
from path_connection_protocol import (atomic_json,atomic_npz,cache_tile,infer_cases,binary_metrics,
    matched_indices,own_threshold,compare_cases,visuals)

VERSION='v9_candidate_connection_nonlocal_v1'
PHASE_NAMES={'B':'B_relation','C':'C_nonlocal','D':'D_path_losses','E':'E_negative_suppression','F':'F_joint'}


def final_test(args,model,identity,comparison):
    """Freeze choices on val before accessing test; never rematch test thresholds."""
    from run_q3_r3_mass import full_dataset
    from diagnostics.v91_counterfactual import evaluate
    candidates=['A_original','P0_surface']+[name for name in PHASE_NAMES.values() if name in comparison]
    chosen=max(candidates,key=lambda name:comparison[name]['own_val_threshold']['metrics']['iou'])
    selected_phase=next((phase for phase,name in PHASE_NAMES.items() if name==chosen),None)
    frozen={'selected_on':'val','selected_model':chosen,'identity':identity,'points':{name:{point:row['threshold'] for point,row in data.items()}
             for name,data in comparison.items()},'test_policy':'evaluate original V9 and val-selected model once; all thresholds and matching work points frozen on val'}
    target=args.output_dir/'final_test_selection.json'
    if target.is_file() and json.loads(target.read_text())!=frozen: raise ValueError('Frozen final-test selection already differs')
    atomic_json(target,frozen)
    # Reset original modules explicitly even if the joint phase just modified them.
    for name,state in model.initial_downstream.items(): getattr(model.base,name).load_state_dict(state,strict=True)
    dataset=full_dataset(args.root_path,'test')
    original,_=infer_cases(model.base,dataset,args.device,output=args.output_dir/'predictions_test/A_original',variant='A_original')
    cases=[{**case,'P0_surface':case['p0']} for case in original]; names=['A_original','P0_surface']
    if selected_phase and chosen!='B_relation':
        restore(model,args.output_dir/PHASE_NAMES[selected_phase]/'best.pth',identity)
        repaired,_=infer_cases(model,dataset,args.device,output=args.output_dir/'predictions_test'/chosen,variant=chosen)
        names.append(chosen)
        for base,value in zip(cases,repaired):
            if base['case']!=value['case'] or not np.array_equal(base['p0'],value['p0']): raise ValueError('Final test P0 changed')
            base[chosen]=value[chosen]
    points={name:frozen['points'][name] for name in names}
    output=args.output_dir/'test_final'; output.mkdir(parents=True,exist_ok=True)
    reports=evaluate(cases,points,args.reference,None,output,modes=names)
    atomic_json(output/'test_metrics.json',{'selection':frozen,'reports':reports,'proxy_warning':'approximate APLS and closure/bridge proxies'})


def load_model(args):
    checkpoint=args.v9_run/'stage_D/best.pth'
    selection=args.v9_run/'direct_val_selection.json'
    if not checkpoint.is_file() or not selection.is_file():
        raise FileNotFoundError(f'Requires existing V9 checkpoint and val selection under {args.v9_run}')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    selected=json.loads(selection.read_text())
    if payload['identity'].get('initialization')!='direct_random_downstream':
        raise ValueError('This experiment explicitly targets direct_random_downstream V9')
    if payload['identity']!=selected['identity'] or selected.get('selected_on')!='val' or Path(selected['checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('V9 checkpoint and validation selection identities differ')
    if Path(payload['training_config']['root_path']).resolve()!=args.root_path.resolve():
        raise ValueError('Dataset root differs from the trained V9')
    if payload['training_config']['seed']!=args.seed or payload['training_config']['batch_size']!=args.batch_size:
        raise ValueError('Preserve the V9 training seed and physical batch size')
    args.reference=float(payload['training_config']['reference_threshold'])
    args.expected_gaps=int(selected['report']['paired']['eligible_p0_gaps'])
    if args.reference!=.45 or args.expected_gaps!=206:
        raise ValueError('Expected the fixed P0 .45 / 206-gap validation protocol')
    base,source,identity=v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(payload['identity']['q3']['path']),device=args.device))
    v9.restore_checkpoint(base,checkpoint,identity,'D'); base.eval()
    for key in ('lambda_bg','lambda_missing','lambda_cont','missing_threshold'):
        source[key]=payload['training_config'][key]
    cfg=PathConfig(max_endpoints=args.max_endpoints,max_pairs=args.max_pairs,max_paths=args.max_paths)
    model=PathConnectionRepair(base,cfg,args.hidden).to(args.device); model.partition_seed=args.seed
    model.initial_downstream={name:{key:value.detach().cpu().clone() for key,value in state.items()}
        for name,state in downstream_state(model).items()}
    record={'version':VERSION,'v9_sha256':v9.digest(checkpoint),'v9_path':str(checkpoint.resolve()),
        'upstream_identity':identity,'candidate_protocol':asdict(cfg),'seed':args.seed,'hidden':args.hidden,
        'root_path':str(args.root_path.resolve()),'p0_reference':args.reference,'expected_val_reference_gaps':args.expected_gaps,
        'training_protocol':'native1500 -> one random512 crop/image/epoch, physical batch2 accumulation1, FP32, no EMA',
        'selection':'B uses val natural candidate AP; C/D/F use max global val IoU; test excluded',
        'training_holdout':'deterministic image-disjoint 20% of original train is reserved from PathRelationHead and subsequent repair training; V9 backbone previously saw original training split',
        'run_settings':{key:getattr(args,key) for key in ('epochs','batch_size','lr','val_interval','warmup_epochs',
            'lambda_relation','lambda_path','lambda_neg','min_long_coverage','min_long_gaps','min_relation_auc','min_class_count')},
        'code_sha256':{name:v9.digest(Path(__file__).parent/name) for name in (
            'run_path_connection_mass.py','path_connection_protocol.py','networks/connection_paths.py',
            'networks/path_connection_repair.py','networks/path_connection_losses.py')}}
    # JSON canonicalization turns tuple config into the same representation on resume.
    record=json.loads(json.dumps(record))
    manifest=args.output_dir/'experiment.json'
    if manifest.is_file() and json.loads(manifest.read_text())!=record:
        raise ValueError('Existing output belongs to a different model, code or candidate protocol')
    atomic_json(manifest,record)
    return model,source,record


def train_dataset(args,augment):
    ds=RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=augment)
    ds.set_epoch(1)
    return ds


@torch.no_grad()
def audit_train(args,model):
    directory=args.output_dir/'candidate_cache/train'; directory.mkdir(parents=True,exist_ok=True)
    dataset=train_dataset(args,False); model.configure('B'); model.eval()
    for i in range(len(dataset)):
        # Cache keys based on source order; manifest locks code/seed/data identity.
        target=directory/f'{i:06d}.npz'
        if target.is_file(): continue
        sample=dataset[i]; result=model(sample['image'][None].to(args.device),capture_diagnostics=True)
        cache_tile(model,result,sample['mask'][None].to(args.device),sample['skeleton'][None].to(args.device),
            str(sample['case_name']),'train',target.stem,directory,args.reference,
            (sample.get('tile_top',0),sample.get('tile_left',0)))
        print(f'[CANDIDATE AUDIT] {i+1}/{len(dataset)} paths={sum(map(len,result["paths"]))}',flush=True)
    return summarize_cache(directory,args.output_dir/'candidate_audit_train.json')


def summarize_cache(directory,output):
    classes=Counter(); bins=defaultdict(Counter); deviations=[]; images=set(); boundaries=0
    for file in sorted(Path(directory).glob('*.npz')):
        with np.load(file,allow_pickle=False) as data:
            meta=json.loads(str(data['meta'])); images.add(meta['case'])
            for label,subtype,deviation in zip(data['labels'],data['subtypes'],data['deviation_native']):
                classes[f'{int(label)}|{subtype}']+=1
                if label==1: deviations.append(float(deviation))
            for gap in meta['gaps']:
                if gap['boundary_excluded']: boundaries+=1; continue
                row=bins[gap['length_bin']]; row['eligible']+=1
                row['any_hit']+=int(gap['any_hit']); row['full_coverage']+=int(gap['full_coverage'])
                row['coverage_sum']+=gap['coverage_fraction']
    report={'split':'train','images':len(images),'native_full_image_census':False,
        'sample_protocol':'one seeded native512 crop per training image, crop epoch1, unaugmented; boundary gaps excluded',
        'labels':dict(classes),'by_length':{key:{**dict(value),'full_coverage_rate':value['full_coverage']/max(value['eligible'],1),
            'mean_coverage':value['coverage_sum']/max(value['eligible'],1)} for key,value in bins.items()},
        'boundary_gap_exclusions':boundaries,'positive_path_mean_deviation_native':float(np.mean(deviations)) if deviations else None,
        'candidate_labels':'native-coordinate GT snap/path geometry proxies, uncertain=-1 excluded from classification',
        'limits':'These are sampled-crop candidate discovery rates, not a complete stitched native-image census. Candidate budgets bound recall. Positive GT paths are never used for proposal generation.'}
    atomic_json(output,report); return report


def read_cache(directory,partition=None):
    result={key:[] for key in ('features','labels','strata','subtypes','lengths_native','source_confidence','endpoint_p0','distance_work','baseline_gate_scores')}; cases=[]
    for path in sorted(Path(directory).glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['meta']))
            if partition is not None and meta['partition']!=partition: continue
            for key in result: result[key].append(data[key].copy())
            cases.extend([meta['case']]*len(data['labels']))
    if not result['features']: raise ValueError(f'No candidate cache files in {directory}, partition={partition}')
    return {**{key:np.concatenate(value) for key,value in result.items()},'cases':np.asarray(cases,dtype='U128')}


@torch.no_grad()
def head_report(model,data,device,seed=1234):
    scores=[]
    for start in range(0,len(data['labels']),4096):
        features=torch.from_numpy(data['features'][start:start+4096]).to(device)
        scores.extend(model.path_head(features).sigmoid().cpu().tolist())
    scores=np.asarray(scores); y=data['labels']; valid=y>=0
    matched=matched_indices(y,data['strata'],seed)
    report={'natural':binary_metrics(y,scores),'matched':binary_metrics(y[matched],scores[matched]),
        'original_pixel_G_path_mean_proxy':binary_metrics(y,data['baseline_gate_scores']),
        'images':len(set(data['cases'])),'uncertain':int((~valid).sum()),
        'by_subtype':{},'by_length':{},'sample_protocol':'natural candidate population; matched length/source-R/endpoint-P0/separation strata',
        'label_warning':'classification of prediction-generated GT-audited proxy labels, not certified true road graphs'}
    report['matching_quality']={}
    for key in ('lengths_native','source_confidence','endpoint_p0','distance_work'):
        value=data[key]; value=value.mean(1) if value.ndim==2 else value
        positive=value[matched[y[matched]==1]]; negative=value[matched[y[matched]==0]]
        report['matching_quality'][key]={'positive_mean':float(positive.mean()) if len(positive) else None,
            'negative_mean':float(negative.mean()) if len(negative) else None,
            'mean_absolute_distribution_shift':abs(float(positive.mean()-negative.mean())) if min(len(positive),len(negative)) else None}
    for subtype in sorted(set(data['subtypes'])):
        # Each negative subtype vs all positives, preserving those samples' natural counts.
        keep=(data['subtypes']==subtype)|(y==1)
        report['by_subtype'][str(subtype)]=binary_metrics(y[keep],scores[keep])
    for name,lower,upper in (('3-7',3,8),('8-15',8,16),('16-31',16,32),('32+',32,float('inf'))):
        keep=(data['lengths_native']>=lower)&(data['lengths_native']<upper)
        report['by_length'][name]=binary_metrics(y[keep],scores[keep])
    return report


def downstream_state(model):
    return {**{name:getattr(model.base,name).state_dict() for name in v9.MODULE_NAMES},
            'gate_relation':model.base.gate_relation.state_dict(),'state_gate':model.base.state_gate.state_dict()}


def save(args,model,phase,epoch,score,optimizer,identity,path):
    payload={'version':VERSION,'identity':identity,'phase':phase,'epoch':epoch,'score':score,
        'path_head':model.path_head.state_dict(),'long_gain':model.long_gain.detach().cpu(),
        'downstream':downstream_state(model) if phase=='F' else None,
        'optimizer':optimizer.state_dict() if optimizer else None,'torch_rng':torch.get_rng_state(),
        'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device.type=='cuda' else None,
        'config':vars_for_save(args)}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
    torch.save(payload,temp); temp.replace(path)


def restore(model,path,identity):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('version')!=VERSION or payload.get('identity')!=identity:
        raise ValueError(f'Path experiment checkpoint identity mismatch: {path}')
    model.path_head.load_state_dict(payload['path_head'],strict=True)
    with torch.no_grad(): model.long_gain.copy_(payload['long_gain'])
    if payload.get('downstream'):
        expected=set(v9.MODULE_NAMES)|{'gate_relation','state_gate'}
        if set(payload['downstream'])!=expected: raise ValueError('Joint downstream state is incomplete')
        for name in expected: getattr(model.base,name).load_state_dict(payload['downstream'][name],strict=True)
    elif hasattr(model,'initial_downstream'):
        for name,state in model.initial_downstream.items(): getattr(model.base,name).load_state_dict(state,strict=True)
    model.configure(payload['phase']); model.eval()
    return payload


def vars_for_save(args):
    return {key:(str(value) if isinstance(value,(Path,torch.device)) else value) for key,value in vars(args).items()
            if key not in ('command','phase')}


def finite_step(model,loss,optimizer):
    if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite path training loss')
    if loss.requires_grad:
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError('Nonfinite parameter gradient')
        if any(p.grad is not None for p in model.base.frozen.parameters()): raise RuntimeError('Frozen BASE/Q3 received gradients')
        if not model.joint and any(p.grad is not None for p in model.base.parameters()): raise RuntimeError('Frozen V9 received gradients')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
        optimizer.step()


def gradient_summary(model):
    modules={'path_head':model.path_head,'R':model.base.reliable_head,'W':model.base.relations,
             'G':model.base.gate_relation,'state_gate':model.base.state_gate,'rates':model.base.dynamics.rates}
    return {name:v9.grad_norm(module) for name,module in modules.items()}


def train_phase(args,model,source,identity,phase,val_dataset):
    directory=args.output_dir/PHASE_NAMES[phase]; directory.mkdir(parents=True,exist_ok=True)
    complete=directory/'complete.json'
    if complete.is_file():
        restore(model,directory/'best.pth',identity); print(f'[RESUME] completed {phase}',flush=True); return
    optimizer=torch.optim.AdamW(model.configure(phase,args.lr),weight_decay=.01)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    begin=1; best=-float('inf'); last=directory/'last.pth'
    if last.is_file():
        resume=restore(model,last,identity)
        old={k:v for k,v in resume['config'].items() if k!='epochs'}
        new={k:v for k,v in vars_for_save(args).items() if k!='epochs'}
        if old!=new: raise ValueError('Resume training settings differ')
        model.configure(phase,args.lr); optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['torch_rng']); np.random.set_state(resume['numpy_rng'])
        if resume['cuda_rng'] is not None: torch.cuda.set_rng_state_all(resume['cuda_rng'])
        begin=resume['epoch']+1; best=resume['score']
    if phase=='B':
        train=read_cache(args.output_dir/'candidate_cache/train','train')
        val=read_cache(args.output_dir/'candidate_cache/val')
        heldout=read_cache(args.output_dir/'candidate_cache/train','heldout')
    else:
        dataset=train_dataset(args,True)
        # The immutable per-image train cache supplies the exact source partition.
        indices=[]
        for i in range(len(dataset)):
            with np.load(args.output_dir/f'candidate_cache/train/{i:06d}.npz',allow_pickle=False) as data:
                meta=json.loads(str(data['meta']))
            if meta['partition']=='train': indices.append(i)
        loader=DataLoader(Subset(dataset,indices),batch_size=args.batch_size,shuffle=True,drop_last=True,
                          num_workers=args.num_workers,pin_memory=args.device.type=='cuda')
    for epoch in range(begin,args.epochs+1):
        model.train(); model.base.set_accept_floor(0.); sums=Counter(); steps=0
        if phase=='B':
            chosen=matched_indices(train['labels'],train['strata'],args.seed+epoch)
            if len(chosen)<2: raise RuntimeError('No matched relation candidates; candidate generation needs correction')
            chosen=np.random.permutation(chosen)
            for start in range(0,len(chosen),256):
                ids=chosen[start:start+256]; optimizer.zero_grad(set_to_none=True)
                logits=model.path_head(torch.from_numpy(train['features'][ids]).to(args.device))
                labels=torch.from_numpy(train['labels'][ids].astype(np.float32)).to(args.device)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
                finite_step(model,loss,optimizer); sums['relation']+=float(loss.detach()); steps+=1
        else:
            dataset.set_epoch(epoch)
            for step,batch in enumerate(loader):
                # Warm the new correction from exactly zero without changing local alpha.
                model.long_ramp=min(1.,((epoch-1)+(step/max(len(loader),1)))/max(args.warmup_epochs,1)) if phase in ('C','D') else 1.
                optimizer.zero_grad(set_to_none=True)
                image,mask,skel=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
                result=model(image,capture_diagnostics=True)
                labels,_=attach_labels(result,mask,skel,args.reference,model.config)
                loss,terms=path_losses(result,mask,skel,labels,source,phase,args.seed+epoch*len(loader)+step,
                    args.lambda_relation,args.lambda_path,args.lambda_neg)
                finite_step(model,loss,optimizer)
                for key,value in terms.items(): sums[key]+=value
                steps+=1
                if step%25==0: print(f'[PATH TRAIN] phase={phase} epoch={epoch}/{args.epochs} batch={step+1}/{len(loader)} loss={float(loss.detach()):.5f}',flush=True)
        model.long_ramp=0. if phase=='B' else 1.; model.eval()
        record={'phase':phase,'epoch':epoch,'steps':steps,**{key:value/max(steps,1) for key,value in sums.items()},
                'gradients_last_batch':gradient_summary(model),'lambda_long':float(model.long_gain.sigmoid()) if phase!='B' else 0.}
        checked=(epoch%args.val_interval==0 or epoch==args.epochs)
        if checked:
            if phase=='B':
                record['val_classification']=head_report(model,val,args.device,args.seed)
                score=record['val_classification']['natural']['pr_auc']
                if score is None: raise RuntimeError('Val candidates do not include both relation classes')
                record['train_image_holdout']=head_report(model,heldout,args.device,args.seed)
            else:
                cases,_=infer_cases(model,val_dataset,args.device,reference=args.reference)
                threshold,curve=own_threshold(cases,'model'); score=curve[threshold]['iou']
                record['val_global_iou']=score; record['val_threshold']=threshold
            if score>best:
                best=float(score); save(args,model,phase,epoch,best,None,identity,directory/'best.pth')
        # Last carries the best score, plus optimizer and RNG, so interruptions resume at epoch boundaries.
        save(args,model,phase,epoch,best,optimizer,identity,last)
        with (directory/'epochs.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
        print('[PATH EPOCH] '+json.dumps(record,allow_nan=False),flush=True)
    restore(model,directory/'best.pth',identity)
    atomic_json(complete,{'phase':phase,'epochs':args.epochs,'best_score':best,'selection_split':'val'})


def smoke(args,model,source,identity):
    dataset=train_dataset(args,False); samples=[dataset[i] for i in range(min(args.batch_size,len(dataset)))]
    image=torch.stack([sample['image'] for sample in samples]).to(args.device)
    mask=torch.stack([sample['mask'] for sample in samples]).to(args.device)
    skel=torch.stack([sample['skeleton'] for sample in samples]).to(args.device)
    original={k:v.detach().clone() for k,v in model.path_head.state_dict().items()}; gain=model.long_gain.detach().clone()
    original_local={name:{key:value.detach().cpu().clone() for key,value in state.items()}
                    for name,state in downstream_state(model).items()}
    model.configure('B'); model.eval()
    if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
    start=time.perf_counter(); inference_times={}
    with torch.no_grad():
        result=model(image,capture_diagnostics=True)
        if not torch.equal(result['z1'],result['v9_z1']): raise RuntimeError('lambda_long=0 does not exactly reproduce V9')
        for name,forward in (('v9',lambda:model.base(image[:1],capture_diagnostics=False)),
                             ('candidate_classifier',lambda:model(image[:1],capture_diagnostics=True))):
            if args.device.type=='cuda': torch.cuda.synchronize(args.device)
            begin=time.perf_counter(); forward()
            if args.device.type=='cuda': torch.cuda.synchronize(args.device)
            inference_times[name]=time.perf_counter()-begin
    reports={}
    for phase in ('B','C','D','F'):
        optimizer=torch.optim.AdamW(model.configure(phase,args.lr),weight_decay=.01)
        model.long_ramp=0. if phase=='B' else 1.; optimizer.zero_grad(set_to_none=True)
        result=model(image,capture_diagnostics=True); labels,_=attach_labels(result,mask,skel,args.reference,model.config)
        loss,terms=path_losses(result,mask,skel,labels,source,phase,args.seed,args.lambda_relation,args.lambda_path,args.lambda_neg)
        finite_step(model,loss,optimizer)
        reports[phase]={'loss':float(loss.detach()),'counts':terms,'gradient_norms':gradient_summary(model),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'nonlocal_delta_abs_mean':float(result['long_delta'].detach().abs().mean()),
            'state_range':[float(result['u_final'].min().detach()),float(result['u_final'].max().detach())]}
    model.path_head.load_state_dict(original,strict=True)
    for name,state in original_local.items(): getattr(model.base,name).load_state_dict(state,strict=True)
    with torch.no_grad(): model.long_gain.copy_(gain)
    model.zero_grad(set_to_none=True); model.configure('B'); model.eval()
    report={'passed':True,'source_identity':identity,'physical_batch':len(samples),'zero_strength_exact_identity':True,
        'tensor_shapes':{key:list(result[key].shape) for key in ('p0','u0','u3','u_final','z1','r','w','q')},
        'inference_seconds_single_tile':inference_times,'timing_note':'single illustrative timing per variant, not a warmed multi-repeat benchmark',
        'phases':reports,'parameter_counts':{'path_head':sum(p.numel() for p in model.path_head.parameters()),
         'new_repair_parameters':1,'original_v9':sum(p.numel() for p in model.base.parameters()),
         'frozen_BASE_Q3':sum(p.numel() for p in model.base.frozen.parameters())},
        'seconds':time.perf_counter()-start,'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None,
        'warning':'A real batch with no matched labels or no executable nonlocal action may have zero head gradients; explicit synthetic invariants are covered by checks.test_path_connection. The full candidate/val gates remain mandatory.'}
    atomic_json(args.output_dir/'smoke.json',report); return report


def phase_cases(args,model,identity,phase,dataset):
    name=PHASE_NAMES[phase]; ckpt=args.output_dir/name/'best.pth'
    restore(model,ckpt,identity); model.long_ramp=0. if phase=='B' else 1.
    output=args.output_dir/'predictions_val'/name
    marker=output/'checkpoint.json'; expected={'sha256':v9.digest(ckpt),'identity':identity}
    if marker.is_file() and json.loads(marker.read_text())!=expected:
        raise ValueError(f'Prediction cache belongs to a different checkpoint: {output}')
    atomic_json(marker,expected)
    features=args.output_dir/'candidate_cache_val_phase'/name
    predictions,cost=infer_cases(model,dataset,args.device,output=output,cache_features=features,
                                 reference=args.reference,variant=name)
    if phase=='F':
        # Joint fine tuning changes R/W/semantic features: old cached descriptors
        # cannot stand in for current held-out-image predictions.
        heldout=args.output_dir/'candidate_cache_heldout_F'; heldout.mkdir(parents=True,exist_ok=True)
        train=train_dataset(args,False)
        for i in range(len(train)):
            with np.load(args.output_dir/f'candidate_cache/train/{i:06d}.npz',allow_pickle=False) as data:
                meta=json.loads(str(data['meta']))
            if meta['partition']!='heldout' or (heldout/f'{i:06d}.npz').is_file(): continue
            sample=train[i]
            with torch.no_grad(): result=model(sample['image'][None].to(args.device),capture_diagnostics=True)
            cache_tile(model,result,sample['mask'][None].to(args.device),sample['skeleton'][None].to(args.device),
                str(sample['case_name']),'train',f'{i:06d}',heldout,args.reference,(sample.get('tile_top',0),sample.get('tile_left',0)))
            print(f'[F HELDOUT] {i+1}/{len(train)}',flush=True)
        heldout_data=read_cache(heldout,'heldout')
    else: heldout_data=read_cache(args.output_dir/'candidate_cache/train','heldout')
    classification={'phase':phase,'checkpoint_sha256':expected['sha256'],
        'val':head_report(model,read_cache(features),args.device,args.seed),
        'train_image_holdout':head_report(model,heldout_data,args.device,args.seed)}
    atomic_json(args.output_dir/name/'relation_classification.json',classification)
    return predictions,cost


def final_report(args,comparisons,classifier,audit,status):
    lines=['# V9 candidate connection repair','',f'Pipeline status: {status}.','',
        'Results use direct_random_downstream V9 and its exact BASE/Q3, not an ablation of the original trained V8.',
        'All topology/closure/bridge results are proxies. Test has not been used for selection.','',
        f'Train crop candidate coverage: {json.dumps(audit.get("by_length",{}))}',
        f'Image-disjoint held-out training candidate classifier: {json.dumps(classifier.get("heldout",{}))}',
        f'Val candidate classifier: {json.dumps(classifier.get("val",{}))}','']
    if comparisons:
        base=comparisons['A_original']['matched_recall']; b=base['paired']; m=base['metrics']
        lines.append(f'Original V9 at matching target: closed={b["closed"]}/{b["eligible"]}, bridge={b["background_bridge_proxy"]}, bidirectional APLS={m["apls_bidirectional_approx"]:.5f}.')
        for phase in ('C','D','E','F'):
            name=PHASE_NAMES[phase]
            if name not in comparisons: continue
            row=comparisons[name]['matched_recall']; p=row['paired']; metric=row['metrics']
            balanced=p['closed']>b['closed'] and p['background_bridge_proxy']<=b['background_bridge_proxy'] and metric['apls_bidirectional_approx']>=m['apls_bidirectional_approx']
            lines.append(f'- {name}: matched-Recall delta closed={p["closed"]-b["closed"]}, bridge={p["background_bridge_proxy"]-b["background_bridge_proxy"]}, '
                f'bidirectional APLS={metric["apls_bidirectional_approx"]-m["apls_bidirectional_approx"]:+.5f}; simultaneous improvement={balanced}.')
        lines+=['','The per-stage comparison JSON contains exact matching errors, length bins, short components and added GT/background pixels.',
            'Relation-only B preserves P1 exactly; its usefulness is assessed by classification, not segmentation change.',
            'C vs B tests nonlocal reach; D vs C tests deduplicated path losses; E vs D is a frozen-weight suppression intervention; F tests joint R/W/G/rates fine tuning when enabled by validation evidence.']
    else: lines+=['','Later stages were not run because the candidate or classifier evidence gate failed. Inspect pipeline_state.json for the exact reason.']
    lines+=['','## Evidence limits for the research questions',
        '1. Feature sufficiency is measured by held-out-image candidate AP/AUC against the same natural and matched proxy-label populations, including the original G path-mean baseline.',
        '2. These A-F experiments isolate the listed additions. They do not by themselves isolate a pure relation-loss vs pixel-loss effect at identical architecture; no such causal claim is made.',
        '3. C vs B and the length-bin closure deltas measure whether nonlocal repair helps medium/long gaps; geometric coverage is not counted as actual closure.',
        '4. E vs D uses identical weights and is the controlled test of negative-candidate suppression.',
        '5. Simultaneous success requires more true closures, no more bridge proxies and higher or equal bidirectional approximate APLS at matched Recall; IoU and short components must also be inspected.']
    for phase,name in PHASE_NAMES.items():
        path=args.output_dir/name/'relation_classification.json'
        if path.is_file():
            data=json.loads(path.read_text())
            lines.append(f'- {name} current head: val={json.dumps(data["val"]["natural"])}, image holdout={json.dumps(data["train_image_holdout"]["natural"])}.')
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def all_stages(args,model,source,identity):
    from run_q3_r3_mass import full_dataset
    val_dataset=full_dataset(args.root_path,'val')
    if len(val_dataset)!=14: raise ValueError('Expected the same 14 val images')
    if not (args.output_dir/'smoke.json').is_file(): smoke(args,model,source,identity)
    audit=audit_train(args,model)
    long=audit['by_length'].get('32+',{})
    status={'stage':'candidate_gate','long_gaps':long,'required_long_full_coverage_rate':args.min_long_coverage}
    if long.get('eligible',0)<args.min_long_gaps or long.get('full_coverage_rate',0)<args.min_long_coverage:
        status['status']='blocked_candidate_reachability'; atomic_json(args.output_dir/'pipeline_state.json',status)
        final_report(args,None,{},audit,status['status'])
        print(f'[AUTOMATIC STOP] Long-gap candidate evidence is insufficient. Read {args.output_dir / "REPORT.md"}',flush=True)
        return 2
    # Original V9 and val feature cache are computed once and reused in every comparison.
    model.configure('B'); model.eval()
    baseline,baseline_cost=infer_cases(model,val_dataset,args.device,output=args.output_dir/'predictions_val/A_original',
        cache_features=args.output_dir/'candidate_cache/val',reference=args.reference,variant='A_original')
    # Validate fixed P0 fingerprint and 206 gaps before any new training starts.
    baseline_compare=args.output_dir/'baseline_comparison'
    compare_cases(baseline,['A_original'],args.reference,args.expected_gaps,baseline_compare,args.reference_set)
    train_phase(args,model,source,identity,'B',val_dataset)
    classifier={'heldout':head_report(model,read_cache(args.output_dir/'candidate_cache/train','heldout'),args.device,args.seed),
                'val':head_report(model,read_cache(args.output_dir/'candidate_cache/val'),args.device,args.seed)}
    atomic_json(args.output_dir/'relation_classification.json',classifier)
    natural=classifier['val']['natural']; held=classifier['heldout']['natural']
    ready=all(row['positive']>=args.min_class_count and row['negative']>=args.min_class_count and
              row['roc_auc'] is not None and row['roc_auc']>=args.min_relation_auc and row['pr_auc']>row['positive_fraction'] for row in (natural,held))
    if not ready:
        status.update(status='blocked_relation_discrimination',classifier=classifier)
        atomic_json(args.output_dir/'pipeline_state.json',status); final_report(args,None,classifier,audit,status['status'])
        print(f'[AUTOMATIC STOP] Candidate classifier evidence is insufficient. Read {args.output_dir / "REPORT.md"}',flush=True)
        return 2
    all_cases=[{**case,'B_relation':case['A_original'],'P0_surface':case['p0']} for case in baseline]
    names=['A_original','P0_surface','B_relation']; costs={'baseline_feature_cache_pass_including_candidate_search':baseline_cost,
        'B_relation':{'extra_surface_effect':0.,'single_tile_timing_in':'smoke.json'}}
    for phase in ('C','D'):
        # Matched ablations: C and D start from identical B weights and RNG,
        # rather than giving D an unreported extra continuation budget.
        restore(model,args.output_dir/PHASE_NAMES['B']/'best.pth',identity)
        train_phase(args,model,source,identity,phase,val_dataset)
        result,cost=phase_cases(args,model,identity,phase,val_dataset); name=PHASE_NAMES[phase]; names.append(name); costs[name]=cost
        for target,case in zip(all_cases,result):
            if target['case']!=case['case'] or not np.array_equal(target['p0'],case['p0']): raise ValueError('P0 or image identity changed across phases')
            target[name]=case[name]
    # E is deliberately an independent, no-update intervention on D weights.
    e_dir=args.output_dir/PHASE_NAMES['E']; e_dir.mkdir(parents=True,exist_ok=True)
    restore(model,args.output_dir/PHASE_NAMES['D']/'best.pth',identity); model.configure('E')
    save(args,model,'E',0,0.,None,identity,e_dir/'best.pth')
    result,cost=phase_cases(args,model,identity,'E',val_dataset); names.append(PHASE_NAMES['E']); costs[PHASE_NAMES['E']]=cost
    for target,case in zip(all_cases,result): target[PHASE_NAMES['E']]=case[PHASE_NAMES['E']]
    comparison=compare_cases(all_cases,names,args.reference,args.expected_gaps,args.output_dir/'comparison_val',args.reference_set)
    baseline_row=comparison['A_original']['matched_recall']; e_row=comparison[PHASE_NAMES['E']]['matched_recall']
    # Automated evidence check, no interactive enter or test-set information.
    joint_ready=(e_row['paired']['closed']>baseline_row['paired']['closed'] and
        e_row['paired']['background_bridge_proxy']<=baseline_row['paired']['background_bridge_proxy'] and
        e_row['metrics']['apls_bidirectional_approx']>=baseline_row['metrics']['apls_bidirectional_approx'])
    if joint_ready:
        train_phase(args,model,source,identity,'F',val_dataset)
        result,cost=phase_cases(args,model,identity,'F',val_dataset); names.append(PHASE_NAMES['F']); costs[PHASE_NAMES['F']]=cost
        for target,case in zip(all_cases,result): target[PHASE_NAMES['F']]=case[PHASE_NAMES['F']]
        comparison=compare_cases(all_cases,names,args.reference,args.expected_gaps,args.output_dir/'comparison_val',args.reference_set)
    status.update(status='complete' if joint_ready else 'complete_B_to_E_F_skipped_by_val_evidence',
                  joint_gate={'passed':joint_ready,'rule':'E at matched V9 Recall: more closed gaps, no more bridge proxies and no lower bidirectional approximate APLS'})
    atomic_json(args.output_dir/'pipeline_state.json',status); atomic_json(args.output_dir/'inference_costs.json',costs)
    for name in names:
        visuals(all_cases,name,comparison[name]['own_val_threshold']['threshold'],args.output_dir/'visuals'/name)
    final_report(args,comparison,classifier,audit,status['status'])
    if args.final_test: final_test(args,model,identity,comparison)
    print(f'[DONE] {args.output_dir / "REPORT.md"}',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','smoke','candidate_audit','train'))
    p.add_argument('--root_path',type=Path,required=True); p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--v9_run',type=Path,default=Path('/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407'))
    p.add_argument('--reference_set',type=Path); p.add_argument('--device',default='cuda')
    p.add_argument('--final_test',action='store_true',help='After freezing model/work points on val, evaluate only V9 and the selected model on test')
    p.add_argument('--phase',choices=('B','C','D','F'),default='B'); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch_size',type=int,default=2); p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=1234); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--hidden',type=int,default=64); p.add_argument('--warmup_epochs',type=int,default=5)
    p.add_argument('--val_interval',type=int,default=5)
    p.add_argument('--max_endpoints',type=int,default=32); p.add_argument('--max_pairs',type=int,default=48)
    p.add_argument('--max_paths',type=int,default=128)
    p.add_argument('--lambda_relation',type=float,default=1.); p.add_argument('--lambda_path',type=float,default=.2)
    p.add_argument('--lambda_neg',type=float,default=.1)
    p.add_argument('--min_long_coverage',type=float,default=.05); p.add_argument('--min_long_gaps',type=int,default=8)
    p.add_argument('--min_relation_auc',type=float,default=.60); p.add_argument('--min_class_count',type=int,default=8)
    args=p.parse_args(); args.device=torch.device(args.device)
    if args.batch_size!=2 or min(args.epochs,args.val_interval,args.num_workers+1,args.max_paths,args.max_pairs,args.max_endpoints)<=0:
        p.error('This protocol requires physical batch2, positive budgets/epochs, and nonnegative workers')
    if not 0<=args.min_long_coverage<=1 or not 0<=args.min_relation_auc<=1: p.error('Invalid evidence gate')
    if min(args.lr,args.lambda_relation,args.lambda_path,args.lambda_neg)<0: p.error('Negative learning rate or loss weights')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model,source,identity=load_model(args)
    if args.command=='all': raise SystemExit(all_stages(args,model,source,identity))
    if args.command=='smoke': smoke(args,model,source,identity)
    elif args.command=='candidate_audit': audit_train(args,model)
    else:
        from run_q3_r3_mass import full_dataset
        prior={'C':'B','D':'B','F':'E'}.get(args.phase)
        if prior: restore(model,args.output_dir/PHASE_NAMES[prior]/'best.pth',identity)
        train_phase(args,model,source,identity,args.phase,full_dataset(args.root_path,'val'))


if __name__=='__main__': main()
