"""Frozen-backbone proposal-only supervision with random/hard crop controls."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.direct_candidate_data import CandidateCropDataset,reference_regions
from diagnostics.audit_full_h0_saved_predictions import component_data,gap_state
from diagnostics.diagnose_full_h0_optimistic_reachability import (atomic_json,atomic_npz,rebuild_source,signature,
                                                                  validate_saved_prediction,verify_recorded_code)
from diagnostics.diagnose_unscaled_proposal_effect import verify_adaptation_code
from evaluation.road_comparison.road_metrics import skeletonize
from mass_protocol import blend_weight,sliding_positions
from networks.direct_candidate import DirectCandidateJoint,candidate_segmentation_loss,freeze_except_proposal
from networks.full_h0_joint import VERSION,build_h0
from run_q3_r3_mass import full_dataset
from run_slim_end_to_end_mass import atomic_checkpoint,digest,schedule,seed_all

ROOT=Path(__file__).resolve().parent
THRESHOLDS=(.1,.2,.3,.4,.45,.5,.6,.7,.8,.9)


def build_model(checkpoint,device,size=512):
    if checkpoint.get('version')!=VERSION or checkpoint.get('config',{}).get('variant')!='B':
        raise ValueError('Expected a full H0 B checkpoint')
    source,q=rebuild_source(checkpoint['model']);base,_=build_h0(checkpoint['config']['h0_args'],size=size)
    model=DirectCandidateJoint(base,q,source);model.load_state_dict(checkpoint['model'],strict=True)
    freeze_except_proposal(model)
    return model.to(device).eval()


def frozen_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith('original.proposal_head.'):continue
        h.update(name.encode()+b'\0');h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def head_norm_difference(model,initial):
    return float(np.sqrt(sum(float((p.detach().cpu().double()-initial[name].double()).square().sum())
                             for name,p in model.original.proposal_head.named_parameters())))


@torch.inference_mode()
def stitch(model,sample,device):
    image=sample['image'][None].to(device);h,w=image.shape[-2:]
    if (h,w)!=(1500,1500):raise ValueError('Expected native 1500x1500 image')
    canvas=torch.zeros((3,1,h,w),device=device);divisor=torch.zeros((1,1,h,w),device=device)
    weight=blend_weight(512,device)
    positions=[(y,x) for y in sliding_positions(h,512,256) for x in sliding_positions(w,512,256)]
    for index,(y,x) in enumerate(positions):
        result=model(image[:,:,y:y+512,x:x+512])
        b=F.interpolate(result['b_logit'],size=(512,512),mode='bilinear',align_corners=False)
        canvas[:,:,y:y+512,x:x+512]+=torch.cat((result['z0'],b,result['z1']),0)*weight
        divisor[:,:,y:y+512,x:x+512]+=weight
        if index==0 or (index+1)%5==0 or index+1==len(positions):
            print(f'[TILE] {sample["case_name"]} {index+1}/{len(positions)}',flush=True)
    values=(canvas/divisor).sigmoid().cpu().numpy()[:,0]
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite stitched prediction')
    return {'p0':values[0],'b':values[1],'p1':values[2]}


def reference_case(model,sample,device,output):
    name=str(sample['case_name']);path=output/(name+'.npz');input_sha=signature(sample)
    if path.is_file():
        with np.load(path) as f:
            if str(f['input_sha256'])!=input_sha:raise ValueError('Cached input changed: '+name)
        print('[CACHE] '+name,flush=True);return
    pred=stitch(model,sample,device)
    gt=sample['mask'][0].numpy()>.5;skel=sample['skeleton'][0].numpy()>.5
    arrays=reference_regions(pred['p0'],pred['b'],pred['p1'],gt,skel)
    arrays.update(pred);arrays['input_sha256']=np.asarray(input_sha)
    atomic_npz(path,arrays)
    print('[REFERENCE SAVED] '+name,flush=True)


def prepare(args,model,split='train'):
    if split not in ('train','val'):raise ValueError('Reference split must be train or val')
    folder=args.output_dir/('reference_train' if split=='train' else 'reference_val_evaluation_only');folder.mkdir(parents=True,exist_ok=True)
    dataset=full_dataset(args.root_path,split)
    names=[Path(name).stem.replace('_sat','') for name in dataset.image_files]
    if len(set(names))!=len(names):raise ValueError('Duplicate training case IDs')
    identity={'checkpoint_sha256':args.source_sha,'root_path':str(args.root_path.resolve()),
              'split':split,'images':names,'inference':'native1500 512/256 weighted logit stitch, no TTA',
              'b':'sigmoid of stitched bilinear-upsampled B_logit','threshold':.45,
              'code_sha256':args.code_sha,'numpy_version':np.__version__,
              'torch_version':str(torch.__version__),'opencv_version':cv2.__version__}
    path=folder/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Training cache identity mismatch')
    if not path.exists() and any(folder.iterdir()):raise ValueError('Nonempty training cache lacks identity')
    atomic_json(path,identity)
    for index in range(len(dataset)):
        sample=dataset[index];print(f'[REFERENCE {split.upper()}] {index+1}/{len(dataset)} {sample["case_name"]}',flush=True)
        reference_case(model,sample,args.device,folder)
    atomic_json(folder/'complete.json',{'images':len(dataset),'identity_sha256':digest(path)})
    print(f'[REFERENCE {split.upper()} DONE] '+str(folder),flush=True)


def require_reference(args,split):
    folder=args.output_dir/('reference_train' if split=='train' else 'reference_val_evaluation_only')
    complete=json.loads((folder/'complete.json').read_text());identity=json.loads((folder/'identity.json').read_text())
    if complete['identity_sha256']!=digest(folder/'identity.json') or identity['checkpoint_sha256']!=args.source_sha:
        raise ValueError('Reference completion or checkpoint fingerprint differs')
    if identity['split']!=split or identity['root_path']!=str(args.root_path.resolve()) or identity['code_sha256']!=args.code_sha:
        raise ValueError('Reference data/code identity differs')
    if complete['images']!=len(identity['images']):raise ValueError('Reference image count mismatch')
    return folder


def fixed_input(args):
    dataset=full_dataset(args.root_path,'train');sample=dataset[0]
    return sample['image'][None,:,:512,:512].to(args.device)


@torch.inference_mode()
def upstream_snapshot(model,image):
    model.eval();out=model(image,capture_diagnostics=True)
    return {name:out[name].detach().cpu().clone() for name in ('p0','q','proposal_input')}


def verify_upstream(model,image,initial,expected_frozen):
    current=upstream_snapshot(model,image);report={}
    for name in initial:
        difference=(current[name]-initial[name]).abs()
        report[name+'_max_abs']=float(difference.max())
        if report[name+'_max_abs']>1e-6:raise RuntimeError('Frozen upstream output changed: '+name)
    report['frozen_state_sha256']=frozen_digest(model)
    if report['frozen_state_sha256']!=expected_frozen:raise RuntimeError('Frozen parameter or buffer changed')
    return report


def mode_config(args,mode,cache_sha):
    return {'version':'direct_candidate_v1','mode':mode,'source_checkpoint':str(args.checkpoint.resolve()),
        'source_sha256':args.source_sha,'cache_identity_sha256':cache_sha,'root_path':str(args.root_path.resolve()),
        'epochs':args.epochs,'batch_size':2,'fp32':True,'num_workers':args.num_workers,'seed':args.seed,
        'optimizer':'AdamW','lr':args.lr,'weight_decay':.0001,'schedule':'original 10-epoch warmup then cosine to 0.1*lr',
        'proposal_formula':'sigmoid(logit(P0_work)+delta_plus-delta_minus)',
        'loss':'dynamic_losses segmentation only: focal + 0.5*dice; no continuity',
        'loss_weights':{'lambda_bg':1.,'lambda_missing':2.,'missing_threshold':.45},
        'sampling':'random: 100% random; hard: configured random/missing/hard probabilities',
        'sampling_probabilities':list(args.probabilities),
        'augmentation':'crop then synchronized flip/rotation and original color-degrade on crop; differs from full-image augmentation',
        'checkpoint_selection':'validation B IoU at threshold 0.45 every 5 epochs; max',
        'code_sha256':args.code_sha}


def save_head(path,model,optimizer,epoch,best,config,record=None):
    state={'version':'direct_candidate_v1','config':config,'epoch':epoch,'score':best,
           'proposal_head':model.original.proposal_head.state_dict(),'record':record}
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                     python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    atomic_checkpoint(path,state)


def restore_head(path,model,config,optimizer=None):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('version')!='direct_candidate_v1' or saved['config']!=config:raise ValueError('Candidate checkpoint/config mismatch')
    model.original.proposal_head.load_state_dict(saved['proposal_head'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        if saved['cuda_rng'] is not None:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved


def val_iou(args,model):
    dataset=full_dataset(args.root_path,'val');counts=Counter()
    for i in range(len(dataset)):
        sample=dataset[i];pred=stitch(model,sample,args.device)['b']>=.45;gt=sample['mask'][0].numpy()>.5
        counts['tp']+=int((pred&gt).sum());counts['fp']+=int((pred&~gt).sum());counts['fn']+=int((~pred&gt).sum())
        print(f'[VAL SCORE] {i+1}/{len(dataset)}',flush=True)
    return counts['tp']/max(sum(counts.values()),1),dict(counts)


def smoke(args,model,mode,reference_dir,output,config):
    dataset=CandidateCropDataset(args.root_path,reference_dir,mode,args.seed,args.probabilities);dataset.set_epoch(1)
    batch=next(iter(DataLoader(dataset,batch_size=2,num_workers=0)))
    image=batch['image'].to(args.device);mask=batch['mask'].to(args.device);skel=batch['skeleton'].to(args.device)
    fixed=fixed_input(args);initial=upstream_snapshot(model,fixed);frozen=frozen_digest(model)
    initial_head={name:p.detach().cpu().clone() for name,p in model.original.proposal_head.named_parameters()}
    optimizer=torch.optim.AdamW(model.original.proposal_head.parameters(),lr=args.lr,weight_decay=.0001)
    model.eval();optimizer.zero_grad(set_to_none=True);result=model(image)
    loss,parts=candidate_segmentation_loss(result,mask,skel)
    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite candidate loss')
    loss.backward()
    gradients={name:float(p.grad.detach().double().norm()) if p.grad is not None else 0.
               for name,p in model.original.proposal_head.named_parameters()}
    if not any(value>0 for value in gradients.values()):raise RuntimeError('Proposal gradient is zero')
    if any(p.grad is not None for name,p in model.named_parameters() if not name.startswith('original.proposal_head.')):
        raise RuntimeError('Frozen module received gradients')
    optimizer.step();change=head_norm_difference(model,initial_head)
    check=verify_upstream(model,fixed,initial,frozen)
    if change<=0:raise RuntimeError('Proposal optimizer did not change parameters')
    with torch.no_grad():
        for name,p in model.original.proposal_head.named_parameters():p.copy_(initial_head[name].to(p.device))
    model.zero_grad(set_to_none=True);seed_all(args.seed)
    atomic_json(output/'smoke.json',{'passed':True,'mode':mode,'loss':float(loss.detach()),
        'focal':float(parts['focal']),'dice':float(parts['dice']),'proposal_gradient_norms':gradients,
        'temporary_proposal_parameter_change':change,'upstream_after_temporary_step':check,
        'B_sigmoid_matches_logit':True,'frozen_model_eval':not model.training,'GT_in_forward':False})
    print('[SMOKE] '+mode+' passed',flush=True)


def train(args,model,mode):
    ref=require_reference(args,'train');out=args.output_dir/mode;out.mkdir(parents=True,exist_ok=True)
    config=mode_config(args,mode,digest(ref/'identity.json'));marker=out/'experiment.json'
    if marker.is_file() and json.loads(marker.read_text())!=config:raise ValueError('Mode configuration changed')
    atomic_json(marker,config);freeze_except_proposal(model)
    optimizer=torch.optim.AdamW(model.original.proposal_head.parameters(),lr=args.lr,weight_decay=.0001)
    optimizer.param_groups[0].update(initial_lr=args.lr,minimum_lr=.1*args.lr)
    included={id(p) for group in optimizer.param_groups for p in group['params']}
    expected={id(p) for p in model.original.proposal_head.parameters()}
    if included!=expected or any(p.requires_grad for n,p in model.named_parameters() if not n.startswith('original.proposal_head.')):
        raise RuntimeError('Optimizer/freeze scope incorrect')
    fixed=fixed_input(args);initial=upstream_snapshot(model,fixed);frozen=frozen_digest(model)
    initial_head={name:p.detach().cpu().clone() for name,p in model.original.proposal_head.named_parameters()}
    last=out/'last.pth';start=1;best=-float('inf')
    if last.is_file():
        state=restore_head(last,model,config,optimizer);start=state['epoch']+1;best=state['score']
        verify_upstream(model,fixed,initial,frozen)
    else:smoke(args,model,mode,ref,out,config)
    dataset=CandidateCropDataset(args.root_path,ref,mode,args.seed,args.probabilities)
    for epoch in range(start,args.epochs+1):
        dataset.set_epoch(epoch);schedule(optimizer,epoch-1,args.epochs);model.eval()
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed+epoch*1000003),pin_memory=args.device.type=='cuda')
        sums=Counter();sampling=Counter();coverage={'missing':set(),'hard':set()};order=hashlib.sha256();grad_sum=grad_max=0.
        for index,batch in enumerate(loader):
            for case in batch['case_name']:order.update(str(case).encode()+b'\0')
            for j,case in enumerate(batch['case_name']):
                requested=batch['requested'][j];actual=batch['actual'][j]
                sampling['requested_'+requested]+=1;sampling['actual_'+actual]+=1
                sampling['fallback_'+requested]+=int(batch['fallback'][j]);sampling['boundary_failed']+=int(batch['boundary_failed'][j])
                sampling['success_'+requested]+=int(not bool(batch['fallback'][j]))
                sampling['priority_hard']+=int(batch['priority'][j])
                if actual in coverage:coverage[actual].add((case,int(batch['component'][j])))
            image=batch['image'].to(args.device);mask=batch['mask'].to(args.device);skeleton=batch['skeleton'].to(args.device)
            optimizer.zero_grad(set_to_none=True);result=model(image)
            loss,parts=candidate_segmentation_loss(result,mask,skeleton)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite candidate loss')
            loss.backward()
            if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('original.proposal_head.')):
                raise RuntimeError('A frozen parameter received gradient')
            grad=float(torch.sqrt(sum(p.grad.detach().double().square().sum() for p in model.original.proposal_head.parameters() if p.grad is not None)))
            if not np.isfinite(grad):raise RuntimeError('Nonfinite proposal gradient')
            grad_sum+=grad;grad_max=max(grad_max,grad);optimizer.step()
            sums['loss']+=float(loss.detach());sums['focal']+=float(parts['focal']);sums['dice']+=float(parts['dice']);sums['batches']+=1
            if index%25==0:print(f'[TRAIN {mode}] epoch={epoch}/{args.epochs} batch={index+1}/{len(loader)} loss={float(loss.detach()):.6f} grad={grad:.6g}',flush=True)
        check=verify_upstream(model,fixed,initial,frozen)
        record={'epoch':epoch,'case_order_sha256':order.hexdigest(),'image_visits':len(dataset),
                'loss':{k:v/max(sums['batches'],1) for k,v in sums.items() if k!='batches'},
                'sampling':dict(sampling),'unique_components':{k:len(v) for k,v in coverage.items()},
                'available_components':dataset.available_components,
                'component_coverage_fraction':{k:len(v)/dataset.available_components[k] if dataset.available_components[k] else None for k,v in coverage.items()},
                'proposal_gradient_norm_mean':grad_sum/max(sums['batches'],1),'proposal_gradient_norm_max':grad_max,
                'proposal_parameter_change_l2':head_norm_difference(model,initial_head),'upstream_check':check,
                'learning_rate':optimizer.param_groups[0]['lr']}
        if epoch%5==0 or epoch==args.epochs:
            score,counts=val_iou(args,model);record['val_B_iou_at_045']=score;record['val_counts']=counts
            if score>best:best=score;save_head(out/'best.pth',model,None,epoch,best,config)
            print(f'[VAL {mode}] epoch={epoch} B_IoU_045={score:.6f}',flush=True)
        atomic_json(out/f'epoch_{epoch:03d}.json',record);save_head(last,model,optimizer,epoch,best,config,record)
    atomic_json(out/'training_complete.json',{'epochs':args.epochs,'best_B_iou_at_045':best})


def candidate_metrics(cases,field,threshold,cldice=True):
    total=Counter();scores=[]
    for case in cases:
        pred=case[field]>=threshold;gt=case['gt']
        total['tp']+=int((pred&gt).sum());total['fp']+=int((pred&~gt).sum());total['fn']+=int((~pred&gt).sum())
        if cldice:
            ps=skeletonize(pred);gs=case['eval_skeleton']
            precision=float((ps&gt).sum()/max(ps.sum(),1));recall=float((gs&pred).sum()/max(gs.sum(),1))
            scores.append(2*precision*recall/max(precision+recall,1e-12))
    tp,fp,fn=total['tp'],total['fp'],total['fn']
    return {'threshold':float(threshold),'tp':tp,'fp':fp,'fn':fn,'iou':tp/max(tp+fp+fn,1),
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
            'cldice':float(np.mean(scores)) if scores else None}


def distribution(values):
    if not len(values):return {'pixels':0,'mean':None,'p10':None,'p25':None,'p50':None,'p75':None,'p90':None,'p99':None}
    v=np.asarray(values,dtype=np.float32);q=np.quantile(v,[.1,.25,.5,.75,.9,.99])
    return {'pixels':len(v),'mean':float(v.mean()),**{f'p{n}':float(x) for n,x in zip((10,25,50,75,90,99),q)}}


def masks_for_eval(case):
    gt=case['gt'];p0=case['initial_p0'];skel=case['skeleton']
    covered=cv2.dilate((p0>=.45).astype(np.uint8),np.ones((5,5),np.uint8))>0
    normal=skel&covered
    result={'true_missing_skeleton':case['missing_points'],'normal_road_skeleton':np.flatnonzero(normal),
            'near_road_background':case['near_background_points'],'hard_background':case['hard_points'],
            'initial_added_FP_proxy':case['added_fp_points'],'initial_bridge_proxy':case['bridge_proxy_points']}
    return result


def choose_threshold_for_recall(values,target_count):
    values=np.asarray(values,dtype=np.float32)
    if not len(values):return None,None
    target_count=max(0,min(len(values),int(target_count)))
    if target_count==0:return float(np.nextafter(values.max(),np.float32(np.inf))),0
    cutoff=np.partition(values,len(values)-target_count)[len(values)-target_count]
    choices=[float(cutoff),float(np.nextafter(np.float32(cutoff),np.float32(np.inf)))]
    threshold=min(choices,key=lambda t:abs(int((values>=t).sum())-target_count))
    return threshold,int((values>=threshold).sum())-target_count


def report_fixed_gaps(cases,manifest,p1_threshold,output):
    rows=[]
    for case in cases:
        shape=case['gt'].shape;p0=case['p0']>=.45;p1=case['p1']>=p1_threshold
        labels0=component_data(p0)[1];labels1=component_data(p1)[1]
        for gap in manifest['images'][case['case']]['gaps']:
            before=gap_state(p0,labels0,gap,shape);after=gap_state(p1,labels1,gap,shape)
            rows.append({'case':case['case'],'gap_id':gap['id'],'length_bin':gap.get('length_bin','unknown'),
                'p0_coverage':before['coverage'],'p1_coverage':after['coverage'],
                'p0_local_closed':before['local_closed_proxy'],'p1_local_closed':after['local_closed_proxy'],
                'newly_closed':bool(not before['local_closed_proxy'] and after['local_closed_proxy']),
                'lost_closure':bool(before['local_closed_proxy'] and not after['local_closed_proxy']),
                'p0_threshold':.45,'p1_threshold':p1_threshold})
    if len(rows)!=manifest['count']:raise ValueError('Fixed gap list count mismatch')
    from diagnostics.v91_counterfactual import write_csv
    write_csv(output/'fixed_gaps.csv',rows)
    return {'count':len(rows),'p0_local_closed':sum(r['p0_local_closed'] for r in rows),
            'p1_local_closed':sum(r['p1_local_closed'] for r in rows),
            'newly_closed':sum(r['newly_closed'] for r in rows),'lost_closure':sum(r['lost_closure'] for r in rows),
            'p1_threshold':p1_threshold}


def evaluate_mode(args,model,mode,initial_checkpoint=False):
    val_ref=require_reference(args,'val');dataset=full_dataset(args.root_path,'val')
    if len(dataset)!=14:raise ValueError('Expected 14 validation images')
    out=args.output_dir/('initial' if initial_checkpoint else mode)/'eval';folder=out/'val';folder.mkdir(parents=True,exist_ok=True)
    cases=[];p0_max=0.
    for index in range(len(dataset)):
        sample=dataset[index];name=str(sample['case_name']);initial_path=val_ref/(name+'.npz')
        with np.load(initial_path) as ref:
            if str(ref['input_sha256'])!=signature(sample):raise ValueError('Validation reference input changed: '+name)
            init={k:ref[k].copy() for k in ('p0','b','p1','missing_points','hard_points','added_fp_points','bridge_proxy_points','near_background_points')}
        pred=({'p0':init['p0'],'b':init['b'],'p1':init['p1']} if initial_checkpoint else stitch(model,sample,args.device))
        gt=sample['mask'][0].numpy()>.5
        ref_gt=args.pair_dir/'reference'/(name+'.npz')
        if ref_gt.is_file():
            with np.load(ref_gt) as original:
                if not np.array_equal(original['gt'].astype(bool),gt):raise ValueError('Fixed gap GT mismatch')
        d=float(np.abs(pred['p0']-init['p0']).max());p0_max=max(p0_max,d)
        if d>1e-5:raise RuntimeError('Frozen P0 differs from starting checkpoint: '+name)
        if initial_checkpoint:
            for key in ('b','p1'):
                if not np.allclose(pred[key],init[key],atol=1e-5,rtol=1e-5):raise RuntimeError('Initial reference prediction changed: '+key)
        arrays={'case':name,'gt':gt,'skeleton':sample['skeleton'][0].numpy()>.5,
                'eval_skeleton':skeletonize(gt),'initial_p0':init['p0'],'B_init':init['b'],
                'initial_p1':init['p1'],**pred}
        for key in ('missing_points','hard_points','added_fp_points','bridge_proxy_points','near_background_points'):arrays[key]=init[key]
        cases.append(arrays)
        atomic_npz(folder/(name+'.npz'),{k:v for k,v in arrays.items() if k!='case'})
        print(f'[EVAL {"initial" if initial_checkpoint else mode}] {index+1}/{len(dataset)}',flush=True)
    curve=[]
    for threshold in THRESHOLDS:
        curve.append(candidate_metrics(cases,'b',threshold))
        print(f'[B SWEEP] {threshold:.2f}',flush=True)
    selection=max(curve,key=lambda r:r['iou']);atomic_json(out/'B_threshold_curve.json',curve)
    atomic_json(out/'val_selection.json',{'threshold':selection['threshold'],'rule':'maximum validation B IoU on fixed threshold grid',
        'selected_on':'val','source_checkpoint_sha256':args.source_sha,'best_head_sha256':None if initial_checkpoint else digest(args.output_dir/mode/'best.pth'),
        'P0_max_abs_vs_initial':p0_max,'threshold_grid':list(THRESHOLDS)})
    manifest=json.loads((args.pair_dir/'reference/fixed_gaps_and_anchors.json').read_text())
    if manifest['count']!=198:raise ValueError('Expected fixed 198 gap reference')
    gaps=report_fixed_gaps(cases,manifest,args.p1_threshold,out)
    atomic_json(out/'P1_fixed_gap_summary.json',gaps)
    atomic_json(out/'surface_metrics.json',{
        'p0':candidate_metrics(cases,'p0',.45),
        'p1':candidate_metrics(cases,'p1',args.p1_threshold),
        'p0_threshold':.45,'p1_threshold':args.p1_threshold,
        'p0_max_abs_vs_initial':p0_max})
    return cases


def load_eval_cases(args,label):
    folder=args.output_dir/label/'eval/val';dataset=full_dataset(args.root_path,'val');cases=[]
    for sample in dataset:
        name=str(sample['case_name']);path=folder/(name+'.npz')
        with np.load(path) as f:item={k:f[k].copy() for k in f.files}
        if not np.array_equal(item['gt'].astype(bool),sample['mask'][0].numpy()>.5):raise ValueError('Saved validation GT differs')
        item['case']=name;cases.append(item)
    return cases


def compare(args):
    labels=('initial','random','hard');all_cases={name:load_eval_cases(args,name) for name in labels}
    selection={name:json.loads((args.output_dir/name/'eval/val_selection.json').read_text()) for name in labels}
    configs=[json.loads((args.output_dir/name/'experiment.json').read_text()) for name in ('random','hard')]
    for key in configs[0]:
        if key not in ('mode','sampling') and configs[0].get(key)!=configs[1].get(key):
            raise ValueError('Random/hard experiment setting differs: '+key)
    for name in labels:
        if selection[name]['source_checkpoint_sha256']!=args.source_sha:raise ValueError('Compared checkpoints differ')
        if name!='initial' and selection[name]['best_head_sha256']!=digest(args.output_dir/name/'best.pth'):
            raise ValueError('Saved candidate predictions do not match best head')
    for epoch in range(1,args.epochs+1):
        random_record=json.loads((args.output_dir/'random'/f'epoch_{epoch:03d}.json').read_text())
        hard_record=json.loads((args.output_dir/'hard'/f'epoch_{epoch:03d}.json').read_text())
        if random_record['case_order_sha256']!=hard_record['case_order_sha256']:
            raise ValueError('Random/hard image visit order differs at epoch '+str(epoch))
    initial=all_cases['initial'];initial_threshold=selection['initial']['threshold']
    missing=[c['B_init'].reshape(-1)[c['missing_points']] for c in initial]
    initial_positive=np.concatenate(missing);target=int((initial_positive>=initial_threshold).sum())
    rows=[];regions=[];per_gap=[]
    for label in labels:
        cases=all_cases[label]
        if [c['case'] for c in cases]!=[c['case'] for c in initial]:raise ValueError('Validation cases differ')
        for a,b in zip(cases,initial):
            if not np.array_equal(a['gt'],b['gt']) or not np.allclose(a['p0'],b['p0'],atol=1e-5,rtol=0):
                raise ValueError('GT or frozen P0 differs across modes')
            for key in ('B_init','initial_p0','missing_points','hard_points','added_fp_points','bridge_proxy_points','near_background_points'):
                if not np.array_equal(a[key],b[key]):raise ValueError('Reference candidates differ: '+key)
        values=np.concatenate([c['b'].reshape(-1)[c['missing_points']] for c in cases])
        matched,error=choose_threshold_for_recall(values,target)
        for workpoint,threshold in (('own_val',selection[label]['threshold']),('same_initial_threshold',initial_threshold),('matched_missing_recall',matched)):
            if threshold is None:continue
            metrics=candidate_metrics(cases,'b',threshold);row={'mode':label,'workpoint':workpoint,**metrics,
                'missing_recall':float((values>=threshold).mean()) if len(values) else None,
                'missing_match_count_error':error if workpoint=='matched_missing_recall' else None}
            for region in ('hard_background','initial_added_FP_proxy','initial_bridge_proxy'):
                pooled=np.concatenate([c['b'].reshape(-1)[masks_for_eval(c)[region]] for c in cases])
                row[region+'_pixels']=len(pooled);row[region+'_acceptance']=float((pooled>=threshold).mean()) if len(pooled) else None
            rows.append(row)
        for region in ('true_missing_skeleton','normal_road_skeleton','near_road_background','hard_background',
                       'initial_added_FP_proxy','initial_bridge_proxy'):
            pooled=np.concatenate([c['b'].reshape(-1)[masks_for_eval(c)[region]] for c in cases])
            regions.append({'mode':label,'region':region,**distribution(pooled)})
    out=args.output_dir/'comparison';out.mkdir(parents=True,exist_ok=True)
    from diagnostics.v91_counterfactual import write_csv
    write_csv(out/'candidate_comparison.csv',rows);write_csv(out/'candidate_distributions.csv',regions)
    summaries={name:json.loads((args.output_dir/name/'eval/P1_fixed_gap_summary.json').read_text()) for name in labels}
    surfaces={name:json.loads((args.output_dir/name/'eval/surface_metrics.json').read_text()) for name in labels}
    gap_tables={}
    for name in labels:
        with (args.output_dir/name/'eval/fixed_gaps.csv').open(newline='',encoding='utf-8') as f:
            gap_tables[name]={(row['case'],row['gap_id']):row for row in csv.DictReader(f)}
    if any(set(gap_tables[name])!=set(gap_tables['initial']) for name in labels):raise ValueError('Fixed gap IDs differ')
    gap_rows=[]
    for key,base in gap_tables['initial'].items():
        entry={'case':key[0],'gap_id':key[1],'length_bin':base['length_bin'],'p0_local_closed':base['p0_local_closed']}
        for name in labels:
            item=gap_tables[name][key]
            if item['p0_local_closed']!=base['p0_local_closed']:raise ValueError('Fixed P0 gap state differs')
            for field in ('p1_local_closed','p1_coverage','newly_closed','lost_closure'):
                entry[name+'_'+field]=item[field]
        gap_rows.append(entry)
    write_csv(out/'fixed_gaps_all.csv',gap_rows)
    lines=['# Direct candidate training comparison','',
           'B_init, random and hard share the same source checkpoint. Random/hard image order is verified per epoch.',
           'Candidate thresholds are selected or matched on val only. GT missing/background and bridge labels are proxies.',
           '| Mode | Workpoint | Threshold | IoU | Precision | Recall | clDice | Missing recall | Hard-background accept | Added-FP accept | Bridge-proxy accept |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'n/a' if value is None else f'{value:.5f}'
    for r in rows:
        lines.append(f'| {r["mode"]} | {r["workpoint"]} | {fmt(r["threshold"])} | {fmt(r["iou"])} | {fmt(r["precision"])} | {fmt(r["recall"])} | {fmt(r["cldice"])} | {fmt(r["missing_recall"])} | {fmt(r["hard_background_acceptance"])} | {fmt(r["initial_added_FP_proxy_acceptance"])} | {fmt(r["initial_bridge_proxy_acceptance"])} |')
    lines += ['', '## P0 and P1 segmentation metrics','',f'`{json.dumps(surfaces)}`',
              '', '## Fixed gap P0/P1 proxy','',f'`{json.dumps(summaries)}`',
              '', 'See candidate_distributions.csv for B quantiles and fixed_gaps_all.csv for unchanged gap IDs.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(out/'summary.json',{'source_checkpoint_sha256':args.source_sha,'selection':selection,'results':rows,
                                    'candidate_distributions':regions,'fixed_gap_summaries':summaries,'surface_metrics':surfaces,
                                    'limits':['GT background and bridge labels are proxies, not verified erroneous topology.',
                                    'Matched missing recall threshold is a validation-set descriptive comparison, not a test-selected parameter.',
                                    'P1 gap closure is the existing local connectivity/80% coverage proxy.']})
    print('[COMPARE DONE] '+str(out/'REPORT.md'),flush=True)


def source_selection(args):
    path=args.checkpoint.parent/'eval/val_selection.json'
    if not path.is_file():
        if args.p1_threshold is None:raise FileNotFoundError('Source validation threshold missing; supply --p1_threshold selected on val')
        return args.p1_threshold,'explicit validation threshold'
    selection=json.loads(path.read_text())
    if selection.get('checkpoint_sha256')!=args.source_sha:raise ValueError('Source validation selection/checkpoint mismatch')
    if args.p1_threshold is not None:return args.p1_threshold,'explicit validation threshold'
    return float(selection['threshold']),'source checkpoint validation selection'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','prepare','smoke','train','val','compare'))
    p.add_argument('--checkpoint',type=Path,required=True,help='Absolute path of the new_B full checkpoint')
    p.add_argument('--pair_dir',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--mode',choices=('random','hard','both'),default='both')
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--num_workers',type=int,default=4);p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--p1_threshold',type=float)
    p.add_argument('--random_prob',type=float,default=.5);p.add_argument('--missing_prob',type=float,default=.25)
    p.add_argument('--hard_prob',type=float,default=.25)
    args=p.parse_args()
    if not args.checkpoint.is_absolute():p.error('--checkpoint must be an absolute path')
    if args.epochs<1 or args.lr<=0 or args.num_workers<0:p.error('Invalid training settings')
    args.probabilities=(args.random_prob,args.missing_prob,args.hard_prob)
    if any(v<0 for v in args.probabilities) or abs(sum(args.probabilities)-1)>1e-8:p.error('Sampling probabilities must be nonnegative and sum to 1')
    args.device=torch.device(args.device);args.checkpoint=args.checkpoint.resolve()
    args.pair_dir=args.pair_dir.resolve();args.output_dir=args.output_dir.resolve()
    args.source_sha=digest(args.checkpoint)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=checkpoint['config'];verify_recorded_code(config);verify_adaptation_code(config)
    reference=args.pair_dir/'reference/fixed_gaps_and_anchors.json'
    if digest(reference)!=config['reference_sha256']:raise ValueError('Source checkpoint and fixed gaps differ')
    args.root_path=Path(config['root_path']).resolve()
    args.p1_threshold,args.p1_threshold_source=source_selection(args)
    if not 0.<args.p1_threshold<1.:p.error('P1 threshold must be between 0 and 1')
    files=(ROOT/'run_direct_candidate.py',ROOT/'networks/direct_candidate.py',ROOT/'datasets/direct_candidate_data.py',
           ROOT/'datasets/dataset_road_skeleton.py',ROOT/'mass_protocol.py',ROOT/'run_q3_r3_mass.py')
    args.code_sha={str(path.relative_to(ROOT)):digest(path) for path in files}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.command=='compare':compare(args);return
    seed_all(args.seed);model=build_model(checkpoint,args.device);initial_state=checkpoint['model'];del checkpoint
    modes=('random','hard') if args.mode=='both' else (args.mode,)
    if args.command in ('prepare','all'):
        prepare(args,model,'train')
        prepare(args,model,'val')
    if args.command=='prepare':return
    if args.command=='smoke':
        reference_dir=require_reference(args,'train')
        for mode in modes:
            out=args.output_dir/mode;out.mkdir(parents=True,exist_ok=True)
            smoke(args,model,mode,reference_dir,out,mode_config(args,mode,digest(reference_dir/'identity.json')))
        return
    if args.command in ('all','val'):
        # Evaluation-only val reference never feeds CandidateCropDataset.
        try:require_reference(args,'val')
        except FileNotFoundError:prepare(args,model,'val')
        initial_folder=args.output_dir/'initial/eval'
        initial_eval=initial_folder/'val_selection.json'
        if not all((initial_folder/name).is_file() for name in ('val_selection.json','surface_metrics.json','P1_fixed_gap_summary.json')):
            evaluate_mode(args,model,'initial',True)
        else:
            selected=json.loads(initial_eval.read_text())
            if selected.get('source_checkpoint_sha256')!=args.source_sha:raise ValueError('Initial evaluation source differs')
    for mode in modes:
        # Each independent branch starts from the exact same full source weights.
        model.load_state_dict(initial_state,strict=True);freeze_except_proposal(model);seed_all(args.seed)
        if args.command in ('all','train'):
            print('[BEGIN TRAIN] '+mode,flush=True);train(args,model,mode)
        if args.command in ('all','val'):
            ref=require_reference(args,'train');cfg=mode_config(args,mode,digest(ref/'identity.json'))
            restore_head(args.output_dir/mode/'best.pth',model,cfg);model.eval()
            evaluate_mode(args,model,mode)
    if args.command=='all' and len(modes)==2:compare(args)
    print('[DONE] '+str(args.output_dir),flush=True)


if __name__=='__main__':main()
