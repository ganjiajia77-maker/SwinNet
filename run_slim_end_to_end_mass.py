"""Self-contained 80-epoch Mass end-to-end experiment; no BASE/Q3 checkpoint."""
import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.slim_end_to_end import VERSION, build_model, end_to_end_losses
from networks.v9_selector_transport_training import gradient_norm
from networks.v9_source_relation_labels import RelationLabels
from run_v9_gate_benefit import atomic_json
from mass_protocol import blend_weight, sliding_positions
from evaluation.road_comparison.road_metrics import image_metrics, summarize
from selector_transport_reference import PROTOCOL, prepared, reference_digest, closed_gaps, bridge_proxy, exact_recall_threshold
from run_v9_selector_transport import point_metrics, selection_score
from diagnostics.v91_counterfactual import write_csv


def digest(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def training_dataset(args, augment=True):
    return RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=augment)


def optimizer_for(model, loaded_names):
    prefix='original.upstream.backbone.segmenter.'
    pretrained=[]; new=[]; assigned=[]
    for name,param in model.named_parameters():
        if not param.requires_grad: raise ValueError(f'Unexpected frozen end-to-end parameter: {name}')
        if name.startswith(prefix) and name[len(prefix):] in loaded_names:
            pretrained.append(param); assigned.append(name)
        else: new.append(param)
    groups=[{'name':'pretrained','params':pretrained,'lr':5e-5,'initial_lr':5e-5,'minimum_lr':5e-6},
            {'name':'new','params':new,'lr':2e-4,'initial_lr':2e-4,'minimum_lr':1e-5}]
    return torch.optim.AdamW(groups,weight_decay=.01),assigned


def schedule(optimizer, epoch, epochs):
    for group in optimizer.param_groups:
        base,minimum=group['initial_lr'],group['minimum_lr']
        if epoch<10: group['lr']=base*(epoch+1)/10.
        else:
            length=max(1,epochs-10)
            group['lr']=minimum if length==1 else minimum+.5*(base-minimum)*(1.+math.cos(math.pi*min((epoch-10)/(length-1),1.)))


def atomic_checkpoint(path, state):
    temporary=path.with_suffix('.tmp')
    try:
        torch.save(state,temporary); temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def checkpoint(model,optimizer,args,epoch,best,with_optimizer):
    state={'version':VERSION,'model':model.state_dict(),'config':args.config,'epoch':epoch,'score':best}
    if with_optimizer:
        state.update(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    return state


def load_weights(model,args,path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version')!=VERSION: raise ValueError('Expected a self-contained slim end-to-end checkpoint; old Q3/V9 weights are incompatible')
    if state['config']!=args.config: raise ValueError('Saved experiment configuration differs')
    model.load_state_dict(state['model'],strict=True)
    return state


def group_norms(model):
    groups={'encoder':[],'decoder':[],'Q':[],'R':[],'W':[],'B':[],'A':[],'rates':[],'selector':[],'pair_features':[]}
    for name,param in model.named_parameters():
        if param.grad is None: continue
        if name.startswith('original.upstream.backbone.'):
            key='encoder' if any(part in name for part in ('.layers.','.patch_embed.')) else 'decoder'
        elif name.startswith('original.upstream.'): key='Q'
        elif name.startswith('selector.'): key='selector'
        else:
            key=next((label for prefix,label in (('reliable_head','R'),('relations','W'),('proposal_head','B'),('accept_head','A'),('dynamics','rates'),('gate_relation','pair_features')) if name.startswith('original.'+prefix)),None)
        if key: groups[key].append(param.grad.detach().double().square().sum())
    return {key:float(torch.stack(values).sum().sqrt()) if values else 0. for key,values in groups.items()}


def loss(args,result,mask,skeleton,seed):
    return end_to_end_losses(result,mask,skeleton,RelationLabels(),seed,
        lambda_q=args.config['lambda_q'],lambda_choice=args.config['lambda_choice'],lambda_p0=args.config['lambda_p0'])


def smoke(args,model,optimizer):
    dataset=training_dataset(args,False); dataset.set_epoch(1)
    batch=next(iter(DataLoader(dataset,batch_size=2,num_workers=0)))
    image,mask,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
    buffers={name:value.detach().clone() for name,value in model.named_buffers()}
    optimizer.zero_grad(set_to_none=True); model.train()
    if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device); torch.cuda.synchronize(args.device)
    started=time.perf_counter(); result=model(image,capture_diagnostics=True)
    total,terms=loss(args,result,mask,skeleton,args.seed)
    # Select the first trainable patch-embedding parameter by its actual name.
    probes=[p for name,p in model.named_parameters() if '.patch_embed.' in name][:1]
    # A separate pure P1 loss proves the final output reaches the encoder.
    from networks.reliable_directional_repair import dynamic_losses
    base,base_terms=dynamic_losses(result,mask,skeleton,return_live_terms=True)
    final_encoder_norm=gradient_norm(base-.2*base_terms['cont_live'],probes)
    total.backward(); norms=group_norms(model)
    if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite real-batch loss')
    if final_encoder_norm<=0 or any(value<=0 for value in norms.values()):
        raise RuntimeError(f'Broken end-to-end gradient path: {norms}; P1->encoder={final_encoder_norm}')
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite gradient')
    if not model.original.upstream.backbone.training: raise RuntimeError('Segmentation backbone was forced into eval')
    torch.testing.assert_close(result['source_choice'].sum(2),torch.ones_like(result['source_choice'][:,:,0]),atol=2e-6,rtol=1e-5)
    if bool(result['source_choice'][:,:,1:][~result['source_valid']].any()): raise RuntimeError('Invalid source selected')
    if args.device.type=='cuda': torch.cuda.synchronize(args.device)
    report={'passed':True,'batch_size':2,'input_shape':list(image.shape),'feature_shapes':result['feature_shapes'],
        'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'gradient_norms':norms,'P1_segmentation_to_encoder_norm':final_encoder_norm,
        'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None,
        'forward_backward_seconds':time.perf_counter()-started,'loss':float(total.detach()),
        'choice_counts':terms['choice_counts'],'backbone_training':True,'GT_enters_inference':False}
    # Smoke must not change BN statistics of the official training start.
    with torch.no_grad():
        for name,value in model.named_buffers(): value.copy_(buffers[name])
    optimizer.zero_grad(set_to_none=True); atomic_json(args.output_dir/'smoke.json',report)
    print('[SMOKE] '+json.dumps(report),flush=True)


@torch.no_grad()
def infer(args,model,split,save_predictions=False):
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(args.root_path,split); expected=14 if split=='val' else 49
    if len(dataset)!=expected: raise ValueError(f'Expected {expected} {split} images, got {len(dataset)}')
    model.eval(); cases=[]; folder=args.output_dir/'eval'/split
    if save_predictions: folder.mkdir(parents=True,exist_ok=True)
    weight=blend_weight(512,args.device)
    for index in range(len(dataset)):
        sample=dataset[index]; image=sample['image'][None].to(args.device); h,w=image.shape[-2:]
        if (h,w)!=(1500,1500): raise ValueError('Native Mass evaluation must retain 1500x1500')
        canvas=torch.zeros((2,1,h,w),device=args.device); divisor=torch.zeros((1,1,h,w),device=args.device)
        for y in sliding_positions(h,512,256):
            for x in sliding_positions(w,512,256):
                result=model(image[:,:,y:y+512,x:x+512])
                canvas[:,:,y:y+512,x:x+512]+=torch.cat((result['z0'],result['z1']),0)*weight
                divisor[:,:,y:y+512,x:x+512]+=weight
        probabilities=(canvas/divisor).sigmoid().cpu().numpy()[:,0]
        case={'case':str(sample['case_name']),'gt':sample['mask'][0].numpy()>.5,'p0':probabilities[0],'p1':probabilities[1]}
        if not np.isfinite(probabilities).all(): raise RuntimeError('Nonfinite stitched prediction')
        if save_predictions: np.savez_compressed(folder/(case['case']+'.npz'),p0=case['p0'],p1=case['p1'],gt=case['gt'])
        cases.append(case); print(f'[EVAL {split}] {index+1}/{len(dataset)}',flush=True)
    return cases


def epoch_validation(cases):
    metrics=[{'case_id':case['case'],**image_metrics(case['p1']>=.6,prepared(case),PROTOCOL)} for case in cases]
    summary=summarize(metrics)
    bridges=sum(bridge_proxy(case['p0']>=.45,case['p1']>=.6,case['gt']) for case in cases)
    return {'metrics':summary,'bridge_proxy':bridges,'threshold':.6,'selection_score':selection_score(summary,bridges,len(cases))}


def train(args,model,optimizer,resume):
    seed_all(args.seed); start=1; best=-float('inf')
    if resume:
        saved=load_weights(model,args,resume); optimizer.load_state_dict(saved['optimizer']); start=saved['epoch']+1; best=saved['score']
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        if saved['cuda_rng'] is not None: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    dataset=training_dataset(args)
    for epoch in range(start,args.epochs+1):
        model.train(); dataset.set_epoch(epoch); schedule(optimizer,epoch-1,args.epochs)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed+epoch*1000003),pin_memory=args.device.type=='cuda')
        sums=Counter(); counts=Counter(); started=time.perf_counter(); order=hashlib.sha256(); augmentation=hashlib.sha256(); norms=None
        if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
        for index,batch in enumerate(loader):
            for name in batch['case_name']: order.update(str(name).encode()+b'\0')
            augmentation.update(batch['image'].contiguous().numpy().tobytes())
            image,mask,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
            optimizer.zero_grad(set_to_none=True); result=model(image)
            total,terms=loss(args,result,mask,skeleton,args.seed+epoch*100000+index)
            if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite training loss')
            total.backward()
            if norms is None: norms=group_norms(model)
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite training gradient')
            optimizer.step()
            for key in ('seg','cont','p0','q_loss','relation','choice','q_positive_fraction'): sums[key]+=float(terms[key])
            sums['total']+=float(total.detach()); sums['batches']+=1
            counts.update(terms['relation_counts'])
            for step in terms['choice_counts']['steps']:
                for row in step['images']:
                    for key in ('positions','wrong_bearing_positions','skipped'):
                        counts[f'step{step["step"]}_{key}']+=row[key]
            if index%25==0: print(f'[TRAIN] epoch={epoch}/{args.epochs} batch={index+1}/{len(loader)} loss={float(total.detach()):.6f}',flush=True)
        record={'epoch':epoch,'losses':{key:sums[key]/max(sums['batches'],1) for key in sums if key!='batches'},
            'counts':dict(counts),'gradient_norms':norms,'case_order_sha256':order.hexdigest(),'augmented_images_sha256':augmentation.hexdigest(),
            'seconds':time.perf_counter()-started,'learning_rates':{group['name']:group['lr'] for group in optimizer.param_groups},
            'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None}
        if epoch%5==0 or epoch==args.epochs:
            validation=epoch_validation(infer(args,model,'val')); atomic_json(args.output_dir/f'val_epoch_{epoch:03d}.json',validation)
            score=validation['selection_score']; record['val_score_at_0_60']=score
            if score>best:
                best=score; atomic_checkpoint(args.output_dir/'best.pth',checkpoint(model,optimizer,args,epoch,best,False))
        atomic_json(args.output_dir/f'epoch_{epoch:03d}.json',record)
        atomic_checkpoint(args.output_dir/'last.pth',checkpoint(model,optimizer,args,epoch,best,True))
    atomic_json(args.output_dir/'training_complete.json',{'epochs':args.epochs,'best_score':best})


def fixed_reference(cases):
    # The trainable P0 is different from frozen V9. Freeze selected-model P0@.45
    # once for THIS run; never claim its counts are the original 206-gap set.
    images={}; total=0
    for case in cases:
        p0=case['p0']>=.45; gt=case['gt']; h,w=gt.shape
        covered=cv2.dilate(p0.astype(np.uint8),np.ones((5,5),np.uint8))>0
        n,labels,stats,_=cv2.connectedComponentsWithStats((prepared(case)['skeleton']&~covered).astype(np.uint8),connectivity=8)
        _,old=cv2.connectedComponents(p0.astype(np.uint8),connectivity=8); gaps=[]
        for i in range(1,n):
            length=int(stats[i,cv2.CC_STAT_AREA])
            if length<3: continue
            x,y,width,height=map(int,stats[i,:4]); x0,x1=max(x-7,0),min(x+width+7,w); y0,y1=max(y-7,0),min(y+height+7,h)
            region=labels[y0:y1,x0:x1]==i; flank=cv2.dilate(region.astype(np.uint8),np.ones((15,15),np.uint8))>0
            local=old[y0:y1,x0:x1]; ground=gt[y0:y1,x0:x1]
            components=np.unique(local[flank&ground]); components=components[components>0]
            if len(components)<2: continue
            def indices(mask):
                yy,xx=np.nonzero(mask); return ((yy+y0)*w+xx+x0).tolist()
            gaps.append({'id':i,'length':length,'length_bin':'3-7' if length<8 else '8-15' if length<16 else '16-31' if length<32 else '32+',
                'pixels':indices(region),'anchors':[indices(flank&(local==component)&ground) for component in components]})
        total+=len(gaps); images[case['case']]={'shape':[h,w],'gaps':gaps}
    return {'version':VERSION,'count':total,'images':images,'sha256':reference_digest(cases),'p0_reference_threshold':.45,
        'origin':'This selected end-to-end model P0, frozen once for its P0/P1 threshold comparisons; NOT the old V9 206 gaps.'}


def area_threshold(cases,name,target):
    values=np.concatenate([case[name].reshape(-1) for case in cases]); target=max(0,min(int(target),len(values)))
    if target==0: threshold=float(np.nextafter(values.max(),np.inf,dtype=np.float32))
    elif target==len(values): threshold=float(values.min())
    else:
        cutoff=np.partition(values,len(values)-target)[len(values)-target]
        candidates=[float(cutoff),float(np.nextafter(cutoff,np.inf,dtype=np.float32))]
        threshold=min(candidates,key=lambda t:abs(int((values>=t).sum())-target))
    return threshold,int((values>=threshold).sum())-target


def evaluate(args,model,split):
    best=args.output_dir/'best.pth'; load_weights(model,args,best)
    folder=args.output_dir/'eval'/split; folder.mkdir(parents=True,exist_ok=True)
    if split=='test':
        selection=json.loads((args.output_dir/'eval/val_selection.json').read_text())
        if selection['checkpoint_sha256']!=digest(best): raise ValueError('Val selection belongs to different weights')
    cases=infer(args,model,split,True); manifest=fixed_reference(cases)
    atomic_json(folder/'fixed_gaps_and_anchors.json',manifest)
    if split=='val':
        curves={}
        for name in ('p0','p1'):
            curve=[]
            for threshold in np.round(np.arange(.05,.951,.05),2):
                curve.append(point_metrics(cases,name,float(threshold),manifest))
                print(f'[SWEEP] {name} {threshold:.2f}',flush=True)
            curves[name]=curve; atomic_json(folder/(name+'_threshold_curve.json'),curve)
        selected={name:max(curve,key=lambda row:row['selection_score'])['threshold'] for name,curve in curves.items()}
        selection={'checkpoint_sha256':digest(best),'selected_on':'val','thresholds':selected,'config':args.config}
        atomic_json(args.output_dir/'eval/val_selection.json',selection)
    else: selected=selection['thresholds']
    reports={name+'_own_val':point_metrics(cases,name,selected[name],manifest) for name in ('p0','p1')}
    common=selected['p0']
    for name in ('p0','p1'): reports[name+'_common']=point_metrics(cases,name,common,manifest)
    if split=='val':
        tp=sum(int(((case['p0']>=common)&case['gt']).sum()) for case in cases)
        area=sum(int((case['p0']>=common).sum()) for case in cases)
        recall_t,recall_error=exact_recall_threshold(cases,'p1',tp); area_t,area_error=area_threshold(cases,'p1',area)
        reports['p1_matched_recall']=dict(point_metrics(cases,'p1',recall_t,manifest),matching_error=recall_error)
        reports['p1_matched_area']=dict(point_metrics(cases,'p1',area_t,manifest),matching_error=area_error)
        selection.update(matched_recall_threshold=recall_t,matched_area_threshold=area_t)
        atomic_json(args.output_dir/'eval/val_selection.json',selection)
    else:
        # Test never uses its GT or area to choose any threshold.
        for label in ('matched_recall','matched_area'):
            reports['p1_val_'+label]=point_metrics(cases,'p1',selection[label+'_threshold'],manifest)
    rows=[]
    for name,row in reports.items():
        rows.append({'name':name,'threshold':row['threshold'],**{key:row['metrics'][key] for key in
            ('iou','f1','precision','recall','cldice','apls_gt_to_pred_approx','apls_pred_to_gt_approx','apls_bidirectional_approx','short_pred_components')},
            'closed_fixed_gaps':row['closed'],'reference_gaps':manifest['count'],'bridge_proxy':row['bridge_proxy'],'matching_error':row.get('matching_error')})
    atomic_json(folder/'summary.json',{'reports':reports,'reference_sha256':manifest['sha256'],'checkpoint_sha256':digest(best),
        'limits':'Whole-system end-to-end experiment, not a single-module ablation. APLS/closure/bridge are proxies; references differ from frozen V9.'})
    write_csv(folder/'comparison.csv',rows)
    print('[RESULTS] '+str(folder/'comparison.csv'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('smoke','train','resume','sweep','test','topology','all'))
    parser.add_argument('--root_path',type=Path,required=True); parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--pretrain_ckpt',type=Path); parser.add_argument('--device',default='cuda')
    parser.add_argument('--epochs',type=int,default=80); parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=1234); parser.add_argument('--activation_checkpoint',action='store_true')
    parser.add_argument('--lambda_choice',type=float,default=.05); parser.add_argument('--lambda_q',type=float,default=.1)
    parser.add_argument('--lambda_p0',type=float,default=1.)
    args=parser.parse_args(); args.device=torch.device(args.device)
    if args.epochs<1 or min(args.lambda_choice,args.lambda_q,args.lambda_p0)<0: parser.error('Invalid training budget/loss coefficients')
    args.output_dir.mkdir(parents=True,exist_ok=True); seed_all(args.seed)
    last=args.output_dir/'last.pth'; existing=args.output_dir/'experiment.json'
    if args.command=='train' and last.is_file(): parser.error('Run already has last.pth; use resume')
    if existing.is_file():
        saved=json.loads(existing.read_text())
        if saved['root_path']!=str(args.root_path.resolve()) or saved['epochs']!=args.epochs or saved['seed']!=args.seed:
            raise ValueError('Saved dataset/budget/seed differs')
        for key in ('lambda_choice','lambda_q','lambda_p0','activation_checkpoint'):
            if saved[key]!=getattr(args,key): raise ValueError(f'Experiment configuration changed: {key}')
        args.config=saved; loaded_names=set(saved['pretrained_loaded_names'])
        model=build_model(512,args.activation_checkpoint).to(args.device)
        if not last.is_file() and args.command in ('train','all','smoke'):
            # A preceding smoke is not a model checkpoint. Reapply identical
            # ImageNet initialization, never silently start from random weights.
            source=Path(saved['pretrain_path'])
            if not source.is_file() or digest(source)!=saved['pretrain_sha256']: raise ValueError('Missing/different ImageNet source for fresh training')
            config=model.original.upstream.backbone.segmenter.config.clone(); config.defrost(); config.MODEL.PRETRAIN_CKPT=str(source); config.freeze()
            actual=model.original.upstream.backbone.segmenter.load_from(config)
            if actual!=loaded_names: raise ValueError('ImageNet tensor mapping changed')
    else:
        if args.command not in ('train','smoke','all'): parser.error('Missing experiment.json; start with train or smoke')
        if args.pretrain_ckpt is None or not args.pretrain_ckpt.is_file(): parser.error('Fresh end-to-end training needs the ImageNet --pretrain_ckpt, not BASE/Q3')
        model=build_model(512,args.activation_checkpoint).to(args.device)
        config=model.original.upstream.backbone.segmenter.config.clone(); config.defrost(); config.MODEL.PRETRAIN_CKPT=str(args.pretrain_ckpt); config.freeze()
        loaded_names=model.original.upstream.backbone.segmenter.load_from(config)
        if not loaded_names: raise RuntimeError('ImageNet initialization loaded zero tensors')
        args.config={'version':VERSION,'root_path':str(args.root_path.resolve()),'epochs':args.epochs,'seed':args.seed,
            'num_workers':args.num_workers,'batch_size':2,'accumulation':1,'amp':'none','EMA':False,
            'source_size':1500,'tile_size':512,'stride':256,'random_crops_per_image':1,'val_interval':5,
            'warmup_epochs':10,'pretrained_lr':5e-5,'pretrained_min_lr':5e-6,'new_lr':2e-4,'new_min_lr':1e-5,
            'lambda_p0':args.lambda_p0,'lambda_q':args.lambda_q,'lambda_cont':.2,'lambda_relation':.1,'lambda_choice':args.lambda_choice,
            'activation_checkpoint':args.activation_checkpoint,'pretrained_loaded_names':sorted(loaded_names),
            'pretrain_path':str(args.pretrain_ckpt.resolve()),'pretrain_sha256':digest(args.pretrain_ckpt),
            'choice_steps':[2,3],'relation_steps':[1,2,3],'Q_target':'original tolerance2px, missing threshold0.5, min3px, dilation2px; max pool',
            'experiment_type':'whole-system slim end-to-end, not a single-module ablation',
            'code_sha256':{name:digest(Path(__file__).parent/name) for name in ('run_slim_end_to_end_mass.py','networks/slim_end_to_end.py',
                'networks/v9_source_selector.py','networks/v9_late_choice_loss.py','networks/v9_source_relation_labels.py',
                'networks/v9_selector_transport_training.py','networks/reliable_directional_repair.py','networks/q3_r3_repair.py',
                'networks/swin_transformer_unet_skip_expand_decoder_sys.py','networks/vision_transformer.py','networks/dca_fpn_lite.py')}}
        atomic_json(existing,args.config)
    for name,expected in args.config['code_sha256'].items():
        if digest(Path(__file__).parent/name)!=expected: raise ValueError('Code changed within experiment: '+name)
    optimizer,assigned=optimizer_for(model,loaded_names)
    atomic_json(args.output_dir/'parameters.json',{'total':sum(p.numel() for p in model.parameters()),
        'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'segmentation_backbone':sum(p.numel() for p in model.original.upstream.backbone.parameters()),
        'Q_and_feature_adapters':sum(p.numel() for name,p in model.original.upstream.named_parameters() if not name.startswith('backbone.')),
        'groups':{group['name']:sum(p.numel() for p in group['params']) for group in optimizer.param_groups},'pretrained_parameter_names':assigned})
    if args.command in ('smoke','train','all') and not (args.output_dir/'smoke.json').is_file(): smoke(args,model,optimizer)
    if args.command=='smoke': return
    if args.command in ('train','resume','all'):
        if args.command=='resume' and not last.is_file(): parser.error('Missing last.pth for resume')
        train(args,model,optimizer,last if last.is_file() else None)
    if args.command in ('sweep','all'): evaluate(args,model,'val')
    if args.command in ('test','topology'): evaluate(args,model,'test')


if __name__=='__main__': main()
