"""Frozen BASE/Q3 S_selector + optional RGB appearance; L_choice absent."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_v9_selector_transport as joint
from diagnose_selector_late_transport import load_current
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_selector_transport_training import VERSION as SOURCE_VERSION,configure_joint,canonical_state,load_canonical,live_relation_loss
from networks.v9_source_relation_labels import RelationLabels
from road_appearance import AppearanceConfig,AppearanceStats
from road_appearance_dataset import RoadAppearanceDataset
from export_road_appearance import export_examples
from run_v9_gate_benefit import atomic_json
from selector_transport_reference import reference_digest,exact_recall_threshold
from selector_transport_provenance import check_code_hashes,SELECTION_RULE
from diagnostics.v91_counterfactual import write_csv

VERSION='v9_frozen_road_appearance_v1'


def training_loss(args,result,gt,skeleton,seed):
    base,terms=dynamic_losses(result,gt,skeleton,**args.source_losses)
    relation,info=live_relation_loss(result,gt,skeleton,args.labels,seed)
    total=base+args.config['lambda_relation']*relation
    terms['relation']=relation.detach();terms['relation_counts']=info['matching']
    return total,terms


def save(args,model,optimizer,path,epoch,score):
    state={'version':VERSION,'config':args.config,'identity':args.identity,'epoch':epoch,'score':score,
        'modules':{name:canonical_state(getattr(model.original,name)) for name in joint.v9.MODULE_NAMES},
        'gate_relation':model.original.gate_relation.state_dict(),'state_gate':model.original.state_gate.state_dict(),
        'selector':model.selector.state_dict()}
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary=path.with_suffix('.tmp')
    try: torch.save(state,temporary);temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True);raise


def restore(args,model,path,optimizer=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['version']!=VERSION or state['config']!=args.config or state['identity']!=args.identity: raise ValueError('Appearance experiment identity/configuration differs')
    if set(state['modules'])!=set(joint.v9.MODULE_NAMES): raise ValueError('Incomplete repair checkpoint')
    for name in joint.v9.MODULE_NAMES: load_canonical(getattr(model.original,name),state['modules'][name])
    model.original.gate_relation.load_state_dict(state['gate_relation'],strict=True)
    model.original.state_gate.load_state_dict(state['state_gate'],strict=True)
    model.selector.load_state_dict(state['selector'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def smoke(args,model,optimizer):
    dataset=RoadAppearanceDataset(args.root_path,args.seed,AppearanceConfig(enabled=True));dataset.set_epoch(args.source_epoch+1)
    samples=[];clean=[]
    for index in range(min(len(dataset),100)):
        sample,debug=dataset.sample(index,force=('shadow','road'),debug=True,preview_salt='model-smoke')
        if not debug['metadata']['success']: continue
        samples.append(sample)
        before=(debug['before'].astype(np.float32)/255.-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        clean.append(torch.from_numpy(before.transpose(2,0,1).copy()))
        if len(samples)==2: break
    if len(samples)!=2: raise RuntimeError('Could not obtain two valid real augmented smoke samples')
    image,gt,skeleton=(torch.stack([sample[key] for sample in samples]).to(args.device) for key in ('image','mask','skeleton'))
    model.train();optimizer.zero_grad(set_to_none=True);seen=[]
    handle=model.original.frozen.register_forward_pre_hook(lambda unused,inputs:seen.append(inputs[0].detach()))
    if args.device.type=='cuda':torch.cuda.reset_peak_memory_stats(args.device)
    try: result=model(image,mode='source')
    finally: handle.remove()
    if len(seen)!=1 or not torch.equal(seen[0],image): raise RuntimeError('Frozen network did not read augmented RGB')
    if model.original.frozen.training or any(p.requires_grad for p in model.original.frozen.parameters()): raise RuntimeError('BASE/Q3 not fully frozen/eval')
    if result['p0'].requires_grad or result['q'].requires_grad: raise RuntimeError('Unexpected BASE/Q3 gradient path')
    total,terms=training_loss(args,result,gt,skeleton,args.seed);total.backward()
    if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite smoke loss')
    if any(p.grad is not None for p in model.original.frozen.parameters()): raise RuntimeError('Frozen model received gradients')
    gradients=joint.parameter_gradients(optimizer)
    if gradients['selector']<=0: raise RuntimeError('Repair selector did not receive gradients')
    with torch.no_grad(): clean_result=model(torch.stack(clean).to(args.device),mode='source')
    report={'passed':True,'image_shape':list(image.shape),'model_input_keys':['image'],'debug_mask_enters_model':False,
        'frozen_forward_reads_augmented_rgb':True,'BASE_Q3_eval':True,'BASE_Q3_gradients':False,'lambda_choice':0,
        'P0_augmented_vs_clean_mean_abs':float((result['p0']-clean_result['p0']).abs().mean()),
        'Q_augmented_vs_clean_mean_abs':float((result['q']-clean_result['q']).abs().mean()),'gradients':gradients,
        'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None}
    atomic_json(args.output_dir/'smoke.json',report);optimizer.zero_grad(set_to_none=True)
    print('[FROZEN AUGMENTED SMOKE] '+json.dumps(report),flush=True)


@torch.no_grad()
def infer(args,model,split='val',save_predictions=False):
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(args.root_path,split);model.eval();cases=[]
    if len(dataset)!=(14 if split=='val' else 49): raise ValueError('Mass split cardinality changed')
    folder=args.output_dir/('predictions_'+split)
    if save_predictions:folder.mkdir(parents=True,exist_ok=True)
    for index in range(len(dataset)):
        sample=dataset[index]
        p0,p1=joint.stitch(model,sample['image'][None].to(args.device),args.device)
        case={'case':str(sample['case_name']),'p0':p0,'p1':p1,'gt':sample['mask'][0].numpy()>.5}
        cases.append(case)
        if save_predictions:
            file=folder/f'{index:02d}.npz';np.savez_compressed(file,p0=p0,p1=p1,gt=case['gt'])
            atomic_json(file.with_suffix('.json'),{'case':case['case'],'checkpoint_sha256':joint.v9.digest(args.output_dir/'best.pth'),'appearance_at_eval':False})
        print(f'[CLEAN {split.upper()}] {index+1}/{len(dataset)}',flush=True)
    return cases


def train(args,model,optimizer,manifest):
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    start=1;best=-float('inf');last=args.output_dir/'last.pth'
    if last.is_file():
        state=restore(args,model,last,optimizer);start=state['epoch']+1;best=state['score']
        print(f'[RESUME] completed={state["epoch"]}; target={args.epochs}; next={start}',flush=True)
    dataset=RoadAppearanceDataset(args.root_path,args.seed,args.appearance)
    for epoch in range(start,args.epochs+1):
        crop_epoch=args.source_epoch+epoch;dataset.set_epoch(crop_epoch)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,pin_memory=args.device.type=='cuda',
            generator=torch.Generator().manual_seed(args.seed+crop_epoch*1000003))
        model.train();model.set_accept_floor(0.);stats=AppearanceStats();totals={key:0. for key in ('total','seg','cont','relation')}
        order=hashlib.sha256();base_inputs=hashlib.sha256();augmented=hashlib.sha256();batch_count=0;wait_seconds=iteration_seconds=0.;clock=time.perf_counter()
        with open(args.output_dir/f'appearance_epoch_{epoch:03d}.jsonl','w',encoding='utf-8') as records:
            for index,batch in enumerate(loader):
                arrived=time.perf_counter();wait_seconds+=arrived-clock
                for raw in batch['appearance_metadata']:
                    info=json.loads(raw);stats.add(info);records.write(json.dumps(info)+'\n')
                    stable={key:info[key] for key in ('image_name','crop_ordinal','epoch','tile_top','tile_left','base_geometry','before_local_sha256','road_sha256','skeleton_sha256')}
                    base_inputs.update(json.dumps(stable,sort_keys=True).encode()+b'\0')
                records.flush()
                for name in batch['case_name']: order.update(str(name).encode()+b'\0')
                augmented.update(batch['image'].contiguous().numpy().tobytes())
                image,gt,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
                optimizer.zero_grad(set_to_none=True);result=model(image,mode='source')
                total,terms=training_loss(args,result,gt,skeleton,args.seed+crop_epoch*100000+index)
                if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite training loss')
                total.backward()
                if any(p.grad is not None for p in model.original.frozen.parameters()): raise RuntimeError('BASE/Q3 received gradients')
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite gradient')
                optimizer.step();totals['total']+=float(total.detach())
                for key in ('seg','cont','relation'):totals[key]+=float(terms[key])
                batch_count+=1;clock=time.perf_counter();iteration_seconds+=clock-arrived
                if index%25==0:print(f'[APPEARANCE TRAIN] {epoch}/{args.epochs} batch={index+1}/{len(loader)} loss={float(total.detach()):.6f} applied={stats.counts["success"]}/{stats.counts["tiles"]}',flush=True)
        record={'epoch':epoch,'crop_epoch':crop_epoch,'losses':{key:value/max(batch_count,1) for key,value in totals.items()},
            'appearance':stats.report(),'case_order_sha256':order.hexdigest(),'base_inputs_sha256':base_inputs.hexdigest(),
            'actual_model_images_sha256':augmented.hexdigest(),'data_wait_seconds':wait_seconds,
            'mean_iteration_seconds':iteration_seconds/max(batch_count,1),'lambda_choice':0}
        if epoch%args.val_interval==0 or epoch==args.epochs:
            cases=infer(args,model)
            if reference_digest(cases)!=manifest['sha256']:raise ValueError('Clean validation frozen P0 reference changed')
            val=joint.point_metrics(cases,'p1',.6,manifest);record['val_score_at_0_60']=val['selection_score']
            atomic_json(args.output_dir/f'val_epoch_{epoch:03d}.json',val)
            if val['selection_score']>best:
                best=val['selection_score'];save(args,model,None,args.output_dir/'best.pth',epoch,best)
        atomic_json(args.output_dir/f'epoch_{epoch:03d}.json',record);save(args,model,optimizer,last,epoch,best)
    atomic_json(args.output_dir/'training_complete.json',{'epochs':args.epochs,'best_score':best,'appearance_only_in_train':True})
    print(f'[TRAIN COMPLETE] {args.epochs}/{args.epochs}',flush=True)


def evaluate(args,model,manifest,split='val'):
    restore(args,model,args.output_dir/'best.pth');cases=infer(args,model,split,True)
    if split=='val':
        if reference_digest(cases)!=manifest['sha256']:raise ValueError('Frozen P0/GT reference differs')
        curve=[]
        for t in np.round(np.arange(.35,.751,.01),2):
            curve.append(joint.point_metrics(cases,'p1',float(t),manifest))
            atomic_json(args.output_dir/'threshold_curve.json',curve)
            print(f'[APPEARANCE VAL THRESHOLD] {t:.2f}',flush=True)
        selected=max(curve,key=lambda row:row['selection_score'])
        reports={'p0_common':joint.point_metrics(cases,'p0',args.common_threshold,manifest),
            'p1_common':joint.point_metrics(cases,'p1',args.common_threshold,manifest),'p1_own_val':selected}
        atomic_json(args.output_dir/'threshold_curve.json',curve)
        atomic_json(args.output_dir/'val_selection.json',{'threshold':selected['threshold'],'checkpoint_sha256':joint.v9.digest(args.output_dir/'best.pth'),'selected_on':'val'})
    else:
        # Test is final-only and uses val-selected thresholds, never test tuning.
        selection=json.loads((args.output_dir/'val_selection.json').read_text())
        if selection['checkpoint_sha256']!=joint.v9.digest(args.output_dir/'best.pth'):raise ValueError('Val selection belongs to other weights')
        from run_slim_end_to_end_mass import fixed_reference
        manifest=fixed_reference(cases)
        reports={'p0_common':joint.point_metrics(cases,'p0',args.common_threshold,manifest),
            'p1_common':joint.point_metrics(cases,'p1',args.common_threshold,manifest),
            'p1_val_selected':joint.point_metrics(cases,'p1',selection['threshold'],manifest)}
    atomic_json(args.output_dir/(split+'_summary.json'),{'reports':reports,'appearance_at_eval':False,'lambda_choice':0})
    write_csv(args.output_dir/(split+'_comparison.csv'),[
        {'name':name,'threshold':value['threshold'],**{key:value['metrics'][key] for key in ('iou','f1','precision','recall','cldice','apls_gt_to_pred_approx','apls_bidirectional_approx','short_pred_components')},
         'closed_fixed_gaps':value['closed'],'bridge_proxy':value['bridge_proxy']} for name,value in reports.items()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preview','smoke','train','resume','val','test','all'))
    parser.add_argument('--run_dir',type=Path,required=True,help='Original pre-L_choice actual_transport S_selector run')
    parser.add_argument('--output_dir',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--threshold',type=float)
    parser.add_argument('--epochs',type=int);parser.add_argument('--road_appearance_aug',action='store_true')
    parser.add_argument('--preview_dir',type=Path)
    args=parser.parse_args();args.device=torch.device(args.device);args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.checkpoint or args.run_dir/'best.pth';source=torch.load(path,map_location='cpu',weights_only=False)
    if source.get('version')!=SOURCE_VERSION or source['config'].get('lambda_choice',0)!=0:raise ValueError('Must start from original actual-transport S_selector, NOT late_choice or untrained slim')
    check_code_hashes(source['config']);args.source_epoch=source['epoch'];cfg=source['config']
    args.seed=cfg['seed'];args.root_path=Path(cfg['root_path']);args.num_workers=cfg['num_workers'];args.val_interval=cfg['val_interval']
    args.epochs=args.epochs if args.epochs is not None else cfg['epochs']
    if args.epochs<1:parser.error('Positive epoch budget required')
    args.appearance=AppearanceConfig(enabled=args.road_appearance_aug)
    preview=args.preview_dir or args.output_dir/'preview24'
    if args.command in ('all','train','resume','preview','smoke'):
        report=preview/'preview_report.json'
        if report.is_file():
            saved=json.loads(report.read_text())
            if not saved['passed'] or saved['root_path']!=str(args.root_path.resolve()) or saved['epoch']!=args.source_epoch+1 or saved['config']!=json.loads(json.dumps(asdict(AppearanceConfig(enabled=True)))):
                raise ValueError('Existing preview does not match this training dataset/configuration')
        else:export_examples(args.root_path,preview,args.seed,args.source_epoch+1)
    if args.command=='preview':return
    model,source,path,manifest,args.common_threshold,origin=load_current(args)
    args.identity=source['identity'];args.source_losses=cfg['source_losses'];args.labels=RelationLabels(**cfg['labels'])
    rates={key:value['lr'] for key,value in cfg['optimizer'].items()};rates['gate_relation']=rates['gate_feature_projections_and_pairs']
    optimizer,groups=configure_joint(model,rates,rates['selector'])
    args.config={'version':VERSION,'source_checkpoint':str(path.resolve()),'source_sha256':joint.v9.digest(path),'identity':args.identity,
        'root_path':str(args.root_path.resolve()),'epochs':args.epochs,'seed':args.seed,'source_epoch':args.source_epoch,
        'num_workers':args.num_workers,'val_interval':args.val_interval,'batch_size':2,'accumulation':1,'amp':'none','EMA':False,
        'source_losses':args.source_losses,'lambda_relation':cfg['lambda_relation'],'lambda_choice':0,'optimizer':groups,
        'appearance':json.loads(json.dumps(asdict(args.appearance))),'pipeline':'native crop512 -> original geometry -> original color -> appearance -> normalization',
        'common_threshold':args.common_threshold,'threshold_origin':origin,'checkpoint_threshold':.6,'selection_rule':SELECTION_RULE,
        'code_sha256':{name:joint.v9.digest(Path(__file__).parent/name) for name in ('run_v9_road_appearance.py','road_appearance.py','road_appearance_dataset.py','export_road_appearance.py')}}
    marker=args.output_dir/'experiment.json'
    if marker.is_file() and json.loads(marker.read_text())!=args.config:raise ValueError('Appearance run configuration changed')
    atomic_json(marker,args.config)
    for name in ('fixed_gaps_and_anchors.json','fixed_reference_binding.json'):shutil.copyfile(args.run_dir/name,args.output_dir/name)
    if args.command=='train' and (args.output_dir/'last.pth').is_file():raise ValueError('Existing run: use resume/all')
    if args.command=='resume' and not (args.output_dir/'last.pth').is_file():raise ValueError('No last.pth to resume')
    if args.command in ('all','train','resume','smoke') and not (args.output_dir/'smoke.json').is_file():smoke(args,model,optimizer)
    if args.command=='smoke':return
    if args.command in ('all','train','resume'):train(args,model,optimizer,manifest)
    if args.command in ('all','val'):evaluate(args,model,manifest)
    if args.command=='test':evaluate(args,model,manifest,'test')
    if joint.v9.digest(path)!=args.config['source_sha256']:raise RuntimeError('Input S checkpoint changed during run')


if __name__=='__main__':main()
