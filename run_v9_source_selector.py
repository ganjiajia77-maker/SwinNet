"""Staged source-relation experiment: fixed V9 scoring first, repair only after evidence."""
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
from run_v9_gate_benefit import atomic_json, load_original
from networks.reliable_directional_repair import DIRECTIONS, shifted, dynamic_losses
from networks.v9_source_selector import SourceSelectedRepair, source_support, relation_loss, VERSION
from networks.v9_source_relation_labels import RelationLabels,native_relations,eligible_relations,match_relations
from networks.v9_fixed_action_audit import compare_scores,bootstrap_difference
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnostics.v91_counterfactual import write_csv


def checkpoint(path,model,args,phase,epoch,score,optimizer=None,record=None):
    original=model.original
    payload={'version':VERSION,'identity':args.identity,'source_sha256':args.source_sha,
        'phase':phase,'epoch':epoch,'score':score,'config':args.experiment_config,
        'modules':{name:getattr(original,name).state_dict() for name in v9.MODULE_NAMES},
        'gate_relation':original.gate_relation.state_dict(),'state_gate':original.state_gate.state_dict(),
        'selector':model.selector.state_dict(),'epoch_record':record,
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if optimizer is not None: payload['optimizer']=optimizer.state_dict()
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary); temporary.replace(path)


def restore(path,model,args,phase=None):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('version')!=VERSION or value.get('identity')!=args.identity or value.get('source_sha256')!=args.source_sha:
        raise ValueError('Selector checkpoint version/upstream/source identity differs')
    if value.get('config')!=args.experiment_config or phase is not None and value.get('phase')!=phase:
        raise ValueError('Selector checkpoint experiment configuration/phase differs')
    if set(value.get('modules',{}))!=set(v9.MODULE_NAMES): raise ValueError('Incomplete V9 checkpoint')
    for name in v9.MODULE_NAMES: getattr(model.original,name).load_state_dict(value['modules'][name],strict=True)
    model.original.gate_relation.load_state_dict(value['gate_relation'],strict=True)
    model.original.state_gate.load_state_dict(value['state_gate'],strict=True)
    model.selector.load_state_dict(value['selector'],strict=True)
    return value


def configure(model,phase,args):
    model.set_phase(phase)
    groups=[{'params':[p for p in model.selector.parameters() if p.requires_grad],'lr':args.selector_lr,'name':'selector'}]
    if phase=='joint':
        groups.append({'params':[p for p in model.original.parameters() if p.requires_grad],
                       'lr':args.joint_lr,'name':'existing_downstream'})
    actual=[id(p) for group in groups for p in group['params']]
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if set(actual)!=expected or len(actual)!=len(set(actual)): raise RuntimeError('Incorrect optimizer coverage')
    return torch.optim.AdamW(groups,weight_decay=.01)


def grad_report(model):
    result={}
    for name,module in (('edge',model.selector.edge),('stay',model.selector.stay),
                        ('original',model.original),('BASE_Q3',model.original.frozen)):
        pieces=[p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
        result[name]=float(torch.stack(pieces).sum().sqrt()) if pieces else 0.
    return result


def smoke(args,model):
    dataset=RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=False)
    dataset.set_epoch(1); batch=next(iter(DataLoader(dataset,batch_size=2,num_workers=0)))
    image,gt,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
    if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
    model.eval()
    with torch.no_grad():
        old=model.original(image,capture_diagnostics=True); reference=model(image,mode='scalar')
        if not torch.equal(old['p1'],reference['p1']): raise RuntimeError('Scalar mode no longer reproduces original V9')
        native=native_relations(gt,skeleton,reference['r'].shape[-2:],args.labels)
        rows=eligible_relations(reference,native,args.labels); selected,counts=match_relations(rows,args.labels,args.seed)
    before={key:value.detach().clone() for key,value in model.selector.state_dict().items()}
    optimizer=configure(model,'relation',args); optimizer.zero_grad(set_to_none=True)
    loss,_,count=relation_loss(model.selector,reference['selector_context'],reference,rows['labels'],selected)
    loss.backward(); relation_grad=grad_report(model)
    if relation_grad['original']!=0. or relation_grad['BASE_Q3']!=0.: raise RuntimeError('Fixed-input relation training leaked gradients upstream')
    optimizer=configure(model,'repair',args); optimizer.zero_grad(set_to_none=True)
    result=model(image); total,terms=dynamic_losses(result,gt,skeleton,**args.source_losses)
    if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite selector repair loss')
    total.backward(); repair_grad=grad_report(model)
    if repair_grad['BASE_Q3']!=0. or repair_grad['original']!=0.: raise RuntimeError('Frozen V9 received repair gradients')
    if repair_grad['edge']==0. or repair_grad['stay']==0.: raise RuntimeError('Selector edge/stay cannot receive Surface gradients')
    choice=result['source_choice']
    torch.testing.assert_close(choice.sum(2),torch.ones_like(choice[:,:,0]),atol=2e-6,rtol=1e-6)
    for k in range(3):
        u=result['states'][k]
        expected=result['transport_rate'][:,k]*(choice[:,k,1:]*(result['source_states'][:,k]-u)).sum(1,keepdim=True)
        torch.testing.assert_close(expected,result['transport_delta'][:,k])
    model.zero_grad(set_to_none=True); model.selector.load_state_dict(before,strict=True); model.eval()
    report={'passed':True,'physical_batch':len(image),'work_shape':list(result['r'].shape),
        'source_choice_shape':list(choice.shape),'directions':DIRECTIONS,
        'selector_parameters':sum(p.numel() for p in model.selector.parameters()),
        'original_parameters':sum(p.numel() for p in model.original.parameters()),
        'native_label_counts':{str(value):int((native['labels']==value).sum()) for value in (-1,0,1)},
        'matching':counts,'relation_labeled_samples':count,'relation_gradients':relation_grad,
        'repair_gradients':repair_grad,'legacy_scalar_bitwise_equal':True,'extra_scalar_gate_used':False,
        'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None,
        'limitation':'Native local-corridor labels are proxies. No matched samples in this smoke batch means relation-gradient evidence is unavailable; empty training epochs fail.'}
    atomic_json(args.output_dir/'smoke.json',report); print('[SMOKE] '+json.dumps(report),flush=True)


def relation_scores(model,reference):
    context=reference['selector_context']; baseline,_=original_edge_terms(reference); edge=[]; choice=[]
    for k in range(3):
        state=reference['states'][k]; support=source_support(state,context['p0'],context['r'])
        prediction=model.selector(context,state,support,reference['source_available'][:,k],reference['transport_rate'][:,k])
        edge.append(prediction['edge_logits']); choice.append(prediction['choice'][:,1:].clamp_min(1e-12).log())
    return baseline.clamp_min(1e-12).log(),torch.stack(edge,1),torch.stack(choice,1)


def original_edge_terms(result):
    weights=[]; updates=[]; p0=result['u0']
    for k in range(3):
        state=result[f'u{k}']; support=source_support(state,p0,result['r']); mass=result['source_mass'][:,k]
        edge_weights=[]; edge_updates=[]
        for d,(dy,dx) in enumerate(DIRECTIONS):
            fraction=torch.where(mass>1e-6,result['w'][:,d:d+1]*shifted(support,dy,dx)/mass.clamp_min(1e-6),torch.zeros_like(mass))
            value=result['g_step'][:,k]*fraction
            edge_weights.append(value)
            edge_updates.append(result['transport_rate'][:,k]*value*(shifted(state,dy,dx)-state))
        weights.append(torch.cat(edge_weights,1)); updates.append(torch.cat(edge_updates,1))
    return torch.stack(weights,1),torch.stack(updates,1)


@torch.no_grad()
def audit(args,model,output,save_scores=True):
    """Fixed original V9 inputs, full14 val images, all tiles, shared edge rows."""
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(args.root_path,'val')
    if len(dataset)!=14: raise ValueError('Expected the same 14 validation images')
    output.mkdir(parents=True,exist_ok=True); model.eval()
    collected={key:[] for key in ('label','baseline','edge','choice','matched','image','step','direction','negative_type','native_y','native_x')}
    counts=Counter(); mapping=None; case_ids=[]
    for index in range(len(dataset)):
        sample=dataset[index]; case_ids.append(str(sample['case_name'])); image=sample['image'][None].to(args.device); height,width=image.shape[-2:]
        owned=np.zeros((height,width),bool)
        for y in range_positions(height):
            for x in range_positions(width):
                # The GT never enters this forward; only the native label function below.
                reference=model(image[:,:,y:y+512,x:x+512],mode='scalar')
                mask=sample['mask'][None,:,y:y+512,x:x+512].to(args.device)
                skeleton=sample['skeleton'][None,:,y:y+512,x:x+512].to(args.device)
                native=native_relations(mask,skeleton,reference['r'].shape[-2:],args.labels); mapping=native['native_stride']
                h,w=reference['r'].shape[-2:]
                yy=np.minimum(((np.arange(h)+.5)*512/h).astype(int),511); xx=np.minimum(((np.arange(w)+.5)*512/w).astype(int),511)
                own=(~owned[y:y+512,x:x+512])[yy[:,None],xx[None,:]]; owned[y:y+512,x:x+512]=True
                rows=eligible_relations(reference,native,args.labels,torch.as_tensor(own,device=args.device)[None,None])
                selected,matching=match_relations(rows,args.labels,args.seed+index*100000+y*1000+x)
                counts.update(matching)
                for value in (-1,0,1): counts['native_'+str(value)]+=int((native['labels']==value).sum())
                base,edge,choice=relation_scores(model,reference)
                take=rows['valid'].cpu().numpy(); indexes=np.nonzero(take); n=int(take.sum())
                for key,tensor in (('label',rows['labels']),('baseline',base),('edge',edge),('choice',choice),('matched',selected)):
                    collected[key].append(tensor.cpu().numpy()[take])
                for key,value in (('image',np.full(n,index,np.int8)),('step',indexes[1].astype(np.int8)+1),('direction',indexes[2].astype(np.int8))):
                    collected[key].append(value)
                collected['native_y'].append((y+yy[indexes[3]]).astype(np.int16))
                collected['native_x'].append((x+xx[indexes[4]]).astype(np.int16))
                kind=native['negative_type'][:,None].expand(-1,3,-1,-1,-1).cpu().numpy()
                collected['negative_type'].append(kind[take])
        print(f'[SOURCE RELATION VAL] {index+1}/14 {sample["case_name"]}',flush=True)
    data={key:np.concatenate(value) for key,value in collected.items()}
    if not len(data['label']): raise RuntimeError('No valid fixed relations in validation')
    reports={}; table=[]
    for population in ('natural','matched'):
        select=np.ones(len(data['label']),bool) if population=='natural' else data['matched'].astype(bool)
        for scorer in ('edge','choice'):
            for group,mask in [('all',select),*[(f'step{k}',select&(data['step']==k)) for k in (1,2,3)],
                               *[(f'direction{d}',select&(data['direction']==d)) for d in range(8)]]:
                result=compare_scores(data['label'][mask].astype(bool),data['baseline'][mask],data[scorer][mask],(.8,.9))
                raw=result.pop('raw_at_G_0_5')
                result['ranking']={'original_V9':raw['B_control']['auc'],'source_selector':raw['C_gate3']['auc']}
                result['score_note']='Compare rank/acceptance curves, not a shared zero threshold: original per-source weights and edge logits are different score parameterizations.'
                if group=='all':
                    result['bootstrap']=bootstrap_difference(data['label'][mask].astype(bool),data['baseline'][mask],data[scorer][mask],data['image'][mask],(.8,.9),args.bootstrap,args.seed)
                    reports[population+'_'+scorer]=result
                for row in result['matched_helpful']:
                    if row['B']['available'] and row['C']['available']:
                        table.append({'population':population,'scorer':scorer,'group':group,'correct_relations':result['helpful'],
                            'incorrect_relations':result['harmful'],'correct_acceptance':row['helpful_target'],
                            'V9_wrong_acceptance':row['B']['harmful_acceptance'],'selector_wrong_acceptance':row['C']['harmful_acceptance'],
                            'difference':row['harmful_C_minus_B']})
    screen=selection_screen(reports)
    score=-np.mean([row['harmful_C_minus_B']+row['B']['harmful_acceptance'] for row in reports['matched_edge']['matched_helpful']
                    if row['helpful_target'] in (.8,.9) and row['B']['available']]) if reports['matched_edge']['helpful'] and reports['matched_edge']['harmful'] else None
    atomic_json(output/'relations.json',{'reports':reports,'counts':dict(counts),'native_pixels_per_work_cell':mapping,
        'directions':DIRECTIONS,'case_ids':case_ids,'negative_types':{1:'blocked_native_road_pair',2:'background_extension',3:'background_response'},
        'negative_type_counts':{str(value):int(((data['label']==0)&(data['negative_type']==value)).sum()) for value in (1,2,3)},
        'screening':screen,'selection_score':float(score) if score is not None else None,
        'scope':'Fixed ORIGINAL V9 states/actions and identical relation rows. Baseline is the actual old per-source G*W*SourceSupport/mass. Relation labels are native local-corridor GT proxies, not BCE benefits.'})
    write_csv(output/'matched_correct_relations.csv',table)
    if save_scores: np.savez_compressed(output/'relation_scores.npz',**data)
    lines=['# Source relation discrimination','',
        '| Population | Scorer | Correct acceptance | V9 wrong acceptance | Selector wrong acceptance | Difference |',
        '|---|---|---:|---:|---:|---:|']
    for row in table:
        if row['group']=='all' and row['correct_acceptance'] in (.8,.9):
            lines.append(f'| {row["population"]} | {row["scorer"]} | {row["correct_acceptance"]:.0%} | {row["V9_wrong_acceptance"]:.3%} | {row["selector_wrong_acceptance"]:.3%} | {row["difference"]:+.3%} |')
    lines+=['',f'Selection screen: {screen}', '']
    for population in ('natural','matched'):
        result=reports[population+'_edge']; old=result['ranking']['original_V9']; new=result['ranking']['source_selector']
        lines.append(f'{population}: {result["helpful"]} positive / {result["harmful"]} negative; '
            f'ROC-AUC {old["roc_auc"]} -> {new["roc_auc"]}; PR-AUC {old["pr_auc"]} -> {new["pr_auc"]}.')
    for row in reports['natural_edge']['bootstrap']:
        lines.append(f'Image-bootstrap wrong-acceptance difference at correct={row["helpful_target"]:.0%}: {row["delta_harmful_C_minus_B_ci95"]}.')
    lines+=['',
        'edge is the shared per-neighbour relation logit; choice is its actual nine-way softmax source weight. '
        'All rows come from the same frozen original V9, not separately generated selectors.',
        'These are native short-corridor proxy labels. Correct relation acceptance does not force positive state transport and does not prove final road closure.',
        'Junctions, uncertain labels and boundaries are excluded. Raw label counts include overlapping tiles; scored relations use unique work-cell ownership.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'screening':screen,'score':float(score) if score is not None else None,'reports':reports}


def range_positions(length):
    from mass_protocol import sliding_positions
    return sliding_positions(length,512,256)


def selection_screen(reports):
    checks={}
    for population in ('natural','matched'):
        edge=reports[population+'_edge']; choice=reports[population+'_choice']
        for target in (.8,.9):
            e=next((row for row in edge['matched_helpful'] if row['helpful_target']==target),None)
            c=next((row for row in choice['matched_helpful'] if row['helpful_target']==target),None)
            checks[f'{population}_fewer_edge_errors_at_{target}']=bool(e and e['harmful_C_minus_B'] is not None and e['harmful_C_minus_B']<0)
            checks[f'{population}_no_worse_choice_errors_at_{target}']=bool(c and c['harmful_C_minus_B'] is not None and c['harmful_C_minus_B']<=0)
        old=edge['ranking']['original_V9']['roc_auc']; new=edge['ranking']['source_selector']['roc_auc']
        checks[population+'_edge_ROC_AUC_improved']=old is not None and new is not None and new>old
    intervals=reports['natural_edge']['bootstrap']
    checks['natural_edge_image_CI_below_zero']=any(row['delta_harmful_C_minus_B_ci95'][1]<0 for row in intervals)
    return {'passed':all(checks.values()),'checks':checks,
        'rule':'Fixed relations: at .8/.9 correct acceptance, fewer wrong accepts for edge logits in natural and matched populations, no worse actual choice weights, higher edge ROC-AUC; at least one natural image-bootstrap interval excludes zero.',
        'note':'Conservative val screening, not final topology improvement. No scalar G shrinkage criterion.'}


def train(args,model,phase):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    output=args.output_dir/phase; output.mkdir(parents=True,exist_ok=True)
    budget=args.relation_epochs if phase=='relation' else args.repair_epochs
    if phase!='relation' and not (output/'last.pth').is_file():
        prior='relation' if phase=='repair' else 'repair'
        if not (args.output_dir/prior/'best.pth').is_file(): raise ValueError(f'Run {prior} first')
        if phase=='repair':
            screen=json.loads((args.output_dir/'relation_best/relations.json').read_text())['screening']
            if not screen['passed']: raise ValueError('Fixed-input relation selection has not passed; inspect labels/features before repair training')
        restore(args.output_dir/prior/'best.pth',model,args,prior)
    elif phase=='relation': v9.restore_checkpoint(model.original,args.source,args.identity,'D')
    optimizer=configure(model,phase,args); start=1; best=-float('inf')
    if (output/'last.pth').is_file():
        state=restore(output/'last.pth',model,args,phase); optimizer.load_state_dict(state['optimizer'])
        start=state['epoch']+1; best=state['score']; torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        if state.get('epoch_record') is not None:
            atomic_json(output/f'epoch_{state["epoch"]:03d}.json',state['epoch_record'])
    dataset=RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=True)
    for epoch in range(start,budget+1):
        dataset.set_epoch(epoch); generator=torch.Generator().manual_seed(args.seed+epoch*1000003)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,generator=generator,pin_memory=args.device.type=='cuda')
        model.train(); sums=Counter(); gradients=None; order=hashlib.sha256(); image_hash=hashlib.sha256()
        for index,batch in enumerate(loader):
            for case in batch['case_name']: order.update(str(case).encode()+b'\0')
            image_hash.update(batch['image'].contiguous().numpy().tobytes())
            image,gt,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
            optimizer.zero_grad(set_to_none=True)
            if phase=='relation':
                with torch.no_grad():
                    reference=model(image,mode='scalar'); native=native_relations(gt,skeleton,reference['r'].shape[-2:],args.labels)
                    rows=eligible_relations(reference,native,args.labels); selected,counts=match_relations(rows,args.labels,args.seed+epoch*100000+index)
                loss,_,count=relation_loss(model.selector,reference['selector_context'],reference,rows['labels'],selected)
                sums.update(counts); sums['labeled_samples']+=count
            else:
                result=model(image,mode='source'); loss,terms=dynamic_losses(result,gt,skeleton,**args.source_losses)
                sums['seg']+=float(terms['seg']); sums['cont']+=float(terms['cont'])
                sums['stay_mean']+=float(result['source_choice'][:,:,0].detach().mean())
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite source selector training loss')
            loss.backward()
            if any(p.grad is not None for p in model.original.frozen.parameters()): raise RuntimeError('BASE/Q3 received gradients')
            if phase in ('relation','repair') and any(p.grad is not None for p in model.original.parameters()): raise RuntimeError('Frozen V9 received gradients')
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite gradient')
            if gradients is None and (phase!='relation' or count): gradients=grad_report(model)
            # No weight-decay-only update when this batch has no relation labels.
            if phase!='relation' or count: optimizer.step()
            sums['loss']+=float(loss.detach()); sums['batches']+=1
            if index%25==0: print(f'[SOURCE SELECTOR TRAIN] {phase} epoch={epoch}/{budget} batch={index+1}/{len(loader)} loss={float(loss.detach()):.6f}',flush=True)
        record={'epoch':epoch,'phase':phase,'totals':dict(sums),'gradients_first_valid_batch':gradients,
            'case_order_sha256':order.hexdigest(),'augmented_images_sha256':image_hash.hexdigest()}
        if phase=='relation' and sums['labeled_samples']==0:
            atomic_json(output/'failed_no_labels.json',record); raise RuntimeError('No matched relations in training epoch; no successful checkpoint saved')
        if phase=='relation':
            measured=audit(args,model,output/f'val_epoch_{epoch:03d}',save_scores=False); score=measured['score']
            if score is None: raise RuntimeError('No valid two-class validation relation population')
            record['val_relation_score']=score
        else:
            measured,_,_,_=evaluate_thresholds(model,full_dataset(args.root_path,'val'),args.device,(.6,),reference_threshold=.45)
            score=float(measured[.6]['selection_score']); atomic_json(output/f'val_epoch_{epoch:03d}.json',measured[.6])
            record['val_surface_score_at_0_60']=score
        if score>best: best=score; checkpoint(output/'best.pth',model,args,phase,epoch,best)
        checkpoint(output/'last.pth',model,args,phase,epoch,best,optimizer,record)
        atomic_json(output/f'epoch_{epoch:03d}.json',record)
        print(f'[SOURCE SELECTOR VAL] {phase} epoch={epoch} score={score:.6f}',flush=True)
    atomic_json(output/'training_complete.json',{'epochs':budget,'best_score':best,'phase':phase})


@torch.no_grad()
def compare_surface(args,model,phase):
    from run_q3_r3_mass import full_dataset
    from mass_protocol import blend_weight
    from gate_benefit_protocol import create_variant,threshold_curve,operating_points
    from diagnostics.v91_counterfactual import evaluate
    baseline=create_variant(model.original.frozen,args.source,args.identity,args.device)
    model.eval(); cases=[]; output=args.output_dir/(phase+'_surface_val'); output.mkdir(parents=True,exist_ok=True)
    relation_stats={}
    dataset=full_dataset(args.root_path,'val')
    if len(dataset)!=14: raise ValueError('Expected 14 full validation images')
    for index in range(len(dataset)):
        sample=dataset[index]; image=sample['image'][None].to(args.device); gt=sample['mask'][0].numpy()>.5; h,w=gt.shape
        canvas=torch.zeros((3,1,h,w),device=args.device); divisor=torch.zeros((1,1,h,w),device=args.device); weight=blend_weight(512,args.device)
        owned=np.zeros((h,w),bool)
        for y in range_positions(h):
            for x in range_positions(w):
                frozen=model.original.frozen(image[:,:,y:y+512,x:x+512],capture_diagnostics=True)
                old=baseline(None,capture_diagnostics=True,frozen_override=frozen)
                new=model(None,mode='source',frozen_override=frozen)
                if not torch.equal(old['z0'],new['z0']): raise ValueError('Frozen P0 differs')
                shape=new['r'].shape[-2:]
                native=native_relations(sample['mask'][None,:,y:y+512,x:x+512].to(args.device),
                    sample['skeleton'][None,:,y:y+512,x:x+512].to(args.device),shape,args.labels)
                yy=np.minimum(((np.arange(shape[0])+.5)*512/shape[0]).astype(int),511)
                xx=np.minimum(((np.arange(shape[1])+.5)*512/shape[1]).astype(int),511)
                own=(~owned[y:y+512,x:x+512])[yy[:,None],xx[None,:]]; owned[y:y+512,x:x+512]=True
                own=torch.as_tensor(own,device=args.device)[None,None]
                old_weights,old_updates=original_edge_terms(old)
                for name,weights,updates in (('A_original',old_weights,old_updates),
                    ('C_selector',new['source_choice'][:,:,1:],new['source_transport_delta'])):
                    for k in range(3):
                        for value,label in ((1,'supported_relation'),(0,'blocked_relation'),(-1,'uncertain_relation')):
                            take=(native['labels']==value)&own; vector=updates[:,k][take]
                            row=relation_stats.setdefault((name,k+1,label),Counter()); row['relations']+=int(take.sum())
                            row['weight_sum']+=float(weights[:,k][take].sum())
                            row['transport_signed_sum']+=float(vector.sum()); row['transport_abs_sum']+=float(vector.abs().sum())
                            row['positive_transport_sum']+=float(vector.clamp_min(0.).sum())
                            row['negative_transport_abs_sum']+=float((-vector).clamp_min(0.).sum())
                canvas[:,:,y:y+512,x:x+512]+=torch.cat((old['z0'],old['z1'],new['z1']),0).float()*weight
                divisor[:,:,y:y+512,x:x+512]+=weight
        z=(canvas/divisor).cpu().numpy()[:,0]; p=1./(1.+np.exp(-np.clip(z,-80.,80.)))
        cases.append({'case':str(sample['case_name']),'gt':gt,'p0':p[0],'A_original':p[1],'C_selector':p[2]})
        np.savez_compressed(output/f'image_{index:02d}.npz',p0=p[0],original=p[1],selector=p[2],gt=gt)
        print(f'[SOURCE SURFACE VAL] {index+1}/14',flush=True)
    curves={'A_original':{float(t):{'metrics':value['p1'],'selection_score':value['selection_score']}
            for t,value in args.selection['threshold_curves'].items()},'C_selector':threshold_curve(cases,'C_selector',output)}
    threshold=float(args.selection['threshold']); saved=args.selection['report']['p1']
    counts={'tp':sum(int(((row['A_original']>=threshold)&row['gt']).sum()) for row in cases),
            'fp':sum(int(((row['A_original']>=threshold)&~row['gt']).sum()) for row in cases),
            'fn':sum(int(((row['A_original']<threshold)&row['gt']).sum()) for row in cases)}
    if any(counts[key]!=saved[key] for key in counts): raise ValueError('Original V9 reproduction differs from saved validation counts')
    points=operating_points(cases,['A_original','C_selector'],curves,threshold)
    reports=evaluate(cases,points,.45,206,output,modes=['A_original','C_selector'])
    write_csv(output/'actual_source_relations.csv',[{'model':key[0],'step':key[1],'relation':key[2],**dict(value)}
        for key,value in relation_stats.items()])
    atomic_json(output/'summary.json',{'reports':reports,'points':points,'curves':curves,
        'reference_threshold':.45,'reference_gaps':206,'selected_on':'val','topology_metrics':'approximate proxies'})
    lines=['# Source-selected repair vs original V9','',
        '| Model | Work point | Recall | IoU | clDice | Bidirectional approximate APLS | Closed /206 | Bridge proxy |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,values in reports.items():
        for point,row in values.items():
            m=row['metrics']; pair=row['paired']
            lines.append(f'| {name} | {point} | {m["recall"]:.5f} | {m["iou"]:.5f} | {m["cldice"]:.5f} | {m["apls_bidirectional_approx"]:.5f} | {pair["closed"]}/206 | {pair["background_bridge_proxy"]} |')
    lines+=['','This changes source selection, not source range: eight actual neighbours and three steps. '
        'Original V9 has no extra continuation training budget; these are engineering comparisons, not a pure equal-budget loss ablation. '
        'Inspect the per-image CSV for precision/recall, directional APLS, added TP/FP, gap lengths and short components.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('smoke','relation','audit','repair','joint','val','all'))
    parser.add_argument('--run_dir',type=Path,required=True); parser.add_argument('--root_path',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True); parser.add_argument('--device',default='cuda')
    parser.add_argument('--relation_epochs',type=int,default=5); parser.add_argument('--repair_epochs',type=int,default=20)
    parser.add_argument('--selector_lr',type=float,default=1e-4); parser.add_argument('--joint_lr',type=float,default=5e-6)
    parser.add_argument('--num_workers',type=int,default=4); parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--bootstrap',type=int,default=100); parser.add_argument('--hidden',type=int,default=32)
    parser.add_argument('--phase',choices=('repair','joint'),default='repair')
    parser.add_argument('--audit_phase',choices=('relation','repair'),default='relation')
    args=parser.parse_args(); args.device=torch.device(args.device); args.batch_size=2; args.labels=RelationLabels()
    if min(args.relation_epochs,args.repair_epochs,args.hidden)<1 or args.num_workers<0 or args.bootstrap<0: parser.error('Invalid budget')
    args.output_dir.mkdir(parents=True,exist_ok=True); torch.manual_seed(args.seed); np.random.seed(args.seed)
    original,losses,identity,_,source,selection=load_original(args)
    args.identity=identity; args.source=source; args.source_sha=v9.digest(source); args.source_losses=losses; args.selection=selection
    args.experiment_config={'version':VERSION,'identity':identity,'source_sha256':args.source_sha,
        'root_path':str(args.root_path.resolve()),'seed':args.seed,'relation_epochs':args.relation_epochs,
        'repair_epochs':args.repair_epochs,'num_workers':args.num_workers,'selector_lr':args.selector_lr,
        'joint_lr':args.joint_lr,'hidden':args.hidden,'bootstrap':args.bootstrap,'labels':asdict(args.labels),
        'code_sha256':{name:v9.digest(Path(__file__).parent/name) for name in (
            'run_v9_source_selector.py','networks/v9_source_selector.py','networks/v9_source_relation_labels.py')}}
    marker=args.output_dir/'experiment.json'
    if marker.is_file() and json.loads(marker.read_text())!=args.experiment_config: raise ValueError('Experiment configuration changed')
    atomic_json(marker,args.experiment_config)
    model=SourceSelectedRepair(original,args.hidden).to(args.device)
    if args.command in ('all','smoke') and not (args.output_dir/'smoke.json').is_file(): smoke(args,model)
    if args.command in ('all','relation'): train(args,model,'relation')
    if args.command in ('all','audit','relation'):
        audit_phase=args.audit_phase if args.command=='audit' else 'relation'
        restore(args.output_dir/audit_phase/'best.pth',model,args,audit_phase)
        measured=audit(args,model,args.output_dir/(audit_phase+'_best'))
        atomic_json(args.output_dir/(audit_phase+'_selection_screen.json'),measured['screening'])
        if args.command=='all' and not measured['screening']['passed']:
            atomic_json(args.output_dir/'pipeline_status.json',{'status':'stopped_relation_selection_not_improved',
                'next':'Inspect native labels, same-relation matching and edge/context features. No20-epoch repair was launched.'})
            print('[STOP] Fixed-input relation discrimination has not passed; no repair training started.',flush=True); return
    if args.command in ('all','repair','joint'):
        phase='joint' if args.command=='joint' else 'repair'; train(args,model,phase)
        restore(args.output_dir/phase/'best.pth',model,args,phase)
        if phase=='repair': audit(args,model,args.output_dir/'repair_best')
        compare_surface(args,model,phase)
        atomic_json(args.output_dir/'pipeline_status.json',{'status':phase+'_val_complete',
            'checkpoint':str(args.output_dir/phase/'best.pth'),'test_used':False})
    if args.command=='val':
        restore(args.output_dir/args.phase/'best.pth',model,args,args.phase); compare_surface(args,model,args.phase)
    if v9.digest(source)!=args.source_sha: raise RuntimeError('Original V9 checkpoint was modified')


if __name__=='__main__': main()
