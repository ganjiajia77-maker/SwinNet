"""Actual source transport + live relationship supervision; reuse verified B_control."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_reliable_v9_mass as v9
from run_v9_gate_benefit import atomic_json,load_original
from run_v9_source_selector import original_edge_terms
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_source_selector import SourceSelectedRepair
from networks.v9_source_relation_labels import RelationLabels,native_relations
from networks.v9_selector_transport_training import VERSION,configure_joint,canonical_state,load_canonical,live_relation_loss,gradient_norm
from selector_transport_provenance import inspect_selector,find_control,find_reference,read_json,SELECTION_RULE
from selector_transport_reference import PROTOCOL,prepared,reference_digest,freeze_gaps,closed_gaps,bridge_proxy,exact_recall_threshold
from evaluation.road_comparison.road_metrics import image_metrics,summarize
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from mass_protocol import blend_weight,sliding_positions
from diagnostics.v91_counterfactual import write_csv


def selection_score(metrics,bridge,images):
    return metrics['iou']+.15*metrics['cldice']+.1*metrics['apls_gt_to_pred_approx']+.05*metrics['apls_bidirectional_approx']-.001*metrics['short_pred_components']-.01*bridge/max(images,1)


def tensor_digest(sample):
    digest=hashlib.sha256(str(sample['case_name']).encode())
    for key in ('image','mask'):
        array=sample[key].detach().cpu().contiguous().numpy(); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def save(args,model,path,epoch,best,optimizer=None,record=None):
    original=model.original
    value={'version':VERSION,'config':args.config,'identity':args.identity,'epoch':epoch,'score':best,
        'modules':{name:canonical_state(getattr(original,name)) for name in v9.MODULE_NAMES},
        'gate_relation':original.gate_relation.state_dict(),'state_gate':original.state_gate.state_dict(),
        'selector':model.selector.state_dict(),'epoch_record':record,'torch_rng':torch.get_rng_state(),
        'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if optimizer is not None: value['optimizer']=optimizer.state_dict()
    temporary=path.with_suffix('.tmp'); torch.save(value,temporary); temporary.replace(path)


def restore(args,model,path):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('version')!=VERSION or value.get('config')!=args.config or value.get('identity')!=args.identity:
        raise ValueError('Joint transport checkpoint identity/configuration differs')
    if set(value['modules'])!=set(v9.MODULE_NAMES): raise ValueError('Incomplete original V9 modules')
    for name in v9.MODULE_NAMES: load_canonical(getattr(model.original,name),value['modules'][name])
    model.original.gate_relation.load_state_dict(value['gate_relation'],strict=True)
    model.original.state_gate.load_state_dict(value['state_gate'],strict=True)
    model.selector.load_state_dict(value['selector'],strict=True)
    return value


def dataset_train(args,augment=True):
    return RoadSkeletonDataset(str(args.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=augment)


def parameter_gradients(optimizer):
    result={}
    for group in optimizer.param_groups:
        pieces=[p.grad.detach().double().square().sum() for p in group['params'] if p.grad is not None]
        result[group['name']]=float(torch.stack(pieces).sum().sqrt()) if pieces else 0.
    return result


def smoke(args,model,optimizer):
    ds=dataset_train(args,False); ds.set_epoch(1)
    batch=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))
    image,gt,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
    if args.device.type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
    model.train(); optimizer.zero_grad(set_to_none=True); seen=[]
    handle=model.selector.register_forward_pre_hook(lambda unused,inputs:seen.append(inputs[1]))
    try: result=model(image,mode='source')
    finally: handle.remove()
    if len(seen)!=3 or any(not torch.equal(value,result['states'][k]) for k,value in enumerate(seen)):
        raise RuntimeError('Step2/3 did not read the previous NEW source-selected U')
    base,terms=dynamic_losses(result,gt,skeleton,**args.source_losses,return_live_terms=True)
    relation,info=live_relation_loss(result,gt,skeleton,args.labels,args.seed)
    selector_parameters=tuple(model.selector.parameters())
    seg_selector_norm=gradient_norm(base-.2*terms['cont_live'],selector_parameters)
    if seg_selector_norm<=0: raise RuntimeError('Final segmentation loss cannot reach selector')
    total=base+args.lambda_relation*relation
    if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite joint smoke loss')
    total.backward(); grads=parameter_gradients(optimizer)
    if any(p.grad is not None for p in model.original.frozen.parameters()): raise RuntimeError('BASE/Q3 received gradients')
    for module in (model.original.state_gate,model.original.gate_relation.delta,model.original.relations.direction_head.scalar):
        if any(p.grad is not None for p in module.parameters()): raise RuntimeError('Unused scalar G received gradients')
    for k in range(3):
        a=result['source_choice'][:,k]; u=result['states'][k]
        expected=result['transport_rate'][:,k]*(a[:,1:]*(result['source_states'][:,k]-u)).sum(1,keepdim=True)
        torch.testing.assert_close(expected,result['transport_delta'][:,k],atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(a.sum(1),torch.ones_like(a[:,0]),atol=2e-6,rtol=1e-6)
        if bool((a[:,1:][~result['source_valid'][:,k]]!=0).any()): raise RuntimeError('Invalid neighbour selected')
        if bool(((result['states'][k+1]<-1e-6)|(result['states'][k+1]>1.+1e-6)).any()): raise RuntimeError('Invalid probability update')
    model.eval()
    bias=model.selector.stay[-1].bias; saved_bias=bias.detach().clone()
    old_modules=(model.original.state_gate.net[-1],model.original.gate_relation.delta[-1],model.original.relations.direction_head.scalar)
    old_bias=[module.bias.detach().clone() for module in old_modules]
    try:
        with torch.no_grad():
            normal=model(image,mode='source')
            bias.add_(2.); altered=model(image,mode='source'); bias.copy_(saved_bias)
            change=float((normal['p1']-altered['p1']).abs().mean())
            transport_change=float((normal['transport_delta']-altered['transport_delta']).abs().mean())
            if change<=1e-10 or transport_change<=1e-10: raise RuntimeError('Selector perturbation does not affect actual transport/P1')
            bias.fill_(1000.); holding=model(image,mode='source'); bias.copy_(saved_bias)
            hold_delta=float(holding['transport_delta'].abs().max())
            if hold_delta>1e-8: raise RuntimeError('Stay choice cannot suppress transport')
            for module in old_modules: module.bias.add_(40.)
            unused=model(image,mode='source')
            torch.testing.assert_close(normal['p1'],unused['p1'],atol=0.,rtol=0.)
    finally:
        with torch.no_grad():
            bias.copy_(saved_bias)
            for module,value in zip(old_modules,old_bias): module.bias.copy_(value)
    optimizer.zero_grad(set_to_none=True)
    report={'passed':True,'batch_size':len(image),'selector_changes_P1_mean_abs':change,
        'selector_changes_transport_mean_abs':transport_change,'segmentation_to_selector_grad_norm':seg_selector_norm,
        'live_relation_matching':info['matching'],'gradients':grads,'GT_used_in_forward':False,
        'U2_U3_read_new_previous_state':True,'invalid_neighbour_weights_zero':True,'unused_G_has_no_effect':True,
        'forced_stay_transport_max':hold_delta,'forced_stay_note':'Capacity check only; not a claim that the learned selector always stays when R is low.',
        'optimizer_groups':args.optimizer_report,'lambda_relation':args.lambda_relation,
        'peak_cuda_mib':torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type=='cuda' else None}
    atomic_json(args.output_dir/'actual_update_check.json',report)
    print('[ACTUAL UPDATE CHECK] '+json.dumps(report),flush=True)


@torch.no_grad()
def stitch(model,image,device,on_tile=None):
    h,w=image.shape[-2:]; canvas=torch.zeros((2,1,h,w),device=device); divisor=torch.zeros((1,1,h,w),device=device)
    weight=blend_weight(512,device)
    for y in sliding_positions(h,512,256):
        for x in sliding_positions(w,512,256):
            result=model(image[:,:,y:y+512,x:x+512],capture_diagnostics=True)
            if on_tile is not None: on_tile(y,x,result)
            canvas[:,:,y:y+512,x:x+512]+=torch.cat((result['z0'],result['z1']),0).float()*weight
            divisor[:,:,y:y+512,x:x+512]+=weight
    logits=(canvas/divisor).cpu().numpy()[:,0]
    if not np.isfinite(logits).all(): raise RuntimeError('Nonfinite stitched logits')
    return 1./(1.+np.exp(-np.clip(logits,-80.,80.)))


def source_statistics_callback(args,sample):
    height,width=sample['mask'].shape[-2:]; owned=np.zeros((height,width),bool); totals={}
    def collect(y,x,result):
        h,w=result['r'].shape[-2:]
        target=sample['mask'][None,:,y:y+512,x:x+512].to(args.device)
        skeleton=sample['skeleton'][None,:,y:y+512,x:x+512].to(args.device)
        native=native_relations(target,skeleton,(h,w),args.labels)
        yy=np.minimum(((np.arange(h)+.5)*512/h).astype(int),511); xx=np.minimum(((np.arange(w)+.5)*512/w).astype(int),511)
        own=(~owned[y:y+512,x:x+512])[yy[:,None],xx[None,:]]; owned[y:y+512,x:x+512]=True
        own=torch.as_tensor(own,device=args.device)[None,None]
        if 'source_transport_delta' in result: weights,updates=result['source_choice'][:,:,1:],result['source_transport_delta']
        else: weights,updates=original_edge_terms(result)
        for k in range(3):
            for label,name in ((1,'supported'),(0,'blocked'),(-1,'uncertain')):
                take=(native['labels']==label)&own; vector=updates[:,k][take]
                row=totals.setdefault(f'{k+1}:{name}',Counter()); row['relations']+=int(take.sum())
                row['weight_sum']+=float(weights[:,k][take].sum()); row['transport_signed_sum']+=float(vector.sum())
                row['transport_abs_sum']+=float(vector.abs().sum()); row['positive_transport_sum']+=float(vector.clamp_min(0.).sum())
    return collect,totals


def merge_statistics(values):
    total={}
    for item in values:
        for key,row in item.items(): total.setdefault(key,Counter()).update(row)
    return {key:dict(row) for key,row in total.items()}


@torch.no_grad()
def prepare_control(args,model):
    from run_q3_r3_mass import full_dataset
    from gate_benefit_protocol import create_variant
    dataset=full_dataset(args.root_path,'val')
    if len(dataset)!=14: raise ValueError('Expected the same14 validation images')
    folder=args.output_dir/'control_predictions'; folder.mkdir(exist_ok=True)
    control=create_variant(model.original.frozen,args.control['checkpoint'],args.identity,args.device)
    cases=[]; metadata={}; statistics=[]
    for index in range(len(dataset)):
        sample=dataset[index]; name=str(sample['case_name']); digest=tensor_digest(sample)
        file=folder/f'{index:02d}.npz'; marker=file.with_suffix('.json'); stat_file=folder/f'{index:02d}.stats.json'
        expected={'case':name,'input_sha256':digest,'control_sha256':args.config['control_sha256']}
        if file.is_file() and marker.is_file() and stat_file.is_file():
            if read_json(marker)!=expected: raise ValueError('Control cache input/checkpoint changed')
            with np.load(file,allow_pickle=False) as value: p0,p1,gt=value['p0'],value['p1'],value['gt'].astype(bool)
        else:
            collect,stats=source_statistics_callback(args,sample)
            p0,p1=stitch(control,sample['image'][None].to(args.device),args.device,collect); gt=sample['mask'][0].numpy()>.5
            with file.with_suffix('.tmp').open('wb') as handle: np.savez_compressed(handle,p0=p0,p1=p1,gt=gt)
            file.with_suffix('.tmp').replace(file); atomic_json(marker,expected); atomic_json(stat_file,stats)
        statistics.append(read_json(stat_file))
        cases.append({'case':name,'gt':gt,'p0':p0,'B_control':p1,'input_sha256':digest})
        metadata[name]=expected
        print(f'[CONTROL INFERENCE OR CACHE] {index+1}/14',flush=True)
    old_reference=read_json(args.reference_path); fingerprint=reference_digest(cases)
    if fingerprint!=old_reference['sha256']: raise ValueError('Current original P0 inputs differ from the fixed old reference')
    counts={'tp':sum(int(((case['B_control']>=.6)&case['gt']).sum()) for case in cases),
            'fp':sum(int(((case['B_control']>=.6)&~case['gt']).sum()) for case in cases),
            'fn':sum(int(((case['B_control']<.6)&case['gt']).sum()) for case in cases)}
    if any(counts[key]!=args.control['val_report']['p1'][key] for key in counts):
        raise ValueError('Reused B_control does not reproduce its saved validation counts at0.60')
    file=args.output_dir/'fixed_gaps_and_anchors.json'
    binding=args.output_dir/'fixed_reference_binding.json'
    if file.is_file():
        manifest=read_json(file)
        if manifest['sha256']!=fingerprint or manifest['count']!=206: raise ValueError('Frozen gap/anchor manifest changed')
        if binding.is_file():
            if read_json(binding)['manifest_sha256']!=v9.digest(file): raise ValueError('Frozen gap/anchor pixels were modified')
        elif manifest!=freeze_gaps(cases,old_reference):
            raise ValueError('Unbound gap/anchor manifest does not reproduce the old fixed definition')
    else:
        manifest=freeze_gaps(cases,old_reference); atomic_json(file,manifest)
    atomic_json(binding,{'manifest_sha256':v9.digest(file),'original_reference_sha256':fingerprint})
    atomic_json(args.output_dir/'validation_inputs.json',metadata)
    atomic_json(args.output_dir/'control_source_statistics.json',merge_statistics(statistics))
    return cases,manifest


@torch.no_grad()
def infer_selector(args,model,cases,collect_statistics=False):
    from run_q3_r3_mass import full_dataset
    model.eval(); dataset=full_dataset(args.root_path,'val'); statistics=[]
    for index,case in enumerate(cases):
        sample=dataset[index]
        if str(sample['case_name'])!=case['case'] or tensor_digest(sample)!=case['input_sha256']:
            raise ValueError('S_selector and B_control validation inputs differ')
        callback,stats=source_statistics_callback(args,sample) if collect_statistics else (None,None)
        p0,p1=stitch(model,sample['image'][None].to(args.device),args.device,callback)
        if stats is not None: statistics.append(stats)
        if not np.array_equal(p0>=.45,case['p0']>=.45): raise ValueError('Frozen P0 reference changed')
        case['S_selector']=p1
        print(f'[SELECTOR VAL] {index+1}/14',flush=True)
    if collect_statistics: atomic_json(args.output_dir/'selector_source_statistics.json',merge_statistics(statistics))


def point_metrics(cases,name,threshold,manifest):
    metrics=[]; closed=bridges=0; gap_ids={}; bins={label:{'eligible':0,'closed':0} for label in ('3-7','8-15','16-31','32+')}
    for case in cases:
        pred=case[name]>=threshold
        metrics.append({'case_id':case['case'],**image_metrics(pred,prepared(case),PROTOCOL)})
        bucket,ids=closed_gaps(pred,manifest['images'][case['case']]); gap_ids[case['case']]=ids
        closed+=len(ids); bridges+=bridge_proxy(case['p0']>=.45,pred,case['gt'])
        for label in bins:
            for key in bins[label]: bins[label][key]+=bucket[label][key]
    result=summarize(metrics)
    return {'threshold':float(threshold),'metrics':result,'closed':closed,'bridge_proxy':bridges,
            'closed_ids':gap_ids,'gap_by_length':bins,'selection_score':selection_score(result,bridges,len(cases))}


def train(args,model,optimizer,cases,manifest):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    start=1; best=-float('inf'); last=args.output_dir/'last.pth'
    if last.is_file():
        state=restore(args,model,last); optimizer.load_state_dict(state['optimizer']); start=state['epoch']+1; best=state['score']
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        if state.get('epoch_record'): atomic_json(args.output_dir/f'epoch_{state["epoch"]:03d}.json',state['epoch_record'])
    dataset=dataset_train(args)
    for epoch in range(start,args.epochs+1):
        dataset.set_epoch(epoch); generator=torch.Generator().manual_seed(args.seed+epoch*1000003)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,
                          pin_memory=args.device.type=='cuda',generator=generator)
        model.train(); model.set_accept_floor(0.); order=hashlib.sha256(); images=hashlib.sha256(); sums=Counter(); grad=None
        for index,batch in enumerate(loader):
            for case in batch['case_name']: order.update(str(case).encode()+b'\0')
            images.update(batch['image'].contiguous().numpy().tobytes())
            image,gt,skeleton=(batch[key].to(args.device) for key in ('image','mask','skeleton'))
            optimizer.zero_grad(set_to_none=True); result=model(image,mode='source')
            base,terms=dynamic_losses(result,gt,skeleton,**args.source_losses,return_live_terms=True)
            relation,info=live_relation_loss(result,gt,skeleton,args.labels,args.seed+epoch*100000+index)
            total=base+args.lambda_relation*relation
            if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite joint training loss')
            if grad is None and info['matching']['matched_pairs']:
                parameters=tuple(p for p in model.parameters() if p.requires_grad)
                sg=gradient_norm(base-.2*terms['cont_live'],parameters)
                rg=gradient_norm(args.lambda_relation*relation,parameters)
                grad={'weighted_relation_norm':rg,'segmentation_norm':sg,'relation_to_segmentation_ratio':rg/max(sg,1e-12),
                      'review_relation_dominance':rg>sg}
            total.backward()
            if any(p.grad is not None for p in model.original.frozen.parameters()): raise RuntimeError('BASE/Q3 received gradients')
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            for key,value in (('total',total),('seg',terms['seg']),('cont',terms['cont']),('relation',relation)):
                sums[key]+=float(value.detach())
            sums['matched_pairs']+=info['matching']['matched_pairs']; sums['batches']+=1
            sums['natural_positive']+=info['matching']['natural_positive']; sums['natural_negative']+=info['matching']['natural_negative']
            if index%25==0: print(f'[S_SELECTOR TRAIN] epoch={epoch}/{args.epochs} batch={index+1}/{len(loader)} loss={float(total.detach()):.6f} relation_pairs={info["matching"]["matched_pairs"]}',flush=True)
        record={'epoch':epoch,'case_order_sha256':order.hexdigest(),'augmented_images_sha256':images.hexdigest(),
            'losses':{name:sums[name]/max(sums['batches'],1) for name in ('total','seg','cont','relation')},
            'relation_counts':{name:sums[name] for name in ('matched_pairs','natural_positive','natural_negative')},'gradient_check':grad}
        for key in ('case_order_sha256','augmented_images_sha256'):
            if record[key]!=args.control['records'][epoch][key]:
                atomic_json(args.output_dir/'failed_data_pairing.json',record)
                raise ValueError(f'Existing B_control cannot be reused: {key} differs at epoch {epoch}')
        if not sums['matched_pairs']:
            atomic_json(args.output_dir/'failed_no_relation_labels.json',record)
            raise RuntimeError('No matched relationship labels in a whole epoch; not a successful supervised continuation')
        if epoch%args.val_interval==0 or epoch==args.epochs:
            infer_selector(args,model,cases); validation=point_metrics(cases,'S_selector',.6,manifest)
            atomic_json(args.output_dir/f'val_epoch_{epoch:03d}.json',validation); score=validation['selection_score']
            record['val_score_at_0_60']=score
            if score>best: best=score; save(args,model,args.output_dir/'best.pth',epoch,best)
            print(f'[S_SELECTOR VAL] epoch={epoch} score={score:.6f}',flush=True)
        save(args,model,last,epoch,best,optimizer,record); atomic_json(args.output_dir/f'epoch_{epoch:03d}.json',record)
    atomic_json(args.output_dir/'training_complete.json',{'epochs':args.epochs,'best_score':best,'reused_control':str(args.control['run'])})


def curve(args,cases,name,manifest):
    path=args.output_dir/(name+'_threshold_curve.json')
    marker=path.with_suffix('.identity.json')
    expected={'checkpoint_sha256':args.config['control_sha256'] if name=='B_control' else v9.digest(args.output_dir/'best.pth'),
              'reference_sha256':manifest['sha256']}
    valid_cache=path.is_file() and marker.is_file() and read_json(marker)==expected
    values={float(t):value for t,value in read_json(path).items()} if valid_cache else {}
    if not valid_cache: atomic_json(path,{}) # invalidate stale data BEFORE changing its identity marker
    atomic_json(marker,expected)
    # Reuse completed control metrics only if all cached probabilities and GT
    # are exactly identical to the freshly verified current validation inputs.
    old=args.control['run'].parent
    cached=old/'comparison_3step/B_control_threshold_curve.json'
    probabilities=old/'predictions_val/B_control'
    if name=='B_control' and not values and cached.is_file() and (probabilities/'identity.json').is_file():
        identity=read_json(probabilities/'identity.json'); compatible=identity.get('checkpoint_sha256')==expected['checkpoint_sha256'] and identity.get('steps')==3
        for case in cases:
            file=probabilities/(hashlib.sha256(case['case'].encode()).hexdigest()[:20]+'.npz')
            if not compatible or not file.is_file(): compatible=False; break
            with np.load(file,allow_pickle=False) as data:
                compatible=all(np.array_equal(data[key],case[target]) for key,target in (('p0','p0'),('p1','B_control'),('gt','gt')))
        if compatible:
            candidate={float(t):value for t,value in read_json(cached).items()}
            if all(abs(selection_score(row['metrics'],row['bridge_proxy'],len(cases))-row['selection_score'])<1e-8 for row in candidate.values()):
                values=candidate; print('[CONTROL THRESHOLDS] Reused exact-probability-matched prior val curve',flush=True)
    for threshold in np.round(np.arange(.35,.751,.01),2):
        if float(threshold) in values: continue
        row=point_metrics(cases,name,float(threshold),manifest)
        values[float(threshold)]=row
        atomic_json(path,values)
        print(f'[VAL THRESHOLD] {name} threshold={threshold:.2f} score={row["selection_score"]:.6f}',flush=True)
    atomic_json(path,values); return values


@torch.no_grad()
def compare(args,model,cases,manifest):
    restore(args,model,args.output_dir/'best.pth'); infer_selector(args,model,cases,collect_statistics=True)
    for index,case in enumerate(cases):
        np.savez_compressed(args.output_dir/f'val_probabilities_{index:02d}.npz',p0=case['p0'],control=case['B_control'],selector=case['S_selector'],gt=case['gt'])
    curves={name:curve(args,cases,name,manifest) for name in ('B_control','S_selector')}
    selected={name:max(values,key=lambda t:values[t]['selection_score']) for name,values in curves.items()}
    tp=sum(int(((case['B_control']>=selected['B_control'])&case['gt']).sum()) for case in cases)
    rows=[]; reports={}
    for name in curves:
        own=point_metrics(cases,name,selected[name],manifest); threshold,error=exact_recall_threshold(cases,name,tp)
        matched=point_metrics(cases,name,threshold,manifest)
        reports[name]={'own_val_threshold':own,'matched_recall':dict(matched,target_tp=tp,tp_matching_error=error)}
        for point,value in reports[name].items():
            metrics=value['metrics']
            rows.append({'model':name,'work_point':point,'threshold':value['threshold'],
                **{key:metrics[key] for key in ('iou','f1','precision','recall','cldice','apls_bidirectional_approx','short_pred_components')},
                'closed_fixed_gaps':value['closed'],'bridge_proxy':value['bridge_proxy'],
                'tp_matching_error':value.get('tp_matching_error')})
    atomic_json(args.output_dir/'comparison.json',{'reports':reports,'selection_rule':SELECTION_RULE,
        'checkpoint_selection_threshold':.6,'reference_sha256':manifest['sha256'],'reference_count':manifest['count'],
        'control_reused':str(args.control['run']),'lambda_relation':args.lambda_relation,'metric_limits':'APLS/gap/bridge are proxy metrics.'})
    write_csv(args.output_dir/'comparison.csv',rows)
    stats_rows=[]
    for name,file in (('B_control','control_source_statistics.json'),('S_selector','selector_source_statistics.json')):
        for key,value in read_json(args.output_dir/file).items():
            k,kind=key.split(':')
            stats_rows.append({'model':name,'step':int(k),'native_relation':kind,**value,
                'weight_mean':value['weight_sum']/max(value['relations'],1),
                'transport_abs_mean':value['transport_abs_sum']/max(value['relations'],1)})
    write_csv(args.output_dir/'actual_source_relations.csv',stats_rows)
    # Preserve the already-validated fixed-input test on an ORIGINAL V9 bank.
    # This never supplies the old bank to the actual joint training forward.
    from gate_benefit_protocol import create_variant
    from run_v9_source_selector import audit
    reference_original=create_variant(model.original.frozen,args.run_dir/'stage_D/best.pth',args.identity,args.device)
    reference_model=SourceSelectedRepair(reference_original,args.hidden).to(args.device)
    reference_model.selector.load_state_dict(model.selector.state_dict(),strict=True)
    after=audit(args,reference_model,args.output_dir/'fixed_relation_after')
    before=read_json(args.output_dir/'fixed_relation_before.json'); changes=[]
    for population in ('natural','matched'):
        for scorer in ('edge','choice'):
            key=population+'_'+scorer
            for target in (.8,.9):
                old=next(row for row in before['reports'][key]['matched_helpful'] if row['helpful_target']==target)
                new=next(row for row in after['reports'][key]['matched_helpful'] if row['helpful_target']==target)
                if old['C']['available'] and new['C']['available']:
                    changes.append({'population':population,'scorer':scorer,'correct_acceptance':target,
                        'wrong_before':old['C']['harmful_acceptance'],'wrong_after':new['C']['harmful_acceptance'],
                        'difference':new['C']['harmful_acceptance']-old['C']['harmful_acceptance']})
    write_csv(args.output_dir/'fixed_relation_change.csv',changes)
    lines=['# Actual source transport vs reused B_control','',f'Checkpoint/threshold score: `{SELECTION_RULE}`.',
        'Checkpoints use val@0.60 at the same cadence; final thresholds use val0.35:0.01:0.75. No test selection.',
        f'Frozen reference: {manifest["count"]} gaps; SHA256 {manifest["sha256"]}.','',
        '| Model | Work point | Threshold | IoU | Precision | Recall | clDice | Bidirectional approx APLS | Closed | Bridge proxy | Short fragments |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["work_point"]} | {row["threshold"]:.6f} | {row["iou"]:.5f} | {row["precision"]:.5f} | {row["recall"]:.5f} | {row["cldice"]:.5f} | {row["apls_bidirectional_approx"]:.5f} | {row["closed_fixed_gaps"]} | {row["bridge_proxy"]} | {row["short_pred_components"]:.2f} |')
    lines+=['',f'Relation coefficient: {args.lambda_relation}; {args.config["relation_coefficient_origin"]}.',
        'Input order and complete augmented training tensors were checked against B_control at every epoch. '
        'Selector received its validated relationship pretraining; this extra pretraining is recorded, not a pure loss-only ablation.',
        'See actual_source_relations.csv for transport into blocked native relationships, and fixed_relation_after/REPORT.md for the unchanged fixed-input test. '
        'Joint feature encoders can drift, so that frozen-input replay is a compatibility diagnostic, not a gate-only causal claim.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','smoke','train','val','all'))
    parser.add_argument('--selector_run',type=Path,required=True); parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--results',type=Path,default=Path('/home/gjj/SwinNet-roadbias/model_out'))
    parser.add_argument('--original_run',type=Path); parser.add_argument('--control_run',type=Path); parser.add_argument('--reference_set',type=Path)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--lambda_relation',type=float); parser.add_argument('--bootstrap',type=int,default=100)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        selector=inspect_selector(args.selector_run,args.results,args.original_run)
        args.control=find_control(args.results,selector,args.epochs,args.control_run)
        args.reference_path=find_reference(selector['original_path'].parent.parent,args.reference_set)
    except Exception as error:
        atomic_json(args.output_dir/'preflight.json',{'passed':False,'reason':str(error),'training_started':False}); raise
    args.device=torch.device(args.device); config=selector['config']; args.root_path=Path(config['root_path'])
    args.seed=config['seed']; args.num_workers=args.control['config']['num_workers']; args.batch_size=2
    args.val_interval=args.control['config']['val_interval']; args.run_dir=selector['original_path'].parent.parent
    args.labels=RelationLabels(**config['labels']); args.identity=selector['state']['identity']
    args.hidden=config['hidden']
    origin=('explicit experiment override' if args.lambda_relation is not None else
        'validated selector config' if 'lambda_relation' in config else 'initial experimental0.1; validated pretraining had no reusable joint coefficient')
    args.lambda_relation=args.lambda_relation if args.lambda_relation is not None else config.get('lambda_relation',.1)
    if args.lambda_relation<=0 or args.epochs<1: parser.error('Positive relationship coefficient and training budget required')
    if args.command=='preflight':
        atomic_json(args.output_dir/'preflight.json',{'passed':True,'selector_checkpoint':str(selector['checkpoint']),
            'selector_epoch':selector['state']['epoch'],'selector_score':selector['state']['score'],
            'original_checkpoint':str(selector['original_path']),'control_checkpoint':str(args.control['checkpoint']),
            'reference':str(args.reference_path),'lambda_relation':args.lambda_relation,'coefficient_origin':origin,
            'training_started':False}); print((args.output_dir/'preflight.json').read_text()); return
    original,losses,identity,_,source,_=load_original(args); args.source_losses=losses
    model=SourceSelectedRepair(original,config['hidden']).to(args.device)
    model.selector.load_state_dict(selector['state']['selector'],strict=True)
    optimizer,args.optimizer_report=configure_joint(model,selector['original']['learning_rates'],config['selector_lr'])
    args.config={'version':VERSION,'identity':identity,'selector_run':str(args.selector_run.resolve()),
        'selector_sha256':v9.digest(selector['checkpoint']),'original_sha256':v9.digest(source),
        'control_sha256':v9.digest(args.control['checkpoint']),'root_path':str(args.root_path.resolve()),
        'epochs':args.epochs,'seed':args.seed,'num_workers':args.num_workers,'val_interval':args.val_interval,
        'batch_size':2,'accumulation':1,'amp':'none','EMA':False,'labels':config['labels'],
        'lambda_relation':args.lambda_relation,'relation_coefficient_origin':origin,'source_losses':losses,
        'optimizer':args.optimizer_report,'selection_rule':SELECTION_RULE,'checkpoint_threshold':.6,
        'reference_file_sha256':v9.digest(args.reference_path),'selector_pretrain_epoch':selector['state']['epoch'],
        'code_sha256':{name:v9.digest(Path(__file__).parent/name) for name in ('run_v9_selector_transport.py',
            'selector_transport_provenance.py','selector_transport_reference.py','networks/v9_selector_transport_training.py')}}
    marker=args.output_dir/'experiment.json'
    if marker.is_file() and read_json(marker)!=args.config: raise ValueError('Joint transport experiment configuration changed')
    atomic_json(marker,args.config)
    atomic_json(args.output_dir/'fixed_relation_before.json',selector['report'])
    atomic_json(args.output_dir/'preflight.json',{'passed':True,'selector_checkpoint':str(selector['checkpoint']),
        'selector_epoch':selector['state']['epoch'],'original_checkpoint':str(source),'control_checkpoint':str(args.control['checkpoint']),
        'optimizer':args.optimizer_report,'reference':str(args.reference_path),'training_started':False})
    if args.command in ('all','train','smoke') and not (args.output_dir/'actual_update_check.json').is_file(): smoke(args,model,optimizer)
    if args.command=='smoke': return
    cases,manifest=prepare_control(args,model)
    if args.command in ('all','train'): train(args,model,optimizer,cases,manifest)
    if args.command in ('all','val'): compare(args,model,cases,manifest)
    for file,expected in ((selector['checkpoint'],args.config['selector_sha256']),
                          (source,args.config['original_sha256']),(args.control['checkpoint'],args.config['control_sha256'])):
        if v9.digest(file)!=expected: raise RuntimeError(f'Input checkpoint changed during the experiment: {file}')


if __name__=='__main__': main()
