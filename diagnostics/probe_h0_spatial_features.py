"""Compare frozen 16+16 point features with native H2/G3 3x3 patches.

Reuses the previous probe's exact image splits and validation candidates.
Only diagnostic probes train; the checkpoint and forward remain frozen.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.audit_full_h0_saved_predictions import digest,read_json,write_csv
from diagnostics.diagnose_full_h0_optimistic_reachability import atomic_json,atomic_npz,rebuild_model,signature,verify_recorded_code
from diagnostics.probe_full_h0_gap_features import (binary_metrics,bootstrap_by_image,combine,evaluate_threshold,
    features_from_result,fit_probe,make_candidates,match_image,paired_delta_ci,stable_seed,subset_sample,
    threshold_for_recall,tile_positions,smd_report)
from run_q3_r3_mass import full_dataset
from run_slim_end_to_end_mass import seed_all

FIELDS=('features','label','weight','nuisance','native_y','native_x','component','length','flank_proxy',
        'fixed_gap_id','negative_kind','tile_index','corridor_gt_fraction','corridor_length','raw_patch')


def patch_features(detail,semantic,gy,gx):
    """Spatially ordered 3x3 patch, with zeros outside the work grid."""
    if detail.ndim!=4 or detail.shape[1]!=64 or semantic.ndim!=4 or semantic.shape[1]!=96:
        raise ValueError('Expected raw H2=64 and G3=96 channels')
    if semantic.shape[-2:]!=detail.shape[-2:]:
        semantic=F.interpolate(semantic,size=detail.shape[-2:],mode='bilinear',align_corners=False)
    height,width=detail.shape[-2:]
    if not len(gy):return np.empty((0,1440),np.float32)
    values=F.unfold(torch.cat((detail,semantic),1),kernel_size=3,padding=1)[0]
    index=torch.as_tensor(gy*width+gx,device=values.device,dtype=torch.long)
    return values[:,index].T.detach().cpu().numpy().astype(np.float32)


@torch.inference_mode()
def tile_forward(model,image,top,left):
    captured=[]
    handle=model.original.upstream.register_forward_hook(lambda m,i,o:captured.append(o))
    try: result=model(image[None,:,top:top+512,left:left+512].to(next(model.parameters()).device))
    finally:handle.remove()
    if len(captured)!=1:raise RuntimeError('Expected exactly one live H0/Q forward')
    if captured[0]['detail'].shape[-2:]!=(128,128):raise ValueError('Expected the 128x128 H2 work grid')
    return result,captured[0]['detail'],captured[0]['semantic']


def original_coordinates(name,split,seed,record):
    if split=='val':
        return {i:(v[0],v[1]) for i,v in enumerate(tile_positions()) if (record['tile_index']==i).any()}
    rng=np.random.default_rng(stable_seed(seed,name))
    return {0:(int(rng.integers(0,989)),int(rng.integers(0,989)))}


def extra_coordinates(name,seed,index):
    rng=np.random.default_rng(stable_seed(seed,name+':spatial-extra:'+str(index)))
    return int(rng.integers(0,989)),int(rng.integers(0,989))


@torch.inference_mode()
def extract_original(model,sample,record,split,seed):
    name=str(sample['case_name']);n=len(record['label']);patches=np.empty((n,1440),np.float32)
    max_error=0.;sum_error=0.;count=0
    positions=original_coordinates(name,split,seed,record)
    for completed,(tile,(top,left)) in enumerate(positions.items(),1):
        take=np.flatnonzero(record['tile_index']==tile)
        if not len(take):continue
        result,detail,semantic=tile_forward(model,sample['image'],top,left)
        gy=record['gy'][take];gx=record['gx'][take]
        patches[take]=patch_features(detail,semantic,gy,gx)
        # Verify cached Q and projected H/H3 against the actual same forward.
        ctx=result['selector_context']['feature'][0].detach().cpu().numpy()[:,gy,gx].T
        q=result['q'][0].detach().cpu().numpy()[:,gy,gx].T
        reconstructed=np.concatenate((q,ctx),axis=1)
        difference=np.abs(reconstructed-record['features'][take,4:37])
        max_error=max(max_error,float(difference.max(initial=0)))
        sum_error+=float(difference.sum());count+=difference.size
        if split=='val' and (completed%5==0 or completed==len(positions)):
            print(f'[VAL TILE] {name} {completed}/{len(positions)}',flush=True)
    mean_error=sum_error/max(count,1)
    if max_error>.005 or mean_error>.0001:
        raise ValueError(f'Previous candidate feature reconstruction mismatch {name}: max={max_error}, mean={mean_error}')
    return patches,{'sampled_values':count,'max_abs':max_error,'mean_abs':mean_error}


@torch.inference_mode()
def extra_crop(model,sample,name,seed,crop_index):
    top,left=extra_coordinates(name,seed,crop_index)
    result,detail,semantic=tile_forward(model,sample['image'],top,left)
    p0=result['p0'][0,0].cpu().numpy();p1=result['p1'][0,0].cpu().numpy()
    gt=sample['mask'][0,top:top+512,left:left+512].numpy()>.5
    skeleton=sample['skeleton'][0,top:top+512,left:left+512].numpy()>.5
    cand=make_candidates(p0,p1,gt,skeleton,max_neg_per_type=150,
                         seed=stable_seed(seed,name+':spatial-extra:'+str(crop_index)))
    n=len(cand['label']);row=subset_sample(cand,np.arange(n))
    row['features']=features_from_result(result,cand)
    row['raw_patch']=patch_features(detail,semantic,cand['gy'],cand['gx'])
    row['native_y']=cand['y']+top;row['native_x']=cand['x']+left
    row['tile_index']=np.full(n,crop_index+1,np.int16)
    return row,{'top':top,'left':left,'positive':int((cand['label']==1).sum()),
                'negative':int((cand['label']==0).sum()),'candidate_counts':cand['counts']}


def append_nonduplicate(original,extras):
    """Keep original samples; limit repeated same-label locations in new crops."""
    buckets={(int(label),int(y)//8,int(x)//8) for label,y,x in
             zip(original['label'],original['native_y'],original['native_x'])}
    pieces=[{k:original[k] for k in FIELDS}];removed=0
    for extra in extras:
        keep=[]
        for i,(label,y,x) in enumerate(zip(extra['label'],extra['native_y'],extra['native_x'])):
            key=(int(label),int(y)//8,int(x)//8)
            if key in buckets:removed+=1
            else:buckets.add(key);keep.append(i)
        pieces.append({k:extra[k][keep] for k in FIELDS})
    return {key:np.concatenate([p[key] for p in pieces],axis=0) for key in FIELDS},removed


def pack_record(prior,patch):
    return {key:(patch if key=='raw_patch' else prior[key]) for key in FIELDS}


def evaluate_comparison(records,repetitions):
    base,match_rows=combine(records)
    raw=np.concatenate([r['raw_patch'] for r in records],axis=0)
    y=base['label'];split=base['split'];case=base['case'];weight=base['weight']
    match=np.zeros(len(y),bool);match[base['matched_indices']]=True
    fit=(split=='fit')&match;hold=(split=='hold')&match;val=split=='val';valmatch=val&match
    if any(len(np.unique(y[take]))<2 for take in (fit,hold,valmatch)):
        raise ValueError('Both classes required in fit, image-held-out and matched validation; inspect matching_by_image.csv')
    # Shared nuisance + Q; projected 16+16 point features versus 3x3 raw 64+96.
    inputs={'projected_point':base['features'][:,:37],
            'raw_spatial_3x3':np.concatenate((base['features'][:,:5],raw),axis=1)}
    results=[];per_image=[];fp_rows=[];positive_rows=[];scores_all={}
    for kind in ('linear','mlp'):
        for representation,values in inputs.items():
            print(f'[PROBE] {kind} {representation}',flush=True)
            predict,fit_info=fit_probe(values[fit],y[fit],values[hold],y[hold],kind,101)
            score=predict(values);scores_all[(kind,representation)]=score
            cut=threshold_for_recall(y[hold],score[hold],.9)
            val_cut=threshold_for_recall(y[valmatch],score[valmatch],.9)
            for population,take in (('matched',valmatch),('natural',val)):
                weights=weight[take] if population=='natural' else np.ones(int(take.sum()))
                metrics=binary_metrics(y[take],score[take],weights)
                row={'classifier':kind,'representation':representation,'population':population,**fit_info,**metrics,
                     'image_bootstrap_ci95':json.dumps(bootstrap_by_image(y[take],score[take],weights,case[take],repetitions=repetitions)),
                     'train_holdout_90_threshold':cut,
                     'validation_at_train_holdout_threshold':json.dumps(evaluate_threshold(y[take],score[take],cut,weights))}
                results.append(row)
                for image in np.unique(case[take]):
                    image_take=take&(case==image);w=weight[image_take] if population=='natural' else np.ones(int(image_take.sum()))
                    per_image.append({'classifier':kind,'representation':representation,'population':population,'case':str(image),
                                      **binary_metrics(y[image_take],score[image_take],w)})
            for category,subset in (('added_fp_proxy',val&(y==0)&np.isin(base['negative_kind'],[3,4])),
                                    ('bridge_proxy',val&(y==0)&(base['negative_kind']==4)),
                                    ('two_component_background_proxy',val&(y==0)&(base['nuisance'][:,3]<=64))):
                n=int(subset.sum());w=weight[subset]
                fp_rows.append({'classifier':kind,'representation':representation,'category':category,'natural_sampled':n,
                    'matched_sampled':int((subset&match).sum()),'unmatched_sampled':int((subset&~match).sum()),
                    'weighted_population_estimate':float(w.sum()),'train_holdout_threshold':cut,
                    'accepted_weighted_fraction':float(np.average(score[subset]>=cut,weights=w)) if n and cut is not None else None,
                    'val_matched_90_recall_threshold_descriptive':val_cut,
                    'accepted_at_val_matched_90_recall':float(np.average(score[subset]>=val_cut,weights=w)) if n and val_cut is not None else None})
            for category,subset in (('3_7',val&(y==1)&(base['length']>=3)&(base['length']<=7)),
                                    ('8_15',val&(y==1)&(base['length']>=8)&(base['length']<=15)),
                                    ('16_31',val&(y==1)&(base['length']>=16)&(base['length']<=31)),
                                    ('32_plus',val&(y==1)&(base['length']>=32)),
                                    ('centre_proxy',val&(y==1)&(base['flank_proxy']==0)),
                                    ('flank_proxy',val&(y==1)&(base['flank_proxy']==1))):
                n=int(subset.sum());w=weight[subset]
                positive_rows.append({'classifier':kind,'representation':representation,'category':category,
                    'natural_sampled':n,'matched_sampled':int((subset&match).sum()),
                    'train_holdout_threshold':cut,
                    'accepted_weighted_fraction':float(np.average(score[subset]>=cut,weights=w)) if n and cut is not None else None})
    increments=[]
    for kind in ('linear','mlp'):
        for population,take in (('matched',valmatch),('natural',val)):
            w=weight[take] if population=='natural' else np.ones(int(take.sum()))
            old=scores_all[(kind,'projected_point')][take];new=scores_all[(kind,'raw_spatial_3x3')][take]
            old_metrics=binary_metrics(y[take],old,w);new_metrics=binary_metrics(y[take],new,w)
            ci=paired_delta_ci(y[take],new,old,w,case[take],repetitions)
            row={'classifier':kind,'population':population,'samples':int(take.sum())}
            for key in ('roc_auc','pr_auc','fpr_at_80_recall','fpr_at_90_recall'):
                row[key+'_delta']=(new_metrics[key]-old_metrics[key]) if new_metrics[key] is not None and old_metrics[key] is not None else None
                row[key+'_paired_ci95']=json.dumps(ci[key])
            increments.append(row)
    return base,match_rows,results,per_image,fp_rows,increments,positive_rows


def write_output(output,identity,records,match_rows,results,per_image,fp_rows,increments,reconstruction,base,positive_rows):
    def save(name,rows):write_csv(output/name,rows,list(rows[0]) if rows else ['empty'])
    save('matching_by_image.csv',match_rows);save('probe_comparison.csv',results)
    save('per_image_metrics.csv',per_image);save('false_positive_subgroups.csv',fp_rows)
    save('positive_subgroups.csv',positive_rows)
    save('paired_increment.csv',increments)
    matched=np.zeros(len(base['label']),bool);matched[base['matched_indices']]=True
    smd=[]
    for split in ('fit','hold','val'):
        for population in (('matched','natural') if split=='val' else ('matched',)):
            take=(base['split']==split)&(matched if population=='matched' else True)
            weights=np.ones(int(take.sum())) if population=='matched' else base['weight'][take]
            for row in smd_report(base['label'][take],base['nuisance'][take],weights):
                smd.append({'split':split,'population':population,'samples':int(take.sum()),**row})
    save('matching_smd.csv',smd)
    count={}
    for split in ('fit','hold','val'):
        items=[r for r in match_rows if r['split']==split]
        count[split]={'images':len(items),'positive':sum(r['positive'] for r in items),
                      'negative':sum(r['negative'] for r in items),'matched_pairs':sum(r['matched_pairs'] for r in items)}
    lines=['# Raw H2/G3 spatial feature comparison','',
           'Original B checkpoint frozen. Original image splits and all sampled validation candidates are reused.',
           'Both probes use the same samples and matching. Baseline: nuisance + Q + 16+16 projected point features.',
           'Spatial: nuisance + Q + raw H2(64)/G3(96) 3x3 patches, flattened (1445 total inputs).',
           '3x3 is a local work-grid neighbourhood, not full-image or long-gap context.',
           'This single comparison changes both channel representation and spatial context; it does not isolate their individual effects.',
           f'Crop settings: original crop plus {identity["extra_train_crops"]} extra crops per training image.',
           f'Sample counts: `{json.dumps(count)}`.',
           f'Original cached Q/projected feature reconstruction: `{json.dumps(reconstruction)}`.','',
           '| Probe | Representation | Population | N | Positive fraction | ROC-AUC | PR-AUC | FPR at 90% recall |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'n/a' if v is None else f'{v:.4f}'
    for r in results:
        lines.append(f'| {r["classifier"]} | {r["representation"]} | {r["population"]} | {r["n"]} | {fmt(r["weighted_positive_fraction"])} | {fmt(r["roc_auc"])} | {fmt(r["pr_auc"])} | {fmt(r["fpr_at_90_recall"])} |')
    lines += ['', 'FPR at validation recall is descriptive; operating thresholds are selected on the image-disjoint training holdout.',
              'Actual added-FP and bridge categories are GT/prediction proxies, reported separately for matched and unmatched sampled candidates.',
              'Natural prevalence is inverse-sampling-weighted for the original prediction-defined validation candidate population, after uncertain labels are excluded.',
              'Validation tile-owner features are not the original weighted-logit stitched feature maps.',
              'Paired confidence intervals resample whole validation images; images, not neighbouring pixels, are the resampling units.',
              'This diagnostic does not prove candidate connections or repaired gap closure.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(output/'summary.json',{'complete':True,'identity':identity,'counts':count,
                                     'reconstruction':reconstruction,'results':results,'increments':increments,
                                     'false_positive_subgroups':fp_rows,'limits':lines[-6:]})


def verify_original_validation_matches(prior_path,base):
    path=prior_path/'val_candidate_scores.csv'
    if not path.is_file():raise FileNotFoundError('Previous validation candidate score file is required')
    expected={}
    with path.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            key=(row['case'],int(row['y']),int(row['x']))
            if key in expected:raise ValueError('Previous validation sample coordinates are not unique')
            expected[key]=(row['matched'].lower()=='true',int(row['label']),
                           float(row['sampling_weight']),float(row['p0']))
    matched=np.zeros(len(base['label']),bool);matched[base['matched_indices']]=True
    actual={}
    for i in np.flatnonzero(base['split']=='val'):
        key=(str(base['case'][i]),int(base['native_y'][i]),int(base['native_x'][i]))
        if key in actual:raise ValueError('Current validation sample coordinates are not unique')
        actual[key]=(bool(matched[i]),int(base['label'][i]),float(base['weight'][i]),float(base['nuisance'][i,0]))
    if set(expected)!=set(actual):raise ValueError('Validation candidates differ from the previous probe')
    for key in expected:
        if expected[key][:2]!=actual[key][:2] or not np.allclose(expected[key][2:],actual[key][2:],atol=1e-6,rtol=1e-6):
            raise ValueError('Validation labels, matching or sampling weights differ from the previous probe')
    return len(actual)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair_dir',type=Path,required=True);p.add_argument('--previous_probe',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--extra_train_crops',type=int,default=3);p.add_argument('--bootstrap',type=int,default=100)
    args=p.parse_args()
    if args.extra_train_crops<0 or args.bootstrap<0:p.error('Counts must be non-negative')
    previous=args.previous_probe.resolve();old=read_json(previous/'identity.json')
    old_summary=read_json(previous/'summary.json')
    if not old_summary.get('complete'):raise ValueError('Previous probe is not complete')
    if old['script_sha256']!=digest(ROOT/'diagnostics/probe_full_h0_gap_features.py'):
        raise ValueError('Previous probe code does not match the installed original probe')
    pair=args.pair_dir.resolve();checkpoint_path=pair/'B_transport_0p01_proposal_0p001/best.pth'
    reference=pair/'reference/fixed_gaps_and_anchors.json'
    if digest(checkpoint_path)!=old['checkpoint_sha256'] or digest(reference)!=old['reference_sha256']:
        raise ValueError('Previous probe, checkpoint and fixed reference do not match')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=checkpoint['config'];verify_recorded_code(config)
    if config['reference_sha256']!=old['reference_sha256']:raise ValueError('Checkpoint reference differs')
    root=Path(old['root_path']);train=full_dataset(root,'train');val=full_dataset(root,'val')
    def indexes(ds):
        result={Path(name).stem.replace('_sat',''):i for i,name in enumerate(ds.image_files)}
        if len(result)!=len(ds.image_files):raise ValueError('Duplicate case names')
        return result
    train_index=indexes(train);val_index=indexes(val);splits=old['train_image_splits'];val_names=old['val_image_names']
    if set(splits)-set(train_index) or set(val_names)!=set(val_index) or set(splits)&set(val_names):
        raise ValueError('Previous train/validation image identities differ from dataset')
    identity={'version':'h0_raw_spatial_probe_v1','previous_probe':str(previous),
              'previous_identity_sha256':digest(previous/'identity.json'),
              'previous_val_scores_sha256':digest(previous/'val_candidate_scores.csv'),
              'checkpoint_sha256':old['checkpoint_sha256'],'reference_sha256':old['reference_sha256'],
              'script_sha256':digest(Path(__file__)),'root_path':str(root),'seed':old['seed'],
              'train_image_splits':splits,'val_image_names':val_names,'extra_train_crops':args.extra_train_crops,
              'bootstrap':args.bootstrap,'device':torch.device(args.device).type,'torch_version':str(torch.__version__),
              'raw_features':'H2 64 + G3 96 channels, aligned 128x128, 3x3 local patch',
              'previous_probe_code_sha256':old['script_sha256']}
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True);id_path=output/'identity.json'
    if id_path.exists() and read_json(id_path)!=identity:raise ValueError('Output belongs to another experiment')
    if not id_path.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty')
    atomic_json(id_path,identity);cache=output/'cases';cache.mkdir(exist_ok=True)
    seed_all(int(old['seed']));model=rebuild_model(checkpoint,torch.device(args.device));model.requires_grad_(False);del checkpoint
    tasks=[(train,train_index[name],name,split) for name,split in splits.items()]
    tasks += [(val,val_index[name],name,'val') for name in val_names]
    records=[];recon=[];crop_reports=[]
    for number,(dataset,index,name,split) in enumerate(tasks,1):
        sample=dataset[index]
        if str(sample['case_name'])!=name:raise ValueError('Dataset case mapping mismatch')
        prior_path=previous/'cases'/(split+'_'+name+'.npz')
        with np.load(prior_path) as f: prior={key:f[key].copy() for key in f.files}
        if str(prior['input_sha256'])!=signature(sample):raise ValueError('Image or GT differs from previous probe: '+name)
        cache_path=cache/(split+'_'+name+'.npz');meta_path=cache/(split+'_'+name+'.json')
        if cache_path.is_file() and meta_path.is_file():
            meta=read_json(meta_path)
            if meta['prior_sha256']!=digest(prior_path):raise ValueError('Previous case cache changed: '+name)
            with np.load(cache_path) as f: current={key:f[key].copy() for key in f.files}
            print(f'[CACHE] {number}/{len(tasks)} {split} {name}',flush=True)
        else:
            print(f'[EXTRACT] {number}/{len(tasks)} {split} {name}',flush=True)
            patches,verified=extract_original(model,sample,prior,split,int(old['seed']))
            initial=pack_record(prior,patches);extras=[];crop_info=[]
            if split!='val':
                for crop in range(args.extra_train_crops):
                    item,info=extra_crop(model,sample,name,int(old['seed']),crop)
                    extras.append(item);crop_info.append(info)
                    print(f'[EXTRA CROP] {name} {crop+1}/{args.extra_train_crops}',flush=True)
            current,removed=append_nonduplicate(initial,extras)
            meta={'prior_sha256':digest(prior_path),'reconstruction':verified,'extra_crops':crop_info,
                  'extra_duplicate_locations_removed':removed,'original_samples':len(prior['label']),
                  'total_samples':len(current['label'])}
            atomic_npz(cache_path,current);atomic_json(meta_path,meta)
        current['case']=name;current['split']=split;records.append(current)
        recon.append(meta['reconstruction']);crop_reports.append({'case':name,'split':split,**meta})
    atomic_json(output/'crop_coverage.json',crop_reports)
    base,match_rows=combine(records)
    write_csv(output/'matching_by_image.csv',match_rows,list(match_rows[0]) if match_rows else ['case'])
    val_count=verify_original_validation_matches(previous,base)
    reconstruction={'previous_validation_candidates_verified':val_count,
                    'images':len(recon),'max_abs':max((r['max_abs'] for r in recon),default=0.),
                    'weighted_mean_abs':sum(r['mean_abs']*r['sampled_values'] for r in recon)/max(1,sum(r['sampled_values'] for r in recon))}
    try:
        base,match_rows,results,per_image,fp_rows,increments,positive_rows=evaluate_comparison(records,args.bootstrap)
    except ValueError as error:
        atomic_json(output/'summary.json',{'complete':False,'reason':str(error),'identity':identity})
        (output/'REPORT.md').write_text('# Probe unavailable\n\n'+str(error)+'\nSee matching_by_image.csv.\n',encoding='utf-8')
        print('[INCOMPLETE] '+str(error),flush=True);return
    write_output(output,identity,records,match_rows,results,per_image,fp_rows,increments,reconstruction,base,positive_rows)
    print('[DONE] '+str(output/'REPORT.md'),flush=True)


if __name__=='__main__':main()
