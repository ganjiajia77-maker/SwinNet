"""Val-only native1500, 512/256 matched-operating-point paired evaluation."""
import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from datasets.native_relation_targets import native_targets
from evaluation.road_comparison.road_metrics import DEFAULT_PROTOCOL,image_metrics,prepare_target,summarize
from mass_protocol import blend_weight,sliding_positions
from networks.vision_transformer import SwinUnet
from networks.native_h0_relations import install_native_relations
from selector_transport_reference import bridge_proxy,closed_gaps,exact_recall_threshold,prepared,reference_digest

PROTOCOL=dict(DEFAULT_PROTOCOL,image_size=1500)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_model(path,device,native):
    state=torch.load(path,map_location='cpu',weights_only=False);saved=state['args']
    if bool(saved.get('native_relation_experiment',False)) != native:raise ValueError('Wrong checkpoint variant')
    opts=SimpleNamespace(cfg=str(ROOT/'configs/swin_tiny_patch4_window7_224_lite.yaml'),opts=None,batch_size=2,
        img_size=512,zip=False,cache_mode=None,resume=None,accumulation_steps=1,use_checkpoint=False,
        amp_opt_level=None,tag=None,eval=False,throughput=False)
    config=get_config(opts);names=inspect.signature(SwinUnet).parameters
    kwargs={k:v for k,v in saved.items() if k in names and k not in ('config','img_size','num_classes','return_skeleton')}
    model=SwinUnet(config,img_size=512,num_classes=1,return_skeleton=True,**kwargs)
    if native:install_native_relations(model)
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model.to(device).eval(),saved


def owner(top,left):
    mapping={0:(0,384),256:(384,640),512:(640,896),768:(896,1134),988:(1134,1500)}
    y0,y1=mapping[top];x0,x1=mapping[left]
    yy=top+np.arange(128)*4+2;xx=left+np.arange(128)*4+2
    return (yy[:,None]>=y0)&(yy[:,None]<y1)&(xx[None,:]>=x0)&(xx[None,:]<x1)


def update_diagnostics(stats,outputs,mask,skel,core):
    labels=native_targets(mask,skel);valid=labels['connectivity_valid']&core[None,:,:]
    positive=labels['connectivity']>.5;dv=labels['direction_valid']&core
    stats['ambiguous_cells']+=int((labels['ambiguous_cells']&core).sum())
    occupied=mask.reshape(128,4,128,4).any(axis=(1,3));padded=np.pad(occupied,1)
    from topology_direction_constants import CONNECTIVITY_DIRECTIONS
    neighbour=np.stack([padded[1+dy:129+dy,1+dx:129+dx] for dy,dx in CONNECTIVITY_DIRECTIONS])
    two_roads=occupied[None,:,:]&neighbour
    for row in outputs[4]:
        stage=row.get('stage')
        if stage not in (2,3):continue
        c=row['connectivity'][0].sigmoid().detach().cpu().numpy();d=row['direction'][0].detach().cpu().numpy()
        for label,condition in (('pos',positive),('neg',~positive)):
            select=valid&condition;stats[f'C{stage}_{label}_edges']+=int(select.sum())
            stats[f'C{stage}_{label}_accepted']+=int(((c>=.5)&select).sum())
        difficult=valid&(~positive)&two_roads
        stats[f'C{stage}_negative_road_road_edges']+=int(difficult.sum())
        stats[f'C{stage}_negative_road_road_accepted']+=int(((c>=.5)&difficult).sum())
        norm=np.linalg.norm(d,axis=0);axis=(d*labels['direction']).sum(axis=0)/np.maximum(norm,1e-6)
        errors=.5*np.degrees(np.arccos(np.clip(axis,-1,1)))
        stats[f'D{stage}_valid_cells']+=int(dv.sum());stats[f'D{stage}_angle_sum']+=float(errors[dv].sum())
        if stage==3 and 'relation_weights' in row:
            weights=row['relation_weights'][0,1:].detach().cpu().numpy()
            for label,condition in (('correct',positive),('incorrect',~positive)):
                select=valid&condition;stats[f'choice_{label}_sum']+=float(weights[select].sum())
                stats[f'choice_{label}_edges']+=int(select.sum())
            stats['choice_incorrect_road_road_sum']+=float(weights[difficult].sum())
            stats['choice_incorrect_road_road_edges']+=int(difficult.sum())


@torch.inference_mode()
def infer(model,sample,device,collect):
    image=sample['image'][None].to(device);gt=sample['mask'][0].numpy()>.5;skel=sample['skeleton'][0].numpy()>.5
    if gt.shape!=(1500,1500):raise ValueError('Expected native 1500 validation image')
    canvas=torch.zeros((1,1,1500,1500),device=device);divisor=torch.zeros_like(canvas);weight=blend_weight(512,device)
    stats=Counter();positions=sliding_positions(1500,512,256)
    for top in positions:
        for left in positions:
            result=model(image[:,:,top:top+512,left:left+512])
            canvas[:,:,top:top+512,left:left+512]+=result[0].float()*weight
            divisor[:,:,top:top+512,left:left+512]+=weight
            if collect:
                update_diagnostics(stats,result,gt[top:top+512,left:left+512],skel[top:top+512,left:left+512],owner(top,left))
    prob=(canvas/divisor).sigmoid()[0,0].cpu().numpy()
    if not np.isfinite(prob).all():raise ValueError('Invalid probability')
    return prob,gt,stats


@torch.inference_mode()
def intervention(model,sample,device,parameter):
    """Diagnostic-only forward; restore the checkpoint parameter immediately."""
    saved=parameter.detach().clone()
    try:
        parameter.zero_()
        probability,_,_=infer(model,sample,device,False)
        return probability
    finally:parameter.copy_(saved)


def fixed_reference(cases):
    images={};total=0
    for case in cases:
        p0=case['A']>=.45;gt=case['gt'];h,w=gt.shape
        covered=cv2.dilate(p0.astype(np.uint8),np.ones((5,5),np.uint8))>0
        n,labels,stats,_=cv2.connectedComponentsWithStats((prepared(case)['skeleton']&~covered).astype(np.uint8),connectivity=8)
        _,old=cv2.connectedComponents(p0.astype(np.uint8),connectivity=8);gaps=[]
        for i in range(1,n):
            length=int(stats[i,cv2.CC_STAT_AREA])
            if length<3:continue
            x,y,width,height=map(int,stats[i,:4]);x0,x1=max(x-7,0),min(x+width+7,w);y0,y1=max(y-7,0),min(y+height+7,h)
            region=labels[y0:y1,x0:x1]==i;flank=cv2.dilate(region.astype(np.uint8),np.ones((15,15),np.uint8))>0
            local=old[y0:y1,x0:x1];ground=gt[y0:y1,x0:x1];components=np.unique(local[flank&ground]);components=components[components>0]
            if len(components)<2:continue
            def indices(value):
                yy,xx=np.nonzero(value);return ((yy+y0)*w+xx+x0).tolist()
            gaps.append({'id':i,'length':length,'length_bin':'3-7' if length<8 else '8-15' if length<16 else '16-31' if length<32 else '32+',
                         'pixels':indices(region),'anchors':[indices(flank&(local==component)&ground) for component in components]})
        total+=len(gaps);images[case['case']]={'shape':[h,w],'gaps':gaps}
    return {'images':images,'count':total,'p0_reference_threshold':.45,'sha256':reference_digest(cases),
            'origin':'Frozen once from baseline A best-checkpoint validation output at 0.45; gap/anchors are proxies.'}


def point(cases,name,threshold,manifest):
    rows=[];closed=bridges=0;ids={};local_closed=0;local_ids={}
    bins={key:{'eligible':0,'global_closed':0,'local_closed':0} for key in ('3-7','8-15','16-31','32+')}
    for case in cases:
        pred=case[name]>=threshold;rows.append({'case_id':case['case'],**image_metrics(pred,prepared(case),PROTOCOL)})
        per_bin,gap_ids=closed_gaps(pred,manifest['images'][case['case']]);ids[case['case']]=gap_ids;closed+=len(gap_ids)
        for key,counts in per_bin.items():
            bins[key]['eligible']+=counts['eligible'];bins[key]['global_closed']+=counts['closed']
        local_ids[case['case']]=[]
        for gap in manifest['images'][case['case']]['gaps']:
            pixels=np.asarray(gap['pixels']);anchors=[np.asarray(a) for a in gap['anchors']]
            if pred.ravel()[pixels].mean()<.8:continue
            points=np.concatenate([pixels]+anchors);yy,xx=np.divmod(points,pred.shape[1])
            y0,y1=max(0,int(yy.min())-8),min(pred.shape[0],int(yy.max())+9)
            x0,x1=max(0,int(xx.min())-8),min(pred.shape[1],int(xx.max())+9)
            _,local=cv2.connectedComponents(pred[y0:y1,x0:x1].astype(np.uint8),connectivity=8)
            groups=[]
            for anchor in anchors:
                ay,ax=np.divmod(anchor,pred.shape[1]);groups.append(set(np.unique(local[ay-y0,ax-x0]))-{0})
            if groups and set.intersection(*groups):
                local_ids[case['case']].append(gap['id']);local_closed+=1;bins[gap['length_bin']]['local_closed']+=1
        bridges+=bridge_proxy(case['A']>=.45,pred,case['gt'])
    return {'threshold':float(threshold),'metrics':summarize(rows),'closed':closed,'closed_ids':ids,
            'local_closed':local_closed,'local_closed_ids':local_ids,'gap_by_length':bins,'bridge_proxy':bridges}


def area_threshold(cases,name,target):
    values=np.concatenate([c[name].ravel() for c in cases]);target=max(0,min(int(target),len(values)))
    if target==0:return float(np.nextafter(values.max(),np.float32(np.inf))),0
    cutoff=np.partition(values,len(values)-target)[len(values)-target]
    candidates=[float(cutoff),float(np.nextafter(cutoff,np.float32(np.inf)))]
    value=min(candidates,key=lambda v:abs(int((values>=v).sum())-target))
    return value,int((values>=value).sum())-target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--root_path',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    device=torch.device(args.device);base,config_a=load_model(args.baseline,device,False)
    native,config_b=load_model(args.native,device,True)
    for key in ('seed','root_path','batch_size','img_size','source_patch_size','max_epochs','pretrained_lr','new_lr','warmup_epochs','use_ema'):
        if config_a[key]!=config_b[key]:raise ValueError('Experiment settings differ: '+key)
    if Path(config_a['root_path']).resolve()!=args.root_path.resolve():raise ValueError('Evaluation data root differs from training')
    for checkpoint in (args.baseline,args.native):
        last=torch.load(checkpoint.parent/'last.pth',map_location='cpu',weights_only=False)
        if last.get('epoch')!=80:raise ValueError('Expected a completed 80-epoch run: '+str(checkpoint.parent))
        del last
    initial_a=json.loads((args.baseline.parent/'initial_state.json').read_text())
    initial_b=json.loads((args.native.parent/'initial_state.json').read_text())
    if initial_a['seed']!=initial_b['seed'] or initial_a['pretrain_sha256']!=initial_b['pretrain_sha256']:
        raise ValueError('Initial seed/pretraining differs')
    shared=set(initial_a['tensors'])&set(initial_b['tensors'])
    shared={key for key in shared if not key.startswith('swin_unet.decoder_structure_blocks.3.skeleton_head.')}
    if not shared or any(initial_a['tensors'][key]!=initial_b['tensors'][key] for key in shared):
        raise ValueError('Shared initialized tensors differ')
    for epoch in range(1,81):
        a=json.loads((args.baseline.parent/f'input_epoch_{epoch:03d}.json').read_text())
        b=json.loads((args.native.parent/f'input_epoch_{epoch:03d}.json').read_text())
        if a!=b:raise ValueError(f'Training input differs at epoch {epoch}')
    dataset=RoadSkeletonDataset(str(args.root_path),'val',image_size=None,source_patch_size=1500,augment=False,return_full_image=True)
    if len(dataset)!=14:raise ValueError('Expected 14 validation images')
    cases=[];diagnostics={'A':{},'B':{}};args.output_dir.mkdir(parents=True,exist_ok=True)
    for i in range(len(dataset)):
        sample=dataset[i];a,gt,da=infer(base,sample,device,True);b,gt_b,db=infer(native,sample,device,True)
        if not np.array_equal(gt,gt_b):raise ValueError('Validation GT mismatch')
        no_global=intervention(native,sample,device,native.swin_unet.native_global_context.raw_alpha)
        no_local=intervention(native,sample,device,native.swin_unet.decoder_structure_blocks['3'].gamma)
        name=sample['case_name'];cases.append({'case':name,'gt':gt,'p0':a,'A':a,'B':b,
                                                'no_global':no_global,'no_local':no_local})
        diagnostics['A'][name]=da;diagnostics['B'][name]=db
        np.savez_compressed(args.output_dir/(name+'.npz'),gt=gt,A=a,B=b,no_global=no_global,no_local=no_local)
        print(f'[VAL PAIR] {i+1}/14 {name}',flush=True)
    manifest=fixed_reference(cases)
    (args.output_dir/'fixed_gaps_and_anchors.json').write_text(json.dumps(manifest))
    (args.output_dir/'structure_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    summary={}
    for name,images in diagnostics.items():
        total=Counter()
        for values in images.values():total.update(values)
        summary[name]=dict(total)
        for stage in (2,3):
            for label in ('pos','neg','negative_road_road'):
                denom=total[f'C{stage}_{label}_edges']
                summary[name][f'C{stage}_{label}_acceptance']=total[f'C{stage}_{label}_accepted']/denom if denom else None
            count=total[f'D{stage}_valid_cells']
            summary[name][f'D{stage}_mean_angle_degrees']=total[f'D{stage}_angle_sum']/count if count else None
        for label in ('correct','incorrect','incorrect_road_road'):
            count=total[f'choice_{label}_edges']
            summary[name][f'choice_{label}_mean_weight']=total[f'choice_{label}_sum']/count if count else None
    (args.output_dir/'structure_summary.json').write_text(json.dumps(summary,indent=2))
    thresholds=np.round(np.arange(.05,.951,.05),2);curves={}
    for name in ('A','B'):
        curves[name]=[]
        for threshold in thresholds:
            curves[name].append(point(cases,name,float(threshold),manifest))
            print(f'[SWEEP] {name} {threshold:.2f}',flush=True)
    selected={name:max(curve,key=lambda x:x['metrics']['f1']) for name,curve in curves.items()}
    threshold_b=selected['B']['threshold'];interventions={}
    for key in ('no_global','no_local'):
        total=0;absolute=0.;flips=0
        for case in cases:
            diff=np.abs(case['B']-case[key]);absolute+=float(diff.sum(dtype=np.float64));total+=diff.size
            flips+=int(((case['B']>=threshold_b)!=(case[key]>=threshold_b)).sum())
        interventions[key]={'mean_abs_P_difference':absolute/total,'threshold':threshold_b,'binary_flips':flips,
                            'note':'fixed-checkpoint inference intervention; diagnostic only'}
    (args.output_dir/'interventions.json').write_text(json.dumps(interventions,indent=2))
    a_t=selected['A']['threshold'];target_tp=sum(int(((c['A']>=a_t)&c['gt']).sum()) for c in cases)
    target_area=sum(int((c['A']>=a_t).sum()) for c in cases)
    recall_t,recall_err=exact_recall_threshold(cases,'B',target_tp);area_t,area_err=area_threshold(cases,'B',target_area)
    reports={'A_own_val':selected['A'],'B_own_val':selected['B'],'B_same_threshold':point(cases,'B',a_t,manifest),
             'B_matched_recall':dict(point(cases,'B',recall_t,manifest),matching_error=recall_err),
             'B_matched_area':dict(point(cases,'B',area_t,manifest),matching_error=area_err)}
    gap_rows=[]
    for case in cases:
        name=case['case'];old=reports['A_own_val']['local_closed_ids'][name]
        for gap in manifest['images'][name]['gaps']:
            gid=gap['id'];after=gid in reports['B_matched_recall']['local_closed_ids'][name];before=gid in old
            gap_rows.append({'case':name,'gap_id':gid,'length_bin':gap['length_bin'],'A_local_closed':before,
                             'B_matched_recall_local_closed':after,'newly_closed':after and not before,'lost':before and not after})
    with (args.output_dir/'fixed_gap_changes.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['case','gap_id','length_bin','A_local_closed','B_matched_recall_local_closed','newly_closed','lost']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(gap_rows)
    payload={'selected_on':'val only; max F1 threshold per model','checkpoint_A_sha256':digest(args.baseline),
             'checkpoint_B_sha256':digest(args.native),'fixed_gaps':manifest['count'],'curves':curves,'reports':reports,
             'limitations':['APLS, bridge and fixed gaps are proxies.','Source/local C is not a distant path reachability label.']}
    (args.output_dir/'comparison.json').write_text(json.dumps(payload,indent=2))
    with (args.output_dir/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['name','threshold','iou','f1','precision','recall','cldice','apls_gt_to_pred_approx',
                'apls_pred_to_gt_approx','apls_bidirectional_approx','short_pred_components','closed','local_closed','bridge_proxy','matching_error']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,report in reports.items():
            row={'name':name,'threshold':report['threshold'],'closed':report['closed'],'local_closed':report['local_closed'],'bridge_proxy':report['bridge_proxy'],
                 'matching_error':report.get('matching_error',0)}
            row.update({k:report['metrics'][k] for k in fields if k in report['metrics']});writer.writerow(row)
    print('[DONE]',args.output_dir,flush=True)


if __name__=='__main__':main()
