"""Val-only matched operating points and actual on-policy transport diagnostics."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import run_reliable_v9_mass as v9
from networks.reliable_directional_repair import ReliableDirectionalRepair
from networks.v9_transport_benefit import forward_trace,gate_benefit_loss,BenefitMetrics,restore_model
from diagnostics.diagnose_v9_transport import matched_threshold
from diagnostics.v91_counterfactual import evaluate,write_csv
from evaluation.road_comparison.road_metrics import prepare_target,image_metrics,summarize,skeletonize
from mass_protocol import blend_weight,sliding_positions


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8'); temp.replace(path)


class StepModel(nn.Module):
    def __init__(self,model,steps): super().__init__(); self.model=model; self.steps=steps
    def forward(self,image,capture_diagnostics=False): return forward_trace(self.model,image,self.steps)


def create_variant(frozen,path,identity,device):
    model=ReliableDirectionalRepair(frozen,width=32,steps=3,bridge_route=True,
        conditional_transport=True,transport_gate_mode='v7',gate_relation_mode='full').to(device)
    model.state_gate=v9.StateGate().to(device)
    restore_model(model,path,identity); model.eval(); return model


@torch.no_grad()
def infer_cases(model,dataset,device,steps,name,output,checkpoint):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    marker=output/'identity.json'; expected={'checkpoint_sha256':v9.digest(checkpoint),'steps':steps}
    if marker.is_file() and json.loads(marker.read_text())!=expected: raise ValueError(f'Prediction cache identity changed: {output}')
    atomic_json(marker,expected); cases=[]; model.eval()
    for index in range(len(dataset)):
        sample=dataset[index]; case=str(sample['case_name']); file=output/(hashlib.sha256(case.encode()).hexdigest()[:20]+'.npz')
        if file.is_file():
            with np.load(file,allow_pickle=False) as data: cases.append({'case':case,'gt':data['gt'].astype(bool),'p0':data['p0'],name:data['p1']})
            continue
        image=sample['image'][None].to(device); gt=sample['mask'][0].numpy()>.5; h,w=gt.shape
        canvas=torch.zeros((2,1,h,w),device=device); divisor=torch.zeros((1,1,h,w),device=device); weight=blend_weight(512,device)
        for y in sliding_positions(h,512,256):
            for x in sliding_positions(w,512,256):
                result=forward_trace(model,image[:,:,y:y+512,x:x+512],steps)
                region=(slice(None),slice(None),slice(y,y+512),slice(x,x+512))
                canvas[region]+=torch.cat((result['z0'],result['z1']),0).float()*weight; divisor[region]+=weight
        if not bool((divisor>0).all()): raise RuntimeError('Uncovered native image pixels')
        logits=(canvas/divisor).cpu().numpy()[:,0]
        probability=1./(1.+np.exp(-np.clip(logits,-80.,80.)))
        if not np.isfinite(probability).all(): raise RuntimeError('Nonfinite prediction')
        temp=file.with_suffix('.tmp')
        with temp.open('wb') as handle: np.savez_compressed(handle,p0=probability[0],p1=probability[1],gt=gt)
        temp.replace(file)
        cases.append({'case':case,'gt':gt,'p0':probability[0],name:probability[1]})
        print(f'[VAL INFERENCE] {name} steps={steps} image={index+1}/{len(dataset)}',flush=True)
    return cases


def threshold_curve(cases,name,output):
    from run_q3_r3_mass import PROTOCOL
    from run_reliable_directional_mass import gap_lengths,selection_score
    output=Path(output); marker=output/(name+'_threshold_curve.json')
    if marker.is_file(): return {float(key):value for key,value in json.loads(marker.read_text()).items()}
    thresholds=tuple(np.round(np.arange(.35,.751,.01),2).tolist())
    rows={threshold:[] for threshold in thresholds}; bridges=Counter(); per_image=[]
    for index,case in enumerate(cases):
        prepared=prepare_target(case['gt'],PROTOCOL); p0=case['p0']>=.45
        for threshold in thresholds:
            pred=case[name]>=threshold
            row={'case_id':case['case'],**image_metrics(pred,prepared,PROTOCOL)}
            _,bridge=gap_lengths(p0,pred,prepared['skeleton'],case['gt']); bridges[threshold]+=bridge
            rows[threshold].append(row); per_image.append({'case':case['case'],'threshold':threshold,**row,'background_bridge_proxy':bridge})
        print(f'[THRESHOLD METRICS] {name} image={index+1}/{len(cases)}',flush=True)
    curve={}
    for threshold in thresholds:
        metrics=summarize(rows[threshold]); score=selection_score(metrics,{'background_bridge_proxy':bridges[threshold]},len(cases))
        curve[threshold]={'metrics':metrics,'selection_score':score,'bridge_proxy':bridges[threshold]}
    write_csv(output/(name+'_threshold_per_image.csv'),per_image); atomic_json(marker,curve); return curve


def operating_points(cases,names,curves,original_threshold):
    target_tp=sum(int(((case['A_original']>=original_threshold)&case['gt']).sum()) for case in cases)
    target_area=sum(int((case['A_original']>=original_threshold).sum()) for case in cases)
    return {name:{'same_threshold':original_threshold,
        'own_val_threshold':max(curves[name],key=lambda threshold:curves[name][threshold]['selection_score']),
        'matched_recall':original_threshold if name=='A_original' else matched_threshold(cases,name,target_tp,'tp',10000)[0],
        'matched_foreground_area':original_threshold if name=='A_original' else matched_threshold(cases,name,target_area,'area',10000)[0]}
        for name in names}


def save_step_visuals(output,name,case,result,gt,origin):
    directory=Path(output)/'visuals'/name; directory.mkdir(parents=True,exist_ok=True)
    states=np.stack([state[0,0].detach().cpu().numpy() for state in result['states']])
    gate=result['g_step'][0,:,0].detach().cpu().numpy()
    source=result['source_available'][0,:,0].detach().cpu().numpy()
    transport=result['transport_delta'][0,:,0].detach().cpu().numpy()
    reliable=result['r'][0,0].detach().cpu().numpy()
    np.savez_compressed(directory/(case+'.npz'),states=states,gates=gate,source_V=source,transport=transport,
        R=reliable,gt=gt,tile_origin_native=np.asarray(origin),native_per_work_pixel=np.asarray(gt.shape)/np.asarray(states.shape[-2:]))
    panels=[]
    def add(array,label,signed=False):
        values=np.clip(array*2.+.5 if signed else array,0.,1.)
        color=cv2.applyColorMap(np.uint8(values*255),cv2.COLORMAP_TURBO if signed else cv2.COLORMAP_VIRIDIS)
        color=cv2.resize(color,(256,256),interpolation=cv2.INTER_NEAREST)
        cv2.putText(color,label,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        panels.append(color)
    for k,state in enumerate(states): add(state,'U'+str(k))
    add(reliable,'R initial'); add(gate[-1],'G final'); add(source[-1],'V final'); add(transport[-1],'transport final',True)
    add(gt.astype(np.float32),'GT (display only)')
    while len(panels)%4: panels.append(np.zeros((256,256,3),np.uint8))
    sheet=np.concatenate([np.concatenate(panels[k:k+4],axis=1) for k in range(0,len(panels),4)],axis=0)
    cv2.imwrite(str(directory/(case+'.png')),sheet)


@torch.no_grad()
def decision_audit(args,models,steps_map,cases,points,benefit,output):
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(args.root_path,'val'); by_case={case['case']:case for case in cases}
    decision={name:BenefitMetrics(steps_map[name]) for name in models}
    regional=defaultdict(Counter)
    for index in range(len(dataset)):
        sample=dataset[index]; case=by_case[str(sample['case_name'])]
        image=sample['image'][None].to(args.device); gt=case['gt']; h,w=gt.shape; p0=case['p0']>=.45
        missing=skeletonize(gt)&~(cv2.dilate(p0.astype(np.uint8),np.ones((5,5),np.uint8))>0)
        n,components,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),connectivity=8)
        length_masks={label:np.zeros_like(gt) for label in ('3-7','8-15','16-31','32+')}
        for component in range(1,n):
            size=int(stats[component,cv2.CC_STAT_AREA])
            if size<3: continue
            label='3-7' if size<8 else '8-15' if size<16 else '16-31' if size<32 else '32+'
            length_masks[label]|=components==component
        masks={}
        for name in models:
            pred=case[name]>=points[name]['matched_recall']
            isolated,bridge=v9.background_components(p0,pred,gt)
            masks[name]={'missing_center':missing,'recovered_gt':gt&~p0&pred,'unrecovered_gt':gt&~p0&~pred,
                         'isolated_new_fp':isolated,'false_bridge':bridge}
            masks[name].update({'missing_center_'+label:value for label,value in length_masks.items()})
        positions=[(y,x) for y in sliding_positions(h,512,256) for x in sliding_positions(w,512,256)]
        best_tile=max(positions,key=lambda position:int(missing[position[0]:position[0]+512,position[1]:position[1]+512].sum()))
        owned=np.zeros((h,w),bool)
        for y in sliding_positions(h,512,256):
            for x in sliding_positions(w,512,256):
                frozen=next(iter(models.values())).frozen(image[:,:,y:y+512,x:x+512],capture_diagnostics=True)
                own=~owned[y:y+512,x:x+512]; owned[y:y+512,x:x+512]=True
                target=sample['mask'][None,:,y:y+512,x:x+512].to(args.device)
                for name,model in models.items():
                    result=forward_trace(model,None,steps_map[name],frozen_override=frozen)
                    wh,ww=result['r'].shape[-2:]
                    yy=np.minimum(((np.arange(wh)+.5)*512/wh).astype(int),511)
                    xx=np.minimum(((np.arange(ww)+.5)*512/ww).astype(int),511)
                    own_work=own[yy[:,None],xx[None,:]]
                    ownership=torch.as_tensor(own_work,device=args.device)[None,None]
                    _,labels,selected,_=gate_benefit_loss(result,target,benefit,args.seed+index*1000+y+x,ownership)
                    decision[name].add(labels,selected,result)
                    if index<4 and (y,x)==best_tile:
                        save_step_visuals(output,name,str(sample['case_name']),result,gt[y:y+512,x:x+512],(y,x))
                    region_work={key:value[y:y+512,x:x+512][yy[:,None],xx[None,:]]&own_work for key,value in masks[name].items()}
                    fraction=F.interpolate(target.float(),size=(wh,ww),mode='area')[0,0].cpu().numpy()
                    region_work['hard_background']=(fraction<=.05)&(result['q'][0,0].cpu().numpy()>=.5)&own_work
                    p0work=result['states'][0]
                    for k in range(steps_map[name]):
                        u=result['states'][k]; gained=((u-p0work).clamp_min(0.)/(1.-p0work).clamp_min(1e-5)).clamp_max(1.)
                        fields={'V':result['source_available'][:,k],'G':result['g_step'][:,k],
                            'potential':result['transport_rate'][:,k]*(result['m'][:,k]-u),
                            'actual_transport':result['transport_delta'][:,k],
                            'proposal':result['proposal_delta'][:,k],
                            'U_change_from_P0':result['states'][k+1]-p0work,
                            'generated_source':(1.-result['r'])*gained}
                        for field,tensor in fields.items():
                            array=tensor[0,0].cpu().numpy()
                            for region,mask in region_work.items():
                                values=array[mask]
                                if not len(values): continue
                                row=regional[(name,k+1,region,field)]; row['n']+=len(values)
                                row['sum']+=float(values.sum(dtype=np.float64)); row['abs_sum']+=float(np.abs(values).sum(dtype=np.float64))
                                row['positive']+=int((values>benefit.min_action).sum())
                                row['near_zero']+=int((np.abs(values)<benefit.min_action).sum())
        print(f'[TRANSPORT AUDIT] image={index+1}/{len(dataset)}',flush=True)
    rows=[{'model':key[0],'step':key[1],'region':key[2],'quantity':key[3],'pixels':row['n'],
           'mean':row['sum']/row['n'],'mean_abs':row['abs_sum']/row['n'],
           'positive_fraction':row['positive']/row['n'],'near_zero_fraction':row['near_zero']/row['n']} for key,row in regional.items()]
    write_csv(Path(output)/'dynamics_regions.csv',rows)
    reports={name:value.report() for name,value in decision.items()}
    atomic_json(Path(output)/'decision_metrics.json',{'models':reports,
        'sampling':'all windows, first-window work-cell-center ownership; on-policy model states/actions, not a fixed-input gate-only comparison',
        'regions':'Matched-Recall masks per model on full native images. Length-specific missing-center components are not the same denominator as the206 closure-eligible gaps. All gap/bridge/APLS measures are proxies.'})
    return reports


def write_comparison(output,reports,decisions,points,curves,label):
    atomic_json(Path(output)/'comparison.json',{'label':label,'selected_on':'val','points':points,'reports':reports,
        'decisions':decisions,'selection_rule':'original V9 weighted score, val thresholds .35:.01:.75',
        'threshold_curves':curves,'note':'APLS/closure/bridges are proxies; histogram matching resolution .0001; decision populations are on-policy.'})
    lines=['# V9 transport benefit gate comparison','',label,'',
        '| model | work point | threshold | recall | IoU | clDice | bidirectional approximate APLS | closed / 206 | bridge proxy | short components |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,data in reports.items():
        for point,row in data.items():
            m=row['metrics']; p=row['paired']
            lines.append(f'| {name} | {point} | {row["threshold"]:.4f} | {m["recall"]:.5f} | {m["iou"]:.5f} | {m["cldice"]:.5f} | '
                f'{m["apls_bidirectional_approx"]:.5f} | {p["closed"]}/206 | {p["background_bridge_proxy"]} | {m["short_pred_components"]:.2f} |')
    lines+=['','| model | matched helpful accept | matched harmful accept | matched ROC-AUC | harmful actual BCE loss / valid action | beneficial actual BCE gain / valid action |',
            '|---|---:|---:|---:|---:|---:|']
    for name,row in decisions.items():
        lines.append(f'| {name} | {row["matched_helpful_acceptance_rate"]} | {row["matched_harmful_acceptance_rate"]} | '
                     f'{row["matched_auc_exact"]["roc_auc"]} | '
                     f'{row["harmful_actual_loss_per_valid_action"]:.7f} | {row["beneficial_actual_gain_per_valid_action"]:.7f} |')
    lines+=['','Matching controls image/step/action sign and V/action-magnitude bands. Different models still create different action populations; these results do not alone isolate a gate-only classifier effect.',
            'Inspect matching residuals, gap length bins and added TP/FP in the JSON/CSV, and per-step missing-center/source/transport dynamics in dynamics_regions.csv.']
    (Path(output)/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_comparison(args,model,identity,source,selection,benefit,include6=False):
    from run_q3_r3_mass import full_dataset
    output=args.output_dir/('comparison_with_trained6' if include6 else 'comparison_3step'); output.mkdir(parents=True,exist_ok=True)
    checkpoints={'A_original':source,'B_control':args.output_dir/'control_3step/best.pth','C_gate3':args.output_dir/'gate_3step/best.pth'}
    if include6: checkpoints['D_gate6']=args.output_dir/'gate_6step/best.pth'
    steps_map={name:6 if name=='D_gate6' else 3 for name in checkpoints}
    dataset=full_dataset(args.root_path,'val')
    if len(dataset)!=14: raise ValueError('Expected the same 14 validation images')
    cases=[]; curves={}
    for name,path in checkpoints.items():
        payload=restore_model(model,path,identity,steps_map[name])
        if name!='A_original' and (payload.get('benefit_version')!='v9_detached_transport_benefit_gate_v1' or
                                  payload.get('benefit_config',{}).get('steps')!=steps_map[name]):
            raise ValueError('Checkpoint lacks the expected benefit-label or step-count protocol')
        if name!='A_original':
            config=payload['benefit_config']
            if config.get('variant')!=('control' if name=='B_control' else 'gate') or config.get('original_sha256')!=v9.digest(source):
                raise ValueError('Continuation variant or original checkpoint provenance differs')
        values=infer_cases(model,dataset,args.device,steps_map[name],name,args.output_dir/'predictions_val'/name,path)
        if name=='A_original': cases=[dict(case) for case in values]
        else:
            for target,case in zip(cases,values):
                if target['case']!=case['case'] or not np.array_equal(target['p0'],case['p0']): raise ValueError('P0 or case identity differs')
                target[name]=case[name]
        if name=='A_original':
            curves[name]={float(key):{'metrics':value['p1'],'selection_score':value['selection_score'],
                'bridge_proxy':value['paired']['background_bridge_proxy']} for key,value in selection['threshold_curves'].items()}
        else: curves[name]=threshold_curve(cases,name,output)
    original_threshold=float(selection['threshold'])
    saved=selection['report']['p1']
    counts={'tp':sum(int(((case['A_original']>=original_threshold)&case['gt']).sum()) for case in cases),
            'fp':sum(int(((case['A_original']>=original_threshold)&~case['gt']).sum()) for case in cases),
            'fn':sum(int(((case['A_original']<original_threshold)&case['gt']).sum()) for case in cases)}
    if any(counts[key]!=saved[key] for key in counts): raise ValueError('Original V9 predictions do not reproduce saved validation counts')
    names=list(checkpoints); points=operating_points(cases,names,curves,original_threshold)
    reports=evaluate(cases,points,.45,206,output,modes=names)
    if json.loads((output/'reference_set.json').read_text())!=json.loads(args.reference_set.read_text()): raise ValueError('The fixed P0/GT/skeleton fingerprint differs')
    models={name:create_variant(model.frozen,path,identity,args.device) for name,path in checkpoints.items()}
    decisions=decision_audit(args,models,steps_map,cases,points,benefit,output)
    write_comparison(output,reports,decisions,points,curves,'Equal-budget control and L_G continuations from the exact original V9.')
    left=reports['B_control']['matched_recall']; right=reports['C_gate3']['matched_recall']
    a=decisions['B_control']; b=decisions['C_gate3']
    rates=all(value is not None for value in (a['matched_harmful_acceptance_rate'],b['matched_harmful_acceptance_rate'],
             a['matched_helpful_acceptance_rate'],b['matched_helpful_acceptance_rate']))
    success=(rates and b['matched_harmful_acceptance_rate']<a['matched_harmful_acceptance_rate'] and
        b['matched_auc_exact']['roc_auc'] is not None and a['matched_auc_exact']['roc_auc'] is not None and
        b['matched_auc_exact']['roc_auc']>a['matched_auc_exact']['roc_auc']+1e-6 and
        b['matched_helpful_acceptance_rate']>=a['matched_helpful_acceptance_rate'] and
        b['harmful_actual_loss_per_valid_action']<=a['harmful_actual_loss_per_valid_action'] and
        right['paired']['background_bridge_proxy']<left['paired']['background_bridge_proxy'] and
        right['paired']['closed']>=left['paired']['closed'] and
        abs(right['metrics']['recall']-left['metrics']['recall'])<=.001)
    screening={'passed':bool(success),'rule':'At matched Recall (residual difference<=.001): higher matched decision ROC-AUC, fewer harmful accepts and actual BCE harm, no lower helpful acceptance, fewer bridge proxies, no fewer closed fixed reference gaps.',
               'note':'A conservative val-only screening condition, not proof of a fixed-input gate-only causal effect.'}
    atomic_json(output/'transport_selection_screen.json',screening)
    return {'cases':cases,'curves':curves,'reports':reports,'decisions':decisions,'points':points,'screening':screening}


def run_step_diagnostics(args,model,identity,source,selection,benefit,paired,train_variant):
    output=args.output_dir/'steps_inference_diagnostic'; output.mkdir(parents=True,exist_ok=True)
    if not paired['screening']['passed']:
        atomic_json(output/'status.json',{'status':'skipped_transport_selection_not_improved','screening':paired['screening']})
        print('[STEP DIAGNOSTICS SKIPPED] First improve selective transport; see comparison_3step/REPORT.md',flush=True)
        return
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(args.root_path,'val'); checkpoint=args.output_dir/'gate_3step/best.pth'
    cases=[{'case':case['case'],'gt':case['gt'],'p0':case['p0'],'A_original':case['A_original'],'S3':case['C_gate3']} for case in paired['cases']]
    curves={'A_original':paired['curves']['A_original'],'S3':paired['curves']['C_gate3']}
    for steps in (6,12):
        name='S'+str(steps); restore_model(model,checkpoint,identity,3)
        values=infer_cases(model,dataset,args.device,steps,name,args.output_dir/'predictions_val'/name,checkpoint)
        for target,case in zip(cases,values):
            if target['case']!=case['case'] or not np.array_equal(target['p0'],case['p0']): raise ValueError('Step diagnostic P0 mismatch')
            target[name]=case[name]
        curves[name]=threshold_curve(cases,name,output)
    names=['A_original','S3','S6','S12']; points=operating_points(cases,names,curves,float(selection['threshold']))
    reports=evaluate(cases,points,.45,206,output,modes=names)
    models={name:create_variant(model.frozen,checkpoint,identity,args.device) for name in ('S3','S6','S12')}
    decisions=decision_audit(args,models,{'S3':3,'S6':6,'S12':12},cases,points,benefit,output)
    write_comparison(output,reports,decisions,points,curves,'Inference-only 3/6/12 steps using identical learned three-step weights. Not evidence of a trained 6/12-step architecture benefit.')
    three=reports['S3']['matched_recall']; six=reports['S6']['matched_recall']
    better=(six['paired']['closed']>three['paired']['closed'] and
        six['paired']['background_bridge_proxy']<=three['paired']['background_bridge_proxy'] and
        six['metrics']['apls_bidirectional_approx']>=three['metrics']['apls_bidirectional_approx'] and
        decisions['S6']['harmful_actual_loss_per_valid_action']<=decisions['S3']['harmful_actual_loss_per_valid_action'] and
        abs(six['metrics']['recall']-three['metrics']['recall'])<=.001)
    status={'inference_only':True,'six_steps_balanced_val_improvement':bool(better),'auto_train6_enabled':args.auto_train6,
            'trained6_started':False,'rule':'More closed gaps, no more bridge proxy, no lower bidirectional APLS, no more per-action BCE harm at matched Recall.'}
    if better and args.auto_train6:
        status['trained6_started']=True; atomic_json(output/'status.json',status)
        train_variant(args,model,source,identity,'gate',6)
        # Check the same complete image order and augmentation tensors for the equal-budget step ablation.
        original=[json.loads(line) for line in (args.output_dir/'gate_3step/epochs.jsonl').read_text().splitlines() if line.strip()]
        new=[json.loads(line) for line in (args.output_dir/'gate_6step/epochs.jsonl').read_text().splitlines() if line.strip()]
        for a,b in zip(original,new):
            if a['epoch']!=b['epoch'] or any(a[key]!=b[key] for key in ('case_order_sha256','augmented_images_sha256')):
                raise ValueError('Three/six-step training data schedules differ')
        if len(original)!=len(new): raise ValueError('Three/six-step epoch budgets differ')
        run_comparison(args,model,identity,source,selection,benefit,include6=True)
    atomic_json(output/'status.json',status)
    print(f'[STEP DIAGNOSTICS DONE] {output}',flush=True)
