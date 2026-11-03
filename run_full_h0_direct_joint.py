"""Continue from the original A H0 plus pretrained unscaled repair; val only."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnostics.diagnose_full_h0_optimistic_reachability import verify_recorded_code
from diagnostics.diagnose_unscaled_proposal_effect import verify_adaptation_code
from diagnostics.audit_full_h0_saved_predictions import component_data, gap_state
from diagnostics.v91_counterfactual import write_csv
from networks.full_h0_direct_joint import enable_joint_training, joint_loss
from networks.full_h0_joint import original_criterion, parameter_groups
from networks.v9_selector_transport_training import gradient_norm
from run_direct_candidate import build_model, candidate_metrics, reference_regions, stitch
from run_full_h0_joint import check_reusable_A, load_h0, norms, restore, save, training_dataset
from run_q3_r3_mass import full_dataset
from run_slim_end_to_end_mass import digest, schedule, seed_all
from run_v9_gate_benefit import atomic_json
from networks.v9_source_relation_labels import RelationLabels
from run_v9_selector_transport import point_metrics
from selector_transport_reference import exact_recall_threshold
from evaluation.road_comparison.road_metrics import skeletonize

ROOT = Path(__file__).resolve().parent
EPOCHS = 20


def configure(pair, source_path, run_name, device):
    pair, source_path = pair.resolve(), source_path.resolve()
    if not source_path.is_file(): raise FileNotFoundError(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    previous = source['config']
    verify_recorded_code(previous)
    verify_adaptation_code(previous)
    if previous['epochs'] != EPOCHS: raise ValueError('Source must use the existing 20-epoch paired experiment')
    check_reusable_A(pair, previous)
    selection = json.loads((source_path.parent/'eval/val_selection.json').read_text())
    if selection['checkpoint_sha256'] != digest(source_path): raise ValueError('Source checkpoint and val selection differ')
    manifest_path = pair/'reference/fixed_gaps_and_anchors.json'
    if digest(manifest_path) != previous['reference_sha256']: raise ValueError('Fixed reference differs')
    h0_path = Path(previous['h0_checkpoint'])
    if not h0_path.is_file() or digest(h0_path) != previous['h0_sha256']: raise ValueError('A initial H0 missing or changed')
    seed_all(int(previous['seed']))
    model = build_model(source, device)
    initial, _, original_args = load_h0(h0_path, torch.device('cpu'))
    if original_args != previous['h0_args']: raise ValueError('Original H0 configuration differs')
    model.base.load_state_dict(initial.state_dict(), strict=True)
    if any(not torch.equal(v.detach().cpu(), initial.state_dict()[k]) for k, v in model.base.state_dict().items()):
        raise RuntimeError('H0 initialization does not match A')
    source_state = source['model']
    for name, value in model.state_dict().items():
        if not name.startswith('original.upstream.base.') and not torch.equal(value.cpu(), source_state[name].cpu()):
            raise RuntimeError('Repair initialization differs: '+name)
    enable_joint_training(model)
    h0_lr = float(previous['h0_lr']); repair_lr = 10*h0_lr
    cfg = copy.deepcopy(previous)
    cfg['run_directory'] = run_name
    cfg['direct_joint'] = {
        'version':'joint_direct_candidate_v1','source_checkpoint':str(source_path),
        'source_sha256':digest(source_path),'source_epoch':source['epoch'],
        'h0_initial_checkpoint':str(h0_path),'h0_initial_sha256':digest(h0_path),
        'A_checkpoint_sha256':digest(pair/'A/best.pth'),
        'objective':'0.5*seg(P0)+0.5*seg(P1)+0.25*seg(B)+H0_aux+0.2*continuity+original_relation',
        'h0_aux':'original H0 criterion minus its original surface term',
        'segmentation':'existing Q3/missing-weighted focal+dice for P0,P1,B',
        'h0_lr':h0_lr,'repair_lr':repair_lr,'choice_loss':0.,'gate_benefit_loss':0.,
        'epochs':EPOCHS,'sampling':'original random 512, same A input hashes',
        'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'run_full_h0_direct_joint.py',ROOT/'networks/full_h0_direct_joint.py',ROOT/'networks/direct_candidate.py')},
    }
    args = argparse.Namespace(pair_dir=pair, output_dir=pair/run_name, checkpoint=source_path,
        root_path=Path(previous['root_path']), device=device, seed=int(previous['seed']),
        num_workers=int(previous['num_workers']), epochs=EPOCHS, variant='B', b_run_name=run_name,
        h0_lr=h0_lr, config=cfg, manifest=json.loads(manifest_path.read_text()))
    if args.manifest['count'] != 198: raise ValueError('Expected fixed 198 gaps')
    del initial, source, source_state
    return args, model, original_criterion(previous['h0_args'], device), repair_lr


def make_optimizer(model, h0_lr, repair_lr):
    optimizer = parameter_groups(model, h0_lr, repair_lr=repair_lr)
    for group in optimizer.param_groups:
        lr = h0_lr if group['name']=='H0' else repair_lr
        group.update(lr=lr,initial_lr=lr,minimum_lr=.1*lr)
    return optimizer


def smoke(args,model,criterion):
    dataset=training_dataset(args,False);dataset.set_epoch(1)
    batch=next(iter(DataLoader(dataset,batch_size=2,num_workers=0)))
    batch={k:(v.to(args.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    model.eval();out=model(batch['image'])
    total,terms=joint_loss(model,criterion,out,batch,RelationLabels(**args.config['source_training']['labels']),
                           args.config['lambda_relation'],args.seed,args.config['source_training']['source_losses'])
    grad={
        'B_loss_to_proposal':gradient_norm(terms['seg_b'],tuple(model.original.proposal_head.parameters())),
        'P1_loss_to_selector':gradient_norm(terms['seg_p1'],tuple(model.selector.parameters())),
        'P1_loss_to_rates':gradient_norm(terms['seg_p1'],tuple(model.original.dynamics.rates[-1].parameters())),
        'P1_loss_to_feature_adaptation':gradient_norm(terms['seg_p1'],tuple(model.original.visual.parameters())),
        'P1_loss_to_H0':gradient_norm(terms['seg_p1'],tuple(model.base.parameters())),
    }
    if not torch.isfinite(total) or any(not np.isfinite(x) or x<=0 for x in grad.values()):
        raise RuntimeError('Task gradient blocked: '+str(grad))
    total.backward();report={'passed':True,'loss':float(total.detach()),'task_gradient_norms':grad,
        'all_loss_gradient_norms':norms(model),'GT_in_forward':False,
        'transport_abs_mean':float(out['transport_delta'].detach().abs().mean()),
        'proposal_abs_mean':float(out['proposal_delta'].detach().abs().mean()),
        'P1_minus_P0_abs_mean':float((out['p1']-out['p0']).detach().abs().mean())}
    if any(report[key] == 0 for key in ('transport_abs_mean','proposal_abs_mean','P1_minus_P0_abs_mean')):
        raise RuntimeError('Actual dynamics is numerically inactive: '+str(report))
    atomic_json(args.output_dir/'smoke.json',report);model.zero_grad(set_to_none=True);seed_all(args.seed)
    print('[SMOKE] '+json.dumps(report),flush=True)


@torch.inference_mode()
def val_cases(args,model,save_predictions=False):
    model.eval();dataset=full_dataset(args.root_path,'val');cases=[];candidate_rows=[]
    if len(dataset)!=14:raise ValueError('Expected 14 validation images')
    for i in range(len(dataset)):
        sample=dataset[i];name=str(sample['case_name']);pred=stitch(model,sample,args.device)
        gt=sample['mask'][0].numpy()>.5;skel=sample['skeleton'][0].numpy()>.5
        with np.load(args.pair_dir/'reference'/(name+'.npz')) as ref:
            if not np.array_equal(gt,ref['gt'].astype(bool)):raise ValueError('Validation GT changed')
            initial=ref['p0'].copy()
        regions=reference_regions(pred['p0'],pred['b'],pred['p1'],gt,skel)
        missing=pred['b'].reshape(-1)[regions['missing_points']]
        hard=pred['b'].reshape(-1)[regions['hard_points']]
        for label,values in (('missing',missing),('hard_background',hard)):
            candidate_rows.append({'case':name,'region':label,'pixels':len(values),
                'mean':float(values.mean()) if len(values) else None,
                'p10':float(np.quantile(values,.1)) if len(values) else None,
                'p50':float(np.quantile(values,.5)) if len(values) else None,
                'p90':float(np.quantile(values,.9)) if len(values) else None,
                'fraction_ge_045':float((values>=.45).mean()) if len(values) else None,
                'fraction_ge_075':float((values>=.75).mean()) if len(values) else None})
        case={'case':name,'gt':gt,'p0':initial,'current_p0':pred['p0'],'p1':pred['p1'],
              'b':pred['b'],'eval_skeleton':skeletonize(gt)}
        cases.append(case)
        if save_predictions:
            folder=args.output_dir/'eval/val';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/(name+'.npz'),**{k:v for k,v in case.items() if k not in ('case','eval_skeleton')})
        print(f'[VAL IMAGE] {i+1}/14 {name}',flush=True)
    return cases,candidate_rows


def activity(cases,threshold=.45):
    sums=Counter();absolute=0.
    for case in cases:
        p0,p1,gt=case['current_p0'],case['p1'],case['gt'];before=p0>=threshold;after=p1>=threshold
        absolute+=float(np.abs(p1-p0).sum(dtype=np.float64));sums['pixels']+=gt.size
        sums['added_tp']+=int((after&~before&gt).sum());sums['lost_tp']+=int((before&~after&gt).sum())
        sums['added_fp']+=int((after&~before&~gt).sum());sums['removed_fp']+=int((before&~after&~gt).sum())
    return dict(sums,threshold=threshold,P1_P0_mean_abs=absolute/max(sums['pixels'],1))


def train(args,model,criterion,optimizer):
    last=args.output_dir/'last.pth';start=1;best=-float('inf')
    if last.exists():
        state=restore(args,model,last,optimizer);start=state['epoch']+1;best=state['score']
    else:smoke(args,model,criterion)
    dataset=training_dataset(args);labels=RelationLabels(**args.config['source_training']['labels'])
    for epoch in range(start,EPOCHS+1):
        dataset.set_epoch(epoch);schedule(optimizer,epoch-1,EPOCHS)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=args.num_workers,
                          generator=torch.Generator().manual_seed(args.seed+epoch*1000003),pin_memory=args.device.type=='cuda')
        model.train();order=hashlib.sha256();images=hashlib.sha256();sums=Counter();first=None
        for index,batch in enumerate(loader):
            for case in batch['case_name']:order.update(str(case).encode()+b'\0')
            images.update(batch['image'].contiguous().numpy().tobytes())
            torch.manual_seed(args.seed+epoch*1000003+index)
            batch={k:(v.to(args.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True);result=model(batch['image'])
            loss,terms=joint_loss(model,criterion,result,batch,labels,args.config['lambda_relation'],args.seed+epoch*100000+index,
                                  args.config['source_training']['source_losses'])
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if first is None:first=norms(model)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(grad)):raise RuntimeError('Nonfinite gradient')
            optimizer.step();sums['batches']+=1;sums['loss']+=float(loss.detach())
            for key in ('seg_p0','seg_p1','seg_b','h0_aux','cont','relation'):sums[key]+=float(terms[key].detach())
            sums['transport_abs_mean']+=float(result['transport_delta'].detach().abs().mean())
            sums['proposal_abs_mean']+=float(result['proposal_delta'].detach().abs().mean())
            sums['P1_P0_abs_mean']+=float((result['p1']-result['p0']).detach().abs().mean())
            if index%25==0:print(f'[TRAIN] epoch={epoch}/{EPOCHS} batch={index+1}/{len(loader)} loss={float(loss.detach()):.6f}',flush=True)
        record={'epoch':epoch,'case_order_sha256':order.hexdigest(),'augmented_images_sha256':images.hexdigest(),
                'losses':{k:v/sums['batches'] for k,v in sums.items() if k!='batches'},'gradients':first}
        if epoch==1 and any(record['losses'][key]==0 for key in ('transport_abs_mean','proposal_abs_mean','P1_P0_abs_mean')):
            raise RuntimeError('Epoch 1 repair activity is numerically zero: '+str(record['losses']))
        control=json.loads((args.pair_dir/'A'/f'epoch_{epoch:03d}.json').read_text())
        for key in ('case_order_sha256','augmented_images_sha256'):
            if record[key]!=control[key]:raise ValueError(f'A/B input differs at epoch {epoch}: {key}')
        record['A_input_hashes_match']=True
        if epoch%5==0:
            cases,regions=val_cases(args,model);report=point_metrics(cases,'p1',.6,args.manifest)
            a_epoch=args.pair_dir/'A'/f'val_epoch_{epoch:03d}.json'
            val={'selection_p1':report,'p0':candidate_metrics(cases,'current_p0',.45),
                 'b':candidate_metrics(cases,'b',.45),'p1':candidate_metrics(cases,'p1',.45),
                 'same_threshold_repair':activity(cases),'candidate_regions':regions,
                 'A_same_epoch_saved_report':json.loads(a_epoch.read_text()) if a_epoch.is_file() else None}
            atomic_json(args.output_dir/f'val_epoch_{epoch:03d}.json',val)
            record['val_score_at_0_60']=report['selection_score']
            if report['selection_score']>best:best=report['selection_score'];save(args,model,None,args.output_dir/'best.pth',epoch,best)
            print(f'[VAL] epoch={epoch} score={report["selection_score"]:.6f}',flush=True)
        atomic_json(args.output_dir/f'epoch_{epoch:03d}.json',record);save(args,model,optimizer,last,epoch,best,record)
    atomic_json(args.output_dir/'training_complete.json',{'epochs':EPOCHS,'best_score':best})


def evaluate(args,model):
    restore(args,model,args.output_dir/'best.pth')
    cases,regions=val_cases(args,model,True)
    curve=[];p0_curve=[]
    for threshold in np.round(np.arange(.35,.751,.01),2):
        curve.append(point_metrics(cases,'p1',float(threshold),args.manifest))
        p0_curve.append(point_metrics(cases,'current_p0',float(threshold),args.manifest))
        print(f'[SWEEP] P1={threshold:.2f}',flush=True)
    best=max(curve,key=lambda row:row['selection_score']);t=float(best['threshold'])
    out=args.output_dir/'eval';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'threshold_curve.json',curve)
    atomic_json(out/'p0_threshold_curve.json',p0_curve)
    atomic_json(out/'val_selection.json',{'threshold':t,'report':best,'selected_on':'val',
        'checkpoint_sha256':digest(args.output_dir/'best.pth'),'reference_sha256':args.config['reference_sha256'],
        'selection_rule':args.config['selection_rule']})
    candidate_curve=[candidate_metrics(cases,'b',float(x)) for x in (.1,.2,.3,.4,.45,.5,.6,.7,.8,.9)]
    atomic_json(out/'candidate_threshold_curve.json',candidate_curve)
    atomic_json(out/'candidate_regions.json',regions)
    a_selection=json.loads((args.pair_dir/'A/eval/val_selection.json').read_text())
    for case in cases:
        path=args.pair_dir/'A/eval/val'/(case['case']+'.npz')
        with np.load(path) as a:
            if not np.array_equal(a['gt'].astype(bool),case['gt']) or not np.array_equal(a['p0'],case['p0']):
                raise ValueError('A/B reference mismatch: '+case['case'])
            case['A']=a['p1'].copy()
    a_threshold=float(a_selection['threshold'])
    target=sum(int(((case['A']>=a_threshold)&case['gt']).sum()) for case in cases)
    match,err=exact_recall_threshold(cases,'p1',target)
    repair_cases=[dict(case,p0=case['current_p0']) for case in cases]
    reports={'A_own_val':point_metrics(cases,'A',a_threshold,args.manifest),
             'B_P1_own_val':best,
             'B_P0_own_val':max(p0_curve,key=lambda row:row['selection_score']),
             'B_P1_matched_A_recall':dict(point_metrics(cases,'p1',match,args.manifest),matching_error=int(err)),
             'B_P0_same_threshold':point_metrics(repair_cases,'current_p0',t,args.manifest),
             'B_P1_same_threshold':point_metrics(repair_cases,'p1',t,args.manifest)}
    rows=[]
    for name,report in reports.items():
        row={'workpoint':name,'threshold':report['threshold']}
        row.update({key:report['metrics'][key] for key in ('iou','precision','recall','cldice','apls_bidirectional_approx','short_pred_components')})
        row.update(closed_fixed_gaps=report['closed'],bridge_proxy=report['bridge_proxy'],matching_error=report.get('matching_error',0))
        rows.append(row)
    write_csv(out/'comparison.csv',rows)
    gap_rows=[]
    for case in cases:
        name=case['case']
        p0_mask=case['current_p0']>=t;p1_mask=case['p1']>=t
        p0_labels=component_data(p0_mask)[1];p1_labels=component_data(p1_mask)[1]
        for gap in args.manifest['images'][name]['gaps']:
            g=gap['id'];before=gap_state(p0_mask,p0_labels,gap,case['gt'].shape)
            after=gap_state(p1_mask,p1_labels,gap,case['gt'].shape)
            gap_rows.append({'case':name,'gap_id':g,'length_bin':gap.get('length_bin','unknown'),
                'P0_coverage':before['coverage'],'P1_coverage':after['coverage'],
                'P0_local_closed':before['local_closed_proxy'],'P1_local_closed':after['local_closed_proxy'],
                'newly_closed':after['local_closed_proxy'] and not before['local_closed_proxy'],
                'lost':before['local_closed_proxy'] and not after['local_closed_proxy']})
    write_csv(out/'fixed_gap_changes.csv',gap_rows)
    atomic_json(out/'local_gap_summary.json',{'threshold':t,'gaps':len(gap_rows),
        'newly_closed':sum(r['newly_closed'] for r in gap_rows),'lost':sum(r['lost'] for r in gap_rows),
        'definition':'80% GT gap pixel coverage plus connected anchor groups inside local gap bounding window; proxy'})
    atomic_json(out/'same_threshold_changes.json',activity(cases,t))
    atomic_json(out/'comparison_reports.json',reports)
    lines=['# A vs joint B: validation only','','Original H0 initialization, input stream and 20-epoch budget match A.',
           'B repair is pretrained; this compares the complete training strategies, not an isolated architecture ablation.',
           'APLS, gap closure and bridge counts are proxy metrics. Table gap counts use the existing global-anchor proxy.',
           'The same-threshold repair rows use current B P0 as bridge baseline.', '',
           '| Workpoint | Threshold | IoU | Precision | Recall | clDice | Bidirectional APLS | Closed | Bridge proxy | Short components |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {workpoint} | {threshold:.4f} | {iou:.4f} | {precision:.4f} | {recall:.4f} | {cldice:.4f} | {apls_bidirectional_approx:.4f} | {closed_fixed_gaps} | {bridge_proxy} | {short_pred_components:.2f} |'.format(**row))
    lines+=['','See local_gap_summary.json, fixed_gap_changes.csv and same_threshold_changes.json for B P0 to P1 at the same threshold.',
            'Candidate region distributions are stored per image in candidate_regions.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[EVAL DONE] '+str(out),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('all','smoke','train','val'))
    parser.add_argument('--pair_dir',type=Path,required=True)
    parser.add_argument('--repair_checkpoint',type=Path,required=True)
    parser.add_argument('--run_name',default='B_joint_direct_20e')
    parser.add_argument('--device',default='cuda')
    opt=parser.parse_args()
    if Path(opt.run_name).name!=opt.run_name or not opt.run_name.startswith('B_'):parser.error('run_name must be a single B_-prefixed name')
    if not opt.repair_checkpoint.is_absolute():parser.error('repair_checkpoint must be absolute')
    args,model,criterion,repair_lr=configure(opt.pair_dir,opt.repair_checkpoint,opt.run_name,torch.device(opt.device))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    marker=args.output_dir/'experiment.json'
    if marker.is_file() and json.loads(marker.read_text())!=args.config:raise ValueError('Existing experiment settings differ')
    atomic_json(marker,args.config)
    optimizer=make_optimizer(model,args.h0_lr,repair_lr)
    atomic_json(args.output_dir/'parameter_groups.json',{g['name']:{'lr':g['initial_lr'],'parameters':sum(p.numel() for p in g['params'])} for g in optimizer.param_groups})
    atomic_json(args.output_dir/'initialization_check.json',args.config['direct_joint'])
    if opt.command=='smoke':smoke(args,model,criterion);return
    if opt.command in ('all','train'):train(args,model,criterion,optimizer)
    if opt.command in ('all','val'):evaluate(args,model)
    print('[DONE] '+str(args.output_dir),flush=True)


if __name__=='__main__':main()
