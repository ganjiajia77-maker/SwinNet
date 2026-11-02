"""Val-only A, old B, and proposal-unscaled B comparison."""
import argparse
import json
from pathlib import Path
import numpy as np

from diagnostics.v91_counterfactual import write_csv
from run_slim_end_to_end_mass import area_threshold,digest
from run_v9_gate_benefit import atomic_json
from run_v9_selector_transport import point_metrics
from selector_transport_reference import exact_recall_threshold


def compare(pair,new_run,old_run='B_transport_0p01_proposal_0p001'):
    pair=Path(pair);runs={'A':pair/'A','old_B':pair/old_run,'new_B':pair/new_run}
    manifest=json.loads((pair/'reference/fixed_gaps_and_anchors.json').read_text())
    reference_sha=digest(pair/'reference/fixed_gaps_and_anchors.json')
    selections={}
    for name,folder in runs.items():
        selection=json.loads((folder/'eval/val_selection.json').read_text())
        if selection['checkpoint_sha256']!=digest(folder/'best.pth') or selection['reference_sha256']!=reference_sha:
            raise ValueError('Checkpoint/reference mismatch in '+name)
        selections[name]=selection
    old_sha=digest(runs['old_B']/'best.pth')
    config=json.loads((runs['new_B']/'experiment.json').read_text())
    if config['proposal_adaptation']['source_B_sha256']!=old_sha:raise ValueError('Adapted run did not start from this old B')
    cases=[];files=sorted((runs['A']/'eval/val').glob('*.npz'))
    if len(files)!=14 or len(manifest['images'])!=14:raise ValueError('Expected all 14 validation images')
    for path in files:
        with np.load(path) as a,np.load(runs['old_B']/'eval/val'/path.name) as b,np.load(runs['new_B']/'eval/val'/path.name) as c:
            if not (np.array_equal(a['gt'],b['gt']) and np.array_equal(a['gt'],c['gt']) and
                    np.array_equal(a['p0'],b['p0']) and np.array_equal(a['p0'],c['p0'])):
                raise ValueError('GT or fixed initial H0 differs for '+path.name)
            cases.append({'case':path.stem,'gt':a['gt'].astype(bool),'p0':a['p0'].copy(),
                          'A':a['p1'].copy(),'old_B':b['p1'].copy(),'new_B':c['p1'].copy(),
                          'new_B_P0':c['current_p0'].copy(),'old_B_P0':b['current_p0'].copy()})
    anchor=float(selections['old_B']['threshold'])
    target_tp=sum(int(((c['old_B']>=anchor)&c['gt']).sum()) for c in cases)
    target_area=sum(int((c['old_B']>=anchor).sum()) for c in cases)
    reports={}
    for name in runs:
        reports[name+'_own_val']=point_metrics(cases,name,float(selections[name]['threshold']),manifest)
        reports[name+'_at_old_B_threshold']=point_metrics(cases,name,anchor,manifest)
        reports[name+'_at_0_45']=(reports[name+'_at_old_B_threshold'] if abs(anchor-.45)<1e-9
                                   else point_metrics(cases,name,.45,manifest))
        if name!='old_B':
            for label,func,target in (('matched_recall',exact_recall_threshold,target_tp),('matched_area',area_threshold,target_area)):
                threshold,error=func(cases,name,target)
                reports[name+'_'+label]=dict(point_metrics(cases,name,threshold,manifest),matching_error=int(error))
    reports['new_B_P0_at_old_B_threshold']=point_metrics(cases,'new_B_P0',anchor,manifest)
    reports['old_B_P0_at_old_B_threshold']=point_metrics(cases,'old_B_P0',anchor,manifest)
    rows=[]
    for name,result in reports.items():
        row={'name':name,'threshold':result['threshold']}
        row.update({k:result['metrics'][k] for k in ('iou','f1','precision','recall','cldice',
            'apls_gt_to_pred_approx','apls_pred_to_gt_approx','apls_bidirectional_approx','short_pred_components')})
        row.update(closed_fixed_gaps=result['closed'],reference_gaps=manifest['count'],bridge_proxy=result['bridge_proxy'],
                   matching_error=result.get('matching_error',0))
        rows.append(row)
    pixel_rows=[]
    for label,before_name,after_name,before_threshold,after_threshold in (
            ('old_B_to_new_same_threshold','old_B','new_B',anchor,anchor),
            ('old_B_to_new_matched_recall','old_B','new_B',anchor,reports['new_B_matched_recall']['threshold']),
            ('old_B_to_new_matched_area','old_B','new_B',anchor,reports['new_B_matched_area']['threshold']),
            ('new_P0_to_P1_own_threshold','new_B_P0','new_B',selections['new_B']['threshold'],selections['new_B']['threshold'])):
        changes={'comparison':label,'before_threshold':float(before_threshold),'after_threshold':float(after_threshold),
                 'added_tp':0,'lost_tp':0,'added_fp':0,'removed_fp':0,'absolute_probability_sum':0.,'pixels':0}
        for case in cases:
            before=case[before_name]>=before_threshold;after=case[after_name]>=after_threshold;gt=case['gt']
            changes['added_tp']+=int((after&~before&gt).sum());changes['lost_tp']+=int((before&~after&gt).sum())
            changes['added_fp']+=int((after&~before&~gt).sum());changes['removed_fp']+=int((before&~after&~gt).sum())
            changes['absolute_probability_sum']+=float(np.abs(case[after_name]-case[before_name]).sum(dtype=np.float64))
            changes['pixels']+=gt.size
        changes['mean_abs_probability_change']=changes['absolute_probability_sum']/max(changes['pixels'],1)
        pixel_rows.append(changes)
    old_closed=reports['old_B_at_old_B_threshold']['closed_ids']
    new_closed=reports['new_B_at_old_B_threshold']['closed_ids']
    matched_closed=reports['new_B_matched_recall']['closed_ids']
    gap_rows=[]
    for case in cases:
        name=case['case']
        for gap in manifest['images'][name]['gaps']:
            g=gap['id'];old=g in old_closed[name];new=g in new_closed[name];matched=g in matched_closed[name]
            gap_rows.append({'case':name,'gap_id':g,'length_bin':gap.get('length_bin','unknown'),
                             'old_closed':old,'new_closed_same_threshold':new,'new_closed_matched_recall':matched,
                             'newly_closed_same_threshold':bool(new and not old),'lost_same_threshold':bool(old and not new),
                             'newly_closed_matched_recall':bool(matched and not old),'lost_matched_recall':bool(old and not matched)})
    folder=runs['new_B']/'comparison';folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/'comparison.csv',rows);write_csv(folder/'fixed_gap_changes.csv',gap_rows)
    write_csv(folder/'pixel_changes.csv',pixel_rows)
    length_rows=[]
    for name,report in reports.items():
        for length,values in report['gap_by_length'].items():length_rows.append({'work_point':name,'length_bin':length,**values})
    write_csv(folder/'fixed_gap_length_bins.csv',length_rows)
    atomic_json(folder/'reports.json',reports)
    lines=['# Full H0 / original B / adapted proposal comparison','',
           'Validation only. New B starts from old B best and trains 20 additional epochs; it is not a same-initialization or equal-total-training-budget ablation.',
           'The old B positive_support multiplier is removed only in the new proposal logit; A/rates formulas remain unchanged.',
           'APLS, fixed gap closure and bridge counts are proxies; no test set was used for model or threshold selection.',
           f'Common threshold and matching target: old B validation threshold {anchor}.','',
           '| Work point | Threshold | IoU | Precision | Recall | clDice | APLS bidirectional | Closed | Bridge proxy | Short components |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["name"]} | {r["threshold"]:.5f} | {r["iou"]:.5f} | {r["precision"]:.5f} | {r["recall"]:.5f} | {r["cldice"]:.5f} | {r["apls_bidirectional_approx"]:.5f} | {r["closed_fixed_gaps"]} | {r["bridge_proxy"]} | {r["short_pred_components"]:.2f} |')
    lines += ['', 'See pixel_changes.csv for added/lost TP and FP, and fixed_gap_changes.csv for each fixed gap.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[COMPARISON DONE] '+str(folder/'REPORT.md'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair_dir',type=Path,required=True)
    p.add_argument('--new_run',required=True);p.add_argument('--old_run',default='B_transport_0p01_proposal_0p001')
    a=p.parse_args()
    for name in (a.new_run,a.old_run):
        if Path(name).name!=name or not name.startswith('B'):p.error('Invalid run name')
    compare(a.pair_dir,a.new_run,a.old_run)
