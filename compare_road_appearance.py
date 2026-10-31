"""CPU comparison of paired clean-val predictions. No training/model loading."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_v9_selector_transport import point_metrics
from selector_transport_reference import exact_recall_threshold,reference_digest
from run_slim_end_to_end_mass import area_threshold
from diagnostics.v91_counterfactual import write_csv
from run_v9_gate_benefit import atomic_json


def compare(control,augmented,output):
    a=json.loads((control/'experiment.json').read_text());b=json.loads((augmented/'experiment.json').read_text())
    aa=dict(a);bb=dict(b);aa['appearance']=dict(aa['appearance']);bb['appearance']=dict(bb['appearance'])
    if aa['appearance'].pop('enabled') or not bb['appearance'].pop('enabled'):raise ValueError('Expected off/on pair')
    if aa!=bb:raise ValueError('Pair differs beyond augmentation switch: source, seed, losses, LR, budget, pipeline or code')
    if a['lambda_choice']!=0:raise ValueError('This comparison requires L_choice absent')
    for epoch in range(1,a['epochs']+1):
        off=json.loads((control/f'epoch_{epoch:03d}.json').read_text());on=json.loads((augmented/f'epoch_{epoch:03d}.json').read_text())
        for field in ('case_order_sha256','base_inputs_sha256'):
            if off[field]!=on[field]:raise ValueError(f'Base order/crop/geometry/color mismatch: epoch{epoch}, {field}')
    cases=[]
    off_files=sorted((control/'predictions_val').glob('*.npz'));on_by_case={}
    for file in (augmented/'predictions_val').glob('*.npz'):
        meta=json.loads(file.with_suffix('.json').read_text());on_by_case[meta['case']]=file
    for file in off_files:
        meta=json.loads(file.with_suffix('.json').read_text());name=meta['case']
        with np.load(file,allow_pickle=False) as off,np.load(on_by_case.pop(name),allow_pickle=False) as on:
            if not np.array_equal(off['gt'],on['gt']) or not np.array_equal(off['p0'],on['p0']):raise ValueError('Frozen clean P0/GT changed between augmentation variants')
            cases.append({'case':name,'gt':off['gt'].astype(bool),'p0':off['p0'].copy(),'control':off['p1'].copy(),'appearance':on['p1'].copy()})
    if len(cases)!=14 or on_by_case:raise ValueError('Val image sets differ')
    manifest=json.loads((control/'fixed_gaps_and_anchors.json').read_text())
    if reference_digest(cases)!=manifest['sha256']:raise ValueError('Clean frozen reference fingerprint differs')
    output.mkdir(parents=True,exist_ok=True)
    thresholds={'control':json.loads((control/'val_selection.json').read_text())['threshold'],
                'appearance':json.loads((augmented/'val_selection.json').read_text())['threshold']}
    common=a['common_threshold'];reports={}
    tp=sum(int(((case['control']>=thresholds['control'])&case['gt']).sum()) for case in cases)
    area=sum(int((case['control']>=thresholds['control']).sum()) for case in cases)
    for name in thresholds:
        reports[name+'_own_val']=point_metrics(cases,name,thresholds[name],manifest)
        reports[name+'_common']=point_metrics(cases,name,common,manifest)
        rt,re=exact_recall_threshold(cases,name,tp);at,ae=area_threshold(cases,name,area)
        reports[name+'_matched_recall']=dict(point_metrics(cases,name,rt,manifest),matching_error=re)
        reports[name+'_matched_area']=dict(point_metrics(cases,name,at,manifest),matching_error=ae)
    reports['P0_common']=point_metrics(cases,'p0',common,manifest)
    rows=[]
    for name,value in reports.items():
        rows.append({'name':name,'threshold':value['threshold'],**{key:value['metrics'][key] for key in ('iou','f1','precision','recall','cldice','apls_gt_to_pred_approx','apls_bidirectional_approx','short_pred_components')},
            'closed_fixed_gaps':value['closed'],'bridge_proxy':value['bridge_proxy'],'matching_error':value.get('matching_error')})
    write_csv(output/'comparison.csv',rows)
    atomic_json(output/'summary.json',{'reports':reports,'same_initial_checkpoint':True,'same_training_order_crops_geometry_color':True,
        'P0_clean_val_exact_equal':True,'BASE_Q3_frozen':True,'lambda_choice':0,'limits':'APLS, closure and bridge counts are raster proxies. Crop-first control is distinct from the old whole-image-augment-then-crop training.'})
    lines=['# Frozen S_selector appearance experiment','',
        'Same source checkpoint, architecture, optimizer groups/LR, epoch budget, data order, clean crops and base geometry/color hashes verified.',
        'BASE/Q3 remain frozen. Every augmented training RGB is forwarded through them; no clean-feature cache is used. Clean val P0 is exactly equal.',
        'L_choice absent. Augmentation is train-only. Candidates/anchors remain the original frozen206 reference.',
        '', '| Variant / point | IoU | Recall | clDice | Bidirectional approximate APLS | Closed | Bridge proxy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f'| {row["name"]} | {row["iou"]:.6f} | {row["recall"]:.6f} | {row["cldice"]:.6f} | {row["apls_bidirectional_approx"]:.6f} | {row["closed_fixed_gaps"]} | {row["bridge_proxy"]} |')
    lines+=['','Per-epoch augmentation requests/success/skips and dimensions/alpha/timing: augmentation_on/epoch_*.json and appearance_epoch_*.jsonl. Forced24 previews do not estimate training proportions.',
            'The frozen experiment can improve only repair behavior; it cannot train the base to handle occlusion. Treat this separately from slim end-to-end training.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[PAIRED COMPARISON] '+str(output/'REPORT.md'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--control_run',type=Path,required=True)
    parser.add_argument('--augmented_run',type=Path,required=True);parser.add_argument('--output_dir',type=Path,required=True)
    args=parser.parse_args();compare(args.control_run,args.augmented_run,args.output_dir)
