"""CPU comparison using shared fixed references and exact val operating points."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_slim_end_to_end_mass import digest,area_threshold
from run_v9_selector_transport import point_metrics
from selector_transport_reference import exact_recall_threshold
from run_v9_gate_benefit import atomic_json
from diagnostics.v91_counterfactual import write_csv


def compare(pair):
    pair=Path(pair)
    if not all((pair/v/'eval/val_selection.json').is_file() for v in ('A','B')):
        print('[COMPARE WAIT] The other group has not finished validation yet. Run compare after both finish.',flush=True)
        return
    configs={v:json.loads((pair/v/'experiment.json').read_text()) for v in ('A','B')}
    for key in ('h0_sha256','epochs','seed','root_path','batch_size','num_workers','h0_lr','selection_rule','reference_sha256'):
        if configs['A'][key]!=configs['B'][key]: raise ValueError('A/B settings differ: '+key)
    for epoch in range(1,configs['A']['epochs']+1):
        records={v:json.loads((pair/v/f'epoch_{epoch:03d}.json').read_text()) for v in ('A','B')}
        for key in ('case_order_sha256','augmented_images_sha256'):
            if records['A'][key]!=records['B'][key]: raise ValueError(f'A/B input data differ at epoch {epoch}: {key}')
    cases=[]
    files=sorted((pair/'A/eval/val').glob('*.npz'))
    if len(files)!=14: raise ValueError('Expected 14 complete validation predictions')
    for path in files:
        with np.load(path) as a,np.load(pair/'B/eval/val'/path.name) as b:
            if not np.array_equal(a['gt'],b['gt']) or not np.array_equal(a['p0'],b['p0']): raise ValueError('Different GT/reference P0')
            cases.append({'case':path.stem,'gt':a['gt'].copy(),'p0':a['p0'].copy(),
                'A':a['p1'].copy(),'B':b['p1'].copy(),'B_P0':b['current_p0'].copy()})
    selection={v:json.loads((pair/v/'eval/val_selection.json').read_text()) for v in ('A','B')}
    for v in ('A','B'):
        if selection[v]['checkpoint_sha256']!=digest(pair/v/'best.pth'): raise ValueError('Saved predictions refer to different weights')
    manifest=json.loads((pair/'reference/fixed_gaps_and_anchors.json').read_text());reports={}
    threshold=selection['A']['threshold']
    target_tp=sum(int(((c['A']>=threshold)&c['gt']).sum()) for c in cases)
    target_area=sum(int((c['A']>=threshold).sum()) for c in cases)
    for v in ('A','B'):
        reports[v+'_own_val']=point_metrics(cases,v,selection[v]['threshold'],manifest)
        reports[v+'_common']=point_metrics(cases,v,threshold,manifest)
    for label,function,target in (('matched_recall',exact_recall_threshold,target_tp),('matched_area',area_threshold,target_area)):
        t,error=function(cases,'B',target)
        reports['B_'+label]=dict(point_metrics(cases,'B',t,manifest),matching_error=error)
    reports['B_P0_common']=point_metrics(cases,'B_P0',threshold,manifest)
    rows=[]
    for name,report in reports.items():
        rows.append({'name':name,'threshold':report['threshold'],**{k:report['metrics'][k] for k in
            ('iou','f1','precision','recall','cldice','apls_gt_to_pred_approx','apls_pred_to_gt_approx','apls_bidirectional_approx','short_pred_components')},
            'closed_fixed_gaps':report['closed'],'reference_gaps':manifest['count'],'bridge_proxy':report['bridge_proxy'],
            'matching_error':report.get('matching_error',0)})
    folder=pair/'comparison';folder.mkdir(exist_ok=True)
    write_csv(folder/'comparison.csv',rows)
    atomic_json(folder/'reports.json',reports)
    lines=['# Full H0 A/B validation comparison','',
        'A: original full H0 continuation. B: same full H0 + trained source-selector structure, jointly trained.',
        'Same initial H0, epoch budget, augmented RGB tensors and order verified for every epoch.',
        'B uses the previously trained repair/Q/selector; this is a whole-system comparison, not a single-loss ablation.',
        'No L_choice, L_G, path losses or appearance occlusion. Eight neighbours, three steps; no extra scalar G.',
        f'Fixed {manifest["count"]} initial FULL H0 reference gaps, shared across A/B; NOT the previous local-gap 206.',
        'APLS, closure and bridge counts are proxies. Matching thresholds use validation only.','',
        '| Model/work point | Threshold | IoU | Precision | Recall | clDice | Approx bidirectional APLS | Closed | Bridge proxy | Short fragments |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["name"]} | {r["threshold"]:.6f} | {r["iou"]:.5f} | {r["precision"]:.5f} | {r["recall"]:.5f} | {r["cldice"]:.5f} | {r["apls_bidirectional_approx"]:.5f} | {r["closed_fixed_gaps"]} | {r["bridge_proxy"]} | {r["short_pred_components"]:.2f} |')
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[COMPARE DONE] '+str(folder/'REPORT.md'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair_dir',type=Path,required=True)
    compare(parser.parse_args().pair_dir)
