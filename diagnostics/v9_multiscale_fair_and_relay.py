"""Read-only val-selected operating points and fixed-checkpoint relay intervention."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native,expandable_segments:False'
os.environ['PYTORCH_ALLOC_CONF'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

import run_reliable_v9_mass as v9
import run_v9_multiscale as experiment
from diagnostics.audit_v9_multiscale import fixed_gaps, connected, parallel_background_bridges
from diagnostics.diagnose_v9_transport import matched_threshold
from diagnostics.v92_reachability import longest_false_run
from evaluation.road_comparison.road_metrics import prepare_target, image_metrics, summarize
from mass_protocol import blend_weight, sliding_positions
from networks.v9_multiscale_transport import install
from run_q3_r3_mass import full_dataset, PROTOCOL
from run_reliable_directional_mass import gap_lengths

MODES = ('local', 'multiscale', 'multiscale_fixed_sources')
FIELDS = ('u', 'm', 'available', 'g_step', 'transport_rate', 'alpha', 'beta', 'gamma',
          'transport_delta', 'proposal_delta', 'anchor_delta')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def models(args):
    source = args.source.resolve()
    payload = torch.load(source, map_location='cpu', weights_only=False)
    train = payload['training_config']
    setup = SimpleNamespace(source_checkpoint=source, root_path=args.data.resolve(),
        seed=train['seed'], batch_size=train['batch_size'], device=args.device)
    model, _ = experiment.original(setup)
    states, selections = {}, {}
    for mode in ('local', 'multiscale'):
        ckpt = args.run/mode/'best.pth'
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        config = state['multiscale_config']
        selection = load_json(ckpt.parent/'val_selection.json')
        if (selection.get('selected_on') != 'val' or selection['checkpoint_sha256'] != v9.digest(ckpt)
                or selection['config'] != config or config['source_sha256'] != v9.digest(source)
                or config['mode'] != mode or config['identity'] != payload['identity']):
            raise ValueError(f'{mode}: checkpoint/selection/upstream identity mismatch')
        states[mode] = state
        selections[mode] = float(selection['threshold'])
    a, b = (states[k]['multiscale_config'] for k in ('local', 'multiscale'))
    if {k:v for k,v in a.items() if k != 'mode'} != {k:v for k,v in b.items() if k != 'mode'}:
        raise ValueError('Paired experiments do not share training configuration')
    selections['multiscale_fixed_sources'] = selections['multiscale']
    return model, states, selections, payload


def set_mode(model, state, mode):
    which = 'local' if mode == 'local' else 'multiscale'
    install(model, which)
    for name in experiment.NAMES:
        getattr(model, name).load_state_dict(state['modules'][name], strict=True)
    model.eval(); model.dynamics.blend = 1.


@torch.inference_mode()
def infer(model, sample, device, frozen_sources=False, points=None):
    image = sample['image'][None].to(device); h, w = image.shape[-2:]
    canvas = torch.zeros((2, 1, h, w), device=device)
    denominator = torch.zeros((1, 1, h, w), device=device)
    weight = blend_weight(512, device); owner_weight = weight[0,0].cpu().numpy()
    if points is not None:
        points = np.asarray(points, dtype=np.int64).reshape(-1, 2)
        point_owner = np.full(len(points), -1., np.float32)
        trace = {key: np.full((3, len(points)), np.nan, np.float32) for key in FIELDS}
    for y in sliding_positions(h, 512, 256):
        for x in sliding_positions(w, 512, 256):
            result = model(image[:,:,y:y+512,x:x+512], capture_diagnostics=points is not None,
                           freeze_generated_sources=frozen_sources)
            canvas[:,:,y:y+512,x:x+512] += torch.cat((result['z0'],result['z1']),0).float()*weight
            denominator[:,:,y:y+512,x:x+512] += weight
            if points is None or not len(points):
                continue
            inside = ((points[:,0]>=y)&(points[:,0]<y+512)&(points[:,1]>=x)&(points[:,1]<x+512))
            indices = np.flatnonzero(inside)
            py, px = points[indices,0]-y, points[indices,1]-x
            keep = owner_weight[py,px] > point_owner[indices]
            indices, py, px = indices[keep], py[keep], px[keep]
            if not len(indices): continue
            point_owner[indices] = owner_weight[py,px]
            wh, ww = result['u0'].shape[-2:]
            wy, wx = py*wh//512, px*ww//512
            for step in range(3):
                for key in FIELDS:
                    value = result[f'u{step}'] if key == 'u' else (
                        result['source_available'][:,step] if key == 'available' else result[key][:,step])
                    trace[key][step,indices] = value[0,0,wy,wx].float().cpu().numpy()
    if not bool((denominator > 0).all()): raise RuntimeError('Uncovered native pixels')
    logits = (canvas/denominator).cpu().numpy()[:,0]
    probabilities = 1./(1.+np.exp(-np.clip(logits,-80.,80.)))
    if points is not None and (point_owner < 0).any(): raise RuntimeError('Unowned gap points')
    return probabilities, trace if points is not None else None


def prediction_pass(args, model, states, dataset, output):
    caches = []
    for i in range(len(dataset)):
        sample = dataset[i]; case = str(sample['case_name']); predictions = {}
        for mode in MODES:
            state = states['local' if mode == 'local' else 'multiscale']
            set_mode(model, state, mode)
            result, _ = infer(model, sample, args.device, mode == 'multiscale_fixed_sources')
            if predictions and not np.array_equal(predictions['p0'], result[0]):
                raise ValueError('Frozen base P0 differs across modes')
            predictions['p0'] = result[0]; predictions[mode] = result[1]
        gt = sample['mask'][0].numpy() > .5
        caches.append({'case':case,'gt':gt,**predictions})
        print(f'[INFER {args.split}] {i+1}/{len(dataset)} {case}',flush=True)
    return caches


def choose_points(caches, selections):
    local_t = selections['local']
    target_tp = sum(int(((c['local']>=local_t)&c['gt']).sum()) for c in caches)
    target_area = sum(int((c['local']>=local_t).sum()) for c in caches)
    recall_t, recall_error = matched_threshold(caches, 'multiscale', target_tp, 'tp', 10000)
    area_t, area_error = matched_threshold(caches, 'multiscale', target_area, 'area', 10000)
    return {'own_val': {'local':local_t,'multiscale':selections['multiscale'],
                        'multiscale_fixed_sources':selections['multiscale']},
        'matched_recall': {'local':local_t,'multiscale':recall_t,'multiscale_fixed_sources':recall_t},
        'matched_area': {'local':local_t,'multiscale':area_t,'multiscale_fixed_sources':area_t},
        'matching': {'selected_on':'val','reference':'local own-val threshold','bins':10000,
                     'target_tp':target_tp,'target_area':target_area,
                     'recall_tp_residual':recall_error,'area_pixels_residual':area_error}}


def assess(caches, points, p0_threshold, output):
    rows=[]; per_image=[]
    prepared_cases={case['case']:prepare_target(case['gt'],PROTOCOL) for case in caches}
    metric_cache={}
    for point in ('own_val','matched_recall','matched_area'):
        for mode in MODES:
            metrics=[]; bridge=parallel=0
            for case in caches:
                key=(case['case'],mode,points[point][mode])
                if key not in metric_cache:
                    gt=case['gt']; p0=case['p0']>=p0_threshold; pred=case[mode]>=points[point][mode]
                    prepared=prepared_cases[case['case']]
                    item={'case_id':case['case'],**image_metrics(pred,prepared,PROTOCOL)}
                    _, old=cv2.connectedComponents(p0.astype(np.uint8),connectivity=8)
                    bridges, adjacent=parallel_background_bridges(p0,pred,gt,old)
                    _, existing_bridge=gap_lengths(p0,pred,prepared['skeleton'],gt)
                    metric_cache[key]=(item,bridges,adjacent,existing_bridge)
                item,bridges,adjacent,existing_bridge=metric_cache[key]
                bridge+=bridges; parallel+=adjacent; metrics.append(item)
                per_image.append({'point':point,'mode':mode,'threshold':points[point][mode],
                    **item,'background_bridge_proxy':bridges,'parallel_bridge_proxy':adjacent,
                    'v9_background_bridge_proxy':existing_bridge})
            rows.append({'point':point,'mode':mode,'threshold':points[point][mode],
                **summarize(metrics),'background_bridge_proxy_total':bridge,
                'parallel_bridge_proxy_total':parallel,
                'v9_background_bridge_proxy_total':sum(r['v9_background_bridge_proxy'] for r in per_image if r['point']==point and r['mode']==mode)})
    experiment.write_csv(output/'fair_operating_points.csv',rows)
    experiment.write_csv(output/'fair_per_image.csv',per_image)
    return rows


def trace_gaps(args, model, state, dataset, caches, threshold, p0_threshold, output):
    gap_rows=[]; pixel_rows=[]; paired=[]
    for i, case in enumerate(caches):
        sample=dataset[i]; gt=case['gt']; prepared=prepare_target(gt,PROTOCOL)
        gaps,_=fixed_gaps(case['p0']>=p0_threshold,prepared['skeleton'],gt)
        gaps=[g for g in gaps if g['length']>=16]
        all_points=np.concatenate([g['points'] for g in gaps],axis=0) if gaps else np.zeros((0,2),np.int64)
        traces={}; outputs={}
        for mode in ('multiscale','multiscale_fixed_sources'):
            set_mode(model,state,mode)
            prediction, trace=infer(model,sample,args.device,mode=='multiscale_fixed_sources',all_points)
            if not np.allclose(prediction[1],case[mode],rtol=0,atol=1e-7):
                raise ValueError(f'{case["case"]}: diagnostic rerun changed {mode} prediction')
            traces[mode]=trace; outputs[mode]=prediction[1]>=threshold
        offset=0
        for gap in gaps:
            n=len(gap['points']); ys,xs=gap['points'].T
            truth={'case':case['case'],'gap_id':gap['id'],'coordinate_sha256':gap['coordinate_sha256'],
                   'length':n,'length_bin':gap['length_bin'],'eligible':gap['eligible'],
                   'simple_path':gap['order'] is not None}
            closures={mode:connected(gap,outputs[mode],gt) for mode in outputs}
            paired.append({**truth,'on_connected':closures['multiscale'],
                'fixed_sources_connected':closures['multiscale_fixed_sources']})
            for mode in outputs:
                data=traces[mode]; sl=slice(offset,offset+n); p=case[mode][ys,xs]
                row={**truth,'mode':mode,'threshold':threshold,'connected':closures[mode],
                    'unclosed_on':closures['multiscale'] is False,
                    'p0_mean':float(case['p0'][ys,xs].mean()),
                    'final_probability_mean':float(p.mean()),'final_probability_min':float(p.min()),
                    'fraction_below_threshold':float((p<threshold).mean()),
                    'longest_uncovered_simple_path':None}
                if gap['order'] is not None:
                    row['longest_uncovered_simple_path']=longest_false_run([
                        bool(outputs[mode][y,x]) for y,x in gap['order']])
                for step in range(3):
                    for key in FIELDS:
                        values=data[key][step,sl]
                        row[f'step{step+1}_{key}_mean']=float(values.mean())
                    row[f'step{step+1}_m_minus_u_mean']=float((data['m'][step,sl]-data['u'][step,sl]).mean())
                work_final=(data['u'][2,sl]+data['transport_delta'][2,sl]+data['proposal_delta'][2,sl]+data['anchor_delta'][2,sl])
                potential=np.maximum(data['m'][:,sl]-data['u'][:,sl],0.)
                delivered=np.maximum(data['transport_delta'][:,sl],0.)
                active=delivered>1e-4
                positive=float(delivered[active].mean()) if active.any() else 0.
                pull=np.maximum(-(data['proposal_delta'][:,sl]+data['anchor_delta'][:,sl]),0.)
                pullback=float(pull[active].mean()) if active.any() else 0.
                strong=potential>.02
                row['work_u3_mean']=float(work_final.mean())
                row['work_u3_above_threshold_fraction']=float((work_final>=threshold).mean())
                # Review flags, not exclusive categories or calibrated classifiers.
                row['review_low_source_availability']=bool(data['available'][:,sl].mean()<.1)
                row['review_high_m_small_transport']=bool(strong.any() and delivered[strong].mean()<.1*potential[strong].mean())
                row['review_other_updates_counter_transport']=bool(active.any() and pullback>.5*positive)
                row['review_work_high_native_weak']=bool(((work_final>=threshold)&(p<threshold)).mean()>=.2)
                gap_rows.append(row)
                for j,(y,x) in enumerate(gap['points']):
                    item={**truth,'mode':mode,'y':int(y),'x':int(x),'p0':float(case['p0'][y,x]),
                          'final_probability':float(case[mode][y,x]),'below_threshold':bool(case[mode][y,x]<threshold)}
                    for step in range(3):
                        for key in FIELDS: item[f'step{step+1}_{key}']=float(data[key][step,offset+j])
                    pixel_rows.append(item)
            offset+=n
        print(f'[GAP TRACE {args.split}] {i+1}/{len(dataset)} {case["case"]}: {len(gaps)} long gaps',flush=True)
    experiment.write_csv(output/'long_gap_components.csv',gap_rows)
    experiment.write_csv(output/'long_gap_pixels.csv',pixel_rows)
    experiment.write_csv(output/'relay_gap_connectivity.csv',paired)
    flags=('review_low_source_availability','review_high_m_small_transport',
           'review_other_updates_counter_transport','review_work_high_native_weak')
    review={}
    for mode in ('multiscale','multiscale_fixed_sources'):
        for group in ('16-31','32+'):
            subset=[row for row in gap_rows if row['mode']==mode and row['length_bin']==group
                    and row['eligible'] and row['unclosed_on']]
            review[mode+'/'+group]={'unclosed_on_eligible':len(subset),
                                   **{key:sum(row[key] for row in subset) for key in flags}}
    return {'long_gaps':len(paired),'eligible':sum(bool(x['eligible']) for x in paired),
        'on_to_fixed_sources_connected':sum(x['on_connected'] is False and x['fixed_sources_connected'] is True for x in paired),
        'on_to_fixed_sources_disconnected':sum(x['on_connected'] is True and x['fixed_sources_connected'] is False for x in paired),
        'review_flags_for_unclosed_eligible':review}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),default='val'); p.add_argument('--device',default='cuda')
    args=p.parse_args(); args.run=args.run.resolve(); args.data=args.data.resolve(); args.device=torch.device(args.device)
    args.output.mkdir(parents=True,exist_ok=False)
    model,states,selections,source=models(args)
    dataset=full_dataset(args.data,args.split)
    ids={mode:v9.digest(args.run/mode/'best.pth') for mode in ('local','multiscale')}
    source_sha=v9.digest(args.source)
    selection_path=args.run/'fair_operating_points_val.json'
    if args.split=='test':
        selection=load_json(selection_path)
        if selection['checkpoint_sha256']!=ids or selection['source_sha256']!=source_sha or selection['data_root']!=str(args.data):
            raise ValueError('Validation operating points do not match current checkpoints/data')
        points=selection['points']
    caches=prediction_pass(args,model,states,dataset,args.output)
    if args.split=='val':
        # Detect stale selections, changed data or inconsistent model loading.
        for mode in ('local','multiscale'):
            selection=load_json(args.run/mode/'val_selection.json')
            counts={'tp':sum(int(((c[mode]>=selections[mode])&c['gt']).sum()) for c in caches),
                    'fp':sum(int(((c[mode]>=selections[mode])&~c['gt']).sum()) for c in caches),
                    'fn':sum(int(((c[mode]<selections[mode])&c['gt']).sum()) for c in caches)}
            if any(counts[k]!=selection['report']['p1'][k] for k in counts):
                raise ValueError(f'{mode}: validation TP/FP/FN do not reproduce the saved sweep: {counts}')
    if args.split=='val':
        points=choose_points(caches,selections)
    rows=assess(caches,points,source['training_config']['reference_threshold'],args.output)
    trace=trace_gaps(args,model,states['multiscale'],dataset,caches,selections['multiscale'],
                     source['training_config']['reference_threshold'],args.output)
    old_pointer=args.run/('latest_audit_'+args.split+'.txt'); previous=None; previous_summary=None
    if old_pointer.is_file():
        old=Path(old_pointer.read_text().strip()); summary=load_json(old/'summary.json')
        if summary['source_sha256']!=source_sha: raise ValueError('Existing audit source differs')
        fingerprint=hashlib.sha256()
        for c in caches:
            for value in (c['case'].encode('utf-8')+b'\0',np.asarray(c['gt'].shape,dtype=np.int64).tobytes(),
                *(np.packbits(mask,axis=None).tobytes() for mask in (
                    c['p0']>=source['training_config']['reference_threshold'],c['gt'],
                    prepare_target(c['gt'],PROTOCOL)['skeleton']))):
                fingerprint.update(value)
        if summary['reference_set']['sha256']!=fingerprint.hexdigest():
            raise ValueError('Previous audit has different image ordering, P0, GT or skeleton')
        csvpath=old/'generated_source_reuse_and_bridges.csv'
        with csvpath.open(encoding='utf-8',newline='') as f: previous=list(csv.DictReader(f))
        # The older CSV has no saved per-arm checkpoint SHA; label its provenance
        # and do not merge it into matched-threshold results.
        if summary.get('own_val_thresholds',{}).get('local') != selections['local'] or summary.get('own_val_thresholds',{}).get('multiscale') != selections['multiscale']:
            raise ValueError('Existing audit uses different validation thresholds')
        for mode in ('local','multiscale'):
            cases=[row['case'] for row in previous if row['mode']==mode]
            if len(cases)!=len(caches) or set(cases)!={c['case'] for c in caches}:
                raise ValueError(f'Previous {mode} CSV case set is incomplete or duplicated')
        experiment.write_csv(args.output/'previous_fixed_threshold_reuse.csv',previous)
        keys=('step1_generated_fp_weight_mean','step2_generated_fp_weight_mean',
              'step3_generated_fp_weight_mean','same_generated_fp_source_used_twice_fraction',
              'background_bridge_proxy','parallel_background_bridge_proxy')
        previous_summary={'path':str(csvpath),'sha256':v9.digest(csvpath),'fixed_threshold':summary['same_threshold'],
            'means_by_mode':{mode:{key:float(np.mean([float(row[key]) for row in previous if row['mode']==mode]))
                for key in keys} for mode in ('local','multiscale')},
            'limitation':'Same fixed P0/GT fingerprint and thresholds checked; old file has no per-arm checkpoint SHA.'}
    else:
        print(f'[NOTE] No previous {args.split} audit pointer; old source-reuse CSV unavailable',flush=True)
    if source_sha!=v9.digest(args.source) or any(ids[m]!=v9.digest(args.run/m/'best.pth') for m in ids):
        raise RuntimeError('A checkpoint changed during diagnostic execution')
    experiment.write_json(args.output/'summary.json',{'split':args.split,'images':len(dataset),
        'checkpoint_sha256':ids,'source_sha256':source_sha,'points':points,'metrics':rows,
        'relay_gap_connectivity':trace,'previous_reuse_csv_loaded':previous is not None,'previous_fixed_threshold_reuse':previous_summary,
        'notes':['Matching is val only; local own threshold is the reference and exact residual counts are provided.',
        'Frozen source support uses R only; U, gate, proposal and anchor still update normally; same checkpoint, no retraining.',
        'Gap trace uses highest blend-weight owning window, while final probabilities are native stitched logits.',
        'Delta values are on the work grid and cannot simply be added to stitched probabilities.',
        'Review flags are heuristic: mean availability<0.1; among points/steps where M-U>0.02, mean positive transport<10% of M-U; among points/steps with positive transport>1e-4, other negative updates>50% of transport; or >=20% of the same native points have owning-window work U3 over threshold but stitched output under it. They are not causal diagnoses.',
        'Gap closure, background bridges and parallel bridges are proxies; APLS is the existing raster approximation.',
        'The old CSV is from its own fixed threshold; it lacks per-arm checkpoint hashes and cannot be interpreted as matched or proven to use identical arm weights.',
        'GT is used only for evaluation and tracing; no test-set thresholds are selected.']})
    if args.split=='val':
        # Publish test operating points only after all validation checks finish.
        experiment.write_json(selection_path,{'checkpoint_sha256':ids,'source_sha256':source_sha,
            'data_root':str(args.data),'val_images':len(dataset),'points':points})
    print(f'[DONE] {args.output}',flush=True)


if __name__=='__main__': main()
