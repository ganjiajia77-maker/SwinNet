"""Three-step optimistic action reachability from one saved full H0+B model.

The max update deliberately ignores learned gates and is only an upper-bound
diagnostic for the fixed P0/B candidates. It is never a deployable prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.audit_full_h0_saved_predictions import (component_data, digest, gap_state,
                                                         read_json, write_csv, write_json)
from mass_protocol import blend_weight, sliding_positions
from networks.full_h0_joint import VERSION, FullH0Joint, build_h0
from networks.reliable_directional_repair import compose_surface_logits
from run_q3_r3_mass import full_dataset
from run_slim_end_to_end_mass import seed_all


def optimistic_sequence(p0, proposal, steps=3):
    """Simultaneous 8-neighbour max, plus fixed proposal and fixed P0."""
    if p0.shape!=proposal.shape or p0.ndim!=4 or p0.shape[1]!=1:
        raise ValueError('Expected matching one-channel work-grid P0 and B')
    if not bool(torch.isfinite(p0).all() and torch.isfinite(proposal).all()):
        raise ValueError('Nonfinite P0 or B')
    if bool(((p0<0)|(p0>1)|(proposal<0)|(proposal>1)).any()):
        raise ValueError('P0 and B must be probabilities')
    states=[p0]
    for _ in range(steps):
        neighbourhood=F.max_pool2d(states[-1],kernel_size=3,stride=1,padding=1)
        states.append(torch.maximum(torch.maximum(neighbourhood,proposal),p0))
    return states


def state_section(state, prefix):
    result={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    if not result: raise ValueError('Missing checkpoint section '+prefix)
    return result


def rebuild_source(state):
    """Extract all constructor weights from B itself, including Q3 and H0."""
    source={'modules':{name:state_section(state,'original.'+name+'.') for name in
                       ('visual','reliable_head','relations','proposal_head','accept_head','dynamics')},
            'gate_relation':state_section(state,'original.gate_relation.'),
            'selector':state_section(state,'selector.')}
    relations=source['modules']['relations']
    for suffix in ('weight','bias'):
        scalar='direction_head.scalar.'+suffix; directions='direction_head.directions.'+suffix
        if scalar not in relations or directions not in relations:
            raise ValueError('Checkpoint lacks both trained direction-head parts')
        relations['direction_head.'+suffix]=torch.cat((relations.pop(scalar),relations.pop(directions)),0)
    q={'q_features':state_section(state,'original.upstream.q_features.'),
       'q_head':state_section(state,'original.upstream.q_head.')}
    return source,q


def rebuild_model(checkpoint,device,size=512):
    if checkpoint.get('version')!=VERSION or checkpoint.get('config',{}).get('variant')!='B':
        raise ValueError('Expected the saved complete H0+source-selector B checkpoint')
    state=checkpoint['model'];source,q=rebuild_source(state)
    base,_=build_h0(checkpoint['config']['h0_args'],size=size)
    model=FullH0Joint(base,q,source)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def verify_recorded_code(config):
    checked={}
    for label,source in (('B',config),('selector_source',config.get('source_training') or {})):
        records=source.get('code_sha256',{})
        for name,expected in records.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or digest(path)!=expected:
                raise ValueError(f'Training code hash mismatch or missing file: {name}')
        checked[label]=len(records)
    if not checked['B']:
        raise ValueError('B checkpoint lacks original code fingerprints')
    return checked


def tile_gap(gap,shape,top,left,tile):
    """Only include gaps and all frozen anchors fully present in this tile."""
    arrays=[np.asarray(gap['pixels'],dtype=np.int64)] + [np.asarray(a,dtype=np.int64) for a in gap['anchors']]
    local=[]
    for values in arrays:
        if not values.size: return None
        ys,xs=np.divmod(values,shape[1])
        if not bool(((ys>=top)&(ys<top+tile)&(xs>=left)&(xs<left+tile)).all()): return None
        local.append((ys-top)*tile+(xs-left))
    return {'id':gap['id'],'pixels':local[0],'anchors':local[1:]}


def grid_gap(gap,tile=512,grid=128):
    """Map pixel centres to nearest quarter-grid cell and flag collisions."""
    if tile%grid: raise ValueError('Work grid is not an integer scale of its tile')
    factor=tile//grid
    def mapped(indices):
        y,x=np.divmod(np.asarray(indices,dtype=np.int64),tile)
        return np.unique((y//factor)*grid+(x//factor))
    pixels=mapped(gap['pixels']); anchors=[mapped(value) for value in gap['anchors']]
    reasons=[]
    if len(pixels)<2: reasons.append('fewer_than_2_unique_gap_cells')
    if any(np.intersect1d(pixels,group).size for group in anchors): reasons.append('gap_anchor_cell_overlap')
    if any(np.intersect1d(a,b).size for index,a in enumerate(anchors) for b in anchors[index+1:]):
        reasons.append('different_anchors_same_cell')
    return {'id':gap['id'],'pixels':pixels,'anchors':anchors},reasons


def work_gap_state(probability, labels, gap, threshold):
    prediction=probability>=threshold
    pixels=np.asarray(gap['pixels'],dtype=np.int64)
    sets=[set(np.unique(labels.reshape(-1)[group]))-{0} for group in gap['anchors']]
    connected=bool(sets and set.intersection(*sets))
    coverage=float(prediction.reshape(-1)[pixels].mean())
    values=probability.reshape(-1)[pixels]
    # 2 work cells correspond to the 8 native pixels used by full local ROI.
    points=np.concatenate([pixels]+gap['anchors'])
    ys,xs=np.divmod(points,probability.shape[1])
    y0=max(int(ys.min())-2,0);y1=min(int(ys.max())+3,probability.shape[0])
    x0=max(int(xs.min())-2,0);x1=min(int(xs.max())+3,probability.shape[1])
    _,local,_,_=component_data(prediction[y0:y1,x0:x1])
    local_sets=[]
    for group in gap['anchors']:
        gy,gx=np.divmod(group,probability.shape[1])
        local_sets.append(set(np.unique(local[gy-y0,gx-x0]))-{0})
    local_connected=bool(local_sets and set.intersection(*local_sets))
    return {'coverage':coverage,'min_probability':float(values.min()),'mean_probability':float(values.mean()),
            'connected_inside_tile':connected,'closed_proxy':bool(coverage>=.8 and connected),
            'local_connected':local_connected,'local_closed_proxy':bool(coverage>=.8 and local_connected)}


def tile_gap_diagnostics(result,optimistic,full_upper,entry,shape,top,left,thresholds):
    """Compare work and composed full pixels within the same tile and anchors."""
    present=[]
    for gap in entry['gaps']:
        local=tile_gap(gap,shape,top,left,result['z0'].shape[-1])
        if local is not None: present.append((gap,local))
    if not present: return []
    p0=result['u0'][0,0].detach().cpu().numpy()
    candidate=result['b'][0,0].detach().cpu().numpy()
    actual=result['u3'][0,0].detach().cpu().numpy()
    work=[value[0,0].detach().cpu().numpy() for value in optimistic]
    upper_full=full_upper.sigmoid()[0,0].detach().cpu().numpy()
    actual_full=result['p1'][0,0].detach().cpu().numpy()
    work_shape=p0.shape;tile=result['z0'].shape[-1]
    if work_shape[0]!=work_shape[1] or work_shape[0]*4!=tile:
        raise ValueError('Expected the actual 1/4-scale work grid')
    rows=[]
    for threshold in thresholds:
        work_labels=[component_data(value>=threshold)[1] for value in work]
        actual_work_labels=component_data(actual>=threshold)[1]
        full_mask=upper_full>=threshold;full_labels=component_data(full_mask)[1]
        actual_mask=actual_full>=threshold;actual_labels=component_data(actual_mask)[1]
        for original,local in present:
            mapped,reasons=grid_gap(local,tile,work_shape[0])
            work_states=[work_gap_state(value,labels,mapped,threshold)
                         for value,labels in zip(work,work_labels)]
            true_work=work_gap_state(actual,actual_work_labels,mapped,threshold)
            upper_tile=gap_state(full_mask,full_labels,local,(tile,tile))
            true_tile=gap_state(actual_mask,actual_labels,local,(tile,tile))
            row={'gap_id':original['id'],'threshold':float(threshold),'tile_top':top,'tile_left':left,
                 'work_gap_cells':len(mapped['pixels']),'work_mapping_ambiguous':bool(reasons),
                 'work_mapping_reason':','.join(reasons),
                 'candidate_B_gap_mean':float(candidate.reshape(-1)[mapped['pixels']].mean()),
                 'candidate_B_gap_above_threshold_fraction':float((candidate.reshape(-1)[mapped['pixels']]>=threshold).mean()),
                 'actual_work_U3_coverage':true_work['coverage'],
                 'actual_work_U3_closed':true_work['closed_proxy'],
                 'actual_work_U3_local_closed':true_work['local_closed_proxy'],
                 'actual_full_tile_coverage':true_tile['coverage'],
                 'actual_full_tile_closed_global':true_tile['global_closed_proxy'],
                 'upper_full_tile_coverage':upper_tile['coverage'],
                 'upper_full_tile_closed_global':upper_tile['global_closed_proxy'],
                 'upper_full_tile_closed_local':upper_tile['local_closed_proxy']}
            for index,status in enumerate(work_states):
                for key in ('coverage','min_probability','mean_probability','closed_proxy','local_closed_proxy'):
                    row[f'upper_work_U{index}_{key}']=status[key]
            row['work_closed_but_same_tile_full_not']=bool(not reasons and work_states[-1]['closed_proxy'] and
                                                            not upper_tile['global_closed_proxy'])
            row['work_local_closed_but_same_tile_full_local_not']=bool(not reasons and work_states[-1]['local_closed_proxy'] and
                                                                        not upper_tile['local_closed_proxy'])
            rows.append(row)
    return rows


@torch.inference_mode()
def stitch_case(model,image,entry,thresholds,device,tile=512,stride=256,progress=None):
    """Original logit stitch; each optimistic update stays within its tile."""
    if image.ndim!=4 or image.shape[:2]!=(1,3): raise ValueError('Expected one RGB image')
    h,w=image.shape[-2:]
    weight=blend_weight(tile,device)
    canvas=torch.zeros((3,1,h,w),device=device,dtype=torch.float32)
    divisor=torch.zeros((1,1,h,w),device=device,dtype=torch.float32)
    rows=[];violations=[0,0,0];max_violation=[0.,0.,0.]
    positions=[(top,left) for top in sliding_positions(h,tile,stride)
               for left in sliding_positions(w,tile,stride)]
    for index,(top,left) in enumerate(positions):
        result=model(image[:,:,top:top+tile,left:left+tile])
        if result['z0'].shape!=(1,1,tile,tile) or result['u0'].shape[-2:]!=(tile//4,tile//4):
            raise ValueError('Unexpected full-resolution or quarter-grid shape')
        optimistic=optimistic_sequence(result['u0'],result['b'],3)
        for step in range(1,4):
            difference=(result['states'][step]-optimistic[step]).clamp_min(0)
            violations[step-1]+=int((difference>1e-6).sum())
            max_violation[step-1]=max(max_violation[step-1],float(difference.max()))
        z_upper,_=compose_surface_logits(result['z0'],result['u0'],optimistic[-1])
        actual=torch.cat((result['z0'],result['z1'],z_upper),0)
        region=(slice(None),slice(None),slice(top,top+tile),slice(left,left+tile))
        canvas[region]+=actual.float()*weight
        divisor[:,:,top:top+tile,left:left+tile]+=weight
        rows.extend(tile_gap_diagnostics(result,optimistic,z_upper,entry,(h,w),top,left,thresholds))
        if progress and (index==0 or (index+1)%5==0 or index+1==len(positions)):
            progress(index+1,len(positions))
    if bool((divisor<=0).any()): raise RuntimeError('Uncovered native pixels')
    predictions=(canvas/divisor).sigmoid().cpu().numpy()[:,0]
    if not np.isfinite(predictions).all(): raise RuntimeError('Nonfinite stitched probabilities')
    full_violation=np.maximum(predictions[1]-predictions[2],0)
    if max(max_violation)>1e-4 or float(full_violation.max())>1e-4:
        raise RuntimeError('Optimistic max failed to upper-bound the actual fixed-candidate dynamics')
    return {'p0':predictions[0],'p1':predictions[1],'upper3':predictions[2]},rows,{
        'tiles':len(positions),'work_actual_above_upper_pixels_by_step':violations,
        'work_actual_above_upper_max_by_step':max_violation,
        'full_actual_above_upper_pixels':int((full_violation>1e-6).sum()),
        'full_actual_above_upper_max':float(full_violation.max())}


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    write_json(temporary,value);temporary.replace(path)


def atomic_npz(path,arrays):
    path=Path(path);temporary=path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream: np.savez_compressed(stream,**arrays)
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def signature(sample):
    value=hashlib.sha256()
    for key in ('image','mask'):
        tensor=sample[key].detach().cpu().contiguous().numpy()
        value.update(str(tensor.shape).encode());value.update(tensor.tobytes())
    return value.hexdigest()


def validate_saved_prediction(pair,b_name,name,gt,predictions):
    """Compare reconstructed real P0/P1 with the previously saved B maps."""
    path=pair/b_name/'eval/val'/(name+'.npz')
    if not path.is_file(): return {'available':False}
    with np.load(path) as saved:
        if not np.array_equal(saved['gt'].astype(bool),gt):
            raise ValueError('Saved B GT differs from current validation data: '+name)
        report={'available':True}
        for key,saved_key in (('p0','current_p0'),('p1','p1')):
            old=saved[saved_key]
            if old.shape!=predictions[key].shape: raise ValueError('Saved probability shape changed: '+name)
            difference=np.abs(predictions[key]-old)
            mean,max_value=float(difference.mean()),float(difference.max())
            report[key+'_mean_abs']=mean;report[key+'_max_abs']=max_value
            if mean>1e-4 or max_value>.005:
                raise ValueError(f'Reconstructed {key} differs from saved B predictions: {name}; mean={mean}, max={max_value}')
        return report


def full_gap_rows(name,entry,maps,thresholds,tile_rows):
    shape=tuple(entry['shape']);rows=[]
    for threshold in thresholds:
        masks={key:value>=threshold for key,value in maps.items()}
        labels={key:component_data(value)[1] for key,value in masks.items()}
        for gap in entry['gaps']:
            pixels=np.asarray(gap['pixels'],dtype=np.int64)
            row={'case':name,'gap_id':gap['id'],'threshold':float(threshold),
                 'missing_skeleton_pixels':len(pixels),'length_bin':gap.get('length_bin','unknown')}
            for key in ('p0','p1','upper3'):
                status=gap_state(masks[key],labels[key],gap,shape)
                row.update({key+'_'+field:value for field,value in status.items()})
                values=maps[key].reshape(-1)[pixels]
                row.update({key+'_gap_mean_probability':float(values.mean()),
                            key+'_gap_min_probability':float(values.min()),
                            key+'_gap_max_probability':float(values.max()),
                            key+'_gap_above_threshold_pixels':int((values>=threshold).sum()),
                            key+'_gap_all_pixels_above_threshold':bool((values>=threshold).all())})
            matching=[value for value in tile_rows if value['gap_id']==gap['id'] and value['threshold']==float(threshold)]
            unambiguous=[value for value in matching if not value['work_mapping_ambiguous']]
            row['fully_containing_tiles']=len(matching)
            row['unambiguous_work_tiles']=len(unambiguous)
            row['work_U3_closed_any_tile']=any(value['upper_work_U3_closed_proxy'] for value in unambiguous)
            row['work_U3_local_closed_any_tile']=any(value['upper_work_U3_local_closed_proxy'] for value in unambiguous)
            row['work_closed_full_tile_not_any']=any(value['work_closed_but_same_tile_full_not'] for value in unambiguous)
            row['work_local_closed_full_tile_local_not_any']=any(value['work_local_closed_but_same_tile_full_local_not'] for value in unambiguous)
            row['work_closed_but_stitched_full_not']=bool(row['work_U3_closed_any_tile'] and not row['upper3_global_closed_proxy'])
            row['work_U3_best_gap_coverage']=max((value['upper_work_U3_coverage'] for value in unambiguous),default=None)
            row['candidate_B_best_gap_coverage']=max((value['candidate_B_gap_above_threshold_fraction'] for value in unambiguous),default=None)
            row['upper_newly_closed_global_vs_actual']=bool(not row['p1_global_closed_proxy'] and row['upper3_global_closed_proxy'])
            row['upper_newly_closed_local_vs_actual']=bool(not row['p1_local_closed_proxy'] and row['upper3_local_closed_proxy'])
            row['upper_gap_coverage_below_80pct']=row['upper3_coverage']<.8
            row['upper_80pct_but_anchors_not_connected_global']=bool(row['upper3_coverage']>=.8 and not row['upper3_global_connected'])
            rows.append(row)
    return rows


def summarize(gaps,thresholds,case_reports,identity):
    verified=[case['reconstructed_vs_saved_B'] for case in case_reports if case['reconstructed_vs_saved_B']['available']]
    reconstruction={'processed_images':len(case_reports),'expected_images':identity['expected_images'],
                    'images_with_saved_B_comparison':len(verified),
                    'P0_max_abs_vs_saved_B':max((value['p0_max_abs'] for value in verified),default=None),
                    'P1_max_abs_vs_saved_B':max((value['p1_max_abs'] for value in verified),default=None),
                    'actual_above_upper_max':max((case['full_actual_above_upper_max'] for case in case_reports),default=None),
                    'actual_above_upper_pixels':sum(case['full_actual_above_upper_pixels'] for case in case_reports)}
    result={'identity':identity,'complete':False,'reconstruction_summary':reconstruction,
            'threshold_results':{},'case_reconstruction':case_reports,
            'limits':['This is a deliberately optimistic fixed-P0/fixed-B three-step max, not model performance or deployable output.',
                      'All neighbours are allowed regardless of learned direction, reliability, selector and update rates; B is freely accepted everywhere.',
                      'Native output follows the original per-tile logit composition and 512/256 weighted-logit stitching.',
                      'Work-grid tests use only tiles containing the entire frozen gap and all anchor pixels; gap/anchor collisions on the 1/4 grid are excluded.',
                      'A within-tile work/full mismatch is a diagnostic candidate, not proof that interpolation alone caused a failure.',
                      'Gap closure remains a proxy: >=80% reference-pixel coverage and connected anchors; 100% gap-pixel coverage is reported separately.',
                      'Global anchor connectivity may use a detour; local connectivity uses the gap/anchor bounding box plus 8 pixels and can miss curved paths.',
                      'Work-grid local connectivity uses a 2-cell margin (8 native pixels); discretization can still change coverage and topology.',
                      'An optimistic closure does not establish GT support or rule out false bridges; the max update can create large background errors.',
                      'A gap not fully contained in any 512 tile has no valid within-tile work-grid conclusion.']}
    for threshold in thresholds:
        rows=[row for row in gaps if row['threshold']==float(threshold)]
        counts={'fixed_gaps':len(rows)}
        for key in ('p0','p1','upper3'):
            counts[key+'_coverage_ge_80pct']=sum(row[key+'_coverage']>=.8 for row in rows)
            counts[key+'_all_gap_pixels_above_threshold']=sum(row[key+'_gap_all_pixels_above_threshold'] for row in rows)
            counts[key+'_global_closed_proxy']=sum(row[key+'_global_closed_proxy'] for row in rows)
            counts[key+'_local_closed_proxy']=sum(row[key+'_local_closed_proxy'] for row in rows)
        for key in ('upper_newly_closed_global_vs_actual','upper_newly_closed_local_vs_actual',
                    'upper_gap_coverage_below_80pct','upper_80pct_but_anchors_not_connected_global',
                    'work_U3_closed_any_tile','work_U3_local_closed_any_tile','work_closed_full_tile_not_any',
                    'work_local_closed_full_tile_local_not_any','work_closed_but_stitched_full_not'):
            counts[key]=sum(row[key] for row in rows)
        counts['gaps_with_fully_containing_tile']=sum(row['fully_containing_tiles']>0 for row in rows)
        counts['gaps_with_unambiguous_work_tile']=sum(row['unambiguous_work_tiles']>0 for row in rows)
        counts['gaps_without_unambiguous_work_tile']=sum(row['unambiguous_work_tiles']==0 for row in rows)
        counts['by_missing_skeleton_pixels']={}
        for label in ('3-7','8-15','16-31','32+'):
            subset=[row for row in rows if row['length_bin']==label]
            counts['by_missing_skeleton_pixels'][label]={'gaps':len(subset),
                'upper_below_80pct':sum(row['upper_gap_coverage_below_80pct'] for row in subset),
                'upper_newly_closed_local_vs_actual':sum(row['upper_newly_closed_local_vs_actual'] for row in subset),
                'upper_all_pixels_above_threshold':sum(row['upper3_gap_all_pixels_above_threshold'] for row in subset),
                'unambiguous_work_tiles':sum(row['unambiguous_work_tiles']>0 for row in subset)}
        result['threshold_results'][str(threshold)]=counts
    return result


def output_report(output,gaps,tile_rows,summary):
    output=Path(output)
    gap_fields=list(gaps[0]) if gaps else ['case','gap_id','threshold']
    tile_fields=list(tile_rows[0]) if tile_rows else ['case','gap_id','threshold','tile_top','tile_left']
    write_csv(output/'gap_upper_bound.csv',gaps,gap_fields)
    write_csv(output/'tile_gap_diagnostics.csv',tile_rows,tile_fields)
    atomic_json(output/'summary.json',summary)
    lines=['# Three-step optimistic max reachability','',
           'Fixed trained B P0/B candidates, three max steps, validation only; not a deployable model.',
           f'Complete: {summary["complete"]}; processed {summary["reconstruction_summary"]["processed_images"]}/{summary["identity"]["expected_images"]} images.',
           'Reconstruction checks: `'+json.dumps(summary['reconstruction_summary'],ensure_ascii=False)+'`','']
    for threshold,stats in summary['threshold_results'].items():
        lines += [f'## Threshold {threshold}','',
                  '| Measure | Gaps |','|---|---:|']
        for key,value in stats.items():
            if isinstance(value,dict):continue
            lines.append(f'| {key} | {value} |')
        lines += ['', 'Length groups (missing skeleton pixel count; not geometric distance):', '',
                  '| Group | Gaps | Upper below 80% coverage | Upper newly locally closed vs actual | Upper covers all gap pixels |',
                  '|---|---:|---:|---:|---:|']
        for name,value in stats['by_missing_skeleton_pixels'].items():
            lines.append(f'| {name} | {value["gaps"]} | {value["upper_below_80pct"]} | {value["upper_newly_closed_local_vs_actual"]} | {value["upper_all_pixels_above_threshold"]} |')
        lines.append('')
    lines += ['## Limits','']+['- '+value for value in summary['limits']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair_dir',type=Path,required=True)
    parser.add_argument('--b_run_name',default='B_transport_0p01_proposal_0p001')
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--root_path',type=Path,help='Override the recorded dataset location')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--expected_gaps',type=int,default=198)
    parser.add_argument('--expected_images',type=int,default=14)
    args=parser.parse_args()
    if Path(args.b_run_name).name!=args.b_run_name or not args.b_run_name.startswith('B'):
        parser.error('Invalid B run name')
    pair=args.pair_dir.resolve();run=pair/args.b_run_name
    checkpoint_path=run/'best.pth'
    if not checkpoint_path.is_file(): raise FileNotFoundError('B best checkpoint is required: '+str(checkpoint_path))
    manifest_path=pair/'reference/fixed_gaps_and_anchors.json'
    manifest=read_json(manifest_path);manifest_sha=digest(manifest_path)
    if manifest['count']!=args.expected_gaps or sum(len(value['gaps']) for value in manifest['images'].values())!=args.expected_gaps:
        raise ValueError(f'Expected {args.expected_gaps} fixed gaps, got {manifest.get("count")}')
    checkpoint_sha=digest(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=checkpoint['config']
    if config['reference_sha256']!=manifest_sha: raise ValueError('Checkpoint and fixed gap manifest differ')
    selection_path=run/'eval/val_selection.json'
    selection=read_json(selection_path) if selection_path.is_file() else None
    if selection and (selection['checkpoint_sha256']!=checkpoint_sha or selection['reference_sha256']!=manifest_sha):
        raise ValueError('B selection metadata differs from checkpoint or fixed reference')
    thresholds=[.45]
    if selection and abs(float(selection['threshold'])-.45)>1e-8:
        thresholds.append(float(selection['threshold']))
    if any(not 0.<value<1. for value in thresholds): raise ValueError('Invalid saved threshold')
    root=(args.root_path or Path(config['root_path'])).resolve()
    code_checks=verify_recorded_code(config)
    device=torch.device(args.device)
    identity={'version':'full_h0_optimistic_three_steps_v1','checkpoint_sha256':checkpoint_sha,
              'checkpoint_path':str(checkpoint_path),'checkpoint_epoch':checkpoint['epoch'],
              'fixed_reference_sha256':manifest_sha,'fixed_gaps':manifest['count'],
              'dataset_root':str(root),'dataset_root_override':args.root_path is not None,
              'thresholds':thresholds,'threshold_sources':['reference P0=.45']+
                    (['B validation selected'] if len(thresholds)>1 else []),
              'expected_images':args.expected_images,'tile':512,'stride':256,'steps':3,
              'diagnostic_code_sha256':digest(Path(__file__)),'recorded_code_files_checked':code_checks,
              'torch_version':str(torch.__version__),'cuda_version':str(torch.version.cuda),
              'numpy_version':str(np.__version__),'opencv_version':str(cv2.__version__),'device_type':device.type,
              'checkpoint_loading':'strict reconstruction entirely from B checkpoint; no separate H0/Q3 weights',
              'update':'max(current state, all eight neighbours, fixed candidate B, fixed work-grid P0)',
              'split':'val','GT_in_model_forward':False}
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    identity_path=output/'identity.json'
    if identity_path.is_file() and read_json(identity_path)!=identity:
        raise ValueError('Existing diagnostic output has a different checkpoint, reference, code or dataset')
    if not identity_path.is_file() and any(output.iterdir()):
        raise ValueError('Output directory is not empty and lacks a matching identity; choose a new directory')
    atomic_json(identity_path,identity)
    cache=output/'cases';cache.mkdir(exist_ok=True)
    seed_all(int(config['seed']))
    model=rebuild_model(checkpoint,device)
    del checkpoint
    dataset=full_dataset(root,'val')
    if len(dataset)!=args.expected_images or len(manifest['images'])!=args.expected_images:
        raise ValueError(f'Expected exactly {args.expected_images} fixed validation images')
    print(f'[LOADED] strict B checkpoint; {len(dataset)} val images; {manifest["count"]} fixed gaps; thresholds={thresholds}',flush=True)
    all_gaps=[];all_tiles=[];case_reports=[];seen=set();started=time.perf_counter()
    for index in range(len(dataset)):
        sample=dataset[index];name=str(sample['case_name'])
        if name in seen or name not in manifest['images']: raise ValueError('Validation case differs from fixed manifest: '+name)
        seen.add(name);entry=manifest['images'][name]
        gt=sample['mask'][0].detach().cpu().numpy()>.5
        if list(gt.shape)!=entry['shape'] or gt.shape!=(1500,1500):
            raise ValueError('Expected native 1500x1500 coordinates for '+name)
        ref_path=pair/'reference'/(name+'.npz');saved_path=run/'eval/val'/(name+'.npz')
        check_path=ref_path if ref_path.is_file() else saved_path
        if not check_path.is_file(): raise FileNotFoundError('Need saved GT to verify fixed reference: '+name)
        with np.load(check_path) as saved:
            if not np.array_equal(saved['gt'].astype(bool),gt): raise ValueError('Current GT differs from fixed reference: '+name)
        sample_sha=signature(sample);map_file=cache/(name+'.npz');row_file=cache/(name+'.json')
        cached=map_file.is_file() and row_file.is_file()
        if cached:
            row_data=read_json(row_file)
            if row_data['sample_sha256']!=sample_sha: raise ValueError('Cached image/GT changed: '+name)
            with np.load(map_file) as saved: maps={key:saved[key].copy() for key in ('p0','p1','upper3')}
            tile_rows=row_data['tile_rows'];stats=row_data['stats']
            print(f'[CACHE] {index+1}/{len(dataset)} {name}',flush=True)
        else:
            image=sample['image'][None].to(device)
            def progress(done,total):print(f'[UPPER] {index+1}/{len(dataset)} {name} tile={done}/{total}',flush=True)
            maps,tile_rows,stats=stitch_case(model,image,entry,thresholds,device,progress=progress)
            del image
            row_data={'sample_sha256':sample_sha,'tile_rows':tile_rows,'stats':stats}
            atomic_npz(map_file,maps);atomic_json(row_file,row_data)
        original=validate_saved_prediction(pair,args.b_run_name,name,gt,maps)
        case_reports.append({'case':name,'input_sha256':sample_sha,'reconstructed_vs_saved_B':original,**stats})
        for row in tile_rows: row['case']=name
        all_tiles.extend(tile_rows)
        all_gaps.extend(full_gap_rows(name,entry,maps,thresholds,tile_rows))
        output_report(output,all_gaps,all_tiles,summarize(all_gaps,thresholds,case_reports,identity))
    if seen!=set(manifest['images']): raise ValueError('Validation cases differ from fixed manifest')
    for threshold in thresholds:
        if sum(row['threshold']==threshold for row in all_gaps)!=args.expected_gaps:
            raise ValueError('Did not analyze every fixed gap')
    summary=summarize(all_gaps,thresholds,case_reports,identity)
    summary['elapsed_seconds']=time.perf_counter()-started
    summary['complete']=True
    output_report(output,all_gaps,all_tiles,summary)
    print('[UPPER DONE] '+str(output/'REPORT.md'),flush=True)


if __name__=='__main__':main()
