"""Pair saved A/B validation maps, without model inference or training.

Reuse the original 198-gap reference and report the inherited global closure
criterion beside a local-ROI proxy.  All geometry categories are GT proxies.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from evaluation.road_comparison.road_metrics import skeletonize
from selector_transport_reference import exact_recall_threshold


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows([{key:(json.dumps(value,ensure_ascii=False) if isinstance(value,dict) else value)
                           for key,value in row.items()} for row in rows])


def load_inputs(pair, b_name, expected_gaps, expected_images):
    """Check available provenance and frozen reference without requiring weights."""
    pair=Path(pair)
    if Path(b_name).name != b_name or not b_name.startswith('B'):
        raise ValueError('Invalid B run directory')
    runs={'A':pair/'A', 'B':pair/b_name}
    cfg={key:read_json(path/'experiment.json') for key,path in runs.items()}
    for field in ('h0_sha256','epochs','seed','root_path','batch_size','num_workers',
                  'h0_lr','val_interval','selection_rule','reference_sha256'):
        if field not in cfg['A'] or field not in cfg['B'] or cfg['A'][field] != cfg['B'][field]:
            raise ValueError('A/B setting differs: '+field)
    reference_code_count=0
    for key,value in cfg['A'].get('code_sha256',{}).items():
        if key.startswith('reference_h0_599a410/') and cfg['B'].get('code_sha256',{}).get(key)!=value:
            raise ValueError('Original H0 code differs: '+key)
        if key.startswith('reference_h0_599a410/'): reference_code_count+=1
    for epoch in range(1,int(cfg['A']['epochs'])+1):
        records={key:read_json(path/f'epoch_{epoch:03d}.json') for key,path in runs.items()}
        for field in ('case_order_sha256','augmented_images_sha256'):
            if records['A'][field]!=records['B'][field]:
                raise ValueError(f'A/B input fingerprints differ: epoch {epoch}, {field}')
    selection={key:read_json(path/'eval/val_selection.json') for key,path in runs.items()}
    manifest_path=pair/'reference/fixed_gaps_and_anchors.json'
    manifest=read_json(manifest_path)
    manifest_sha=digest(manifest_path)
    if manifest_sha != cfg['A']['reference_sha256']:
        raise ValueError('Fixed reference manifest SHA does not match experiments')
    if manifest['count']!=expected_gaps or sum(len(row['gaps']) for row in manifest['images'].values())!=expected_gaps:
        raise ValueError(f'Expected {expected_gaps} fixed gaps but manifest reports {manifest.get("count")}')
    verification={'reference_H0_code_hashes_checked':reference_code_count,
                  'checkpoint_hash_status':{},'reference_images_checked':0}
    for key,path in runs.items():
        checkpoint=path/'best.pth'
        if checkpoint.is_file():
            if selection[key]['checkpoint_sha256']!=digest(checkpoint):
                raise ValueError(f'{key} selection metadata and best checkpoint differ')
            verification['checkpoint_hash_status'][key]='selection SHA matches existing best checkpoint'
        else:
            verification['checkpoint_hash_status'][key]='best checkpoint missing; selection SHA cannot be independently verified'
        if selection[key].get('reference_sha256')!=manifest_sha:
            raise ValueError(f'{key} selection refers to another fixed reference')
    files=sorted((runs['A']/'eval/val').glob('*.npz'))
    b_files=sorted((runs['B']/'eval/val').glob('*.npz'))
    if len(files)!=expected_images or {p.name for p in files}!={p.name for p in b_files}:
        raise ValueError(f'Expected {expected_images} matching saved A/B images')
    if {p.stem for p in files}!=set(manifest['images']):
        raise ValueError('Saved case names differ from the fixed manifest')
    cases=[]
    for path in files:
        with np.load(path) as a,np.load(runs['B']/'eval/val'/path.name) as b:
            for key in ('gt','p0','p1'):
                if key not in a or key not in b: raise ValueError(f'{path.name} lacks {key}')
            if 'current_p0' not in b: raise ValueError(f'{path.name} lacks trained B current_p0')
            if not np.array_equal(a['gt'],b['gt']) or not np.array_equal(a['p0'],b['p0']):
                raise ValueError(f'Different GT or frozen P0: {path.name}')
            ref_path=pair/'reference'/path.name
            if ref_path.is_file():
                with np.load(ref_path) as reference:
                    if not np.array_equal(reference['gt'],a['gt']) or not np.array_equal(reference['p0'],a['p0']):
                        raise ValueError(f'Saved predictions differ from frozen reference: {path.name}')
                verification['reference_images_checked']+=1
            gt=a['gt'].astype(bool)
            case={'case':path.stem,'gt':gt,'A':a['p1'].copy(),
                  'B_P0':b['current_p0'].copy(),'B_P1':b['p1'].copy()}
            if list(gt.shape)!=manifest['images'][path.stem]['shape']:
                raise ValueError(f'Fixed coordinates do not match image: {path.name}')
            for key in ('A','B_P0','B_P1'):
                p=case[key]
                if p.shape!=gt.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                    raise ValueError(f'{path.name}: {key} is not a finite probability map')
            cases.append(case)
    selection['_verification']=verification
    return cases,manifest,cfg,selection,manifest_sha


def component_data(mask):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)


def frozen_gap_mask(shape, entry):
    mask=np.zeros(shape,dtype=np.uint8)
    flat=mask.reshape(-1)
    for gap in entry['gaps']:
        indices=np.asarray(gap['pixels'],dtype=np.int64)
        if len(indices)==0 or indices.min()<0 or indices.max()>=flat.size:
            raise ValueError('Fixed gap pixels out of bounds')
        flat[indices]=1
    return cv2.dilate(mask,np.ones((5,5),np.uint8))>0


def prepare_gt(gt, entry, thin_width):
    skeleton=skeletonize(gt)
    gap=frozen_gap_mask(gt.shape,entry)
    skeleton_near=cv2.dilate(skeleton.astype(np.uint8),np.ones((3,3),np.uint8))>0
    inside=cv2.distanceTransform(gt.astype(np.uint8),cv2.DIST_L2,5)
    outside=cv2.distanceTransform((~gt).astype(np.uint8),cv2.DIST_L2,5)
    # Estimate full local width at the nearest GT skeleton pixel, rather than
    # confusing a wide road edge's small distance with a genuinely thin road.
    thin=np.zeros(gt.shape,dtype=bool)
    if skeleton.any():
        nearest=distance_transform_edt(~skeleton,return_distances=False,return_indices=True)
        thin=gt & ((2.*inside[nearest[0],nearest[1]])<=thin_width)
        del nearest
    regions={'fixed_gap_corridor_2px':gap,
             'GT_skeleton_near_outside_gap':~gap & gt & skeleton_near,
             'GT_road_edge_outside_gap':~gap & gt & ~skeleton_near & (inside<=2.),
             'GT_road_interior_outside_gap':~gap & gt & ~skeleton_near & (inside>2.),
             'GT_background_near_outside_gap':~gap & ~gt & (outside<=3.),
             'GT_background_far_outside_gap':~gap & ~gt & (outside>3.)}
    return {'thin':thin,'regions':regions,'gap_corridor':gap}


def scores(pred,gt):
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/max(tp+fp+fn,1),
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
            'f1':2*tp/max(2*tp+fp+fn,1)}


def new_short_components(before, after, gt, thin, short_area, case, scenario, transition):
    """Full after-mask components that are newly isolated or split from old ones."""
    old_n,old,old_stats,_=component_data(before)
    count,labels,stats,_=component_data(after)
    length=count
    overlap=np.bincount(labels[before].ravel(),minlength=length)
    gt_count=np.bincount(labels[gt].ravel(),minlength=length)
    thin_count=np.bincount(labels[thin].ravel(),minlength=length)
    both=before&after
    had_short_parent=np.zeros(count,dtype=bool)
    if both.any():
        pairs=np.unique(old[both].astype(np.int64)*count+labels[both].astype(np.int64))
        parents=pairs//count;children=pairs%count
        child_count=np.bincount(parents,minlength=old_n)
        split_child=np.zeros(count,dtype=bool)
        split_child[children[child_count[parents]>1]]=True
        np.logical_or.at(had_short_parent,children,old_stats[parents,cv2.CC_STAT_AREA]<short_area)
    else:
        split_child=np.zeros(count,dtype=bool)
    rows=[]; after_categories={}
    for index in range(1,count):
        x,y,w,h,area=map(int,stats[index])
        if area>=short_area:
            continue
        gt_pixels=int(gt_count[index]); thin_pixels=int(thin_count[index])
        if gt_pixels/area>=.5:
            category='mostly_GT_road_thin_proxy' if thin_pixels/max(gt_pixels,1)>=.5 else 'mostly_GT_road_other'
        elif gt_pixels/area<=.1:
            category='mostly_GT_background'
        else:
            category='mixed_GT_and_background'
        item=after_categories.setdefault(category,{'components':0,'area_pixels':0,'GT_road_pixels':0,'GT_background_pixels':0})
        item['components']+=1;item['area_pixels']+=area
        item['GT_road_pixels']+=gt_pixels;item['GT_background_pixels']+=area-gt_pixels
        if overlap[index]>0 and not split_child[index] and had_short_parent[index]:
            continue
        origin=('new_no_previous_overlap' if overlap[index]==0 else
                'split_descendant' if split_child[index] else 'became_short_after_shrink')
        rows.append({'scenario':scenario,'transition':transition,'case':case,'component_id':index,
                     'origin':origin,
                     'category':category,'area_pixels':area,'GT_road_pixels':gt_pixels,
                     'GT_background_pixels':area-gt_pixels,'GT_thin_road_pixels':thin_pixels,
                     'GT_road_fraction':gt_pixels/area,'x':x,'y':y,'width':w,'height':h})
    old_short=int((old_stats[1:,cv2.CC_STAT_AREA]<short_area).sum())
    new_short=int((stats[1:,cv2.CC_STAT_AREA]<short_area).sum())
    return rows,{'before_short_components':old_short,'after_short_components':new_short,
                 'short_component_net_change':new_short-old_short,'all_after_short_categories':after_categories}


def bridge_candidates(before,after,gt,gap_corridor,case,scenario,transition,strict,gaps=()):
    """Reproduce the existing GT-buffered bridge proxy, or a broader sensitivity check."""
    _,old,_,_=component_data(before)
    road_buffer=cv2.dilate(gt.astype(np.uint8),np.ones((7,7),np.uint8))>0
    addition=after&~before&~(road_buffer if strict else gt)
    count,labels,stats,_=component_data(addition)
    rows=[]
    for index in range(1,count):
        x,y,width,height,area=map(int,stats[index])
        if area<3: continue
        x0,x1=max(x-2,0),min(x+width+2,gt.shape[1])
        y0,y1=max(y-2,0),min(y+height+2,gt.shape[0])
        region=labels[y0:y1,x0:x1]==index
        near=cv2.dilate(region.astype(np.uint8),np.ones((5,5),np.uint8))>0
        old_ids=np.unique(old[y0:y1,x0:x1][near]);old_ids=old_ids[old_ids>0]
        if len(old_ids)<2: continue
        # The strict proxy excludes a 3px GT buffer, so associating it with
        # gap pixels within only 2px would be identically empty. 16px is a
        # deliberately labelled proximity proxy, not causal attribution.
        gx0,gx1=max(x-16,0),min(x+width+16,gt.shape[1])
        gy0,gy1=max(y-16,0),min(y+height+16,gt.shape[0])
        region16=labels[gy0:gy1,gx0:gx1]==index
        near16=cv2.dilate(region16.astype(np.uint8),np.ones((33,33),np.uint8))>0
        gap_ids=[]
        for gap in gaps:
            gy,gx=np.divmod(np.asarray(gap['pixels'],dtype=np.int64),gt.shape[1])
            inside=(gy>=gy0)&(gy<gy1)&(gx>=gx0)&(gx<gx1)
            if inside.any() and bool(near16[gy[inside]-gy0,gx[inside]-gx0].any()):gap_ids.append(str(gap['id']))
        rows.append({'scenario':scenario,'transition':transition,'case':case,
                     'definition':'strict_outside_GT_3px' if strict else 'broad_GT_background',
                     'component_id':index,'pixels':area,'previous_components_near':len(old_ids),
                     'near_gap_corridor_2px':bool((region&gap_corridor[y0:y1,x0:x1]).any()),
                     'gap_ids_within_16px':','.join(gap_ids),
                     'x':x,'y':y,'width':width,'height':height})
    return rows


def gap_state(pred, labels, gap, shape):
    """Official global anchor proxy plus local anchor connectivity diagnostic."""
    total=int(np.prod(shape)); pixels=np.asarray(gap['pixels'],dtype=np.int64)
    anchors=[np.asarray(group,dtype=np.int64) for group in gap['anchors']]
    if pixels.size==0 or pixels.min()<0 or pixels.max()>=total or not anchors:
        raise ValueError('Invalid frozen gap pixels or anchor groups')
    if any(a.size==0 or a.min()<0 or a.max()>=total for a in anchors):
        raise ValueError('Invalid frozen anchor coordinates')
    coverage=float(pred.reshape(-1)[pixels].mean())
    global_sets=[set(np.unique(labels.reshape(-1)[a]))-{0} for a in anchors]
    global_connected=bool(global_sets and set.intersection(*global_sets))
    points=np.concatenate([pixels]+anchors)
    ys,xs=np.divmod(points,shape[1])
    y0=max(int(ys.min())-8,0);y1=min(int(ys.max())+9,shape[0])
    x0=max(int(xs.min())-8,0);x1=min(int(xs.max())+9,shape[1])
    _,local,_,_=component_data(pred[y0:y1,x0:x1])
    local_sets=[]
    for anchors_flat in anchors:
        ay,ax=np.divmod(anchors_flat,shape[1])
        local_sets.append(set(np.unique(local[ay-y0,ax-x0]))-{0})
    local_connected=bool(local_sets and set.intersection(*local_sets))
    return {'coverage':coverage,'missing_anchor_groups':sum(not group for group in global_sets),
            'global_connected':global_connected,'global_closed_proxy':bool(coverage>=.8 and global_connected),
            'local_connected':local_connected,'local_closed_proxy':bool(coverage>=.8 and local_connected)}


def region_rows(before_prob,after_prob,gt,regions,case,scenario,transition,before_threshold,after_threshold):
    before=before_prob>=before_threshold;after=after_prob>=after_threshold
    delta=after_prob-before_prob
    result=[]
    for name,mask in regions.items():
        total=int(mask.sum())
        road=mask&gt;background=mask&~gt
        result.append({'scenario':scenario,'transition':transition,'case':case,'region':name,
                       'region_pixels':total,'probability_delta_mean':float(delta[mask].mean()) if total else None,
                       'probability_delta_abs_mean':float(np.abs(delta[mask]).mean()) if total else None,
                       'GT_road_pixels':int(road.sum()),'GT_background_pixels':int(background.sum()),
                       'GT_road_probability_delta_mean':float(delta[road].mean()) if road.any() else None,
                       'GT_background_probability_delta_mean':float(delta[background].mean()) if background.any() else None,
                       'added_tp':int((mask&after&~before&gt).sum()),'lost_tp':int((mask&before&~after&gt).sum()),
                       'added_fp':int((mask&after&~before&~gt).sum()),'removed_fp':int((mask&before&~after&~gt).sum())})
    return result


def addition_spatial_rows(before,after,gt,gap_corridor,after_labels,case,scenario,transition):
    """Assign every gross addition to gap, island, previous-edge, or extension."""
    added=after&~before
    overlap=np.bincount(after_labels[before].ravel(),minlength=int(after_labels.max())+1)
    attached=(after_labels>0)&(overlap[after_labels]>0)
    distance=cv2.distanceTransform((~before).astype(np.uint8),cv2.DIST_L2,5)
    remaining=added&~gap_corridor
    masks={'fixed_gap_corridor_2px':added&gap_corridor,
           'isolated_after_component':remaining&~attached,
           'previous_prediction_edge_2px':remaining&attached&(distance<=2.),
           'attached_extension_beyond_2px':remaining&attached&(distance>2.)}
    return [{'scenario':scenario,'transition':transition,'case':case,'category':name,
             'added_pixels':int(mask.sum()),'added_GT_road_pixels':int((mask&gt).sum()),
             'added_GT_background_pixels':int((mask&~gt).sum())}
            for name,mask in masks.items()]


def status_rows(case,entry,predictions,label_maps,scenario,thresholds):
    result=[]
    for gap in entry['gaps']:
        pixels=np.asarray(gap['pixels'],dtype=np.int64)
        ys,xs=np.divmod(pixels,case['gt'].shape[1])
        row={'scenario':scenario,'case':case['case'],'gap_id':gap['id'],
             'missing_skeleton_pixels':len(pixels),'length_bin':gap.get('length_bin','unknown'),
             'x_min':int(xs.min()),'y_min':int(ys.min()),'x_max':int(xs.max()),'y_max':int(ys.max()),
             'anchor_groups':len(gap['anchors'])}
        for key in ('A','B_P0','B_P1'):
            state=gap_state(predictions[key],label_maps[key],gap,case['gt'].shape)
            row[key+'_threshold']=thresholds[key]
            for name,value in state.items(): row[key+'_'+name]=value
            row[key+'_mean_probability']=float(case[key].reshape(-1)[pixels].mean())
        for label in ('global','local'):
            before=row['B_P0_'+label+'_closed_proxy'];after=row['B_P1_'+label+'_closed_proxy']
            row['repair_newly_closed_'+label]=not before and after
            row['repair_lost_closed_'+label]=before and not after
        result.append(row)
    return result


def save_visual(path,case,pred,thresholds,entry):
    """Downsampled GT/prediction overlays; no RGB inputs were saved in NPZ."""
    def color_mask(mask): return cv2.cvtColor(mask.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)
    def changes(before,after,gt):
        canvas=color_mask(gt)//3
        canvas[after&~before&gt]=(0,255,0)
        canvas[after&~before&~gt]=(0,0,255)
        canvas[before&~after&gt]=(0,255,255)
        canvas[before&~after&~gt]=(255,255,0)
        return canvas
    def label(image,text):
        small=cv2.resize(image,(750,750),interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(small,(0,0),(750,28),(0,0,0),-1)
        cv2.putText(small,text,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        return small
    a,b,c=pred['A'],pred['B_P0'],pred['B_P1'];gt=case['gt']
    panels=[label(color_mask(gt),'GT road'),label(color_mask(a),f'A threshold {thresholds["A"]:.4f}'),
            label(color_mask(b),f'B-P0 threshold {thresholds["B_P0"]:.4f}'),
            label(color_mask(c),f'B-P1 threshold {thresholds["B_P1"]:.4f}'),
            label(changes(a,b,gt),'B-P0 vs A: green +TP, red +FP, yellow lost TP, cyan removed FP'),
            label(changes(b,c,gt),'B-P1 vs B-P0: green +TP, red +FP, yellow lost TP, cyan removed FP')]
    h,w=gt.shape
    for gap in entry['gaps']:
        ys,xs=np.divmod(np.asarray(gap['pixels'],dtype=np.int64),w)
        x0=max(int(xs.min()*750/w)-3,0);x1=min(int(xs.max()*750/w)+3,749)
        y0=max(int(ys.min()*750/h)-3,29);y1=min(int(ys.max()*750/h)+3,749)
        if y1<y0: continue
        for panel in (panels[1],panels[2],panels[3]):
            cv2.rectangle(panel,(x0,y0),(x1,y1),(255,0,255),1)
            cv2.putText(panel,str(gap['id']),(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,0,255),1)
    grid=np.concatenate((np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)),0)
    if not cv2.imwrite(str(path),grid): raise OSError('Could not write '+str(path))


def operating_points(cases,selection):
    threshold_a=float(selection['A']['threshold'])
    reference_tp=sum(int(((case['A']>=threshold_a)&case['gt']).sum()) for case in cases)
    b0_threshold,b0_error=exact_recall_threshold(cases,'B_P0',reference_tp)
    b0_at_common=sum(int(((case['B_P0']>=threshold_a)&case['gt']).sum()) for case in cases)
    b1_threshold,b1_error=exact_recall_threshold(cases,'B_P1',b0_at_common)
    return {
        'same_threshold':{'A':threshold_a,'B_P0':threshold_a,'B_P1':threshold_a,
                          'matching_note':'All predictions use the A-selected validation threshold.'},
        'B_P0_matched_to_A_recall':{'A':threshold_a,'B_P0':b0_threshold,'B_P1':b0_threshold,
                                    'matching_error_TP':b0_error,
                                    'matching_note':'B-P0 TP count is matched to A; B-P1 uses the same threshold as B-P0.'},
        'B_P1_matched_to_B_P0_recall':{'A':threshold_a,'B_P0':threshold_a,'B_P1':b1_threshold,
                                       'matching_error_TP':b1_error,
                                       'matching_note':'B-P1 TP count is matched to B-P0 at the A threshold.'}}


def analyze(cases,manifest,selection,output,thin_width,short_area,visualizations,provenance=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    points=operating_points(cases,selection)
    gaps,regions,spatial_additions,fragments,bridges,images=[],[],[],[],[],[]
    vis=output/'overviews'
    if visualizations:
        vis.mkdir()
    for case_index,case in enumerate(cases):
        entry=manifest['images'][case['case']]
        prepared=prepare_gt(case['gt'],entry,thin_width)
        for scenario,thresholds in points.items():
            predictions={key:case[key]>=thresholds[key] for key in ('A','B_P0','B_P1')}
            labels={key:component_data(value)[1] for key,value in predictions.items()}
            case_gaps=status_rows(case,entry,predictions,labels,scenario,thresholds)
            case_bridges=[]
            image_row={'scenario':scenario,'case':case['case']}
            for key,prediction in predictions.items():
                image_row.update({key+'_'+name:value for name,value in scores(prediction,case['gt']).items()})
            for before,after,transition in (('A','B_P0','B_P0_vs_A'),('B_P0','B_P1','B_P1_vs_B_P0')):
                rows=region_rows(case[before],case[after],case['gt'],prepared['regions'],case['case'],
                                 scenario,transition,thresholds[before],thresholds[after])
                regions.extend(rows)
                spatial_additions.extend(addition_spatial_rows(predictions[before],predictions[after],case['gt'],
                    prepared['gap_corridor'],labels[after],case['case'],scenario,transition))
                for field in ('added_tp','lost_tp','added_fp','removed_fp'):
                    image_row[transition+'_'+field]=sum(row[field] for row in rows)
                new_fragments,fragment_counts=new_short_components(predictions[before],predictions[after],case['gt'],
                    prepared['thin'],short_area,case['case'],scenario,transition)
                fragments.extend(new_fragments)
                image_row.update({transition+'_'+key:value for key,value in fragment_counts.items()})
                for strict in (True,False):
                    candidates=bridge_candidates(predictions[before],predictions[after],case['gt'],
                        prepared['gap_corridor'],case['case'],scenario,transition,strict,entry['gaps'])
                    bridges.extend(candidates)
                    case_bridges.extend(candidates)
                    suffix='strict_bridges' if strict else 'broad_bridges'
                    image_row[transition+'_'+suffix]=len(candidates)
            for row in case_gaps:
                for transition in ('B_P0_vs_A','B_P1_vs_B_P0'):
                    for definition,suffix in (('strict_outside_GT_3px','strict'),('broad_GT_background','broad')):
                        row[transition+'_'+suffix+'_bridges_within_16px']=sum(
                            candidate['transition']==transition and candidate['definition']==definition and
                            str(row['gap_id']) in candidate['gap_ids_within_16px'].split(',') for candidate in case_bridges)
            gaps.extend(case_gaps)
            images.append(image_row)
            if scenario=='same_threshold' and case_index<visualizations:
                save_visual(vis/(case['case']+'.png'),case,predictions,thresholds,entry)
        print(f'[PAIR AUDIT] {case_index+1}/{len(cases)} {case["case"]}',flush=True)

    summary={'protocol':{'images':len(cases),'fixed_gaps':manifest['count'],
                'split':'validation only; no inference or training; matching thresholds are computed on validation',
                'short_component_area_lt':short_area,'GT_thin_width_proxy_le_pixels':thin_width,
                'fixed_reference_threshold':manifest.get('p0_reference_threshold'),
                'selected_thresholds':{key:selection[key]['threshold'] for key in ('A','B')},
                'operating_points':points},'provenance':provenance or {},'results':{},
             'limits':['These are paired saved predictions; no RGB or training-time feature maps are inferred.',
                       'GT labels are treated as given; missing or incorrect GT can make true roads look like background.',
                       'Thin-road width is estimated from GT skeleton and distance transform, not a semantic road class.',
                       'All after short components are classified; selected new fragments are newly isolated, split descendants, or previously non-short components that became short.',
                       'The 198 fixed reference gaps are automatic candidates, not manually verified breaks; skeleton pixel counts are not geometric lengths.',
                       'Inherited global gap closure uses >=80% exact reference pixels and a common prediction component for all anchors; distant detours may count.',
                       'Local closure uses the same 80% rule and requires anchor connectivity inside the frozen gap/anchor bounding box plus 8px; it may miss legitimate curved paths.',
                       'Both bridge definitions are local-component proxies. Strict excludes a 3px GT-road buffer; broad includes near-road background and can overcount.',
                       'A bridge candidate associated with a gap is within 16 pixels of its frozen missing skeleton; proximity is not causal attribution.',
                       'Matched thresholds are chosen only on validation; TP-count error reports threshold-tie mismatch.']}
    summary['limits'].append('Prediction NPZ files do not have separately recorded hashes; checkpoint hash checks cannot independently prove which weights generated a modified NPZ.')
    for scenario,thresholds in points.items():
        image_set=[row for row in images if row['scenario']==scenario]
        gap_set=[row for row in gaps if row['scenario']==scenario]
        result={'thresholds':thresholds,'models':{},'transitions':{},'fixed_gaps':{}}
        for model in ('A','B_P0','B_P1'):
            total={key:sum(row[model+'_'+key] for row in image_set) for key in ('tp','fp','fn')}
            tp,fp,fn=total['tp'],total['fp'],total['fn']
            total.update(iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),
                         recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))
            result['models'][model]=total
            result['fixed_gaps'][model+'_global_closed_proxy']=sum(row[model+'_global_closed_proxy'] for row in gap_set)
            result['fixed_gaps'][model+'_local_closed_proxy']=sum(row[model+'_local_closed_proxy'] for row in gap_set)
        result['fixed_gaps']['eligible']=len(gap_set)
        for label in ('global','local'):
            result['fixed_gaps']['repair_newly_closed_'+label]=sum(row['repair_newly_closed_'+label] for row in gap_set)
            result['fixed_gaps']['repair_lost_closed_'+label]=sum(row['repair_lost_closed_'+label] for row in gap_set)
        result['fixed_gaps']['by_missing_skeleton_pixel_count']={}
        for label in ('3-7','8-15','16-31','32+'):
            bucket=[row for row in gap_set if row['length_bin']==label]
            result['fixed_gaps']['by_missing_skeleton_pixel_count'][label]={
                'eligible':len(bucket),
                'repair_newly_closed_global':sum(row['repair_newly_closed_global'] for row in bucket),
                'repair_lost_closed_global':sum(row['repair_lost_closed_global'] for row in bucket),
                'repair_newly_closed_local':sum(row['repair_newly_closed_local'] for row in bucket),
                'repair_lost_closed_local':sum(row['repair_lost_closed_local'] for row in bucket)}
        for transition in ('B_P0_vs_A','B_P1_vs_B_P0'):
            subset=[row for row in fragments if row['scenario']==scenario and row['transition']==transition]
            bridge_set=[row for row in bridges if row['scenario']==scenario and row['transition']==transition]
            totals={key:sum(row[transition+'_'+key] for row in image_set) for key in
                    ('added_tp','lost_tp','added_fp','removed_fp','before_short_components','after_short_components',
                     'short_component_net_change','strict_bridges','broad_bridges')}
            categories={}
            for row in subset:
                target=categories.setdefault(row['category'],{'components':0,'area_pixels':0,
                    'GT_road_pixels':0,'GT_background_pixels':0,'GT_thin_road_pixels':0})
                target['components']+=1
                for key in ('area_pixels','GT_road_pixels','GT_background_pixels','GT_thin_road_pixels'):
                    target[key]+=row[key]
            totals['new_split_or_became_short_component_categories']=categories
            all_categories={}
            for image in image_set:
                for category,values in image[transition+'_all_after_short_categories'].items():
                    target=all_categories.setdefault(category,{key:0 for key in values})
                    for key,value in values.items():target[key]+=value
            totals['all_after_short_component_categories']=all_categories
            totals['gross_additions_by_spatial_category']={}
            for category in ('fixed_gap_corridor_2px','isolated_after_component',
                             'previous_prediction_edge_2px','attached_extension_beyond_2px'):
                chosen=[row for row in spatial_additions if row['scenario']==scenario and
                        row['transition']==transition and row['category']==category]
                totals['gross_additions_by_spatial_category'][category]={key:sum(row[key] for row in chosen)
                    for key in ('added_pixels','added_GT_road_pixels','added_GT_background_pixels')}
            totals['strict_bridge_pixels']=sum(row['pixels'] for row in bridge_set if row['definition']=='strict_outside_GT_3px')
            totals['broad_bridge_pixels']=sum(row['pixels'] for row in bridge_set if row['definition']=='broad_GT_background')
            totals['regions']={}
            for region in sorted({row['region'] for row in regions if row['scenario']==scenario and row['transition']==transition}):
                subset_regions=[row for row in regions if row['scenario']==scenario and row['transition']==transition and row['region']==region]
                region_sum={key:sum(row[key] for row in subset_regions) for key in
                            ('region_pixels','GT_road_pixels','GT_background_pixels','added_tp','lost_tp','added_fp','removed_fp')}
                for value_key,weight_key in (('probability_delta_mean','region_pixels'),
                    ('probability_delta_abs_mean','region_pixels'),('GT_road_probability_delta_mean','GT_road_pixels'),
                    ('GT_background_probability_delta_mean','GT_background_pixels')):
                    denominator=region_sum[weight_key]
                    region_sum[value_key]=(sum((row[value_key] or 0.)*row[weight_key] for row in subset_regions)/denominator
                                           if denominator else None)
                totals['regions'][region]=region_sum
            def impact(image):
                return (image[transition+'_lost_tp']+image[transition+'_added_fp']-
                        image[transition+'_added_tp']-image[transition+'_removed_fp'])
            totals['largest_net_error_increases']=[{'case':row['case'],'net_error_increase_pixels':impact(row),
                **{key:row[transition+'_'+key] for key in ('added_tp','lost_tp','added_fp','removed_fp')}}
                for row in sorted(image_set,key=impact,reverse=True)[:5]]
            result['transitions'][transition]=totals
        summary['results'][scenario]=result
    write_json(output/'summary.json',summary)
    def fields(rows,default): return list(rows[0]) if rows else default
    same=[row for row in gaps if row['scenario']=='same_threshold']
    write_csv(output/'fixed_gaps_common.csv',same,fields(same,['scenario','case','gap_id']))
    write_csv(output/'fixed_gaps_all_workpoints.csv',gaps,fields(gaps,['scenario','case','gap_id']))
    write_csv(output/'change_regions.csv',regions,fields(regions,['scenario','transition','case','region']))
    write_csv(output/'added_pixel_categories.csv',spatial_additions,
              fields(spatial_additions,['scenario','transition','case','category']))
    write_csv(output/'new_short_components.csv',fragments,fields(fragments,['scenario','transition','case','component_id']))
    write_csv(output/'bridge_candidates.csv',bridges,fields(bridges,['scenario','transition','case','definition']))
    write_csv(output/'per_image.csv',images,fields(images,['scenario','case']))
    lines=['# A / B paired validation audit','',
           f'Images: {len(cases)}; fixed reference gaps: {manifest["count"]}; existing saved predictions only.',
           'Prediction NPZ files are not individually hash-bound to checkpoints; see summary.json for available provenance checks.',
           'See summary.json for thresholds, net short components, bridge pixels and classification details.','']
    for scenario,result in summary['results'].items():
        lines += [f'## {scenario}','',result['thresholds']['matching_note'],'',
                  '| Model | threshold | IoU | Precision | Recall |','|---|---:|---:|---:|---:|']
        for key,metrics in result['models'].items():
            lines.append(f'| {key} | {result["thresholds"][key]:.6f} | {metrics["iou"]:.6f} | {metrics["precision"]:.6f} | {metrics["recall"]:.6f} |')
        lines += ['','| Transition | +TP | lost TP | +FP | removed FP | strict bridge proxy | broad bridge proxy |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name,value in result['transitions'].items():
            lines.append(f'| {name} | {value["added_tp"]} | {value["lost_tp"]} | {value["added_fp"]} | {value["removed_fp"]} | {value["strict_bridges"]} | {value["broad_bridges"]} |')
        lines += ['', 'Fixed-gap closure proxies: `'+json.dumps(result['fixed_gaps'],ensure_ascii=False)+'`','']
        for name,value in result['transitions'].items():
            lines += [f'### {name}: gross additions by predicted geometry','',
                      '| Category | Added GT road pixels | Added GT background pixels |',
                      '|---|---:|---:|']
            for category,stats in value['gross_additions_by_spatial_category'].items():
                lines.append(f'| {category} | {stats["added_GT_road_pixels"]} | {stats["added_GT_background_pixels"]} |')
            lines += ['']
            lines += [f'### {name}: changed pixels by region','',
                      '| Region | +TP | lost TP | +FP | removed FP | GT-road mean probability change |',
                      '|---|---:|---:|---:|---:|---:|']
            for region,stats in value['regions'].items():
                delta=stats['GT_road_probability_delta_mean']
                text='n/a' if delta is None else f'{delta:.6g}'
                lines.append(f'| {region} | {stats["added_tp"]} | {stats["lost_tp"]} | {stats["added_fp"]} | {stats["removed_fp"]} | {text} |')
            lines += ['', 'New/split/became-short component categories: `'+json.dumps(value['new_split_or_became_short_component_categories'],ensure_ascii=False)+'`',
                      '', 'All after-short component categories: `'+json.dumps(value['all_after_short_component_categories'],ensure_ascii=False)+'`',
                      '', 'Largest per-image net pixel-error increases: `'+json.dumps(value['largest_net_error_increases'],ensure_ascii=False)+'`','']
    lines += ['## Interpretation limits','']+['- '+value for value in summary['limits']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[PAIR AUDIT DONE] '+str(output/'REPORT.md'),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair_dir',type=Path,required=True)
    parser.add_argument('--b_run_name',default='B_transport_0p01_proposal_0p001')
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--expected_gaps',type=int,default=198)
    parser.add_argument('--expected_images',type=int,default=14)
    parser.add_argument('--thin_width',type=float,default=8.)
    parser.add_argument('--short_area',type=int,default=20)
    parser.add_argument('--visualizations',type=int,default=14)
    args=parser.parse_args()
    if args.expected_gaps<0 or args.expected_images<1 or args.thin_width<=0 or args.short_area<1 or args.visualizations<0:
        parser.error('Invalid count or width')
    cases,manifest,cfg,selection,manifest_sha=load_inputs(args.pair_dir,args.b_run_name,args.expected_gaps,args.expected_images)
    print(f'[VERIFIED] {len(cases)} saved val images; {manifest["count"]} frozen gaps; reference SHA256={manifest_sha}',flush=True)
    print('[PROVENANCE] '+json.dumps(selection.get('_verification',{}),ensure_ascii=False),flush=True)
    provenance={'pair_directory':str(args.pair_dir.resolve()),'B_directory':args.b_run_name,
                'fixed_reference_sha256':manifest_sha,'initial_H0_sha256':cfg['A']['h0_sha256'],
                'A_best_sha256':selection['A']['checkpoint_sha256'],
                'B_best_sha256':selection['B']['checkpoint_sha256'],
                'training_epoch_input_fingerprints':'all verified equal',
                'verification':selection.get('_verification',{}),
                'prediction_NPZ_provenance':'not independently hash-verifiable; GT/frozen P0 compared across A/B and with available references'}
    analyze(cases,manifest,selection,args.output_dir,args.thin_width,args.short_area,args.visualizations,provenance)


if __name__=='__main__':main()
