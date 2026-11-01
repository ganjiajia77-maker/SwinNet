"""Offline spatial audit of saved fixed-batch startup predictions; no model load."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from diagnostics.v91_counterfactual import write_csv
from evaluation.road_comparison.road_metrics import skeletonize
from run_v9_gate_benefit import atomic_json


def components(mask):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def fixed_gaps(p0, gt, skeleton, threshold):
    binary = p0 >= threshold
    covered = cv2.dilate(binary.astype(np.uint8), np.ones((5,5),np.uint8)) > 0
    count, labels, stats, _ = components(skeleton & ~covered)
    _, old, _, _ = components(binary)
    h, w = gt.shape
    result = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        x,y,width,height = map(int,stats[index,:4])
        x0,x1 = max(x-7,0),min(x+width+7,w)
        y0,y1 = max(y-7,0),min(y+height+7,h)
        missing = labels[y0:y1,x0:x1] == index
        flank = cv2.dilate(missing.astype(np.uint8), np.ones((15,15),np.uint8)) > 0
        local = old[y0:y1,x0:x1]
        ground = gt[y0:y1,x0:x1]
        ids = np.unique(local[flank & ground])
        ids = ids[ids > 0]
        if len(ids) < 2:
            continue
        def coordinates(mask):
            ys,xs = np.nonzero(mask)
            return np.stack((ys+y0,xs+x0),1)
        result.append({'id':index,'pixels':coordinates(missing),
                       'anchors':[coordinates(flank & (local == value) & ground) for value in ids]})
    return result


def gap_status(prediction, gap):
    """Use the same frozen anchors; restrict connectivity to the gap neighbourhood."""
    ys,xs = gap['pixels'].T
    all_points = np.concatenate([gap['pixels']] + gap['anchors'])
    y0,x0 = np.maximum(all_points.min(0)-8,0)
    y1,x1 = np.minimum(all_points.max(0)+9,prediction.shape)
    _,labels,_,_ = components(prediction[y0:y1,x0:x1])
    anchor_ids = [set(np.unique(labels[a[:,0]-y0,a[:,1]-x0])) - {0} for a in gap['anchors']]
    connected = bool(anchor_ids and set.intersection(*anchor_ids))
    coverage = float(prediction[ys,xs].mean())
    tolerant = cv2.dilate(prediction.astype(np.uint8),np.ones((3,3),np.uint8)) > 0
    return {'coverage':coverage,'coverage_tolerance_1px':float(tolerant[ys,xs].mean()),
            'local_anchor_connection':connected,'closed_proxy':bool(connected and coverage >= .8),
            'missing_anchor_groups':sum(not value for value in anchor_ids)}


def addition_components(p0, p1, gt):
    """Partition gross added TP/FP components; bridge labels are geometry proxies."""
    _,old,_,_ = components(p0)
    _,final,_,_ = components(p1)
    distance = cv2.distanceTransform((~p0).astype(np.uint8),cv2.DIST_L2,5)
    rows = []
    for target,mask in (('added_tp',p1 & ~p0 & gt),('added_fp',p1 & ~p0 & ~gt)):
        count,labels,stats,_ = components(mask)
        for index in range(1,count):
            region = labels == index
            x,y,width,height,area = map(int,stats[index])
            near = cv2.dilate(region.astype(np.uint8),np.ones((3,3),np.uint8)) > 0
            adjacent = np.unique(old[near]); adjacent = adjacent[adjacent > 0]
            parent_ids = np.unique(final[region]); parent_ids = parent_ids[parent_ids > 0]
            attached = any(bool((old[final == value] > 0).any()) for value in parent_ids)
            if not attached:
                category = 'isolated_island'
            elif len(adjacent) >= 2:
                category = 'bridge_proxy'
            elif float(distance[region].max()) <= 2.:
                category = 'road_edge_expansion'
            else:
                category = 'extension_or_other'
            rows.append({'kind':target,'component':index,'category':category,'pixels':area,
                         'skeleton_pixel_count':int(skeletonize(region[y:y+height,x:x+width]).sum()),
                         'x':x,'y':y,'width':width,'height':height,
                         'adjacent_old_components':len(adjacent),'attached_to_old_road':attached,
                         'min_distance_to_old_road':float(distance[region].min()) if p0.any() else None,
                         'mean_distance_to_old_road':float(distance[region].mean()) if p0.any() else None})
    return rows


def save_overview(path, rgb, gt, p0, p1, threshold, gaps=()):
    """Six aligned native crop panels; overlay colours represent gross changes."""
    def label(image, text):
        image = image.copy()
        cv2.rectangle(image,(0,0),(image.shape[1],28),(0,0,0),-1)
        cv2.putText(image,text,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        return image
    def probability(p):
        return cv2.applyColorMap(np.clip(p*255,0,255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)
    original = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    changes = original.copy()
    old,new = p0 >= threshold,p1 >= threshold
    changes[new & ~old & gt] = (0,255,0)
    changes[new & ~old & ~gt] = (0,0,255)
    changes[~new & old & gt] = (0,255,255)
    changes[~new & old & ~gt] = (255,255,0)
    delta = np.clip((p1-p0)/.1,-1,1)
    signed = np.full((*gt.shape,3),255.,dtype=np.float32)
    signed[:,:,0] -= np.maximum(delta,0)*255
    signed[:,:,1] -= np.abs(delta)*255
    signed[:,:,2] -= np.maximum(-delta,0)*255
    panels = [label(original,'RGB: fixed training crop'),
              label(cv2.cvtColor(gt.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR),'GT road'),
              label(probability(p0),'Current P0 probability'),label(probability(p1),'P1 probability'),
              label(signed.astype(np.uint8),'P1-P0: red +, blue -, scale +/-0.1'),
              label(changes,'+TP green; +FP red; -TP yellow; -FP cyan')]
    for gap in sorted(gaps,key=lambda value:len(value['pixels']),reverse=True)[:30]:
        ys,xs = gap['pixels'].T
        x0,y0 = max(int(xs.min())-3,0),max(int(ys.min())-3,28)
        x1,y1 = min(int(xs.max())+3,gt.shape[1]-1),min(int(ys.max())+3,gt.shape[0]-1)
        if y1 < y0:
            continue
        for panel in (panels[0],panels[2],panels[3]):
            cv2.rectangle(panel,(x0,y0),(x1,y1),(0,255,255),1)
            cv2.putText(panel,str(gap['id']),(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
    grid = np.concatenate((np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)),0)
    if not cv2.imwrite(str(path),grid):
        raise OSError('Could not save overview: '+str(path))


def analyze(folder, update=None):
    folder = Path(folder)
    report = json.loads((folder/'startup_report.json').read_text(encoding='utf-8'))
    metadata = report['metadata']
    threshold = float(metadata['threshold'])
    files = sorted((folder/'snapshots').glob('update_*.npz'))
    if not files:
        raise FileNotFoundError('No saved spatial snapshots; old summary-only reports cannot locate pixels.')
    final = files[-1] if update is None else folder/'snapshots'/f'update_{update:04d}.npz'
    with np.load(folder/'batch_inputs.npz') as batch, np.load(files[0]) as start, np.load(final) as saved:
        rgb,gt,skeleton = batch['rgb'],batch['gt'][:,0].astype(bool),batch['skeleton'][:,0].astype(bool)
        initial_p0,p0,p1 = start['p0'][:,0],saved['p0'][:,0],saved['p1'][:,0]
        actions = {key:saved[key] for key in ('transport_delta','proposal_delta','anchor_delta')}
    if p0.shape != gt.shape or p1.shape != gt.shape or initial_p0.shape != gt.shape:
        raise ValueError('Spatial grids differ')
    output = folder/'spatial_analysis'
    output.mkdir(exist_ok=True)
    component_rows,gap_rows,action_rows,case_rows = [],[],[],[]
    for index in range(len(gt)):
        name = f'{index}_{metadata["training_cases"][index]}'
        old,new = p0[index] >= threshold,p1[index] >= threshold
        masks = {'added_tp':new & ~old & gt[index],'added_fp':new & ~old & ~gt[index],
                 'lost_tp':old & ~new & gt[index],'removed_fp':old & ~new & ~gt[index]}
        rows = addition_components(old,new,gt[index])
        component_rows.extend(dict(case=name,**row) for row in rows)
        gaps = fixed_gaps(initial_p0[index],gt[index],skeleton[index],threshold)
        for gap in gaps:
            before,after = gap_status(old,gap),gap_status(new,gap)
            ys,xs = gap['pixels'].T
            length = len(ys)
            gap_rows.append({'case':name,'gap_id':gap['id'],'missing_skeleton_pixels':length,
                             'length_bin':'3-7' if length<8 else '8-15' if length<16 else '16-31' if length<32 else '32+',
                             'x_min':int(xs.min()),'y_min':int(ys.min()),'x_max':int(xs.max()),'y_max':int(ys.max()),
                             'anchor_groups':len(gap['anchors']),
                             'crop_boundary':bool((ys<2).any() or (xs<2).any() or (ys>=gt.shape[1]-2).any() or (xs>=gt.shape[2]-2).any()),
                             **{'P0_'+key:value for key,value in before.items()},
                             **{'P1_'+key:value for key,value in after.items()}})
        for region,mask in masks.items():
            for step in range(3):
                values = {'case':name,'region':region,'step':step+1,'pixels':int(mask.sum())}
                for key,maps in actions.items():
                    field = cv2.resize(maps[index,step,0],(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_LINEAR)
                    values[key+'_signed_mean'] = float(field[mask].mean()) if mask.any() else None
                    values[key+'_abs_mean'] = float(np.abs(field[mask]).mean()) if mask.any() else None
                action_rows.append(values)
        case_rows.append({'case':name,**{key:int(value.sum()) for key,value in masks.items()},'reference_gaps':len(gaps)})
        save_overview(output/(name+'.png'),rgb[index],gt[index],p0[index],p1[index],threshold,gaps)
    summary = {'scope':'Two fixed training crops, not a validation estimate. Coordinates/length bins are crop pixels.',
               'update':int(final.stem.split('_')[-1]),'threshold':threshold,'cases':case_rows,
               'totals':{key:sum(row[key] for row in case_rows) for key in ('added_tp','added_fp','lost_tp','removed_fp')},
               'component_categories':{},
               'fixed_gaps':{'eligible':len(gap_rows),'P0_closed_proxy':sum(row['P0_closed_proxy'] for row in gap_rows),
                             'P1_closed_proxy':sum(row['P1_closed_proxy'] for row in gap_rows),
                             'repair_newly_closed_proxy':sum(not row['P0_closed_proxy'] and row['P1_closed_proxy'] for row in gap_rows),
                             'repair_lost_closed_proxy':sum(row['P0_closed_proxy'] and not row['P1_closed_proxy'] for row in gap_rows)},
               'repair_parameter_updates':metadata.get('repair_parameter_updates',{}),
               'replay_final_minus_previous':metadata.get('replay_final_minus_previous'),
               'limits':['Gross additions/removals are located; net counts are arithmetic differences, not pixel sets.',
                         'Reference gaps/anchors come from step-0 P0 in these crops; not the 14-image validation reference.',
                         'Closure proxy requires >=80% exact centreline coverage and connected frozen anchors inside the local ROI.',
                         'Anchor selection and bridge categories are unverified proxies, not confirmed true/false graph edges.',
                         'Skeleton pixel count is not a geometric path-length measurement.',
                         'Overview boxes show up to 30 largest reference missing components per crop; all are listed in CSV.',
                         'Action values are interpolated work-grid probability increments, not additive final-P1 contributions.']}
    for kind in ('added_tp','added_fp'):
        category = {}
        for row in component_rows:
            if row['kind'] != kind:
                continue
            item = category.setdefault(row['category'],{'components':0,'pixels':0})
            item['components'] += 1
            item['pixels'] += row['pixels']
        summary['component_categories'][kind] = category
    atomic_json(output/'spatial_summary.json',summary)
    write_csv(output/'added_components.csv',component_rows)
    write_csv(output/'fixed_gap_changes.csv',gap_rows)
    write_csv(output/'action_regions.csv',action_rows)
    lines = ['# Startup spatial diagnostic','',
             'Fixed training crops only. All TP/FP additions compare P1 against the same update current P0.',
             f'Update: {summary["update"]}; threshold: {threshold}.','',
             '| Gross change | Pixels |','|---|---:|']
    lines += [f'| {key} | {value} |' for key,value in summary['totals'].items()]
    lines += ['', '## Added component categories','', '| Kind | Category | Components | Pixels |','|---|---|---:|---:|']
    for kind,categories in summary['component_categories'].items():
        lines += [f'| {kind} | {category} | {value["components"]} | {value["pixels"]} |' for category,value in categories.items()]
    lines += ['', '## Fixed local gap proxies','',json.dumps(summary['fixed_gaps'],ensure_ascii=False),'',
              'See fixed_gap_changes.csv for per-gap coverage and frozen-anchor connectivity, and action_regions.csv for the three update channels.','',
              '## Limits',''] + ['- '+value for value in summary['limits']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[SPATIAL DONE] '+str(output/'REPORT.md'),flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir',type=Path,required=True)
    parser.add_argument('--update',type=int)
    args = parser.parse_args()
    analyze(args.run_dir,args.update)
