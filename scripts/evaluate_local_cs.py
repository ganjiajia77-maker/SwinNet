"""Paired val-only local contact changes, frozen gap IDs and mask export.

This does not call any path supervision or change model forward. Full TOPO/APLS
must be computed using the shared six-metric tool and the exported manifest.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_native_h0_pair import load_model, owner, digest
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from datasets.native_relation_targets import native_targets
from losses.local_cs_loss import contact_indices
from mass_protocol import blend_weight, sliding_positions
from selector_transport_reference import bridge_proxy
from evaluation.road_comparison.road_metrics import skeletonize
from topology_direction_constants import CONNECTIVITY_DIRECTIONS


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def csv_out(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def counts(pred, gt):
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()))


def metrics(c):
    tp,fp,fn = c['tp'],c['fp'],c['fn']
    return dict(iou=tp/max(tp+fp+fn,1), precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1))


def local_closed(pred, gap):
    # Restrict connectivity to the frozen gap/anchor bounding box + 16 native pixels.
    flat = pred.reshape(-1)
    if flat[gap['pixels']].mean() < .8: return False
    indices = np.array(gap['pixels']+[i for group in gap['anchors'] for i in group])
    y,x=np.divmod(indices,pred.shape[1]);y0=max(int(y.min())-16,0);y1=min(int(y.max())+17,pred.shape[0])
    x0=max(int(x.min())-16,0);x1=min(int(x.max())+17,pred.shape[1])
    _,labels=cv2.connectedComponents(pred[y0:y1,x0:x1].astype(np.uint8),connectivity=8)
    groups=[]
    for anchor in gap['anchors']:
        yy,xx=np.divmod(np.array(anchor),pred.shape[1]);groups.append(set(labels[yy-y0,xx-x0])-{0})
    return bool(groups and set.intersection(*groups))


def check_comparison(paths, kind):
    """Distinguish checkpoint continuation from matched-initialization fresh runs."""
    evidence = {}
    if kind == 'continued':
        for name, path in paths.items():
            if name == 'initial':
                continue
            item = json.loads((path.parent/'local_cs_initial.json').read_text())
            if item['sha256'] != digest(paths['initial']):
                raise ValueError('Run initialized from a different checkpoint')
            evidence[name] = item
    else:
        manifests = {name: json.loads((path.parent/'initial_state.json').read_text())
                     for name, path in paths.items()}
        base = manifests['initial']
        for name, item in manifests.items():
            if not item.get('native_relations'):
                raise ValueError(f'{name}: expected original native-B initialization')
            same_tensors = item['tensors'] == base['tensors']
            if not same_tensors or any(item[k] != base[k] for k in ('seed', 'pretrain_sha256')):
                raise ValueError(f'{name}: independent run initialization does not match original B')
            evidence[name] = dict(seed=item['seed'], pretrain_sha256=item['pretrain_sha256'],
                                  tensors_checked=len(item['tensors']), identical_initialization=True)
    return dict(kind=kind, reference_role='original_B' if kind == 'independent' else 'initial_checkpoint',
                reference_checkpoint=str(paths['initial'].resolve()), evidence=evidence)


def relation_changes(good, y, difficult, confidence, initial, current, threshold, c_threshold):
    original = initial >= threshold
    response = current >= threshold
    missed = good & y & (confidence >= c_threshold) & ~original
    correct = good & y & original
    negative = good & ~y & ~original
    return dict(
        initial_c_supported_missed_positive=int(missed.sum()),
        initial_correct_positive=int(correct.sum()),
        initial_correct_negative=int(negative.sum()),
        initial_correct_hard_negative=int((negative & difficult).sum()),
        recovered_positive=int((missed & response).sum()),
        lost_positive=int((correct & ~response).sum()),
        new_negative_connection=int((negative & response).sum()),
        new_hard_negative_connection=int((negative & difficult & response).sum()))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial', '--reference', dest='initial', type=Path, required=True)
    p.add_argument('--comparison_kind', choices=['continued', 'independent'], default='continued')
    p.add_argument('--run', action='append', required=True, help='name=/absolute/checkpoint.pth')
    p.add_argument('--root_path', type=Path, required=True)
    p.add_argument('--fixed_gaps', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threshold', type=float, default=.2, help='fixed diagnostic contact/surface threshold')
    p.add_argument('--diagnostic_thresholds', type=float, nargs='+',
                   help='multiple fixed thresholds using the same forward, e.g. .2 .46')
    p.add_argument('--threshold_selection', type=Path,
                   help='optional existing val selection; check candidate SHA and selected threshold')
    p.add_argument('--c_threshold', type=float, default=.5, help='diagnostic only; never a training gate')
    a=p.parse_args()
    device=torch.device(a.device)
    paths={'initial':a.initial}
    for item in a.run:
        name, value=item.split('=',1)
        if name not in ('off','surface_only','c_guided') or name in paths:raise ValueError(item)
        paths[name]=Path(value)
    thresholds = [float(np.float32(t)) for t in (a.diagnostic_thresholds or [a.threshold])]
    if any(not 0 < t < 1 for t in thresholds) or len(set(thresholds)) != len(thresholds):
        p.error('Diagnostic thresholds must be unique and strictly between zero and one')
    provenance = check_comparison(paths, a.comparison_kind)
    if a.threshold_selection:
        selected = json.loads(a.threshold_selection.read_text())
        if (selected.get('selected_on') != 'val' or 'c_guided' not in paths
                or selected.get('checkpoint_sha256') != digest(paths['c_guided'])
                or not any(abs(t-float(selected['threshold'])) < 1e-6 for t in thresholds)):
            raise ValueError('Validation threshold evidence does not match the candidate checkpoint/thresholds')
        provenance['existing_val_selection'] = dict(path=str(a.threshold_selection.resolve()),
                                                   threshold=selected['threshold'])
    if a.output_dir.exists():raise ValueError('Use a new output directory')
    a.output_dir.mkdir(parents=True)
    reference=json.loads(a.fixed_gaps.read_text())
    data=RoadSkeletonDataset(str(a.root_path),'val',image_size=None,source_patch_size=1500,return_full_image=True)
    models={}; configs={}
    for name, path in paths.items():
        models[name], configs[name] = load_model(path, device, True)
        models[name].requires_grad_(False)
        if a.comparison_kind == 'independent':
            actual_mode = configs[name].get('local_cs_mode', 'off')
            if actual_mode != ('off' if name == 'initial' else name):
                raise ValueError(f'{name}: checkpoint local_cs_mode is {actual_mode}')
            if configs[name].get('local_cs_initial_checkpoint') or configs[name].get('path_training_mode', 'off') != 'off':
                raise ValueError('Expected fresh B/local-CS training, not continuation or path supervision')
    if a.comparison_kind == 'independent':
        fields = ('max_epochs', 'batch_size', 'seed', 'pretrained_lr', 'new_lr', 'warmup_epochs',
                  'source_patch_size', 'img_size', 'random_crops_per_image', 'use_ema')
        for name, config in configs.items():
            if any(config.get(k) != configs['initial'].get(k) for k in fields):
                raise ValueError(f'{name}: training budget/config differs from original B')
    provenance['checkpoints'] = {name:dict(path=str(path.resolve()), sha256=digest(path)) for name,path in paths.items()}
    dump(a.output_dir/'comparison_provenance.json', provenance)
    print('[COMPARISON CHECK]', json.dumps(provenance), flush=True)
    totals={(name,t):Counter() for name in paths for t in thresholds};fixed=[];cases=[]
    weight=blend_weight(512,device)
    for i in range(len(data)):
        sample=data[i];image=sample['image'][None].to(device);gt=sample['mask'][0].numpy()>.5
        sk=sample['skeleton'][0].numpy()>.5;case=sample['case_name']
        if gt.shape!=(1500,1500) or case not in reference['images']:raise ValueError('Fixed val reference mismatch')
        canvases={name:torch.zeros((1,1,1500,1500),device=device) for name in paths}
        divisor=torch.zeros_like(canvases['initial'])
        for top in sliding_positions(1500,512,256):
            for left in sliding_positions(1500,512,256):
                tile=image[:,:,top:top+512,left:left+512]
                target=native_targets(gt[top:top+512,left:left+512],sk[top:top+512,left:left+512])
                road=torch.as_tensor(gt[top:top+512,left:left+512].reshape(-1),device=device)
                occupied=road.reshape(128,4,128,4).any(3).any(1)
                core=torch.as_tensor(owner(top,left).reshape(-1),device=device)
                lab=torch.as_tensor(target['connectivity'].reshape(8,-1),device=device)>.5
                valid=torch.as_tensor(target['connectivity_valid'].reshape(8,-1),device=device)
                responses={};confidence=None
                for name,model in models.items():
                    outputs=model(tile);z=outputs[0]
                    canvases[name][:,:,top:top+512,left:left+512]+=z*weight
                    prob=z.sigmoid().reshape(-1);edge_values=[]
                    for d,cells,first,second in contact_indices(512,512,4,prob.device):
                        edge_values.append(torch.minimum(prob[first],prob[second]).max(-1).values)
                    responses[name]=edge_values
                    if name=='initial':
                        c3=next(row['connectivity'] for row in outputs[4] if row.get('stage')==3)
                        confidence=c3.sigmoid().reshape(8,-1)
                for k,(d,cells,_,_) in enumerate(contact_indices(512,512,4,road.device)):
                    good=valid[d,cells]&core[cells];y=lab[d,cells]
                    dy,dx=CONNECTIVITY_DIRECTIONS[d];cy,cx=cells//128,cells%128
                    difficult=(~y)&occupied.flatten()[cells]&occupied[cy+dy,cx+dx]
                    for threshold in thresholds:
                        for name in paths:
                            totals[name,threshold].update(relation_changes(
                                good, y, difficult, confidence[d,cells], responses['initial'][k],
                                responses[name][k], threshold, a.c_threshold))
                divisor[:,:,top:top+512,left:left+512]+=weight
        probabilities={name:(value/divisor).sigmoid()[0,0].cpu().numpy() for name,value in canvases.items()}
        cases.append((case,gt,probabilities))
        for threshold in thresholds:
            base=probabilities['initial']>=threshold
            for gap in reference['images'][case]['gaps']:
                old=local_closed(base,gap)
                for name,prob in probabilities.items():
                    closed=local_closed(prob>=threshold,gap)
                    fixed.append(dict(case=case,gap_id=gap['id'],model=name,threshold=threshold,
                                      initial_closed=old,closed=closed,
                                      newly_closed=closed and not old,lost=old and not closed))
        print(f'[LOCAL CS VAL] {i+1}/{len(data)} {case}',flush=True)
    curves=[];summary=[];manifest_models=[]
    for name,path in paths.items():
        for threshold in np.arange(.05,.951,.01):
            c=Counter()
            for _,gt,probs in cases:c.update(counts(probs[name]>=threshold,gt))
            curves.append(dict(model=name,threshold=round(float(threshold),2),**metrics(c)))
        best=max((r for r in curves if r['model']==name),key=lambda r:(r['f1'],-r['threshold']))
        for setting,threshold in [('fixed',t) for t in thresholds]+[('own_val',best['threshold'])]:
            c=Counter();cldice=[];bridges=0
            for case,gt,probs in cases:
                pred=probs[name]>=threshold;c.update(counts(pred,gt));ps=skeletonize(pred);gs=skeletonize(gt)
                precision=(ps & gt).sum()/max(int(ps.sum()),1);recall=(gs & pred).sum()/max(int(gs.sum()),1)
                cldice.append(2*precision*recall/max(precision+recall,1e-12))
                bridges+=bridge_proxy(probs['initial']>=threshold,pred,gt)
            summary.append(dict(model=name,setting=setting,threshold=threshold,**metrics(c),
                                cldice=float(np.mean(cldice)),new_bridge_proxy=bridges))
        directory=a.output_dir/name;directory.mkdir()
        for case,_,probs in cases:
            if not cv2.imwrite(str(directory/(case+'.png')),(probs[name]>=best['threshold']).astype(np.uint8)*255):
                raise IOError(case)
        manifest_models.append(dict(name=name,provenance=dict(checkpoint=str(path.resolve()),sha256=digest(path)),
            predictions=dict(val=[dict(path=str(directory.resolve()),threshold=best['threshold'])])))
    relation_rows=[]
    for (name,threshold),t in totals.items():
        r=dict(model=name,threshold=threshold,c_support_threshold=a.c_threshold,**t)
        for key,num,den in [('recovery_rate','recovered_positive','initial_c_supported_missed_positive'),
                            ('loss_rate','lost_positive','initial_correct_positive'),
                            ('new_negative_rate','new_negative_connection','initial_correct_negative'),
                            ('new_hard_negative_rate','new_hard_negative_connection','initial_correct_hard_negative')]:
            r[key]=t[num]/t[den] if t[den] else None
        relation_rows.append(r)
    csv_out(a.output_dir/'relations.csv',relation_rows);csv_out(a.output_dir/'fixed_gap_changes.csv',fixed)
    gap_summary=[]
    for name in paths:
        for threshold in thresholds:
            group=[r for r in fixed if r['model']==name and r['threshold']==threshold]
            totals_gap={k:sum(r[k] for r in group) for k in ('initial_closed','closed','newly_closed','lost')}
            gap_summary.append(dict(model=name,threshold=threshold,gaps=len(group),**totals_gap,
                                    net=totals_gap['newly_closed']-totals_gap['lost']))
    csv_out(a.output_dir/'fixed_gap_summary.csv',gap_summary)
    csv_out(a.output_dir/'comparison.csv',summary);csv_out(a.output_dir/'val_threshold_scores.csv',curves)
    dump(a.output_dir/'manifest.json',dict(data_root=str(a.root_path.resolve()),protocol=dict(image_size=1500),models=manifest_models))
    dump(a.output_dir/'protocol.json',dict(fixed_gaps=str(a.fixed_gaps),fixed_gaps_sha256=digest(a.fixed_gaps),
        gap_count=reference['count'],local_gap_box_margin=16,gap_coverage=.8,diagnostic_thresholds=thresholds,
        c_support_threshold=a.c_threshold,comparison_kind=a.comparison_kind,reference_role=provenance['reference_role'],
        relation_scope='owned work-grid edges from individual 512 tiles; contact=max of min endpoint probabilities',
        gap_scope='weighted-logit stitched native1500 predictions, fixed original IDs/anchors',
        available_models=list(paths),
        notes=['Local contact is not whole-road connectivity.',
               'Bridge count is a proxy, not manually confirmed false bridges.',
               'Full TOPO/APLS require the shared six-metric tool on manifest.json.']))
    print('[DONE]',a.output_dir,flush=True)


if __name__=='__main__':main()
