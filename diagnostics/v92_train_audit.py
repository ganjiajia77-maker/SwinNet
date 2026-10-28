"""Full training images: prediction-first path proposals and local reach audit."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

import run_reliable_v9_mass as v9
from mass_protocol import sliding_positions
from diagnostics.v91_counterfactual import write_json
from diagnostics.v92_connection_candidates import generate,label,classifier_audit,split_name
from diagnostics.v92_reachability import inspect_gaps


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir',type=Path,required=True); p.add_argument('--root_path',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--thresholds',type=float,nargs='+',default=[.3,.45,.6]); p.add_argument('--min_effect',type=float,default=1e-5)
    p.add_argument('--reliable_threshold',type=float,default=.1); p.add_argument('--min_weight',type=float,default=.01)
    p.add_argument('--min_support',type=float,default=1e-6); p.add_argument('--max_endpoints',type=int,default=32)
    p.add_argument('--max_pairs',type=int,default=64); p.add_argument('--reservoir_per_class',type=int,default=20000)
    p.add_argument('--seed',type=int,default=1234); p.add_argument('--max_images',type=int,default=0)
    args=p.parse_args()
    if any(not 0<t<1 for t in args.thresholds) or min(args.min_effect,args.min_support,args.max_endpoints,args.max_pairs,args.reservoir_per_class)<=0 or not 0<=args.min_weight<1 or not 0<=args.reliable_threshold<1 or args.max_images<0: p.error('Invalid audit parameters')
    ckpt=args.run_dir/'stage_D/best.pth'; sel_path=args.run_dir/'direct_val_selection.json'
    if not ckpt.is_file() or not sel_path.is_file(): p.error('Missing V9 checkpoint or val selection')
    if args.output_dir.exists() and any(args.output_dir.iterdir()): p.error('Use an empty output directory')
    payload=torch.load(ckpt,map_location='cpu',weights_only=False); sel=json.loads(sel_path.read_text())
    if sel.get('selected_on')!='val' or sel['identity']!=payload['identity'] or Path(sel['checkpoint']).resolve()!=ckpt.resolve(): p.error('Checkpoint and validation selection differ')
    if Path(payload['training_config']['root_path']).resolve()!=args.root_path.resolve(): p.error('Data root differs from V9')
    reference=float(payload['training_config']['reference_threshold']); selected=float(sel['threshold']); checksum=v9.digest(ckpt)
    device=torch.device(args.device)
    model,_,identity=v9.load_initial(SimpleNamespace(initialization='direct',q3_checkpoint=Path(payload['identity']['q3']['path']),device=device))
    v9.restore_checkpoint(model,ckpt,identity,'D'); model.eval()
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(args.root_path,'train')
    image_count=min(len(dataset),args.max_images or len(dataset))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    proposal_fields=['case','tile_y','tile_x','threshold','kind','a_y','a_x','b_y','b_x','length_work',
        'direction_channel','endpoint_tangent_alignment','endpoint_tangent_similarity','source_component_area_work',
        'endpoint_p0_a','endpoint_p0_b','endpoint_r_a','endpoint_r_b','gt_label','gt_reason','subtype',
        'gt_snap_distance_a','gt_snap_distance_b','gt_graph_distance','gt_line_background_fraction',
        'gt_line_near_skeleton_fraction','features']
    gap_fields=['case','tile_y','tile_x','gap_id','centroid_y','centroid_x','length_pixels','length_bin',
        'touches_tile_margin','positive_candidate_fraction','at_least_one_candidate','candidate_full_coverage',
        'simple_path','longest_consecutive_no_candidate_pixels','anchor_left_count','anchor_right_count',
        'center_reachable_3_steps','both_flanks_reach_center_3_steps','full_path_reachable_3_steps',
        'path_reachable_fraction_step1','path_reachable_fraction_step2','path_reachable_fraction_step3',
        'actual_closure_eligible','actual_binary_closed','positive_pair_path_hit','positive_pair_path_coverage_best']
    total=Counter(); proposal_counts=Counter(); gap_counts=defaultdict(Counter); population=Counter()
    reservoirs=defaultdict(list); seen=Counter(); rng=np.random.default_rng(args.seed)
    image_sets=defaultdict(set)
    with (args.output_dir/'train_candidates.csv').open('w',newline='',encoding='utf-8') as cf, (args.output_dir/'train_gap_reachability.csv').open('w',newline='',encoding='utf-8') as gf:
        cw=csv.DictWriter(cf,fieldnames=proposal_fields,extrasaction='ignore'); cw.writeheader()
        gw=csv.DictWriter(gf,fieldnames=gap_fields,extrasaction='ignore'); gw.writeheader()
        for index in range(image_count):
            sample=dataset[index]; case=str(sample['case_name']); image=sample['image']; fullgt=sample['mask'][0].numpy()>0.5
            fullskel=sample['skeleton'][0].numpy()>0.5; height,width=fullgt.shape
            positions=[(y,x) for y in sliding_positions(height,512,256) for x in sliding_positions(width,512,256)]
            owner=np.full((height,width),-1,np.int16)
            for tile_index,(y,x) in enumerate(positions):
                area=owner[y:y+512,x:x+512]; area[area<0]=tile_index
            for tile_index,(y,x) in enumerate(positions):
                print(f'[TRAIN TILE] image={index+1}/{image_count} case={case} '
                      f'tile={tile_index+1}/{len(positions)}',flush=True)
                tile=image[None,:,y:y+512,x:x+512].to(device)
                gt=fullgt[y:y+512,x:x+512]; skel=fullskel[y:y+512,x:x+512]
                result=model(tile,capture_diagnostics=True)
                gap_rows,gap_work=inspect_gaps(result,gt,skel,case,reference,selected,args.min_effect,
                    args.reliable_threshold,args.min_weight,args.min_support)
                selected_gaps=[]
                for row in gap_rows:
                    gy=min(height-1,y+int(row['centroid_y'])); gx=min(width-1,x+int(row['centroid_x']))
                    if owner[gy,gx]!=tile_index: continue
                    row.update(tile_y=y,tile_x=x); selected_gaps.append(row)
                best={row['gap_id']:0. for row in selected_gaps}
                p0=result['u0'][0,0].cpu().numpy(); r=result['r'][0,0].cpu().numpy()
                w=result['w'][0].cpu().numpy(); q=result['q'][0,0].cpu().numpy()
                for threshold in args.thresholds:
                    proposals,meta=generate(p0,r,w,q,threshold,args.max_endpoints,args.max_pairs)
                    total['endpoints_before_cap']+=meta['endpoints_before_cap']; total['endpoints_used']+=meta['endpoints_used']
                    total['pairs_before_cap']+=meta['pairs_before_cap']; total['pairs_used']+=meta['pairs_used']
                    kept=[]
                    for c in proposals:
                        my=int(round((c['a_y']+c['b_y'])*.5*512/p0.shape[0])); mx=int(round((c['a_x']+c['b_x'])*.5*512/p0.shape[1]))
                        if owner[min(height-1,y+my),min(width-1,x+mx)]==tile_index: kept.append(c)
                    kept,found=label(kept,gt,p0,reference,{row['gap_id']:gap_work[row['gap_id']] for row in selected_gaps})
                    for gap,value in found.items(): best[gap]=max(best[gap],value)
                    for c in kept:
                        c.update(case=case,tile_y=y,tile_x=x,threshold=threshold)
                        proposal_counts[f"{threshold:.2f}|{c['gt_label']}|{c['subtype']}"]+=1
                        export=dict(c); export['features']=json.dumps(c['features']); cw.writerow(export)
                        if c['gt_label'] in ('positive','negative'):
                            group=f"{split_name(case,args.seed)}_{c['gt_label']}"; population[group]+=1; image_sets[group].add(case)
                            seen[group]+=1; pool=reservoirs[group]
                            if len(pool)<args.reservoir_per_class: pool.append(c)
                            else:
                                j=int(rng.integers(seen[group]))
                                if j<len(pool): pool[j]=c
                for row in selected_gaps:
                    score=best[row['gap_id']]; row['positive_pair_path_hit']=bool(score>0)
                    row['positive_pair_path_coverage_best']=float(score); gw.writerow(row)
                    g=gap_counts[row['length_bin']]; g['all_owned']+=1
                    if row['touches_tile_margin']: g['boundary_excluded']+=1; continue
                    g['non_boundary_gaps']+=1; g['at_least_one_candidate']+=int(row['at_least_one_candidate'])
                    g['candidate_full_coverage']+=int(row['candidate_full_coverage'])
                    g['positive_path_hit']+=int(score>0); g['positive_path_near_full_proxy']+=int(score>=.99)
                    if row['simple_path']:
                        g['simple_paths']+=1; g['center_reachable_3_steps']+=int(row['center_reachable_3_steps'])
                        g['both_flanks_reach_center_3_steps']+=int(row['both_flanks_reach_center_3_steps'])
                        g['full_path_reachable_3_steps']+=int(row['full_path_reachable_3_steps'])
                        g['no_left_or_right_reliable_anchor']+=int(not row['anchor_left_count'] or not row['anchor_right_count'])
                        g['longest_missing_run_sum']+=row['longest_consecutive_no_candidate_pixels']
                    else: g['non_simple_or_branch_excluded']+=1
                    if row['actual_closure_eligible']:
                        g['actual_closure_eligible']+=1; g['actual_binary_closed_proxy']+=int(row['actual_binary_closed'])
                total['tiles']+=1
            print(f'[TRAIN AUDIT] {index+1}/{image_count} {case} tiles={len(positions)}',flush=True)
    summary_gaps={}
    for name,counts in gap_counts.items():
        c=dict(counts); base=c.get('non_boundary_gaps',0); simple=c.get('simple_paths',0)
        for numerator,denominator in (('at_least_one_candidate',base),('candidate_full_coverage',base),
            ('positive_path_hit',base),('positive_path_near_full_proxy',base),
            ('center_reachable_3_steps',simple),('both_flanks_reach_center_3_steps',simple),
            ('full_path_reachable_3_steps',simple),('actual_binary_closed_proxy',c.get('actual_closure_eligible',0))):
            c[numerator+'_rate']=c.get(numerator,0)/denominator if denominator else None
        c['mean_longest_missing_run_pixels']=c.get('longest_missing_run_sum',0)/simple if simple else None
        summary_gaps[name]=c
    sampled=[item for group in reservoirs.values() for item in group]
    classifier=classifier_audit(sampled,args.seed,dict(population))
    classifier['population_image_counts']={k:len(v) for k,v in image_sets.items()}
    if v9.digest(ckpt)!=checksum: raise RuntimeError('Original checkpoint changed')
    report={'checkpoint_sha256':checksum,'identity':identity,'split':'train','images':image_count,
        'dataset_total_images':len(dataset),'complete_train_split':image_count==len(dataset),'tiles':total['tiles'],
        'p0_reference_threshold':reference,'p1_threshold_from_val':selected,'candidate_thresholds':args.thresholds,
        'settings':{'min_effect':args.min_effect,'reliable_threshold':args.reliable_threshold,'min_weight':args.min_weight,
                    'min_support':args.min_support,'max_endpoints':args.max_endpoints,'max_pairs':args.max_pairs,
                    'reservoir_per_class':args.reservoir_per_class,'seed':args.seed},
        'proposal_generation_counters':dict(total),'candidate_labels_by_threshold_and_subtype':dict(proposal_counts),
        'gap_reachability_by_length':summary_gaps,'diagnostic_classifier':classifier,
        'limits':['The selected training images are scanned using 512/256 windows. First-window centroid/midpoint ownership reduces overlap but tile-specific predictions may split or duplicate a gap; this is not a stitched full-image path census.',
                  'Boundary-truncated and branched/non-simple skeleton pieces are separately flagged and excluded from ordered-path rates.',
                  'Three-step reach uses the original-run per-step pre-gate positive candidate, source support, W and an explicit reliable-source threshold. It is a graph reachability proxy, not a causal rerun or proof of binary closure.',
                  'GT is used for labels and analysis only; proposals use only P0/R/W/Q and predicted geometry. GT snap/detour, extension and bridge subtypes are imperfect proxies; uncertain labels are not forced into the classifier.',
                  'Full-path candidate coverage and GT-path proposal hit are distinct from actual P1 binary closure. Candidate endpoint and pair caps are recorded and limit recall.',
                  'The image-disjoint classifier holdout is within the training split; prevalence-weighted PR-AUC is an estimate if reservoirs truncate samples.']}
    write_json(args.output_dir/'train_path_audit.json',report)
    lines=['# V9 train path reach and candidate audit','',
           '| length (pixels) | non-boundary gaps | any local candidate | full local coverage | simple paths | center reachable / simple paths | full path reachable / simple paths | closure / eligible |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('3-7','8-15','16-31','32+'):
        c=summary_gaps.get(name,{})
        lines.append(f'| {name} | {c.get("non_boundary_gaps",0)} | {c.get("at_least_one_candidate",0)} | '
                     f'{c.get("candidate_full_coverage",0)} | {c.get("simple_paths",0)} | '
                     f'{c.get("center_reachable_3_steps",0)} | {c.get("full_path_reachable_3_steps",0)} | '
                     f'{c.get("actual_binary_closed_proxy",0)}/{c.get("actual_closure_eligible",0)} |')
    lines+=['','Candidate labels by threshold and proxy subtype:']
    lines += [f'- {key}: {value}' for key,value in sorted(proposal_counts.items())]
    if classifier['available']:
        natural=classifier['natural_holdout']; matched=classifier['matched_holdout']
        lines += ['',f'Image-disjoint held-out training candidates: natural prevalence '
                   f'{natural["weighted_positive_fraction"]:.4f}; estimated PR-AUC {natural["pr_auc"]:.4f}; '
                   f'ROC-AUC {natural["roc_auc"]:.4f}.',
                  f'Matched candidate PR-AUC: {matched["pr_auc"] if matched["pr_auc"] is not None else "unavailable"}; '
                  f'ROC-AUC: {matched["roc_auc"] if matched["roc_auc"] is not None else "unavailable"}; '
                  f'sampled candidates: {matched["n"]}.']
    else:
        lines += ['',f'Candidate classifier unavailable: {classifier["reason"]}.']
    lines += ['', 'Classifier: `diagnostic_classifier` in train_path_audit.json. '
              'See the JSON for denominators, positive path proposals, natural/matched PR-AUC and sampling limits. '
              'The CSV files contain individual gaps and candidates. All findings are train-only window-level proxies.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[DONE] {args.output_dir}',flush=True)

if __name__=='__main__': main()
