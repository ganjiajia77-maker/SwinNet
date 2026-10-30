"""Continue the trained S_selector; add only late wrong-choice mass loss."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_reliable_v9_mass as v9
import run_v9_selector_transport as joint
from diagnose_selector_late_transport import load_current, LateIntervention
from networks.reliable_directional_repair import dynamic_losses, compose_surface_logits
from networks.v9_selector_transport_training import configure_joint, canonical_state, load_canonical, live_relation_loss, gradient_norm
from networks.v9_source_relation_labels import RelationLabels
from networks.v9_late_choice_loss import late_choice_loss
from run_v9_gate_benefit import atomic_json
from selector_transport_provenance import read_json, SELECTION_RULE
from selector_transport_reference import reference_digest, exact_recall_threshold
from mass_protocol import sliding_positions, blend_weight
from diagnostics.v91_counterfactual import write_csv

VERSION = 'v9_late_wrong_choice_continuation_v1'


def pack(model, args, epoch, score, optimizer=None):
    state = {'version': VERSION, 'config': args.config, 'identity': args.identity,
        'epoch': epoch, 'score': score,
        'modules': {name: canonical_state(getattr(model.original, name)) for name in v9.MODULE_NAMES},
        'gate_relation': model.original.gate_relation.state_dict(),
        'state_gate': model.original.state_gate.state_dict(), 'selector': model.selector.state_dict()}
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    return state


def save(path, state):
    temporary = path.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(path)


def restore(model, args, path, optimizer=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['version'] != VERSION or state['config'] != args.config or state['identity'] != args.identity:
        raise ValueError('Continuation checkpoint configuration/identity changed')
    load_weights(model, state)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer']); torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def load_weights(model, state):
    if set(state['modules']) != set(v9.MODULE_NAMES): raise ValueError('Incomplete downstream checkpoint')
    for name in v9.MODULE_NAMES: load_canonical(getattr(model.original, name), state['modules'][name])
    model.original.gate_relation.load_state_dict(state['gate_relation'], strict=True)
    model.original.state_gate.load_state_dict(state['state_gate'], strict=True)
    model.selector.load_state_dict(state['selector'], strict=True)


def losses(args, result, gt, skeleton, seed):
    base, terms = dynamic_losses(result, gt, skeleton, **args.source_losses, return_live_terms=True)
    relation, info = live_relation_loss(result, gt, skeleton, args.labels, seed)
    choice, counts = late_choice_loss(result, info)
    total = base + .1 * relation + args.lambda_choice * choice
    return total, base, relation, choice, terms, info, counts


def smoke(args, model, optimizer):
    dataset = joint.dataset_train(args, False); dataset.set_epoch(1)
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    image, gt, skeleton = (batch[key].to(args.device) for key in ('image', 'mask', 'skeleton'))
    model.train(); optimizer.zero_grad(set_to_none=True)
    result = model(image, mode='source')
    total, base, relation, choice, terms, info, counts = losses(args, result, gt, skeleton, args.seed)
    logits = torch.cat((result['source_stay_logit'], result['source_edge_logits'].masked_fill(~result['source_valid'], -torch.inf)), 2)
    torch.testing.assert_close(logits.softmax(2), result['source_choice'], atol=2e-6, rtol=1e-5)
    parameters = tuple(model.selector.parameters())
    seg_norm = gradient_norm(base - .2 * terms['cont_live'], parameters)
    choice_norm = gradient_norm(args.lambda_choice * choice, parameters)
    edge_grad, stay_grad = torch.autograd.grad(choice, (result['source_edge_logits'], result['source_stay_logit']), retain_graph=True)
    if bool(edge_grad[:, 0].any()) or bool(stay_grad[:, 0].any()): raise RuntimeError('Choice loss reached step1 logits directly')
    supervised = sum(row['wrong_bearing_positions'] for step in counts['steps'] for row in step['images'])
    if supervised and (choice_norm <= 0 or not bool(stay_grad[:, 1:].abs().sum() > 0)):
        raise RuntimeError('Choice loss cannot train selector/stay')
    total.backward()
    if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite smoke loss')
    if any(p.grad is not None for p in model.original.frozen.parameters()): raise RuntimeError('BASE/Q3 received gradients')
    report = {'passed': True, 'choice': float(choice.detach()), 'relation': float(relation.detach()),
        'choice_supervision_exercised': bool(supervised),
        'choice_counts': counts, 'selector_seg_gradient_norm': seg_norm, 'selector_weighted_choice_gradient_norm': choice_norm,
        'choice_to_seg_ratio': choice_norm / max(seg_norm, 1e-12),
        'review_choice_dominance': choice_norm > seg_norm, 'stay_logit_gradient_abs': float(stay_grad.abs().sum()),
        'step1_direct_choice_gradient_zero': True, 'actual_softmax_verified': True,
        'peak_cuda_mib': torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type == 'cuda' else None}
    atomic_json(args.output_dir/'choice_smoke.json', report); optimizer.zero_grad(set_to_none=True)
    print('[CHOICE CHECK] ' + json.dumps(report), flush=True)


def validation_cases(args):
    from run_q3_r3_mass import full_dataset
    dataset = full_dataset(args.root_path, 'val')
    if len(dataset) != 14: raise ValueError('Expected the unchanged 14-image val split')
    return dataset


@torch.no_grad()
def infer(args, model, dataset, name, cases=None, statistics=False):
    model.eval(); result_cases = []; stats = []
    for index in range(len(dataset)):
        sample = dataset[index]
        callback, totals = joint.source_statistics_callback(args, sample) if statistics else (None, None)
        p0, p1 = joint.stitch(model, sample['image'][None].to(args.device), args.device, callback)
        case = {'case': str(sample['case_name']), 'p0': p0, 'gt': sample['mask'][0].numpy() > .5,
                'input_sha256': joint.tensor_digest(sample), name: p1}
        if cases is not None:
            old = cases[index]
            if old['case'] != case['case'] or old['input_sha256'] != case['input_sha256'] or not np.array_equal(old['p0'], p0):
                raise ValueError('Validation/P0 input changed between checkpoints')
            old[name] = p1; case = old
        result_cases.append(case)
        if totals is not None: stats.append(totals)
        print(f'[CHOICE VAL] {name} {index+1}/{len(dataset)}', flush=True)
    if statistics: atomic_json(args.output_dir/(name+'_source_statistics.json'), joint.merge_statistics(stats))
    return result_cases


def train(args, model, optimizer, dataset, manifest):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    start = 1; best = -float('inf'); last = args.output_dir/'last.pth'
    if last.is_file():
        state = restore(model, args, last, optimizer); start = state['epoch'] + 1; best = state['score']
    trainset = joint.dataset_train(args)
    for epoch in range(start, args.epochs+1):
        # Continue crop epochs after the selected source epoch, not epoch1.
        crop_epoch = args.source_epoch + epoch; trainset.set_epoch(crop_epoch)
        generator = torch.Generator().manual_seed(args.seed + crop_epoch * 1000003)
        loader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=args.num_workers,
                            pin_memory=args.device.type == 'cuda', generator=generator)
        model.train(); model.set_accept_floor(0.); sums = Counter(); position_counts = Counter(); batch_rows = []
        order = hashlib.sha256(); augmented = hashlib.sha256(); gradients = None
        for index, batch in enumerate(loader):
            for case in batch['case_name']: order.update(str(case).encode()+b'\0')
            augmented.update(batch['image'].contiguous().numpy().tobytes())
            image, gt, skeleton = (batch[key].to(args.device) for key in ('image', 'mask', 'skeleton'))
            optimizer.zero_grad(set_to_none=True); result = model(image, mode='source')
            total, base, relation, choice, terms, info, counts = losses(args, result, gt, skeleton, args.seed+crop_epoch*100000+index)
            wrong_count = sum(row['wrong_bearing_positions'] for step in counts['steps'] for row in step['images'])
            if gradients is None and wrong_count:
                parameters = tuple(p for p in model.parameters() if p.requires_grad)
                sg = gradient_norm(base-.2*terms['cont_live'], parameters)
                cg = gradient_norm(args.lambda_choice*choice, parameters)
                gradients = {'weighted_choice_norm': cg, 'seg_norm': sg, 'ratio': cg/max(sg,1e-12), 'review_dominance': cg>sg}
            if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite continuation loss')
            total.backward()
            if any(p.grad is not None for p in model.original.frozen.parameters()): raise RuntimeError('Frozen BASE/Q3 has gradients')
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            for key, value in (('total',total),('seg',terms['seg']),('cont',terms['cont']),('relation',relation),('choice',choice)):
                sums[key] += float(value.detach())
            sums['batches'] += 1; sums['matched_relation_pairs'] += info['matching']['matched_pairs']
            for step in counts['steps']:
                for row in step['images']:
                    for key in ('positions','wrong_bearing_positions','all_valid_sources_wrong_positions','wrong_mass_sum','loss_sum','skipped'):
                        position_counts[f'step{step["step"]}_{key}'] += row[key]
            batch_rows.append({'batch':index+1,'cases':list(batch['case_name']),'choice':float(choice.detach()),'counts':counts})
            if index%25 == 0: print(f'[CHOICE TRAIN] {epoch}/{args.epochs} batch={index+1}/{len(loader)} total={float(total.detach()):.6f} choice={float(choice.detach()):.6f}', flush=True)
        record = {'epoch':epoch,'crop_epoch':crop_epoch,'losses':{k:sums[k]/max(sums['batches'],1) for k in ('total','seg','cont','relation','choice')},
                  'counts':dict(position_counts),'matched_relation_pairs':sums['matched_relation_pairs'],'gradient_check':gradients,
                  'case_order_sha256':order.hexdigest(),'augmented_images_sha256':augmented.hexdigest()}
        atomic_json(args.output_dir/f'epoch_{epoch:03d}_choice_positions.json', batch_rows)
        if not sum(position_counts[f'step{k}_wrong_bearing_positions'] for k in (2,3)):
            atomic_json(args.output_dir/'failed_no_choice_labels.json',record); raise RuntimeError('No supervised wrong-source positions in the whole epoch')
        if not sums['matched_relation_pairs']: raise RuntimeError('No matched relation supervision in whole epoch')
        if epoch%args.val_interval == 0 or epoch == args.epochs:
            cases = infer(args,model,dataset,'choice')
            if reference_digest(cases) != manifest['sha256']: raise ValueError('Frozen P0 reference fingerprint changed')
            val = joint.point_metrics(cases,'choice',.6,manifest); record['val_score_at_0_60'] = val['selection_score']
            atomic_json(args.output_dir/f'val_epoch_{epoch:03d}.json',val)
            if val['selection_score'] > best:
                best = val['selection_score']; save(args.output_dir/'best.pth',pack(model,args,epoch,best))
        atomic_json(args.output_dir/f'epoch_{epoch:03d}.json',record)
        state = pack(model,args,epoch,best,optimizer); state['epoch_record'] = record
        save(last,state)
    atomic_json(args.output_dir/'training_complete.json',{'epochs':args.epochs,'best_score':best})


@torch.no_grad()
def compare(args, model, source_state, dataset, manifest, common_threshold):
    load_weights(model,source_state)
    cases = infer(args,model,dataset,'current_S',statistics=True)
    # Transfer all late source probability to stay; proposal/anchor still run.
    stopped = []
    for sample in dataset:
        def stopped_forward(image, **kwargs):
            # A new intervention per tile: exactly three selector calls.
            with LateIntervention(model,'stop_late'):
                return model(image, **kwargs)
        _,prob = joint.stitch(stopped_forward,sample['image'][None].to(args.device),args.device)
        stopped.append(prob)
    for case, prob in zip(cases,stopped): case['stop_late'] = prob
    restore(model,args,args.output_dir/'best.pth')
    cases = infer(args,model,dataset,'choice',cases,statistics=True)
    if reference_digest(cases) != manifest['sha256']: raise ValueError('Fixed reference mismatch')
    curve = []
    for threshold in np.round(np.arange(.35,.751,.01),2):
        curve.append(joint.point_metrics(cases,'choice',float(threshold),manifest))
        atomic_json(args.output_dir/'choice_threshold_curve.json',curve)
    own = max(curve,key=lambda row:row['selection_score'])
    target_tp = sum(int(((case['current_S']>=common_threshold)&case['gt']).sum()) for case in cases)
    threshold,error = exact_recall_threshold(cases,'choice',target_tp)
    reports = {'current_S_common':joint.point_metrics(cases,'current_S',common_threshold,manifest),
        'stop_late_common':joint.point_metrics(cases,'stop_late',common_threshold,manifest),
        'choice_common':joint.point_metrics(cases,'choice',common_threshold,manifest),
        'choice_own_val':own,'choice_matched_recall':dict(joint.point_metrics(cases,'choice',threshold,manifest),tp_matching_error=error)}
    rows=[]; late=[]
    for name,value in reports.items():
        rows.append({'name':name,'threshold':value['threshold'],**{key:value['metrics'][key] for key in ('tp','fp','fn','iou','f1','precision','recall','cldice','apls_bidirectional_approx','short_pred_components')},
                     'closed_fixed_gaps':value['closed'],'bridge_proxy':value['bridge_proxy']})
    for name in ('current_S','choice'):
        recovered=added_fp=lost_gt=0
        for case in cases:
            pred=case[name]>=common_threshold; stop=case['stop_late']>=common_threshold; gt=case['gt']
            recovered+=int((pred&~stop&gt).sum()); added_fp+=int((pred&~stop&~gt).sum()); lost_gt+=int((~pred&stop&gt).sum())
        late.append({'model':name,'threshold':common_threshold,'gt_retained_beyond_current_stop_late':recovered,
                     'fp_beyond_current_stop_late':added_fp,'gt_lost_vs_current_stop_late':lost_gt})
    for i,case in enumerate(cases):
        np.savez_compressed(args.output_dir/f'val_probabilities_{i:02d}.npz',**{k:case[k] for k in ('p0','gt','current_S','stop_late','choice')})
    write_csv(args.output_dir/'comparison.csv',rows); write_csv(args.output_dir/'late_road_retention.csv',late)
    source_rows=[]
    for name in ('current_S','choice'):
        for key,row in read_json(args.output_dir/(name+'_source_statistics.json')).items():
            source_rows.append({'model':name,'step_region':key,**row,
                'weight_mean':row['weight_sum']/max(row['relations'],1),
                'positive_transport_mean':row['positive_transport_sum']/max(row['relations'],1)})
    write_csv(args.output_dir/'actual_source_relations.csv',source_rows)
    atomic_json(args.output_dir/'comparison.json',{'reports':reports,'late_retention':late,
        'reference_count':manifest['count'],'selection_rule':SELECTION_RULE,
        'limits':'This is extra training from current S. Without an equal-budget lambda_choice=0 continuation, changes do not isolate this loss causally. APLS, gaps and bridges are proxies.'})
    lines=['# Late wrong-choice continuation','',f'Common threshold: {common_threshold}. Fixed reference gaps: {manifest["count"]}.',
        f'Loss: Lseg + 0.2 Lcont + 0.1 Lrelation + {args.lambda_choice} Lchoice (unvalidated initial coefficient).',
        'Only steps 2/3 receive direct choice supervision. Native labels never enter inference.',
        '', '| Model | Threshold | FP | FN | IoU | clDice | Closed | Bridge proxy |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows: lines.append(f'| {row["name"]} | {row["threshold"]:.5f} | {row["fp"]} | {row["fn"]} | {row["iou"]:.6f} | {row["cldice"]:.6f} | {row["closed_fixed_gaps"]} | {row["bridge_proxy"]} |')
    lines+=['','Read actual_source_relations.csv for supported/blocked actual weight and positive transport at each step. Read late_road_retention.csv for current stop_late-referenced GT retention and FP. The previously reported 38,128/60,667 are not hardcoded as achieved outcomes.',
            'Merely moving toward stop_late is not evidence of successful selectivity. Check matched Recall, true-road retention and bridges together.',
            'Extra continuation training confounds loss-only attribution unless compared with a same-start, same-budget lambda_choice=0 continuation. No B_control is retrained here.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('smoke','train','val','all'))
    parser.add_argument('--run_dir',type=Path,required=True); parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path); parser.add_argument('--threshold',type=float)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--lambda_choice',type=float,default=.05)
    args=parser.parse_args(); args.device=torch.device(args.device)
    if args.epochs<1 or args.lambda_choice<0: parser.error('Invalid training budget/coefficient')
    if args.output_dir.resolve()==args.run_dir.resolve(): parser.error('Output must be a new continuation directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model,state,path,manifest,threshold,origin=load_current(args)
    source=state['config']
    if abs(source['lambda_relation']-.1)>1e-12 or abs(source['source_losses']['lambda_cont']-.2)>1e-12:
        raise ValueError('Existing losses differ from required .1 relation / .2 continuity; refusing a simultaneous coefficient change')
    args.root_path=Path(source['root_path']); args.seed=source['seed']; args.num_workers=source['num_workers']
    args.val_interval=source['val_interval']; args.labels=RelationLabels(**source['labels'])
    args.source_losses=source['source_losses']; args.identity=state['identity']; args.source_epoch=state['epoch']
    rates={key:value['lr'] for key,value in source['optimizer'].items()}
    rates['gate_relation']=rates['gate_feature_projections_and_pairs']
    optimizer,groups=configure_joint(model,rates,rates['selector'])
    args.config={'version':VERSION,'source_checkpoint':str(path.resolve()),'source_sha256':v9.digest(path),
        'source_epoch':args.source_epoch,'identity':args.identity,'source_config':source,
        'epochs':args.epochs,'lambda_choice':args.lambda_choice,'lambda_relation':.1,'lambda_cont':.2,
        'choice_steps':[2,3],'optimizer':groups,'optimizer_initialization':'fresh AdamW, same LR and trainable groups; current best has no optimizer',
        'common_threshold':threshold,'threshold_origin':origin,'reference_file_sha256':v9.digest(args.run_dir/'fixed_gaps_and_anchors.json'),
        'code_sha256':{name:v9.digest(Path(__file__).parent/name) for name in ('run_v9_late_choice.py','networks/v9_late_choice_loss.py')}}
    config_path=args.output_dir/'experiment.json'
    if config_path.is_file() and read_json(config_path)!=args.config: raise ValueError('Continuation config/source changed')
    atomic_json(config_path,args.config)
    for filename in ('fixed_gaps_and_anchors.json','fixed_reference_binding.json'):
        shutil.copyfile(args.run_dir/filename,args.output_dir/filename)
    if args.command in ('smoke','all','train') and not (args.output_dir/'choice_smoke.json').is_file(): smoke(args,model,optimizer)
    if args.command=='smoke': return
    dataset=validation_cases(args)
    if args.command in ('all','train'): train(args,model,optimizer,dataset,manifest)
    if args.command in ('all','val'): compare(args,model,state,dataset,manifest,threshold)
    if v9.digest(path)!=args.config['source_sha256']: raise RuntimeError('Input S_selector checkpoint changed')


if __name__=='__main__': main()
