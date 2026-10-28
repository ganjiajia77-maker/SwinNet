"""Independent V9 continuation: control versus matched transport-gate BCE.

Only training GT is used for candidate labels. Validation uses the original
V9 score and fixed P0 gap reference. Original checkpoint and code stay intact.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_reliable_v9_mass as v9
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.reliable_directional_repair import dynamic_losses
from networks.v91_gate_candidates import gate_candidates


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_v9(args):
    checkpoint = args.run_dir/'stage_D/best.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing original V9 checkpoint: {checkpoint}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['identity'].get('initialization') != 'direct_random_downstream':
        raise ValueError('Expected the direct-trained V9 checkpoint')
    if Path(payload['training_config']['root_path']).resolve() != args.root_path.resolve():
        raise ValueError('Training data root differs from V9')
    model, source, identity = v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(payload['identity']['q3']['path']), device=args.device))
    v9.restore_checkpoint(model, checkpoint, identity, 'D')
    return model, source, identity, payload, checkpoint


def training_dataset(args, augment):
    return RoadSkeletonDataset(str(args.root_path), 'train', image_size=512, source_patch_size=1500,
        random_crop_train=True, random_crops_per_image=1, random_crop_seed=args.seed, augment=augment)


def candidate_loss(result, batch, source, args, seed, include_samples=False):
    return gate_candidates(result, batch['mask'].to(args.device), batch['skeleton'].to(args.device),
        threshold=source['missing_threshold'], min_action=args.min_action,
        near_gt_work_pixels=args.near_gt_work_pixels, max_pairs_per_stratum=args.max_pairs,
        max_pairs_per_step=args.max_pairs_per_step, seed=seed, include_samples=include_samples,
        output_threshold=args.output_threshold)


def aggregate(reports):
    counters = ('executable_candidates', 'positive_before_matching', 'negative_before_matching',
                'matched_positive', 'matched_negative')
    out = {key: sum(row[key] for row in reports) for key in counters}
    out['batches'] = len(reports)
    out['no_matched_positive_batches'] = sum(row['matched_positive'] == 0 for row in reports)
    details = [detail for row in reports for detail in row['details']]
    for key in ('negative_near_gt_before', 'negative_high_q_before',
                'negative_bridge_proxy_before', 'negative_isolated_proxy_before',
                'negative_bridge_proxy_matched', 'negative_isolated_proxy_matched'):
        out[key] = sum(row[key] for row in details)
    out['gt_missing_segment_coverage'] = {length: {key: sum(row['gt_missing_segment_coverage'][length][key]
        for row in reports) for key in ('missing_segments', 'covered_before', 'covered_after')}
        for length in ('3-7', '8-15', '16-31', '32+')}
    total = out['matched_positive']
    positive_hist = np.array([r['g_positive_histogram'] for r in reports], dtype=np.int64).sum(axis=0)
    negative_hist = np.array([r['g_negative_histogram'] for r in reports], dtype=np.int64).sum(axis=0)
    if total and out['matched_negative']:
        tp = positive_hist[::-1].cumsum()/total
        fp = negative_hist[::-1].cumsum()/out['matched_negative']
        precision = positive_hist[::-1].cumsum()/np.maximum(
            positive_hist[::-1].cumsum()+negative_hist[::-1].cumsum(), 1)
        tpr, fpr = np.r_[0., tp], np.r_[0., fp]
        out['matched_G_roc_auc_hist_approx'] = float(np.sum((tpr[:-1]+tpr[1:])*.5*np.diff(fpr)))
        out['matched_G_pr_auc_hist_approx'] = float(np.sum(np.diff(np.r_[0., tp])*precision))
        mid = (np.arange(100)+.5)/100.
        out['matched_G_positive_mean_hist_approx'] = float((positive_hist*mid).sum()/total)
        out['matched_G_negative_mean_hist_approx'] = float((negative_hist*mid).sum()/out['matched_negative'])
    else:
        out.update(matched_G_roc_auc_hist_approx=None, matched_G_pr_auc_hist_approx=None,
                   matched_G_positive_mean_hist_approx=None, matched_G_negative_mean_hist_approx=None)
    out['weighted_mean_absolute_pair_differences'] = {
        key: (sum(row['mean_absolute_pair_differences'][key]*row['matched_positive']
                  for row in reports if row['matched_positive'])/total if total else None)
        for key in ('v', 'potential_action', 'realized_action',
                    'prediction_distance_work', 'component_area_work')}
    out['limitations'] = ('Matched at pixel/action level by binned V, pre-gate potential and '
        'P0-road distance and action-component area; action-component area is not path length. '
        'GT skeleton hit is a coverage proxy, not true positive path-pair recall. '
        'Parallel-road subtypes are not automatically certified; bridge and isolated labels are component proxies. '
        'Training candidate PR-AUC uses a balanced matched sample, not natural prevalence; '
        'both AUC values are 100-bin approximations.')
    return out


def audit_candidates(args, model, source, identity, checkpoint):
    dataset = training_dataset(args, augment=False); dataset.set_epoch(1)
    count = min(len(dataset), args.max_images or len(dataset))
    loader = DataLoader(torch.utils.data.Subset(dataset, list(range(count))), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f'Candidate output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_reports = []; rows = []
    sample_path = args.output_dir/'matched_candidates_train.csv'
    model.eval()
    with torch.no_grad(), sample_path.open('w', newline='', encoding='utf-8') as sample_file:
        sample_fields = ['case', 'step', 'stratum', 'pair_number', 'label', 'y_work', 'x_work',
            'availability', 'potential_transport_effect', 'realized_transport_effect',
            'prediction_distance_work', 'component_area_work',
            'near_gt', 'high_q', 'false_bridge_proxy', 'isolated_fp_proxy', 'g']
        sample_writer = csv.DictWriter(sample_file, fieldnames=sample_fields)
        sample_writer.writeheader()
        for index, batch in enumerate(loader):
            result = model(batch['image'].to(args.device), capture_diagnostics=True)
            _, report = candidate_loss(result, batch, source, args, args.seed+index, include_samples=True)
            for item in report.pop('samples'):
                item['case'] = batch['case_name'][item.pop('batch_index')]
                sample_writer.writerow(item)
            all_reports.append(report)
            for row in report['details']:
                case = batch['case_name'][row['batch_index']]
                rows.append({'case': case, **row})
            print(f'[CANDIDATES] batch={index+1}/{len(loader)} '
                  f'pos={report["positive_before_matching"]} neg={report["negative_before_matching"]} '
                  f'matched_pairs={report["matched_positive"]}', flush=True)
    summary = aggregate(all_reports)
    summary.update({'source_checkpoint': str(checkpoint), 'source_sha256': v9.digest(checkpoint),
        'identity': identity, 'split': 'train', 'images': count, 'random_crop_seed': args.seed,
        'crop_epoch': 1, 'augmented': False, 'min_action': args.min_action,
        'near_gt_work_pixels': args.near_gt_work_pixels, 'max_pairs': args.max_pairs,
        'max_pairs_per_step': args.max_pairs_per_step,
        'output_threshold': args.output_threshold,
        'batch_size': args.batch_size,
        'matching': 'within image and step; V, positive transport effect, P0 distance and component area bins'})
    write_json(args.output_dir/'candidates_train.json', summary)
    with (args.output_dir/'candidates_train.csv').open('w', newline='', encoding='utf-8') as f:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


def save(path, model, identity, epoch, score, optimizer, config, initial_sha):
    checkpoint = {'version': v9.VERSION, 'identity': identity, 'stage': 'D', 'epoch': epoch, 'score': score,
        'modules': {name: getattr(model, name).state_dict() for name in v9.MODULE_NAMES},
        'gate_relation': model.gate_relation.state_dict(), 'state_gate': model.state_gate.state_dict(),
        'training_config': model.v9_training_config, 'learning_rates': dict(v9.LR),
        'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'v91_config': config, 'v91_initial_checkpoint_sha256': initial_sha}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    temp = path.with_suffix('.tmp'); torch.save(checkpoint, temp); temp.replace(path)


def train(args, model, source, identity, baseline, checkpoint):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    if args.max_images:
        raise ValueError('Training and validation must use all images')
    if args.candidate_report is None or not args.candidate_report.is_file():
        raise ValueError('Run the complete train-set candidate audit and pass --candidate_report')
    candidate_report = json.loads(args.candidate_report.read_text())
    initial_sha = v9.digest(checkpoint)
    dataset = training_dataset(args, augment=True)
    expected = {'source_sha256': initial_sha, 'split': 'train', 'images': len(dataset),
        'random_crop_seed': args.seed, 'min_action': args.min_action,
        'near_gt_work_pixels': args.near_gt_work_pixels, 'max_pairs': args.max_pairs,
        'max_pairs_per_step': args.max_pairs_per_step,
        'output_threshold': args.output_threshold,
        'batch_size': args.batch_size}
    mismatches = {k: [candidate_report.get(k), value] for k, value in expected.items()
                  if candidate_report.get(k) != value}
    if mismatches or candidate_report.get('matched_positive', 0) == 0:
        raise ValueError(f'Full matching candidate audit missing, incompatible, or no matched pairs: {mismatches}')
    if args.seed != baseline['training_config']['seed'] or args.batch_size != baseline['training_config']['batch_size']:
        raise ValueError('Continuation must preserve V9 seed and physical batch size')
    out = args.output_dir/args.variant
    resuming = getattr(args, 'command', 'train') == 'resume'
    if not resuming and out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory exists; use a new directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    resume = None
    if resuming:
        last = out/'last.pth'
        if not last.is_file():
            raise ValueError(f'No last.pth for resume: {last}')
        resume = v9.restore_checkpoint(model, last, identity, 'D')
        if resume.get('v91_initial_checkpoint_sha256') != initial_sha:
            raise ValueError('Resume checkpoint belongs to a different original V9')
    model.v9_training_config = dict(baseline['training_config'])
    v9.LR.update(baseline['learning_rates'])
    optimizer = v9.configure(model, 'D')
    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    config = {'variant': args.variant, 'seed': seed, 'batch_size': args.batch_size,
        'epochs': args.epochs, 'val_interval': args.val_interval, 'lambda_gate': args.lambda_gate,
        'min_action': args.min_action, 'near_gt_work_pixels': args.near_gt_work_pixels,
        'output_threshold': args.output_threshold,
        'max_pairs': args.max_pairs, 'max_pairs_per_step': args.max_pairs_per_step,
        'same_initial_checkpoint': str(checkpoint),
        'candidate_report': str(args.candidate_report.resolve()),
        'candidate_report_sha256': v9.digest(args.candidate_report),
        'data_augmentation': True, 'lr_groups': dict(v9.LR),
        'continuation_note': 'Both control and gate use a fresh AdamW starting from the identical trained V9 checkpoint.'}
    first_epoch, best = 1, -float('inf')
    if resume:
        old_config = resume['v91_config']
        if {k: v for k, v in old_config.items() if k != 'epochs'} != {k: v for k, v in config.items() if k != 'epochs'}:
            raise ValueError('Resume settings differ from the saved continuation')
        if args.epochs < resume['epoch']:
            raise ValueError('Requested total epochs precede the saved epoch')
        if resume.get('learning_rates') != v9.LR or 'optimizer' not in resume:
            raise ValueError('Resume optimizer or learning rates differ')
        optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['torch_rng']); np.random.set_state(resume['numpy_rng'])
        if resume.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        first_epoch, best = resume['epoch']+1, resume['score']
    else:
        write_json(out/'config.json', config)
    for epoch in range(first_epoch, args.epochs+1):
        model.train(); model.set_accept_floor(0.); dataset.set_epoch(epoch)
        reports = []; totals = dict(total=0., seg=0., cont=0., gate=0.)
        gate_grad_norms = []
        for index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            image, mask, skeleton = (batch[key].to(args.device) for key in ('image', 'mask', 'skeleton'))
            result = model(image, capture_diagnostics=True)
            surface_loss, terms = dynamic_losses(result, mask, skeleton,
                **{k: source[k] for k in ('lambda_bg', 'lambda_missing', 'lambda_cont', 'missing_threshold')},
                return_live_terms=True)
            gate_loss, report = candidate_loss(result, batch, source, args, seed+epoch*100000+index)
            reports.append(report)
            actual = surface_loss + (args.lambda_gate * gate_loss if args.variant == 'gate' else 0.)
            if not bool(torch.isfinite(actual)):
                raise RuntimeError('Nonfinite loss')
            if args.variant == 'gate' and index == 0:
                g = torch.autograd.grad(gate_loss, tuple(model.state_gate.parameters()), retain_graph=True,
                                        allow_unused=True)
                gate_grad_norms.append(float(torch.stack([x.detach().square().sum() for x in g if x is not None]).sum().sqrt())
                    if any(x is not None for x in g) else 0.)
            actual.backward()
            if any(p.grad is not None for p in model.frozen.parameters()):
                raise RuntimeError('Frozen BASE or Q3 received gradients')
            optimizer.step()
            totals['total'] += float(actual.detach()); totals['seg'] += float(terms['seg'])
            totals['cont'] += float(terms['cont']); totals['gate'] += float(gate_loss.detach())
        summary = aggregate(reports)
        summary.update({'epoch': epoch, 'variant': args.variant,
                        **{k: val/max(len(loader), 1) for k, val in totals.items()},
                        'gate_loss_gradient_to_state_gate_first_batch': gate_grad_norms[0] if gate_grad_norms else None})
        with (out/'epochs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(summary, allow_nan=False)+'\n')
        print('[TRAIN] '+json.dumps(summary), flush=True)
        if args.variant == 'gate' and summary['matched_positive'] == 0:
            raise RuntimeError('No matched training candidates in this epoch; inspect epochs.jsonl and candidate audit')
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            values, _, _, _ = evaluate_thresholds(model, full_dataset(args.root_path, 'val'), args.device,
                (.6,), reference_threshold=float(baseline['training_config']['reference_threshold']))
            report = values[.6]; score = report['selection_score']
            write_json(out/f'val_epoch_{epoch:03d}.json', report)
            if score > best:
                best = score
                save(out/'best.pth', model, identity, epoch, best, None, config, initial_sha)
            print(f'[VAL] {args.variant} epoch={epoch} score={score:.6f}', flush=True)
        save(out/'last.pth', model, identity, epoch, best, optimizer, config, initial_sha)
    if v9.digest(checkpoint) != initial_sha:
        raise RuntimeError('Original V9 checkpoint changed during continuation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('candidates', 'train', 'resume'))
    p.add_argument('--run_dir', type=Path, required=True); p.add_argument('--root_path', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--variant', choices=('control', 'gate'), default='control')
    p.add_argument('--candidate_report', type=Path)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--val_interval', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=2); p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234); p.add_argument('--max_images', type=int, default=0)
    p.add_argument('--lambda_gate', type=float, default=.1); p.add_argument('--min_action', type=float, default=1e-5)
    p.add_argument('--near_gt_work_pixels', type=float, default=8.); p.add_argument('--max_pairs', type=int, default=256)
    p.add_argument('--max_pairs_per_step', type=int, default=256)
    p.add_argument('--output_threshold', type=float, default=.68)
    args = p.parse_args()
    if (args.epochs < 1 or args.val_interval < 1 or args.batch_size < 1 or args.num_workers < 0 or
            args.max_images < 0 or args.lambda_gate < 0 or not 0 < args.min_action < .001 or
            args.near_gt_work_pixels <= 0 or args.max_pairs < 1 or args.max_pairs_per_step < 1 or
            not 0 < args.output_threshold < 1):
        p.error('Invalid experiment configuration')
    args.device = torch.device(args.device); torch.manual_seed(args.seed); np.random.seed(args.seed)
    model, source, identity, baseline, checkpoint = load_v9(args)
    if args.command == 'candidates':
        audit_candidates(args, model, source, identity, checkpoint)
    else:
        train(args, model, source, identity, baseline, checkpoint)


if __name__ == '__main__':
    main()
