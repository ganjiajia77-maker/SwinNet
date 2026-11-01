"""Fixed-training-batch startup audit for the existing complete H0 + source selector.

This diagnostic starts from the original H0 and validated repair checkpoints,
changes only the rates output initialization and runs short in-memory updates.
No checkpoint or held-out prediction is written, and GT never enters forward.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from networks.full_h0_joint import FullH0Joint, original_criterion, parameter_groups
from run_full_h0_joint import (DEFAULT_H0, DEFAULT_REPAIR, batch_loss, inspect_source,
                               load_h0, training_dataset)
from run_slim_end_to_end_mass import digest, schedule, seed_all
from run_v9_gate_benefit import atomic_json


def initialize_rates(model, transport, proposal):
    """Initialize 4-way softmax budgets exactly; anchor/stay retain their ratio."""
    if not (0. < transport < 1. and 0. < proposal < 1. and transport + proposal < 1.):
        raise ValueError('transport and proposal must be positive and sum to less than one')
    head = model.original.dynamics.rates[-1]
    if head.out_channels != 4 or head.bias is None:
        raise ValueError('Expected the four-channel rate output layer')
    stay, anchor = 2.5, 1.
    residual_mass = math.exp(stay) + math.exp(anchor)
    remaining = 1. - transport - proposal
    logits = [stay, math.log(residual_mass * transport / remaining),
              math.log(residual_mass * proposal / remaining), anchor]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(head.bias.new_tensor(logits))
    return {'transport': transport, 'proposal': proposal, 'logits': logits,
            'proposal_note': 'effective beta also multiplies 0.2 and the learned acceptance A'}


def segmentation_counts(probability, gt, threshold):
    prediction = probability >= threshold
    road = gt > .5
    tp = int((prediction & road).sum())
    fp = int((prediction & ~road).sum())
    fn = int((~prediction & road).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'iou': tp / max(tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1)}


def differences(p0, p1, gt, initial_p0, threshold):
    current = p0 >= threshold
    repaired = p1 >= threshold
    road = gt > .5
    added = ~current & repaired
    removed = current & ~repaired
    repair_delta = p1 - p0
    base_delta = p0 - initial_p0
    return {
        'P1_minus_P0_mean': float(repair_delta.mean()),
        'P1_minus_P0_abs_mean': float(repair_delta.abs().mean()),
        'P1_minus_P0_abs_max': float(repair_delta.abs().max()),
        'P1_P0_flipped_pixels': int((added | removed).sum()),
        'repair_added_tp': int((added & road).sum()),
        'repair_lost_tp': int((removed & road).sum()),
        'repair_added_fp': int((added & ~road).sum()),
        'repair_removed_fp': int((removed & ~road).sum()),
        'P0_change_from_initial_abs_mean': float(base_delta.abs().mean()),
        'P0_change_from_initial_abs_max': float(base_delta.abs().max()),
        'P0_changed_binary_pixels': int(((p0 >= threshold) != (initial_p0 >= threshold)).sum()),
        'P0': segmentation_counts(p0, gt, threshold),
        'P1': segmentation_counts(p1, gt, threshold)}


@torch.no_grad()
def evaluate(model, batch, initial_p0, threshold):
    model.eval()
    raw_rates = []
    def capture(unused, inputs, output):
        raw_rates.append(output.softmax(1).detach())
    handle = model.original.dynamics.rates.register_forward_hook(capture)
    try:
        result = model(batch['image'], capture_diagnostics=True)
    finally:
        handle.remove()
    if len(raw_rates) != 3:
        raise RuntimeError(f'Expected three dynamic steps, got {len(raw_rates)}')
    rate = torch.stack(raw_rates, 1)
    p0, p1, gt = (result['p0'].detach().cpu(), result['p1'].detach().cpu(), batch['mask'].detach().cpu())
    if initial_p0 is None:
        initial_p0 = p0.clone()
    if not all(torch.isfinite(value).all() for value in (p0, p1, rate, result['transport_delta'])):
        raise RuntimeError('Nonfinite probability, rate, or transport')
    data = differences(p0, p1, gt, initial_p0, threshold)
    data.update({
        'transport_rate_mean': float(rate[:, :, 1].mean()),
        'proposal_rate_mean': float(rate[:, :, 2].mean()),
        'proposal_coefficient_mean': float(result['beta'].mean()),
        'transport_abs_mean': float(result['transport_delta'].abs().mean()),
        'transport_abs_max': float(result['transport_delta'].abs().max()),
        'transport_signed_mean': float(result['transport_delta'].mean()),
        'proposal_abs_mean': float(result['proposal_delta'].abs().mean()),
        'anchor_abs_mean': float(result['anchor_delta'].abs().mean()),
        'steps': [{'step': k + 1,
                   'transport_rate_mean': float(rate[:, k, 1].mean()),
                   'transport_abs_mean': float(result['transport_delta'][:, k].abs().mean()),
                   'transport_signed_mean': float(result['transport_delta'][:, k].mean()),
                   'state_change_abs_mean': float((result['states'][k+1] - result['states'][k]).abs().mean())}
                  for k in range(3)]})
    return data, initial_p0


def write_results(folder, metadata, history):
    atomic_json(folder / 'startup_report.json', {'metadata': metadata, 'history': history})
    fields = ['update', 'loss', 'transport_rate_mean', 'proposal_rate_mean', 'proposal_coefficient_mean',
              'transport_abs_mean', 'transport_abs_max', 'transport_signed_mean', 'proposal_abs_mean',
              'anchor_abs_mean', 'P1_minus_P0_abs_mean', 'P1_minus_P0_abs_max', 'P1_P0_flipped_pixels',
              'repair_added_tp', 'repair_lost_tp', 'repair_added_fp', 'repair_removed_fp',
              'P0_change_from_initial_abs_mean', 'P0_change_from_initial_abs_max', 'P0_changed_binary_pixels',
              'P0_tp', 'P0_fp', 'P0_fn', 'P0_iou', 'P1_tp', 'P1_fp', 'P1_fn', 'P1_iou']
    temporary = folder / 'startup_metrics.csv.tmp'
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in history:
            flat = dict(record)
            for prefix in ('P0', 'P1'):
                flat.update({prefix + '_' + key: value for key, value in record[prefix].items()})
            writer.writerow(flat)
    temporary.replace(folder / 'startup_metrics.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    parser.add_argument('--h0_checkpoint', type=Path, default=Path(DEFAULT_H0))
    parser.add_argument('--repair_run', type=Path, default=Path(DEFAULT_REPAIR))
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--report_every', type=int, default=25)
    parser.add_argument('--transport_rate', type=float, default=.01)
    parser.add_argument('--proposal_rate', type=float, default=.001)
    parser.add_argument('--threshold', type=float, default=.45)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--h0_lr', type=float, default=5e-5)
    args = parser.parse_args()
    if args.steps < 1 or args.report_every < 1 or not 0. < args.threshold < 1.:
        parser.error('steps/report_every must be positive and threshold must be in (0,1)')
    device = torch.device(args.device)
    seed_all(args.seed)
    base, h0, saved = load_h0(args.h0_checkpoint, device)
    source, q, source_meta = inspect_source(args.repair_run)
    source_config = source['config']
    model = FullH0Joint(base, q, source).to(device)
    rate_init = initialize_rates(model, args.transport_rate, args.proposal_rate)
    criterion = original_criterion(saved, device)
    config = {'source_training': source_config, 'lambda_relation': source_config['lambda_relation']}
    loss_args = argparse.Namespace(variant='B', config=config)
    lrs = {key: value['lr'] for key, value in source_config.get('optimizer', {}).items()}
    optimizer = parameter_groups(model, args.h0_lr, repair_rates=lrs)
    # Match the first epoch's warmup rates from the paired 20-epoch setup.
    schedule(optimizer, 0, 20)
    dataset = training_dataset(args, augment=True)
    dataset.set_epoch(1)
    generator = torch.Generator().manual_seed(args.seed + 1000003)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, generator=generator)))
    image_hash = hashlib.sha256(batch['image'].contiguous().numpy().tobytes()).hexdigest()
    case_names = [str(value) for value in batch['case_name']]
    batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
    metadata = {'scope': 'training-set fixed batch, diagnostic only; not validation or model selection',
                'source_h0_checkpoint': str(args.h0_checkpoint.resolve()), 'source_h0_epoch': h0['epoch'],
                'source_h0_sha256': digest(args.h0_checkpoint),
                'source_repair': source_meta, 'training_cases': case_names, 'image_sha256': image_hash,
                'image_shape': list(batch['image'].shape), 'seed': args.seed, 'steps': args.steps,
                'report_every': args.report_every, 'threshold': args.threshold, 'rate_initialization': rate_init,
                'learning_rates': {group['name']: group['lr'] for group in optimizer.param_groups},
                'weights': 'updated in memory only; no training checkpoint is written',
                'baseline': 'step 0 P0 from the same starting H0 in eval mode; all points use eval mode',
                'loss': 'original H0 loss + existing repair losses + existing relation loss',
                'repair_loss_config': source_config['source_losses'],
                'lambda_relation': source_config['lambda_relation'],
                'warnings': ['Repeated training on one batch can overfit; no conclusion about validation quality.']}
    del h0, source, q
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    initial, initial_p0 = evaluate(model, batch, None, args.threshold)
    initial.update(update=0, loss=None)
    history = [initial]
    write_results(args.output_dir, metadata, history)
    print('[STARTUP] ' + json.dumps(initial, ensure_ascii=False), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        torch.manual_seed(args.seed + step)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch['image'])
        total, _ = batch_loss(loss_args, model, criterion, result, batch, args.seed + step)
        if not torch.isfinite(total):
            raise RuntimeError(f'Nonfinite loss at update {step}')
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise RuntimeError(f'Nonfinite gradient at update {step}')
        optimizer.step()
        current_loss = float(total.detach())
        del total, result
        if step == 1 or step % args.report_every == 0 or step == args.steps:
            point, _ = evaluate(model, batch, initial_p0, args.threshold)
            point.update(update=step, loss=current_loss, gradient_norm_before_clip=float(grad),
                         elapsed_seconds=time.perf_counter() - start)
            if device.type == 'cuda':
                point['peak_cuda_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
            history.append(point)
            write_results(args.output_dir, metadata, history)
            print('[STARTUP] ' + json.dumps(point, ensure_ascii=False), flush=True)
    print('[DONE] ' + str(args.output_dir / 'startup_report.json'), flush=True)


if __name__ == '__main__':
    main()
