"""Train a zero-initialized semantic-relation residual on a frozen v7 model."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.reliable_directional_repair import MultiScaleGateRelation, dynamic_losses
from run_q3_r3_mass import full_dataset
from run_reliable_directional_mass import (
    V7_VERSION, evaluate_thresholds, load_model as load_v7, write_evaluation,
)


VERSION = 'reliable_directional_dynamics_v8_source_available_gate'
MODES = ('local', 'bilateral', 'directional', 'full')


def load_experiment(args):
    v7_path = args.v7_checkpoint.resolve()
    source = torch.load(v7_path, map_location='cpu', weights_only=False)
    if source.get('version') != V7_VERSION:
        raise ValueError('The initialization checkpoint must be trained v7')
    base = Path(getattr(args, 'base_checkpoint_override', None) or
                source['base_checkpoint']).resolve()
    q3 = Path(getattr(args, 'q3_checkpoint_override', None) or
              source['q3_checkpoint']).resolve()
    if not base.is_file() or not q3.is_file():
        raise FileNotFoundError(f'Missing frozen model checkpoints: {base}, {q3}')
    config = SimpleNamespace(
        mode='val', dynamic_checkpoint=v7_path, init_dynamic_checkpoint=None,
        base_checkpoint=base, q3_checkpoint=q3, device=args.device,
        width=source['width'], accept_floor_initial=source['accept_floor_initial'],
        accept_floor_anneal_epochs=source['accept_floor_anneal_epochs'],
        missing_threshold=source['missing_threshold'], lambda_bg=source['lambda_bg'],
        lambda_missing=source['lambda_missing'], lambda_cont=source['lambda_cont'],
        transport_gate_mode='v7', allow_checkpoint_source_override=(
            getattr(args, 'base_checkpoint_override', None) is not None or
            getattr(args, 'q3_checkpoint_override', None) is not None))
    model = load_v7(config)
    model.dynamics.transport_availability_power = float(
        getattr(args, 'transport_availability_power', 1.))
    stage = model.frozen.base.swin_unet.decoder_structure_blocks['3']
    decoder_channels = stage.structure_branch[0].block[0].in_channels
    highres_channels = model.frozen.base.swin_unet.highres_structure_channels
    model.requires_grad_(False)
    model.gate_relation = MultiScaleGateRelation(
        highres_channels, decoder_channels, source['width']).to(args.device)
    model.gate_relation_mode = args.relation_mode
    if args.v8_checkpoint:
        payload = torch.load(args.v8_checkpoint, map_location='cpu', weights_only=False)
        expected = {'version': VERSION, 'v7_checkpoint': str(v7_path),
                    'relation_mode': args.relation_mode,
                    'transport_availability_power':
                        model.dynamics.transport_availability_power}
        errors = {key: (payload.get(key), value) for key, value in expected.items()
                  if payload.get(key) != value}
        if errors:
            raise ValueError(f'v8 checkpoint mismatch: {errors}')
        model.gate_relation.load_state_dict(payload['gate_relation'], strict=True)
        print(f'[LOAD] v8 epoch={payload["epoch"]}, strict relation state', flush=True)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[PARAMS] frozen={frozen:,} new_trainable={trainable:,} '
          f'gate_mode={args.relation_mode}', flush=True)
    return model, source


def save(model, args, epoch, optimizer, best_score):
    payload = {'version': VERSION, 'epoch': epoch,
               'v7_checkpoint': str(args.v7_checkpoint.resolve()),
               'relation_mode': args.relation_mode,
               'transport_availability_power':
                   model.dynamics.transport_availability_power,
               'gate_relation': model.gate_relation.state_dict(),
               'optimizer': optimizer.state_dict(), 'best_score': best_score}
    target = args.output_dir / 'last.pth'
    temp = target.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(target)


def smoke(args, model):
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
                                  source_patch_size=1500, random_crop_train=True,
                                  random_crops_per_image=1)
    sample = dataset[0]
    image = sample['image'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    mask = sample['mask'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    skeleton = sample['skeleton'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    model.eval()
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    with torch.no_grad():
        v8 = model(image, capture_diagnostics=True)
        frozen_outputs = model.frozen(image[:1], capture_diagnostics=True)
        saved = model.gate_relation
        model.gate_relation = None
        try:
            v7 = model(image)
        finally:
            model.gate_relation = saved
        baseline_error = float((v8['p1'] - v7['p1']).abs().max())
        if baseline_error > 1e-7:
            raise RuntimeError(f'Zero-initialized v8 differs from v7: {baseline_error}')
        with torch.no_grad():
            model.gate_relation.delta[-1].bias.add_(.5)
            perturbed = model(image)
            model.gate_relation.delta[-1].bias.sub_(.5)
        for key in ('r', 'w', 'b', 'a'):
            if not torch.equal(perturbed[key], v7[key]):
                raise RuntimeError(f'Gate-only perturbation modified {key}')
        gate_effect = float((perturbed['g_transport'] - v7['g_transport']).abs().max())
        surface_effect = float((perturbed['p1'] - v7['p1']).abs().max())
        if gate_effect <= 0 or surface_effect <= 0:
            raise RuntimeError('Gate delta did not reach G and Surface')
    model.train()
    model.set_accept_floor(0.)
    result = model(image)
    loss, terms = dynamic_losses(
        result, mask, skeleton, lambda_bg=args.lambda_bg,
        lambda_missing=args.lambda_missing, lambda_cont=args.lambda_cont,
        missing_threshold=args.missing_threshold)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss')
    loss.backward()
    gradients = {}
    for name in ('highres_project', 'decoder_project', 'local', 'context', 'pair', 'delta'):
        module = getattr(model.gate_relation, name)
        squared = [p.grad.detach().square().sum() for p in module.parameters()
                   if p.grad is not None]
        gradients[name] = float(torch.stack(squared).sum().sqrt()) if squared else 0.
    gradients['gate_relation'] = sum(gradients.values())
    if not gradients['gate_relation'] > 0:
        raise RuntimeError('No gradient reached the new gate relation')
    if any(p.grad is not None for name, module in model.named_children()
           if name != 'gate_relation' for p in module.parameters()):
        raise RuntimeError('A frozen v7 module received gradients')
    if not torch.isfinite(result['p1']).all():
        raise RuntimeError('Nonfinite surface prediction')
    report = {'mode': args.relation_mode, 'v7_identity_max_abs': baseline_error,
              'gate_perturbation_max_abs': gate_effect,
              'surface_perturbation_max_abs': surface_effect,
              'condition_shapes': {key: list(frozen_outputs[key].shape) for key in
                                   ('z0', 'q', 'highres_h', 'stage3_h')},
              'shapes': {key: list(result[key].shape) for key in
                         ('p0', 'q', 'r', 'w', 'g_transport', 'p1')},
              'loss': float(loss), 'L_seg': float(terms['seg']),
              'L_cont': float(terms['cont']), 'gradients': gradients,
              'peak_cuda_mib': (torch.cuda.max_memory_allocated(args.device) / 2**20
                                if args.device.type == 'cuda' else None)}
    print(json.dumps(report, indent=2), flush=True)


def train(args, model):
    if args.command == 'train':
        args.output_dir.mkdir(parents=True, exist_ok=False)
        first_epoch, best_score = 1, -float('inf')
    else:
        payload = torch.load(args.v8_checkpoint, map_location='cpu', weights_only=False)
        first_epoch, best_score = payload['epoch'] + 1, payload['best_score']
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
                                  source_patch_size=1500, random_crop_train=True,
                                  random_crops_per_image=1, random_crop_seed=args.seed,
                                  augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.gate_relation.parameters(), lr=args.lr,
                                  weight_decay=.01)
    if args.command == 'resume':
        optimizer.load_state_dict(payload['optimizer'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        model.set_accept_floor(0.)
        dataset.set_epoch(epoch)
        total = seg = cont = 0.
        missing = hard_bg = 0
        gate_missing_sum = gate_background_sum = 0.
        gate_missing_count = gate_background_count = 0
        start = time.perf_counter()
        for batch in loader:
            image = batch['image'].to(args.device, non_blocking=True)
            mask = batch['mask'].to(args.device, non_blocking=True)
            skeleton = batch['skeleton'].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = model(image)
            loss, terms = dynamic_losses(
                result, mask, skeleton, lambda_bg=args.lambda_bg,
                lambda_missing=args.lambda_missing, lambda_cont=args.lambda_cont,
                missing_threshold=args.missing_threshold)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            seg += float(terms['seg'])
            cont += float(terms['cont'])
            missing += terms['missing_weighted_pixels']
            hard_bg += terms['hard_background_weighted_pixels']
            with torch.no_grad():
                missed = (skeleton > .5) & (
                    F.max_pool2d(result['p0'], 5, stride=1, padding=2) <
                    args.missing_threshold)
                missed_work = F.adaptive_max_pool2d(
                    missed.float(), result['g_transport'].shape[-2:]) > 0
                background_work = (F.interpolate(mask, size=result['q'].shape[-2:],
                                                  mode='nearest') < .5) & (result['q'] > .5)
                gate_missing_sum += float(result['g_transport'][missed_work].sum())
                gate_background_sum += float(result['g_transport'][background_work].sum())
                gate_missing_count += int(missed_work.sum())
                gate_background_count += int(background_work.sum())
        report = {'epoch': epoch, 'loss': total/len(loader),
                  'L_seg': seg/len(loader), 'L_cont': cont/len(loader),
                  'missing_weighted_pixels': missing,
                  'hard_background_weighted_pixels': hard_bg,
                  'g_missing_mean': gate_missing_sum/max(gate_missing_count, 1),
                  'g_hard_background_mean': gate_background_sum/max(gate_background_count, 1),
                  'seconds': time.perf_counter()-start}
        print('[TRAIN] ' + json.dumps(report), flush=True)
        with (args.output_dir / 'epoch_metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(report) + '\n')
        save(model, args, epoch, optimizer, best_score)
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            reports, _, _, _ = evaluate_thresholds(
                model, full_dataset(args.root_path, 'val'), args.device,
                (.6,), args.max_images)
            score = reports[.6]['selection_score']
            print(f'[VAL] epoch={epoch} score={score:.6f} '
                  f'IoU={reports[.6]["p1"]["iou"]:.5f}', flush=True)
            if score > best_score:
                best_score = score
                target = args.output_dir / 'best.pth'
                temp = target.with_suffix('.tmp')
                torch.save({'version': VERSION, 'epoch': epoch,
                            'v7_checkpoint': str(args.v7_checkpoint.resolve()),
                            'relation_mode': args.relation_mode,
                            'transport_availability_power':
                                model.dynamics.transport_availability_power,
                            'gate_relation': model.gate_relation.state_dict(),
                            'best_score': best_score}, temp)
                temp.replace(target)
            save(model, args, epoch, optimizer, best_score)


def evaluate(args, model):
    dataset = full_dataset(args.root_path, args.command)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_file = args.output_dir / 'val_selection.json'
    if args.command == 'val':
        thresholds = tuple(round(float(t), 2) for t in np.arange(.35, .751, .01))
        reports, _, _, _ = evaluate_thresholds(
            model, dataset, args.device, thresholds, args.max_images)
        selected = max(reports, key=lambda t: reports[t]['selection_score'])
        selection = {'selected_on': 'val', 'threshold': selected,
                     'v8_checkpoint': str(args.v8_checkpoint.resolve()),
                     'v7_checkpoint': str(args.v7_checkpoint.resolve()),
                     'relation_mode': args.relation_mode}
        selection_file.write_text(json.dumps(selection, indent=2), encoding='utf-8')
    else:
        selection = json.loads(selection_file.read_text(encoding='utf-8'))
        expected = {'selected_on': 'val',
                    'v8_checkpoint': str(args.v8_checkpoint.resolve()),
                    'v7_checkpoint': str(args.v7_checkpoint.resolve()),
                    'relation_mode': args.relation_mode}
        if any(selection.get(key) != value for key, value in expected.items()):
            raise ValueError('Test requires the matching validation selection')
        selected = selection['threshold']
    reports, details, _, _ = evaluate_thresholds(
        model, dataset, args.device, (selected,), args.max_images)
    report = reports[selected]
    report.update(version=VERSION, relation_mode=args.relation_mode,
                  checkpoint=str(args.v8_checkpoint.resolve()),
                  val_selected_threshold=selected)
    write_evaluation(args.output_dir / args.command, report, details[selected])
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'train', 'resume', 'val', 'test'))
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--v7_checkpoint', type=Path, required=True)
    parser.add_argument('--v8_checkpoint', type=Path)
    parser.add_argument('--relation_mode', choices=MODES, required=True)
    parser.add_argument('--transport_availability_power', type=float, default=1.0,
                        help='Exponent on V_k in alpha_k; 0 keeps the original v8 rule')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_images', type=int, default=0)
    args = parser.parse_args()
    if not args.v7_checkpoint.is_file():
        parser.error(f'Missing v7 checkpoint: {args.v7_checkpoint}')
    if args.command in ('resume', 'val', 'test') and (
            args.v8_checkpoint is None or not args.v8_checkpoint.is_file()):
        parser.error('An existing --v8_checkpoint is required')
    if args.command == 'train' and args.v8_checkpoint:
        parser.error('Use resume for an existing v8 checkpoint')
    if args.batch_size < 1 or args.epochs < 1 or args.val_interval < 1:
        parser.error('Invalid training configuration')
    args.device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, source = load_experiment(args)
    args.lambda_bg = source['lambda_bg']
    args.lambda_missing = source['lambda_missing']
    args.lambda_cont = source['lambda_cont']
    args.missing_threshold = source['missing_threshold']
    if args.command == 'smoke':
        smoke(args, model)
    elif args.command in ('train', 'resume'):
        train(args, model)
    else:
        evaluate(args, model.eval())


if __name__ == '__main__':
    main()
