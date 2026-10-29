"""Equal-budget local/multiscale V9 continuations; frozen BASE/Q3, original loss."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import run_reliable_v9_mass as v9
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from mass_protocol import blend_weight, sliding_positions
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_multiscale_transport import install, transition, SOURCES

VERSION = 'v9_multiscale_source_selector_v1'
NAMES = (*v9.MODULE_NAMES, 'gate_relation', 'state_gate')
ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def seed_run(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def original(args):
    source = args.source_checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Original V9 checkpoint not found: {source}')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if payload.get('version') != v9.VERSION or payload.get('stage') != 'D':
        raise ValueError('Requires an original V9 stage-D checkpoint')
    if any(key in payload for key in ('benefit_version', 'v91_config', 'multiscale_config')):
        raise ValueError('Use the original V9 checkpoint for both continuations, not a later loss/architecture experiment')
    if payload['identity'].get('initialization') != 'direct_random_downstream':
        raise ValueError('This paired protocol requires the direct V9 initialization')
    config = payload['training_config']
    if Path(config['root_path']).resolve() != args.root_path.resolve():
        raise ValueError('Dataset root differs from the source checkpoint')
    if config['seed'] != args.seed or config['batch_size'] != args.batch_size:
        raise ValueError('Preserve the original V9 seed and physical batch size')
    if config['lambda_cont'] != .2:
        raise ValueError('The source must use L_seg + 0.2 L_cont')
    seed_run(args.seed)
    model, _, identity = v9.load_initial(SimpleNamespace(initialization='direct',
        q3_checkpoint=Path(payload['identity']['q3']['path']), device=args.device))
    # The original model/runner files are not edited in this branch, including their hashes.
    v9.restore_checkpoint(model, source, identity, 'D')
    model.v9_training_config = dict(config)
    model.set_accept_floor(0.)
    return model, payload


def configure(model, rates, selector_lr):
    model.requires_grad_(False)
    groups = []
    for name in v9.STAGE_MODULES['D']:
        selected = v9.module(model, name)
        selected.requires_grad_(True)
        groups.append({'name': name, 'params': list(selected.parameters()), 'lr': rates[name]})
    if model.dynamics.selector is not None:
        model.dynamics.selector.requires_grad_(True)
        groups.append({'name': 'selector', 'params': list(model.dynamics.selector.parameters()), 'lr': selector_lr})
    flat = [id(p) for group in groups for p in group['params']]
    if len(flat) != len(set(flat)) or set(flat) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer parameter coverage differs')
    if any(p.requires_grad for p in model.frozen.parameters()):
        raise RuntimeError('BASE/Q3 must remain frozen')
    return torch.optim.AdamW(groups, weight_decay=.01)


def dataset(args, augment=True):
    return RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
        source_patch_size=1500, random_crop_train=True, random_crops_per_image=1,
        random_crop_seed=args.seed, augment=augment)


def configuration(args, payload):
    rates = dict(payload['learning_rates'])
    selector_lr = args.selector_lr if args.selector_lr is not None else max(rates.values())
    return {'version': VERSION, 'source_checkpoint': str(args.source_checkpoint.resolve()),
        'source_sha256': v9.digest(args.source_checkpoint), 'identity': payload['identity'],
        'mode': args.transport_mode, 'steps': 3, 'epochs': args.epochs,
        'transition_epochs': args.transition_epochs, 'selector_lr': selector_lr,
        'selector': {'embedding': 8, 'hidden': 32, 'radii': [1, 2, 4, 8],
                     'directions_dy_dx': [list(v[:2]) for v in SOURCES[::4]],
                     'corridor_fractions': [.25, .5, .75], 'lateral_offsets': [-.5, 0., .5],
                     'prior': 'log(evolving_V9_source_support.clamp_min(1e-6))',
                     'stay_bias_init': 2., 'epsilon': 1e-6, 'recompute': True},
        'training': payload['training_config'], 'learning_rates': rates,
        'num_workers': args.num_workers, 'val_interval': args.val_interval,
        'checkpoint_selection_threshold': .60, 'threshold_grid': [.35, .75, .01],
        'protocol': 'FP32/no EMA; native1500 random512 x1; batch2/accum1; val/test512 stride256 weighted logits',
        'loss': 'original L_seg + 0.2 L_cont; no L_G or source relation loss',
        'code_sha256': {p: v9.digest(ROOT/p) for p in (
            'run_v9_multiscale.py', 'networks/v9_multiscale_transport.py',
            'networks/reliable_directional_repair.py', 'run_reliable_v9_mass.py')}}


def save(path, model, payload, config, optimizer, epoch, best, record):
    state = {'version': VERSION, 'multiscale_config': config,
        'identity': payload['identity'], 'epoch': epoch, 'score': best,
        'modules': {name: getattr(model, name).state_dict() for name in NAMES},
        'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
        'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'epoch_record': record}
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(state, temporary); temporary.replace(path)


def restore(model, path, config):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('version') != VERSION or saved.get('multiscale_config') != config:
        raise ValueError('Checkpoint architecture, upstream identity or paired training configuration differs')
    if set(saved['modules']) != set(NAMES):
        raise ValueError('Downstream checkpoint module set differs')
    for name in NAMES:
        getattr(model, name).load_state_dict(saved['modules'][name], strict=True)
    return saved


def loss_for(result, batch, config):
    return dynamic_losses(result, batch['mask'], batch['skeleton'],
        **{name: config[name] for name in ('lambda_bg', 'lambda_missing', 'lambda_cont', 'missing_threshold')})


def check_gradients(model):
    if any(p.grad is not None for p in model.frozen.parameters()):
        raise RuntimeError('Frozen BASE/Q3 received a gradient')
    for p in model.parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            raise RuntimeError('Nonfinite gradient')


def train(args, model, payload, config):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    out = args.output_dir / args.transport_mode
    out.mkdir(parents=True, exist_ok=args.command == 'resume')
    optimizer = configure(model, config['learning_rates'], config['selector_lr'])
    start, best = 1, -float('inf')
    if args.command == 'resume':
        state = restore(model, out/'last.pth', config)
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['epoch']+1, state['score']
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if args.device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    write_json(out/'experiment.json', config)
    ds = dataset(args)
    val = full_dataset(args.root_path, 'val')
    for epoch in range(start, args.epochs+1):
        # Epoch-local RNG and loader seed make image order/augmentation identical
        # for both arms and for resumed versus uninterrupted continuations.
        seed_run(args.seed + epoch*100003)
        ds.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch*100003)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=generator,
            num_workers=args.num_workers, pin_memory=args.device.type == 'cuda', persistent_workers=False)
        model.train(); model.dynamics.blend = transition(epoch, args.transition_epochs)
        sums = {'loss': 0., 'seg': 0., 'cont': 0., 'selector_grad_l2': 0., 'stay': 0.,
                'far_source_weight_r4_r8': 0., 'transport_abs_mean': 0.}
        for index, batch in enumerate(loader, 1):
            batch = {key: batch[key].to(args.device) for key in ('image', 'mask', 'skeleton')}
            optimizer.zero_grad(set_to_none=True)
            result = model(batch['image'], capture_diagnostics=True)
            loss, terms = loss_for(result, batch, config['training'])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss.backward(); check_gradients(model)
            grad = v9.grad_norm(model.dynamics.selector) if model.dynamics.selector is not None else 0.
            optimizer.step()
            sums['loss'] += float(loss.detach()); sums['seg'] += float(terms['seg']); sums['cont'] += float(terms['cont'])
            sums['selector_grad_l2'] += grad
            sums['transport_abs_mean'] += float(result['transport_delta'].detach().abs().mean())
            if 'selector_weights' in result:
                w = result['selector_weights'].detach()
                sums['stay'] += float(w[:, :, :1].mean())
                indices = [1+i for i, (_, _, r) in enumerate(SOURCES) if r >= 4]
                sums['far_source_weight_r4_r8'] += float(w[:, :, indices].sum(2).mean())
            if index == 1 or index % 20 == 0 or index == len(loader):
                print(f'[TRAIN] {args.transport_mode} epoch={epoch}/{args.epochs} batch={index}/{len(loader)} '
                      f'blend={model.dynamics.blend:.3f} loss={float(loss.detach()):.6f} selector_grad={grad:.6g}', flush=True)
        record = {'epoch': epoch, 'mode': args.transport_mode, 'blend': model.dynamics.blend,
                  **{key: value/max(len(loader), 1) for key, value in sums.items()}}
        improved = False
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.dynamics.blend = 1.
            reports = evaluate_thresholds(model, val, args.device, (.60,),
                reference_threshold=config['training']['reference_threshold'])[0]
            score = reports[.60]['selection_score']
            record['validation_score'] = score
            improved = score > best
            best = max(best, score)
            model.dynamics.blend = transition(epoch, args.transition_epochs)
        with (out/'train.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')
        save(out/'last.pth', model, payload, config, optimizer, epoch, best, record)
        if improved:
            save(out/'best.pth', model, payload, config, optimizer, epoch, best, record)
        print(json.dumps(record), flush=True)
    if v9.digest(args.source_checkpoint) != config['source_sha256']:
        raise RuntimeError('Original checkpoint changed during continuation')


@torch.no_grad()
def native_prediction(model, sample, device):
    image = sample['image'][None].to(device)
    h, w = image.shape[-2:]
    canvas = torch.zeros((2, 1, h, w), device=device)
    denominator = torch.zeros((1, 1, h, w), device=device)
    weight = blend_weight(512, device)
    for y in sliding_positions(h, 512, 256):
        for x in sliding_positions(w, 512, 256):
            result = model(image[:, :, y:y+512, x:x+512])
            region = (slice(None), slice(None), slice(y, y+512), slice(x, x+512))
            canvas[region] += torch.cat((result['z0'], result['z1']), 0).float()*weight
            denominator[region] += weight
    if not bool((denominator > 0).all()):
        raise RuntimeError('Uncovered native image pixels')
    logits = (canvas/denominator).cpu().numpy()[:, 0]
    return 1./(1.+np.exp(-np.clip(logits, -80., 80.)))


def evaluate(args, model, config):
    from run_q3_r3_mass import full_dataset, PROTOCOL
    from run_reliable_directional_mass import evaluate_thresholds
    from evaluation.road_comparison.road_metrics import prepare_target, image_metrics, summarize
    out = args.output_dir / args.transport_mode
    ckpt = out/'best.pth'
    restore(model, ckpt, config)
    model.eval(); model.dynamics.blend = 1.
    if args.command == 'sweep':
        reports = evaluate_thresholds(model, full_dataset(args.root_path, 'val'), args.device,
            tuple(np.round(np.arange(.35, .751, .01), 2).tolist()),
            reference_threshold=config['training']['reference_threshold'])[0]
        threshold = max(reports, key=lambda key: reports[key]['selection_score'])
        write_json(out/'val_selection.json', {'selected_on': 'val', 'threshold': threshold,
            'checkpoint_sha256': v9.digest(ckpt), 'config': config, 'report': reports[threshold],
            'threshold_curves': reports})
        print(f'[SELECTED] {args.transport_mode} threshold={threshold}', flush=True)
        return
    selection = json.loads((out/'val_selection.json').read_text(encoding='utf-8'))
    if selection['checkpoint_sha256'] != v9.digest(ckpt) or selection['config'] != config:
        raise ValueError('Validation selection does not belong to this checkpoint')
    threshold = float(selection['threshold'])
    test = full_dataset(args.root_path, 'test')
    pred_dir = out/'unified/predictions/test'
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, cases = [], []
    with torch.inference_mode():
        for index in range(len(test)):
            sample = test[index]; probability = native_prediction(model, sample, args.device)[1]
            case = str(sample['case_name']); target = sample['mask'][0].numpy() > .5
            pred = probability >= threshold
            path = pred_dir/(Path(case).stem + '_pred.png')
            if not cv2.imwrite(str(path), pred.astype(np.uint8)*255):
                raise OSError(f'Failed to write prediction: {path}')
            rows.append({'case_id': case, **image_metrics(pred, prepare_target(target, PROTOCOL), PROTOCOL)})
            cases.append({'case_id': case, 'prediction': str(path.resolve())})
            print(f'[TEST] {args.transport_mode} {index+1}/{len(test)} {case}', flush=True)
    write_csv(out/'test_comparison.csv', [{'name': args.transport_mode, 'threshold': threshold, **summarize(rows)}])
    write_csv(out/'test_per_image.csv', rows)
    write_json(out/'test_predictions.json', {'checkpoint_sha256': v9.digest(ckpt), 'threshold': threshold,
        'selected_on': 'val', 'native_size': 1500, 'cases': cases, 'protocol': PROTOCOL})


def smoke(args, model, payload, config):
    # Actual source checkpoint + an actual training crop. No optimizer state or
    # module parameter from this preflight is subsequently used for formal training.
    ds = dataset(args, augment=False); ds.set_epoch(1)
    sample = ds[0]
    batch = {key: sample[key][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
             for key in ('image', 'mask', 'skeleton')}
    model.eval()
    with torch.inference_mode():
        before = model(batch['image'], capture_diagnostics=True)
    install(model, 'local', args.seed)
    with torch.inference_mode():
        after = model(batch['image'], capture_diagnostics=True)
    errors = {key: float((before[key]-after[key]).abs().max()) for key in ('z1', 'u1', 'u2', 'u3', 'g_step', 'alpha')}
    if any(errors.values()):
        raise RuntimeError(f'Local V9 equivalence failed: {errors}')
    install(model, 'multiscale', args.seed)
    optimizer = configure(model, config['learning_rates'], config['selector_lr'])
    model.train(); model.dynamics.blend = 1.
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter(); norms = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(batch['image'], capture_diagnostics=True)
        loss, _ = loss_for(result, batch, config['training'])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite smoke loss')
        loss.backward(); check_gradients(model)
        norms.append(v9.grad_norm(model.dynamics.selector)); optimizer.step()
    if not all(value > 0. for value in norms):
        raise RuntimeError(f'Selector has no useful gradient: {norms}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/'smoke.json', {'passed': True, 'local_exact_errors': errors,
        'source_sha256': v9.digest(args.source_checkpoint), 'batch_size': args.batch_size,
        'code_sha256': config['code_sha256'], 'training': config['training'],
        'selector_gradients': norms, 'seconds': time.perf_counter()-started,
        'peak_cuda_mib': torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.type == 'cuda' else None})
    print('[SMOKE PASSED] strict actual checkpoint, local equivalence, actual multiscale forward/backward', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'train', 'resume', 'sweep', 'test', 'audit'))
    parser.add_argument('--source_checkpoint', type=Path, required=True)
    parser.add_argument('--root_path', type=Path, default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--transport_mode', choices=('local', 'multiscale'), default='multiscale')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--transition_epochs', type=int, default=3)
    parser.add_argument('--selector_lr', type=float)
    parser.add_argument('--audit_split', choices=('val', 'test'), default='val')
    parser.add_argument('--min_effect', type=float, default=1e-5)
    parser.add_argument('--reference_set', type=Path)
    args = parser.parse_args(); args.device = torch.device(args.device)
    if args.batch_size != 2 or args.epochs < 1 or args.transition_epochs < 0 or args.val_interval < 1:
        parser.error('Requires physical batch2, positive epochs/val_interval and nonnegative transition_epochs')
    if args.transition_epochs >= args.epochs:
        parser.error('Budget must include at least one epoch after the transition')
    if args.selector_lr is not None and args.selector_lr <= 0:
        parser.error('selector_lr must be positive')
    model, payload = original(args)
    config = configuration(args, payload)
    if args.command == 'smoke':
        smoke(args, model, payload, config); return
    install(model, args.transport_mode, args.seed)
    if args.command in ('train', 'resume'):
        marker = json.loads((args.output_dir/'smoke.json').read_text(encoding='utf-8'))
        if (not marker.get('passed') or marker['source_sha256'] != config['source_sha256']
                or marker['batch_size'] != args.batch_size or marker['code_sha256'] != config['code_sha256']
                or marker['training'] != config['training']):
            raise ValueError('Run actual-checkpoint smoke first with the same source and physical batch size')
        train(args, model, payload, config)
    elif args.command == 'audit':
        from diagnostics.audit_v9_multiscale import audit
        audit(args, model, config)
    else:
        evaluate(args, model, config)


if __name__ == '__main__':
    main()
