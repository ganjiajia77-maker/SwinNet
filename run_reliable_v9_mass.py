"""State-aware v9 on an intact v8: staged training and validation-only comparison."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from mass_protocol import blend_weight, sliding_positions
from networks.reliable_directional_repair import dynamic_losses

VERSION = 'reliable_directional_dynamics_v9_state_gate_v1'
MODULE_NAMES = ('visual', 'reliable_head', 'relations', 'proposal_head', 'accept_head', 'dynamics')
STAGE_MODULES = {
    'B': ('gate_relation', 'state_gate'),
    'C': ('gate_relation', 'state_gate', 'reliable_head', 'relations', 'rates'),
    'D': ('gate_relation', 'state_gate', 'reliable_head', 'relations', 'rates',
          'visual', 'proposal_head', 'accept_head'),
}
LR = {'state_gate': 1e-4, 'gate_relation': 5e-5, 'reliable_head': 1e-5,
      'relations': 1e-5, 'rates': 1e-5, 'visual': 5e-6,
      'proposal_head': 5e-6, 'accept_head': 5e-6}


class StateGate(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, width, 1), nn.GELU(),
                                 nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def module(model, name):
    return model.dynamics.rates if name == 'rates' else getattr(model, name)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def preflight(args):
    paths = {'v7': args.v7_checkpoint.resolve(), 'v8': args.v8_checkpoint.resolve()}
    report = {'checkpoints': {}, 'upstream_cache_status': 'No cache manifest supplied or audited in this command',
              'required_upstream_tensors': ['z0', 'q', 'highres_h', 'stage3_h'],
              'cache_note': 'R/W/B/A maps are not substitutes for trainable module inputs.'}
    source = None
    payloads = {}
    errors = []
    for name, path in paths.items():
        entry = {'path': str(path), 'exists': path.is_file()}
        if path.is_file():
            state = torch.load(path, map_location='cpu', weights_only=False)
            payloads[name] = state
            entry.update(version=state.get('version'), keys=list(state), epoch=state.get('epoch'))
            if name == 'v7':
                source = state
                entry['saved_modules'] = list(state.get('modules', {}))
            else:
                entry['gate_relation_tensors'] = len(state.get('gate_relation', {}))
        report['checkpoints'][name] = entry
    if 'v7' in payloads:
        if payloads['v7'].get('version') != 'reliable_directional_dynamics_v7_continuity_transport':
            errors.append('v7 checkpoint is not the continuity transport baseline')
        if set(payloads['v7'].get('modules', {})) != set(MODULE_NAMES):
            errors.append('v7 checkpoint has missing/different dynamics modules')
    if 'v8' in payloads:
        if payloads['v8'].get('version') != 'reliable_directional_dynamics_v8_semantic_gate':
            errors.append('v8 must be the original semantic gate baseline')
        if payloads['v8'].get('v7_checkpoint') != str(paths['v7']):
            errors.append('v8 references another v7 checkpoint path')
        if payloads['v8'].get('relation_mode') != 'full':
            errors.append('v8 is not the full semantic relationship mode')
    if source is not None:
        for name, key in [('base', 'base_checkpoint'), ('q3', 'q3_checkpoint')]:
            path = Path(source[key])
            report['checkpoints'][name] = {'path': str(path), 'exists': path.is_file()}
    report['errors'] = errors
    report['strict_upstream_model_loading_still_required'] = True
    report['can_run_full_model'] = (not errors and len(report['checkpoints']) == 4 and
        all(item['exists'] for item in report['checkpoints'].values()))
    report['original_v8_freezing'] = {
        'trainable': ['gate_relation'],
        'frozen': ['BASE', 'Q3', 'visual', 'reliable_head', 'relations',
                   'proposal_head', 'accept_head', 'dynamics.rates'],
        'base_outputs': 'no_grad and detach; downstream operations remain differentiable',
        'optimizer': 'AdamW(model.gate_relation.parameters())'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not report['can_run_full_model']:
        raise SystemExit('Missing original upstream weights; use a complete, consistently rebuilt v7/v8 baseline before stage0.')


def load_initial(args):
    if getattr(args, 'initialization', 'v8') == 'direct':
        return load_direct(args)
    from run_reliable_v8_mass import load_experiment as load_v8
    model, source = load_v8(args)
    model.state_gate = StateGate().to(args.device)
    paths = {'v7': args.v7_checkpoint.resolve(), 'v8': args.v8_checkpoint.resolve(),
             'base': Path(source['base_checkpoint']).resolve(),
             'q3': Path(source['q3_checkpoint']).resolve()}
    identity = {key: {'path': str(path), 'sha256': digest(path)}
                for key, path in paths.items()}
    identity['baseline_label'] = args.baseline_label
    root = Path(__file__).resolve().parent
    identity['code_sha256'] = {name: digest(root/name) for name in (
        'run_reliable_v9_mass.py', 'networks/reliable_directional_repair.py',
        'run_reliable_directional_mass.py')}
    return model, source, identity


def load_direct(args):
    from types import SimpleNamespace
    from run_q3_r3_mass import load_experiment as load_q3
    from networks.reliable_directional_repair import ReliableDirectionalRepair
    if args.q3_checkpoint is None or not args.q3_checkpoint.is_file():
        raise ValueError('Direct V9 requires an existing --q3_checkpoint; /best.pth is not a valid run path')
    q3_path = args.q3_checkpoint.resolve()
    q3_state = torch.load(q3_path, map_location='cpu', weights_only=False)
    if q3_state.get('repair_version') != 'dual_recovery_v1':
        raise ValueError('Expected a dual-recovery Q3 checkpoint')
    base_path = Path(q3_state['base_checkpoint']).resolve()
    if not base_path.is_file():
        raise ValueError(f'Q3 requires its exact frozen base: {base_path}')
    frozen = load_q3(SimpleNamespace(base_checkpoint=base_path, repair_checkpoint=q3_path,
        init_repair_checkpoint=None, eta_plus=.5, eta_minus=.5,
        enable_stage2_local_propagation=False, enable_gap_refinement=False, device=args.device))
    model = ReliableDirectionalRepair(frozen, width=32, steps=3,
        bridge_route=True, conditional_transport=True, transport_gate_mode='v7',
        gate_relation_mode='full').to(args.device)
    model.state_gate = StateGate().to(args.device)
    source = {'lambda_bg': 1., 'lambda_missing': 2., 'lambda_cont': .2, 'missing_threshold': .45}
    root = Path(__file__).resolve().parent
    identity = {'initialization': 'direct_random_downstream',
        'baseline_label': 'new V9 experiment; not an ablation of trained V8',
        'base': {'path': str(base_path), 'sha256': digest(base_path)},
        'q3': {'path': str(q3_path), 'sha256': digest(q3_path)},
        'code_sha256': {name: digest(root/name) for name in ('run_reliable_v9_mass.py',
            'networks/reliable_directional_repair.py', 'run_reliable_directional_mass.py')}}
    return model, source, identity


def configure(model, stage):
    model.requires_grad_(False)
    groups = []
    for name in STAGE_MODULES[stage]:
        selected = module(model, name)
        selected.requires_grad_(True)
        params = list(selected.parameters())
        groups.append({'params': params, 'lr': LR[name], 'name': name})
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in groups for p in group['params']]
    if expected != set(actual) or len(actual) != len(set(actual)):
        raise RuntimeError('Optimizer does not cover each trainable parameter exactly once')
    print('[OPTIMIZER] ' + json.dumps({g['name']: {'parameters': sum(p.numel() for p in g['params']),
                                                  'lr': g['lr']} for g in groups}), flush=True)
    return torch.optim.AdamW(groups, weight_decay=.01)


def save_checkpoint(path, model, identity, stage, epoch, score, optimizer=None):
    payload = {'version': VERSION, 'identity': identity, 'stage': stage, 'epoch': epoch,
               'score': score, 'modules': {name: getattr(model, name).state_dict()
                                           for name in MODULE_NAMES},
               'gate_relation': model.gate_relation.state_dict(),
               'state_gate': model.state_gate.state_dict(),
               'training_config': getattr(model, 'v9_training_config', None),
               'learning_rates': dict(LR),
               'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def restore_checkpoint(model, path, identity, expected_stage=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('version') != VERSION or payload.get('identity') != identity:
        raise ValueError('v9 checkpoint version or upstream checkpoint identity differs')
    if expected_stage is not None and payload.get('stage') != expected_stage:
        raise ValueError(f'Expected stage {expected_stage}, got {payload.get("stage")}')
    if set(payload['modules']) != set(MODULE_NAMES):
        raise ValueError('Checkpoint dynamics modules differ from v7/v8')
    for name in MODULE_NAMES:
        getattr(model, name).load_state_dict(payload['modules'][name], strict=True)
    model.gate_relation.load_state_dict(payload['gate_relation'], strict=True)
    model.state_gate.load_state_dict(payload['state_gate'], strict=True)
    return payload


@torch.no_grad()
def stage0(args, model, identity):
    from run_q3_r3_mass import full_dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'stage0.json').write_text(json.dumps({'passed': False,
        'identity': identity, 'reason': 'validation in progress'}), encoding='utf-8')
    model.eval()
    dataset = full_dataset(args.root_path, 'val')
    count = min(len(dataset), args.max_images or len(dataset))
    errors = {key: 0. for key in ('g_step', 'alpha', 'u1', 'u2', 'u3', 'p1')}
    tp = [0, 0]; fp = [0, 0]; fn = [0, 0]; tiles = 0
    for i in range(count):
        sample = dataset[i]
        image = sample['image'][None].to(args.device)
        height, width = image.shape[-2:]
        sums = [torch.zeros((1, 1, height, width), device=args.device) for _ in range(2)]
        denom = torch.zeros_like(sums[0]); weight = blend_weight(512, args.device)
        for y in sliding_positions(height, 512, 256):
            for x in sliding_positions(width, 512, 256):
                patch = image[:, :, y:y+512, x:x+512]
                state_gate = model.state_gate
                model.state_gate = None
                try:
                    old = model(patch, capture_diagnostics=True)
                finally:
                    model.state_gate = state_gate
                new = model(patch, capture_diagnostics=True)
                if any(not bool(torch.isfinite(result[key]).all())
                       for result in (old, new) for key in errors):
                    raise RuntimeError('Nonfinite stage0 output')
                for key in errors:
                    errors[key] = max(errors[key], float((old[key] - new[key]).abs().max()))
                region = (slice(None), slice(None), slice(y, y+512), slice(x, x+512))
                sums[0][region] += old['z1'].float() * weight
                sums[1][region] += new['z1'].float() * weight
                denom[region] += weight
                tiles += 1
        if not bool((denom > 0).all()):
            raise RuntimeError('Sliding windows do not cover image')
        gt = sample['mask'][0].numpy() > .5
        for k in range(2):
            pred = (sums[k] / denom).sigmoid()[0, 0].cpu().numpy() >= args.reference_threshold
            tp[k] += int((pred & gt).sum()); fp[k] += int((pred & ~gt).sum())
            fn[k] += int((~pred & gt).sum())
        print(f'[STAGE0] {i+1}/{count} images', flush=True)
    ious = [tp[k] / max(tp[k] + fp[k] + fn[k], 1) for k in range(2)]
    if not count or max(errors.values()) > args.tolerance or tp[0] != tp[1] or fp[0] != fp[1] or fn[0] != fn[1]:
        raise RuntimeError(f'V8 identity check failed: errors={errors}, IoU={ious}')
    report = {'identity': identity, 'passed': True, 'images': count, 'tiles': tiles,
              'tolerance': args.tolerance, 'max_abs': errors, 'old_iou': ious[0],
              'new_iou': ious[1], 'reference_threshold': args.reference_threshold,
              'root_path': str(args.root_path.resolve())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'stage0.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def grad_norm(layer):
    pieces = [p.grad.detach().float().square().sum() for p in layer.parameters() if p.grad is not None]
    return float(torch.stack(pieces).sum().sqrt()) if pieces else 0.


def smoke(args, model, source, identity):
    """Check two real optimizer steps per phase and then restore all downstream weights."""
    names = (*MODULE_NAMES, 'gate_relation', 'state_gate')
    saved = {name: {key: value.detach().cpu().clone() for key, value in getattr(model, name).state_dict().items()}
             for name in names}
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
        source_patch_size=1500, random_crop_train=True, random_crops_per_image=1,
        random_crop_seed=args.seed)
    sample = dataset[0]
    image = sample['image'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    mask = sample['mask'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    skeleton = sample['skeleton'][None].repeat(args.batch_size, 1, 1, 1).to(args.device)
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    report = {'identity': identity, 'batch_size': args.batch_size, 'phases': {}}
    start = time.perf_counter()
    try:
        stages = ('D',) if getattr(args, 'initialization', 'v8') == 'direct' else STAGE_MODULES
        for stage in stages:
            for name in names:
                getattr(model, name).load_state_dict(saved[name], strict=True)
            optimizer = configure(model, stage)
            model.train(); model.set_accept_floor(0.)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = model(image, capture_diagnostics=True)
                loss, terms = dynamic_losses(result, mask, skeleton,
                    lambda_bg=source['lambda_bg'], lambda_missing=source['lambda_missing'],
                    lambda_cont=source['lambda_cont'], missing_threshold=source['missing_threshold'])
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(result['p1']).all()):
                    raise RuntimeError('Nonfinite real-batch smoke loss/output')
                loss.backward()
                norms = {name: grad_norm(module(model, name)) for name in STAGE_MODULES[stage]}
                if any(p.grad is not None for p in model.frozen.parameters()):
                    raise RuntimeError('Frozen upstream received a gradient')
                optimizer.step()
            report['phases'][stage] = {'loss': float(loss.detach()), 'gradients': norms}
            if any(not np.isfinite(v) or v <= 0 for v in norms.values()):
                raise RuntimeError(f'No finite nonzero gradient for a real-batch phase: {stage}: {norms}')
        report['passed'] = True
    finally:
        for name in names:
            getattr(model, name).load_state_dict(saved[name], strict=True)
        model.zero_grad(set_to_none=True)
        model.eval()
        report['elapsed_seconds'] = time.perf_counter()-start
        report['peak_cuda_mib'] = (torch.cuda.max_memory_allocated(args.device)/2**20
                                  if args.device.type == 'cuda' else None)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


def train(args, model, source, identity):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    direct = getattr(args, 'initialization', 'v8') == 'direct'
    if direct:
        if args.stage != 'D' or args.previous_checkpoint:
            raise ValueError('Direct initialization jointly trains stage D without an earlier stage checkpoint')
    else:
        marker = json.loads((args.output_dir / 'stage0.json').read_text(encoding='utf-8'))
        if (not marker.get('passed') or marker.get('identity') != identity or
                marker.get('images') != len(full_dataset(args.root_path, 'val')) or
                marker.get('root_path') != str(args.root_path.resolve()) or
                marker.get('reference_threshold') != args.reference_threshold):
            raise ValueError('A successful stage0 over the full validation set with identical checkpoints is required')
    smoke_report = json.loads((args.output_dir / 'smoke.json').read_text(encoding='utf-8'))
    if not smoke_report.get('passed') or smoke_report.get('identity') != identity:
        raise ValueError('Successful real-batch B/C/D smoke checks are required')
    prior = None if direct else {'B': None, 'C': 'B', 'D': 'C'}[args.stage]
    training_config = {'root_path': str(args.root_path.resolve()), 'batch_size': args.batch_size,
        'seed': args.seed, 'reference_threshold': args.reference_threshold,
        **{name: source[name] for name in ('lambda_bg', 'lambda_missing', 'lambda_cont', 'missing_threshold')}}
    model.v9_training_config = training_config
    resume = None
    if args.command == 'resume':
        if not args.dynamic_checkpoint:
            raise ValueError('resume requires --dynamic_checkpoint pointing to last.pth')
        resume = restore_checkpoint(model, args.dynamic_checkpoint, identity, args.stage)
        if resume.get('learning_rates') != LR:
            raise ValueError('Resume learning-rate groups differ from the saved run')
        if resume.get('training_config') != training_config:
            raise ValueError('Resume data/loss/seed/batch configuration differs')
    elif prior:
        if not args.previous_checkpoint:
            raise ValueError(f'Stage {args.stage} requires the best checkpoint from stage {prior}')
        restore_checkpoint(model, args.previous_checkpoint, identity, prior)
    elif args.previous_checkpoint:
        raise ValueError('Stage B starts from v8; do not pass a prior stage checkpoint')
    output = args.output_dir / ('stage_' + args.stage)
    output.mkdir(parents=True, exist_ok=resume is not None)
    optimizer = configure(model, args.stage)
    first_epoch = 1
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        first_epoch = resume['epoch'] + 1
        torch.set_rng_state(resume['torch_rng'])
        np.random.set_state(resume['numpy_rng'])
        if resume.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
        source_patch_size=1500, random_crop_train=True, random_crops_per_image=1,
        random_crop_seed=args.seed, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    best = resume['score'] if resume else -float('inf')
    for epoch in range(first_epoch, args.epochs + 1):
        model.train(); model.set_accept_floor(0.); dataset.set_epoch(epoch)
        sums = {'loss': 0., 'seg': 0., 'cont': 0., 'delta_plus_saturated_fraction': 0.,
                'delta_minus_mean': 0., 'g_mean': 0., 'r_mean': 0., 'w_entropy': 0.,
                'missing_segments': 0., 'continuity_paths': 0., 'a_mean': 0.,
                'r_near_one_fraction': 0., 'g_near_one_fraction': 0.}
        for step in range(1, 4):
            for key in ('transport_rate', 'alpha', 'beta', 'gamma', 'transport_delta',
                        'proposal_delta', 'anchor_delta', 'g_step'):
                sums[f'{key}_{step}'] = 0.
        grads = {name: 0. for name in STAGE_MODULES[args.stage]}
        cont_grads = {}
        for batch_index, batch in enumerate(loader):
            image = batch['image'].to(args.device); mask = batch['mask'].to(args.device)
            skeleton = batch['skeleton'].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            result = model(image, capture_diagnostics=True)
            loss, terms = dynamic_losses(result, mask, skeleton,
                lambda_bg=source['lambda_bg'], lambda_missing=source['lambda_missing'],
                lambda_cont=source['lambda_cont'], missing_threshold=source['missing_threshold'],
                return_live_terms=True)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(result['p1']).all()):
                raise RuntimeError('Nonfinite output or loss')
            if batch_index == 0:
                for name in grads:
                    params = tuple(module(model, name).parameters())
                    live = torch.autograd.grad(terms['cont_live'], params, retain_graph=True,
                                               allow_unused=True)
                    pieces = [g.detach().square().sum() for g in live if g is not None]
                    cont_grads[name] = float(torch.stack(pieces).sum().sqrt()) if pieces else 0.
            loss.backward()
            if any(p.grad is not None for p in model.frozen.parameters()):
                raise RuntimeError('Frozen BASE/Q3 received gradients')
            for name in grads:
                grads[name] += grad_norm(module(model, name))
            optimizer.step()
            sums['loss'] += float(loss.detach()); sums['seg'] += float(terms['seg'])
            sums['cont'] += float(terms['cont'])
            sums['delta_plus_saturated_fraction'] += float((result['delta_z_plus'] >= 7.9).float().mean())
            sums['delta_minus_mean'] += float(result['delta_z_minus'].mean())
            sums['g_mean'] += float(result['g_step'].mean()); sums['r_mean'] += float(result['r'].mean())
            sums['a_mean'] += float(result['a'].mean())
            sums['r_near_one_fraction'] += float((result['r'] > .99).float().mean())
            sums['g_near_one_fraction'] += float((result['g_step'] > .99).float().mean())
            sums['missing_segments'] += terms['missing_segments']; sums['continuity_paths'] += terms['paths']
            for step in range(1, 4):
                for key in ('transport_rate', 'alpha', 'beta', 'gamma', 'transport_delta',
                            'proposal_delta', 'anchor_delta', 'g_step'):
                    sums[f'{key}_{step}'] += float(result[key][:, step-1].detach().mean())
            w = result['w'].clamp_min(1e-8)
            sums['w_entropy'] += float((-(w * w.log()).sum(1)).mean())
        count = max(len(loader), 1)
        report = {'epoch': epoch, 'stage': args.stage,
                  **{k: v/count for k, v in sums.items()},
                  'gradient_norms': {k: v/count for k, v in grads.items()},
                  'continuity_gradient_norms_first_batch': cont_grads}
        print('[TRAIN] ' + json.dumps(report), flush=True)
        with (output / 'epoch_metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(report) + '\n')
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            reports, _, _, _ = evaluate_thresholds(model, full_dataset(args.root_path, 'val'),
                args.device, (.6,), args.max_images, reference_threshold=args.reference_threshold)
            evaluation = reports[.6]
            (output / f'val_epoch_{epoch:03d}.json').write_text(json.dumps(evaluation, indent=2), encoding='utf-8')
            score = evaluation['selection_score']
            print(f'[VAL] stage={args.stage} epoch={epoch} score={score:.6f}', flush=True)
            if score > best:
                best = score
                save_checkpoint(output / 'best.pth', model, identity, args.stage, epoch, best)
        save_checkpoint(output / 'last.pth', model, identity, args.stage, epoch, best, optimizer)


def compare(args, model, identity):
    from run_q3_r3_mass import full_dataset, stitch_three
    from run_reliable_directional_mass import evaluate_thresholds
    marker = json.loads((args.output_dir / 'stage0.json').read_text(encoding='utf-8'))
    if (not marker.get('passed') or marker.get('identity') != identity or
            marker.get('images') != len(full_dataset(args.root_path, 'val')) or
            marker.get('root_path') != str(args.root_path.resolve()) or
            marker.get('reference_threshold') != args.reference_threshold):
        raise ValueError('Stage0 identity check required')
    dataset = full_dataset(args.root_path, 'val')
    thresholds = tuple(round(float(x), 2) for x in np.arange(.35, .751, .01))
    reports = {}
    checkpoints = {'A_v8': None, 'B': args.b_checkpoint, 'C': args.c_checkpoint, 'D': args.d_checkpoint}
    def snapshot(layer):
        return {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
    initial = {name: snapshot(getattr(model, name)) for name in MODULE_NAMES}
    initial_gate = snapshot(model.gate_relation)
    initial_state = snapshot(model.state_gate)
    for name, checkpoint in checkpoints.items():
        if checkpoint is None and name != 'A_v8':
            raise ValueError(f'Missing checkpoint for {name}')
        if name == 'A_v8':
            for key in MODULE_NAMES:
                getattr(model, key).load_state_dict(initial[key], strict=True)
            model.gate_relation.load_state_dict(initial_gate, strict=True)
            model.state_gate.load_state_dict(initial_state, strict=True)
        else:
            restore_checkpoint(model, checkpoint, identity, name)
        reports[name], _, _, _ = evaluate_thresholds(model, dataset, args.device, thresholds,
            args.max_images, reference_threshold=args.reference_threshold)
    selected = {name: max(data, key=lambda t: data[t]['selection_score']) for name, data in reports.items()}
    ref = reports['A_v8'][selected['A_v8']]['p1']
    matches = {'A_v8': (selected['A_v8'], selected['A_v8'])}
    # Fine matching uses probability histograms, then evaluates the actual
    # selected masks. It does not reuse the coarse model-selection grid.
    for name in ('B', 'C', 'D'):
        restore_checkpoint(model, checkpoints[name], identity, name)
        model.eval()
        all_hist = np.zeros(args.matching_bins, dtype=np.int64)
        gt_hist = np.zeros_like(all_hist)
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                _, z1, _ = stitch_three(model, sample['image'][None].to(args.device))
                probability = 1. / (1. + np.exp(-np.clip(z1, -80., 80.)))
                gt = sample['mask'][0].numpy() > .5
                all_hist += np.histogram(probability, bins=args.matching_bins, range=(0., 1.))[0]
                gt_hist += np.histogram(probability[gt], bins=args.matching_bins, range=(0., 1.))[0]
        fg_counts = all_hist[::-1].cumsum()[::-1]
        tp_counts = gt_hist[::-1].cumsum()[::-1]
        recall_index = int(np.argmin(np.abs(tp_counts[1:] - ref['tp']))) + 1
        area_index = int(np.argmin(np.abs(fg_counts[1:] - ref['tp'] - ref['fp']))) + 1
        matches[name] = (recall_index/args.matching_bins, area_index/args.matching_bins)
        extra, _, _, _ = evaluate_thresholds(model, dataset, args.device,
            tuple(sorted(set(matches[name]))), reference_threshold=args.reference_threshold)
        reports[name].update(extra)
    comparisons = {}
    for name, data in reports.items():
        same = min(thresholds, key=lambda t: abs(t - args.reference_threshold))
        match_recall, match_area = matches[name]
        comparisons[name] = {key: {**data[t],
                'recall_difference_vs_v8': data[t]['p1']['recall']-ref['recall'],
                'foreground_area_difference_vs_v8': data[t]['p1']['tp']+data[t]['p1']['fp']-ref['tp']-ref['fp']}
            for key, t in
            [('own_val_threshold', selected[name]), ('same_threshold', same),
             ('matched_v8_recall', match_recall), ('matched_v8_foreground_area', match_area)]}
    out = args.output_dir / 'comparison_val.json'
    out.write_text(json.dumps({'identity': identity, 'p0_reference_threshold': args.reference_threshold,
        'comparisons': comparisons, 'threshold_curves': reports,
        'matching_probability_resolution': 1./args.matching_bins,
        'note': 'Own thresholds use 0.01 grid; matching uses fine probability histograms with actual residual errors. '
                'APLS is approximate; closure and bridge are proxies.'}, indent=2), encoding='utf-8')
    print(f'[COMPARE] {out}', flush=True)


def background_components(p0, p1, gt):
    added = (p1 & ~p0 & ~gt).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(added, connectivity=8)
    _, old_labels = cv2.connectedComponents(p0.astype(np.uint8), connectivity=8)
    isolated = np.zeros_like(p0, dtype=bool)
    bridges = np.zeros_like(p0, dtype=bool)
    for i in range(1, count):
        component = labels == i
        near = cv2.dilate(component.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sources = np.unique(old_labels[near]); sources = sources[sources > 0]
        if len(sources) == 0:
            isolated |= component
        elif len(sources) >= 2:
            bridges |= component
    return isolated, bridges


def validate_direct(args, model, identity):
    from run_q3_r3_mass import full_dataset
    from run_reliable_directional_mass import evaluate_thresholds
    if args.dynamic_checkpoint is None:
        raise ValueError('val requires --dynamic_checkpoint from the direct V9 run')
    restore_checkpoint(model, args.dynamic_checkpoint, identity, 'D')
    thresholds = tuple(round(float(x), 2) for x in np.arange(.35, .751, .01))
    reports, _, _, _ = evaluate_thresholds(model, full_dataset(args.root_path, 'val'),
        args.device, thresholds, reference_threshold=args.reference_threshold)
    selected = max(reports, key=lambda t: reports[t]['selection_score'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'direct_val_selection.json').write_text(json.dumps({
        'selected_on': 'val', 'threshold': selected, 'identity': identity,
        'checkpoint': str(args.dynamic_checkpoint.resolve()), 'report': reports[selected],
        'threshold_curves': reports}, indent=2), encoding='utf-8')
    print(json.dumps(reports[selected], indent=2), flush=True)


@torch.no_grad()
def audit(args, model, identity):
    from run_q3_r3_mass import full_dataset, stitch_three
    if args.audit_stage == 'A_v8':
        model.state_gate = None
    else:
        if not args.dynamic_checkpoint:
            raise ValueError('Audit B/C/D requires --dynamic_checkpoint')
        restore_checkpoint(model, args.dynamic_checkpoint, identity, args.audit_stage)
    model.eval()
    dataset = full_dataset(args.root_path, 'val')
    limit = min(len(dataset), args.max_images or len(dataset))
    sums = {}
    for index in range(limit):
        sample = dataset[index]
        image = sample['image'][None].to(args.device)
        z0, z1, q = stitch_three(model, image)
        p0 = 1. / (1. + np.exp(-np.clip(z0, -80., 80.))) >= args.reference_threshold
        p1 = 1. / (1. + np.exp(-np.clip(z1, -80., 80.))) >= args.audit_threshold
        gt = sample['mask'][0].numpy() > .5
        isolated, bridges = background_components(p0, p1, gt)
        masks = {'recovered_gt': gt & ~p0 & p1, 'unrecovered_gt': gt & ~p0 & ~p1,
                 'hard_background': ~gt & (q >= .5), 'isolated_new_fp': isolated,
                 'false_bridge': bridges}
        owned = np.zeros_like(gt, dtype=bool)
        for y in sliding_positions(gt.shape[0], 512, 256):
            for x in sliding_positions(gt.shape[1], 512, 256):
                result = model(image[:, :, y:y+512, x:x+512], capture_diagnostics=True)
                h, w = result['r'].shape[-2:]
                owned_tile = ~owned[y:y+512, x:x+512]
                owned[y:y+512, x:x+512] = True
                per_region = {name: cv2.resize((mask[y:y+512, x:x+512] & owned_tile).astype(np.uint8),
                                               (w, h), interpolation=cv2.INTER_NEAREST) > 0
                              for name, mask in masks.items()}
                values = {'r': result['r'][0, 0], 'b_minus_p0': result['b'][0, 0]-result['u0'][0, 0],
                          'delta_z_plus': result['delta_z_plus'][0, 0],
                          'delta_z_minus': result['delta_z_minus'][0, 0],
                          'w_entropy': -(result['w'][0].clamp_min(1e-8) * result['w'][0].clamp_min(1e-8).log()).sum(0)}
                for step in range(3):
                    for key in ('g_step', 'alpha', 'beta', 'gamma', 'transport_rate',
                                'transport_delta', 'proposal_delta', 'anchor_delta', 'm', 'source_available'):
                        values[f'{key}_{step+1}'] = result[key][0, step, 0]
                    values[f'candidate_{step+1}'] = result['m'][0, step, 0] - result[f'u{step}'][0, 0]
                    values[f'high_g_negligible_transport_{step+1}'] = (
                        (result['g_step'][0, step, 0] > .9) &
                        (result['transport_delta'][0, step, 0].abs() < 1e-6)).float()
                for region, mask in per_region.items():
                    target = sums.setdefault(region, {})
                    if not mask.any():
                        continue
                    for key, value in values.items():
                        arr = value.detach().cpu().numpy()[mask].astype(np.float64)
                        low, high = ((0., 8.) if key == 'delta_z_plus' else
                                     (0., 4.) if key == 'delta_z_minus' else
                                     (0., float(np.log(8.))) if key == 'w_entropy' else (-1., 1.))
                        stat = target.setdefault(key, {'count': 0, 'sum': 0., 'abs_sum': 0.,
                            'hist': np.zeros(512, dtype=np.int64), 'low': low, 'high': high})
                        stat['count'] += int(arr.size); stat['sum'] += float(arr.sum())
                        stat['abs_sum'] += float(np.abs(arr).sum())
                        stat['hist'] += np.histogram(arr, bins=512, range=(low, high))[0]
        print(f'[AUDIT] {index+1}/{limit}', flush=True)
    summary = {region: {key: {'count': stat['count'], 'mean': stat['sum']/stat['count'],
                              'mean_abs': stat['abs_sum']/stat['count'],
                              **{f'p{int(q*100)}_hist_approx': float(stat['low'] +
                                  (min(int(np.searchsorted(stat['hist'].cumsum(), q*stat['count'])), 511) + .5) *
                                  (stat['high']-stat['low'])/512.) for q in (.1, .5, .9)}}
                        for key, stat in stats.items()} for region, stats in sums.items()}
    path = args.output_dir / f'audit_{args.audit_stage}_val.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'stage': args.audit_stage, 'identity': identity,
        'images': limit, 'p0_reference_threshold': args.reference_threshold,
        'p1_threshold': args.audit_threshold, 'regions': summary,
        'note': 'Full image binary regions projected to uniquely owned working-grid tile pixels; '
                'bridge is an old-component proximity proxy, not a GT connection label.'}, indent=2), encoding='utf-8')
    print(f'[AUDIT] {path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'stage0', 'smoke', 'train', 'resume', 'compare', 'audit', 'val'))
    p.add_argument('--root_path', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--initialization', choices=('v8', 'direct'), default='v8')
    p.add_argument('--q3_checkpoint', type=Path)
    p.add_argument('--v7_checkpoint', type=Path)
    p.add_argument('--v8_checkpoint', type=Path)
    p.add_argument('--relation_mode', choices=('full',), default='full')
    p.add_argument('--baseline_label', choices=('original', 'rebuilt'), default='original')
    p.add_argument('--stage', choices=('B', 'C', 'D'))
    p.add_argument('--previous_checkpoint', type=Path)
    p.add_argument('--b_checkpoint', type=Path); p.add_argument('--c_checkpoint', type=Path)
    p.add_argument('--d_checkpoint', type=Path)
    p.add_argument('--audit_stage', choices=('A_v8', 'B', 'C', 'D'), default='A_v8')
    p.add_argument('--dynamic_checkpoint', type=Path)
    p.add_argument('--audit_threshold', type=float, default=.6)
    p.add_argument('--device', default='cuda'); p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--val_interval', type=int, default=5); p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=4); p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--max_images', type=int, default=0); p.add_argument('--reference_threshold', type=float, default=.45)
    p.add_argument('--tolerance', type=float, default=1e-6)
    p.add_argument('--matching_bins', type=int, default=10000)
    p.add_argument('--lr_state_gate', type=float, default=1e-4)
    p.add_argument('--lr_gate_relation', type=float, default=5e-5)
    p.add_argument('--lr_rw_rates', type=float, default=1e-5)
    p.add_argument('--lr_visual_ba', type=float, default=5e-6)
    p.add_argument('--direct_lr', type=float, default=1e-4)
    args = p.parse_args()
    if args.initialization == 'v8' and (args.v7_checkpoint is None or args.v8_checkpoint is None):
        p.error('v8 initialization requires --v7_checkpoint and --v8_checkpoint')
    if args.initialization == 'direct':
        if args.q3_checkpoint is None:
            p.error('direct initialization requires --q3_checkpoint')
        if args.command in ('stage0', 'compare', 'preflight') or args.audit_stage == 'A_v8' and args.command == 'audit':
            p.error('Original V8 equivalence/comparison commands require v8 initialization; direct V9 is a separate experiment')
    if args.command in ('train', 'resume') and not args.stage:
        p.error('train requires --stage B, C or D')
    if args.command in ('train', 'resume', 'compare') and args.max_images:
        p.error('training/selection and comparison require the full validation set')
    if (args.epochs < 1 or args.batch_size < 1 or args.val_interval < 1 or args.num_workers < 0 or
            args.max_images < 0 or args.tolerance <= 0 or args.matching_bins < 1000 or not 0 < args.reference_threshold < 1 or
            not 0 < args.audit_threshold < 1):
        p.error('Invalid training configuration')
    args.device = torch.device(args.device)
    LR.update(state_gate=args.lr_state_gate, gate_relation=args.lr_gate_relation,
              reliable_head=args.lr_rw_rates, relations=args.lr_rw_rates, rates=args.lr_rw_rates,
              visual=args.lr_visual_ba, proposal_head=args.lr_visual_ba, accept_head=args.lr_visual_ba)
    if args.initialization == 'direct':
        LR.update({name: args.direct_lr for name in LR})
    if any(value <= 0 for value in LR.values()):
        p.error('Learning rates must be positive')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.command == 'preflight':
        preflight(args)
        return
    model, source, identity = load_initial(args)
    if args.command == 'stage0':
        stage0(args, model, identity)
    elif args.command == 'smoke':
        smoke(args, model, source, identity)
    elif args.command in ('train', 'resume'):
        train(args, model, source, identity)
    elif args.command == 'compare':
        compare(args, model, identity)
    elif args.command == 'val':
        if args.initialization != 'direct':
            p.error('Use compare for staged v8-based validation')
        validate_direct(args, model, identity)
    else:
        audit(args, model, identity)


if __name__ == '__main__':
    main()
