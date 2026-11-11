"""Configuration inheritance and startup checks for original-H0 phase exchange."""
import hashlib
import json
from pathlib import Path

import torch

SAMPLING_PROTOCOL = 'mass-h0-shared-epoch-affine-crop-v1-workers-restarted'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digests(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def inherit_h0_config(args, parser):
    """Read only saved args, never load trained reference weights into the model."""
    if getattr(args, 'complete_no_h_ablation', False) and not args.encoder_h_postmerge_phase:
        raise ValueError('Complete-no-H must use the inherited ccc7992 phase training protocol')
    if not args.encoder_h_postmerge_phase:
        return None
    path = Path(args.h0_config_checkpoint)
    if not path.is_file():
        raise FileNotFoundError('Specify the original full H0 configuration checkpoint: '+str(path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint.get('args')
    if not isinstance(saved, dict):
        raise ValueError('Original H0 checkpoint must contain its training args')
    for key in ('native_relation_experiment', 'static_h0_stage_fusion', 'h_pixel_reassembly',
                'h_encoder_exchange', 'encoder_h_postmerge_phase', 'complete_no_h_ablation'):
        if saved.get(key):
            raise ValueError('Configuration reference is not the original full H0: '+key)
    state = checkpoint.get('model_state_dict', {})
    for stage in (2, 3):
        key = f'swin_unet.decoder_structure_blocks.{stage}.highres_skeleton_fusion.stage_to_highres.0.weight'
        if key not in state:
            raise ValueError('Reference lacks original dynamic H/G interaction: '+key)
    runtime = {'root_path', 'output_dir', 'run_name', 'cfg', 'pretrain_ckpt', 'resume',
               'warm_start_ckpt', 'h0_config_checkpoint', 'encoder_h_postmerge_phase',
               'h_phase_initial_strength', 'h_phase_smoke_only', 'h_phase_audit_interval',
               'h_phase_smoke_manifest', 'complete_no_h_ablation'}
    inherited = []
    for action in parser._actions:
        key = action.dest
        if key in saved and key not in runtime:
            setattr(args, key, saved[key])
            inherited.append(key)
    # Explicit user override: recreate workers each epoch. Keep the original
    # shared epoch as well, so the dataset itself is safe under either policy.
    args.disable_persistent_workers = True
    args.sampling_protocol = SAMPLING_PROTOCOL
    if args.warm_start_ckpt:
        raise ValueError('This experiment starts from ImageNet, not a trained road checkpoint')
    required = dict(max_epochs=80, img_size=512, source_patch_size=1500, batch_size=2,
        random_crop_train=True, random_crops_per_image=1, direct_resize_train=False,
        use_ema=False, amp_dtype='none', enable_highres_structure_stream=True,
        enable_global_topology=True, enable_h3_surface_fusion=True, highres_structure_channels=64,
        highres_structure_fusion_mode='stage23', stage_skeleton_mode='direct',
        max_train_batches=0, finetune_epochs=0, tiny_overfit_samples=0, no_pretrain=False)
    mismatch = {key: dict(actual=getattr(args, key, None), required=value)
                for key, value in required.items() if getattr(args, key, None) != value}
    if mismatch:
        raise ValueError('Original H0 configuration differs from the full-80-epoch protocol: '+str(mismatch))
    reference_pretrain = saved.get('pretrain_ckpt')
    if not args.pretrain_ckpt:
        args.pretrain_ckpt = reference_pretrain or './pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth'
    pretrain_sha = file_digest(args.pretrain_ckpt)
    if reference_pretrain and Path(reference_pretrain).is_file() and file_digest(reference_pretrain) != pretrain_sha:
        raise ValueError('ImageNet initialization differs from the reference H0 pretrain file')
    report = dict(base_commit='599a410de8d341850e1810417307e4b3bbdfc82d',
        reference_checkpoint=str(path.resolve()), reference_checkpoint_sha256=file_digest(path),
        reference_role='configuration only; trained model weights are never loaded',
        reference_epoch=checkpoint.get('epoch'), inherited_keys=inherited,
        reference_args=saved, pretrain_sha256=pretrain_sha,
        cfg_sha256=file_digest(args.cfg) if hasattr(args, 'cfg') and Path(args.cfg).is_file() else None,
        accumulation_steps_inherited=args.accumulation_steps)
    report['sampling_protocol'] = SAMPLING_PROTOCOL
    report['explicit_overrides'] = dict(disable_persistent_workers=dict(
        reference=saved.get('disable_persistent_workers', False), actual=True,
        reason='User requested restarting workers and different crop positions every epoch'))
    if getattr(args, 'complete_no_h_ablation', False):
        if getattr(args, 'highres_structure_skeleton_weight', 0.) != 0.:
            raise ValueError('H-specific auxiliary loss must already have zero weight in this ablation')
        report['variant'] = 'complete_no_h'
        report['effective_architecture'] = dict(H_stream=False, H_encoder_exchange=False,
            H_decoder_exchange=False, H_global_topology=False, A1A2=False,
            G_refinement=True, G3_surface_fusion=True,
            construction_args='Original H0 is initialized first; H-specific modules are then removed')
    del checkpoint
    return report


def write_report(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n', encoding='utf-8')


def phase_gradients(model):
    if getattr(model.swin_unet, 'complete_no_h_ablation', False):
        from networks.complete_no_h import task_gradients
        return task_gradients(model)
    result = {}
    for index, module in enumerate(model.swin_unet.encoder_h_postmerge_phases, 1):
        for name in ('to_h', 'to_e'):
            grad = getattr(module, name)[0].weight.grad
            result[f'stage{index}_{name}'] = float(grad.float().norm()) if grad is not None else 0.
    return result


def phase_stats(model):
    if getattr(model.swin_unet, 'complete_no_h_ablation', False):
        from networks.complete_no_h import forward_stats
        return forward_stats(model)
    core = model.swin_unet
    return dict(stages=[{key: float(value) for key, value in module.last_stats.items()}
                        for module in core.encoder_h_postmerge_phases],
                H_shape=list(core.last_encoder_features['H_E3'].shape),
                H_consumed_by_decoder=core.last_stage_features['H_decoder_input'] is core.last_encoder_features['H_E3'])


def smoke(model, criterion, loader, args, device):
    if getattr(model.swin_unet, 'complete_no_h_ablation', False):
        from networks.complete_no_h import smoke as no_h_smoke
        return no_h_smoke(model, criterion, loader, args, device)
    model.train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(loader))
    images = batch['image'].to(device)
    mask, skel, dilated = (batch[key].to(device) for key in ('mask', 'skeleton', 'skeleton_dilate'))
    outputs = model(images)
    loss, _ = criterion(outputs[0], surface_gt=mask, skeleton_gt=skel,
        skeleton_dilate_gt=dilated, boundary_logits=outputs[1], skeleton_logits=outputs[2],
        connectivity_logits=outputs[3], stage_outputs=outputs[4])
    surface = criterion.surface_loss(outputs[0], mask)[0]
    names, parameters = [], []
    for index, module in enumerate(model.swin_unet.encoder_h_postmerge_phases, 1):
        for name in ('to_h', 'to_e'):
            names.append(f'stage{index}_{name}')
            parameters.append(getattr(module, name)[0].weight)
    grads = torch.autograd.grad(surface, parameters, retain_graph=True, allow_unused=True)
    surface_grads = {name: float(grad.float().norm()) if grad is not None else 0.
                     for name, grad in zip(names, grads)}
    loss.backward()
    total_grads = phase_gradients(model)
    stats = phase_stats(model)
    passed = (torch.isfinite(loss).item() and stats['H_consumed_by_decoder'] and
        stats['H_shape'] == [args.batch_size, 64, args.img_size//4, args.img_size//4] and
        all(value > 0 and torch.isfinite(torch.tensor(value)) for value in [*surface_grads.values(), *total_grads.values()]))
    report = dict(passed=bool(passed), loss=float(loss.detach()),
        surface_gradient_norms=surface_grads, original_total_gradient_norms=total_grads,
        image_shape=list(images.shape), **stats,
        peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
        optimizer_steps=0, no_new_loss=True, initial_strength=args.h_phase_initial_strength,
        reference_checkpoint_sha256=file_digest(args.h0_config_checkpoint),
        pretrain_sha256=file_digest(args.pretrain_ckpt), seed=args.seed)
    write_report(Path(args.output_dir)/'phase_smoke.json', report)
    print('[PHASE SMOKE]', json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError('Encoder-H post-merge smoke check failed')
