"""Remove the entire H path from the initialized ccc7992 phase model."""
import json
from pathlib import Path
import torch
from torch import nn


class GOnlyFeatureFusion(nn.Module):
    """Keep exactly the old G convolution and activation, without an H input."""
    def __init__(self, previous):
        super().__init__()
        self.stage_refine = previous.stage_refine
        self.activation = previous.activation

    def forward(self, highres_feature, stage_feature):
        if highres_feature is not None:
            raise ValueError('The complete-no-H model must not receive H features')
        return self.activation(self.stage_refine(stage_feature)), None


def install_complete_no_h(model):
    core = model.swin_unet
    if getattr(core, 'complete_no_h_ablation', False):
        raise ValueError('Complete-no-H ablation already installed')
    if not hasattr(core, 'encoder_h_postmerge_phases'):
        raise ValueError('First construct the exact initialized ccc7992 phase baseline')
    before = set(core.state_dict())
    del core.encoder_h_postmerge_phases
    for name in ('prepatch_structure_encoder', 'highres_skeleton_adapter',
                 'highres_structure_skeleton_head', 'highres_structure_fusion',
                 'structure_surface_correction_head', 'global_topology'):
        setattr(core, name, None)
    for stage in (2, 3):
        block = core.decoder_structure_blocks[str(stage)]
        block.highres_skeleton_fusion = GOnlyFeatureFusion(block.highres_skeleton_fusion)
        block.complete_no_h_ablation = True
    core.guided_head.post_refine_structure_interaction = None
    core.guided_head.enable_post_refine_structure_interaction = False
    core.enable_highres_structure_stream = False
    core.enable_global_topology = False
    core.enable_post_refine_structure_interaction = False
    core.complete_no_h_ablation = True
    core.last_encoder_features = {}
    core.last_highres_z_struct = core.last_z_struct = None
    core.last_highres_structure_skeleton = None
    core.no_h_removed_keys = ['swin_unet.'+key for key in sorted(before-set(core.state_dict()))]
    print('[COMPLETE NO H] H stem/exchanges/H fusion/global topology removed; '
          'G refinement, G2->G3, S/C/D, bottleneck context, G3 surface fusion retained.', flush=True)
    return model


def architecture_checks(model):
    core = model.swin_unet
    checks = dict(no_encoder_H=not hasattr(core, 'encoder_h_postmerge_phases'),
        no_H_stem=core.prepatch_structure_encoder is None,
        no_H_adapter=core.highres_skeleton_adapter is None,
        no_H_aux_head=core.highres_structure_skeleton_head is None,
        no_H_surface_modules=core.highres_structure_fusion is None and core.structure_surface_correction_head is None,
        no_H_global=core.global_topology is None and not core.enable_global_topology,
        bottleneck_context_retained=core.bottleneck_context_fusion is not None and core.bottleneck_swin_block is not None,
        stage3_context_retained=core.global_context_head is not None,
        G_only_fusions=all(isinstance(core.decoder_structure_blocks[str(i)].highres_skeleton_fusion,
                                     GOnlyFeatureFusion) for i in (2, 3)),
        G3_surface_retained=core.h3_surface_proj is not None and core.surface_h3_fusion is not None,
        no_A1A2=not any('road_attention_head' in key or 'road_bias_scale_a' in key for key in core.state_dict()))
    return checks


def task_parameters(model, include_heads=False):
    core = model.swin_unet
    parameters = {}
    for stage in (2, 3):
        block = core.decoder_structure_blocks[str(stage)]
        parameters[f'G{stage}_refine'] = block.highres_skeleton_fusion.stage_refine.block[0].weight
        if include_heads:
            parameters[f'S{stage}'] = block.skeleton_head.out.weight
            parameters[f'C{stage}'] = next(block.connectivity_head.parameters())
            parameters[f'D{stage}'] = block.direction_head[-1].weight
    return parameters


def task_gradients(model, include_heads=False):
    return {name: float(p.grad.float().norm()) if p.grad is not None else 0.
            for name, p in task_parameters(model, include_heads).items()}


def forward_stats(model):
    core = model.swin_unet
    return dict(variant='complete_no_h', checks=architecture_checks(model),
        G_shapes={f'G{i}': list(core.last_stage_features[f'G{i}'].shape) for i in (2, 3)},
        no_H_tensors=all(value is None for key, value in core.last_stage_features.items() if key.startswith('H')),
        H_consumed_by_decoder=False, H_global_enabled=False)


def smoke(model, criterion, loader, args, device):
    from h0_phase_training import file_digest, write_report
    model.train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(loader))
    images = batch['image'].to(device)
    mask, skel, dilated = (batch[key].to(device) for key in ('mask', 'skeleton', 'skeleton_dilate'))
    outputs = model(images)
    stages = {item['stage']: item for item in outputs[4]}
    if set(stages) != {2, 3}:
        raise RuntimeError('Expected only retained Stage2/3 S/C/D outputs')
    for stage in (2, 3):
        for field, channels in (('skeleton', 1), ('connectivity', 8), ('direction', 2)):
            value = stages[stage][field]
            if value.shape != (args.batch_size, channels, args.img_size//4, args.img_size//4):
                raise RuntimeError(f'Incorrect Stage{stage} {field} shape: {value.shape}')
    loss, _ = criterion(outputs[0], surface_gt=mask, skeleton_gt=skel,
        skeleton_dilate_gt=dilated, boundary_logits=outputs[1], skeleton_logits=outputs[2],
        connectivity_logits=outputs[3], stage_outputs=outputs[4])
    surface = criterion.surface_loss(outputs[0], mask)[0]
    named = task_parameters(model)
    grads = torch.autograd.grad(surface, tuple(named.values()), retain_graph=True, allow_unused=True)
    surface_grads = {name: float(grad.float().norm()) if grad is not None else 0.
                     for name, grad in zip(named, grads)}
    loss.backward()
    total_grads = task_gradients(model, include_heads=True)
    stats = forward_stats(model)
    passed = (bool(torch.isfinite(loss)) and all(stats['checks'].values()) and stats['no_H_tensors']
        and all(value > 0 and bool(torch.isfinite(torch.tensor(value)))
                for value in [*surface_grads.values(), *total_grads.values()]))
    report = dict(passed=passed, **stats, loss=float(loss.detach()),
        surface_gradient_norms=surface_grads, original_total_gradient_norms=total_grads,
        image_shape=list(images.shape), optimizer_steps=0, no_new_loss=True,
        reference_checkpoint_sha256=file_digest(args.h0_config_checkpoint),
        pretrain_sha256=file_digest(args.pretrain_ckpt), seed=args.seed,
        peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None)
    write_report(Path(args.output_dir)/'phase_smoke.json', report)
    print('[NO H SMOKE]', json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError('Complete-no-H smoke failed')
