import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from h0_phase_training import inherit_h0_config, tensor_digests
from networks.complete_no_h import (GOnlyFeatureFusion, architecture_checks,
    install_complete_no_h, smoke)
from networks.encoder_h_postmerge_phase import install_encoder_h_postmerge_phase
from scripts import evaluate_h0_encoder_phase as evaluate
from test_encoder_h_postmerge_phase import fake_reference, make_core


def make_model():
    core = make_core()
    model = nn.Module()
    model.swin_unet = core
    model.forward = lambda image: model.swin_unet(image)
    install_encoder_h_postmerge_phase(model)
    return model


def test_removal_preserves_every_surviving_tensor_and_G_refinement_and_rng():
    model = make_model()
    core = model.swin_unet
    before = tensor_digests(model)
    context = core.bottleneck_context_fusion
    stage3_context = core.global_context_head
    g_refs = [core.decoder_structure_blocks[str(i)].highres_skeleton_fusion.stage_refine for i in (2, 3)]
    g3 = core.h3_surface_proj
    fuse = core.surface_h3_fusion
    rng = torch.get_rng_state().clone()
    install_complete_no_h(model)
    after = tensor_digests(model)
    assert all(architecture_checks(model).values())
    assert set(after) < set(before)
    assert all(after[key] == before[key] for key in after)
    assert set(before)-set(after) == set(core.no_h_removed_keys)
    assert torch.equal(torch.get_rng_state(), rng)
    assert core.h3_surface_proj is g3 and core.surface_h3_fusion is fuse
    assert [core.decoder_structure_blocks[str(i)].highres_skeleton_fusion.stage_refine for i in (2, 3)] == g_refs
    assert core.bottleneck_context_fusion is context
    assert core.global_context_head is stage3_context
    assert not any('encoder_h_postmerge_phases' in key or 'global_topology.' in key
                   or 'prepatch_structure_encoder.' in key for key in after)
    assert not any('highres_to_stage' in key or 'stage_to_highres' in key or 'highres_refine' in key for key in after)


def test_G_refinement_keeps_original_convolution_and_activation_without_H():
    from networks.skeleton_guided_head import BidirectionalSkeletonFeatureFusion
    original = BidirectionalSkeletonFeatureFusion(8, 16).eval()
    sample = torch.randn(2, 16, 8, 8, requires_grad=True)
    expected = original.activation(original.stage_refine(sample))
    replacement = GOnlyFeatureFusion(original).eval()
    actual, h = replacement(None, sample)
    assert torch.equal(expected, actual) and h is None
    actual.sum().backward()
    assert replacement.stage_refine.block[0].weight.grad.norm() > 0
    with pytest.raises(ValueError, match='must not receive H'):
        replacement(torch.zeros(2, 8, 8, 8), sample)


def test_real_original_loss_smoke_has_G_S_C_D_gradients_and_only_two_stage_outputs(tmp_path):
    from losses.road_losses import SurfaceStructureLoss
    model = make_model()
    install_complete_no_h(model)
    mask = torch.zeros(2, 1, 128, 128)
    mask[:, :, 60:67, 8:120] = 1
    skel = torch.zeros_like(mask)
    skel[:, :, 63, 8:120] = 1
    loader = [dict(image=torch.randn(2, 3, 128, 128), mask=mask,
                   skeleton=skel, skeleton_dilate=skel)]
    criterion = SurfaceStructureLoss(stage_structure_weights=(0., 0., .008, .012),
                                    stage_direction_factor=.2, stage_connectivity_factor=3.)
    reference = tmp_path/'reference.pth'; reference.write_bytes(b'config only')
    pretrain = tmp_path/'pretrain.pth'; pretrain.write_bytes(b'ImageNet')
    args = SimpleNamespace(output_dir=str(tmp_path), batch_size=2, img_size=128,
        h0_config_checkpoint=str(reference), pretrain_ckpt=str(pretrain), seed=1234)
    smoke(model, criterion, loader, args, torch.device('cpu'))
    report = json.loads((tmp_path/'phase_smoke.json').read_text())
    assert report['passed'] and report['optimizer_steps'] == 0 and report['no_new_loss']
    assert all(report['checks'].values()) and report['no_H_tensors']
    assert not report['H_consumed_by_decoder'] and not report['H_global_enabled']
    assert min(report['surface_gradient_norms'].values()) > 0
    assert set(report['original_total_gradient_norms']) == {'G2_refine', 'G3_refine', 'S2', 'S3', 'C2', 'C3', 'D2', 'D3'}
    assert min(report['original_total_gradient_norms'].values()) > 0


def test_H_phase_baseline_forward_is_unchanged_when_ablation_not_installed():
    model = make_model().eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        first = model(image)[0]
        second = model(image)[0]
    assert torch.equal(first, second)
    assert model.swin_unet.last_stage_features['H_decoder_input'] is model.swin_unet.last_encoder_features['H_E3']


def test_strict_no_H_checkpoint_restore_and_wrong_variant_rejected(tmp_path, monkeypatch):
    saved = dict(encoder_h_postmerge_phase=True, complete_no_h_ablation=True, use_ema=False,
        img_size=128, structure_profile='stage23_boundary_0626', stage_skeleton_mode='direct',
        enable_highres_structure_stream=True, highres_structure_channels=64,
        enable_global_topology=True, enable_h3_surface_fusion=True, remove_stage2_pre_topology_source=True)
    original = evaluate.build_model(saved).eval()
    state = dict(args=saved, epoch=80, model_state_dict=copy.deepcopy(original.state_dict()))
    checkpoint = tmp_path/'best.pth'; checkpoint.write_bytes(b'identity')
    monkeypatch.setattr(evaluate.torch, 'load', lambda *a, **k: state)
    restored, _, epoch = evaluate.load_model(checkpoint, torch.device('cpu'))
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.equal(original(image)[0], restored(image)[0])
    assert epoch == 80 and all(architecture_checks(restored).values())
    state['model_state_dict'].pop('swin_unet.h3_surface_proj.weight')
    with pytest.raises(RuntimeError, match='Missing key'):
        evaluate.load_model(checkpoint, torch.device('cpu'))
    from scripts.evaluate_h0_no_h import build_model
    with pytest.raises(ValueError, match='complete-no-H'):
        build_model(dict(saved, complete_no_h_ablation=False))


def test_no_H_inherits_exact_budget_and_loss_settings_and_records_effective_architecture(tmp_path):
    args, parser, path, saved = fake_reference(tmp_path)
    args.complete_no_h_ablation = True
    report = inherit_h0_config(args, parser)
    assert report['variant'] == 'complete_no_h'
    assert report['effective_architecture']['H_stream'] is False
    assert report['effective_architecture']['G3_surface_fusion'] is True
    assert args.complete_no_h_ablation and args.max_epochs == 80
    assert args.stage2_skeleton_weight == .008 and args.stage3_skeleton_weight == .012
    assert args.use_ema is False and args.amp_dtype == 'none'
    assert args.disable_persistent_workers and args.random_crop_train
    assert args.warm_start_ckpt == '' and args.resume == ''
