import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from h0_phase_training import inherit_h0_config, tensor_digests
from networks.encoder_h_postmerge_phase import (EncoderHPostMergePhase,
    install_encoder_h_postmerge_phase)
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


@pytest.mark.parametrize('factor', [2, 4, 8])
def test_phase_read_keeps_exact_subpixel_order_and_before_merge_leaves_E_untouched(factor):
    module=EncoderHPostMergePhase(8,factor,h_channels=8)
    h=torch.arange(8*16*16,dtype=torch.float32).reshape(1,8,16,16)
    observed={}
    hook=module.to_e[0].register_forward_pre_hook(lambda m,args: observed.update(phases=args[0].clone()))
    resolution=(16//factor,16//factor)
    result=module.after_merge(torch.zeros(1,resolution[0]*resolution[1],16),h,resolution)
    hook.remove()
    assert torch.equal(F.pixel_shuffle(observed['phases'],factor),h)
    assert observed['phases'][0,1,0,0]==h[0,0,0,1]
    assert observed['phases'][0,factor,0,0]==h[0,0,1,0]
    assert result.shape==(1,resolution[0]*resolution[1],16)
    e=torch.randn(1,16,8);before=e.clone()
    updated=module.before_merge(e,h,(4,4))
    assert torch.equal(e,before) and updated.shape==h.shape


def make_core():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    return SwinTransformerSys(img_size=128,window_size=8,num_classes=1,return_skeleton=True,
        structure_profile='stage23_boundary_0626',enable_highres_structure_stream=True,
        enable_h3_surface_fusion=True,enable_global_topology=True,stage_skeleton_mode='direct',
        remove_stage2_pre_topology_source=True)


def test_installation_preserves_every_shared_tensor_rng_and_original_decoder():
    core=make_core()
    before=tensor_digests(core)
    original_global=core.global_topology
    decoder_types=[type(core.decoder_structure_blocks[str(i)]) for i in (2,3)]
    random=torch.get_rng_state().clone()
    install_encoder_h_postmerge_phase(SimpleNamespace(swin_unet=core))
    assert torch.equal(random,torch.get_rng_state())
    after=tensor_digests(core)
    assert all(before[k]==after[k] for k in before if k in after)
    assert not set(before)-set(after)
    assert not any(part in name for name in core.state_dict()
                   for part in ('road_attention_head', 'road_bias_scale_a', 'road_alpha'))
    assert core.global_topology is original_global
    assert [type(core.decoder_structure_blocks[str(i)]) for i in (2,3)]==decoder_types
    assert not hasattr(core,'native_global_context') and not hasattr(core,'h_pixel_reassembly')


def test_zero_exchange_matches_same_model_without_encoder_exchange():
    core=make_core().eval()
    image=torch.randn(1,3,128,128)
    with torch.no_grad():expected=core(image)[0]
    target=make_core().eval()
    install_encoder_h_postmerge_phase(SimpleNamespace(swin_unet=target))
    own=target.state_dict()
    own.update(core.state_dict());target.load_state_dict(own,strict=True)
    with torch.no_grad():
        for module in target.encoder_h_postmerge_phases:
            module.theta_eh.fill_(-100);module.theta_he.fill_(-100)
        actual=target(image)[0]
    assert torch.equal(actual,expected)


def test_full_model_order_no_premerge_H_injection_live_H_and_original_task_gradients():
    core=make_core().eval()
    install_encoder_h_postmerge_phase(SimpleNamespace(swin_unet=core))
    events=[];handles=[];block_outputs={};counts={'stem':0,'adapter':0}
    def counter(key):
        def hook(*unused):counts[key]+=1
        return hook
    handles += [core.prepatch_structure_encoder.register_forward_hook(counter('stem')),
                core.highres_skeleton_adapter.register_forward_hook(counter('adapter'))]
    for stage in range(3):
        def blocks(m,args,output,i=stage):
            events.append(('blocks',i));block_outputs[i]=output.detach().clone()
        def write(m,args,output,i=stage):events.append(('E_to_H',i))
        def merge_pre(m,args,i=stage):
            events.append(('merge_pre',i));assert torch.equal(args[0],block_outputs[i])
        def merge_post(m,args,output,i=stage):events.append(('merge_post',i))
        def read(m,args,output,i=stage):events.append(('H_to_E',i))
        handles += [core.layers[stage].blocks[-1].register_forward_hook(blocks),
            core.encoder_h_postmerge_phases[stage].to_h.register_forward_hook(write),
            core.layers[stage].downsample.register_forward_pre_hook(merge_pre),
            core.layers[stage].downsample.register_forward_hook(merge_post),
            core.encoder_h_postmerge_phases[stage].to_e.register_forward_hook(read)]
    outputs=core(torch.randn(1,3,128,128))
    for hook in handles:hook.remove()
    assert events==[(kind,i) for i in range(3) for kind in ('blocks','E_to_H','merge_pre','merge_post','H_to_E')]
    assert counts=={'stem':1,'adapter':1}
    assert core.last_stage_features['H_decoder_input'] is core.last_encoder_features['H_E3']
    for name,h in core.last_encoder_features.items():assert h.shape==(1,64,32,32)
    assert core.highres_structure_skeleton_head is not None
    assert [row['stage'] for row in outputs[4]]==[2,3,'highres_structure']
    from losses.road_losses import SurfaceStructureLoss
    criterion=SurfaceStructureLoss(stage_structure_weights=(0.,0.,.008,.012),stage_direction_factor=.2,stage_connectivity_factor=3.)
    mask=torch.zeros(1,1,128,128);mask[:,:,60:67,8:120]=1
    skel=torch.zeros_like(mask);skel[:,:,63,8:120]=1
    surface=criterion.surface_loss(outputs[0],mask)[0]
    parameters=tuple(getattr(module,name)[0].weight for module in core.encoder_h_postmerge_phases for name in ('to_h','to_e'))
    gradients=torch.autograd.grad(surface,parameters,retain_graph=True)
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)
    loss,_=criterion(outputs[0],surface_gt=mask,skeleton_gt=skel,skeleton_dilate_gt=skel,
        boundary_logits=outputs[1],skeleton_logits=outputs[2],connectivity_logits=outputs[3],stage_outputs=outputs[4])
    loss.backward()
    assert all(p.grad is not None and p.grad.norm()>0 for p in parameters)
    saved=copy.deepcopy(core.state_dict())
    restored=make_core().eval()
    install_encoder_h_postmerge_phase(SimpleNamespace(swin_unet=restored))
    restored.load_state_dict(saved,strict=True)


def fake_reference(tmp_path):
    pretrained=tmp_path/'pretrained.pth';pretrained.write_bytes(b'the exact initialization file')
    saved=dict(max_epochs=80,img_size=512,source_patch_size=1500,batch_size=2,random_crop_train=True,
        random_crops_per_image=1,direct_resize_train=False,use_ema=False,amp_dtype='none',
        enable_highres_structure_stream=True,enable_global_topology=True,enable_h3_surface_fusion=True,
        highres_structure_channels=64,highres_structure_fusion_mode='stage23',stage_skeleton_mode='direct',
        max_train_batches=0,finetune_epochs=0,tiny_overfit_samples=0,no_pretrain=False,
        stage2_skeleton_weight=.008,stage3_skeleton_weight=.012,accumulation_steps=2,seed=1234,
        pretrain_ckpt=str(pretrained),output_dir='old_output',run_name='old_run',resume='old_resume')
    state={f'swin_unet.decoder_structure_blocks.{i}.highres_skeleton_fusion.stage_to_highres.0.weight':torch.ones(1) for i in (2,3)}
    path=tmp_path/'reference.pth';torch.save(dict(args=saved,model_state_dict=state,epoch=65),path)
    args=SimpleNamespace(**saved)
    args.encoder_h_postmerge_phase=True;args.global_topology_stage2=True;args.h0_config_checkpoint=str(path)
    args.output_dir='new_output';args.run_name='new_run';args.resume='';args.warm_start_ckpt=''
    import argparse
    parser=argparse.ArgumentParser()
    for key in saved:parser.add_argument('--'+key)
    return args,parser,path,saved


def test_reference_is_configuration_only_and_preserves_budget_accumulation_and_losses(tmp_path):
    args,parser,path,saved=fake_reference(tmp_path)
    args.max_epochs=1;args.accumulation_steps=1;args.stage2_skeleton_weight=.020
    report=inherit_h0_config(args,parser)
    assert args.max_epochs==80 and args.accumulation_steps==2 and args.stage2_skeleton_weight==.008
    assert args.output_dir=='new_output' and args.run_name=='new_run' and args.resume==''
    assert 'configuration only' in report['reference_role']
    assert 'model_state_dict' not in report
    assert report['reference_epoch']==65


@pytest.mark.parametrize('kind',['static','native_B','missing_interaction','EMA','warm_start'])
def test_rejects_wrong_reference_or_continued_training(tmp_path,kind):
    args,parser,path,saved=fake_reference(tmp_path)
    checkpoint=torch.load(path,weights_only=False)
    if kind=='static':checkpoint['args']['static_h0_stage_fusion']=True
    elif kind=='native_B':checkpoint['args']['native_relation_experiment']=True
    elif kind=='missing_interaction':checkpoint['model_state_dict']={}
    elif kind=='EMA':checkpoint['args']['use_ema']=True
    else:args.warm_start_ckpt='trained_road_model.pth'
    torch.save(checkpoint,path)
    with pytest.raises(ValueError):inherit_h0_config(args,parser)


def test_profile_defaults_cannot_overwrite_inherited_original_loss_budget(monkeypatch):
    source=Path(__file__).resolve().parents[1]/'train_image.py'
    module=ast.parse(source.read_text(encoding='utf-8'))
    functions=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in ('_cli_has','apply_structure_profile_defaults')]
    context=dict(sys=sys,h_phase_reference_report={'inherited_keys':['stage2_skeleton_weight','stage3_skeleton_weight','max_epochs']},
        STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626='stage23_boundary_0626',
        STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE='stage23_boundary_0626_final_ske')
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),context)
    args=SimpleNamespace(structure_profile='stage23_boundary_0626',stage2_skeleton_weight=.008,
        stage3_skeleton_weight=.012,warmup_epochs=10,max_epochs=80)
    monkeypatch.setattr(sys,'argv',['train_image.py','--encoder_h_postmerge_phase'])
    context['apply_structure_profile_defaults'](args)
    context['apply_structure_profile_defaults'](args)
    assert args.stage2_skeleton_weight==.008 and args.stage3_skeleton_weight==.012 and args.max_epochs==80


def test_real_original_loss_smoke_writes_a_passed_report_without_optimizer_steps(tmp_path):
    import json
    from h0_phase_training import smoke
    from losses.road_losses import SurfaceStructureLoss
    core=make_core()
    model=nn.Module();model.swin_unet=core
    # Module wrapper with the same SwinUnet forwarding contract.
    model.forward=lambda image: model.swin_unet(image)
    install_encoder_h_postmerge_phase(model)
    criterion=SurfaceStructureLoss(stage_structure_weights=(0.,0.,.008,.012),stage_direction_factor=.2,stage_connectivity_factor=3.)
    mask=torch.zeros(2,1,128,128);mask[:,:,60:67,8:120]=1
    skel=torch.zeros_like(mask);skel[:,:,63,8:120]=1
    loader=[dict(image=torch.randn(2,3,128,128),mask=mask,skeleton=skel,skeleton_dilate=skel)]
    reference=tmp_path/'reference.pth';reference.write_bytes(b'configuration reference')
    pretrain=tmp_path/'pretrain.pth';pretrain.write_bytes(b'ImageNet')
    args=SimpleNamespace(output_dir=str(tmp_path),batch_size=2,img_size=128,
        h_phase_initial_strength=.05,h0_config_checkpoint=str(reference),pretrain_ckpt=str(pretrain),seed=1234)
    smoke(model,criterion,loader,args,torch.device('cpu'))
    value=json.loads((tmp_path/'phase_smoke.json').read_text())
    assert value['passed'] and value['optimizer_steps']==0 and value['no_new_loss']
    assert value['H_consumed_by_decoder'] and value['H_shape']==[2,64,32,32]
    assert min(value['surface_gradient_norms'].values())>0
