import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import pytest
import torch
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from networks.native_h0_relations import install_native_relations
from diagnostics.stage_feature_readout import StageCapture, LinearReadout, token_map, road_loss


def test_token_layout_preserves_spatial_coordinates():
    tokens=torch.arange(2*16*3).reshape(2,16,3)
    feature=token_map(tokens)
    assert feature.shape==(2,3,4,4)
    assert torch.equal(feature[1,:,2,3],tokens[1,11])
    with pytest.raises(ValueError):token_map(torch.zeros(1,15,3))


def test_every_sampling_stage_and_readonly_hooks():
    torch.set_num_threads(2);torch.manual_seed(11)
    core=SwinTransformerSys(img_size=128,window_size=8,num_classes=1,return_skeleton=True,
        structure_profile='stage23_boundary_0626',enable_highres_structure_stream=True,
        enable_h3_surface_fusion=True,enable_global_topology=True,stage_skeleton_mode='direct',
        remove_stage2_pre_topology_source=True)
    model=SimpleNamespace(swin_unet=core);install_native_relations(model);core.eval().requires_grad_(False)
    before={k:v.clone() for k,v in core.state_dict().items()}
    image=torch.randn(1,3,128,128)
    with torch.no_grad():expected=core(image)[0].clone()
    capture=StageCapture(model)
    with torch.no_grad():actual=core(image)[0]
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    values=capture.values
    for i in (1,2,3):
        pre,post=values[f'enc{i}_blocks_before_down'],values[f'enc{i}_after_down']
        assert pre.shape[-1]==post.shape[-1]*2
        assert pre.shape[1]*2==post.shape[1]
    for i in (1,2):
        pre,post=values[f'dec{i}_blocks_before_up'],values[f'dec{i}_after_up']
        assert pre.shape[-1]*2==post.shape[-1]
        assert pre.shape[1]==post.shape[1]*2
    for key in ('enc4_after_blocks','dec0_after_up','dec3_after_blocks','dec2_after_structure',
                'dec3_after_structure','H_stage2_before_global','H_stage2_after_global',
                'H_stage3_before_relation','H_final_after_relation','final_x4_after_up',
                'final_H_G3_surface_fusion','surface_head_input_after_boundary',
                'decoder_final_norm','dec2_DCA_refined_skip','dec3_DCA_refined_skip'):
        assert key in values,key
    assert values['final_x4_after_up'].shape[-2:]==(128,128)
    readout=LinearReadout(values['enc2_after_down'].shape[1])
    loss=road_loss(readout(values['enc2_after_down'],(128,128)),torch.ones(1,1,128,128))
    loss.backward()
    assert readout.head.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in core.parameters())
    for key,value in core.state_dict().items():torch.testing.assert_close(before[key],value,rtol=0,atol=0)
    assert all(not v.requires_grad for v in capture.values.values())
    capture.close();assert not capture.handles and not capture.values
