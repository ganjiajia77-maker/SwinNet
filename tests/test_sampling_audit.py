import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from diagnostics.sampling_audit import PhaseReadout,expansion_matrix_report,reconstruction_report,prior_intervention
from diagnostics.stage_feature_readout import LinearReadout,token_map
from networks.swin_transformer_unet_skip_expand_decoder_sys import FinalPatchExpand_X4


def test_phase_head_starts_equal_shared_and_has_phase_specific_gradient():
    torch.manual_seed(7)
    shared=LinearReadout(8)
    with torch.no_grad():shared.head.weight.normal_();shared.head.bias.fill_(.2)
    phase=PhaseReadout(shared)
    x=torch.randn(2,8,8,8)
    torch.testing.assert_close(shared(x),phase(x),rtol=1e-5,atol=1e-6)
    phase(x)[:,:,0::4,0::4].sum().backward()
    assert phase.weight.grad[0].abs().sum()>0
    assert phase.weight.grad[1:].abs().sum()==0


def test_expansion_reconstruction_and_rank():
    torch.manual_seed(9)
    up=FinalPatchExpand_X4((2,2),8)
    parent=torch.randn(1,4,8)
    raw=[]
    handle=up.norm.register_forward_pre_hook(lambda m,args:raw.append(token_map(args[0]).detach()))
    post=token_map(up(parent));handle.remove()
    report=expansion_matrix_report(up)
    assert report['full_column_rank'] and report['rank_at_float32_tolerance']==8
    reconstruction=reconstruction_report(up,dict(parent=token_map(parent),raw=raw[0],post=post))
    assert reconstruction['pre_layernorm_parent_reconstruction_max_abs']<1e-5
    assert reconstruction['layernorm_reconstruction_with_saved_stats_max_abs']<1e-5


def test_prior_intervention_restores_even_on_exception():
    merge=[nn.Parameter(torch.tensor(.2)),nn.Parameter(torch.tensor(.3))]
    bias=[nn.Parameter(torch.tensor(.4)),nn.Parameter(torch.tensor(.5))]
    core=SimpleNamespace(layers=[SimpleNamespace(downsample=SimpleNamespace(road_alpha=p)) for p in merge]+
        [SimpleNamespace(blocks=[SimpleNamespace(road_bias_scale_a1=bias[0],road_bias_scale_a2=bias[1])])])
    saved=[p.detach().clone() for p in merge+bias]
    with pytest.raises(ValueError):
        with prior_intervention(core,'both_off'):
            assert all(p==0 for p in merge+bias)
            raise ValueError('test')
    for p,v in zip(merge+bias,saved):torch.testing.assert_close(p,v,rtol=0,atol=0)
    with prior_intervention(core,'merge12_off'):
        assert all(p==0 for p in merge)
        assert all(p!=0 for p in bias)
