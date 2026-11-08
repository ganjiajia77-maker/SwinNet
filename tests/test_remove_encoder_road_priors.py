from types import SimpleNamespace
import torch
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from networks.native_h0_relations import install_native_relations
from networks.h_pixel_reassembly import install_reassembly_from_args
from networks.remove_encoder_road_priors import remove_encoder_road_priors


def model():
    torch.manual_seed(1234)
    core=SwinTransformerSys(img_size=128,window_size=8,num_classes=1,return_skeleton=True,
        structure_profile='stage23_boundary_0626',enable_highres_structure_stream=True,
        enable_h3_surface_fusion=True,enable_global_topology=True,stage_skeleton_mode='direct',
        remove_stage2_pre_topology_source=True)
    wrapper=SimpleNamespace(swin_unet=core)
    install_native_relations(wrapper)
    install_reassembly_from_args(wrapper,dict(h_pixel_reassembly=True,h_reassembly_chunk_rows=16))
    return wrapper


def test_delete_exact_scope_and_preserve_initialization_rng():
    torch.set_num_threads(2)
    wrapper=model();core=wrapper.swin_unet
    before={k:v.clone() for k,v in core.state_dict().items()}
    rng=torch.get_rng_state().clone()
    remove_encoder_road_priors(wrapper)
    assert torch.equal(rng,torch.get_rng_state())
    after=core.state_dict()
    assert all(torch.equal(value,before[key]) for key,value in after.items())
    removed=set(before)-set(after)
    assert removed
    for key in removed:
        assert (key.startswith(('encoder_stage1_road_attention_head.','encoder_stage2_road_attention_head.',
                                'layers.0.road_attention_head.','layers.1.road_attention_head.'))
                or key in ('layers.0.downsample.road_alpha','layers.1.downsample.road_alpha')
                or (key.startswith('layers.2.blocks.') and '.road_bias_scale_a' in key)),key
    assert {'swin_unet.'+k for k in removed}==set(core.encoder_road_prior_ablation['removed_keys'])
    assert core.layers[0].road_attention_head is core.layers[1].road_attention_head is None
    assert not core.layers[2].use_road_bias
    for block in core.layers[2].blocks:
        assert not block.use_road_bias
        assert block._build_road_attention_bias((torch.ones(1,1,32,32),)*2) is None
    for prefix in ('patch_embed.edge_stem.','surface_h3_fusion.','h3_surface_proj.',
                   'native_h_surface_proj.','h_pixel_reassembly.','native_global_context.'):
        assert any(k.startswith(prefix) for k in after),prefix


def test_equivalent_to_zeroing_A_uses_and_live_H_reassembly_gradient():
    torch.set_num_threads(2)
    wrapper=model();core=wrapper.swin_unet;core.eval()
    # Reference: retain the A heads but disable only their two actual uses.
    with torch.no_grad():
        for i in (0,1):core.layers[i].downsample.road_alpha.zero_()
        for block in core.layers[2].blocks:
            block.road_bias_scale_a1.zero_();block.road_bias_scale_a2.zero_()
        core.h_pixel_reassembly.out.weight.normal_(std=.01)
    image=torch.randn(1,3,128,128)
    with torch.no_grad():reference=core(image)[0]
    remove_encoder_road_priors(wrapper)
    with torch.no_grad():actual=core(image)[0]
    assert torch.equal(reference,actual)
    outputs=core(image)
    assert not any('road_attention' in row for row in outputs[4])
    outputs[0].square().mean().backward()
    assert core.h_pixel_reassembly.out.weight.grad.norm()>0
    assert core.h_pixel_reassembly.query.weight.grad.norm()>0
    # Recreate from saved options and restore the reduced state strictly.
    restored=model()
    remove_encoder_road_priors(restored)
    restored.swin_unet.load_state_dict(core.state_dict(),strict=True)
