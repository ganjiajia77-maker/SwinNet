from types import SimpleNamespace
import torch

from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from networks.native_h0_relations import install_native_relations
from networks.h_pixel_reassembly import install_reassembly_from_args
from networks.encoder_h_exchange import install_encoder_h_exchange
from losses.road_losses import SurfaceStructureLoss


def base():
    torch.manual_seed(1234)
    core=SwinTransformerSys(img_size=128,window_size=8,num_classes=1,return_skeleton=True,
        structure_profile='stage23_boundary_0626',enable_highres_structure_stream=True,
        enable_h3_surface_fusion=True,enable_global_topology=True,stage_skeleton_mode='direct',
        remove_stage2_pre_topology_source=True)
    wrapper=SimpleNamespace(swin_unet=core)
    install_native_relations(wrapper)
    install_reassembly_from_args(wrapper,dict(h_pixel_reassembly=True,no_encoder_a1a2=True))
    return wrapper


def test_shared_initialization_and_zero_exchange_reproduce_c77():
    torch.set_num_threads(2)
    wrapper=base();core=wrapper.swin_unet;core.eval()
    image=torch.randn(1,3,128,128)
    with torch.no_grad():expected=core(image)[0]
    before={k:v.clone() for k,v in core.state_dict().items()}
    rng=torch.get_rng_state().clone()
    install_encoder_h_exchange(wrapper)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in before.items():assert torch.equal(v,core.state_dict()[k])
    with torch.no_grad():
        for module in core.encoder_h_exchanges:
            module.theta_eh.fill_(-100);module.theta_he.fill_(-100)
        actual=core(image)[0]
    assert torch.equal(actual,expected)


def test_three_premerge_exchanges_one_H_and_original_tasks_have_gradients():
    torch.set_num_threads(2)
    wrapper=base();core=wrapper.swin_unet
    install_encoder_h_exchange(wrapper);core.eval()
    events=[];handles=[];counts={'stem':0,'adapter':0}
    def counter(key):
        def hook(*args):counts[key]+=1
        return hook
    handles.append(core.prepatch_structure_encoder.register_forward_hook(counter('stem')))
    handles.append(core.highres_skeleton_adapter.register_forward_hook(counter('adapter')))
    for i in range(3):
        def block_hook(m,inputs,out,stage=i):events.append(('blocks',stage))
        def exchange_hook(m,inputs,out,stage=i):
            events.append(('exchange',stage))
            assert out[1].shape==(1,64,32,32)
        def merge_hook(m,inputs,stage=i):events.append(('merge',stage))
        handles.extend([core.layers[i].blocks[-1].register_forward_hook(block_hook),
                        core.encoder_h_exchanges[i].register_forward_hook(exchange_hook),
                        core.layers[i].downsample.register_forward_pre_hook(merge_hook)])
    captured={}
    def decoder_H(m,inputs,kwargs):captured['H']=kwargs['highres_skeleton_feat']
    handles.append(core.decoder_structure_blocks['2'].register_forward_pre_hook(decoder_H,with_kwargs=True))
    outputs=core(torch.randn(1,3,128,128))
    for handle in handles:handle.remove()
    assert counts=={'stem':1,'adapter':1}
    assert events==[(kind,i) for i in range(3) for kind in ('blocks','exchange','merge')]
    assert core.last_stage_features['H_decoder_input'] is core.last_encoder_features['H_E3']
    assert captured['H'] is core.last_encoder_features['H_E3']
    assert core.last_stage_features['H0'] is core.last_encoder_features['H_initial']
    assert [r['stage'] for r in outputs[4]]==[2,3]
    assert not hasattr(core,'encoder_stage1_road_attention_head')
    mask=torch.zeros(1,1,128,128);mask[:,:,60:67,8:120]=1
    skel=torch.zeros_like(mask);skel[:,:,63,8:120]=1
    criterion=SurfaceStructureLoss(stage_structure_weights=(0.,0.,.008,.012),
                                  stage_direction_factor=.2,stage_connectivity_factor=3.)
    criterion.native_relation_enabled=True
    surface=criterion.surface_loss(outputs[0],mask)[0]
    parameters=tuple(layer[0].weight for module in core.encoder_h_exchanges
                     for layer in (module.to_h,module.to_e))
    gradients=torch.autograd.grad(surface,parameters,retain_graph=True)
    assert all(g.norm()>0 for g in gradients)
    loss,_=criterion(outputs[0],surface_gt=mask,skeleton_gt=skel,skeleton_dilate_gt=skel,
                     stage_outputs=outputs[4],boundary_logits=outputs[1],skeleton_logits=outputs[2],connectivity_logits=outputs[3])
    loss.backward()
    for module in core.encoder_h_exchanges:
        for layer in (module.to_h[0],module.to_e[0]):
            assert layer.weight.grad is not None and layer.weight.grad.norm()>0
        assert module.theta_eh.grad is not None and module.theta_he.grad is not None
    assert core.h_pixel_reassembly.out.weight.grad.norm()>0
    # Same saved options recreate this architecture for strict inference loading.
    restored=base()
    install_reassembly_from_args(restored,dict(h_encoder_exchange=True))
    restored.swin_unet.load_state_dict(core.state_dict(),strict=True)
