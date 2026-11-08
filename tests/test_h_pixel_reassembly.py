from types import SimpleNamespace
import copy
import torch
import pytest

from networks.h_pixel_reassembly import HPixelReassembly, install_h_pixel_reassembly
from networks.native_h0_relations import install_native_relations
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE


def inputs():
    torch.manual_seed(7)
    return [torch.randn(2,96,16,20,requires_grad=True),
            torch.randn(2,64,4,5,requires_grad=True),
            torch.randn(2,96,4,5,requires_grad=True),
            torch.randn(2,8,4,5,requires_grad=True)]


def test_identity_bounds_and_per_pixel_weights():
    module=HPixelReassembly(chunk_rows=3).eval()
    x,h,g,c=inputs()
    delta,weights=module(x,h,g,c,return_weights=True)
    assert torch.equal(delta,torch.zeros_like(x))
    torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]))
    for d,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        if dy<0: assert weights[:,d+1,:4].count_nonzero()==0
        if dy>0: assert weights[:,d+1,-4:].count_nonzero()==0
        if dx<0: assert weights[:,d+1,:,:4].count_nonzero()==0
        if dx>0: assert weights[:,d+1,:,-4:].count_nonzero()==0
    # Distinct queries inside one coarse cell must produce distinct source choices.
    assert not torch.allclose(weights[:,:,4,4],weights[:,:,5,5])


def test_reverse_C_and_detach_and_uniform():
    module=HPixelReassembly()
    c=torch.full((1,8,3,3),-100.,requires_grad=True)
    d=CONNECTIVITY_DIRECTIONS.index((0,1));reverse=CONNECTIVITY_OPPOSITE[d]
    with torch.no_grad():
        c[0,d,1,1]=100.;c[0,reverse,1,2]=100.
    prior=module.log_priors(c).exp()
    assert prior[0,0,1,1]==1
    torch.testing.assert_close(prior[0,d+1,1,1],torch.tensor(1.))
    torch.testing.assert_close(prior[0,d+1,1,0],torch.tensor(.1))
    assert not prior.requires_grad
    module.prior_mode='uniform'
    assert torch.equal(module.log_priors(c),torch.zeros_like(prior))


def test_C_changes_choices_and_control_has_identical_parameters():
    torch.manual_seed(19)
    predicted=HPixelReassembly(prior_mode='predicted').eval()
    torch.manual_seed(19)
    uniform=HPixelReassembly(prior_mode='uniform').eval()
    for key,value in predicted.state_dict().items():
        assert torch.equal(value,uniform.state_dict()[key])
    x,h,g,c=inputs()
    _,with_c=predicted(x,h,g,c,return_weights=True)
    _,without_c=uniform(x,h,g,c,return_weights=True)
    assert not torch.allclose(with_c,without_c)


def test_gradients_activate_after_output_step_and_C_has_no_new_gradient():
    torch.set_num_threads(2)
    module=HPixelReassembly(chunk_rows=3).train()
    x,h,g,c=inputs()
    delta=module(x,h,g,c)
    delta.square().mean().add(delta.mean()).backward()
    assert module.out.weight.grad.norm()>0
    assert module.query.weight.grad.count_nonzero()==0
    assert c.grad is None
    with torch.no_grad():module.out.weight.add_(-.01*module.out.weight.grad)
    module.zero_grad(set_to_none=True)
    delta=module(x,h,g,c)
    (delta.square().mean()+delta.mean()).backward()
    for layer in (module.query,module.key,module.value,module.position[0]):
        assert layer.weight.grad.norm()>0
    assert c.grad is None
    assert h.grad.norm()>0 and g.grad.norm()>0 and x.grad.norm()>0


def test_chunks_checkpoint_and_unfolded_reference_agree():
    torch.set_num_threads(2)
    torch.manual_seed(15)
    module=HPixelReassembly(chunk_rows=3).train()
    with torch.no_grad():module.out.weight.normal_(std=.01)
    # Independent full source stack reference: allowed only in this tiny test.
    x,h,g,c=inputs()
    q=module.query(x);k=module.key(torch.cat((h,g),1));v=module.value(h)
    priors=module.log_priors(c)
    y=torch.arange(16)[:,None].expand(16,20);z=torch.arange(20)[None,:].expand(16,20)
    py,px=y//4,z//4
    scores=[];values=[]
    for i,(dy,dx) in enumerate([(0,0)]+list(CONNECTIVITY_DIRECTIONS)):
        sy,sx=py+dy,px+dx
        index=(sy.clamp(0,3)*5+sx.clamp(0,4)).flatten()
        key=k.flatten(2)[:,:,index].reshape(2,16,16,20)
        value=v.flatten(2)[:,:,index].reshape(2,16,16,20)
        pos=torch.stack(((z%4+.5)/4-.5,(y%4+.5)/4-.5,
                         torch.full_like(z,dx).float(),torch.full_like(y,dy).float()),-1)
        score=(q*key).sum(1,keepdim=True)/4+module.position(pos).permute(2,0,1)[None]
        score=score+priors[:,i:i+1].repeat_interleave(4,2).repeat_interleave(4,3)
        scores.append(score.masked_fill(~((sy>=0)&(sy<4)&(sx>=0)&(sx<5))[None,None],-torch.inf))
        values.append(value)
    weights=torch.softmax(torch.cat(scores,1),1)
    stacked=torch.stack(values,1)
    difference=(weights[:,:,None]*(stacked-values[0][:,None])).sum(1)
    reference=module.out(difference)
    actual=module(x,h,g,c)
    torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)
    params=(x,h,g,module.query.weight,module.key.weight,module.value.weight,module.position[0].weight,module.out.weight)
    ga=torch.autograd.grad(actual.square().mean(),params,retain_graph=True)
    gb=torch.autograd.grad(reference.square().mean(),params)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)


def test_full_native_B_initialization_output_loss_and_strict_restore():
    torch.set_num_threads(2)
    options=dict(img_size=128,window_size=8,num_classes=1,return_skeleton=True,
                 structure_profile='stage23_boundary_0626',enable_highres_structure_stream=True,
                 enable_h3_surface_fusion=True,enable_global_topology=True,stage_skeleton_mode='direct',
                 remove_stage2_pre_topology_source=True)
    torch.manual_seed(1234)
    core=SwinTransformerSys(**options)
    model=SimpleNamespace(swin_unet=core)
    install_native_relations(model);core.eval()
    before={k:v.clone() for k,v in core.state_dict().items()}
    state=torch.get_rng_state().clone()
    image=torch.randn(1,3,128,128)
    with torch.no_grad():old=core(image)[0]
    rng=torch.get_rng_state().clone()
    install_h_pixel_reassembly(model,chunk_rows=16)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in before.items():assert torch.equal(value,core.state_dict()[key])
    with torch.no_grad():new=core(image)[0]
    assert torch.equal(old,new)
    saved=copy.deepcopy(core.state_dict())
    core.load_state_dict(saved,strict=True)
    # Same loss input means the entire original surface loss is identical.
    from losses.road_losses import SurfaceStructureLoss
    criterion=SurfaceStructureLoss()
    gt=torch.zeros_like(old);gt[:,:,60:65]=1
    assert torch.equal(criterion.surface_loss(old,gt)[0],criterion.surface_loss(new,gt)[0])
    criterion.surface_loss(core(image)[0],gt)[0].backward()
    assert core.h_pixel_reassembly.out.weight.grad.norm()>0
