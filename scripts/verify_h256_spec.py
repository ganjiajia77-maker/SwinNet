"""Targeted numerical, shape, initialization, checkpoint and CUDA cost checks."""
import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_protocol import (H256_ID, EXPERIMENT_BASE, BASE_COMMIT, architecture,
    build_model, initialize_from_h0, load_checkpoint, save_atomic)
from losses.h0_diffusion_loss import H0DiffusionLoss
from networks.encoder_h256 import EncoderHExchange, HSurfaceResidual
from networks.skeleton_guided_head import BidirectionalSkeletonFeatureFusion


def arc(mode='semantic_directional', **kwargs):
    return architecture(mode, architecture_id=H256_ID, **kwargs)


def strength(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def numerical_checks():
    torch.manual_seed(1234)
    exchange = EncoderHExchange(12, 8).eval()
    e, h = torch.randn(2,12,8,8), torch.randn(2,8,16,16)
    actual_e, actual_h = exchange(e,h)
    hbar = h + exchange.refine(h)
    expected_e = e + .05*exchange.to_e(F.adaptive_avg_pool2d(hbar,(8,8)))
    expected_h = hbar + .05*F.interpolate(exchange.to_h(e),(16,16),mode='bilinear',align_corners=False)
    torch.testing.assert_close(actual_e,expected_e)
    torch.testing.assert_close(actual_h,expected_h)
    assert not torch.equal(exchange(e+1,h)[1],actual_h)
    assert not torch.equal(exchange(e,h+1)[0],actual_e)
    fusion = BidirectionalSkeletonFeatureFusion(8,12,True).eval()
    g,h_next = fusion(h,e)
    hr,gr = fusion.highres_refine(h),fusion.stage_refine(e)
    torch.testing.assert_close(g,fusion.activation(gr+fusion.highres_to_stage(F.adaptive_avg_pool2d(hr,(8,8)))))
    torch.testing.assert_close(h_next,fusion.activation(hr+F.interpolate(fusion.stage_to_highres(gr),(16,16),mode='bilinear',align_corners=False)))
    residual = HSurfaceResidual(12,8).eval()
    surface = torch.randn(2,12,32,32)
    torch.testing.assert_close(residual(surface,h),surface,rtol=0,atol=0)
    with torch.no_grad(): residual.out.weight.normal_(std=.02)
    assert not torch.equal(residual(surface,h),residual(surface,h+1))
    return {'synchronous_encoder_formula':True,'native_decoder_formula':True,
            'E_changes_H_and_H_changes_E':True,'final_H_initial_identity_and_nonzero_effect':True}


def timed_cost(model, device, steps=3):
    model.to(device).train()
    criterion = H0DiffusionLoss().to(device)
    image = torch.randn(2,3,512,512,device=device)
    target = torch.zeros(2,1,512,512,device=device); target[:,:,250:254,25:485] = 1
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.0001)
    def step():
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss,terms = criterion(outputs[0],surface_gt=target,skeleton_gt=target,stage_outputs=outputs[4])
        loss.backward()
        assert torch.isfinite(loss)
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        return float(loss.detach())
    step()  # Includes AdamW state allocation before timing/peak reset.
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    losses=[step() for _ in range(steps)]
    torch.cuda.synchronize()
    training_ms=(time.perf_counter()-start)*1000/steps
    peak=torch.cuda.max_memory_allocated()
    reserved=torch.cuda.max_memory_reserved()
    del optimizer,image,target
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    model.eval(); image=torch.randn(1,3,512,512,device=device)
    with torch.inference_mode():
        for _ in range(2): model(image)
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(5): model(image)
        torch.cuda.synchronize(); inference_ms=(time.perf_counter()-start)*1000/5
    return dict(parameters=sum(p.numel() for p in model.parameters()),
        train_batch2_ms=training_ms, train_peak_allocated_MiB=peak/2**20,
        train_peak_reserved_MiB=reserved/2**20,inference_batch1_ms=inference_ms,
        synthetic_losses=losses, training_warmup_steps=1,training_timed_steps=steps,
        inference_warmup_passes=2,inference_timed_passes=5,
        scope='FP32 model/loss/backward/clip/AdamW training step; inference model only; no IO; CUDA synchronized')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrain_ckpt',required=True)
    p.add_argument('--output',default=str(ROOT/'H256_VERIFICATION.json'))
    args=p.parse_args()
    torch.set_num_threads(2)
    report={'base_experiment_commit':EXPERIMENT_BASE,'numerics':numerical_checks()}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        report['cuda_checks']='not run: CUDA unavailable'
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n'); return
    report.update(device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                  cuda_version=torch.version.cuda,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_benchmark=torch.backends.cudnn.benchmark)
    torch.manual_seed(1234)
    with patch('networks.swin_transformer_unet_skip_expand_decoder_sys.RoadAttentionHead',side_effect=AssertionError('A1/A2 constructed')), \
         patch('networks.swin_transformer_unet_skip_expand_decoder_sys.KeypointGuidedGlobalTopology',side_effect=AssertionError('global nodes constructed')), \
         patch('networks.skeleton_guided_head.PairwiseConnectivityHead',side_effect=AssertionError('C constructed')):
        model=build_model(arc())
    loaded,migration=initialize_from_h0(model,arc(),1234,args.pretrain_ckpt)
    report['initialization']=migration
    core=model.swin_unet
    forbidden=('road_attention_head','road_bias_scale','road_alpha','.connectivity_head.','.structure_gate.','.feature_residual.','global_topology.')
    assert not [k for k in model.state_dict() if any(term in k for term in forbidden)]
    assert core.prepatch_structure_encoder.down2[0].stride==(1,1)
    assert all(len(layer.blocks)==n for layer,n in zip(core.layers,[2,2,6,2]))
    assert all(len(layer.blocks)==n for layer,n in zip(core.layers_up[1:],[6,2,2]))
    events=[]; hooks=[]; layer_inputs={}; feature_results=[]
    for index,layer in enumerate(core.layers):
        hooks.append(layer.register_forward_pre_hook(lambda m,a,i=index: layer_inputs.__setitem__(i,a[0])))
        for block in layer.blocks:
            hooks.append(block.register_forward_hook(lambda m,a,o,i=index:events.append(('swin',i))))
        if index<3:
            hooks.append(core.encoder_h_exchanges[index].register_forward_hook(lambda m,a,o,i=index:events.append(('exchange',i))))
            def before_merge(module, inputs, i=index):
                assert inputs[0] is core.layers[i].last_pre_downsample
                events.append(('merge',i))
            hooks.append(layer.downsample.register_forward_pre_hook(before_merge))
    hooks.append(core.highres_skeleton_adapter.register_forward_hook(lambda m,a,o:events.append(('adapter',0))))
    original_features=core.forward_features
    def capture_features(*a,**kw):
        out=original_features(*a,**kw); feature_results.append(out); return out
    core.forward_features=capture_features
    for block in core.decoder_structure_blocks.values(): block.capture_diagnostics=True
    model.to(device).eval()
    report['shapes']={}
    for batch in (1,2):
        events.clear();feature_results.clear()
        with torch.inference_mode(): outputs=model(torch.randn(batch,3,512,512,device=device))
        assert outputs[0].shape==(batch,1,512,512)
        for name,tensor in core.last_encoder_h_features.items():
            assert tensor.shape==(batch,64,256,256), (name,tensor.shape)
        for name in ('H0','H_enc','H1','H2'):
            assert core.last_stage_features[name].shape==(batch,64,256,256)
        for name in ('G2','G3'): assert core.last_stage_features[name].shape==(batch,96,128,128)
        assert feature_results[0][3] is core.last_stage_features['H_enc']
        assert all(feature_results[0][1][i] is layer_inputs[i] for i in range(4))
        expected=[('adapter',0)]
        for i,n in enumerate((2,2,6,2)):
            expected.extend([('swin',i)]*n)
            if i<3: expected.extend([('exchange',i),('merge',i)])
        assert events==expected,events
        s2,s3=[entry for entry in outputs[4] if entry.get('stage') in (2,3)]
        assert s2['skeleton'].shape==(batch,1,128,128) and s3['direction'].shape==(batch,2,128,128)
        assert s2['direction'] is None and s2['connectivity'] is None and s3['connectivity'] is None
        second=core.decoder_structure_blocks['2'].last_diagnostics
        third=core.decoder_structure_blocks['3'].last_diagnostics
        assert second['X'] is second['out']
        assert third['H'].shape==(batch,64,256,256) and third['H_diff'].shape==(batch,64,128,128)
        assert third['previous_G2'] is core.last_stage_features['G2']
        report['shapes'][str(batch)]={k:list(v.shape) for k,v in core.last_stage_features.items()}
    for hook in hooks:hook.remove()
    core.forward_features=original_features
    for block in core.decoder_structure_blocks.values(): block.capture_diagnostics=False;block.last_diagnostics=None
    del outputs,feature_results,layer_inputs,s2,s3,second,third
    # A disabled exchange must not execute either R_i or a cross projection.
    core.encoder_h_exchange=False
    core.enable_h_surface_residual=False
    with patch.object(core.encoder_h_exchanges[0],'forward',side_effect=AssertionError('disabled exchange executed')), \
         patch.object(core.h_surface_residual,'forward',side_effect=AssertionError('disabled H residual executed')):
        with torch.inference_mode(): model(torch.randn(1,3,512,512,device=device))
    assert core.last_encoder_h_features['H0'] is core.last_encoder_h_features['H_E3']
    core.encoder_h_exchange=True
    core.enable_h_surface_residual=True
    report['ordering_adapter_skip_and_disabled_exchange_and_surface_residual']=True
    # Update both zero output projections before inspecting upstream gradients.
    criterion=H0DiffusionLoss().to(device)
    model.train();opt=torch.optim.SGD(model.parameters(),lr=.01)
    image=torch.randn(2,3,512,512,device=device)
    target=torch.zeros(2,1,512,512,device=device);target[:,:,250:254,25:485]=1
    outputs=model(image)
    loss,terms=criterion(outputs[0],surface_gt=target,skeleton_gt=target,stage_outputs=outputs[4])
    expected=terms['surface_loss']+.008*terms['S2']+.012*terms['S3']+.0024*terms['D3']
    torch.testing.assert_close(loss.detach(),expected)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert strength(core.decoder_structure_blocks['2'].skeleton_head)>0
    assert strength(core.decoder_structure_blocks['3'].skeleton_head)>0
    assert strength(core.decoder_structure_blocks['3'].direction_head)>0
    opt.step();opt.zero_grad(set_to_none=True)
    model(image)[0].square().mean().backward()
    for module in (core.encoder_h_exchanges,core.prepatch_structure_encoder,core.h_surface_residual.project,
                   core.decoder_structure_blocks['3'].diffusion.guide_h_proj):
        assert strength(module)>0
    report['surface_gradient_to_H_encoder_and_updated_projections']=True
    model.zero_grad(set_to_none=True);del opt,outputs,loss
    model.eval()
    with torch.inference_mode(): expected_logits=model(image[:1])[0].cpu()
    validation_dir = ROOT / '.validation'
    validation_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=validation_dir) as directory:
        path=Path(directory)/'model.pth'
        save_atomic(dict(architecture_id=H256_ID,architecture_config=arc(),base_h0_commit=BASE_COMMIT,
            base_experiment_commit=EXPERIMENT_BASE,model_config=model.config.dump(),
            model_state_dict=model.state_dict(),args={'use_ema':False}),path)
        restored,_=load_checkpoint(path);restored.to(device).eval()
        with torch.inference_mode():torch.testing.assert_close(restored(image[:1])[0].cpu(),expected_logits,rtol=0,atol=0)
        del restored
    core.last_stage_features = {}
    core.last_encoder_h_features = {}
    del image,target,core,original_features,capture_features
    model.cpu();gc.collect();torch.cuda.empty_cache()
    report['strict_checkpoint_roundtrip']=True
    report['H256_cost']=timed_cost(model,device)
    model.swin_unet.last_stage_features = {}
    model.swin_unet.last_encoder_h_features = {}
    model.cpu();del model;gc.collect();torch.cuda.empty_cache()
    baseline=build_model(architecture())
    initialize_from_h0(baseline,architecture(),1234,args.pretrain_ckpt)
    report['a668711_cost']=timed_cost(baseline,device)
    baseline.cpu();del baseline;gc.collect();torch.cuda.empty_cache()
    # Constructor/migration equality in all three control modes.
    digests=[]
    for mode in ('identity','semantic','semantic_directional'):
        torch.manual_seed(1234);control=build_model(arc(mode))
        initialize_from_h0(control,arc(mode),1234,args.pretrain_ckpt)
        value=hashlib.sha256()
        for k,v in control.state_dict().items():value.update(k.encode());value.update(v.cpu().numpy().tobytes())
        digests.append(value.hexdigest());del control
    assert len(set(digests))==1
    report['three_modes_complete_initial_state_equal']=digests[0]
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='initialization'},indent=2))


if __name__=='__main__':main()
