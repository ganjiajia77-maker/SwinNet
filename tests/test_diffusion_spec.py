import contextlib
import gc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diffusion_protocol import (ARCHITECTURE_ID, BASE_COMMIT, architecture, build_model,
                                initialize_from_h0, load_checkpoint, save_atomic)
from diffusion_experiment import benchmark, compare, export_masks, model_name
from losses.h0_diffusion_loss import H0DiffusionLoss
from losses.road_losses import SurfaceStructureLoss
from networks.directional_diffusion import SemanticDirectionalDiffusion, neighbor
from networks.skeleton_guided_head import H0SkeletonDiffusionStage
from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE

PRETRAIN=Path(os.environ.get('H0_PRETRAIN_CKPT', str(ROOT.parents[1]/'pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth')))
EVIDENCE={}


def strength(module):
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)


class NumericalTests(unittest.TestCase):
    def test_native1500_export_validation_selection_and_unified_test(self):
        class EmptyRoadModel(torch.nn.Module):
            def forward(self, image):
                return torch.full_like(image[:, :1], -10.), None, None

        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as temporary:
            temporary=Path(temporary)
            data, report=temporary/'data', temporary/'report'
            for split in ('val','test'):
                (data/split).mkdir(parents=True)
                (data/f'{split}_labels').mkdir()
                image=np.zeros((1500,1500,3),dtype=np.uint8)
                label=np.zeros((1500,1500),dtype=np.uint8)
                cv2.line(label,(400,750),(1100,750),255,5)
                self.assertTrue(cv2.imwrite(str(data/split/'example.tiff'),image))
                self.assertTrue(cv2.imwrite(str(data/f'{split}_labels'/'example.tif'),label))
            args=SimpleNamespace(root_path=str(data),output_dir=str(report),seed=1234,
                                 model_path=str(temporary/'example_checkpoint.pth'))
            checkpoint={'epoch':5,'architecture_config':architecture(),
                        'args':{'seed':1234},
                        'initialization_manifest':{'shared_tensor_sha256':'synthetic'}}
            model=EmptyRoadModel()
            export_masks(model,args,'val',[.45,.5],torch.device('cpu'))
            compare(args,checkpoint,'val')
            selection=report/'val'/'val_selection.json'
            selected=json.loads(selection.read_text())['models'][model_name(checkpoint)]['threshold']
            self.assertEqual(selected,.45)
            export_masks(model,args,'test',[selected],torch.device('cpu'))
            compare(args,checkpoint,'test',selection)
            mask=cv2.imread(str(report/'predictions'/'test'/'threshold_0.45'/'surface'/'example_pred.png'),0)
            self.assertEqual(mask.shape,(1500,1500))
            self.assertTrue((report/'test'/'test_comparison.csv').is_file())

    def test_symmetric_edges_constants_boundaries_and_no_wrap(self):
        module=SemanticDirectionalDiffusion(channels=4, structure_channels=4, highres_channels=4)
        a=module.edge_weights(torch.randn(2,8,24,24),torch.randn(2,2,24,24))
        valid=torch.ones(2,1,24,24)
        for d,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            mask=neighbor(valid,dy,dx).bool()
            opposite=neighbor(a[:,CONNECTIVITY_OPPOSITE[d]:CONNECTIVITY_OPPOSITE[d]+1],dy,dx)
            torch.testing.assert_close(a[:,d:d+1][mask],opposite[mask],rtol=0,atol=0)
            self.assertTrue(bool((a[:,d:d+1][~mask]==0).all()))
        constant=torch.full((2,32,24,24),3.)
        torch.testing.assert_close(module.propagate(constant,a),constant,atol=2e-6,rtol=0)
        impulse=torch.zeros(1,1,24,24); impulse[:,:,23,23]=1
        uniform=torch.cat([neighbor(torch.ones_like(impulse),dy,dx) for dy,dx in CONNECTIVITY_DIRECTIONS],1)
        result=module.propagate(impulse,uniform)
        self.assertEqual(float(result[0,0,0,0]),0.)
        z=torch.randn(2,4,24,24,requires_grad=True)
        module.propagate(z,a).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(z.grad).all()))

    def test_D_axes_rotation_zero_and_semantic_beta_zero(self):
        module=SemanticDirectionalDiffusion(channels=4,structure_channels=4,highres_channels=4)
        ell=torch.zeros(1,8,12,12)
        horizontal=torch.zeros(1,2,12,12); horizontal[:,0]=1
        h=module.edge_weights(ell,horizontal)
        v=module.edge_weights(ell,-horizontal)
        self.assertGreater(float(h[0,2,5,5]),float(h[0,0,5,5]))
        torch.testing.assert_close(h[:,2,1:-1,1:-1],v[:,0,1:-1,1:-1])
        torch.testing.assert_close(h[:,2,1:-1,1:-1],h[:,6,1:-1,1:-1])
        self.assertTrue(bool(torch.isfinite(module.edge_weights(ell,torch.zeros_like(horizontal))).all()))
        semantic=SemanticDirectionalDiffusion(channels=4,structure_channels=4,highres_channels=4,mode='semantic')
        semantic.load_state_dict(module.state_dict(),strict=True)
        x,g,hf=(torch.randn(1,4,12,12) for _ in range(3))
        with torch.no_grad(): module.out_proj.weight.normal_(); semantic.load_state_dict(module.state_dict())
        module.beta=0.
        torch.testing.assert_close(module(x,g,hf,horizontal),semantic(x,g,hf,horizontal),rtol=0,atol=0)

    def test_identity_zero_initialization_and_live_upstream_after_update(self):
        module=SemanticDirectionalDiffusion(channels=4,structure_channels=4,highres_channels=4)
        x,g,h=(torch.randn(2,4,12,12,requires_grad=True) for _ in range(3))
        d=torch.randn(2,2,12,12,requires_grad=True)
        out=module(x,g,h,d)
        torch.testing.assert_close(out,x,rtol=0,atol=0)
        optimizer=torch.optim.SGD(module.parameters(),lr=.1)
        (out-torch.randn_like(out)).square().mean().backward()
        self.assertGreater(strength(module.out_proj),0.)
        self.assertEqual(strength(module.guide_head),0.)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        for tensor in (x,g,h,d): tensor.grad=None
        module(x,g,h,d).square().mean().backward()
        for part in (module.guide_head,module.value_proj,module.guide_g_proj,module.guide_h_proj):
            self.assertGreater(strength(part),0.)
        self.assertGreater(float(g.grad.abs().sum()),0.)
        self.assertGreater(float(h.grad.abs().sum()),0.)
        self.assertIsNone(d.grad)
        module.mode='identity'
        self.assertIs(module(x,g,h,d),x)

    def test_stage2_identity_HG_and_S2_D3_losses_have_gradients(self):
        stage2=H0SkeletonDiffusionStage(8,2,4)
        stage3=H0SkeletonDiffusionStage(8,3,4,previous_structure_channels=8)
        for module in (stage2,stage3): module.capture_diagnostics=True
        x=torch.randn(2,8,16,16); h=torch.randn(2,4,16,16)
        second=stage2(x,highres_skeleton_feat=h)
        self.assertIs(second[0],x)
        self.assertIsNone(second[2]); self.assertIsNone(second[3]); self.assertIsNone(second[4])
        criterion=H0DiffusionLoss()
        target=torch.zeros(2,1,64,64); target[:,:,28:32,5:58]=1
        surface=torch.randn(2,1,64,64,requires_grad=True)
        third=stage3(x,highres_skeleton_feat=second[-1]['highres_skeleton_feat'],
                     previous_structure_feat=second[-1]['structure_feat'])
        outputs=[dict(stage=2,skeleton=second[1],connectivity=None,direction=None,structure_gate=None),
                 dict(stage=3,skeleton=third[1],connectivity=None,direction=third[3],structure_gate=None)]
        loss,terms=criterion(surface,surface_gt=target,skeleton_gt=target,stage_outputs=outputs)
        original=SurfaceStructureLoss(stage_structure_weights=(0.,0.,.008,.012), stage_direction_factor=.2,
                                     stage_skeleton_loss_factor=1.,stage_skeleton_only_loss_factor=1.,skeleton_pos_weight=2.)
        expected=original.stage_structure_loss(outputs,target,target)
        torch.testing.assert_close(loss-terms['surface_loss'],expected)
        loss.backward()
        self.assertGreater(strength(stage2.skeleton_head),0.)
        self.assertGreater(strength(stage2.highres_skeleton_fusion),0.)
        self.assertGreater(strength(stage3.direction_head),0.)
        self.assertIs(stage3.last_diagnostics['previous_G2'],second[-1]['structure_feat'])
        self.assertIsNone(terms['C2']);self.assertIsNone(terms['D2']);self.assertIsNone(terms['C3'])


@unittest.skipUnless(torch.cuda.is_available(),'real model CUDA verification')
class IntegrationTests(unittest.TestCase):
    def test_architecture_batch12_training_checkpoint_initialization_and_original(self):
        device=torch.device('cuda')
        torch.manual_seed(1234)
        with (patch('networks.swin_transformer_unet_skip_expand_decoder_sys.KeypointGuidedGlobalTopology',
                   side_effect=AssertionError('Global module must not be constructed')),
             patch('networks.skeleton_guided_head.PairwiseConnectivityHead',
                   side_effect=AssertionError('C heads must not be constructed'))):
            model=build_model(architecture())
        loaded,migration=initialize_from_h0(model,architecture(),1234,PRETRAIN)
        EVIDENCE['migration']=migration
        EVIDENCE['shared_count']=len(migration['shared_tensors'])
        core=model.swin_unet
        self.assertIsNone(core.global_topology); self.assertIsNone(core.global_context_head)
        self.assertNotIn('2',core.highres_structure_fusion)
        for block in core.decoder_structure_blocks.values():
            for name in ('connectivity_head','connectivity_context','structure_gate','feature_residual','gamma1','context_to_gate'):
                self.assertFalse(hasattr(block,name),name)
            block.capture_diagnostics=True
        self.assertIsNone(core.decoder_structure_blocks['2'].direction_head)
        counts={0:0,1:0,2:0,3:0}; real_apply=core._apply_highres_structure_fusion
        def counted(x,h,stage,hw):
            counts[stage]+=1
            return real_apply(x,h,stage,hw)
        core._apply_highres_structure_fusion=counted
        model=model.to(device).eval()
        EVIDENCE['benchmark']=benchmark(model,device)
        for batch in (1,2):
            with torch.inference_mode(): out=model(torch.randn(batch,3,512,512,device=device))
            self.assertEqual(tuple(out[0].shape),(batch,1,512,512))
            for name,value in core.last_stage_features.items():
                channels=96 if name.startswith('G') else 64
                self.assertEqual(tuple(value.shape),(batch,channels,128,128))
            s2,s3=[item for item in out[4] if item.get('stage') in (2,3)]
            self.assertEqual(tuple(s2['skeleton'].shape),(batch,1,128,128))
            self.assertIsNone(s2['connectivity']);self.assertIsNone(s2['direction'])
            self.assertIsNone(s3['connectivity']);self.assertEqual(s3['direction'].shape[1],2)
            self.assertIs(core.decoder_structure_blocks['2'].last_diagnostics['X'],
                          core.decoder_structure_blocks['2'].last_diagnostics['out'])
        self.assertEqual(counts[2],0);self.assertGreater(counts[3],0)
        # Single actual batch2 training step; no full training run.
        del out
        model.train()
        criterion=H0DiffusionLoss().to(device)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        image=torch.randn(2,3,512,512,device=device)
        target=torch.zeros(2,1,512,512,device=device);target[:,:,250:254,30:480]=1
        out=model(image)
        loss,terms=criterion(out[0],surface_gt=target,skeleton_gt=target,stage_outputs=out[4])
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        EVIDENCE['single_batch2_loss']=float(loss.detach())
        del out,loss
        model.eval()
        with torch.no_grad(): expected=model(image[:1])[0].cpu()
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as temp:
            checkpoint=Path(temp)/'model.pth'
            save_atomic(dict(architecture_id=ARCHITECTURE_ID,base_h0_commit=BASE_COMMIT,
                architecture_config=architecture(),model_state_dict=model.state_dict(),args={'use_ema':False}),checkpoint)
            model.cpu();del optimizer,core;gc.collect();torch.cuda.empty_cache()
            restored,_=load_checkpoint(checkpoint)
            restored=restored.to(device).eval()
            with torch.no_grad(): actual=restored(image[:1])[0].cpu()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            del restored
        # Actual shared initializers in all controls, not merely equal seeds.
        digests=[]
        new_digests=[]
        import hashlib
        for mode in ('identity','semantic','semantic_directional'):
            torch.manual_seed(1234)
            control=build_model(architecture(mode))
            _,manifest=initialize_from_h0(control,architecture(mode),1234,PRETRAIN)
            digests.append(manifest['shared_tensor_sha256'])
            hasher=hashlib.sha256()
            for name,tensor in control.state_dict().items():
                if '.diffusion.' in name:hasher.update(tensor.numpy().tobytes())
            new_digests.append(hasher.hexdigest())
            del control
        self.assertEqual(len(set(digests)),1);self.assertEqual(len(set(new_digests)),1)
        EVIDENCE['all_control_initialization_digests_equal']=True
        original=build_model(architecture(),PRETRAIN,original=True).to(device).eval()
        self.assertIsNotNone(original.swin_unet.global_topology)
        with torch.no_grad(): original_out=original(image[:1])
        self.assertEqual(tuple(original_out[0].shape),(1,1,512,512))
        EVIDENCE['original_H0_forward']=True
        (ROOT/'VERIFICATION_EVIDENCE.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
