import unittest
import torch
from types import SimpleNamespace
from networks.skeleton_guided_head import DecoderStructureRefinement
from networks.keypoint_global_topology import KeypointGuidedGlobalTopology
from networks.native_h0_relations import NativeGlobalContext, NativeStage3, install_native_relations
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from losses.road_losses import SurfaceStructureLoss


class NativeH0RelationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_live_global_and_local_gradient_and_boundary_choices(self):
        torch.manual_seed(5)
        old=DecoderStructureRefinement(96,previous_structure_channels=96,highres_structure_channels=64)
        local=NativeStage3(old).eval()
        global_head=NativeGlobalContext(KeypointGuidedGlobalTopology(96,64,max_nodes=8,heads=4,enabled=True)).eval()
        h=torch.randn(1,64,16,16,requires_grad=True);g=torch.randn(1,96,16,16,requires_grad=True)
        s=torch.randn(1,1,16,16,requires_grad=True);c=torch.randn(1,8,16,16,requires_grad=True)
        d=torch.randn(1,2,16,16,requires_grad=True)
        context=global_head(h,g,s,c,d)
        x,sh,c3,d3,_,info=local(g,previous_structure_feat=g,highres_skeleton_feat=context)
        self.assertEqual(sh.shape,(1,1,16,16));self.assertEqual(info['highres_skeleton_feat'].shape,h.shape)
        choices=info['relation_weights'];torch.testing.assert_close(choices.sum(1),torch.ones_like(choices[:,0]))
        self.assertEqual(float(choices[:,1,0,:].sum()),0.)
        loss=info['highres_skeleton_feat'].square().mean()+sh.square().mean()+c3.square().mean()+d3.square().mean()
        loss.backward()
        for parameter in (global_head.raw_alpha,global_head.pair[0].weight,local.edge[0].weight,
                          local.skeleton_head.out.weight,old.connectivity_head.edge_linear.weight):
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_full_H0_forward_and_native_loss(self):
        torch.manual_seed(11)
        model=SwinTransformerSys(img_size=128,window_size=8,num_classes=1,return_skeleton=True,
            structure_profile='stage23_boundary_0626',enable_highres_structure_stream=True,
            enable_h3_surface_fusion=True,enable_global_topology=True,stage_skeleton_mode='direct',
            remove_stage2_pre_topology_source=True)
        install_native_relations(SimpleNamespace(swin_unet=model));model.eval()
        mask=torch.zeros(1,1,128,128);mask[:,:,59:68,10:118]=1
        skel=torch.zeros_like(mask);skel[:,:,63,10:118]=1
        out=model(torch.randn(1,3,128,128));stage=out[4]
        self.assertEqual(model.native_global_context.calls,1)
        rows=[x for x in stage if x.get('stage') in (2,3)]
        self.assertEqual([x['stage'] for x in rows],[2,3])
        self.assertFalse(any(x.get('stage')=='highres_structure' for x in stage))
        self.assertEqual(rows[1]['skeleton'].shape,(1,1,32,32))
        self.assertEqual(model.last_stage_features['H2'].shape,(1,64,32,32))
        criterion=SurfaceStructureLoss(stage_structure_weights=(0.,0.,.008,.012),
                                       stage_direction_factor=.2,stage_connectivity_factor=3.)
        criterion.native_relation_enabled=True
        surface=criterion.surface_loss(out[0],mask)[0]
        connected=(model.native_global_context.pair[0].weight,model.decoder_structure_blocks['3'].edge[0].weight,
                   model.decoder_structure_blocks['3'].connectivity_head.edge_linear.weight,
                   model.decoder_structure_blocks['3'].direction_head[-1].weight,
                   model.decoder_structure_blocks['3'].skeleton_head.out.weight)
        for gradient in torch.autograd.grad(surface,connected,retain_graph=True,allow_unused=True):
            self.assertIsNotNone(gradient);self.assertGreater(float(gradient.abs().sum()),0.)
        loss,_=criterion(out[0],surface_gt=mask,skeleton_gt=skel,skeleton_dilate_gt=skel,
                         stage_outputs=stage,boundary_logits=out[1],skeleton_logits=out[2],connectivity_logits=out[3])
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(float(model.native_global_context.pair[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.decoder_structure_blocks['3'].edge[0].weight.grad.abs().sum()),0.)

    def test_shared_H0_initialization_is_identical(self):
        opts=dict(img_size=128,window_size=8,num_classes=1,return_skeleton=True,
            structure_profile='stage23_boundary_0626',enable_highres_structure_stream=True,
            enable_h3_surface_fusion=True,enable_global_topology=True,stage_skeleton_mode='direct',
            remove_stage2_pre_topology_source=True)
        torch.manual_seed(1234);base=SwinTransformerSys(**opts)
        reference={k:v.clone() for k,v in base.state_dict().items()};del base
        torch.manual_seed(1234);model=SwinTransformerSys(**opts);install_native_relations(SimpleNamespace(swin_unet=model))
        candidate={k.replace('native_global_context.topology.','global_topology.'):v for k,v in model.state_dict().items()}
        skip='decoder_structure_blocks.3.skeleton_head.'
        common=[k for k in reference.keys() & candidate.keys() if not k.startswith(skip)]
        self.assertGreater(len(common),100)
        for key in common:torch.testing.assert_close(reference[key],candidate[key],rtol=0,atol=0,msg=key)

    def test_full_image_validation_without_stage_outputs(self):
        criterion=SurfaceStructureLoss(stage_structure_weights=(0.,0.,.008,.012))
        criterion.native_relation_enabled=True
        logits=torch.zeros(1,1,1500,1500);mask=torch.zeros_like(logits)
        loss,_=criterion(logits,surface_gt=mask,skeleton_gt=mask,skeleton_dilate_gt=mask)
        self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':unittest.main()
