import unittest
import numpy as np
import torch
from datasets.native_relation_targets import native_targets
from losses.native_path_supervision import path_supervision
from losses.road_losses import SurfaceStructureLoss


class PathSupervisionTest(unittest.TestCase):
    def test_path_losses_backpropagate_to_both_heads(self):
        road=np.zeros((512,512),bool)
        road[120,70:250]=True;road[144,70:250]=True
        labels=native_targets(road,road)
        relations={k:torch.from_numpy(np.array([labels[k]])) for k in ('connectivity','connectivity_valid')}
        s=torch.randn(1,1,512,512,requires_grad=True);c=torch.randn(1,8,128,128,requires_grad=True)
        target=torch.from_numpy(road[None,None].astype(np.float32))
        surface,connectivity,stats=path_supervision(s,c,target,target,relations,'joint',1234)
        self.assertGreater(stats['positive_used'],0);self.assertGreater(stats['negative_used'],0)
        (surface+connectivity).backward()
        self.assertGreater(float(s.grad.abs().sum()),0);self.assertGreater(float(c.grad.abs().sum()),0)

    def test_full_criterion_path_mode_and_disabled_mode(self):
        road=np.zeros((512,512),bool);road[120,70:250]=True;road[144,70:250]=True
        target=torch.from_numpy(road[None,None].astype(np.float32))
        s=torch.randn(1,1,512,512,requires_grad=True);c=torch.randn(1,8,128,128,requires_grad=True)
        stages=[{'stage':3,'skeleton':torch.randn(1,1,128,128,requires_grad=True),
                 'connectivity':c,'direction':torch.randn(1,2,128,128,requires_grad=True)}]
        criterion=SurfaceStructureLoss(stage_connectivity_factor=3,stage_direction_factor=.2)
        criterion.native_relation_enabled=True;criterion.path_training_mode='joint'
        criterion.path_surface_weight=.01;criterion.path_c_weight=.01
        torch.manual_seed(1234)
        loss,details=criterion(s,surface_gt=target,skeleton_gt=target,stage_outputs=stages)
        self.assertGreater(float(details['path_surface_weighted']),0.)
        self.assertGreater(float(details['path_connectivity_weighted']),0.)
        self.assertGreater(criterion.last_path_stats.get('positive_used',0),0)
        grads=torch.autograd.grad(loss,(s,c))
        self.assertTrue(all(float(x.abs().sum())>0 for x in grads))
        criterion.path_training_mode='off'
        torch.manual_seed(1234)
        _,off=criterion(s,surface_gt=target,skeleton_gt=target,stage_outputs=stages)
        self.assertEqual(float(off['path_surface_weighted']),0.)
        self.assertEqual(float(off['path_connectivity_weighted']),0.)

if __name__=='__main__':unittest.main()
