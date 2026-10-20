import unittest

import numpy as np
import torch

from networks.surface_gap_repair import (GapCandidate, SurfaceGapRepair, candidate_tensors,
    generate_candidates, label_candidate, merge_candidate_deltas, taper_weight)


class SurfaceGapRepairTests(unittest.TestCase):
    def test_taper_matches_baseline_outer_product(self):
        w=taper_weight(16)
        axis=torch.linspace(-1,1,16).abs().neg().add(1).clamp_min(.1)
        expected=torch.outer(axis,axis)[None,None]
        torch.testing.assert_close(w,expected,rtol=0,atol=0)
        self.assertAlmostEqual(float(w[0,0,0,0]),.01,places=6)

    def test_candidate_proposals_have_no_gt_input_and_are_tile_assigned(self):
        probability=np.zeros((512,512),np.float32)
        probability[250:263,80:210]=.9
        probability[250:263,226:390]=.9
        candidates,stats=generate_candidates(probability,threshold=.45,tile=512,stride=256)
        self.assertGreaterEqual(stats["endpoint_count"],2)
        self.assertTrue(candidates)
        self.assertTrue(all(c.tile_top==0 and c.tile_left==0 for c in candidates))
        self.assertTrue(all(8<=c.distance<=128 for c in candidates))

    @staticmethod
    def candidate(a,b,kind="endpoint_endpoint"):
        return GapCandidate(kind,a,b,(0.,1.),True,(0.,-1.),True,
                            float(np.hypot(b[0]-a[0],b[1]-a[1])),1.,1.,0,0)

    def test_positive_straight_curved_and_t_junction_labels(self):
        gt=np.zeros((256,256),bool)
        gt[100,40:221]=True
        pred=np.zeros_like(gt,dtype=np.float32)
        straight=label_candidate(self.candidate((100,100),(100,160)),gt,pred)
        self.assertEqual(straight.label,1)

        gt[:]=False
        # A curved, local GT route can be positive even when the straight corridor is empty.
        curve=[(80,120),(100,115),(120,105),(140,95),(160,95),(180,105),(200,115)]
        for (x0,y0),(x1,y1) in zip(curve,curve[1:]):
            steps=max(abs(x1-x0),abs(y1-y0))
            for step in range(steps+1):
                t=step/steps
                x=round(x0+(x1-x0)*t); y=round(y0+(y1-y0)*t)
                gt[y-2:y+3,x-2:x+3]=True
        curved=label_candidate(self.candidate((120,80),(115,200)),gt,pred)
        self.assertEqual(curved.label,1)

        # T-branch connection across a small missing span.
        gt[:]=False; gt[120,40:221]=True; gt[60:121,150]=True
        tee=label_candidate(self.candidate((120,100),(120,124),"endpoint_road"),gt,pred)
        self.assertEqual(tee.label,1)

    def test_true_dead_end_and_distant_parallel_join_are_negative(self):
        gt=np.zeros((256,512),bool); gt[100,40:151]=True
        pred=np.zeros_like(gt,dtype=np.float32)
        dead_end=label_candidate(self.candidate((100,120),(100,180)),gt,pred)
        self.assertEqual(dead_end.label,0)

        gt[:]=False
        gt[80,30:480]=True; gt[120,30:480]=True
        gt[80:121,470]=True  # real connection, well outside the proposed local gap ROI
        parallel=label_candidate(self.candidate((80,200),(120,200)),gt,pred)
        self.assertEqual(parallel.label,0)

    def test_repair_zero_delta_and_joint_gradient(self):
        torch.manual_seed(1)
        repair=SurfaceGapRepair()
        rgb=torch.randn(1,3,512,512)
        p0=torch.sigmoid(torch.randn(1,1,512,512))
        candidate=GapCandidate("endpoint_endpoint",(256,200),(256,240),(0,1),True,(0,-1),True,
                               40.,1.,1.,0,0,1)
        prompts,geometry=candidate_tensors(candidate,"cpu")
        self.assertEqual(tuple(prompts.shape),(1,2,512,512))
        self.assertEqual(tuple(geometry.shape),(1,8))
        shallow=torch.randn(1,96,128,128,requires_grad=True)
        decoder=torch.randn(1,96,512,512,requires_grad=True)
        self.assertEqual(repair.classifier[0].in_features,256)
        cls,delta=repair(rgb,p0,prompts,geometry,shallow,decoder)
        self.assertEqual(tuple(cls.shape),(1,))
        self.assertEqual(tuple(delta.shape),(1,1,512,512))
        self.assertEqual(float(delta.detach().abs().max()),0.0)
        # Connection BCE updates its classifier; pixel loss reaches the zero-initialized delta head.
        loss=torch.nn.functional.binary_cross_entropy_with_logits(cls,torch.ones_like(cls))+delta.square().mean()
        loss.backward()
        self.assertIsNotNone(repair.classifier[-1].weight.grad)
        self.assertIsNotNone(repair.delta512[-1].weight.grad)
        self.assertIsNotNone(shallow.grad)
        self.assertIsNotNone(decoder.grad)

    def test_merged_delta_is_candidate_order_invariant(self):
        base=torch.zeros(1,1,8,8)
        d1=torch.ones_like(base); d2=-2*torch.ones_like(base)
        p1=torch.tensor([.7]); p2=torch.tensor([.3])
        w=torch.ones_like(base)
        a=merge_candidate_deltas(base,[d1,d2],[p1,p2],[w,w])
        b=merge_candidate_deltas(base,[d2,d1],[p2,p1],[w,w])
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
