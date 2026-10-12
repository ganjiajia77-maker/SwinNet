"""Static H0 still guides stages without receiving stage updates."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from networks.skeleton_guided_head import BidirectionalSkeletonFeatureFusion


class StaticH0FusionTests(unittest.TestCase):
    def test_default_fusion_still_updates_highres(self):
        fusion = BidirectionalSkeletonFeatureFusion(4, 8)
        fusion.eval()
        h0 = torch.randn(1, 4, 8, 8)
        stage = torch.randn(1, 8, 8, 8)
        _, updated_h = fusion(h0, stage)
        self.assertIsNotNone(fusion.stage_to_highres)
        self.assertFalse(torch.allclose(updated_h, h0))

    def test_stage_receives_h0_while_h0_stays_unchanged(self):
        torch.manual_seed(7)
        fusion = BidirectionalSkeletonFeatureFusion(4, 8, update_highres=False)
        fusion.eval()
        h0 = torch.randn(1, 4, 8, 8, requires_grad=True)
        stage = torch.randn(1, 8, 8, 8, requires_grad=True)

        updated_stage, returned_h0 = fusion(h0, stage)
        self.assertIs(returned_h0, h0)
        self.assertIsNone(fusion.stage_to_highres)
        changed_stage, _ = fusion(h0 + 1, stage)
        self.assertFalse(torch.allclose(updated_stage, changed_stage))

        updated_stage.square().mean().backward()
        self.assertGreater(h0.grad.abs().sum().item(), 0)
        self.assertGreater(stage.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
