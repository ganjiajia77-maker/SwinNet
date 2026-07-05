import unittest

import torch

from networks.skeleton_guided_head import SkeletonGuidedHead


class GapMaskTest(unittest.TestCase):
    def test_gap_mask_is_detached_and_requires_nearby_structure(self):
        logits = torch.zeros(1, 1, 9, 9, requires_grad=True)
        skeleton = torch.zeros(1, 1, 9, 9, requires_grad=True)
        connectivity = torch.zeros(1, 8, 9, 9, requires_grad=True)

        no_structure = SkeletonGuidedHead.build_gap_mask(
            logits,
            skeleton,
            connectivity,
        )
        self.assertFalse(no_structure.requires_grad)
        self.assertEqual(float(no_structure.max()), 0.0)

        with torch.no_grad():
            skeleton[:, :, 4, 4] = 1.0
            connectivity[:, :, 4, 4] = 1.0
        gap = SkeletonGuidedHead.build_gap_mask(
            logits,
            skeleton,
            connectivity,
        )
        self.assertFalse(gap.requires_grad)
        self.assertGreater(float(gap[:, :, 4, 5]), 0.0)
        self.assertEqual(float(gap[:, :, 4, 4]), 0.0)

    def test_weak_surface_is_preferred_over_stable_or_empty_surface(self):
        skeleton = torch.zeros(1, 1, 9, 9)
        connectivity = torch.zeros(1, 8, 9, 9)
        skeleton[:, :, 4, 4] = 1.0
        connectivity[:, :, 4, 4] = 1.0

        weak_logits = torch.full((1, 1, 9, 9), torch.logit(torch.tensor(0.3)))
        stable_logits = torch.full(
            (1, 1, 9, 9),
            torch.logit(torch.tensor(0.95)),
        )
        empty_logits = torch.full(
            (1, 1, 9, 9),
            torch.logit(torch.tensor(0.001)),
        )
        weak = SkeletonGuidedHead.build_gap_mask(
            weak_logits,
            skeleton,
            connectivity,
        )
        stable = SkeletonGuidedHead.build_gap_mask(
            stable_logits,
            skeleton,
            connectivity,
        )
        empty = SkeletonGuidedHead.build_gap_mask(
            empty_logits,
            skeleton,
            connectivity,
        )
        location = (0, 0, 4, 5)
        self.assertGreater(float(weak[location]), float(stable[location]))
        self.assertGreater(float(weak[location]), float(empty[location]))

    def test_global_alpha_is_inert_but_gap_rho_receives_gradient(self):
        torch.manual_seed(7)
        head = SkeletonGuidedHead(
            in_channels=8,
            hidden_channels=8,
            topology_eta_init=0.005,
            gap_rho_init=0.005,
        )
        head.eval()
        feature = torch.randn(1, 8, 16, 16)

        with torch.no_grad():
            head.alpha.fill_(-1.0)
            minus = head(feature)[0]
            head.alpha.fill_(1.0)
            plus = head(feature)[0]
        self.assertTrue(torch.equal(minus, plus))

        head.zero_grad(set_to_none=True)
        surface = head(feature)[0]
        surface.mean().backward()
        self.assertIsNone(head.alpha.grad)
        self.assertIsNotNone(head.raw_rho_gap.grad)
        self.assertTrue(torch.isfinite(head.raw_rho_gap.grad))
        self.assertIsNotNone(head.structure_residual[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
