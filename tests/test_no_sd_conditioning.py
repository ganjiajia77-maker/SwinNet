"""Verify the ablation removes only direct S/D inputs to connectivity."""

import unittest

import torch

from networks.checkpoint_compat import adapt_connectivity_modules_for_checkpoint
from networks.skeleton_guided_head import DecoderStructureRefinement, PairwiseConnectivityHead


class NoSDConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.block = DecoderStructureRefinement(8).eval()
        self.image_feature = torch.randn(2, 8, 12, 12, requires_grad=True)

    def test_changing_skeleton_and_direction_heads_cannot_change_connectivity(self):
        original = self.block(self.image_feature)
        with torch.no_grad():
            for head in (self.block.skeleton_head, self.block.direction_head):
                for parameter in head.parameters():
                    parameter.add_(3.0)
        changed = self.block(self.image_feature)
        self.assertFalse(torch.equal(original[1], changed[1]))
        self.assertFalse(torch.equal(original[3], changed[3]))
        torch.testing.assert_close(original[2], changed[2], rtol=0, atol=0)
        self.assertFalse(hasattr(self.block.connectivity_head, 'prior_embed'))
        self.assertEqual(self.block.connectivity_head.edge_linear.in_channels, 4 * 8)

    def test_connectivity_keeps_feature_gradients_and_sd_keep_their_own_gradients(self):
        outputs = self.block(self.image_feature)
        outputs[2].square().mean().backward()
        self.assertTrue(torch.isfinite(self.image_feature.grad).all())
        self.assertGreater(float(self.image_feature.grad.abs().sum()), 0)
        self.assertIsNotNone(self.block.connectivity_head.edge_linear.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.block.skeleton_head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.block.direction_head.parameters()))
        self.block.zero_grad(set_to_none=True)
        outputs = self.block(self.image_feature)
        (outputs[1].square().mean() + outputs[3].square().mean()).backward()
        for head in (self.block.skeleton_head, self.block.direction_head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in head.parameters()))

    def test_checkpoint_loader_rejects_baseline_prior_classifier(self):
        model = torch.nn.Module()
        model.connectivity_head = PairwiseConnectivityHead(8)
        weights = model.state_dict()
        adapt_connectivity_modules_for_checkpoint(model, weights, 'standard')
        weights['connectivity_head.edge_linear.weight'] = torch.zeros(1, 48, 1, 1)
        with self.assertRaisesRegex(ValueError, 'feature-pair-only'):
            adapt_connectivity_modules_for_checkpoint(model, weights, 'standard')


if __name__ == '__main__':
    unittest.main()
