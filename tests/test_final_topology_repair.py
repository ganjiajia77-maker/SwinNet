import unittest

import torch

from networks.skeleton_guided_head import FinalTopologyRepairAttention


class FinalTopologyRepairAttentionTest(unittest.TestCase):
    def test_eta_zero_is_strict_identity_for_output_and_feature_gradient(self):
        torch.manual_seed(11)
        module = FinalTopologyRepairAttention(
            channels=4,
            window_size=4,
            tau=4.0,
            eta_max=0.05,
            eta_init=0.0,
        ).eval()
        feature = torch.randn(1, 4, 8, 8, requires_grad=True)
        skeleton = torch.rand(1, 1, 8, 8)
        connectivity = torch.rand(1, 8, 8, 8)

        output = module(feature, skeleton, connectivity)
        self.assertTrue(torch.equal(output, feature))
        output.square().mean().backward()
        expected_gradient = 2.0 * feature.detach() / feature.numel()
        self.assertTrue(torch.equal(feature.grad, expected_gradient))

    def test_direction_distance_and_weak_target_selectivity(self):
        module = FinalTopologyRepairAttention(
            channels=2,
            window_size=4,
            tau=4.0,
            eta_max=0.05,
            eta_init=0.005,
        )
        skeleton = torch.zeros(1, 1, 4, 4)
        connectivity = torch.zeros(1, 8, 4, 4)
        source_y, source_x = 1, 0
        source_index = source_y * 4 + source_x
        skeleton[0, 0, source_y, source_x] = 1.0
        connectivity[0, 3, source_y, source_x] = 1.0

        weak_skeleton = skeleton.clone()
        weak_skeleton[0, 0, source_y, source_x + 1] = 0.1
        weak_topology, _, _ = module._topology_term(
            weak_skeleton,
            connectivity,
        )

        strong_skeleton = skeleton.clone()
        strong_skeleton[0, 0, source_y, source_x + 1] = 0.9
        strong_topology, _, _ = module._topology_term(
            strong_skeleton,
            connectivity,
        )

        right_one = source_index + 1
        right_three = source_index + 3
        up_one = source_index - 4
        self.assertGreater(
            weak_topology[0, source_index, right_one],
            weak_topology[0, source_index, up_one],
        )
        self.assertGreater(
            weak_topology[0, source_index, right_one],
            weak_topology[0, source_index, right_three],
        )
        self.assertGreater(
            weak_topology[0, source_index, right_one],
            strong_topology[0, source_index, right_one],
        )

    def test_unreliable_window_has_zero_topology(self):
        module = FinalTopologyRepairAttention(
            channels=2,
            window_size=4,
            eta_init=0.005,
        )
        skeleton = torch.zeros(1, 1, 4, 4)
        connectivity = torch.zeros(1, 8, 4, 4)
        topology, reliability, _ = module._topology_term(
            skeleton,
            connectivity,
        )
        self.assertEqual(reliability.max().item(), 0.0)
        self.assertEqual(topology.max().item(), 0.0)

    def test_nonzero_eta_preserves_shape_and_backpropagates(self):
        module = FinalTopologyRepairAttention(
            channels=4,
            window_size=4,
            eta_init=0.005,
        )
        feature = torch.randn(2, 4, 8, 8, requires_grad=True)
        skeleton = torch.rand(2, 1, 8, 8)
        connectivity = torch.rand(2, 8, 8, 8)
        output = module(feature, skeleton, connectivity)
        output.mean().backward()

        self.assertEqual(output.shape, feature.shape)
        self.assertIsNotNone(feature.grad)
        self.assertIsNotNone(module.raw_eta.grad)
        self.assertIsNotNone(module.qkv.weight.grad)


if __name__ == "__main__":
    unittest.main()
