import unittest

import torch

from networks.skeleton_guided_head import DirectionalValueAggregation
from networks.swin_transformer_unet_skip_expand_decoder_sys import (
    TopologyAwareSwinBlock,
)


DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


class TopologyDirectionTest(unittest.TestCase):
    def test_attention_direction_mapping(self):
        block = TopologyAwareSwinBlock(
            dim=8,
            input_resolution=(8, 8),
            num_heads=1,
            window_size=8,
        )
        center_y, center_x = 3, 3
        center = center_y * 8 + center_x

        for channel, (dy, dx) in enumerate(DIRECTIONS):
            neighbor = (center_y + dy) * 8 + center_x + dx
            mapped_channel = int(
                block.direction_one_hot[center, neighbor].argmax().item()
            )
            self.assertEqual(mapped_channel, channel)

    def test_dva_samples_neighbor_named_by_connectivity_channel(self):
        aggregation = DirectionalValueAggregation(channels=1, gamma_max=0.05)
        aggregation.value_proj.weight.data.fill_(1.0)
        aggregation.gamma.data.fill_(20.0)
        center_y, center_x = 3, 3

        for channel, (dy, dx) in enumerate(DIRECTIONS):
            feature = torch.zeros(1, 1, 8, 8)
            feature[0, 0, center_y + dy, center_x + dx] = 8.0
            connectivity = torch.zeros(1, 8, 8, 8)
            connectivity[0, channel, center_y, center_x] = 1.0

            output = aggregation(feature, connectivity)
            self.assertAlmostEqual(
                output[0, 0, center_y, center_x].item(),
                0.05,
                places=5,
                msg=f"channel={channel}, direction={(dy, dx)}",
            )

    def test_effective_coefficients_are_nonnegative_and_bounded(self):
        block = TopologyAwareSwinBlock(
            dim=8,
            input_resolution=(8, 8),
            num_heads=1,
            window_size=8,
            topology_alpha_max=0.05,
        )
        aggregation = DirectionalValueAggregation(channels=1, gamma_max=0.05)

        for raw_value in (-100.0, -8.0, 0.0, 100.0):
            block.topology_alpha.data.fill_(raw_value)
            aggregation.gamma.data.fill_(raw_value)
            alpha = block.effective_topology_alpha().item()
            gamma = aggregation.effective_gamma().item()
            self.assertGreaterEqual(alpha, 0.0)
            self.assertLessEqual(alpha, 0.05 + 1e-7)
            self.assertGreaterEqual(gamma, 0.0)
            self.assertLessEqual(gamma, 0.05 + 1e-7)


if __name__ == "__main__":
    unittest.main()
