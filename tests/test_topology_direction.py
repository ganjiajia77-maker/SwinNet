import copy
import unittest

import torch

from networks.skeleton_guided_head import LightweightTopologyGate
from networks.swin_transformer_unet_skip_expand_decoder_sys import (
    SwinTransformerBlock,
    TopologyAttentionScale,
    window_partition,
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
        block = SwinTransformerBlock(
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
                block.topology_direction_one_hot[center, neighbor].argmax().item()
            )
            self.assertEqual(mapped_channel, channel)

    def test_effective_coefficients_are_nonnegative_and_bounded(self):
        scale = TopologyAttentionScale(
            alpha_max=0.20,
            alpha_init=0.02,
            trainable=True,
        )

        for raw_value in (-100.0, -8.0, 0.0, 100.0):
            scale.topology_alpha.data.fill_(raw_value)
            alpha = scale.effective_topology_alpha().item()
            self.assertGreaterEqual(alpha, 0.0)
            self.assertLessEqual(alpha, 0.20 + 1e-7)

    def test_fixed_alpha(self):
        scale = TopologyAttentionScale(
            alpha_max=0.20,
            alpha_init=0.02,
            trainable=False,
        )
        self.assertFalse(scale.topology_alpha.requires_grad)
        self.assertAlmostEqual(
            scale.effective_topology_alpha().item(),
            0.02,
            places=6,
        )

    def test_fixed_zero_alpha_is_exact_and_finite(self):
        scale = TopologyAttentionScale(
            alpha_max=0.20,
            alpha_init=0.0,
            trainable=False,
        )
        self.assertEqual(scale.effective_topology_alpha().item(), 0.0)
        self.assertTrue(
            all(torch.isfinite(value).all() for value in scale.state_dict().values())
        )

    def test_lightweight_gate_matches_topology_confidence_formula(self):
        gate = LightweightTopologyGate(
            channels=1,
            gamma_max=0.05,
            gamma_init=0.01,
            trainable=False,
        )
        gate.feature_proj.weight.data.fill_(1.0)
        feature = torch.ones(1, 4, 1)
        skeleton = torch.full((1, 1, 2, 2), 0.5)
        connectivity = torch.zeros(1, 8, 2, 2)
        connectivity[:, 0] = 0.8
        connectivity[:, 1] = 0.6

        output = gate(feature, skeleton, connectivity)
        expected = 1.0 + 0.01 * (0.5 * 0.7)
        self.assertTrue(
            torch.allclose(output, torch.full_like(output, expected))
        )

    def test_zero_lightweight_gate_is_strict_identity(self):
        gate = LightweightTopologyGate(
            channels=3,
            gamma_max=0.05,
            gamma_init=0.0,
            trainable=False,
        )
        feature = torch.randn(2, 16, 3)
        skeleton = torch.rand(2, 1, 4, 4)
        connectivity = torch.rand(2, 8, 4, 4)
        output = gate(feature, skeleton, connectivity)
        self.assertTrue(torch.equal(output, feature))

    def test_shifted_windows_keep_feature_skeleton_connectivity_aligned(self):
        block = SwinTransformerBlock(
            dim=8,
            input_resolution=(16, 16),
            num_heads=1,
            window_size=8,
            shift_size=4,
        )
        coordinates = torch.arange(256, dtype=torch.float32).view(1, 16, 16, 1)
        feature = coordinates.repeat(1, 1, 1, 8)
        skeleton = coordinates.clone()
        connectivity = coordinates.repeat(1, 1, 1, 8)
        shifts = (-block.shift_size, -block.shift_size)
        feature = torch.roll(feature, shifts=shifts, dims=(1, 2))
        skeleton = torch.roll(skeleton, shifts=shifts, dims=(1, 2))
        connectivity = torch.roll(connectivity, shifts=shifts, dims=(1, 2))
        feature_windows = window_partition(feature, block.window_size).view(-1, 64, 8)
        skeleton_windows = window_partition(skeleton, block.window_size).view(-1, 64, 1)
        connectivity_windows = window_partition(
            connectivity,
            block.window_size,
        ).view(-1, 64, 8)
        self.assertTrue(
            torch.equal(feature_windows[..., 0], skeleton_windows[..., 0])
        )
        self.assertTrue(
            torch.equal(feature_windows[..., 0], connectivity_windows[..., 0])
        )

    def test_final_topology_term_has_direction_selectivity(self):
        block = SwinTransformerBlock(
            dim=8,
            input_resolution=(8, 8),
            num_heads=1,
            window_size=8,
        )
        scale = TopologyAttentionScale(alpha_max=0.20, alpha_init=0.02)
        center = 3 * 8 + 3

        def topology_term(indices, channels):
            skeleton = torch.zeros(1, 64, 1)
            connectivity = torch.zeros(1, 64, 8)
            skeleton[0, indices, 0] = 1.0
            for channel in channels:
                connectivity[0, indices, channel] = 1.0
            return (
                scale.effective_topology_alpha()
                * block._topology_bias(skeleton, connectivity)[0]
            )

        horizontal = topology_term([3 * 8 + x for x in range(8)], (2, 3))
        self.assertGreater(horizontal[center, center - 1], horizontal[center, center - 8])
        self.assertGreater(horizontal[center, center + 1], horizontal[center, center - 8])
        self.assertGreater(horizontal[center, center + 2], horizontal[center, center - 8])

        vertical = topology_term([y * 8 + 3 for y in range(8)], (0, 1))
        self.assertGreater(vertical[center, center - 8], vertical[center, center - 1])
        self.assertGreater(vertical[center, center + 8], vertical[center, center - 1])

        diagonal = topology_term([y * 8 + y for y in range(8)], (4, 7))
        self.assertGreater(diagonal[center, center - 9], diagonal[center, center - 1])
        self.assertGreater(diagonal[center, center + 9], diagonal[center, center + 1])

    def test_alpha_zero_strictly_matches_original_decoder_block(self):
        torch.manual_seed(7)
        original_block = SwinTransformerBlock(
            dim=8,
            input_resolution=(16, 16),
            num_heads=1,
            window_size=8,
            shift_size=4,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
        ).eval()
        topology_block = copy.deepcopy(original_block)
        original_feature = torch.randn(2, 256, 8, requires_grad=True)
        topology_feature = original_feature.detach().clone().requires_grad_(True)
        skeleton = torch.rand(2, 1, 16, 16)
        connectivity = torch.rand(2, 8, 16, 16)

        original_output = original_block(original_feature)
        topology_zero_output = topology_block(
            topology_feature,
            skeleton,
            connectivity,
            torch.tensor(0.0),
        )
        self.assertTrue(torch.equal(original_output, topology_zero_output))
        original_output.square().mean().backward()
        topology_zero_output.square().mean().backward()
        self.assertTrue(
            torch.equal(original_feature.grad, topology_feature.grad)
        )
        for original_parameter, topology_parameter in zip(
            original_block.parameters(),
            topology_block.parameters(),
        ):
            self.assertTrue(
                torch.equal(
                    original_parameter.grad,
                    topology_parameter.grad,
                )
            )


if __name__ == "__main__":
    unittest.main()
