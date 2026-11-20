"""Regression check for the original H0 decoder with C removed."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses.road_losses import SurfaceStructureLoss
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class NoConnectivityTests(unittest.TestCase):
    def test_same_initialization_and_forward_with_only_c_removed(self):
        settings = dict(
            img_size=64, depths=[1, 1, 1, 1], num_classes=1, return_skeleton=True,
            structure_profile="stage23_boundary_0626", stage_skeleton_mode="direct",
            enable_highres_structure_stream=True, enable_h3_surface_fusion=True,
            enable_global_topology=True, remove_stage2_pre_topology_source=True,
        )
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(min(threads, 4))
        torch.manual_seed(1234)
        original = SwinTransformerSys(**settings).eval()
        original_rng = torch.get_rng_state()
        torch.manual_seed(1234)
        ablated = SwinTransformerSys(**settings, disable_stage_connectivity=True).eval()
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        old_state = original.state_dict()
        new_state = ablated.state_dict()
        removed = set(old_state) - set(new_state)
        self.assertTrue(removed)
        self.assertTrue(all(
            ".connectivity_head." in key or ".connectivity_context." in key
            for key in removed
        ))
        for key, value in new_state.items():
            self.assertTrue(torch.equal(old_state[key], value), key)
        ablated.load_state_dict(new_state, strict=True)

        for block in original.decoder_structure_blocks.values():
            block.runtime_ablate_connectivity_gate = True
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad(), patch.object(
            original, "_latest_local_topology_features", return_value=None
        ):
            expected, *_ = original(image)
        with patch.object(
            ablated.global_topology, "forward_feature_anchors",
            wraps=ablated.global_topology.forward_feature_anchors,
        ) as global_call:
            surface, boundary, skeleton, connectivity, stages = ablated(image)
            global_call.assert_called_once()
            self.assertIsNone(global_call.call_args.kwargs["connectivity_feature"])

        self.assertTrue(torch.equal(expected, surface))
        self.assertIsNone(connectivity)
        decoder_stages = [item for item in stages if item["stage"] in (2, 3)]
        self.assertEqual([item["stage"] for item in decoder_stages], [2, 3])
        for item in decoder_stages:
            self.assertIsNone(item["connectivity"])
            self.assertEqual(item["skeleton"].shape, (1, 1, 16, 16))
            self.assertEqual(item["direction"].shape, (1, 2, 16, 16))

        criterion = SurfaceStructureLoss(
            skeleton_weight=0.0, connectivity_weight=0.0,
            stage_structure_weights=(0.0, 0.0, 0.008, 0.012),
            stage_direction_factor=0.2, use_masked_connectivity_center_experiment=True,
        )
        mask = torch.zeros_like(surface)
        mask[..., 28:36, 8:56] = 1
        centerline = torch.zeros_like(surface)
        centerline[..., 31, 8:56] = 1
        with patch("losses.road_losses.build_connectivity_target") as build_c:
            loss, parts = criterion(
                surface, boundary_logits=boundary, skeleton_logits=skeleton,
                connectivity_logits=connectivity, stage_outputs=stages,
                surface_gt=mask, skeleton_gt=centerline,
            )
            build_c.assert_not_called()
        self.assertEqual(parts["connectivity_loss"].item(), 0.0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for block in ablated.decoder_structure_blocks.values():
            for head in (block.skeleton_head.out, block.direction_head[-1]):
                self.assertGreater(head.bias.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
