import unittest
import runpy
import shlex
import sys
from unittest.mock import patch

import torch

from config import get_config
from networks.skeleton_guided_head import DecoderStructureRefinement, PairwiseConnectivityHead
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from networks.vision_transformer import SwinUnet
from types import SimpleNamespace
from pathlib import Path


class ConvConnectivityAblationTests(unittest.TestCase):
    def test_server_runner_matches_manifest_protocol(self):
        root = Path(__file__).resolve().parents[1]
        from evaluation.road_comparison.build_e202_mass_manifest import EXPECTED

        script = (root / "scripts/run_e202_conv_mass1500.sh").read_text(encoding="utf-8")
        command = script.split("  python -u train_image.py ", 1)[1].split("    2>&1", 1)[0]
        command = command.replace('"${resume_args[@]}"', "").replace("\\\n", " ")
        for key, value in {
            "$DATA": "/fixture/data", "$RESULTS": "/fixture/results", "$RUN": "fixture",
            "$CFG": str(root / "configs/swin_tiny_patch4_window7_224_lite.yaml"),
            "$PRETRAIN": "/fixture/swin.pth", "$SEED": "1234",
        }.items():
            command = command.replace(key, value)
        with patch.object(sys, "argv", ["train_image.py"] + shlex.split(command)):
            args = runpy.run_path(str(root / "train_image.py"), run_name="parser_test")["args"]
        mismatches = {name: (getattr(args, name, None), expected)
                      for name, expected in EXPECTED.items()
                      if getattr(args, name, None) != expected}
        self.assertEqual(mismatches, {})

    def test_head_uses_context_only_and_preserves_other_outputs(self):
        block = DecoderStructureRefinement(channels=96)
        block.connectivity_head = torch.nn.Conv2d(96, 8, 1)
        self.assertEqual(sum(p.numel() for p in block.connectivity_head.parameters()), 776)
        self.assertEqual(sum(p.numel() for p in PairwiseConnectivityHead(96).parameters()), 481)

        feature = torch.randn(2, 96, 8, 8, requires_grad=True)
        context = block.connectivity_context(feature)
        logits = block.connectivity_head(context)
        self.assertEqual(tuple(logits.shape), (2, 8, 8, 8))
        logits.sum().backward()
        self.assertIsNotNone(block.connectivity_head.weight.grad)
        self.assertIsNotNone(block.connectivity_context.fuse[0].weight.grad)

    def test_shared_initialization_is_identical(self):
        kwargs = dict(img_size=128, patch_size=4, in_chans=3, num_classes=1,
                      embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                      window_size=8, return_skeleton=True,
                      structure_profile="stage23_boundary_0626",
                      enable_highres_structure_stream=True,
                      enable_e128_stage_fusion=True, enable_h3_surface_fusion=True,
                      enable_global_topology=True, stage_skeleton_mode="direct",
                      remove_stage2_pre_topology_source=True)
        torch.manual_seed(1234)
        pair = SwinTransformerSys(**kwargs, connectivity_head_type="pairwise")
        pair_state = {k: v.clone() for k, v in pair.state_dict().items()
                      if not k.startswith(("decoder_structure_blocks.2.connectivity_head.",
                                           "decoder_structure_blocks.3.connectivity_head."))}
        del pair
        torch.manual_seed(1234)
        conv = SwinTransformerSys(**kwargs, connectivity_head_type="conv")
        conv_state = conv.state_dict()
        self.assertEqual(set(pair_state), set(conv_state) - {
            k for k in conv_state if k.startswith((
                "decoder_structure_blocks.2.connectivity_head.",
                "decoder_structure_blocks.3.connectivity_head."))})
        for name, value in pair_state.items():
            self.assertTrue(torch.equal(value, conv_state[name]), name)
        self.assertIsInstance(conv.decoder_structure_blocks["2"].connectivity_head, torch.nn.Conv2d)
        self.assertIsInstance(conv.decoder_structure_blocks["3"].connectivity_head, torch.nn.Conv2d)

    def test_actual_block_c_ignores_s_and_d_but_trains_shared_features(self):
        block = DecoderStructureRefinement(channels=96).eval()
        block.connectivity_head = torch.nn.Conv2d(96, 8, 1)
        feature = torch.randn(2, 96, 8, 8, requires_grad=True)
        original = block(feature)
        # Perturb the actual S/D outputs, rather than just calling an isolated C head.
        hooks = [
            block.skeleton_head.register_forward_hook(lambda module, args, out: out + 20.),
            block.direction_head.register_forward_hook(lambda module, args, out: out - 30.),
        ]
        try:
            block.runtime_connectivity_skeleton_override = torch.ones(2, 1, 8, 8)
            with torch.no_grad(), patch.object(
                PairwiseConnectivityHead, "direction_alignment",
                side_effect=AssertionError("Conv-C must not compute direction matching"),
            ):
                changed = block(feature)
        finally:
            for hook in hooks:
                hook.remove()
        torch.testing.assert_close(original[2], changed[2], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[1], changed[1]))
        self.assertFalse(torch.equal(original[3], changed[3]))
        original[2].square().mean().backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.)
        self.assertGreater(float(block.structure_branch[0].block[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in block.skeleton_head.parameters()))
        self.assertTrue(all(p.grad is None for p in block.direction_head.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_model_fp32_forward_backward(self):
        root = Path(__file__).resolve().parents[1]
        options = SimpleNamespace(
            cfg=str(root / "configs/swin_tiny_patch4_window7_224_lite.yaml"),
            img_size=128, batch_size=2, opts=None, zip=False, cache_mode="",
            resume="", accumulation_steps=1, use_checkpoint=False,
            amp_opt_level="", tag="", eval=False, throughput=False,
        )
        model = SwinUnet(
            get_config(options), img_size=128, num_classes=1, return_skeleton=True,
            structure_profile="stage23_boundary_0626", enable_highres_structure_stream=True,
            enable_e128_stage_fusion=True, enable_h3_surface_fusion=True,
            enable_global_topology=True, stage_skeleton_mode="direct",
            remove_stage2_pre_topology_source=True, connectivity_head_type="conv",
        ).cuda().train()
        outputs = model(torch.randn(2, 3, 128, 128, device="cuda"))
        self.assertEqual(tuple(outputs[0].shape), (2, 1, 128, 128))
        self.assertEqual(outputs[0].dtype, torch.float32)
        stage_outputs = outputs[4]
        self.assertTrue({2, 3}.issubset({item["stage"] for item in stage_outputs}))
        loss = outputs[0].square().mean()
        for stage in (2, 3):
            connectivity = next(item["connectivity"] for item in stage_outputs
                                if item["stage"] == stage)
            self.assertEqual(connectivity.shape[1], 8)
            loss = loss + connectivity.square().mean()
        loss.backward()
        for stage in (2, 3):
            head = model.swin_unet.decoder_structure_blocks[str(stage)].connectivity_head
            self.assertTrue(torch.isfinite(head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
