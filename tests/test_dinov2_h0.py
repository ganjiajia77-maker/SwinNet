import argparse
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from encoder_options import add_encoder_arguments, inherit_encoder_arguments
from eval_dinov2_h0 import metrics, overlap_logits
from networks.dinov2_encoder import DinoRoadEncoder, convert_dino_state


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 1024, 1)

    def forward_intermediates(self, image, **kwargs):
        feature = nn.functional.adaptive_avg_pool2d(self.proj(image), 4)
        return [feature] * 4


class Tests(unittest.TestCase):
    def test_converted_prefix_and_pos_grid(self):
        target = {"patch_embed.proj.weight": torch.ones(1024, 3, 16, 16), "pos_embed": torch.zeros(1, 17, 1024)}
        source = {"module.backbone.patch_embed.projection.weight": target["patch_embed.proj.weight"],
                  "module.backbone.pos_embed": torch.ones(1, 10, 1024), "module.backbone.mask_token": torch.zeros(1, 1, 1024)}
        state, ignored = convert_dino_state({"state_dict": source}, target)
        self.assertEqual(state["pos_embed"].shape, (1, 17, 1024))
        self.assertEqual(ignored, ["mask_token"])

    def test_reject_patch14_and_missing(self):
        target = {"patch_embed.proj.weight": torch.ones(1024, 3, 16, 16), "pos_embed": torch.zeros(1, 17, 1024)}
        with self.assertRaisesRegex(ValueError, "patch14"):
            convert_dino_state({"patch_embed.proj.weight": torch.zeros(1024, 3, 14, 14)}, target)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            convert_dino_state({"patch_embed.proj.weight": target["patch_embed.proj.weight"]}, target)

    def test_reject_nonfinite(self):
        target = {"patch_embed.proj.weight": torch.zeros(1024, 3, 16, 16), "pos_embed": torch.zeros(1, 17, 1024)}
        source = dict(target)
        source["pos_embed"] = torch.full_like(source["pos_embed"], float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            convert_dino_state(source, target)

    def test_metadata_mismatch(self):
        parser = argparse.ArgumentParser()
        add_encoder_arguments(parser)
        args = parser.parse_args([])
        checkpoint = {"args": {"encoder_type": "dinov2_l16"}, "model_state_dict": {"swin_unet.dino_encoder.backbone.foo": torch.zeros(1)}}
        inherit_encoder_arguments(args, checkpoint)
        self.assertTrue(args.freeze_pretrained_encoder)
        with self.assertRaises(ValueError):
            inherit_encoder_arguments(parser.parse_args(["--encoder_type", "swin"]), checkpoint)

    def test_full_h0_forward_gradients_and_ema(self):
        from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
        with patch("networks.dinov2_encoder.VisionTransformer", return_value=FakeDino()):
            core = SwinTransformerSys(img_size=64, window_size=8, depths=[2, 2, 6, 2],
                num_classes=1, encoder_type="dinov2_l16", return_skeleton=True,
                enable_highres_structure_stream=True, enable_h3_surface_fusion=True,
                enable_global_topology=True, structure_profile="stage23_boundary_0626",
                stage_skeleton_mode="direct", remove_stage2_pre_topology_source=True)
        core.train()
        self.assertFalse(core.dino_encoder.backbone.training)
        with patch("sys.argv", ["train_image.py"]):
            train = importlib.import_module("train_image")
        wrapper = nn.Module()
        wrapper.swin_unet = core
        ema = train.ModelEMA(wrapper)
        self.assertIs(ema.ema.swin_unet.dino_encoder.backbone, core.dino_encoder.backbone)
        image = torch.randn(2, 3, 64, 64)
        output = core(image)
        self.assertEqual(output[0].shape, (2, 1, 64, 64))
        self.assertTrue(torch.isfinite(output[0]).all())
        loss = output[0].square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in core.dino_encoder.backbone.parameters()))
        for module in (core.dino_encoder.projections, core.encoder_stage1_road_attention_head,
                       core.encoder_stage2_road_attention_head, core.highres_skeleton_adapter):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
        before = core.dino_encoder.backbone.proj.weight.detach().clone()
        ema.update(wrapper)
        torch.testing.assert_close(before, core.dino_encoder.backbone.proj.weight, rtol=0, atol=0)

    def test_overlap_fusion_matches_constant_and_global_counts(self):
        class Constant(nn.Module):
            def forward(self, image):
                return image[:, :1] * 0 + 2
        result = overlap_logits(Constant(), torch.zeros(1, 3, 80, 80), 32, 16)
        torch.testing.assert_close(result, torch.full_like(result, 2))
        self.assertAlmostEqual(metrics(5, 2, 3)["iou"], 0.5)

    def test_random_crop_changes_epoch_without_resizing(self):
        from datasets.dataset_road_skeleton import RoadSkeletonDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train/image").mkdir(parents=True)
            (root / "train/label").mkdir(parents=True)
            image = np.zeros((1024, 1024, 3), np.uint8)
            image[:, :, 0] = np.arange(1024)[None, :] % 256
            cv2.imwrite(str(root / "train/image/1.png"), image)
            cv2.imwrite(str(root / "train/label/1.png"), np.zeros((1024, 1024), np.uint8))
            dataset = RoadSkeletonDataset(directory, image_size=512, tile_size=512,
                source_patch_size=1024, random_crop_train=True, random_crops_per_image=1)
            self.assertEqual(len(dataset), 1)
            first = dataset[0]
            dataset.set_epoch(1)
            second = dataset[0]
            self.assertEqual(first["image"].shape, (3, 512, 512))
            self.assertNotEqual((first["tile_top"], first["tile_left"]), (second["tile_top"], second["tile_left"]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
