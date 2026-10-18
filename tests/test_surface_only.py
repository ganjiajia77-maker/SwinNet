import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet, get_topology_coefficients, load_topology_checkpoint_state


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = dict(return_skeleton=True, structure_profile="stage23_boundary_0626",
                 enable_highres_structure_stream=True, enable_e128_stage_fusion=True,
                 enable_h3_surface_fusion=True, remove_stage2_pre_topology_source=True,
                 enable_global_topology=True, stage_skeleton_mode="direct")


def config(size=128):
    return get_config(SimpleNamespace(
        cfg=str(ROOT / "configs/swin_tiny_patch4_window7_224_lite.yaml"), opts=None,
        batch_size=2, img_size=size, zip=False, cache_mode="", resume="",
        accumulation_steps=1, use_checkpoint=False, amp_opt_level="", tag="",
        eval=False, throughput=False))


class SurfaceOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        cls.full = SwinUnet(config(), num_classes=1, **REFERENCE)
        torch.manual_seed(1234)
        cls.surface = SwinUnet(config(), num_classes=1, surface_only=True, **REFERENCE)

    def test_common_initialization_and_actual_removal(self):
        original = self.full.state_dict()
        for name, tensor in self.surface.state_dict().items():
            self.assertIn(name, original)
            self.assertTrue(torch.equal(tensor, original[name]), name)
        core = self.surface.swin_unet
        self.assertEqual([len(x.blocks) for x in core.layers], [2, 2, 6, 2])
        self.assertEqual([len(x.blocks) for x in core.layers_up[1:]], [6, 2, 2])
        self.assertEqual(len(core.dca_blocks), 2)
        self.assertEqual(len(core.decoder_structure_blocks), 0)
        forbidden = ("connectivity", "skeleton", "boundary", "road_attention", "global_topology",
                     "prepatch_structure", "highres_structure", "h3_surface", "surface_h3")
        self.assertFalse([name for name, _ in self.surface.named_parameters()
                          if any(part in name for part in forbidden)])
        self.assertFalse(get_topology_coefficients(self.surface)["global_topology_enabled"])
        load_topology_checkpoint_state(self.surface, self.surface.state_dict(), "test", strict=True)

    def test_same_crop_without_building_skeleton_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "train_labels"):
                (root / name).mkdir()
            rng = np.random.RandomState(2)
            cv2.imwrite(str(root / "train" / "case.tiff"), rng.randint(0, 256, (96, 96, 3), dtype=np.uint8))
            cv2.imwrite(str(root / "train_labels" / "case.tif"), rng.randint(0, 2, (96, 96), dtype=np.uint8) * 255)
            options = dict(root_dir=str(root), image_size=64, source_patch_size=96,
                           tile_size=64, random_crop_train=True, augment=True)
            full = RoadSkeletonDataset(**options)
            surface = RoadSkeletonDataset(surface_only=True, **options)
            full.set_epoch(3)
            surface.set_epoch(3)
            with mock.patch.object(full, "_skeletonize_binary", side_effect=lambda x: x * 255):
                expected = full[0]
            with mock.patch.object(surface, "_skeletonize_binary", side_effect=AssertionError("unused")):
                actual = surface[0]
            for key in ("image", "mask"):
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
            self.assertEqual((actual["tile_top"], actual["tile_left"]),
                             (expected["tile_top"], expected["tile_left"]))
            self.assertEqual(float(actual["skeleton"].sum()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_surface_only_training_loss_and_backward(self):
        with mock.patch("sys.argv", ["train_image.py", "--surface_only",
                                    "--structure_profile", "stage23_boundary_0626",
                                    "--surface_focal_gamma", "1.0"]):
            trainer = importlib.import_module("train_image")
        criterion = trainer.build_criterion(trainer.args, trainer.get_final_loss_weights(trainer.args), "cuda")
        model = self.surface.cuda().train()
        try:
            image = torch.randn(2, 3, 128, 128, device="cuda")
            mask = (torch.rand(2, 1, 128, 128, device="cuda") > 0.9).float()
            outputs = model(image)
            self.assertEqual(outputs[0].shape, mask.shape)
            self.assertEqual(outputs[1:], (None, None, None, []))
            loss, details = criterion(outputs[0], surface_gt=mask, skeleton_gt=torch.zeros_like(mask),
                                      stage_outputs=outputs[4])
            expected, _, _ = criterion.surface_loss(outputs[0], mask)
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            self.assertEqual(float(details["stage_structure_loss"]), 0.0)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            unused = set()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None:
                        unused.add(name)
                        continue
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertEqual(unused, set())
            for block in model.swin_unet.dca_blocks:
                self.assertGreater(float(block.weight_conv[-1].weight.grad.abs().sum()), 0.0)
        finally:
            model.zero_grad(set_to_none=True)
            model.cpu()


if __name__ == "__main__":
    unittest.main()
