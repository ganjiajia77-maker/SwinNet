import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from config import get_config
from diagnostics.probe_stage_injections import InjectionHooks, infer_full, load_cases, load_model
from networks.vision_transformer import SwinUnet


class StageInjectionProbeTests(unittest.TestCase):
    def test_data1_sat_image_matches_mask_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "test" / "image"
            label_dir = root / "test" / "label"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image = image_dir / "100703_sat.jpg"
            label = label_dir / "100703_mask.png"
            image.touch()
            label.touch()
            self.assertEqual(list(load_cases(root, "test", 1)), [(image, label)])

    def test_overlap_positions_and_hook_identity(self):
        from diagnostics.probe_stage_injections import positions

        self.assertEqual(positions(1024, 512, 256), [0, 256, 512])
        self.assertEqual(positions(1500, 512, 256), [0, 256, 512, 768, 988])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_checkpoint_load_and_paired_inference(self):
        root = Path(__file__).resolve().parents[1]
        cfg = root / "configs/swin_tiny_patch4_window7_224_lite.yaml"
        options = SimpleNamespace(cfg=str(cfg), opts=None, batch_size=1, img_size=128,
                                  zip=False, cache_mode="", resume="", accumulation_steps=1,
                                  use_checkpoint=False, amp_opt_level="", tag="", eval=False,
                                  throughput=False)
        architecture = dict(structure_profile="stage23_boundary_0626",
                            enable_highres_structure_stream=True,
                            highres_structure_fuse_stages="stage23",
                            highres_structure_fusion_mode="stage23",
                            stage_skeleton_mode="direct")
        model = SwinUnet(get_config(options), img_size=128, num_classes=1,
                         return_skeleton=True, **architecture)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pth"
            torch.save(dict(model_state_dict=model.state_dict(), args=dict(cfg=str(cfg), **architecture),
                            structure_profile="stage23_boundary_0626", epoch=3), checkpoint)
            args = SimpleNamespace(code_root=None, checkpoint=checkpoint, tile_size=128,
                                   device="cuda", overlap_stride=64)
            loaded, _, source = load_model(args)
            self.assertEqual(source, "H0 adapter")
            hooks = InjectionHooks(loaded)
            image = np.zeros((128, 128, 3), np.uint8)
            weight = torch.ones((128, 128), device="cuda")
            try:
                with torch.inference_mode():
                    baseline, stats = infer_full(loaded, image, args, hooks, "baseline", weight)
                    stage2_off, _ = infer_full(loaded, image, args, hooks, "stage2_off", weight)
                    stage3_off, _ = infer_full(loaded, image, args, hooks, "stage3_off", weight)
                self.assertEqual(baseline.shape, (128, 128))
                self.assertTrue(np.isfinite(baseline).all())
                self.assertIn("stage2_relative_l2", stats)
                self.assertIn("stage3_relative_l2", stats)
                self.assertEqual(stage2_off.shape, baseline.shape)
                self.assertEqual(stage3_off.shape, baseline.shape)
            finally:
                hooks.close()


if __name__ == "__main__":
    unittest.main()
