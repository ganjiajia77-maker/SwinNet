"""Checkpoint audit recognizes the Massachusetts ablation and legacy Data1."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.road_comparison import build_h0_manifest


class H0ManifestTests(unittest.TestCase):
    def test_full_and_ablated_mass_and_legacy_data1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size, ablated, static, use_ema in (
                (1500, False, False, True), (1500, True, False, True),
                (1500, False, True, True), (1500, False, True, False),
                (1500, False, False, False), (1024, False, False, True),
            ):
                saved = dict(
                    img_size=512, source_patch_size=size, random_crop_train=True,
                    random_crops_per_image=1, direct_resize_train=False,
                    amp_dtype="none", use_ema=use_ema, ema_decay=0.999, max_epochs=80,
                    structure_profile="stage23_boundary_0626",
                    enable_highres_structure_stream=not ablated,
                    enable_global_topology=True, stage_skeleton_mode="direct",
                    enable_h3_surface_fusion=True,
                    remove_stage2_pre_topology_source=True,
                )
                if size == 1500:
                    saved.update(ablate_h0_stream=ablated, batch_size=2,
                                 overlap_stride=256, seed=2024)
                    if static:
                        saved["static_h0_stage_fusion"] = True
                checkpoint = root / f"{size}_{ablated}_{static}_{use_ema}.pth"
                torch.save(dict(args=saved, epoch=80,
                                ema_state_dict={} if use_ema else None), checkpoint)
                argv = ["build_h0_manifest.py", "--checkpoint", str(checkpoint),
                        "--data_root", str(root), "--predictions_root", str(root),
                        "--output", str(root / "unused.json"), "--check_only"]
                output = io.StringIO()
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
                    build_h0_manifest.main()
                if static and use_ema:
                    self.assertIn(build_h0_manifest.NAME_B, output.getvalue())
                if not use_ema:
                    self.assertIn("noema80", output.getvalue())


if __name__ == "__main__":
    unittest.main()
