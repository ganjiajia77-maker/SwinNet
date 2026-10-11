import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import torch

import eval_data1
import prepare_manifest
from data1 import case_id
from inference import counts, metrics, positions, predict_full
from train_data1 import criterion


class ConstantModel(torch.nn.Module):
    def forward(self, image):
        return image[:, :1].sigmoid()


class MockRoadModel(torch.nn.Module):
    def __init__(self, pretrained_encoder=False):
        super().__init__()

    def forward(self, image):
        return torch.sigmoid(image[:, :1] * 4)


class PipelineTest(unittest.TestCase):
    def test_case_matching(self):
        self.assertEqual(case_id("123_sat.jpg"), case_id("123_mask.png"))

    def test_positions_cover_edge(self):
        self.assertEqual(positions(1024, 512, 256), [0, 256, 512])
        self.assertEqual(positions(1100, 512, 256), [0, 256, 512, 588])

    def test_weighted_stitch_preserves_constant(self):
        image = torch.full((1, 3, 16, 16), 0.4)
        result = predict_full(ConstantModel(), image, tile_size=8, stride=4)
        self.assertEqual(result.shape, (1, 1, 16, 16))
        self.assertTrue(torch.allclose(result, torch.sigmoid(image[:, :1]), atol=1e-6))

    def test_empty_mask_loss_finite(self):
        prediction = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)
        loss = criterion(prediction, torch.zeros_like(prediction))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(prediction.grad).all().item())

    def test_global_metrics(self):
        prediction = torch.tensor([[0.8, 0.8], [0.1, 0.1]])
        target = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        self.assertEqual(counts(prediction, target, 0.5), (1, 1, 1))
        self.assertEqual(counts(prediction.numpy(), target.numpy(), 0.5), (1, 1, 1))
        self.assertEqual(counts(np.asarray(prediction), target, 0.5), (1, 1, 1))
        self.assertAlmostEqual(metrics(1, 1, 1)["iou"], 1 / 3)

    def test_val_selection_and_test_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, run = root / "data1", root / "run"
            checkpoint = run / "best.pth"
            run.mkdir()
            checkpoint.touch()
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
            image[200:800, 510:515] = 255
            label = image[:, :, 0]
            for split in ("val", "test"):
                image_dir, label_dir = data / split / "image", data / split / "label"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                self.assertTrue(cv2.imwrite(str(image_dir / "1_sat.png"), image))
                self.assertTrue(cv2.imwrite(str(label_dir / "1_mask.png"), label))
            state = {"epoch": 5, "model_state_dict": {},
                     "args": {"encoder_ckpt": None, "batch_size": 2, "accumulation_steps": 2}}
            val_args = ["eval_data1.py", "--data_root", str(data), "--checkpoint", str(checkpoint),
                        "--output_dir", str(run / "unified_val_predictions"), "--split", "val", "--workers", "0"]
            with patch.object(sys, "argv", val_args), patch.object(eval_data1, "WeavingUnet", MockRoadModel), \
                 patch.object(eval_data1.torch, "load", return_value=state):
                eval_data1.main()
            with patch.object(sys, "argv", ["prepare_manifest.py", "--run_dir", str(run),
                                             "--data_root", str(data)]):
                prepare_manifest.main()
            tool = Path(__file__).resolve().parents[1] / "evaluation" / "road_comparison" / "compare_predictions.py"
            subprocess.run([sys.executable, str(tool), "--manifest", str(run / "unified_manifest.json"),
                            "--split", "val", "--output_dir", str(run / "unified_val_metrics")], check=True)
            selection = run / "unified_val_metrics" / "val_selection.json"
            with patch.object(sys, "argv", ["prepare_manifest.py", "--run_dir", str(run),
                                             "--data_root", str(data), "--selection", str(selection)]):
                prepare_manifest.main()
            threshold = json.loads((run / "selected_threshold.json").read_text())["threshold"]
            test_args = ["eval_data1.py", "--data_root", str(data), "--checkpoint", str(checkpoint),
                         "--output_dir", str(run / "unified_test_predictions"), "--split", "test",
                         "--threshold", str(threshold), "--workers", "0"]
            with patch.object(sys, "argv", test_args), patch.object(eval_data1, "WeavingUnet", MockRoadModel), \
                 patch.object(eval_data1.torch, "load", return_value=state):
                eval_data1.main()
            subprocess.run([sys.executable, str(tool), "--manifest", str(run / "unified_manifest.json"),
                            "--split", "test", "--selection", str(selection),
                            "--output_dir", str(run / "unified_test_metrics")], check=True)
            self.assertTrue((run / "unified_test_metrics" / "test_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
