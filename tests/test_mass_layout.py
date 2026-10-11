"""Flat Massachusetts split preflight must agree with training and metrics."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class MassLayoutTests(unittest.TestCase):
    def run_check(self, root):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_mass_layout.py"),
             "--root_path", str(root), "--source_size", "16"],
            cwd=root, capture_output=True, text=True, check=False,
        )

    def test_flat_splits_accept_binary_labels_and_reject_low_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val", "test"):
                (root / split).mkdir()
                (root / f"{split}_labels").mkdir()
                image = np.zeros((16, 16, 3), dtype=np.uint8)
                label = np.zeros((16, 16), dtype=np.uint8)
                label[3:8, 3:8] = 255
                cv2.imwrite(str(root / split / f"{split}.png"), image)
                cv2.imwrite(str(root / f"{split}_labels" / f"{split}.png"), label)

            result = self.run_check(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("test: 1 paired", result.stdout)

            label = cv2.imread(str(root / "val_labels/val.png"), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(root / "val_labels/val.png"), (label > 0).astype(np.uint8))
            result = self.run_check(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Expected binary 0/255", result.stderr)


if __name__ == "__main__":
    unittest.main()
