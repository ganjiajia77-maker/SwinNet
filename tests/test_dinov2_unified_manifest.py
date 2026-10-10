import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_dinov2_unified_manifest import digest, main


class UnifiedManifestTests(unittest.TestCase):
    def test_selected_threshold_reuses_only_matching_test_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            val_dir = run_dir / "unified_val_predictions"
            val_dir.mkdir()
            (val_dir / "best_threshold.json").write_text(json.dumps({
                "split": "val", "checkpoint": str(checkpoint),
                "tile_size": 512, "overlap_stride": 256, "evaluation_size": 1024,
            }), encoding="utf-8")
            with (val_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["threshold", "iou"])
                writer.writeheader()
                writer.writerow({"threshold": "0.45", "iou": "0.7"})
            (val_dir / "threshold_0.45" / "surface").mkdir(parents=True)
            (val_dir / "threshold_0.45" / "surface" / "1_pred.png").write_bytes(b"mask")

            command = ["prepare", "--run_dir", str(run_dir), "--data_root", str(run_dir)]
            with patch("sys.argv", command):
                main()
            manifest = json.loads((run_dir / "unified_manifest.json").read_text(encoding="utf-8"))
            model = manifest["models"][0]
            selection = {"selected_on": "val", "models": {"dinov2_h0": {
                "threshold": 0.45,
                "identity": digest({"name": model["name"], "provenance": model["provenance"]}),
            }}}
            selection_path = run_dir / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            existing = run_dir / "test_selected_threshold"
            (existing / "surface").mkdir(parents=True)
            (existing / "test_results.json").write_text(json.dumps({
                "split": "test", "checkpoint": str(checkpoint), "threshold": 0.45,
                "tile_size": 512, "overlap_stride": 256,
            }), encoding="utf-8")

            with patch("sys.argv", command + ["--selection", str(selection_path)]):
                main()
            manifest = json.loads((run_dir / "unified_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["models"][0]["predictions"]["test"][0]["path"],
                             str(existing / "surface"))
            selected = json.loads((run_dir / "unified_selected_threshold.json").read_text(encoding="utf-8"))
            self.assertEqual(selected["threshold"], 0.45)

            report = json.loads((existing / "test_results.json").read_text(encoding="utf-8"))
            report["threshold"] = 0.5
            (existing / "test_results.json").write_text(json.dumps(report), encoding="utf-8")
            with patch("sys.argv", command + ["--selection", str(selection_path)]):
                main()
            manifest = json.loads((run_dir / "unified_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["models"][0]["predictions"]["test"][0]["path"],
                             str(run_dir / "unified_test_predictions" / "surface"))


if __name__ == "__main__":
    unittest.main()
