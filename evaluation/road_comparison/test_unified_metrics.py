import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from scipy.sparse.csgraph import dijkstra

from compare_predictions import (case_id, evaluate, import_declared_selection,
                                 index_files, prediction_index, read_binary,
                                 select_validation, validate_manifest)
from road_metrics import (DEFAULT_PROTOCOL, image_metrics, prepare_target,
                          skeleton_graph, skeletonize, snap_nodes, summarize)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(DEFAULT_PROTOCOL, image_size=32)
        self.road = np.zeros((32, 32), dtype=bool)
        self.road[15:18, 3:29] = True

    def test_perfect_and_broken_paths(self):
        prepared = prepare_target(self.road, self.protocol)
        perfect = image_metrics(self.road, prepared, self.protocol)
        self.assertEqual(perfect["cldice"], 1)
        self.assertEqual(perfect["apls_bidirectional_approx"], 1)
        broken = self.road.copy()
        broken[:, 14:18] = False
        result = image_metrics(broken, prepared, self.protocol)
        self.assertGreater(result["break_rate"], 0)
        self.assertEqual(result["pred_components"], 2)
        self.assertLess(result["apls_gt_to_pred_approx"], 1)

    def test_snap_inclusive_radius_and_row_major_tie(self):
        self.assertEqual(snap_nodes(np.array([[0, 0]]), np.array([[3, 4]]), 5)[0], 0)
        self.assertEqual(snap_nodes(np.array([[0, 0]]), np.array([[3, 4]]), 4.99)[0], -1)
        self.assertEqual(snap_nodes(np.array([[1, 1]]), np.array([[0, 1], [1, 0]]), 1)[0], 0)

    def test_diagonal_edge_lengths(self):
        mask = np.eye(4, dtype=bool)
        _, graph = skeleton_graph(mask)
        paths = dijkstra(graph, directed=False, indices=[0])
        self.assertAlmostEqual(paths[0, 3], 3 * np.sqrt(2))

    def test_empty_and_global_aggregation(self):
        empty = np.zeros_like(self.road)
        result = image_metrics(empty, prepare_target(empty, self.protocol), self.protocol)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        self.assertEqual(result["apls_bidirectional_approx"], 0)
        rows = [dict(result, tp=1, fp=0, fn=0), dict(result, tp=0, fp=99, fn=0)]
        self.assertEqual(summarize(rows)["iou"], 0.01)

    def test_border_thinning_matches_explicit_padding(self):
        road = self.road.copy()
        road[:7, :3] = True
        actual = skeletonize(road)
        self.assertTrue(np.array_equal(actual, skeletonize(np.pad(road, 2))[2:-2, 2:-2]))
        if hasattr(cv2, "ximgproc"):
            expected = cv2.ximgproc.thinning(np.pad(road.astype(np.uint8) * 255, 1),
                                           thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)[1:-1, 1:-1] > 0
            self.assertTrue(np.array_equal(actual, expected))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = dict(DEFAULT_PROTOCOL, image_size=16)
        road = np.zeros((16, 16), dtype=np.uint8)
        road[7, 2:14] = 255
        empty = np.zeros_like(road)
        for split in ("val", "test"):
            self.save(f"data/{split}/mask/1_mask.png", road)
        for name in ("swin", "sam", "coanet"):
            self.save(f"{name}/val_low/1_sat_pred.png", road)
            self.save(f"{name}/val_high/1_sat_pred.png", empty)
            # Test would favor the higher threshold, but selection must stay val-based.
            self.save(f"{name}/test_low/1_sat_pred.png", empty)
            self.save(f"{name}/test_high/1_sat_pred.png", road)
        self.manifest = {"data_root": "data", "protocol": {"image_size": 16}, "models": [
            {"name": name, "provenance": {"checkpoint": "same.pth"}, "predictions": {
                split: [{"threshold": 0.3, "path": f"{name}/{split}_low"},
                        {"threshold": 0.7, "path": f"{name}/{split}_high"}]
                for split in ("val", "test")}}
            for name in ("swin", "sam", "coanet")
        ]}
        self.output = self.root / "output"
        self.output.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, relative, mask):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cv2.imencode(".png", mask)[1].tobytes())

    def test_three_models_test_uses_val_selected_threshold(self):
        protocol = validate_manifest(self.manifest)
        selection = select_validation(self.manifest, self.root, protocol, self.output, 100)
        self.assertEqual(selection["models"]["swin"]["threshold"], 0.3)
        report = evaluate(self.manifest, self.root, protocol, "test", selection, self.output, 100)
        self.assertTrue(all(row["iou"] == 0 for row in report["summaries"]))
        self.assertEqual(len(set(report["binary_content_sha256"][name] for name in
                                 ("swin", "sam", "coanet"))), 1)
        self.assertTrue((self.output / "test_comparison.csv").exists())

    def test_strict_size_encoding_and_duplicates(self):
        with self.assertRaisesRegex(ValueError, "resizing"):
            read_binary(self.root / "data/val/mask/1_mask.png", 32)
        self.save("invalid/1.png", np.full((16, 16), 128, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "binary"):
            read_binary(self.root / "invalid/1.png", 16)
        self.save("swin/val_low/1_mask_pred.png", np.zeros((16, 16), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_files(self.root / "swin/val_low")
        self.assertEqual(case_id(Path("113_sat_surface_pred.png")), "113")

    def test_missing_case_and_skeleton_directory(self):
        self.save("wrong/2.png", np.zeros((16, 16), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Case set mismatch"):
            prediction_index({"path": "wrong"}, self.root, {"1": None})
        self.save("ske/1_skeleton_pred.png", np.zeros((16, 16), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "surface"):
            index_files(self.root / "ske")

    def test_protocol_and_model_identity_are_frozen(self):
        selection = select_validation(self.manifest, self.root, self.protocol, self.output, 100)
        with self.assertRaisesRegex(ValueError, "protocol"):
            evaluate(self.manifest, self.root, dict(self.protocol, apls_snap_radius=4),
                     "test", selection, self.output, 100)
        modified = copy.deepcopy(self.manifest)
        modified["models"][0]["provenance"]["checkpoint"] = "different.pth"
        with self.assertRaisesRegex(ValueError, "provenance"):
            evaluate(modified, self.root, self.protocol, "test", selection, self.output, 100)

    def test_import_is_explicit_and_requires_val_evidence(self):
        evidence = self.root / "val.log"
        evidence.write_text("selected on val: threshold=0.3", encoding="utf-8")
        for model in self.manifest["models"]:
            model["validation_selection"] = {"selected_on": "val", "criterion": "global_iou",
                                             "threshold": 0.3, "evidence": "val.log"}
        selected = import_declared_selection(self.manifest, self.root, self.protocol)
        self.assertIn("evidence_sha256", selected["models"]["swin"])
        self.manifest["models"][0]["validation_selection"]["selected_on"] = "test"
        with self.assertRaisesRegex(ValueError, "val"):
            import_declared_selection(self.manifest, self.root, self.protocol)


if __name__ == "__main__":
    unittest.main()
