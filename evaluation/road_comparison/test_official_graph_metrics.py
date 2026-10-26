"""Invariant checks for the pinned official metric adapter."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.sparse import csr_matrix

from compare_predictions import (compatible_selection, digest, evaluate,
                                 select_validation, threshold_protocol_hash,
                                 validate_manifest)
from official_graph_metrics import _cosmiq, compact_graph
from road_metrics import DEFAULT_PROTOCOL, image_metrics, prepare_target, skeleton_graph


class OfficialGraphMetricsTest(unittest.TestCase):
    def test_upstream_sources_match_pinned_checksums(self):
        vendor = Path(__file__).parent / "vendor"
        manifest = json.loads((vendor / "UPSTREAM.json").read_text(encoding="utf-8"))
        for package, filename, field in (
            ("sat2graph", "topo.upstream.py.txt", "topo_source_sha256"),
            ("sat2graph", "main.upstream.py.txt", "main_source_sha256"),
            ("sat2graph", "graph.upstream.py.txt", "graph_source_sha256"),
            ("cosmiq", "apls.upstream.py.txt", "apls_source_sha256"),
            ("cosmiq", "apls_utils.upstream.py.txt", "utils_source_sha256"),
        ):
            actual = hashlib.sha256((vendor / package / filename).read_bytes()).hexdigest()
            record = "cosmiq_apls" if package == "cosmiq" else package
            self.assertEqual(actual, manifest[record][field])
        self.assertIn("apls.upstream.py.txt", _cosmiq().compute_apls_metric.__code__.co_filename)

    def test_compaction_preserves_every_pixel_edge_and_closed_ring(self):
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        rows = [0, 1, 1, 2, 2, 3, 3, 0]
        cols = [1, 0, 2, 1, 3, 2, 0, 3]
        pixel = csr_matrix((np.ones(8), (rows, cols)), shape=(4, 4))
        graph = compact_graph(points, pixel)
        self.assertEqual(graph.number_of_edges(), 3)
        self.assertAlmostEqual(sum(data["length"] for *_, data in graph.edges(data=True)), 4)
        self.assertAlmostEqual(sum(data["geometry"].length for *_, data in graph.edges(data=True)), 4)

    def test_official_functions_detect_gap(self):
        gt = np.zeros((96, 96), dtype=bool)
        gt[45, 10:86] = True
        broken = gt.copy()
        broken[45, 43:53] = False
        prepared = prepare_target(gt, DEFAULT_PROTOCOL)
        whole = image_metrics(gt, prepared, DEFAULT_PROTOCOL)
        gap = image_metrics(broken, prepared, DEFAULT_PROTOCOL)
        self.assertEqual(whole["topo_f1"], 1.0)
        self.assertEqual(whole["apls_bidirectional"], 1.0)
        self.assertLess(gap["topo_f1"], whole["topo_f1"])
        self.assertLess(gap["apls_bidirectional"], whole["apls_bidirectional"])
        self.assertGreater(gap["tlts_infeasible"], 0)
        self.assertGreater(gap["apls_gt_pairs"], 0)
        again = image_metrics(gt, prepared, DEFAULT_PROTOCOL)
        self.assertEqual(again["apls_bidirectional"], whole["apls_bidirectional"])
        self.assertEqual(again["apls_gt_pairs"], whole["apls_gt_pairs"])

    def test_apls_uses_original_graph_nodes_and_reuses_gt_paths(self):
        from official_graph_metrics import (_official_apls, prepare_apls_reference,
                                            prepare_official_graph)

        mask = np.zeros((16, 900), dtype=bool)
        mask[8, 10:890] = True
        points, pixel = skeleton_graph(mask)
        ground = prepare_official_graph(points, pixel)
        proposed = prepare_official_graph(points, pixel, id_offset=1_000_000_000)
        original_nodes = ground["compact"].number_of_nodes()
        reference = prepare_apls_reference(ground)
        total, forward, reverse, pairs, _ = _official_apls(ground, proposed, DEFAULT_PROTOCOL)
        self.assertIs(prepare_apls_reference(ground), reference)
        self.assertEqual(pairs["gt_controls"], original_nodes)
        self.assertEqual(pairs["pred_controls"], proposed["compact"].number_of_nodes())
        self.assertEqual(original_nodes, 2)
        self.assertEqual(pairs["gt"], 2)
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(forward, 1.0)
        self.assertAlmostEqual(reverse, 1.0)

    def test_node_only_apls_matches_pinned_reference_without_midpoints(self):
        from contextlib import redirect_stdout
        from official_graph_metrics import (_NullWriter, _official_apls,
                                            prepare_official_graph)

        line = np.zeros((24, 120), dtype=bool)
        line[12, 8:112] = True
        broken = line.copy()
        broken[12, 54:61] = False
        gt_points, gt_pixel = skeleton_graph(line)
        pred_points, pred_pixel = skeleton_graph(broken)
        ground = prepare_official_graph(gt_points, gt_pixel)
        proposed = prepare_official_graph(pred_points, pred_pixel,
                                          id_offset=1_000_000_000)
        apls = _cosmiq()
        with patch.object(apls, "create_graph_midpoints",
                          side_effect=lambda graph, **kwargs: (graph, [], [])):
            with redirect_stdout(_NullWriter()):
                result = apls.make_graphs(
                    ground["compact"].copy(), proposed["compact"].copy(),
                    weight="length", max_snap_dist=DEFAULT_PROTOCOL["apls_snap_radius"],
                    allow_renaming=True)
                expected = apls.compute_apls_metric(
                    result[6], result[7], result[8], result[9],
                    result[4], result[5],
                    min_path_length=DEFAULT_PROTOCOL["apls_min_path_px"])
        actual = _official_apls(ground, proposed, DEFAULT_PROTOCOL)
        for expected_score, actual_score in zip(expected, actual[:3]):
            self.assertAlmostEqual(expected_score, actual_score, places=10)

    def test_previous_validation_threshold_selection_is_usable(self):
        selection = {"schema": 2, "selected_on": "val", "protocol_hash": digest({}),
                     "threshold_protocol_hash": threshold_protocol_hash(DEFAULT_PROTOCOL)}
        self.assertTrue(compatible_selection(selection, DEFAULT_PROTOCOL))

    def test_saved_mask_pipeline_preserves_selected_threshold(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            mask = np.zeros((96, 96), np.uint8)
            mask[45, 10:86] = 255
            for split in ("val", "test"):
                for folder in (root / "data" / split / "mask",
                               root / "data" / split / "image",
                               root / "pred" / split / "surface"):
                    folder.mkdir(parents=True)
                    cv2.imwrite(str(folder / "road.png"), mask)
            manifest = {
                "data_root": str(root / "data"), "protocol": {"image_size": 96,
                    "apls_max_nodes": 64},
                "models": [{"name": "single", "provenance": {"checkpoint": "fixed"},
                            "predictions": {
                                split: [{"threshold": 0.3,
                                         "path": str(root / "pred" / split / "surface")}]
                                for split in ("val", "test")}}],
            }
            protocol = validate_manifest(manifest)
            self.assertEqual(protocol["legacy_apls_max_nodes_ignored"], 64)
            val_output, test_output = root / "val_result", root / "test_result"
            val_output.mkdir()
            test_output.mkdir()
            selection = select_validation(manifest, root, protocol, val_output, 1)
            report = evaluate(manifest, root, protocol, "test", selection, test_output, 1)
            self.assertEqual(report["summaries"][0]["threshold"], 0.3)
            self.assertEqual(report["summaries"][0]["apls_bidirectional"], 1.0)
            self.assertTrue((test_output / "test_report.json").exists())


if __name__ == "__main__":
    unittest.main()
