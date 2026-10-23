import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from diagnose_gap_candidates import (average_precision, classify_gap, gap_covered,
                                     gt_path_evidence, oracle_proposals, ranked_pairs,
                                     save_gap_overlay)
from gap_candidates import endpoints, generate_candidates
from local_gap_config import LocalGapConfig


class CandidateDiagnosticTests(unittest.TestCase):
    def test_replay_matches_production(self):
        probability = np.zeros((32, 32), np.float32) + .01
        probability[16, 4:12] = .99
        probability[16, 20:28] = .99
        cfg = LocalGapConfig()
        points, _ = endpoints(probability, cfg)
        replay = ranked_pairs(points, cfg)
        np.testing.assert_array_equal(replay['top64'], generate_candidates(probability[None], cfg)[0]['edges'])
        self.assertEqual([len(replay[name]) for name in
                          ('endpoints', 'distance', 'direction', 'top3', 'top64')],
                         [6, 6, 1, 1, 1])

    def test_opposite_flanks_required(self):
        a = np.zeros((12, 20), bool); a[6, 2] = True
        b = np.zeros_like(a); b[6, 16] = True
        gap = (1, (0, 12, 0, 20), None, [(a, None, None), (b, None, None)], [(0, 1)], 10)
        self.assertTrue(gap_covered(gap, np.array([[[6, 2], [6, 16]]], np.float32), np.ones(2), 1))
        self.assertFalse(gap_covered(gap, np.array([[[6, 2], [6, 3]]], np.float32), np.ones(2), 1))

    def test_ap_ignores_ambiguous_edges(self):
        self.assertEqual(average_precision([1, 0, -1], [.9, .1, 10]), 1.)
        self.assertIsNone(average_precision([0, -1], [.9, .1]))

    def test_gap_endpoint_class_distinguishes_flanks_and_components(self):
        a = np.zeros((12, 20), bool); a[6, 2] = True
        b = np.zeros_like(a); b[6, 16] = True
        gap = (1, (0, 12, 0, 20), None, [(a, None, None), (b, None, None)], [(0, 1)], 10)
        points = np.array([[6, 2], [6, 16]])
        skeleton = np.zeros((12, 20), bool); skeleton[6, 2:17] = True
        components = np.zeros((12, 20), np.int32)
        components[6, 2] = 1; components[6, 16] = 2
        self.assertEqual(classify_gap(gap, skeleton, points, components, 1)[0], 'both_flanks_disconnected')
        components[6, 16] = 1
        self.assertEqual(classify_gap(gap, skeleton, points, components, 1)[0], 'both_flanks_same_component')
        self.assertEqual(classify_gap(gap, skeleton, points[:1], components, 1)[0],
                         'endpoint_on_one_or_more_flanks')

    def test_gt_oracle_proposal_uses_positive_local_gt_path(self):
        skeleton = np.zeros((512, 512), bool)
        skeleton[256, 100:401] = True
        mask = skeleton.copy()
        left = np.zeros((30, 120), bool); left[16, 20] = True
        right = np.zeros_like(left); right[16, 100] = True
        gap = (1, (240, 270, 200, 320), None,
               [(left, np.array([20., 16.]), {1}),
                (right, np.array([100., 16.]), {2})], [(0, 1)], 20)
        edges, geometry = oracle_proposals([gap], (128, 128), LocalGapConfig(), skeleton, mask)
        self.assertEqual(edges.shape, (1, 2, 2))
        self.assertEqual(geometry.shape, (1, 8))

    def test_gt_path_evidence_is_feature_sized_float_map(self):
        region = np.zeros((20, 20), bool); region[8, 5:15] = True
        anchor_a = np.zeros_like(region); anchor_a[8, 4] = True
        anchor_b = np.zeros_like(region); anchor_b[8, 15] = True
        gap = (1, (20, 40, 30, 50), region, [(anchor_a, None, None),
                                               (anchor_b, None, None)], [(0, 1)], 10)
        evidence = gt_path_evidence([gap], (16, 16), 'cpu', torch.float32)
        self.assertEqual(tuple(evidence.shape), (1, 1, 16, 16))
        self.assertEqual(evidence.dtype, torch.float32)
        self.assertGreater(float(evidence.sum()), 0)

    def test_overlay_writer_creates_gap_crop(self):
        image = torch.zeros((3, 64, 64))
        skeleton = np.zeros((64, 64), bool); skeleton[30, 10:50] = True
        region = np.zeros_like(skeleton); region[30, 25:35] = True
        anchor_a = np.zeros((20, 20), bool); anchor_a[10, 5] = True
        anchor_b = np.zeros_like(anchor_a); anchor_b[10, 15] = True
        gap = (1, (20, 40, 20, 40), region[20:40, 20:40],
               [(anchor_a, None, None), (anchor_b, None, None)], [(0, 1)], 10)
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'class' / 'example.png'
            save_gap_overlay(image, skeleton, skeleton, np.asarray([[30, 20]]), gap, output)
            self.assertTrue(output.is_file())


if __name__ == '__main__':
    unittest.main()
