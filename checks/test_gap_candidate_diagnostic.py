import unittest

import numpy as np

from diagnose_gap_candidates import (average_precision, classify_gap, gap_covered,
                                     oracle_proposals, ranked_pairs)
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


if __name__ == '__main__':
    unittest.main()
