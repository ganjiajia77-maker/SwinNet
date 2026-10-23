import unittest

import numpy as np

from diagnose_gap_candidates import average_precision, gap_covered, ranked_pairs
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


if __name__ == '__main__':
    unittest.main()
