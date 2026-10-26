import unittest

import numpy as np

from diagnostics.compare_v6_v7_matched import (
    choose_thresholds,
    owner_bounds,
    threshold_for_count,
)
from mass_protocol import sliding_positions


class MatchedComparisonTests(unittest.TestCase):
    def test_exact_count_matching_uses_separate_operating_points(self):
        record = {
            'gt': np.array([[1, 1, 1], [0, 0, 0]], dtype=bool),
            'v6': np.array([[.7, .6, .4], [.65, .2, .1]], dtype=np.float32),
            'v7': np.array([[.9, .58, .51], [.8, .7, .1]], dtype=np.float32),
        }
        points, residuals, scan = choose_thresholds([record], .55, .001)
        self.assertEqual(residuals['v7_match_recall']['tp_difference'], 0)
        self.assertEqual(residuals['v7_match_area']['foreground_area_difference'], 0)
        self.assertNotEqual(points['v7_match_recall'], points['v7_match_area'])
        self.assertGreater(len(scan), 100)
        at_55 = next(row for row in scan if row['threshold'] == .55)
        self.assertEqual(at_55['tp'], int(((record['v7'] >= .55) & record['gt']).sum()))
        self.assertEqual(at_55['area'], int((record['v7'] >= .55).sum()))

    def test_unavoidable_score_ties_report_nearest_count(self):
        scores = np.array([.1, .5, .5, .9], dtype=np.float32)
        threshold = threshold_for_count(scores, 2)
        self.assertEqual(abs(int((scores >= threshold).sum()) - 2), 1)

    def test_owner_windows_partition_native_image_once(self):
        positions = sliding_positions(1500, 512, 256)
        coverage = np.zeros(1500, dtype=np.uint8)
        for index, start in enumerate(positions):
            lower, upper = owner_bounds(positions, index, 512, 1500)
            coverage[start + lower:start + upper] += 1
        self.assertTrue(np.all(coverage == 1))


if __name__ == '__main__':
    unittest.main()
