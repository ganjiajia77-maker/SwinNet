import unittest

import numpy as np

from h0_paired_changes import fixed_gap_audit, gap_changes, pixel_changes


class PairedChangesTest(unittest.TestCase):
    def test_pixel_categories_and_fixed_gap(self):
        gt = np.zeros((30, 40), bool)
        gt[15, 4:36] = True
        full = gt.copy()
        off = gt.copy()
        off[15, 18:21] = False
        off[4, 4] = True
        full[5, 5] = True
        h_off = full.copy()
        predictions = {'on': full, 'global_off': off, 'h_direct_off': h_off}
        row = pixel_changes(gt, gt, predictions)[0]
        self.assertEqual((row['repaired_fn'], row['added_fn'],
                          row['removed_fp'], row['added_fp']), (3, 0, 1, 1))
        self.assertEqual(row['upward_crossings'], 4)
        self.assertEqual(row['downward_crossings'], 1)
        gaps, stats = fixed_gap_audit(gt, predictions, gt, pad=2)
        self.assertEqual(stats['eligible'], 1)
        self.assertEqual(gaps[0]['gt_skeleton_pixels'], 3)
        self.assertTrue(gaps[0]['on_local_connected'])
        self.assertFalse(gaps[0]['global_off_local_connected'])
        self.assertEqual(gap_changes(gaps)['global_off']['local']['off_to_full_connected'], 1)

    def test_local_and_global_connectivity_are_distinct(self):
        gt = np.zeros((30, 40), bool)
        gt[15, 4:36] = True
        full = gt.copy()
        off = gt.copy()
        off[15, 18:21] = False
        off[10:16, 12] = True
        off[10:16, 27] = True
        off[10, 12:28] = True
        predictions = {'on': full, 'global_off': off, 'h_direct_off': full}
        gaps, _ = fixed_gap_audit(gt, predictions, gt, pad=1)
        self.assertEqual(len(gaps), 1)
        self.assertTrue(gaps[0]['global_off_global_connected'])
        self.assertFalse(gaps[0]['global_off_local_connected'])
        self.assertFalse(gaps[0]['global_off_local_gt_supported_connected'])
        wide, _ = fixed_gap_audit(gt, predictions, gt, pad=10, gt_tolerance=2)
        self.assertTrue(wide[0]['global_off_local_connected'])
        self.assertFalse(wide[0]['global_off_local_gt_supported_connected'])


if __name__ == '__main__':
    unittest.main()
