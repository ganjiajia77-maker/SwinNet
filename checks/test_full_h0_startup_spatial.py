import unittest

import numpy as np

from diagnostics.analyze_full_h0_startup_spatial import addition_components, fixed_gaps, gap_status


class SpatialTests(unittest.TestCase):
    def example(self):
        old = np.zeros((40,40),dtype=bool)
        old[20,4:10] = True
        old[20,22:28] = True
        gt = np.zeros_like(old)
        gt[20,4:28] = True
        return old,gt

    def test_gross_added_components_partition_pixels_and_categories(self):
        old,gt = self.example()
        new = old.copy()
        new[20,10:22] = True  # Actual positive connection.
        new[19,8:24] = True   # Background connection between old road pieces.
        new[2:4,2:4] = True   # Independent FP island.
        new[21,5:8] = True    # Road edge expansion.
        rows = addition_components(old,new,gt)
        fp = [row for row in rows if row['kind']=='added_fp']
        self.assertEqual(sum(row['pixels'] for row in fp),int((new & ~old & ~gt).sum()))
        self.assertEqual({row['category'] for row in fp},
                         {'isolated_island','bridge_proxy','road_edge_expansion'})
        self.assertEqual(sum(row['pixels'] for row in rows if row['kind']=='added_tp'),12)

    def test_frozen_local_gap_can_be_closed(self):
        old,gt = self.example()
        gaps = fixed_gaps(old.astype(float),gt,gt,.45)
        self.assertEqual(len(gaps),1)
        self.assertFalse(gap_status(old,gaps[0])['closed_proxy'])
        self.assertTrue(gap_status(gt,gaps[0])['closed_proxy'])

    def test_far_detour_is_not_counted_as_local_closure(self):
        old,gt = self.example()
        gap = fixed_gaps(old.astype(float),gt,gt,.45)[0]
        new = gt.copy()
        new[20,15] = False
        new[2:21,4] = True
        new[2:21,27] = True
        new[2,4:28] = True
        status = gap_status(new,gap)
        self.assertGreaterEqual(status['coverage'],.8)
        self.assertFalse(status['local_anchor_connection'])
        self.assertFalse(status['closed_proxy'])


if __name__=='__main__':
    unittest.main()
