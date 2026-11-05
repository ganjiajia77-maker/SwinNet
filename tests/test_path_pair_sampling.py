import unittest
import numpy as np
from losses.path_pair_sampling import sample_pairs, _walk


class PairSamplingTest(unittest.TestCase):
    def test_local_disconnected_roads_and_real_positive_paths(self):
        road=np.zeros((128,128),bool)
        road[32,16:112]=True;road[56,16:112]=True
        positive,negative,stats=sample_pairs(road,road,np.zeros((32,32),bool),
            np.random.default_rng(1234),max_pos=4,max_neg=4,attempts=100,margin=8)
        self.assertTrue(positive);self.assertTrue(negative)
        for pair in positive:
            self.assertTrue(all(road[y,x] for y,x in pair['path']))
            self.assertGreaterEqual(len(pair['path'])-1,16)
            self.assertLessEqual(len(pair['path'])-1,64)
        for pair in negative:
            self.assertNotEqual(pair['a'][0],pair['b'][0])

    def test_ambiguous_cells_skip_positive_paths(self):
        road=np.zeros((128,128),bool);road[64,8:120]=True
        amb=np.ones((32,32),bool)
        positive,negative,stats=sample_pairs(road,road,amb,np.random.default_rng(1),attempts=10,margin=4)
        self.assertEqual(positive,[]);self.assertGreater(stats['positive_ambiguous'],0)

    def test_diagonal_arc_length_uses_native_pixel_distance(self):
        road=np.eye(128,dtype=bool)
        path=_walk((20,20),road,16,64,np.random.default_rng(3))
        self.assertIsNotNone(path)
        length=sum(np.hypot(y2-y1,x2-x1) for (y1,x1),(y2,x2) in zip(path,path[1:]))
        self.assertGreaterEqual(length,16);self.assertLessEqual(length,64)

if __name__=='__main__':unittest.main()
