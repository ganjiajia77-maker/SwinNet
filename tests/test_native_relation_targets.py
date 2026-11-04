import unittest
import numpy as np
from datasets.native_relation_targets import native_targets
from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE


class NativeRelationTargetsTest(unittest.TestCase):
    def test_crossing_and_opposite_agree_with_boundaries(self):
        road=np.zeros((32,32),bool);road[13:16,1:31]=1
        skel=np.zeros_like(road);skel[14,1:31]=1
        result=native_targets(road,skel,junction_radius=0)
        c=result['connectivity'];v=result['connectivity_valid']
        self.assertTrue(c[2,3,2]);self.assertTrue(c[6,3,3])
        self.assertFalse(v[0,0].any());self.assertFalse(v[2,:,-1].any())
        for d,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            for y in range(8):
                for x in range(8):
                    q=(y+dy,x+dx)
                    if 0<=q[0]<8 and 0<=q[1]<8:
                        self.assertEqual(c[d,y,x],c[CONNECTIVITY_OPPOSITE[d],q[0],q[1]])
                        self.assertEqual(v[d,y,x],v[CONNECTIVITY_OPPOSITE[d],q[0],q[1]])

    def test_adjacent_occupied_cells_without_native_contact_are_negative(self):
        road=np.zeros((16,16),bool);road[4:7,1:3]=1;road[4:7,5:7]=1
        result=native_targets(road,np.zeros_like(road))
        self.assertTrue(result['connectivity_valid'][2,1,0])
        self.assertEqual(result['connectivity'][2,1,0],0.)

    def test_multiple_fragments_and_boundary_validity(self):
        road=np.zeros((16,16),bool);road[4,4]=1;road[7,7]=1;road[4:8,8:12]=1
        result=native_targets(road,np.zeros_like(road))
        self.assertTrue(result['ambiguous_cells'][1,1])
        self.assertFalse(result['connectivity_valid'][:,1,1].any())
        self.assertFalse(result['connectivity_valid'][6,1,2])

    def test_native_horizontal_and_vertical_axial_direction(self):
        road=np.zeros((64,64),bool);road[30:33,4:60]=1
        skel=np.zeros_like(road);skel[31,4:60]=1
        first=native_targets(road,skel);second=native_targets(np.rot90(road).copy(),np.rot90(skel).copy())
        self.assertGreater(first['direction_valid'].sum(),0)
        self.assertGreater(second['direction_valid'].sum(),0)
        np.testing.assert_allclose(first['direction'][0][first['direction_valid']],1.,atol=1e-6)
        np.testing.assert_allclose(second['direction'][0][second['direction_valid']],-1.,atol=1e-6)

    def test_non_aligned_crop_and_diagonal_crossing(self):
        image=np.zeros((40,40),bool);image[10,5:25]=1
        crop=image[3:35,3:35];result=native_targets(crop,np.zeros_like(crop))
        self.assertTrue(result['connectivity'][2,1,0])
        self.assertTrue(result['connectivity'][6,1,1])
        road=np.zeros((16,16),bool);road[3,3]=1;road[4,4]=1
        diagonal=native_targets(road,np.zeros_like(road))
        self.assertEqual(diagonal['connectivity'][3,0,0],1.)
        self.assertEqual(diagonal['connectivity'][7,1,1],1.)


if __name__=='__main__':unittest.main()
