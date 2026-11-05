import unittest
import numpy as np
import torch
from losses.cross_scale_path_loss import widest_path,selected_vertex_loss,selected_edge_loss
from topology_direction_constants import CONNECTIVITY_DIRECTIONS,CONNECTIVITY_OPPOSITE


class CrossScalePathTest(unittest.TestCase):
    def test_widest_path_avoids_weak_shortcut(self):
        a=np.array([[.9,.1,.9],[.9,.9,.9],[.1,.1,.1]])
        path=widest_path(a,(0,0),(0,2))
        self.assertIsNotNone(path)
        self.assertGreaterEqual(min(a[p] for p in path),.9)
        self.assertNotIn((0,1),path)

    def test_opposite_direction_is_checked(self):
        a=np.ones((2,2));edge=np.ones((8,2,2));valid=np.ones((8,2,2),bool)
        index=list(CONNECTIVITY_DIRECTIONS).index((0,1));opposite=CONNECTIVITY_OPPOSITE[index]
        valid[opposite,0,1]=False
        allowed=np.array([[True,True],[False,False]])
        self.assertIsNone(widest_path(a,(0,0),(0,1),allowed=allowed,edge_scores=edge,edge_valid=valid))

    def test_selected_losses_have_gradients_only_on_labelled_path(self):
        x=torch.zeros((3,3),requires_grad=True)
        mask=np.zeros((3,3),bool);mask[1,1]=True
        selected_vertex_loss(x,[(1,0),(1,1),(1,2)],0,mask).backward()
        self.assertGreater(x.grad[1,1].item(),0)
        self.assertEqual(int(torch.count_nonzero(x.grad)),1)
        edges=torch.zeros((8,3,3),requires_grad=True);valid=np.ones((8,3,3),bool)
        selected_edge_loss(edges,[(1,0),(1,1)],1,valid).backward()
        self.assertEqual(int(torch.count_nonzero(edges.grad)),2)
        self.assertTrue((edges.grad[edges.grad!=0]<0).all())

if __name__=='__main__':unittest.main()
