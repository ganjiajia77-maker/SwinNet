import unittest
from collections import Counter
from unittest.mock import patch
import numpy as np
import torch
from datasets.native_relation_targets import native_targets
from scripts import evaluate_native_h0_pair as audit


class NativePairEvaluationTest(unittest.TestCase):
    def test_C_D_and_edge_statistics_use_native_masks(self):
        mask=np.zeros((512,512),bool);mask[250:253,20:490]=1
        skeleton=np.zeros_like(mask);skeleton[251,20:490]=1
        labels=native_targets(mask,skeleton);target=labels['connectivity']
        logits=torch.from_numpy(np.where(target>0,5.,-5.).astype(np.float32))[None]
        direction=torch.zeros(1,2,128,128);direction[:,0]=1
        weights=torch.full((1,9,128,128),1./9.)
        outputs=(None,None,None,None,[{'stage':2,'connectivity':logits,'direction':direction},
                                      {'stage':3,'connectivity':logits,'direction':direction,'relation_weights':weights}])
        stats=Counter();audit.update_diagnostics(stats,outputs,mask,skeleton,np.ones((128,128),bool))
        self.assertGreater(stats['C3_pos_edges'],0)
        self.assertEqual(stats['C3_pos_edges'],stats['C3_pos_accepted'])
        self.assertEqual(stats['C3_neg_accepted'],0)
        self.assertGreater(stats['D3_valid_cells'],0)
        self.assertLess(stats['D3_angle_sum'],1e-4)
        self.assertGreater(stats['choice_correct_sum'],0)

    def test_fixed_reference_and_local_closure(self):
        gt=np.zeros((40,40),bool);gt[19:22,2:38]=1
        old=gt.astype(np.float32);old[:,14:26]=0
        case={'case':'toy','gt':gt,'p0':old,'A':old,'B':gt.astype(np.float32)}
        manifest=audit.fixed_reference([case]);self.assertGreater(manifest['count'],0)
        result=audit.point([case],'B',.5,manifest)
        self.assertGreater(result['local_closed'],0)


if __name__=='__main__':unittest.main()
