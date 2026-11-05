import unittest
import torch
from scripts.diagnose_h0_oracle_bad_edges import OracleHook
from networks.native_h0_relations import shifted
from topology_direction_constants import CONNECTIVITY_DIRECTIONS


class OracleBadEdgesTest(unittest.TestCase):
    def test_return_weights_to_stay_and_preserve_other_edges(self):
        torch.manual_seed(4)
        module=torch.nn.Module()
        module.out=torch.nn.Conv2d(3,3,1,bias=False)
        module.gamma=torch.tensor(.2)
        h=torch.randn(1,3,5,5)
        choice=torch.softmax(torch.randn(1,9,5,5),dim=1)
        mixture=sum(choice[:,i+1:i+2]*(shifted(h,dy,dx)-h)
                    for i,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS))
        delta=module.gamma*module.out(mixture)
        info={'h2_before_relation':h,'relation_weights':choice,
              'local_delta':delta,'highres_skeleton_feat':h+delta}
        hook=OracleHook();hook.mask=torch.zeros(1,8,5,5,dtype=torch.bool)
        hook.mask[:,2,2,2]=True
        output=hook(module,(),(None,None,None,None,None,info))
        new=output[5]['relation_weights']
        self.assertTrue(torch.allclose(new.sum(1),choice.sum(1),atol=1e-7))
        self.assertEqual(float(new[0,3,2,2]),0.)
        self.assertTrue(torch.allclose(new[:,1:3],choice[:,1:3]))
        self.assertTrue(torch.allclose(new[:,4:],choice[:,4:]))
        expected=choice[0,0,2,2]+choice[0,3,2,2]
        self.assertTrue(torch.allclose(new[0,0,2,2],expected))
        self.assertTrue(torch.allclose(output[5]['highres_skeleton_feat'],h+output[5]['local_delta']))

    def test_zero_mask_preserves_update(self):
        module=torch.nn.Module();module.out=torch.nn.Conv2d(3,3,1,bias=False);module.gamma=torch.tensor(.1)
        h=torch.randn(1,3,5,5);choice=torch.softmax(torch.randn(1,9,5,5),1)
        delta=module.gamma*module.out(sum(choice[:,i+1:i+2]*(shifted(h,dy,dx)-h)
            for i,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS)))
        hook=OracleHook();hook.mask=torch.zeros(1,8,5,5,dtype=torch.bool)
        info={'h2_before_relation':h,'relation_weights':choice,'local_delta':delta,'highres_skeleton_feat':h+delta}
        changed=hook(module,(),(None,None,None,None,None,info))[5]
        self.assertTrue(torch.allclose(changed['local_delta'],delta,atol=1e-7))
        self.assertTrue(torch.equal(changed['relation_weights'],choice))

if __name__=='__main__':unittest.main()
