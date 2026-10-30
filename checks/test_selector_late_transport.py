"""Read-only late-step interventions and immutable step-wise closure accounting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from checks.test_v9_state_gate import model as fake_v9
from networks.v9_source_selector import SourceSelectedRepair
from networks.v9_source_relation_labels import RelationLabels
from diagnose_selector_late_transport import LateIntervention,transfer_to_stay,tile_contrasts,case_metrics,trace_file


class LateTransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234); torch.set_num_threads(1)
        self.model=SourceSelectedRepair(fake_v9()).requires_grad_(False).eval()
        self.image=torch.randn(1,3,64,64)

    def test_rejected_mass_moves_to_stay_without_renormalising(self):
        choice=torch.tensor([.2]+[.1]*8).reshape(1,9,1,1); before=choice.clone()
        blocked=torch.tensor([1,0,0,1,0,0,0,0],dtype=torch.bool).reshape(1,8,1,1)
        result,removed=transfer_to_stay(choice,blocked)
        torch.testing.assert_close(result[:,:1],torch.full_like(removed,.4))
        self.assertTrue(torch.equal(result[:,1:][~blocked],choice[:,1:][~blocked]))
        self.assertTrue(bool((result[:,1:][blocked]==0).all()))
        self.assertTrue(torch.equal(choice,before)); torch.testing.assert_close(result.sum(1),choice.sum(1))

    def test_stop_late_preserves_first_step_and_proposal_anchor(self):
        with torch.no_grad():
            normal=self.model(self.image)
            with LateIntervention(self.model,'stop_late') as intervention: stopped=self.model(self.image)
        self.assertEqual(len(intervention.records),3)
        self.assertTrue(torch.equal(normal['u1'],stopped['u1']))
        self.assertTrue(bool((stopped['transport_delta'][:,1:]==0).all()))
        for k in (1,2):
            state=stopped['states'][k]
            torch.testing.assert_close(stopped['states'][k+1],state+stopped['proposal_delta'][:,k]+stopped['anchor_delta'][:,k])
            torch.testing.assert_close(stopped['proposal_delta'][:,k],stopped['beta'][:,k]*(stopped['b']-state))
            torch.testing.assert_close(stopped['anchor_delta'][:,k],stopped['gamma'][:,k]*(stopped['u0']-state))
        self.assertGreater(float((stopped['u2']-stopped['u1']).abs().max()),0.)

    def test_oracle_keeps_positive_unknown_and_first_step_weights(self):
        labels=torch.full((1,8,16,16),-1,dtype=torch.int8); labels[:,0]=0; labels[:,1]=1
        with torch.no_grad():
            with LateIntervention(self.model,'oracle_late',labels) as intervention: result=self.model(self.image)
        first=intervention.records[0]
        self.assertTrue(torch.equal(first['raw_choice'],first['actual_choice']))
        for record in intervention.records[1:]:
            self.assertTrue(bool((record['actual_choice'][:,1]==0).all()))
            self.assertTrue(torch.equal(record['raw_choice'][:,2:],record['actual_choice'][:,2:]))
            torch.testing.assert_close(record['actual_choice'][:,:1],record['raw_choice'][:,:1]+record['removed_mass'])

    def test_all_blocked_oracle_equals_late_stop_and_unknown_equals_normal(self):
        with torch.no_grad():
            normal=self.model(self.image)
            with LateIntervention(self.model,'stop_late'): stopped=self.model(self.image)
            with LateIntervention(self.model,'oracle_late',torch.zeros(1,8,16,16,dtype=torch.int8)): blocked=self.model(self.image)
            with LateIntervention(self.model,'oracle_late',torch.full((1,8,16,16),-1,dtype=torch.int8)): unknown=self.model(self.image)
        self.assertTrue(torch.equal(stopped['p1'],blocked['p1']))
        self.assertTrue(torch.equal(normal['p1'],unknown['p1']))

    def test_hooks_are_removed_on_error_and_training_mode_is_rejected(self):
        with self.assertRaises(RuntimeError):
            with LateIntervention(self.model,'stop_late'): raise RuntimeError('interrupted')
        self.assertEqual(len(self.model.selector._forward_hooks),0)
        self.model.train()
        with self.assertRaises(ValueError):
            with LateIntervention(self.model,'full_three'): pass

    def test_three_variants_share_backbone_and_do_not_change_weights(self):
        before={key:value.clone() for key,value in self.model.state_dict().items()}
        calls=[]; handle=self.model.original.frozen.register_forward_hook(lambda *args:calls.append(1))
        mask=torch.zeros(1,1,64,64); mask[:,:,28:36,4:60]=1.
        skeleton=torch.zeros_like(mask); skeleton[:,:,30,4:60]=1.
        try: variants,records,labels=tile_contrasts(self.model,self.image,mask,skeleton,RelationLabels())
        finally: handle.remove()
        self.assertEqual(len(calls),1)
        for name,result in variants.items():
            self.assertTrue(torch.equal(result['u1'],variants['full_three']['u1']))
            self.assertEqual(len(result['states']),4)
            self.assertFalse(result['p1'].requires_grad)
        for key,value in self.model.state_dict().items(): self.assertTrue(torch.equal(value,before[key]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.late_transport_test_') as temp:
            path=Path(temp)/'states.npz'; trace_file(path,variants,records,labels,(0,0))
            with np.load(path) as data:
                self.assertEqual(data['full_three_U0_U3'].shape[0],4)
                self.assertEqual(data['oracle_late_raw_choice'].shape[:2],(3,9))

    def test_per_step_counts_use_the_same_gap_pixels_and_anchors(self):
        gt=np.zeros((32,32),bool); gt[14:17,2:30]=True
        p0=gt.copy(); p0[:,13:19]=False
        initial=np.where(p0,.9,.1).astype(np.float32); repaired=np.where(gt,.9,.1).astype(np.float32)
        false=repaired.copy(); false[2:4,2:4]=.9
        entry={'shape':[32,32],'gaps':[{'id':1,'length':6,'length_bin':'3-7',
            'pixels':[15*32+x for x in range(13,19)],'anchors':[[15*32+12],[15*32+19]]}]}
        predictions={'full_three':np.stack((initial,false,repaired)),
            'stop_late':np.stack((initial,initial,initial)),'oracle_late':np.stack((initial,repaired,repaired))}
        rows=case_metrics({'case':'synthetic','gt':gt,'p0':initial},predictions,.5,entry)
        by_key={(row['mode'],row['step']):row for row in rows}
        self.assertEqual(by_key['full_three',2]['closed'],1)
        self.assertEqual(by_key['full_three',2]['newly_closed'],1)
        self.assertEqual(by_key['full_three',2]['new_fp_vs_previous'],4)
        self.assertEqual(by_key['full_three',3]['removed_fp_vs_previous'],4)
        self.assertEqual(by_key['stop_late',3]['closed'],0)
        self.assertEqual(by_key['oracle_late',3]['closed'],1)


if __name__=='__main__': unittest.main()
