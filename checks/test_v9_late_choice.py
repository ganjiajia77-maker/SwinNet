"""Focused choice loss tests and live three-step transport integration."""
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from networks.v9_late_choice_loss import late_choice_loss


def fixture(batch=1, width=1):
    edge = torch.zeros(batch,3,8,1,width,requires_grad=True)
    stay = torch.zeros(batch,3,1,1,width,requires_grad=True)
    valid = torch.ones_like(edge,dtype=torch.bool)
    labels = torch.zeros(batch,8,1,width,dtype=torch.int8)
    result = dict(source_edge_logits=edge,source_stay_logit=stay,source_valid=valid)
    result['source_choice']=torch.cat((stay,edge),2).softmax(2)
    info={'native':{'labels':labels},'rows':{'valid':valid.clone()}}
    return result,info


class ChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_numerical_examples(self):
        for mass in (.6,.1,.01):
            result,info=fixture()
            with torch.no_grad():
                result['source_stay_logit'].fill_(math.log(1-mass))
                result['source_edge_logits'].fill_(math.log(mass/8))
            loss,_=late_choice_loss(result,info)
            self.assertAlmostEqual(float(loss),-math.log(1-mass),places=6)

    def test_matches_probability_formula(self):
        result,info=fixture(2,4); torch.manual_seed(7)
        with torch.no_grad():
            result['source_edge_logits'].normal_(); result['source_stay_logit'].normal_()
        info['native']['labels'][:,4:]=1
        logits=torch.cat((result['source_stay_logit'],result['source_edge_logits']),2)
        result['source_choice']=logits.softmax(2)
        loss,_=late_choice_loss(result,info)
        expected=-torch.log1p(-result['source_choice'][:,1:,1:5].sum(2)).mean()
        torch.testing.assert_close(loss,expected)

    def test_unknown_invalid_not_wrong_and_stay_grad(self):
        result,info=fixture(); info['native']['labels'][:,1:]=-1
        result['source_valid'][:,:,2:]=False
        loss,_=late_choice_loss(result,info)
        self.assertAlmostEqual(float(loss),math.log(3/2),places=6)
        loss.backward()
        self.assertLess(float(result['source_stay_logit'].grad[:,1:].sum()),0)
        self.assertTrue(bool((result['source_edge_logits'].grad[:,:,2:]==0).all()))

    def test_all_wrong_unmatched_still_supervised(self):
        result,info=fixture(); info['selected']=torch.zeros_like(result['source_valid'])
        loss,counts=late_choice_loss(result,info)
        self.assertAlmostEqual(float(loss),math.log(9),places=6)
        self.assertEqual(counts['steps'][0]['images'][0]['all_valid_sources_wrong_positions'],1)

    def test_include_wrong_edges_outside_per_edge_interest(self):
        result,info=fixture(); info['rows']['valid'][:]=False
        info['rows']['valid'][:,:,0]=True
        loss,_=late_choice_loss(result,info)
        self.assertAlmostEqual(float(loss),math.log(9),places=6)

    def test_only_late_logits_directly_supervised(self):
        result,info=fixture(); loss,_=late_choice_loss(result,info); loss.backward()
        for key in ('source_stay_logit','source_edge_logits'):
            self.assertEqual(float(result[key].grad[:,0].abs().sum()),0)
            self.assertGreater(float(result[key].grad[:,1:].abs().sum()),0)

    def test_image_mean_not_edge_or_position_count(self):
        result,info=fixture(2,4)
        info['rows']['valid'][0,:,:,:,1:]=False
        info['native']['labels'][1,:,0,:]=1
        loss,_=late_choice_loss(result,info)
        self.assertAlmostEqual(float(loss),math.log(9)/2,places=6)

    def test_empty_images_skipped_and_empty_step_zero(self):
        result,info=fixture(2); info['rows']['valid'][1]=False
        loss,counts=late_choice_loss(result,info)
        self.assertAlmostEqual(float(loss),math.log(9),places=6)
        self.assertEqual(counts['steps'][0]['skipped_images'],1)
        info['rows']['valid'][:]=False
        loss,counts=late_choice_loss(result,info); loss.backward()
        self.assertEqual(float(loss),0); self.assertEqual(counts['steps'][1]['valid_images'],0)

    def test_extreme_logits_finite_loss_and_gradient(self):
        result,info=fixture()
        with torch.no_grad():
            result['source_stay_logit'].fill_(-1000); result['source_edge_logits'].fill_(1000)
        loss,_=late_choice_loss(result,info); loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss),1999)
        self.assertTrue(bool(torch.isfinite(result['source_edge_logits'].grad).all()))
        self.assertLess(float(result['source_stay_logit'].grad[:,1:].sum()),-.9)

    def test_live_forward_optimizer_and_strict_resume(self):
        from checks.test_v9_state_gate import model as fake_v9
        from networks.v9_source_selector import SourceSelectedRepair
        from networks.v9_source_relation_labels import RelationLabels
        from networks.v9_selector_transport_training import configure_joint,live_relation_loss
        from networks.reliable_directional_repair import dynamic_losses
        import run_v9_late_choice as runner
        torch.manual_seed(10)
        model=SourceSelectedRepair(fake_v9()); rates={key:1e-4 for key in ('visual','reliable_head','relations','proposal_head','accept_head','rates','gate_relation')}
        # Synthetic source must supply model-visible road evidence; the
        # production eligibility rule must not be relaxed to make a test pass.
        with torch.no_grad(): model.original.frozen.q.bias.fill_(2.)
        optimizer,_=configure_joint(model,rates,1e-4)
        image=torch.rand(2,3,64,64); gt=torch.zeros(2,1,64,64); gt[:,:,22:42,4:60]=1
        skeleton=torch.zeros_like(gt); skeleton[:,:,31,4:60]=1
        output=model(image,mode='source')
        relation,info=live_relation_loss(output,gt,skeleton,RelationLabels(),17)
        choice,counts=late_choice_loss(output,info)
        self.assertGreater(sum(r['wrong_bearing_positions'] for s in counts['steps'] for r in s['images']),0)
        base,_=dynamic_losses(output,gt,skeleton)
        total=base+.1*relation+.05*choice
        old=model.selector.stay[-1].bias.detach().clone(); total.backward()
        self.assertTrue(all(p.grad is None for p in model.original.frozen.parameters()))
        self.assertGreater(float(model.selector.stay[-1].bias.grad.abs().sum()),0)
        optimizer.step(); self.assertFalse(torch.equal(old,model.selector.stay[-1].bias))
        args=SimpleNamespace(config={'test':True},identity={'test':True})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pth'; runner.save(path,runner.pack(model,args,1,.3,optimizer))
            expected={name:value.clone() for name,value in model.state_dict().items()}
            with torch.no_grad(): model.selector.stay[-1].bias.add_(10)
            runner.restore(model,args,path,optimizer)
            for name,value in model.state_dict().items(): torch.testing.assert_close(value,expected[name],atol=0,rtol=0)
            from checks.test_v9_selector_transport import TinyDataset
            args.device=torch.device('cpu');args.seed=1234;args.labels=RelationLabels()
            args.lambda_choice=.05;args.source_losses={'lambda_cont':.2};args.output_dir=Path(folder)
            with patch.object(runner.joint,'dataset_train',return_value=TinyDataset()):
                runner.smoke(args,model,optimizer)


if __name__=='__main__': unittest.main()
