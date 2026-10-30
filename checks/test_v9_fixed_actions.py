"""Fixed physical actions, self replay, exact helpful matching and conservative shifts."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from checks.test_v9_state_gate import model as fake_v9
from networks.v9_transport_benefit import BenefitConfig
from networks.v9_fixed_action_audit import (capture_bank, replay_gate, matched_acceptance,
    compare_scores, acceptance_points, bootstrap_difference)
from diagnose_v9_fixed_actions import gather_image, save_tape, load_tapes, analyze, verify_continuations


class FixedActionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234); torch.set_num_threads(1)

    def test_self_replay_matches_actual_gate_and_keeps_states(self):
        model = fake_v9().eval()
        with torch.no_grad():
            model.gate_relation.delta[-1].weight.normal_(0., .02)
            model.state_gate.net[-1].weight.normal_(0., .02)
        bank = capture_bank(model, torch.randn(1,3,64,64))
        before = [value.clone() for value in bank['result']['states']]
        for mode in ('full_gate', 'residual_only'):
            logits = replay_gate(model, bank, mode)
            torch.testing.assert_close(logits, bank['result']['gate_logits'], atol=1e-6, rtol=1e-6)
            self.assertFalse(logits.requires_grad)
        for old, current in zip(before, bank['result']['states']): self.assertTrue(torch.equal(old, current))

    def test_training_mode_is_rejected_for_readonly_capture_and_replay(self):
        model=fake_v9()
        with self.assertRaises(ValueError): capture_bank(model,torch.randn(1,3,64,64))
        model.eval(); bank=capture_bank(model,torch.randn(1,3,64,64)); model.train()
        with self.assertRaises(ValueError): replay_gate(model,bank)

    def test_scoring_never_uses_scorers_source_proposal_or_rates(self):
        bank_model, scorer = fake_v9().eval(), fake_v9().eval()
        bank = capture_bank(bank_model, torch.randn(1,3,64,64))
        before = {mode: replay_gate(scorer, bank, mode).clone() for mode in ('full_gate','residual_only')}
        actions = (bank['result']['m'] - torch.stack(bank['result']['states'][:-1],1)) * bank['result']['transport_rate']
        with torch.no_grad():
            for name in ('reliable_head','proposal_head','accept_head','dynamics'):
                for value in getattr(scorer,name).parameters(): value.add_(100.)
        for mode in before: self.assertTrue(torch.equal(before[mode], replay_gate(scorer, bank, mode)))
        self.assertTrue(torch.equal(actions, (bank['result']['m'] - torch.stack(bank['result']['states'][:-1],1)) * bank['result']['transport_rate']))

    def test_full_gate_keeps_own_visual_coordinates_without_changing_action(self):
        model=fake_v9().eval(); bank=capture_bank(model,torch.randn(1,3,64,64))
        full=replay_gate(model,bank,'full_gate').clone(); residual=replay_gate(model,bank,'residual_only').clone()
        states=[value.clone() for value in bank['result']['states']]
        with torch.no_grad(): model.visual.project[0].weight.mul_(-1.)
        self.assertGreater(float((replay_gate(model,bank,'full_gate')-full).abs().max()),1e-6)
        self.assertTrue(torch.equal(residual,replay_gate(model,bank,'residual_only')))
        for old,current in zip(states,bank['result']['states']): self.assertTrue(torch.equal(old,current))

    def test_relation_change_only_affects_full_gate_scoring(self):
        model = fake_v9().eval(); bank = capture_bank(model, torch.randn(1,3,64,64))
        residual = replay_gate(model, bank, 'residual_only').clone()
        full = replay_gate(model, bank, 'full_gate').clone()
        with torch.no_grad(): model.relations.direction_head.bias[0].add_(2.)
        torch.testing.assert_close(replay_gate(model, bank, 'full_gate')-full, torch.full_like(full,2.), atol=2e-6, rtol=1e-6)
        self.assertTrue(torch.equal(residual, replay_gate(model, bank, 'residual_only')))

    def test_uniform_conservative_shift_has_identical_matched_curve(self):
        labels = np.array([1,1,1,0,0,0], bool)
        b = np.array([4.,2.,0.,3.,1.,-1.]); c = b-100.
        result = compare_scores(labels,b,c)
        self.assertLess(result['raw_at_G_0_5']['C_gate3']['harmful'],result['raw_at_G_0_5']['B_control']['harmful'])
        for row in result['matched_helpful']: self.assertAlmostEqual(row['harmful_C_minus_B'],0.)
        self.assertEqual(result['raw_at_G_0_5']['B_control']['auc'],result['raw_at_G_0_5']['C_gate3']['auc'])

    def test_improved_ranking_rejects_harmful_at_equal_helpful_rate(self):
        labels=np.array([1,1,0,0],bool)
        row=compare_scores(labels,np.array([.9,.5,.8,.4]),np.array([.9,.8,.4,.3]),(.8,))['matched_helpful'][0]
        self.assertAlmostEqual(row['B']['helpful_acceptance'],row['C']['helpful_acceptance'])
        self.assertLess(row['harmful_C_minus_B'],0.)

    def test_tie_rule_is_uniform_for_both_labels(self):
        result=matched_acceptance(np.zeros(10),np.array([1]*3+[0]*7,bool),.8)
        self.assertAlmostEqual(result['tie_fraction'],.8)
        self.assertAlmostEqual(result['helpful_acceptance'],.8)
        self.assertAlmostEqual(result['harmful_acceptance'],.8)
        self.assertEqual(result['harmful_strict'],0.); self.assertEqual(result['harmful_inclusive'],1.)

    def test_fast_curve_equals_single_point_with_ties_and_imbalance(self):
        rng=np.random.default_rng(12); labels=rng.random(100)<.25; scores=rng.integers(-5,5,100)
        targets=np.linspace(0.,1.,101)
        for target,point in zip(targets,acceptance_points(scores,labels,targets)):
            expected=matched_acceptance(scores,labels,target)
            self.assertAlmostEqual(point['harmful_acceptance'],expected['harmful_acceptance'])

    def test_paired_image_bootstrap_preserves_uniform_shift_equivalence(self):
        labels=np.array([1,0,1,0]*4,bool); b=np.arange(16,dtype=float); ids=np.repeat(np.arange(4),4)
        result=bootstrap_difference(labels,b,b-10.,ids,(.8,.9),20,1234)
        for row in result: self.assertEqual(row['delta_harmful_C_minus_B_ci95'],[0.,0.])

    def test_one_class_and_no_actions_do_not_claim_success(self):
        self.assertFalse(matched_acceptance(np.ones(4),np.ones(4,bool),.8)['available'])
        result=compare_scores(np.array([],bool),np.array([]),np.array([]))
        self.assertEqual(result['actions'],0); self.assertIsNone(result['raw_at_G_0_5']['B_control']['auc']['roc_auc'])

    def test_only_complete_same_budget_same_data_continuations_are_accepted(self):
        import json
        import hashlib
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.fixed_actions_test_') as temp:
            root=Path(temp); source=root/'source.pth'; source.write_bytes(b'source')
            with self.assertRaises(FileNotFoundError): verify_continuations(root,source)
            for name in ('control','gate'):
                directory=root/(name+'_3step'); directory.mkdir()
                torch.save({'benefit_config':{'variant':name,'steps':3,'epochs':1,
                    'original_sha256':hashlib.sha256(b'source').hexdigest()}},directory/'best.pth')
                (directory/'training_complete.json').write_text(json.dumps({'epochs':1}))
                (directory/'epochs.jsonl').write_text(json.dumps({'epoch':1,'case_order_sha256':'same',
                    'augmented_images_sha256':'same'})+'\n')
            paths,config=verify_continuations(root,source)
            self.assertEqual(len(paths),2); self.assertEqual(config['epochs'],1)
            (root/'gate_3step/epochs.jsonl').write_text(json.dumps({'epoch':1,'case_order_sha256':'same',
                    'augmented_images_sha256':'different'})+'\n')
            with self.assertRaises(ValueError): verify_continuations(root,source)

    def test_full_tile_shared_tape_cache_and_report(self):
        models={name:fake_v9().eval() for name in ('B_control','C_gate3')}
        models['C_gate3'].load_state_dict(models['B_control'].state_dict())
        with torch.no_grad(): models['C_gate3'].state_gate.net[-1].bias.sub_(1.)
        mask=torch.zeros(1,512,512); mask[:,:,240:272]=1.
        sample={'case_name':'synthetic','image':torch.randn(3,512,512),'mask':mask}
        from unittest.mock import patch
        with patch('builtins.print'):
            rows=gather_image(sample,models,['B_control','C_gate3'],BenefitConfig(min_action=1e-12,gain_epsilon=1e-12),0,torch.device('cpu'),1234)
        for bank, values in rows.items():
            self.assertGreater(len(values['label']),0)
            for mode in ('full_gate','residual_only'):
                np.testing.assert_allclose(values['C_gate3_'+mode]-values['B_control_'+mode],-1.,atol=2e-6)
            # Every physical location/step occurs only once in this image bank.
            positions=np.stack([values['native_y'],values['native_x'],values['step']],1)
            self.assertEqual(len(np.unique(positions,axis=0)),len(positions))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.fixed_actions_test_') as temp:
            output=Path(temp); manifest={'banks':list(rows),'cases':['synthetic']}
            for name, values in rows.items():
                file=output/'actions'/name/'00.npz'; file.parent.mkdir(parents=True)
                save_tape(file,values,{'case':'synthetic','input_sha256':'synthetic'})
            loaded=load_tapes(output,manifest)
            with patch('builtins.print'): report=analyze(output,manifest,loaded,(.8,.9),0,1234)
            self.assertTrue((output/'REPORT.md').is_file()); self.assertTrue((output/'matched_helpful.csv').is_file())
            for bank in report['banks'].values():
                for mode in bank.values():
                    for row in mode['natural_all']['matched_helpful']:
                        if row['harmful_C_minus_B'] is not None: self.assertAlmostEqual(row['harmful_C_minus_B'],0.,places=5)
            file=output/'actions/B_control/00.npz'; file.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): load_tapes(output,manifest)


if __name__=='__main__': unittest.main()
