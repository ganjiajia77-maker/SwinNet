"""Mechanism checks for the GT-only oracle and prediction-first path audits."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from checks.test_v9_state_gate import model
from diagnostics.v92_oracle_gate import oracle_choice,oracle_dynamics,infer as oracle_infer
from diagnostics.v92_reachability import ordered_simple_path,reach_three_steps,longest_false_run,inspect_gaps
from diagnostics.v92_connection_candidates import generate,label,classifier_audit,split_name,cap_pairs_by_length
from networks.reliable_directional_repair import DIRECTIONS
import diagnostics.v92_train_audit as train_audit


class DiagnosticsTest(unittest.TestCase):
    def setUp(self): torch.manual_seed(2); torch.set_num_threads(1)

    def test_oracle_signed_action_and_uncertainty(self):
        state=torch.full((1,1,1,5),.5); zero=torch.zeros_like(state)
        action=torch.tensor([[[[.1,.1,-.1,-.1,.1]]]])
        target=torch.tensor([[[[1.,0.,1.,0.,.5]]]])
        gate=torch.full_like(state,.37)
        chosen,stats=oracle_choice(state,zero,zero,action,target,gate)
        torch.testing.assert_close(chosen,torch.tensor([[[[1.,0.,0.,1.,.37]]]]))
        self.assertEqual(stats['allowed'],2); self.assertEqual(stats['blocked'],2); self.assertEqual(stats['uncertain'],1)

    def test_ambiguous_oracle_reproduces_original_dynamics_and_preserves_weights(self):
        net=model().eval(); image=torch.randn(1,3,64,64); target=torch.full((1,1,64,64),.5)
        before={k:v.clone() for k,v in net.state_dict().items()}
        counts=[dict(allowed=0,blocked=0,uncertain=0,positive_action=0,negative_action=0) for _ in range(3)]
        with torch.no_grad():
            frozen=net.frozen(image,capture_diagnostics=True)
            normal=net(None,frozen_override=frozen,capture_diagnostics=True)
            original_forward=net.dynamics.forward
            with oracle_dynamics(net.dynamics,target,counts):
                out=net(None,frozen_override=frozen,capture_diagnostics=True)
            self.assertEqual(net.dynamics.forward,original_forward)
        for key in ('u1','u2','u3','p1','transport_delta','source_available'):
            torch.testing.assert_close(out[key],normal[key],atol=0,rtol=0)
        self.assertTrue(all(r['uncertain']>0 and r['allowed']==0 and r['blocked']==0 for r in counts))
        for key,value in net.state_dict().items(): torch.testing.assert_close(value,before[key],atol=0,rtol=0)

    def test_order_and_reach_obey_direction_and_three_step_limit(self):
        comp=np.zeros((12,12),bool); comp[5,3:8]=True
        path=ordered_simple_path(comp); self.assertEqual(len(path),5)
        self.assertEqual(longest_false_run([True,False,False,True]),2)
        anchor=np.zeros((12,12),bool); anchor[5,2]=True
        cand=np.zeros((3,12,12),bool); cand[:,5,3:8]=True
        support=np.ones((3,12,12)); weights=np.zeros((8,12,12))
        weights[DIRECTIONS.index((0,-1))]=1
        reached=reach_three_steps(path,anchor,cand,support,weights,.01,1e-6)
        self.assertTrue(reached[3][5,5]); self.assertFalse(reached[3][5,6])
        weights[:]=0; weights[DIRECTIONS.index((0,1))]=1
        self.assertFalse(reach_three_steps(path,anchor,cand,support,weights,.01,1e-6)[3][5,3])

    def test_oracle_runs_full_prediction_tile_with_original_and_gt_intervention(self):
        net=model().eval(); sample={'image':torch.zeros(3,512,512),
            'mask':torch.zeros(1,512,512),'case_name':'sample'}
        counts=[dict(allowed=0,blocked=0,uncertain=0,positive_action=0,negative_action=0) for _ in range(3)]
        cases=oracle_infer(net,[sample],torch.device('cpu'),counts,1e-5,1e-7)
        self.assertEqual(cases[0]['O_oracle'].shape,(512,512))
        self.assertTrue(np.isfinite(cases[0]['O_oracle']).all())
        self.assertTrue(all(sum(row[k] for k in ('allowed','blocked','uncertain'))>0 for row in counts))

    def test_real_module_outputs_can_be_audited_as_gap_paths(self):
        net=model().eval(); image=torch.randn(1,3,64,64)
        gt=np.zeros((64,64),bool); gt[31,5:57]=True
        with torch.no_grad(): result=net(image,capture_diagnostics=True)
        rows,points=inspect_gaps(result,gt,gt,'case',.45,.68)
        self.assertEqual(len(rows),1); self.assertEqual(len(points),1)
        self.assertTrue(rows[0]['simple_path'])
        self.assertTrue(0<=rows[0]['positive_candidate_fraction']<=1)
        self.assertIn('path_reachable_fraction_step3',rows[0])

    def test_candidates_generated_without_gt_and_gt_changes_labels(self):
        p0=np.zeros((32,32),np.float32); p0[16,3:11]=.9; p0[16,18:27]=.9
        r=np.ones_like(p0)*.4; w=np.ones((8,32,32),np.float32)/8; q=np.zeros_like(p0)
        predictions,meta=generate(p0,r,w,q,.5,max_endpoints=32,max_pairs=100)
        self.assertGreater(meta['pairs_used'],0)
        gt=np.zeros_like(p0); gt[16,3:27]=1
        gap={1:{(16,x) for x in range(11,18)}}
        pos,pair_hits=label(copy.deepcopy(predictions),gt,p0,.45,gap)
        separate=np.zeros_like(p0); separate[16,3:11]=1; separate[16,18:27]=1
        neg,_=label(copy.deepcopy(predictions),separate,p0,.45,{})
        self.assertTrue(any(c['gt_label']=='positive' and c['kind']=='pair' for c in pos))
        self.assertTrue(any(c['gt_label']=='negative' and c['kind']=='pair' for c in neg))
        self.assertGreater(pair_hits[1],0)
        _,extension_hits=label([copy.deepcopy(c) for c in predictions if c['kind']=='extension'],gt,p0,.45,gap)
        self.assertEqual(extension_hits[1],0.)

    def test_pair_cap_preserves_length_strata(self):
        proposals=[('pair',i,j,None,None,length)
                   for i,(j,length) in enumerate(((0,3.),(1,6.),(2,12.),(3,24.),(4,5.),(5,10.)))]
        selected=cap_pairs_by_length(proposals,4)
        self.assertEqual(len(selected),4)
        bins={int(np.digitize(row[-1],[4,8,16])) for row in selected}
        self.assertEqual(bins,{0,1,2,3})

    def test_classifier_splits_by_image_and_reports_both_pr_aucs(self):
        train=next('train'+str(i) for i in range(100) if split_name('train'+str(i))=='train')
        held=next('held'+str(i) for i in range(100) if split_name('held'+str(i))=='heldout')
        rows=[]
        for name in (train,held):
            for outcome in ('positive','negative'):
                for _ in range(5):
                    rows.append({'case':name,'gt_label':outcome,'features':[1 if outcome=='positive' else 0]*10,
                                 'threshold':.45,'length_work':6,'kind':'pair'})
        report=classifier_audit(rows,population={'heldout_positive':10,'heldout_negative':90})
        self.assertTrue(report['available']); self.assertEqual(report['split']['train_images'],1)
        self.assertEqual(report['split']['heldout_images'],1)
        self.assertAlmostEqual(report['natural_holdout']['weighted_positive_fraction'],.1)
        self.assertAlmostEqual(report['matched_holdout']['weighted_positive_fraction'],.5)
        self.assertGreater(report['natural_holdout']['pr_auc'],.9)
        unavailable=classifier_audit([r for r in rows if r['case']==train],population={'train_positive':5})
        self.assertFalse(unavailable['available']); self.assertEqual(unavailable['population_counts'],{'train_positive':5})

    def test_training_audit_writes_readable_outputs_without_candidates(self):
        with tempfile.TemporaryDirectory(prefix='.v92_test_',dir=ROOT) as temp:
            base=Path(temp); run=base/'run'; (run/'stage_D').mkdir(parents=True)
            ckpt=run/'stage_D'/'best.pth'; identity={'q3':{'path':str(base/'q3.pth')}}
            torch.save({'identity':identity,'training_config':{'root_path':str(base),'reference_threshold':.45}},ckpt)
            (run/'direct_val_selection.json').write_text(json.dumps({'identity':identity,'checkpoint':str(ckpt),
                'selected_on':'val','threshold':.68}),encoding='utf-8')
            road=torch.zeros(1,512,512); road[0,250,100:300]=1
            sample={'image':torch.zeros(3,512,512),'mask':road,'skeleton':road,'case_name':'sample'}
            net=model().eval(); out=base/'result'
            argv=['v92_train_audit.py','--run_dir',str(run),'--root_path',str(base),
                  '--output_dir',str(out),'--device','cpu']
            fake_q3=SimpleNamespace(full_dataset=lambda root,split:[sample])
            with patch.object(sys,'argv',argv),patch.dict(sys.modules,{'run_q3_r3_mass':fake_q3}), \
                 patch.object(train_audit.v9,'load_initial',return_value=(net,{},identity)), \
                 patch.object(train_audit.v9,'restore_checkpoint'),patch.object(train_audit.v9,'digest',return_value='fixed'):
                train_audit.main()
            report=json.loads((out/'train_path_audit.json').read_text(encoding='utf-8'))
            self.assertEqual(report['images'],1); self.assertEqual(report['tiles'],1)
            self.assertEqual(report['split'],'train')
            self.assertTrue((out/'train_gap_reachability.csv').is_file())

if __name__=='__main__': unittest.main()
