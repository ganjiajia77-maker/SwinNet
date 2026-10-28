"""Behavioral checks with the real V9 downstream and a synthetic frozen source."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from checks.test_v9_state_gate import model as base_model
from networks.connection_paths import PathConfig,generate_paths,label_paths,soft_corridor,path_length,_search
from networks.path_connection_repair import PathConnectionRepair,candidate_features,replay_with_suppression
from networks.path_connection_losses import path_losses,matching_pairs
from path_connection_protocol import binary_metrics,atomic_npz,matched_indices,infer_cases
import run_path_connection_mass as runner


def candidate(shape=(16,16),a=(8,2),b=(8,13)):
    points=np.asarray([(8,x) for x in range(a[1],b[1]+1)],np.int32)
    return {'endpoint_a':np.array(a),'endpoint_b':np.array(b),'path_points':points,
        'path_length_work':path_length(points),'path_length_native':path_length(points,(4.,4.)),
        'scale_y':4.,'scale_x':4.,'native_shape':(64,64),'endpoint_direction':(np.array((0.,1.)),np.array((0.,-1.))),
        'source_confidence':(.8,.8),'endpoint_p0':(.9,.9),'source_component_area':(30,30),
        'kind':'pair','threshold':.45,'alternative':0,'width_work':1.5,'width_native':6.}


class PathTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(3); torch.set_num_threads(1)

    def wrapper(self):
        return PathConnectionRepair(base_model(),PathConfig(max_endpoints=8,max_pairs=4,max_paths=8,paths_per_pair=1))

    def test_lambda_zero_and_empty_paths_exactly_preserve_v9(self):
        net=self.wrapper().eval(); image=torch.randn(1,3,64,64)
        with torch.no_grad():
            base=net.base(image,capture_diagnostics=True)
            result=net(base_override=base,paths_override=[[]])
        self.assertTrue(torch.equal(result['z1'],base['z1']))
        net.suppress=True; net.long_ramp=0.
        with torch.no_grad(): result=net(base_override=base,paths_override=[[candidate()]])
        self.assertTrue(torch.equal(result['z1'],base['z1']))
        net.suppress=False
        net.long_ramp=1.
        with torch.no_grad(): result=net(base_override=base,paths_override=[[]])
        self.assertTrue(torch.equal(result['z1'],base['z1']))
        self.assertEqual(result['path_features'][0].shape,(0,13*32+23))

    def test_symmetric_endpoint_and_path_reversal(self):
        net=self.wrapper().eval(); image=torch.randn(1,3,64,64)
        with torch.no_grad(): result=net.base(image,capture_diagnostics=True)
        path=candidate(); reverse=copy.deepcopy(path)
        reverse['endpoint_a'],reverse['endpoint_b']=path['endpoint_b'],path['endpoint_a']
        reverse['path_points']=path['path_points'][::-1].copy()
        reverse['endpoint_direction']=path['endpoint_direction'][::-1]
        context=net.base.gate_relation.context(result['gate_relation_diagnostics']['center'])
        a=candidate_features(result,context,[[path]])[0]
        b=candidate_features(result,context,[[reverse]])[0]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(net.path_head(a),net.path_head(b),rtol=1e-5,atol=1e-6)

    def test_nonlocal_action_is_local_bounded_and_differentiable(self):
        net=self.wrapper(); net.configure('C'); net.long_ramp=1.
        with torch.no_grad(): base=net.base(torch.randn(1,3,64,64),capture_diagnostics=True)
        base['u0']=torch.full_like(base['u0'],.1); base['u3']=torch.full_like(base['u3'],.1)
        base['r']=torch.full_like(base['r'],.8)
        base['u0'][0,0,8,2]=.9; base['u0'][0,0,8,13]=.9
        out=net(base_override=base,paths_override=[[candidate()]])
        delta=out['long_delta']; self.assertGreater(float(delta.max()),0.)
        self.assertEqual(float(delta[0,0,0,0]),0.)
        self.assertTrue(bool(((out['u_final']>=0)&(out['u_final']<=1)).all()))
        out['p1'].mean().backward()
        self.assertGreater(runner.v9.grad_norm(net.path_head),0.)
        self.assertGreater(float(net.long_gain.grad.abs()),0.)
        self.assertTrue(all(p.grad is None for p in net.base.parameters()))
        # A near-zero connection score removes the effect, without thresholded road painting.
        with torch.no_grad(): net.path_head.net[-1].weight.zero_(); net.path_head.net[-1].bias.fill_(-25.)
        low=net(base_override=base,paths_override=[[candidate()]])
        self.assertLess(float(low['long_delta'].max()),1e-9)
        base['r'].zero_()
        without_source=net(base_override=base,paths_override=[[candidate()]])
        self.assertEqual(float(without_source['long_delta'].max()),0.)

    def test_zero_suppression_replays_original_dynamics(self):
        net=self.wrapper().eval()
        with torch.no_grad():
            result=net.base(torch.randn(1,3,64,64),capture_diagnostics=True)
            u,_,_=replay_with_suppression(net.base,result,torch.ones_like(result['u0']),0.)
        torch.testing.assert_close(u,result['u3'],rtol=0,atol=0)

    def test_curved_search_uses_predictions_and_preserves_native_units(self):
        h=w=32; p=np.zeros((h,w),np.float32); q=np.zeros_like(p)
        semantic=np.ones((2,h,w),np.float32)*-1.
        curve=[]
        for x in range(4,27):
            y=16+int(round(4.*np.sin((x-4)*np.pi/22)))
            q[y-1:y+2,x]=1.; semantic[:,y-1:y+2,x]=1.; curve.append((y,x))
        weights=np.ones((8,h,w),np.float32)/8
        path=_search((16,4),(16,26),p,q,weights,semantic,PathConfig())
        self.assertIsNotNone(path); self.assertGreater(int(path[:,0].max()),17)
        self.assertAlmostEqual(path_length(path,(4.,4.)),4*path_length(path))
        support=soft_corridor(path,(h,w),2.)
        self.assertEqual(float(support[0,0]),0.); self.assertAlmostEqual(float(support[16,4]),1.)

    def test_proposals_precede_gt_and_geometry_labels_ignore_uncertainty(self):
        p=np.zeros((32,32),np.float32); p[16,3:11]=.9; p[16,19:28]=.9
        r=p.copy(); q=np.ones_like(p); w=np.ones((8,32,32),np.float32)/8
        cfg=PathConfig(max_endpoints=8,max_pairs=8,max_paths=16,paths_per_pair=1)
        paths,_=generate_paths(p,r,w,q,None,(128,128),cfg)
        self.assertGreater(len(paths),0)
        gt=np.zeros((128,128),bool); gt[62:70,12:112]=True
        sk=np.zeros_like(gt); sk[65,12:112]=True
        p_native=np.repeat(np.repeat(p,4,0),4,1)
        labels,gaps=label_paths(paths,gt,sk,p_native,.45,cfg)
        self.assertTrue(any(row['label']==1 for row in labels))
        self.assertTrue(any(gap['full_coverage'] for gap in gaps))
        empty,other=label_paths(paths,np.zeros_like(gt),np.zeros_like(sk),p_native,.45,cfg)
        self.assertTrue(all(row['label']==0 for row in empty)); self.assertEqual(other,[])
        unknown=[{'label':-1} for _ in paths]
        self.assertEqual(matching_pairs(paths,unknown),[])

    def test_path_supervision_removes_overlapping_continuity_pixels(self):
        net=self.wrapper(); net.configure('D')
        with torch.no_grad(): result=net.base(torch.randn(1,3,64,64),capture_diagnostics=True)
        result['p0']=torch.zeros_like(result['p0']); result['z1']=result['z1'].detach().requires_grad_(); result['p1']=result['z1'].sigmoid()
        result['paths']=[[candidate()]]; result['path_logits']=[torch.tensor([1.],requires_grad=True)]
        mask=torch.zeros(1,1,64,64); mask[:,:,30:37,8:56]=1.
        skel=torch.zeros_like(mask); skel[:,:,33,8:56]=1.
        source={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45}
        loss,terms=path_losses(result,mask,skel,[[{'label':1}]],source,'D')
        self.assertGreater(terms['removed_duplicate_cont_pixels'],0)
        self.assertEqual(terms['removed_duplicate_cont_pixels'],terms['path_supervised_pixels'])
        self.assertTrue(torch.isfinite(loss)); loss.backward(); self.assertGreater(float(result['z1'].grad.abs().sum()),0.)

    def test_classification_metrics_group_ties_and_match_ignores_unknown(self):
        report=binary_metrics([0,0,1,1],[.5,.5,.5,.5])
        self.assertAlmostEqual(report['roc_auc'],.5); self.assertAlmostEqual(report['pr_auc'],.5)
        ids=matched_indices(np.array([0,1,-1]),np.array(['a','a','a']),1234)
        self.assertEqual(set(ids),{0,1})

    def test_stage_freezing_and_checkpoint_strict_roundtrip(self):
        net=self.wrapper(); net.configure('F')
        self.assertTrue(any(p.requires_grad for p in net.base.relations.parameters()))
        self.assertFalse(any(p.requires_grad for p in net.base.proposal_head.parameters()))
        self.assertFalse(any(p.requires_grad for p in net.base.frozen.parameters()))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.path_test_') as temp:
            args=SimpleNamespace(device=torch.device('cpu'),epochs=1)
            path=Path(temp)/'test.pth'; identity={'source':'synthetic-only'}
            runner.save(args,net,'F',1,.5,None,identity,path)
            other=self.wrapper(); runner.restore(other,path,identity)
            for key,value in net.state_dict().items():
                if key.startswith('base.frozen.'): continue
                torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
            with self.assertRaises(ValueError): runner.restore(other,path,{'source':'wrong'})

    def test_joint_gradients_reach_reused_features_and_original_transport(self):
        net=self.wrapper(); net.configure('F'); net.long_ramp=1.
        out=net(torch.randn(1,3,64,64),paths_override=[[candidate()]])
        loss=out['path_logits'][0].square().mean()+out['p1'].mean()
        loss.backward()
        for module in (net.path_head,net.base.gate_relation.highres_project,
                       net.base.gate_relation.decoder_project,net.base.reliable_head,
                       net.base.relations,net.base.dynamics.rates):
            self.assertGreater(runner.v9.grad_norm(module),0.)
        self.assertTrue(all(p.grad is None for p in net.base.frozen.parameters()))
        self.assertTrue(all(p.grad is None for p in net.base.proposal_head.parameters()))

    def test_cached_classifier_training_and_restart_skip(self):
        net=self.wrapper(); width=net.path_head.net[0].normalized_shape[0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.path_test_') as temp:
            output=Path(temp); identity={'source':'synthetic'}
            for split,partition in (('train','train'),('train','heldout'),('val','train')):
                features=np.zeros((8,width),np.float32); features[4:,:8]=1.
                labels=np.r_[np.zeros(4),np.ones(4)].astype(np.int8)
                atomic_npz(output/f'candidate_cache/{split}/{partition}.npz',features=features,labels=labels,
                    strata=np.array(['matched']*8),subtypes=np.array(['negative']*4+['positive']*4),
                    lengths_native=np.ones(8)*40,source_confidence=np.ones((8,2))*.8,
                    endpoint_p0=np.ones((8,2))*.9,distance_work=np.ones(8)*10,
                    baseline_gate_scores=np.ones(8)*.5,
                    meta=np.asarray(json.dumps({'case':partition,'partition':partition})))
            args=SimpleNamespace(output_dir=output,device=torch.device('cpu'),lr=.001,seed=1234,
                epochs=1,val_interval=1)
            before={key:value.clone() for key,value in net.path_head.state_dict().items()}
            with patch('builtins.print'): runner.train_phase(args,net,{},identity,'B',None)
            self.assertTrue(any(not torch.equal(before[key],value) for key,value in net.path_head.state_dict().items()))
            log=output/'B_relation/epochs.jsonl'; self.assertEqual(len(log.read_text().splitlines()),1)
            with patch('builtins.print'): runner.train_phase(args,net,{},identity,'B',None)
            self.assertEqual(len(log.read_text().splitlines()),1)

    def test_probability_and_feature_cache_resume_without_repeated_forward(self):
        net=self.wrapper().eval()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.path_test_') as temp:
            sample={'image':torch.zeros(3,512,512),'mask':torch.zeros(1,512,512),
                    'skeleton':torch.zeros(1,512,512),'case_name':'synthetic'}
            kwargs={'output':Path(temp)/'prob','cache_features':Path(temp)/'feat','variant':'A_original'}
            first,_=infer_cases(net,[sample],torch.device('cpu'),**kwargs)
            with patch.object(net,'forward',side_effect=AssertionError('Completed cache should avoid inference')):
                second,_=infer_cases(net,[sample],torch.device('cpu'),**kwargs)
            np.testing.assert_array_equal(first[0]['A_original'],second[0]['A_original'])

    def test_automated_pipeline_blocks_repair_when_long_candidates_are_missing(self):
        net=self.wrapper().eval()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.path_test_') as temp:
            out=Path(temp); (out/'smoke.json').write_text('{}')
            args=SimpleNamespace(output_dir=out,device=torch.device('cpu'),reference=.45,
                expected_gaps=206,reference_set=None,min_long_gaps=8,min_long_coverage=.05,root_path=out)
            fake_dataset=SimpleNamespace(__len__=lambda unused:14)
            # A list is sufficient: no GT or actual benchmark results are fabricated here.
            with patch.dict('sys.modules',{'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda root,split:[None]*14)}), \
                 patch.object(runner,'audit_train',return_value={'by_length':{}}), \
                 patch.object(runner,'infer_cases',return_value=([],{})), \
                 patch.object(runner,'compare_cases',return_value={}), \
                 patch.object(runner,'train_phase') as train:
                status=runner.all_stages(args,net,{},{} )
            self.assertEqual(status,2); train.assert_not_called()
            self.assertEqual(json.loads((out/'pipeline_state.json').read_text())['status'],'blocked_candidate_reachability')

    def test_real_batch_smoke_restores_all_weights_after_joint_probe(self):
        net=self.wrapper(); initial={key:value.clone() for key,value in net.state_dict().items()}
        image=torch.randn(3,64,64); mask=torch.zeros(1,64,64); mask[:,30:36,5:59]=1.
        skeleton=torch.zeros_like(mask); skeleton[:,33,5:59]=1.
        samples=[{'image':image,'mask':mask,'skeleton':skeleton,'case_name':f'synthetic{i}'} for i in range(2)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.path_test_') as temp:
            args=SimpleNamespace(output_dir=Path(temp),device=torch.device('cpu'),batch_size=2,lr=.001,
                reference=.45,seed=1234,lambda_relation=1.,lambda_path=.2,lambda_neg=.1)
            source={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45}
            with patch.object(runner,'train_dataset',return_value=samples): report=runner.smoke(args,net,source,{'synthetic':True})
            self.assertTrue(report['passed']); self.assertEqual(set(report['phases']),{'B','C','D','F'})
            self.assertGreater(report['phases']['F']['gradient_norms']['W'],0.)
            for key,value in net.state_dict().items(): torch.testing.assert_close(value,initial[key],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
