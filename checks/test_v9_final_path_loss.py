"""Behavioral tests for training-only path labels and live P1 supervision."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from checks.test_v9_state_gate import model as base_model
from checks.test_path_connection import candidate
from networks.connection_paths import PathConfig
from networks.v9_final_path_supervision import (soft_min_score,connection_loss,
    weighted_connection,predicted_candidates,_matched_pairs,_junction_mask)
import run_v9_final_path_loss as runner
from path_connection_protocol import compare_cases
from evaluation.road_comparison.road_metrics import DEFAULT_PROTOCOL


def item(label,row=5,kind='pair'):
    points=np.asarray([(row,x) for x in range(4,12)],np.int32)
    return {'label':label,'subtype':'positive' if label else 'negative','kind':kind,
        'points':points,'background':points[2:-2] if label==0 else np.empty((0,2),np.int32),
        'length_native':32.,'length_work':8.,'source':.7,'endpoint_p0':.8,
        'separation_native':28.,'path_q':.4,'gt_missing_overlap':8 if label else 0}


class FinalPathTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(1234); torch.set_num_threads(1)

    def test_softmin_prioritizes_weak_link_and_constant_path(self):
        values=torch.tensor([.92,.86,.18,.83,.91],requires_grad=True)
        score=soft_min_score(values,.1); (-score.log()).backward()
        self.assertLess(float(score),float(values.detach().mean()))
        self.assertGreater(abs(float(values.grad[2])),abs(float(values.grad[0])))
        torch.testing.assert_close(soft_min_score(torch.ones(7)*.7,.1),torch.tensor(.7))

    def test_pos_neg_background_and_rank_gradients_have_expected_sign(self):
        p=torch.full((1,1,16,16),.7,requires_grad=True)
        entries=[[item(1,5),item(0,10)]]; pairs=[[(0,1)]]
        terms,counts=connection_loss(p,entries,pairs,tau=.1,margin=.1,temperature=.1)
        self.assertEqual(counts['matched_pairs'],1)
        grad_pos=torch.autograd.grad(terms['pos'],p,retain_graph=True)[0]
        grad_neg=torch.autograd.grad(terms['neg'],p,retain_graph=True)[0]
        grad_bg=torch.autograd.grad(terms['bg'],p,retain_graph=True)[0]
        grad_rank=torch.autograd.grad(terms['rank'],p)[0]
        self.assertLess(float(grad_pos[0,0,5,7]),0.)
        self.assertGreater(float(grad_neg[0,0,10,7]),0.)
        self.assertGreater(float(grad_bg[0,0,10,7]),0.)
        self.assertEqual(float(grad_bg[0,0,10,4]),0.)
        self.assertLess(float(grad_rank[0,0,5,7]),0.)
        self.assertGreater(float(grad_rank[0,0,10,7]),0.)

    def test_no_labeled_candidates_produce_differentiable_zero(self):
        p=torch.rand((1,1,16,16),requires_grad=True)
        terms,counts=connection_loss(p,[[]],[[]])
        loss=weighted_connection(terms,{'pos':.1,'neg':.1,'bg':.05,'rank':.1})
        self.assertEqual(float(loss),0.); self.assertEqual(counts['matched_pairs'],0)
        loss.backward(); self.assertEqual(float(p.grad.abs().sum()),0.)

    def test_matching_rejects_dissimilar_and_never_reuses_negative(self):
        a=item(1); b=item(0); c=item(0); c['source']=.01
        matches=_matched_pairs([a,b,c],10,1234)
        self.assertEqual(matches,[(0,1)])

    def test_prediction_candidate_labels_and_gt_aligned_positive_are_separate(self):
        net=base_model().eval(); image=torch.randn(1,3,64,64)
        with torch.no_grad(): result=net(image,capture_diagnostics=True)
        gt=np.zeros((64,64),np.uint8); sk=np.zeros_like(gt)
        gt[21:24,8:58]=1; gt[45:48,8:58]=1
        sk[22,8:58]=1; sk[46,8:58]=1
        p0=np.zeros((64,64),np.float32); p0[19:26,8:15]=.9; p0[19:26,50:58]=.9
        p0[43:50,8:58]=.9
        result['p0']=torch.from_numpy(p0)[None,None]
        horizontal=candidate(a=(5,2),b=(5,13)); horizontal['path_points']=np.asarray([(5,x) for x in range(2,14)],np.int32)
        vertical=copy.deepcopy(horizontal); vertical['endpoint_a']=np.array((5,8)); vertical['endpoint_b']=np.array((11,8))
        vertical['path_points']=np.asarray([(y,8) for y in range(5,12)],np.int32)
        vertical['path_length_work']=6.; vertical['path_length_native']=24.
        with patch('networks.v9_final_path_supervision.generate_paths',return_value=([horizontal,vertical],{'paths':2})):
            paths,_,stats,_=predicted_candidates(net,result,torch.from_numpy(gt)[None,None],torch.from_numpy(sk)[None,None])
        positives=[path for path in paths[0] if path['label']==1]
        negatives=[path for path in paths[0] if path['label']==0]
        self.assertEqual(len(positives),1); self.assertEqual(len(negatives),1)
        self.assertTrue(sk[positives[0]['points'][:,0],positives[0]['points'][:,1]].all())
        self.assertTrue((gt[negatives[0]['background'][:,0],negatives[0]['background'][:,1]]==0).all())
        self.assertEqual(stats['selected_positive'],1)

    def test_connection_gradient_reaches_existing_v9_dynamics(self):
        net=base_model(); runner.v9.configure(net,'D'); net.train()
        result=net(torch.randn(1,3,64,64),capture_diagnostics=True)
        terms,_=connection_loss(result['p1'],[[item(1,25),item(0,35)]],[[(0,1)]])
        loss=weighted_connection(terms,{'pos':.1,'neg':.1,'bg':.05,'rank':.1})
        gradients=runner.measure_gradients(loss,net)
        for name in ('reliable_head','relations','gate_relation','state_gate','rates','proposal_head','accept_head'):
            self.assertGreater(gradients[name],0.,name)
        loss.backward(); self.assertTrue(all(parameter.grad is None for parameter in net.frozen.parameters()))

    def test_positive_supervision_follows_a_curved_gt_route(self):
        net=base_model().eval()
        with torch.no_grad(): result=net(torch.randn(1,3,64,64),capture_diagnostics=True)
        path=candidate(a=(5,2),b=(5,13))
        work=np.asarray([(5,2),(5,3),(6,4),(6,5),(7,6),(7,7),(7,8),
                         (6,9),(6,10),(5,11),(5,12),(5,13)],np.int32)
        path['path_points']=work
        path['path_length_work']=float(np.linalg.norm(np.diff(work,axis=0),axis=1).sum())
        path['path_length_native']=path['path_length_work']*4.
        native=np.rint((work+.5)*4.-.5).astype(np.int32)
        skeleton=np.zeros((64,64),np.uint8)
        cv2.polylines(skeleton,[native[:,::-1].reshape(-1,1,2)],False,1,1)
        mask=cv2.dilate(skeleton,np.ones((7,7),np.uint8))
        p0=np.zeros((64,64),np.float32)
        for y,x in (native[0],native[-1]): p0[max(0,y-2):y+3,max(0,x-2):x+3]=.9
        result['p0']=torch.from_numpy(p0)[None,None]
        with patch('networks.v9_final_path_supervision.generate_paths',return_value=([path],{'paths':1})) as generator:
            paths,_,_,_=predicted_candidates(net,result,torch.from_numpy(mask)[None,None],torch.from_numpy(skeleton)[None,None])
        self.assertEqual(len(paths[0]),1)
        self.assertEqual(paths[0][0]['label'],1)
        points=paths[0][0]['points']
        self.assertTrue(skeleton[points[:,0],points[:,1]].all())
        self.assertGreater(int(points[:,0].max()),int(points[0,0])+3)
        # The prediction-only generator accepts no GT mask or skeleton argument.
        self.assertEqual(len(generator.call_args.args),7)

    def test_same_gt_component_can_be_a_negative_shortcut(self):
        net=base_model().eval()
        with torch.no_grad(): result=net(torch.randn(1,3,64,64),capture_diagnostics=True)
        path=candidate(a=(5,2),b=(5,13))
        path['path_points']=np.asarray([(5,x) for x in range(2,14)],np.int32)
        skeleton=np.zeros((64,64),np.uint8)
        skeleton[22:51,10]=1; skeleton[50,10:55]=1; skeleton[22:51,54]=1
        mask=cv2.dilate(skeleton,np.ones((5,5),np.uint8))
        p0=np.zeros((64,64),np.float32); p0[20:25,8:13]=.9; p0[20:25,52:57]=.9
        result['p0']=torch.from_numpy(p0)[None,None]
        with patch('networks.v9_final_path_supervision.generate_paths',return_value=([path],{'paths':1})):
            paths,_,_,_=predicted_candidates(net,result,torch.from_numpy(mask)[None,None],torch.from_numpy(skeleton)[None,None])
        self.assertEqual(len(paths[0]),1)
        self.assertEqual(paths[0][0]['label'],0)
        self.assertEqual(paths[0][0]['subtype'],'shortcut')
        bg=paths[0][0]['background']
        self.assertTrue((mask[bg[:,0],bg[:,1]]==0).all())

    def test_val_comparison_supports_original_and_path_models(self):
        gt=np.zeros((32,32),bool); gt[16,4:28]=True
        probability=np.where(gt,.9,.1).astype(np.float32)
        cases=[{'case':'synthetic','gt':gt,'p0':probability,'A_original':probability,
                'P0_surface':probability,'P1_path':probability}]
        bins={name:{'eligible':0,'closed':0} for name in ('3-7','8-15','16-31','32+')}
        modules={'run_q3_r3_mass':SimpleNamespace(PROTOCOL=dict(DEFAULT_PROTOCOL,image_size=1500)),
                 'run_reliable_directional_mass':SimpleNamespace(gap_lengths=lambda *args:(bins,0))}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.v9_final_test_') as temp:
            with patch('builtins.print'),patch.dict('sys.modules',modules):
                report=compare_cases(cases,['A_original','P0_surface','P1_path'],.45,0,Path(temp))
            self.assertEqual(report['P1_path']['same_threshold']['metrics']['iou'],1.)
            self.assertTrue((Path(temp)/'REPORT.md').is_file())

    def test_junction_and_ambiguous_negative_are_not_forced_into_training(self):
        skeleton=np.zeros((64,64),np.uint8)
        skeleton[10:50,34]=1; skeleton[22,10:55]=1
        self.assertTrue(_junction_mask(skeleton)[22,34])
        net=base_model().eval()
        with torch.no_grad(): result=net(torch.randn(1,3,64,64),capture_diagnostics=True)
        path=candidate(a=(5,2),b=(5,13)); path['path_points']=np.asarray([(5,x) for x in range(2,14)],np.int32)
        result['p0']=torch.zeros(1,1,64,64)
        label={'label':0,'subtype':'lateral_bridge','background_fraction':.5}
        with patch('networks.v9_final_path_supervision.generate_paths',return_value=([path],{'paths':1})), \
             patch('networks.v9_final_path_supervision.label_paths',return_value=([label],[])):
            paths,_,stats,_=predicted_candidates(net,result,torch.zeros(1,1,64,64),torch.from_numpy(skeleton)[None,None])
        self.assertEqual(paths,[[]]); self.assertEqual(stats['ignored_ambiguous_junction'],1)
        with patch('networks.v9_final_path_supervision.generate_paths',return_value=([path],{'paths':1})), \
             patch('networks.v9_final_path_supervision.label_paths',return_value=([label],[])):
            paths,_,stats,_=predicted_candidates(net,result,torch.zeros(1,1,64,64),torch.zeros(1,1,64,64))
        self.assertEqual(paths,[[]]); self.assertEqual(stats['ignored_ambiguous_negative'],1)

    def test_training_saves_and_resumes_at_epoch_boundary(self):
        net=base_model(); net.v9_training_config={'synthetic':True}; identity={'synthetic':True}
        batch={'image':torch.randn(3,64,64),'mask':torch.zeros(1,64,64),'skeleton':torch.zeros(1,64,64)}
        class Dataset:
            def __len__(self): return 2
            def __getitem__(self,index): return batch
            def set_epoch(self,epoch): self.epoch=epoch
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.v9_final_test_') as temp:
            output=Path(temp); source=output/'source.pth'; source.write_bytes(b'synthetic-checkpoint')
            args=SimpleNamespace(output_dir=output,device=torch.device('cpu'),root_path=output,seed=1234,
                batch_size=2,num_workers=0,epochs=1,val_interval=1,max_paths=4,max_rank_pairs=4,
                tau=.1,margin=.1,rank_temperature=.1,lambda_pos=.1,lambda_neg=.1,lambda_bg_path=.05,lambda_rank=.1)
            baseline={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45}
            config={'original_v9_sha256':runner.v9.digest(source)}
            (output/'smoke.json').write_text('{"passed":true,"path_loss_config":{"original_v9_sha256":"'+runner.v9.digest(source)+'"},"identity":{"synthetic":true}}')
            fakepaths=([[item(1,25),item(0,35)],[item(1,25),item(0,35)]],[[(0,1)],[(0,1)]],{}, {})
            modules={'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda root,split:[None]),
                     'run_reliable_directional_mass':SimpleNamespace(evaluate_thresholds=lambda *a,**k:({.6:{'selection_score':.5}},None,None,None))}
            with patch.object(runner,'dataset',return_value=Dataset()),patch.object(runner,'predicted_candidates',return_value=fakepaths), \
                 patch.dict('sys.modules',modules),patch('builtins.print'):
                runner.train(args,net,baseline,identity,source,config,PathConfig())
                runner.train(args,net,baseline,identity,source,config,PathConfig())
            self.assertTrue((output/'best.pth').is_file()); self.assertTrue((output/'last.pth').is_file())
            self.assertEqual(len((output/'epochs.jsonl').read_text().splitlines()),1)

    def test_real_batch_smoke_restores_the_v9_weights(self):
        net=base_model(); initial={key:value.clone() for key,value in net.state_dict().items()}
        sample={'image':torch.randn(3,64,64),'mask':torch.zeros(1,64,64),'skeleton':torch.zeros(1,64,64)}
        class Dataset:
            def __len__(self): return 2
            def __getitem__(self,index): return sample
            def set_epoch(self,epoch): self.epoch=epoch
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.v9_final_test_') as temp:
            args=SimpleNamespace(output_dir=Path(temp),device=torch.device('cpu'),seed=1234,batch_size=2,
                max_paths=4,max_rank_pairs=4,tau=.1,margin=.1,rank_temperature=.1,
                lambda_pos=.1,lambda_neg=.1,lambda_bg_path=.05,lambda_rank=.1)
            baseline={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45}
            paths=([[item(1,25),item(0,35)],[item(1,25),item(0,35)]],[[(0,1)],[(0,1)]],{}, {})
            with patch.object(runner,'dataset',return_value=Dataset()),patch.object(runner,'predicted_candidates',return_value=paths),patch('builtins.print'):
                runner.smoke(args,net,baseline,{'synthetic':True},PathConfig(),{'synthetic':True})
            for key,value in net.state_dict().items(): torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
            self.assertTrue((Path(temp)/'smoke.json').is_file())


if __name__=='__main__': unittest.main()
