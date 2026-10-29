"""GT isolation, signed transport labels, pre-logit gradients and paired schedules."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
# Optimizer imports must persist outside the lightweight module mocks below.
import torch._dynamo

from checks.test_v9_state_gate import model as fake_v9
from diagnostics.v92_oracle_gate import oracle_choice
from networks.reliable_directional_repair import compose_surface_logits
from networks.v9_transport_benefit import (BenefitConfig,forward_trace,benefit_labels,
    match_labels,gate_benefit_loss,BenefitMetrics,restore_model,_exact_auc)
import run_v9_gate_benefit as runner
from gate_benefit_protocol import decision_audit,run_comparison,run_step_diagnostics


def synthetic(logits_value=0.):
    logits=torch.full((1,3,1,1,5),float(logits_value),requires_grad=True)
    u=torch.full((1,1,1,5),.5,requires_grad=True)
    delta=torch.tensor([[[[.2,.2,-.2,-.2,.2]]]],requires_grad=True)
    r=torch.full_like(u,.5,requires_grad=True); m=u+delta
    result={'states':[u,u,u,u],'gate_logits':logits,'g_step':logits.sigmoid(),
        'm':m[:,None].expand(-1,3,-1,-1,-1),'transport_rate':r[:,None].expand(-1,3,-1,-1,-1),
        'proposal_delta':torch.zeros_like(logits),'anchor_delta':torch.zeros_like(logits),
        'source_available':torch.full_like(logits,.2),'r':torch.ones_like(u)*.2}
    gt=torch.tensor([[[[1.,0.,1.,0.,.5]]]],requires_grad=True)
    return result,gt,delta


class GateBenefitTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(1234); torch.set_num_threads(1)

    def test_labels_equal_oracle_choice_for_positive_and_negative_actions(self):
        result,gt,_=synthetic()
        labels=benefit_labels(result,gt)
        self.assertEqual(labels['target'][0,0,0,0,:4].tolist(),[1.,0.,0.,1.])
        self.assertFalse(bool(labels['valid'][0,0,0,0,4]))
        action=result['transport_rate'][:,0]*(result['m'][:,0]-result['states'][0])
        chosen,_=oracle_choice(result['states'][0],torch.zeros_like(action),torch.zeros_like(action),
                              action,gt,result['g_step'][:,0])
        torch.testing.assert_close(chosen[labels['valid'][:,0]],labels['target'][:,0][labels['valid'][:,0]])
        self.assertTrue(all(not value.requires_grad for value in labels.values()))

    def test_matching_preserves_image_step_sign_and_source_action_bins(self):
        result,gt,_=synthetic(); labels=benefit_labels(result,gt)
        selected,stats=match_labels(labels,BenefitConfig(max_pairs_per_sign=1))
        self.assertEqual(stats['pairs_positive_action'],3); self.assertEqual(stats['pairs_negative_action'],3)
        self.assertEqual(int(selected.sum()),12)
        self.assertEqual(stats['matching_action_abs_difference_mean'],0.)
        altered={key:value.expand(2,*value.shape[1:]).clone() for key,value in labels.items()}
        altered['target'][0]=1.; altered['target'][1]=0.
        mask,stats=match_labels(altered)
        self.assertEqual(int(mask.sum()),0); self.assertEqual(stats['pairs'],0)

    def test_only_logit_branch_receives_label_loss_gradients(self):
        result,gt,delta=synthetic()
        loss,labels,selected,report=gate_benefit_loss(result,gt)
        loss.backward()
        self.assertGreater(float(result['gate_logits'].grad.abs().sum()),0.)
        self.assertIsNone(delta.grad); self.assertIsNone(gt.grad); self.assertIsNone(result['states'][0].grad)
        self.assertEqual(report['matching']['matched_allow'],report['matching']['matched_reject'])

    def test_saturated_gate_retains_bce_logits_gradient(self):
        result,gt,_=synthetic(40.)
        self.assertTrue(bool((result['g_step']==1.).all()))
        loss,_,_,_=gate_benefit_loss(result,gt); loss.backward()
        self.assertGreater(float(result['gate_logits'].grad[0,0,0,0,1]),0.)

    def test_empty_step_is_explicit_zero_in_K_step_mean(self):
        result,gt,_=synthetic()
        result['transport_rate']=result['transport_rate'].clone()
        result['transport_rate'][:,1]=0.
        loss,_,_,report=gate_benefit_loss(result,gt)
        self.assertAlmostEqual(float(loss),float(np.log(2)*2/3),places=6)
        self.assertTrue(report['steps'][1]['empty'])
        blank,_,selected,_=gate_benefit_loss(result,gt,BenefitConfig(gain_epsilon=100.))
        self.assertEqual(int(selected.sum()),0); self.assertEqual(float(blank),0.)

    def test_training_labels_never_change_actual_forward_states(self):
        net=fake_v9().eval(); image=torch.randn(1,3,64,64)
        with torch.no_grad():
            ordinary=net(image,capture_diagnostics=True); traced=forward_trace(net,image,3)
            before=[state.clone() for state in traced['states']]
            gate_benefit_loss(traced,torch.zeros(1,1,64,64))
            gate_benefit_loss(traced,torch.ones(1,1,64,64))
        self.assertTrue(torch.equal(ordinary['p1'],traced['p1']))
        for a,b in zip(before,traced['states']): self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(traced['gate_logits'].sigmoid(),traced['g_step']))

    def test_6_12_steps_recompute_shared_rates_and_final_surface(self):
        net=fake_v9().eval(); image=torch.randn(1,3,64,64)
        for steps in (6,12):
            calls=[]; handle=net.dynamics.rates.register_forward_hook(lambda *args:calls.append(1))
            with torch.no_grad(): result=forward_trace(net,image,steps)
            handle.remove(); self.assertEqual(len(calls),steps); self.assertEqual(net.dynamics.steps,3)
            self.assertEqual(len(result['states']),steps+1)
            z,_=compose_surface_logits(result['z0'],result['states'][0],result['states'][-1])
            self.assertTrue(torch.equal(z,result['z1']))
            for k in range(steps):
                torch.testing.assert_close(result['states'][k+1],result['states'][k]+result['transport_delta'][:,k]+
                    result['proposal_delta'][:,k]+result['anchor_delta'][:,k],rtol=1e-5,atol=1e-7)

    def test_metrics_distinguish_helpful_and_harmful_acceptance(self):
        result,gt,_=synthetic(); labels=benefit_labels(result,gt)
        result['gate_logits']=torch.where(labels['target']>.5,torch.tensor(3.),torch.tensor(-3.))
        result['g_step']=result['gate_logits'].sigmoid(); labels=benefit_labels(result,gt)
        selected,_=match_labels(labels); metrics=BenefitMetrics(3); metrics.add(labels,selected,result)
        report=metrics.report()
        self.assertEqual(report['matched_helpful_acceptance_rate'],1.)
        self.assertEqual(report['matched_harmful_acceptance_rate'],0.)
        self.assertEqual(report['steps'][0]['matched_auc_hist_approx']['roc_auc'],1.)
        self.assertEqual(report['matched_auc_exact']['roc_auc'],1.)

    def test_uniform_gate_reduction_alone_does_not_improve_auc(self):
        pos=np.array([.6,.9]); neg=np.array([.4,.8])
        before=_exact_auc(pos,neg); after=_exact_auc(pos*.5,neg*.5)
        self.assertEqual(before,after)

    def test_checkpoint_rejects_wrong_trained_step_count_and_legacy_loader(self):
        net=fake_v9(); net.v9_training_config={'synthetic':True}; identity={'synthetic':True}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.gate_benefit_test_') as temp:
            path=Path(temp)/'six.pth'
            runner.save(path,net,identity,1,.5,None,{'steps':6})
            with self.assertRaises(ValueError): restore_model(net,path,identity,3)
            with self.assertRaises(ValueError): runner.v9.restore_checkpoint(net,path,identity,'D')
            state=restore_model(net,path,identity,6)
            self.assertEqual(state['benefit_config']['steps'],6)

    def test_paired_training_checks_full_order_and_augmented_images(self):
        net=fake_v9(); net.v9_training_config={'synthetic':True}; identity={'synthetic':True}
        class Dataset:
            epoch=1
            def set_epoch(self,epoch): self.epoch=epoch
            def __len__(self): return 2
            def __getitem__(self,index):
                image=torch.arange(3*64*64).reshape(3,64,64).float()/10000.+index*.1+self.epoch*.01
                mask=torch.zeros(1,64,64); mask[:,:,:32]=1.
                return {'image':image,'mask':mask,'skeleton':torch.zeros_like(mask),'case_name':str(index)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.gate_benefit_test_') as temp:
            out=Path(temp); source=out/'source.pth'; runner.v9.save_checkpoint(source,net,identity,'D',20,.5)
            args=SimpleNamespace(output_dir=out,root_path=out,device=torch.device('cpu'),seed=1234,
                epochs=1,batch_size=2,num_workers=0,val_interval=1,lambda_gate=.1,
                min_action=1e-12,gain_epsilon=1e-12,max_pairs_per_sign=8,
                source_losses={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45})
            modules={'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *args:[None]),
                     'run_reliable_directional_mass':SimpleNamespace(evaluate_thresholds=lambda *a,**k:({.6:{'selection_score':.5}},None,None,None))}
            with patch.object(runner,'make_dataset',return_value=Dataset()),patch.dict('sys.modules',modules),patch('builtins.print'):
                runner.train_variant(args,net,source,identity,'control',3)
                runner.train_variant(args,net,source,identity,'gate',3)
                runner.verify_pair(args)
                # Completed epochs are skipped without duplicating the history on restart.
                runner.train_variant(args,net,source,identity,'gate',3)
                runner.train_variant(args,net,source,identity,'gate',6)
            self.assertTrue((out/'paired_data_verified.json').is_file())
            self.assertEqual(len((out/'gate_3step/epochs.jsonl').read_text().splitlines()),1)
            six=restore_model(net,out/'gate_6step/best.pth',identity,6)
            self.assertEqual(six['benefit_config']['epochs'],1)

    def test_shared_parameter_step_audit_writes_progress_and_visuals(self):
        net=fake_v9().eval(); image=torch.randn(3,512,512)
        mask=torch.zeros(1,512,512); mask[:,250:262,40:470]=1.
        sample={'image':image,'mask':mask,'case_name':'synthetic'}
        probability=np.full((512,512),.1,np.float32)
        cases=[{'case':'synthetic','gt':mask[0].numpy()>.5,'p0':probability,'A_original':probability,
                'S3':probability,'S6':probability}]
        args=SimpleNamespace(root_path=Path('.'),device=torch.device('cpu'),seed=1234)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.gate_benefit_test_') as temp:
            output=Path(temp)
            with patch.dict('sys.modules',{'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *args:[sample])}),patch('builtins.print'):
                report=decision_audit(args,{'S3':net,'S6':net},{'S3':3,'S6':6},cases,
                    {'S3':{'matched_recall':.5},'S6':{'matched_recall':.5}},BenefitConfig(),output)
            self.assertEqual(len(report['S6']['steps']),6)
            self.assertTrue((output/'dynamics_regions.csv').is_file())
            with np.load(output/'visuals/S6/synthetic.npz') as data: self.assertEqual(data['states'].shape[0],7)

    def test_smoke_restores_weights_and_keeps_gt_out_of_states(self):
        net=fake_v9(); before={key:value.clone() for key,value in net.state_dict().items()}
        sample={'image':torch.randn(3,64,64),'mask':torch.ones(1,64,64),'skeleton':torch.zeros(1,64,64)}
        class Dataset:
            def set_epoch(self,epoch): pass
            def __len__(self): return 2
            def __getitem__(self,index): return sample
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.gate_benefit_test_') as temp:
            source=Path(temp)/'source.pth'; source.write_bytes(b'synthetic')
            args=SimpleNamespace(output_dir=Path(temp),seed=1234,device=torch.device('cpu'),lambda_gate=.1,
                source_losses={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45})
            with patch.object(runner,'make_dataset',return_value=Dataset()),patch('builtins.print'):
                report=runner.smoke(args,net,BenefitConfig(),{'synthetic':True},source)
            self.assertTrue(report['GT_did_not_change_states'])
            for key,value in net.state_dict().items(): torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_step_diagnostics_wait_for_selective_transport_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.gate_benefit_test_') as temp:
            args=SimpleNamespace(output_dir=Path(temp))
            with patch('builtins.print'):
                train=unittest.mock.Mock()
                run_step_diagnostics(args,None,None,None,None,None,{'screening':{'passed':False}},train)
            train.assert_not_called()
            self.assertTrue((Path(temp)/'steps_inference_diagnostic/status.json').is_file())

    def test_comparison_requires_discrimination_not_only_smaller_gate(self):
        net=fake_v9().eval(); net.v9_training_config={'synthetic':True}; identity={'synthetic':True}
        gt=np.zeros((32,32),bool); gt[16,4:28]=True; probability=np.where(gt,.9,.1).astype(np.float32)
        metrics={'tp':24*14,'fp':0,'fn':0,'recall':.7,'iou':.6,'cldice':.8,'apls_bidirectional_approx':.7,'short_pred_components':2.}
        def decisions(auc):
            return {name:{'matched_harmful_acceptance_rate':.8 if name=='B_control' else .4,
                'matched_helpful_acceptance_rate':.8,'harmful_actual_loss_per_valid_action':.1 if name=='B_control' else .05,
                'beneficial_actual_gain_per_valid_action':.1,'matched_auc_exact':{'roc_auc':.6 if name=='B_control' else auc}}
                for name in ('A_original','B_control','C_gate3')}
        def infer(*args,**kwargs):
            return [{'case':str(i),'gt':gt,'p0':probability,args[4]:probability} for i in range(14)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.gate_benefit_test_') as temp:
            out=Path(temp); source=out/'source.pth'; runner.v9.save_checkpoint(source,net,identity,'D',20,.5)
            for variant in ('control','gate'):
                runner.save(out/f'{variant}_3step/best.pth',net,identity,1,.5,None,
                    {'variant':variant,'steps':3,'original_sha256':runner.v9.digest(source)})
            reference=out/'reference.json'; reference.write_text('{"synthetic": true}')
            args=SimpleNamespace(output_dir=out,root_path=out,device=torch.device('cpu'),reference_set=reference)
            selection={'threshold':.6,'report':{'p1':metrics},'threshold_curves':{'.6':{'p1':metrics,
                'selection_score':.5,'paired':{'background_bridge_proxy':10}}}}
            def evaluate(cases,points,threshold,expected,output,modes):
                (output/'reference_set.json').write_text(reference.read_text())
                return {name:{key:{'threshold':value,'metrics':dict(metrics),
                    'paired':{'closed':1,'eligible':206,'background_bridge_proxy':10 if name=='B_control' else 5}}
                    for key,value in points[name].items()} for name in modes}
            with patch.dict('sys.modules',{'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *args:[None]*14)}), \
                 patch('gate_benefit_protocol.infer_cases',side_effect=infer), \
                 patch('gate_benefit_protocol.threshold_curve',return_value={.6:{'metrics':metrics,'selection_score':.5}}), \
                 patch('gate_benefit_protocol.evaluate',side_effect=evaluate), \
                 patch('gate_benefit_protocol.decision_audit',return_value=decisions(.6)):
                result=run_comparison(args,net,identity,source,selection,BenefitConfig())
                self.assertFalse(result['screening']['passed'])
            with patch.dict('sys.modules',{'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *args:[None]*14)}), \
                 patch('gate_benefit_protocol.infer_cases',side_effect=infer), \
                 patch('gate_benefit_protocol.threshold_curve',return_value={.6:{'metrics':metrics,'selection_score':.5}}), \
                 patch('gate_benefit_protocol.evaluate',side_effect=evaluate), \
                 patch('gate_benefit_protocol.decision_audit',return_value=decisions(.7)):
                result=run_comparison(args,net,identity,source,selection,BenefitConfig())
                self.assertTrue(result['screening']['passed'])


if __name__=='__main__': unittest.main()
