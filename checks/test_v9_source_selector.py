"""Actual-neighbour choice, GT isolation, native relation labels and strict stages."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo

from checks.test_v9_state_gate import model as fake_v9
from networks.reliable_directional_repair import DIRECTIONS,shifted,dynamic_losses
from networks.v9_source_selector import SourceSelectedRepair,source_support,relation_loss
from networks.v9_source_relation_labels import native_relations,RelationLabels,eligible_relations,match_relations
import run_v9_source_selector as runner


class SourceSelectorTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(1234); torch.set_num_threads(1)

    def model(self): return SourceSelectedRepair(fake_v9()).eval()

    def test_scalar_mode_bitwise_reproduces_original_checkpoint_forward(self):
        model=self.model(); image=torch.randn(1,3,64,64)
        with torch.no_grad(): old=model.original(image,capture_diagnostics=True); result=model(image,mode='scalar')
        for key in ('p1','u1','u2','u3','g_step','transport_delta'): self.assertTrue(torch.equal(old[key],result[key]))
        self.assertEqual(len(model.original.dynamics._forward_hooks),0)
        self.assertEqual(len(model.original.gate_relation._forward_hooks),0)

    def test_nine_choices_budget_bounds_and_three_live_steps(self):
        model=self.model(); model.set_phase('repair')
        result=model(torch.randn(2,3,64,64)); choice=result['source_choice']
        self.assertEqual(tuple(choice.shape),(2,3,9,16,16)); self.assertEqual(len(result['states']),4)
        torch.testing.assert_close(choice.sum(2),torch.ones_like(choice[:,:,0]))
        for k in range(3):
            expected=result['transport_rate'][:,k]*(choice[:,k,1:]*(result['source_states'][:,k]-result['states'][k])).sum(1,keepdim=True)
            torch.testing.assert_close(expected,result['transport_delta'][:,k])
            torch.testing.assert_close(result['source_transport_delta'][:,k].sum(1,keepdim=True),result['transport_delta'][:,k])
            torch.testing.assert_close(result['alpha'][:,k]*(result['m'][:,k]-result['states'][k]),result['transport_delta'][:,k],atol=1e-7,rtol=1e-5)
            self.assertTrue(result['states'][k+1].requires_grad)
            self.assertGreaterEqual(float(result['states'][k+1].min()),0.)
            self.assertLessEqual(float(result['states'][k+1].max()),1.)
        result['p1'].mean().backward()
        self.assertGreater(runner.grad_report(model)['edge'],0.)
        self.assertGreater(runner.grad_report(model)['stay'],0.)
        self.assertEqual(runner.grad_report(model)['original'],0.)

    def test_new_mode_is_independent_of_unused_scalar_G_heads(self):
        model=self.model(); image=torch.randn(1,3,64,64)
        with torch.no_grad():
            before=model(image)
            model.original.state_gate.net[-1].bias.add_(100.)
            model.original.gate_relation.delta[-1].bias.add_(100.)
            model.original.relations.direction_head.bias[0].add_(100.)
            after=model(image)
        self.assertTrue(torch.equal(before['p1'],after['p1']))
        self.assertFalse(torch.equal(before['legacy_scalar_G'],after['legacy_scalar_G']))

    def test_actual_neighbour_feature_and_state_difference_channel(self):
        model=self.model(); result=model(torch.randn(1,3,64,64),mode='scalar')
        context=result['selector_context']; state=torch.arange(256,dtype=torch.float32).reshape(1,1,16,16)/256
        seen=[]; handle=model.selector.edge[0].register_forward_pre_hook(lambda unused,args:seen.append(args[0].detach()))
        with torch.no_grad(): out=model.selector(context,state,context['r'],torch.ones_like(state),torch.ones_like(state)*.2)
        handle.remove()
        self.assertEqual(len(seen),8)
        # scalar block starts after nine32-channel feature blocks.
        for d,(dy,dx) in enumerate(DIRECTIONS):
            torch.testing.assert_close(seen[d][:,288:289],state)
            torch.testing.assert_close(seen[d][:,289:290],shifted(state,dy,dx))
            torch.testing.assert_close(seen[d][:,290:291],shifted(state,dy,dx)-state)
            torch.testing.assert_close(out['source_states'][:,d:d+1],shifted(state,dy,dx))

    def test_original_per_source_decomposition_matches_real_transport(self):
        model=self.model()
        with torch.no_grad(): result=model(torch.randn(1,3,64,64),mode='scalar')
        weights,updates=runner.original_edge_terms(result)
        self.assertEqual(weights.shape[2],8)
        torch.testing.assert_close(updates.sum(2,keepdim=True),result['transport_delta'],atol=1e-7,rtol=1e-4)

    def test_out_of_bounds_sources_are_not_selected(self):
        model=self.model(); result=model(torch.randn(1,3,64,64))
        invalid=~result['source_valid']; weights=result['source_choice'][:,:,1:]
        self.assertTrue(bool((weights[invalid]==0).all()))

    def test_stay_can_turn_all_transport_off_without_scalar_G(self):
        model=self.model()
        with torch.no_grad(): model.selector.stay[-1].bias.fill_(100.)
        with torch.no_grad(): result=model(torch.randn(1,3,64,64))
        self.assertLess(float(result['transport_delta'].abs().max()),1e-35)

    def test_native_true_road_and_parallel_shortcut_have_different_labels(self):
        mask=torch.zeros(1,1,64,64); mask[:,:,14,4:60]=1.; mask[:,:,18,4:60]=1.
        native=native_relations(mask,mask,(16,16),RelationLabels(junction_radius=0))
        # Work centre (3,3) -> native(14,14); d1 right follows a road,
        # d3 down crosses background between two parallel roads.
        self.assertEqual(int(native['labels'][0,1,3,3]),1)
        self.assertEqual(int(native['labels'][0,3,3,3]),0)
        self.assertEqual(int(native['negative_type'][0,3,3,3]),1)

    def test_distant_GT_connection_does_not_validate_local_shortcut(self):
        mask=torch.zeros(1,1,64,64); mask[:,:,14,4:60]=1.; mask[:,:,18,4:60]=1.; mask[:,:,14:19,50]=1.
        native=native_relations(mask,mask,(16,16),RelationLabels(junction_radius=0))
        self.assertEqual(int(native['labels'][0,3,3,3]),0)

    def test_native_one_pixel_gap_is_not_filled_by_label_interpolation(self):
        mask=torch.zeros(1,1,64,64); mask[:,:,14,4:60]=1.; mask[:,:,14,16]=0.
        native=native_relations(mask,mask,(16,16),RelationLabels(junction_radius=0))
        self.assertEqual(int(native['labels'][0,1,3,3]),0)

    def test_native_mapping_at_two_pixel_stride_keeps_close_roads_separate(self):
        mask=torch.zeros(1,1,64,64); mask[:,:,7,4:60]=1.; mask[:,:,9,4:60]=1.
        native=native_relations(mask,mask,(32,32),RelationLabels(junction_radius=0))
        self.assertEqual(int(native['labels'][0,1,3,3]),1)
        self.assertEqual(int(native['labels'][0,3,3,3]),0)

    def test_junction_and_boundaries_are_ignored(self):
        mask=torch.zeros(1,1,64,64); mask[:,:,14,4:60]=1.; mask[:,:,4:60,14]=1.
        native=native_relations(mask,mask,(16,16))
        self.assertTrue(bool((native['labels'][0,:,3,3]==-1).all()))
        self.assertTrue(bool((native['labels'][0,:,0,:]==-1).all()))

    def test_small_curved_GT_detour_is_ignored_instead_of_a_hard_negative(self):
        mask=torch.zeros(1,1,64,64)
        mask[:,:,14,14:17]=1.; mask[:,:,14:19,16]=1.; mask[:,:,18,14:17]=1.
        native=native_relations(mask,mask,(16,16),RelationLabels(junction_radius=0))
        self.assertEqual(int(native['labels'][0,3,3,3]),-1)

    def test_ambiguous_endpoint_between_two_native_roads_is_not_a_positive(self):
        mask=torch.zeros(1,1,64,64); mask[:,:,13,4:60]=1.; mask[:,:,15,4:60]=1.
        native=native_relations(mask,mask,(16,16),RelationLabels(junction_radius=0))
        self.assertEqual(int(native['labels'][0,1,3,3]),-1)

    def test_native_mapping_rejects_wrong_skeleton_or_fractional_scale(self):
        mask=torch.zeros(1,1,64,64)
        with self.assertRaises(ValueError): native_relations(mask,mask[:,:,:32],(16,16))
        with self.assertRaises(ValueError): native_relations(mask,mask,(15,16))

    def test_phase_freezing_and_optimizer_coverage(self):
        model=self.model(); args=SimpleNamespace(selector_lr=1e-4,joint_lr=5e-6)
        for phase in ('relation','repair','joint'):
            opt=runner.configure(model,phase,args); model.train()
            self.assertFalse(model.original.frozen.training)
            self.assertTrue(all(not p.requires_grad for p in model.original.frozen.parameters()))
            self.assertEqual(len({id(p) for g in opt.param_groups for p in g['params']}),sum(len(g['params']) for g in opt.param_groups))
            if phase!='joint': self.assertTrue(all(not p.requires_grad for p in model.original.parameters()))
            self.assertTrue(all(not p.requires_grad for p in model.original.state_gate.parameters()))

    def test_relation_training_has_no_GT_or_supervision_in_state_forward(self):
        model=self.model(); image=torch.randn(1,3,64,64)
        with torch.no_grad(): reference=model(image,mode='scalar')
        before=[state.clone() for state in reference['states']]
        labels=torch.zeros(1,3,8,16,16,dtype=torch.int8); labels[:,:,:,:,:8]=1
        selected=torch.ones_like(labels,dtype=torch.bool)
        loss,_,count=relation_loss(model.selector,reference['selector_context'],reference,labels,selected)
        loss.backward()
        self.assertGreater(count,0); self.assertGreater(runner.grad_report(model)['edge'],0.)
        self.assertEqual(runner.grad_report(model)['original'],0.); self.assertEqual(runner.grad_report(model)['stay'],0.)
        for old,new in zip(before,reference['states']): self.assertTrue(torch.equal(old,new))

    def test_strict_checkpoint_roundtrip_and_wrong_identity_rejection(self):
        model=self.model(); args=SimpleNamespace(identity={'synthetic':True},source_sha='source',experiment_config={'synthetic':True})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.source_selector_test_') as temp:
            path=Path(temp)/'checkpoint.pth'; runner.checkpoint(path,model,args,'relation',1,.5)
            before={key:value.clone() for key,value in model.selector.state_dict().items()}
            with torch.no_grad(): model.selector.edge[-1].bias.add_(2.)
            runner.restore(path,model,args,'relation')
            for key,value in model.selector.state_dict().items(): self.assertTrue(torch.equal(value,before[key]))
            with self.assertRaises(ValueError): runner.restore(path,model,args,'repair')
            args.source_sha='different'
            with self.assertRaises(ValueError): runner.restore(path,model,args,'relation')

    def test_empty_relation_loss_has_explicit_zero_without_fake_labels(self):
        model=self.model()
        with torch.no_grad(): ref=model(torch.randn(1,3,64,64),mode='scalar')
        labels=torch.full((1,3,8,16,16),-1,dtype=torch.int8); selected=torch.zeros_like(labels,dtype=torch.bool)
        loss,_,count=relation_loss(model.selector,ref['selector_context'],ref,labels,selected)
        self.assertEqual(count,0); self.assertEqual(float(loss),0.); loss.backward()

    def test_matching_never_borrows_opposite_labels_across_images_or_directions(self):
        labels=torch.tensor([1,0,1,0],dtype=torch.int8).reshape(1,1,1,1,4).expand(2,3,8,1,4).clone()
        rows={'labels':labels,'valid':torch.ones_like(labels,dtype=torch.bool),
            'action':torch.tensor([.1,.1,-.1,-.1]).reshape(1,1,1,1,4).expand_as(labels),
            'source':torch.ones_like(labels,dtype=torch.float32)*.2}
        selected,counts=match_relations(rows)
        self.assertEqual(counts['matched_pairs'],96)
        self.assertEqual(int((selected&(labels==1)).sum()),int((selected&(labels==0)).sum()))
        rows['labels'][0]=1; rows['labels'][1]=0
        self.assertEqual(match_relations(rows)[1]['matched_pairs'],0)
        rows['labels'][:,:,:4]=1; rows['labels'][:,:,4:]=0
        self.assertEqual(match_relations(rows)[1]['matched_pairs'],0)

    def test_full_batch_smoke_checks_frozen_gradients_and_shapes(self):
        model=self.model(); image=torch.randn(3,512,512)
        mask=torch.zeros(1,512,512); mask[:,250:259,20:490]=1.
        skeleton=torch.zeros_like(mask); skeleton[:,254,20:490]=1.
        class Dataset:
            def __init__(self,*args,**kwargs): pass
            def set_epoch(self,epoch): pass
            def __len__(self): return 2
            def __getitem__(self,index): return {'image':image,'mask':mask,'skeleton':skeleton,'case_name':str(index)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.source_selector_test_') as temp:
            args=SimpleNamespace(root_path=Path(temp),output_dir=Path(temp),device=torch.device('cpu'),
                seed=1234,selector_lr=1e-4,joint_lr=5e-6,labels=RelationLabels(),
                source_losses={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45})
            with patch.object(runner,'RoadSkeletonDataset',Dataset),patch('builtins.print'): runner.smoke(args,model)
            import json
            value=json.loads((Path(temp)/'smoke.json').read_text())
            self.assertEqual(value['physical_batch'],2); self.assertEqual(value['source_choice_shape'],[2,3,9,128,128])
            self.assertEqual(value['repair_gradients']['original'],0.)
            self.assertGreater(value['repair_gradients']['edge'],0.); self.assertGreater(value['repair_gradients']['stay'],0.)

    def test_one_epoch_relation_then_repair_and_resume_are_runnable(self):
        import json
        model=self.model()
        before={key:value.clone() for key,value in model.original.state_dict().items()}
        class Dataset:
            def __init__(self,*args,**kwargs): pass
            def set_epoch(self,epoch): pass
            def __len__(self): return 2
            def __getitem__(self,index):
                image=torch.arange(3*64*64).reshape(3,64,64).float()/10000.+index*.1
                mask=torch.zeros(1,64,64); mask[:,10:26,4:60]=1.
                skeleton=torch.zeros_like(mask); skeleton[:,14,4:60]=1.
                return {'image':image,'mask':mask,'skeleton':skeleton,'case_name':str(index)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.source_selector_test_') as temp:
            root=Path(temp); source=root/'source.pth'; identity={'synthetic':True}
            runner.v9.save_checkpoint(source,model.original,identity,'D',20,.5)
            args=SimpleNamespace(root_path=root,output_dir=root,device=torch.device('cpu'),seed=1234,
                selector_lr=1e-4,joint_lr=5e-6,labels=RelationLabels(interest_threshold=0.),relation_epochs=1,repair_epochs=1,
                num_workers=0,source=source,source_sha='synthetic',identity=identity,experiment_config={'synthetic':True},
                source_losses={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45})
            modules={'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *a:[None]),
                'run_reliable_directional_mass':SimpleNamespace(evaluate_thresholds=lambda *a,**k:({.6:{'selection_score':.5}},None,None,None))}
            with patch.object(runner,'RoadSkeletonDataset',Dataset),patch.object(runner,'audit',return_value={'score':-.2}),patch.dict('sys.modules',modules),patch('builtins.print'):
                runner.train(args,model,'relation')
                for key,value in model.original.state_dict().items(): self.assertTrue(torch.equal(value,before[key]))
                best=root/'relation_best'; best.mkdir(); (best/'relations.json').write_text(json.dumps({'screening':{'passed':True}}))
                runner.train(args,model,'repair'); runner.train(args,model,'repair')
            self.assertTrue((root/'repair/training_complete.json').is_file())
            state=runner.restore(root/'repair/best.pth',model,args,'repair')
            self.assertEqual(state['epoch'],1)
            with self.assertRaises(ValueError): runner.v9.restore_checkpoint(model.original,root/'repair/best.pth',identity,'D')

    def test_failed_fixed_relation_screen_does_not_allow_repair_training(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.source_selector_test_') as temp:
            root=Path(temp); prior=root/'relation'; prior.mkdir(); (prior/'best.pth').write_bytes(b'not loaded')
            report=root/'relation_best'; report.mkdir(); (report/'relations.json').write_text(json.dumps({'screening':{'passed':False}}))
            args=SimpleNamespace(output_dir=root,repair_epochs=20,relation_epochs=5)
            modules={'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *a:[None]),
                'run_reliable_directional_mass':SimpleNamespace(evaluate_thresholds=lambda *a,**k:None)}
            with patch.dict('sys.modules',modules),patch.object(runner,'configure') as configure:
                with self.assertRaises(ValueError): runner.train(args,self.model(),'repair')
                configure.assert_not_called()

    def test_full_validation_relation_audit_produces_paired_rows(self):
        model=self.model(); image=torch.randn(3,512,512)
        mask=torch.zeros(1,512,512); mask[:,246:262,20:490]=1.
        skeleton=torch.zeros_like(mask); skeleton[:,254,20:490]=1.
        dataset=[{'image':image,'mask':mask,'skeleton':skeleton,'case_name':f'synthetic_{index}'} for index in range(14)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.source_selector_test_') as temp:
            args=SimpleNamespace(root_path=Path(temp),device=torch.device('cpu'),seed=1234,
                labels=RelationLabels(interest_threshold=0.),bootstrap=0)
            modules={'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *a:dataset)}
            with patch.dict('sys.modules',modules),patch('builtins.print'):
                result=runner.audit(args,model,Path(temp))
            self.assertIsNotNone(result['score'])
            self.assertTrue((Path(temp)/'matched_correct_relations.csv').is_file())
            with np.load(Path(temp)/'relation_scores.npz') as rows:
                self.assertEqual(len(rows['label']),len(rows['edge']))
                self.assertEqual(len(rows['label']),len(rows['baseline']))
                self.assertEqual(len(np.unique(rows['image'])),14)
                keys=np.stack([rows['image'],rows['step'],rows['direction'],rows['native_y'],rows['native_x']],1)
                self.assertEqual(len(np.unique(keys,axis=0)),len(keys))
            self.assertFalse(result['screening']['passed']) # no bootstrap evidence, never auto-launch repair

    def test_surface_comparison_runs_real_stitch_and_actual_source_statistics(self):
        model=self.model()
        with torch.no_grad():
            model.original.frozen.surface.weight.zero_(); model.original.frozen.surface.bias.fill_(-2.)
        image=torch.randn(3,512,512); mask=torch.zeros(1,512,512); mask[:,246:262,20:490]=1.
        skeleton=torch.zeros_like(mask); skeleton[:,254,20:490]=1.
        dataset=[{'image':image,'mask':mask,'skeleton':skeleton,'case_name':f'synthetic_{index}'} for index in range(14)]
        metrics={'tp':0,'fp':0,'fn':int(mask.sum())*14,'recall':0.,'iou':0.,'cldice':0.,'apls_bidirectional_approx':0.}
        paired={'closed':0,'background_bridge_proxy':0}
        reports={name:{'same_threshold':{'metrics':metrics,'paired':paired}} for name in ('A_original','C_selector')}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='.source_selector_test_') as temp:
            root=Path(temp); source=root/'source.pth'; identity={'synthetic':True}
            runner.v9.save_checkpoint(source,model.original,identity,'D',20,.5)
            args=SimpleNamespace(root_path=root,output_dir=root,device=torch.device('cpu'),source=source,
                identity=identity,labels=RelationLabels(),selection={'threshold':.6,'report':{'p1':metrics},
                    'threshold_curves':{'0.6':{'p1':metrics,'selection_score':.5}}})
            modules={'run_q3_r3_mass':SimpleNamespace(full_dataset=lambda *a:dataset)}
            with patch.dict('sys.modules',modules),patch('gate_benefit_protocol.threshold_curve',return_value={.6:{'metrics':metrics,'selection_score':.5}}),patch('gate_benefit_protocol.operating_points',return_value={}),patch('diagnostics.v91_counterfactual.evaluate',return_value=reports),patch('builtins.print'):
                runner.compare_surface(args,model,'repair')
            output=root/'repair_surface_val'
            self.assertTrue((output/'actual_source_relations.csv').is_file())
            self.assertTrue((output/'REPORT.md').is_file())
            self.assertEqual(len(list(output.glob('image_*.npz'))),14)


if __name__=='__main__': unittest.main()
