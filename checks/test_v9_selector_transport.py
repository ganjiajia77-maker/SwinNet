"""Joint live transport, artifact correspondence, reused control and frozen anchors."""
import ast
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch._dynamo
from torch.utils.data import DataLoader

from checks.test_v9_state_gate import model as fake_v9
from networks.v9_source_selector import SourceSelectedRepair,VERSION as SELECTOR_VERSION
from networks.v9_source_relation_labels import RelationLabels
from networks.v9_selector_transport_training import configure_joint,canonical_state,load_canonical,live_relation_loss,SplitDirectionHead
from networks.v9_transport_benefit import BENEFIT_VERSION
from selector_transport_provenance import inspect_selector,inspect_control,find_control,CONTROL_SELECTION
from selector_transport_reference import reference_digest,freeze_gaps,closed_gaps,bridge_proxy,prepared,exact_recall_threshold
import run_v9_selector_transport as runner


ROOT=Path(__file__).resolve().parents[1]


class TinyDataset:
    def set_epoch(self,epoch): self.epoch=epoch
    def __len__(self): return 2
    def __getitem__(self,index):
        image=torch.arange(3*64*64).reshape(3,64,64).float()/10000.+index*.1
        mask=torch.zeros(1,64,64); mask[:,10:26,4:60]=1.
        skeleton=torch.zeros_like(mask); skeleton[:,14,4:60]=1.
        return {'image':image,'mask':mask,'skeleton':skeleton,'case_name':str(index)}


def metadata_fixture(root):
    original=fake_v9(); source=root/'old/stage_D/best.pth'; source.parent.mkdir(parents=True)
    dependencies={}
    for name in ('base','q3'):
        file=root/(name+'.pth'); file.write_bytes(name.encode())
        dependencies[name]={'path':str(file.resolve()),'sha256':runner.v9.digest(file)}
    identity=dict(dependencies,initialization='direct_random_downstream')
    original.v9_training_config={'root_path':str(root/'data'),'seed':1234,'batch_size':2,'reference_threshold':.45,
        'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45}
    runner.v9.save_checkpoint(source,original,identity,'D',20,.5)
    (root/'old/direct_val_selection.json').write_text(json.dumps({'identity':identity,'selected_on':'val',
        'checkpoint':str(source.resolve()),'report':{'paired':{'eligible_p0_gaps':206}}}))
    run=root/'selector'; (run/'relation').mkdir(parents=True); (run/'relation_best').mkdir()
    config={'version':SELECTOR_VERSION,'identity':identity,'source_sha256':runner.v9.digest(source),
        'root_path':str(root/'data'),'seed':1234,'num_workers':0,'hidden':32,'selector_lr':1e-4,
        'labels':asdict(RelationLabels()),'code_sha256':{}}
    (run/'experiment.json').write_text(json.dumps(config))
    wrapper=SourceSelectedRepair(original)
    state={'version':SELECTOR_VERSION,'identity':identity,'source_sha256':config['source_sha256'],'config':config,
        'phase':'relation','epoch':1,'score':-.2,'modules':{name:getattr(original,name).state_dict() for name in runner.v9.MODULE_NAMES},
        'gate_relation':original.gate_relation.state_dict(),'state_gate':original.state_gate.state_dict(),'selector':wrapper.selector.state_dict()}
    torch.save(state,run/'relation/best.pth')
    (run/'relation_best/relations.json').write_text(json.dumps({'screening':{'passed':True},'selection_score':-.2}))
    control=root/'control_exp/control_3step'; control.mkdir(parents=True)
    cfg={'variant':'control','steps':3,'epochs':1,'identity':identity,'original_sha256':config['source_sha256'],
        'root_path':config['root_path'],'seed':1234,'batch_size':2,'num_workers':0,'learning_rates':dict(runner.v9.LR),
        'selection':CONTROL_SELECTION,'val_interval':1,'protocol':'FP32; no EMA; accumulation1','code_sha256':{}}
    (control/'config.json').write_text(json.dumps(cfg))
    ctl={'version':BENEFIT_VERSION,'identity':identity,'benefit_config':cfg,'stage':'D','epoch':1,'score':.5,
        'training_config':original.v9_training_config,'modules':state['modules'],'gate_relation':state['gate_relation'],'state_gate':state['state_gate']}
    torch.save(ctl,control/'best.pth')
    ctl['optimizer']={'param_groups':[{'name':name,'lr':runner.v9.LR[name],'weight_decay':.01}
        for name in runner.v9.STAGE_MODULES['D']]}
    torch.save(ctl,control/'last.pth')
    (control/'training_complete.json').write_text(json.dumps({'epochs':1,'variant':'control'}))
    row={'epoch':1,'losses':{'total':.3,'seg':.2,'cont':.5},'val_score_at_0_60':.5,
        'case_order_sha256':'same','augmented_images_sha256':'same',
        'gradient_comparison_first_matched_batch':{'gate_loss_applied_to_optimizer':False}}
    (control/'epochs.jsonl').write_text(json.dumps(row)+'\n')
    (control/'val_epoch_001.json').write_text(json.dumps({'selection_score':.5,'p1':{'tp':0,'fp':0,'fn':0}}))
    return run,source,control,wrapper


class TransportTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(1234); torch.set_num_threads(1)

    def model(self):
        model=SourceSelectedRepair(fake_v9()); optimizer,report=configure_joint(model,dict(runner.v9.LR),1e-4)
        return model,optimizer,report

    def test_optimizer_excludes_scalar_G_even_from_AdamW_decay(self):
        model,optimizer,_=self.model(); head=model.original.relations.direction_head
        self.assertIsInstance(head,SplitDirectionHead)
        scalar=head.scalar.weight.detach().clone(); frozen={key:value.clone() for key,value in model.original.frozen.state_dict().items()}
        old_gate=model.original.gate_relation.delta[-1].weight.detach().clone()
        result=model(torch.randn(2,3,64,64)); result['p1'].mean().backward(); optimizer.step()
        self.assertTrue(torch.equal(head.scalar.weight,scalar)); self.assertTrue(torch.equal(model.original.gate_relation.delta[-1].weight,old_gate))
        self.assertTrue(all(torch.equal(value,frozen[key]) for key,value in model.original.frozen.state_dict().items()))
        ids={id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) not in ids for p in head.scalar.parameters()))
        self.assertTrue(all(id(p) not in ids for p in model.original.state_gate.parameters()))

    def test_split_head_roundtrips_original_canonical_checkpoint(self):
        model=fake_v9(); before={key:value.clone() for key,value in model.relations.state_dict().items()}
        source=SourceSelectedRepair(model); configure_joint(source,dict(runner.v9.LR),1e-4)
        packed=canonical_state(model.relations)
        for key in before: self.assertTrue(torch.equal(before[key],packed[key]))
        other=fake_v9(); other.relations.load_state_dict(packed,strict=True)
        load_canonical(model.relations,before)
        with self.assertRaises(RuntimeError): load_canonical(model.relations,{'bad':torch.zeros(1)})

    def test_relation_loss_uses_live_logits_and_new_state_graph(self):
        model,optimizer,_=self.model(); batch=next(iter(DataLoader(TinyDataset(),batch_size=2)))
        result=model(batch['image']); result['states'][1].retain_grad()
        loss,info=live_relation_loss(result,batch['mask'],batch['skeleton'],RelationLabels(interest_threshold=0.),1234)
        self.assertGreater(info['matching']['matched_pairs'],0)
        self.assertTrue(all(not value.requires_grad for value in info['rows'].values()))
        loss.backward()
        self.assertGreater(float(model.selector.edge[-1].weight.grad.abs().sum()),0.)
        self.assertIsNotNone(result['states'][1].grad)
        self.assertGreater(float(result['states'][1].grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.original.frozen.parameters()))

    def test_artifact_preflight_binds_selector_source_and_control(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.transport_test_') as temp:
            root=Path(temp); run,source,control,_=metadata_fixture(root)
            selector=inspect_selector(run,root)
            self.assertEqual(selector['original_path'],source)
            value=inspect_control(control,selector,1); self.assertEqual(value['run'],control)
            self.assertEqual(find_control(root,selector,1)['checkpoint'],control/'best.pth')
            with self.assertRaises(ValueError): inspect_control(control,selector,20)
            cfg=json.loads((control/'config.json').read_text()); cfg['learning_rates']['visual']*=2
            (control/'config.json').write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): inspect_control(control,selector,1)

    def test_preflight_rejects_changed_upstream_and_unvalidated_selector(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.transport_test_') as temp:
            root=Path(temp); run,_,_,_=metadata_fixture(root)
            value=json.loads((run/'relation_best/relations.json').read_text()); value['screening']['passed']=False
            (run/'relation_best/relations.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError): inspect_selector(run,root)
            value['screening']['passed']=True; (run/'relation_best/relations.json').write_text(json.dumps(value))
            (root/'base.pth').write_bytes(b'wrong')
            with self.assertRaises(ValueError): inspect_selector(run,root)

    def test_preflight_rejects_control_with_LG_in_total_loss(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.transport_test_') as temp:
            root=Path(temp); run,_,control,_=metadata_fixture(root); selector=inspect_selector(run,root)
            row=json.loads((control/'epochs.jsonl').read_text()); row['losses']['total']=.4
            (control/'epochs.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError): inspect_control(control,selector,1)

    def test_frozen_anchors_match_original_gap_definition_for_all_models(self):
        # Execute the exact unchanged old function without importing the full backbone.
        tree=ast.parse((ROOT/'run_reliable_directional_mass.py').read_text())
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('gap_lengths','path_length_bin')]
        scope={'np':np,'cv2':cv2}; exec(compile(ast.Module(body=functions,type_ignores=[]),'old_gap_definition','exec'),scope)
        gt=np.zeros((64,64),bool); gt[30:35,4:60]=True
        p0=gt.copy(); p0[:,27:36]=False
        case={'case':'synthetic','gt':gt,'p0':np.where(p0,.9,.1).astype(np.float32)}
        old={'images':1,'eligible_reference_gaps':1,'sha256':reference_digest([case])}
        manifest=freeze_gaps([case],old); saved=json.dumps(manifest,sort_keys=True)
        for pred in (p0,gt,np.ones_like(gt)):
            bins,ids=closed_gaps(pred,manifest['images']['synthetic'])
            expected,bridges=scope['gap_lengths'](p0,pred,prepared(case)['skeleton'],gt)
            self.assertEqual(bins,expected); self.assertEqual(bridge_proxy(p0,pred,gt),bridges)
        self.assertEqual(json.dumps(manifest,sort_keys=True),saved)
        old['sha256']='wrong'
        with self.assertRaises(ValueError): freeze_gaps([case],old)

    def test_exact_recall_matching_records_unavoidable_score_ties(self):
        case={'gt':np.array([[1,1,1,0]],bool),'S_selector':np.array([[.8,.7,.6,.9]],np.float32)}
        threshold,error=exact_recall_threshold([case],'S_selector',2)
        self.assertEqual(int(((case['S_selector']>=threshold)&case['gt']).sum()),2); self.assertEqual(error,0)
        case['S_selector'][:]=.5; _,error=exact_recall_threshold([case],'S_selector',2)
        self.assertEqual(error,1)

    def test_actual_update_smoke_covers_joint_gradients_and_G_independence(self):
        model,optimizer,report=self.model()
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.transport_test_') as temp:
            args=SimpleNamespace(output_dir=Path(temp),device=torch.device('cpu'),seed=1234,
                labels=RelationLabels(interest_threshold=0.),lambda_relation=.1,optimizer_report=report,
                source_losses={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45})
            with patch.object(runner,'dataset_train',return_value=TinyDataset()),patch('builtins.print'):
                runner.smoke(args,model,optimizer)
            result=json.loads((Path(temp)/'actual_update_check.json').read_text())
            self.assertTrue(result['passed']); self.assertTrue(result['unused_G_has_no_effect'])
            self.assertGreater(result['selector_changes_P1_mean_abs'],0.)

    def test_joint_epoch_continues_original_weights_and_reuses_control_data(self):
        model,optimizer,report=self.model(); scalar=model.original.relations.direction_head.scalar.weight.detach().clone()
        dataset=TinyDataset(); dataset.set_epoch(1)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(1234+1000003))
        order=hashlib.sha256(); images=hashlib.sha256()
        for batch in loader:
            for name in batch['case_name']: order.update(name.encode()+b'\0')
            images.update(batch['image'].contiguous().numpy().tobytes())
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.transport_test_') as temp:
            args=SimpleNamespace(output_dir=Path(temp),device=torch.device('cpu'),seed=1234,epochs=1,num_workers=0,val_interval=1,
                labels=RelationLabels(interest_threshold=0.),lambda_relation=.1,identity={'synthetic':True},config={'synthetic':True},
                control={'run':Path(temp),'records':{1:{'case_order_sha256':order.hexdigest(),'augmented_images_sha256':images.hexdigest()}}},
                source_losses={'lambda_bg':1.,'lambda_missing':2.,'lambda_cont':.2,'missing_threshold':.45})
            before=model.selector.edge[-1].weight.detach().clone()
            with patch.object(runner,'dataset_train',return_value=dataset),patch.object(runner,'infer_selector'),patch.object(runner,'point_metrics',return_value={'selection_score':.5}),patch('builtins.print'):
                runner.train(args,model,optimizer,[],{})
                runner.train(args,model,optimizer,[],{}) # completed epoch resumes without another step
            self.assertTrue((Path(temp)/'training_complete.json').is_file())
            self.assertFalse(torch.equal(model.selector.edge[-1].weight,before))
            self.assertTrue(torch.equal(model.original.relations.direction_head.scalar.weight,scalar))
            runner.restore(args,model,Path(temp)/'best.pth')
            self.assertEqual(runner.read_json(Path(temp)/'epoch_001.json')['relation_counts']['matched_pairs']>0,True)

    def test_actual_source_statistics_do_not_change_inference(self):
        model,_,_=self.model(); model.eval()
        image=torch.randn(3,512,512); mask=torch.zeros(1,512,512); mask[:,246:262,20:490]=1.
        skeleton=torch.zeros_like(mask); skeleton[:,254,20:490]=1.
        sample={'image':image,'mask':mask,'skeleton':skeleton,'case_name':'synthetic'}
        args=SimpleNamespace(device=torch.device('cpu'),labels=RelationLabels())
        with torch.no_grad(): result=model(image[None])
        before=result['p1'].clone(); callback,values=runner.source_statistics_callback(args,sample)
        with torch.no_grad(): callback(0,0,result)
        self.assertTrue(torch.equal(before,result['p1']))
        for k in range(3):
            signed=sum(values[f'{k+1}:{kind}']['transport_signed_sum'] for kind in ('supported','blocked','uncertain'))
            self.assertAlmostEqual(signed,float(result['source_transport_delta'][:,k].sum()),places=4)

    def test_threshold_cache_is_bound_to_checkpoint_and_can_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.transport_test_') as temp:
            root=Path(temp); args=SimpleNamespace(output_dir=root,config={'control_sha256':'first'},control={'run':root/'old/control_3step'})
            with patch.object(runner,'point_metrics',return_value={'selection_score':.5}),patch('builtins.print'):
                first=runner.curve(args,[],'B_control',{'sha256':'reference'})
            self.assertEqual(len(first),41)
            with patch.object(runner,'point_metrics') as metrics:
                runner.curve(args,[],'B_control',{'sha256':'reference'}); metrics.assert_not_called()
            args.config['control_sha256']='changed'
            with patch.object(runner,'point_metrics',return_value={'selection_score':.4}) as metrics,patch('builtins.print'):
                changed=runner.curve(args,[],'B_control',{'sha256':'reference'})
                self.assertEqual(metrics.call_count,41)
            self.assertTrue(all(value['selection_score']==.4 for value in changed.values()))


if __name__=='__main__': unittest.main()
