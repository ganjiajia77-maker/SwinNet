"""Actual original H0 at 128px + actual source repair, CPU integration tests."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict

import torch
from torch import nn

from networks.full_h0_joint import (build_h0,original_criterion,FullH0Control,
    FullH0Joint,parameter_groups,h0_loss)
from networks.q3_r3_repair import MultiScaleFeatures
from networks.v9_selector_transport_training import canonical_state,gradient_norm
from networks.v9_source_selector import SourceRelationSelector
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_source_relation_labels import RelationLabels
from checks.test_v9_state_gate import model as downstream
import run_full_h0_joint as runner


ROOT=Path(__file__).resolve().parents[1]


def arguments():
    args=json.loads((ROOT/'reference_h0_599a410/argument_defaults.json').read_text())
    args.update(enable_highres_structure_stream=True,enable_h3_surface_fusion=True,
        enable_global_topology=True,remove_stage2_pre_topology_source=True,
        structure_profile='stage23_boundary_0626',stage_skeleton_mode='direct',
        stage2_skeleton_weight=.008,stage3_skeleton_weight=.012,
        stage_direction_factor=.2,stage_connectivity_factor=3.,surface_focal_gamma=1.,
        masked_connectivity_center_experiment=True)
    return args


def make_source():
    original=downstream()
    source={'modules':{name:canonical_state(getattr(original,name)) for name in
        ('visual','reliable_head','relations','proposal_head','accept_head','dynamics')},
        'gate_relation':original.gate_relation.state_dict(),'selector':SourceRelationSelector().state_dict()}
    q={'q_features':MultiScaleFeatures(170,32).state_dict(),'q_head':nn.Conv2d(32,1,1).state_dict()}
    return source,q


class FullH0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(1234)
        base,cls.saved=build_h0(arguments(),128)
        source,q=make_source();cls.model=FullH0Joint(base,q,source)
        cls.criterion=original_criterion(cls.saved,'cpu')

    def batch(self):
        image=torch.randn(2,3,128,128)
        mask=torch.zeros(2,1,128,128);mask[:,:,47:79,4:124]=1
        skeleton=torch.zeros_like(mask);skeleton[:,:,63,4:124]=1
        return {'image':image,'mask':mask,'skeleton':skeleton,'skeleton_dilate':torch.nn.functional.max_pool2d(skeleton,7,1,3)}

    def test_full_h0_retained_and_exact_A_B_initial_backbone(self):
        model=self.model.eval();core=model.base.swin_unet
        self.assertTrue(core.enable_global_topology)
        self.assertTrue(core.enable_highres_structure_stream)
        self.assertTrue(core.enable_h3_surface_fusion)
        self.assertTrue(all(str(k) in core.decoder_structure_blocks for k in (2,3)))
        self.assertIsNotNone(core.global_topology)
        self.assertTrue(any('decoder_structure_blocks.3.connectivity_head.' in k for k in model.base.state_dict()))
        batch=self.batch()
        with torch.no_grad():
            control=FullH0Control(model.base).eval()(batch['image'])
            joint=model(batch['image'])
        torch.testing.assert_close(control['z0'],joint['z0'],atol=0,rtol=0)
        self.assertLess(float((joint['p1']-joint['p0']).abs().mean()),.001)

    def test_final_segmentation_reaches_H0_Q_selector_and_live_states(self):
        model=self.model;model.train();model.zero_grad(set_to_none=True)
        batch=self.batch();result=model(batch['image'])
        result['states'][1].retain_grad()
        loss,_=dynamic_losses(result,batch['mask'],batch['skeleton'],lambda_cont=0.)
        for params in (tuple(model.base.parameters()),tuple(model.original.upstream.q_features.parameters()),tuple(model.selector.parameters())):
            params=tuple(p for p in params if p.requires_grad)
            self.assertGreater(gradient_norm(loss,params),0.)
        loss.backward()
        self.assertIsNotNone(result['states'][1].grad)
        for k in range(3):
            choice=result['source_choice'][:,k]
            self.assertEqual(float(choice[:,1:][~result['source_valid'][:,k]].sum()),0.)
            state=result['states'][k]
            transport=result['transport_rate'][:,k]*(choice[:,1:]*(result['source_states'][:,k]-state)).sum(1,keepdim=True)
            torch.testing.assert_close(transport,result['transport_delta'][:,k])
            torch.testing.assert_close(result['states'][k+1],state+transport+result['proposal_delta'][:,k]+result['anchor_delta'][:,k])
        self.assertTrue(all(p.grad is None for p in model.original.gate_relation.delta.parameters()))

    def test_original_criterion_is_used_once_and_no_forbidden_losses(self):
        model=self.model;model.train();model.zero_grad(set_to_none=True)
        batch=self.batch();result=model(batch['image'])
        loss,terms=h0_loss(self.criterion,result['h0_outputs'],batch)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertIn('stage_structure_loss',terms)
        text=(ROOT/'run_full_h0_joint.py').read_text()
        self.assertNotIn('late_choice_loss(',text)
        self.assertNotIn('gate_benefit_loss(',text)

    def test_checkpoint_strict_restore_optimizer_and_rng(self):
        model=FullH0Control(nn.Linear(3,1));optimizer=parameter_groups(model)
        args=SimpleNamespace(config={'variant':'test'})
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.full_h0_test_') as folder:
            path=Path(folder)/'last.pth'
            runner.save(args,model,optimizer,path,2,.7)
            other=FullH0Control(nn.Linear(3,1));opt=parameter_groups(other)
            state=runner.restore(args,other,path,opt)
            self.assertEqual(state['epoch'],2)
            torch.testing.assert_close(model.base.weight,other.base.weight,atol=0,rtol=0)
            bad=torch.load(path,weights_only=False);bad['model'].pop('base.weight');torch.save(bad,path)
            with self.assertRaises(RuntimeError):runner.restore(args,other,path)

    def test_optimizer_contains_H0_Q_and_repair_excludes_unused_G(self):
        model=self.model;opt=parameter_groups(model)
        ids={id(p) for group in opt.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in model.base.parameters() if p.requires_grad))
        self.assertTrue(all(id(p) in ids for p in model.original.upstream.q_head.parameters()))
        self.assertTrue(all(id(p) not in ids for p in model.original.gate_relation.delta.parameters()))
        self.assertTrue(all(id(p) not in ids for p in model.original.relations.direction_head.scalar.parameters()))

    def test_selector_changes_surface_and_old_scalar_G_does_not(self):
        model=self.model.eval();batch=self.batch();bias=model.selector.stay[-1].bias
        scalar=model.original.relations.direction_head.scalar.bias
        with torch.no_grad():
            before=model(batch['image'])['p1'];saved=bias.clone();old=scalar.clone()
            try:
                bias.add_(3.);after=model(batch['image'])['p1'];bias.copy_(saved)
                self.assertGreater(float((after-before).abs().mean()),0.)
                scalar.add_(100.);unused=model(batch['image'])['p1']
                torch.testing.assert_close(unused,before,atol=0,rtol=0)
            finally:bias.copy_(saved);scalar.copy_(old)

    def test_full_training_smoke_restores_BN_and_reports_actual_gradients(self):
        batch=self.batch()
        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self):return 2
            def set_epoch(self,epoch):pass
            def __getitem__(self,index):return {key:value[index] for key,value in batch.items()}
        buffers={name:value.clone() for name,value in self.model.named_buffers()}
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.full_h0_smoke_') as folder:
            args=SimpleNamespace(variant='B',seed=1234,device=torch.device('cpu'),output_dir=Path(folder),
                config={'lambda_relation':.1,'source_training':{'source_losses':{'lambda_cont':.2},'labels':asdict(RelationLabels())}})
            with patch.object(runner,'training_dataset',return_value=TinyDataset()),patch('builtins.print'):
                runner.smoke(args,self.model,self.criterion)
            report=json.loads((Path(folder)/'smoke.json').read_text())
            self.assertTrue(report['passed'])
            self.assertGreater(report['final_segmentation_gradients']['Q3'],0.)
        for name,value in self.model.named_buffers():torch.testing.assert_close(value,buffers[name],atol=0,rtol=0)

    def test_source_identity_uses_evaluation_record_and_exact_Q(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.full_h0_identity_') as folder:
            root=Path(folder);qpath=root/'q3.pth'
            torch.save({'repair_version':'dual_recovery_v1'},qpath)
            identity={'q3':{'path':str(qpath),'sha256':runner.digest(qpath)},'base':{'path':'old_local_gap_not_loaded'}}
            cfg={'epochs':20,'identity':identity}
            state={'version':runner.SOURCE_VERSION,'identity':identity,'config':cfg,'score':.812,'epoch':5}
            torch.save(state,root/'best.pth')
            (root/'experiment.json').write_text(json.dumps(cfg))
            (root/'training_complete.json').write_text(json.dumps({'epochs':20,'best_score':.812}))
            report={'reports':{'S_selector':{'own_val_threshold':{'threshold':.67}}}}
            (root/'comparison.json').write_text(json.dumps(report))
            _,_,meta=runner.inspect_source(root)
            self.assertEqual(meta['epoch'],5)
            report['reports']['S_selector']['own_val_threshold']['threshold']=.55
            (root/'comparison.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):runner.inspect_source(root)
            report['reports']['S_selector']['own_val_threshold']['threshold']=.67
            (root/'comparison.json').write_text(json.dumps(report))
            qpath.write_bytes(b'changed Q weights')
            with self.assertRaises(ValueError):runner.inspect_source(root)


if __name__=='__main__':unittest.main()
