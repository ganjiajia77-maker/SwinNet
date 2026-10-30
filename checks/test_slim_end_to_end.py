"""Real Swin backbone integration; use 128 inputs locally, 512 on server smoke."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from networks.slim_end_to_end import build_model,end_to_end_losses,prune_structure
import run_slim_end_to_end_mass as runner


class TinyDataset(Dataset):
    def __len__(self): return 2
    def set_epoch(self,epoch): pass
    def __getitem__(self,index):
        image=torch.linspace(0,1,3*128*128).reshape(3,128,128)+index*.01
        mask=torch.zeros(1,128,128); mask[:,46:80,4:124]=1
        skeleton=torch.zeros_like(mask); skeleton[:,62,4:124]=1
        return {'image':image,'mask':mask,'skeleton':skeleton,'case_name':str(index)}


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(1234); cls.model=build_model(128)

    def test_removed_modules_and_retained_backbone(self):
        core=self.model.original.upstream.backbone.segmenter.swin_unet
        self.assertEqual([len(layer.blocks) for layer in core.layers],[2,2,6,2])
        self.assertEqual([len(layer.blocks) for layer in core.layers_up[1:]],[6,2,2])
        self.assertEqual(len(core.dca_blocks),2)
        self.assertEqual(len(core.decoder_structure_blocks),0)
        self.assertIsNone(core.prepatch_structure_encoder)
        self.assertIsNone(core.highres_skeleton_adapter)
        self.assertIsNone(core.gap_connection)
        forbidden=('skeleton','connectivity','road_attention','global_topology','prepatch_structure','highres_structure','state_gate','gate_relation.delta')
        self.assertFalse([name for name,_ in self.model.named_parameters() if any(part in name for part in forbidden)])
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_actual_backbone_gradients_and_shapes(self):
        model=self.model; model.train(); model.zero_grad(set_to_none=True)
        batch=[TinyDataset()[i] for i in (0,1)]
        image,mask,skeleton=(torch.stack([sample[key] for sample in batch]) for key in ('image','mask','skeleton'))
        result=model(image,capture_diagnostics=True)
        self.assertEqual(result['feature_shapes']['detail'],[2,96,32,32])
        self.assertEqual(result['feature_shapes']['semantic'],[2,192,16,16])
        self.assertEqual(tuple(result['z1'].shape),(2,1,128,128))
        self.assertTrue(result['z0'].requires_grad)
        total,terms=end_to_end_losses(result,mask,skeleton)
        self.assertTrue(torch.isfinite(total))
        total.backward(); norms=runner.group_norms(model)
        self.assertTrue(all(value>0 for value in norms.values()),norms)
        for k in range(3):
            expected=result['states'][k]+result['transport_delta'][:,k]+result['proposal_delta'][:,k]+result['anchor_delta'][:,k]
            torch.testing.assert_close(expected,result['states'][k+1])
            self.assertTrue(bool(((expected>=-1e-6)&(expected<=1+1e-6)).all()))
        self.assertEqual(float(result['source_choice'][:,:,1:][~result['source_valid']].detach().abs().sum()),0)

    def test_real_batch_smoke_restores_batchnorm_buffers(self):
        model=self.model; model.zero_grad(set_to_none=True)
        optimizer,_=runner.optimizer_for(model,set())
        before={name:buf.clone() for name,buf in model.named_buffers()}
        with tempfile.TemporaryDirectory() as folder:
            args=SimpleNamespace(device=torch.device('cpu'),seed=1234,output_dir=Path(folder),config={'lambda_q':.1,'lambda_choice':.05,'lambda_p0':1.})
            with patch.object(runner,'training_dataset',return_value=TinyDataset()):
                runner.smoke(args,model,optimizer)
            self.assertTrue((Path(folder)/'smoke.json').is_file())
        for name,value in model.named_buffers(): torch.testing.assert_close(value,before[name],atol=0,rtol=0)

    def test_checkpoint_is_self_contained_and_strict(self):
        # Small module exercises the exact full-state/optimizer serialization path.
        model=nn.Linear(4,2); optimizer=torch.optim.AdamW(model.parameters())
        model(torch.ones(2,4)).sum().backward(); optimizer.step()
        args=SimpleNamespace(config={'version':'test','no_upstream_dependencies':True})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pth'
            runner.atomic_checkpoint(path,runner.checkpoint(model,optimizer,args,2,.5,True))
            restored=nn.Linear(4,2); state=runner.load_weights(restored,args,path)
            self.assertIn('optimizer',state); self.assertNotIn('base_checkpoint',state); self.assertNotIn('q3_checkpoint',state)
            torch.testing.assert_close(model.weight,restored.weight,atol=0,rtol=0)
            with self.assertRaises(RuntimeError): runner.load_weights(nn.Linear(5,2),args,path)

    def test_unchanged_lr_schedule_and_matched_area_ties(self):
        optimizer,_=runner.optimizer_for(nn.Linear(4,2),set())
        runner.schedule(optimizer,0,80); self.assertAlmostEqual(optimizer.param_groups[1]['lr'],2e-5)
        runner.schedule(optimizer,9,80); self.assertAlmostEqual(optimizer.param_groups[1]['lr'],2e-4)
        runner.schedule(optimizer,79,80); self.assertAlmostEqual(optimizer.param_groups[1]['lr'],1e-5)
        cases=[{'p1':np.array([.1,.2,.2,.7,.9],np.float32)}]
        threshold,error=runner.area_threshold(cases,'p1',2)
        self.assertEqual(int((cases[0]['p1']>=threshold).sum()),2);self.assertEqual(error,0)


if __name__=='__main__': unittest.main()
