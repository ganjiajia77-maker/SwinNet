import unittest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from checks.test_full_h0_joint import arguments, make_source
from networks.direct_candidate import DirectCandidateJoint
from networks.full_h0_direct_joint import enable_joint_training, joint_loss
from networks.full_h0_joint import build_h0, original_criterion, h0_loss
from networks.v9_source_relation_labels import RelationLabels
from networks.v9_selector_transport_training import gradient_norm
from run_full_h0_direct_joint import make_optimizer
import run_full_h0_direct_joint as runner


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_model_objective_and_gradient_paths(self):
        torch.manual_seed(17);base,args=build_h0(arguments(),128);source,q=make_source()
        model=DirectCandidateJoint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001)
        enable_joint_training(model);model.eval();criterion=original_criterion(args,'cpu')
        image=torch.randn(2,3,128,128);mask=torch.zeros(2,1,128,128);mask[:,:,47:79,4:124]=1
        skeleton=torch.zeros_like(mask);skeleton[:,:,63,4:124]=1
        batch={'image':image,'mask':mask,'skeleton':skeleton,
               'skeleton_dilate':torch.nn.functional.max_pool2d(skeleton,7,1,3)}
        result=model(image);loss,terms=joint_loss(model,criterion,result,batch,RelationLabels(),.1,123)
        original,_=h0_loss(criterion,result['h0_outputs'],batch)
        expected=original-criterion.surface_loss(result['z0'],mask)[0]+.5*terms['seg_p0']+.5*terms['seg_p1']+.25*terms['seg_b']+.2*terms['cont']+.1*terms['relation']
        torch.testing.assert_close(loss,expected)
        self.assertGreater(gradient_norm(terms['seg_b'],tuple(model.original.proposal_head.parameters())),0.)
        self.assertGreater(gradient_norm(terms['seg_p1'],tuple(model.selector.parameters())),0.)
        self.assertGreater(gradient_norm(terms['seg_p1'],tuple(model.original.dynamics.rates[-1].parameters())),0.)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.base.parameters()))
        self.assertTrue(all(p.grad is None for p in model.original.gate_relation.delta.parameters()))
        optimizer=make_optimizer(model,1e-5,1e-4)
        for group in optimizer.param_groups:
            self.assertEqual(group['initial_lr'],1e-5 if group['name']=='H0' else 1e-4)

    def test_initialization_keeps_pretrained_repair_but_restores_A_H0(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.original=nn.Module();self.original.upstream=nn.Module()
                self.original.upstream.base=nn.Linear(1,1,bias=False)
                self.original.upstream.q_head=nn.Linear(1,1,bias=False)
                self.original.gate_relation=nn.Module();self.original.gate_relation.delta=nn.Linear(1,1)
                self.original.relations=nn.Module();self.original.relations.direction_head=nn.Module()
                self.original.relations.direction_head.scalar=nn.Linear(1,1)
                self.original.proposal_head=nn.Linear(1,1)
            @property
            def base(self):return self.original.upstream.base
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            pair=Path(temp);(pair/'reference').mkdir();(pair/'A').mkdir();(pair/'source/eval').mkdir(parents=True)
            (pair/'reference/fixed_gaps_and_anchors.json').write_text(json.dumps({'count':198,'images':{}}))
            (pair/'source/eval/val_selection.json').write_text(json.dumps({'checkpoint_sha256':'digest'}))
            h0=pair/'original_h0.pth';h0.touch();toy=Toy();toy.base.weight.data.fill_(3.)
            pretrained=toy.original.proposal_head.weight.detach().clone()
            config={'epochs':20,'h0_checkpoint':str(h0),'h0_sha256':'digest','h0_args':{},'reference_sha256':'digest',
                    'seed':1234,'h0_lr':1e-5,'root_path':str(pair),'num_workers':0}
            source=pair/'source/best.pth';torch.save({'model':toy.state_dict(),'config':config,'epoch':5},source)
            initial=nn.Linear(1,1,bias=False);initial.weight.data.fill_(2.)
            with patch.object(runner,'digest',return_value='digest'),patch.object(runner,'verify_recorded_code'), \
                 patch.object(runner,'verify_adaptation_code'),patch.object(runner,'check_reusable_A'), \
                 patch.object(runner,'build_model',return_value=toy),patch.object(runner,'load_h0',return_value=(initial,{},{})), \
                 patch.object(runner,'original_criterion',return_value=None):
                args,model,_,repair_lr=runner.configure(pair,source,'B_test',torch.device('cpu'))
            self.assertEqual(float(model.base.weight.detach()),2.)
            torch.testing.assert_close(model.original.proposal_head.weight,pretrained)
            self.assertEqual(repair_lr,1e-4)
            self.assertEqual(args.config['direct_joint']['source_epoch'],5)

    def test_evaluation_writes_paired_and_local_gap_results(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            pair=Path(temp);out=pair/'B_test';(pair/'A/eval/val').mkdir(parents=True);out.mkdir()
            (pair/'A/eval/val_selection.json').write_text(json.dumps({'threshold':.5}))
            (out/'best.pth').touch();gt=np.zeros((8,8),bool);gt[3,3:5]=True
            p0=np.full((8,8),.1,dtype=np.float32);p1=p0.copy();p1[3,3:5]=.9
            np.savez_compressed(pair/'A/eval/val/toy.npz',gt=gt,p0=p0,p1=p0)
            case={'case':'toy','gt':gt,'p0':p0,'current_p0':p0,'p1':p1,'b':p1,'eval_skeleton':gt}
            manifest={'images':{'toy':{'gaps':[{'id':1,'length_bin':'3-7','pixels':[27,28],'anchors':[[27],[28]]}]}}}
            args=SimpleNamespace(pair_dir=pair,output_dir=out,manifest=manifest,config={'reference_sha256':'sha','selection_rule':'rule'})
            metrics={'iou':.5,'precision':.5,'recall':.5,'cldice':.5,'apls_bidirectional_approx':.5,'short_pred_components':1.}
            def point(cases,name,threshold,manifest):
                return {'threshold':float(threshold),'selection_score':.5,'metrics':metrics,'closed':0,'bridge_proxy':0,
                        'closed_ids':{'toy':[]},'gap_by_length':{}}
            def state(mask,labels,gap,shape):
                closed=bool(mask[3,3]);return {'coverage':float(closed),'local_closed_proxy':closed}
            with patch.object(runner,'restore'),patch.object(runner,'val_cases',return_value=([case],[])), \
                 patch.object(runner,'point_metrics',side_effect=point),patch.object(runner,'exact_recall_threshold',return_value=(.5,0)), \
                 patch.object(runner,'candidate_metrics',return_value={}),patch.object(runner,'component_data',return_value=(0,None)), \
                 patch.object(runner,'gap_state',side_effect=state),patch('builtins.print'):
                runner.evaluate(args,None)
            summary=json.loads((out/'eval/local_gap_summary.json').read_text())
            self.assertEqual(summary['newly_closed'],1)
            self.assertTrue((out/'eval/comparison.csv').is_file())
            self.assertTrue((out/'eval/REPORT.md').is_file())


if __name__=='__main__':unittest.main()
