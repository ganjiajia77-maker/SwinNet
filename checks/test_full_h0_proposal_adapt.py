import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from checks.test_full_h0_joint import arguments,make_source
from networks.full_h0_joint import FullH0Joint,build_h0,VERSION
from networks.full_h0_proposal_adapt import ProposalAdaptedJoint
from run_full_h0_proposal_adapt import build_same_initial_models,region_metrics
import run_full_h0_proposal_adapt as experiment
import compare_full_h0_proposal_adapt as comparison


class ProposalAdaptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_original_formula_matches_and_new_proposal_receives_gradient(self):
        torch.manual_seed(44);base,saved=build_h0(arguments(),128);source,q=make_source()
        model=ProposalAdaptedJoint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001).eval()
        image=torch.randn(1,3,128,128)
        with torch.no_grad():
            reference=FullH0Joint.forward(model,image)
            model.proposal_mode='original';old=model(image)
            for key in ('p0','p1','b','a','w','r','u3','transport_rate','source_choice'):
                torch.testing.assert_close(old[key],reference[key],rtol=0,atol=0)
            model.proposal_mode='unscaled';new=model(image)
            torch.testing.assert_close(new['p0'],old['p0'],rtol=0,atol=0)
            self.assertTrue(bool((new['b']>=old['b']-1e-7).all()))
            z=torch.logit(new['u0'].clamp(1e-5,1-1e-5))
            torch.testing.assert_close(new['b'],(z+new['delta_z_plus']-new['delta_z_minus']).sigmoid())
        model.zero_grad(set_to_none=True)
        out=model(image);loss=out['p1'].square().mean()
        grads=torch.autograd.grad(loss,tuple(model.original.proposal_head.parameters()),allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads if g is not None),0.)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads if g is not None))

    def test_reconstructs_identical_pretraining_state_from_recorded_sources(self):
        root=Path(__file__).resolve().parents[1]
        source,q=make_source();source['config']={'test':1}
        _,saved=build_h0(arguments(),128)
        with tempfile.TemporaryDirectory(dir=root,prefix='.proposal_init_') as temp:
            path=Path(temp);h0=path/'h0.pth';h0.write_bytes(b'h0')
            repair=path/'repair';repair.mkdir();(repair/'best.pth').write_bytes(b'repair')
            previous={'rate_initialization':'explicit_softmax_budgets','h0_checkpoint':str(h0),'h0_sha256':'h0hash',
                      'repair':{'checkpoint':str(repair/'best.pth'),'sha256':'repairhash'},
                      'seed':1234,'initial_transport_rate':.01,'initial_proposal_rate':.001,
                      'h0_args':saved,'source_training':source['config']}
            def fake_load(path,device):
                base,args=build_h0(arguments(),128)
                return base,None,args
            with patch.object(experiment,'digest',return_value='h0hash'), \
                 patch.object(experiment,'load_h0',side_effect=fake_load), \
                 patch.object(experiment,'inspect_source',return_value=(source,q,{'sha256':'repairhash','q3':{}})):
                model,proof=build_same_initial_models(previous,torch.device('cpu'))
            self.assertEqual(len(proof['reconstructed_initial_state_sha256']),64)
            self.assertEqual(proof['matching_tensors'],len(model.state_dict()))
            rates=model.original.dynamics.rates[-1].bias.softmax(0)
            self.assertAlmostEqual(float(rates[1]),.01,places=6)
            self.assertAlmostEqual(float(rates[2]),.001,places=6)

    def test_region_counts_and_old_ceiling_are_finite(self):
        p=torch.full((1,1,2,2),.02);support=torch.full_like(p,.05)
        result={'u0':p,'positive_support':support,'delta_z_plus':torch.full_like(p,8),
                'delta_z_minus':torch.zeros_like(p),'b':torch.ones_like(p),'q':torch.ones_like(p)}
        batch={'mask':torch.tensor([[[[1.,0.],[0.,0.]]]]),'skeleton':torch.tensor([[[[1.,0.],[0.,0.]]]])}
        result=region_metrics(result,batch)
        self.assertEqual(result['missed_skeleton_proxy']['pixels'],1)
        self.assertEqual(result['missed_skeleton_proxy']['old_ceiling_below_045'],1)
        self.assertEqual(result['q_high_background_proxy']['pixels'],3)

    def test_smoke_checks_pretraining_and_preserves_buffers(self):
        torch.manual_seed(31);base,_=build_h0(arguments(),128);source,q=make_source()
        model=ProposalAdaptedJoint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001)
        class Dataset:
            def set_epoch(self,epoch):pass
            def __len__(self):return 2
            def __getitem__(self,i):
                return {'image':torch.full((3,128,128),float(i)*.2),
                        'mask':torch.ones(1,128,128),'skeleton':torch.zeros(1,128,128)}
        def loss(args,model,criterion,result,batch,seed):return result['p1'].square().mean(),{}
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.proposal_smoke_') as temp:
            args=SimpleNamespace(seed=1234,device=torch.device('cpu'),output_dir=Path(temp),
                                 pair_dir=Path(temp),source_run='B_old')
            before={name:v.clone() for name,v in model.named_buffers()}
            with patch.object(experiment,'training_dataset',return_value=Dataset()), \
                 patch.object(experiment,'batch_loss',side_effect=loss):
                experiment.smoke(args,model,None)
            report=json.loads((Path(temp)/'smoke.json').read_text())
            self.assertTrue(report['passed']);self.assertGreater(report['final_segmentation_gradients']['proposal'],0)
            for name,value in model.named_buffers():torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_three_model_comparison_checks_shared_reference(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.proposal_cmp_') as temp:
            pair=Path(temp);ref=pair/'reference';ref.mkdir()
            manifest={'count':14,'images':{f'c{i}':{'gaps':[{'id':1,'length_bin':'3-7'}]} for i in range(14)}}
            refpath=ref/'fixed_gaps_and_anchors.json';refpath.write_text(json.dumps(manifest))
            sha=comparison.digest(refpath);folders={}
            for name in ('A','B_old','B_new'):
                folder=pair/name;folder.mkdir();(folder/'best.pth').write_bytes(name.encode())
                ev=folder/'eval/val';ev.mkdir(parents=True)
                (folder/'eval/val_selection.json').write_text(json.dumps({'checkpoint_sha256':comparison.digest(folder/'best.pth'),
                    'reference_sha256':sha,'threshold':.5}))
                for i in range(14):
                    p=np.full((4,4),.6,np.float32);gt=np.ones((4,4),bool)
                    np.savez_compressed(ev/f'c{i}.npz',gt=gt,p0=p,current_p0=p,p1=p)
                folders[name]=folder
            (folders['B_new']/'experiment.json').write_text(json.dumps({'proposal_adaptation':{
                'source_B_sha256':comparison.digest(folders['B_old']/'best.pth'),
                'initialization_mode':'same_reconstructed_pretraining_state_as_old_B'}}))
            def fake_metrics(cases,name,threshold,manifest):
                return {'threshold':float(threshold),'metrics':{key:1. for key in ('iou','f1','precision','recall','cldice',
                    'apls_gt_to_pred_approx','apls_pred_to_gt_approx','apls_bidirectional_approx','short_pred_components')},
                    'closed':14,'bridge_proxy':0,'closed_ids':{case['case']:[1] for case in cases},
                    'gap_by_length':{'3-7':{'eligible':14,'closed':14}}}
            with patch.object(comparison,'point_metrics',side_effect=fake_metrics), \
                 patch.object(comparison,'exact_recall_threshold',return_value=(.5,0)), \
                 patch.object(comparison,'area_threshold',return_value=(.5,0)):
                comparison.compare(pair,'B_new','B_old')
            self.assertTrue((folders['B_new']/'comparison/REPORT.md').is_file())
            self.assertTrue((folders['B_new']/'comparison/fixed_gap_changes.csv').is_file())

    def test_one_epoch_train_and_resume_with_control_input_hashes(self):
        class Dataset:
            def set_epoch(self,epoch):pass
            def __len__(self):return 2
            def __getitem__(self,i):
                return {'case_name':str(i),'image':torch.full((1,2,2),float(i)),
                        'mask':torch.ones(1,2,2),'skeleton':torch.ones(1,2,2)}
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__();self.original=torch.nn.Module();self.original.visual=torch.nn.Linear(1,1)
            def forward(self,image):return {'z':self.original.visual(image.mean((1,2,3)).unsqueeze(1))}
        def loss(args,model,criterion,result,batch,seed):
            v=result['z'].square().mean()
            return v,{'transport_abs_mean':v.detach(),'proposal_abs_mean':v.detach(),'P1_minus_P0_abs_mean':v.detach()}
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.proposal_train_') as temp:
            pair=Path(temp);control=pair/'A';control.mkdir();previous=pair/'B_old';previous.mkdir();out=pair/'B_test';out.mkdir()
            seed=1234;dataset=Dataset();order=hashlib.sha256();images=hashlib.sha256()
            loader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,num_workers=0,
                generator=torch.Generator().manual_seed(seed+1000003))
            for batch in loader:
                for name in batch['case_name']:order.update(str(name).encode()+b'\0')
                images.update(batch['image'].contiguous().numpy().tobytes())
            (control/'epoch_001.json').write_text(json.dumps({'case_order_sha256':order.hexdigest(),
                'augmented_images_sha256':images.hexdigest()}))
            (previous/'epoch_001.json').write_text((control/'epoch_001.json').read_text())
            args=SimpleNamespace(seed=seed,epochs=1,device=torch.device('cpu'),num_workers=0,
                                 output_dir=out,pair_dir=pair,source_run='B_old',manifest={},config={'version':'test'})
            model=Fake();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            # The scheduler expects the same LR metadata as the real optimizer.
            for group in optimizer.param_groups:group.update(initial_lr=.001,minimum_lr=.0001)
            with patch.object(experiment,'training_dataset',return_value=dataset), \
                 patch.object(experiment,'batch_loss',side_effect=loss), \
                 patch.object(experiment,'region_metrics',return_value={}), \
                 patch.object(experiment,'validation',return_value=[]), \
                 patch.object(experiment,'point_metrics',return_value={'selection_score':.5}), \
                 patch.object(experiment,'validation_repair_activity',return_value={}):
                experiment.train(args,model,None,optimizer)
                self.assertTrue((out/'last.pth').is_file());self.assertTrue((out/'best.pth').is_file())
                experiment.train(args,model,None,optimizer)
            self.assertEqual(json.loads((out/'training_complete.json').read_text())['epochs'],1)


if __name__=='__main__':unittest.main()
