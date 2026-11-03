import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from checks.test_full_h0_joint import arguments,make_source
from datasets.direct_candidate_data import CandidateCropDataset,choose_crop,reference_regions
from diagnostics.diagnose_full_h0_optimistic_reachability import signature
from networks.direct_candidate import DirectCandidateJoint,candidate_segmentation_loss,freeze_except_proposal
from networks.full_h0_joint import build_h0
from networks.reliable_directional_repair import dynamic_losses
from run_direct_candidate import choose_threshold_for_recall,frozen_digest,head_norm_difference
import run_direct_candidate as runner


class DirectCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_loss_matches_existing_segmentation_exactly(self):
        torch.manual_seed(5);z=torch.randn(2,1,32,32);p0=torch.rand(2,1,32,32)
        mask=(torch.rand(2,1,32,32)>.6).float();skel=torch.zeros_like(mask);skel[:,:,8,8:16]=1
        q=torch.rand(2,1,8,8)
        result={'b_logit':z,'p0':p0,'q':q}
        new,_=candidate_segmentation_loss(result,mask,skel)
        old,_=dynamic_losses({'z1':z,'p1':z.sigmoid(),'p0':p0,'q':q},mask,skel,lambda_cont=0.)
        torch.testing.assert_close(new,old,rtol=0,atol=0)
        low=torch.randn(2,1,8,8);result['b_logit']=low
        actual,_=candidate_segmentation_loss(result,mask,skel)
        up=F.interpolate(low,size=(32,32),mode='bilinear',align_corners=False)
        expected,_=dynamic_losses({'z1':up,'p1':up.sigmoid(),'p0':p0,'q':q},mask,skel,lambda_cont=0.)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_actual_model_logit_freeze_and_proposal_update(self):
        torch.manual_seed(7);base,_=build_h0(arguments(),128);source,q=make_source()
        model=DirectCandidateJoint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001)
        names=freeze_except_proposal(model)
        self.assertTrue(all(n.startswith('original.proposal_head.') for n in names))
        image=torch.randn(1,3,128,128);mask=(torch.rand(1,1,128,128)>.7).float();skel=torch.zeros_like(mask)
        before=frozen_digest(model);initial={n:p.detach().clone() for n,p in model.original.proposal_head.named_parameters()}
        with torch.no_grad():old=model(image,capture_diagnostics=True)
        optim=torch.optim.AdamW(model.original.proposal_head.parameters(),lr=2e-4)
        result=model(image);loss,_=candidate_segmentation_loss(result,mask,skel);loss.backward()
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('original.proposal_head.')))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.original.proposal_head.parameters()))
        optim.step()
        with torch.no_grad():new=model(image,capture_diagnostics=True)
        for key in ('p0','q','proposal_input'):torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        self.assertEqual(before,frozen_digest(model));self.assertGreater(head_norm_difference(model,initial),0.)
        torch.testing.assert_close(new['b_logit'].sigmoid(),new['b'],rtol=1e-6,atol=1e-7)

    def test_reference_regions_are_gt_restricted_and_missing_segments_counted(self):
        n=128;gt=np.zeros((n,n),bool);gt[60:69,10:118]=1
        skel=np.zeros_like(gt);skel[64,10:118]=1
        p0=np.zeros((n,n),np.float32);p0[60:69,10:35]=.9
        b=np.zeros_like(p0);p1=np.zeros_like(p0);p1[50:53,40:45]=.9
        values=reference_regions(p0,b,p1,gt,skel)
        self.assertGreater(len(values['missing_ids']),0)
        self.assertGreater(len(values['hard_points']),0)
        self.assertFalse(gt.reshape(-1)[values['hard_points']].any())
        self.assertTrue((values['missing_offsets'][1:]-values['missing_offsets'][:-1]>=3).all())

    def test_sampling_proportions_reproducibility_and_fallback(self):
        ref={'missing_ids':np.array([1]),'missing_offsets':np.array([0,1]),
             'missing_points':np.array([750*1500+750]),'hard_ids':np.array([2]),
             'hard_offsets':np.array([0,1]),'hard_points':np.array([700*1500+700]),'priority_ids':np.array([2])}
        choices=[choose_crop(ref,'hard',123,1,i) for i in range(2000)]
        counts={key:sum(x['requested']==key for x in choices) for key in ('random','missing','hard')}
        self.assertTrue(850<counts['random']<1150);self.assertTrue(400<counts['missing']<600);self.assertTrue(400<counts['hard']<600)
        self.assertEqual(choices[5],choose_crop(ref,'hard',123,1,5))
        empty=dict(ref,missing_ids=np.array([]),hard_ids=np.array([]),priority_ids=np.array([]))
        fallback=[choose_crop(empty,'hard',123,1,i) for i in range(50)]
        self.assertTrue(all(v['actual']=='random' for v in fallback))
        self.assertTrue(any(v['fallback'] for v in fallback))

    def test_recall_matching_reports_tie_error(self):
        threshold,error=choose_threshold_for_recall(np.array([.1,.5,.5,.9]),2)
        self.assertIsNotNone(threshold);self.assertLessEqual(abs(error),1)
        self.assertEqual(choose_threshold_for_recall(np.array([]),2),(None,None))

    def test_full_image_stitch_uses_logits(self):
        class Fixed(nn.Module):
            def forward(self,image):
                n,_,h,w=image.shape
                return {'z0':torch.full((n,1,h,w),-1.),'z1':torch.full((n,1,h,w),1.),
                        'b_logit':torch.full((n,1,h//4,w//4),2.)}
        sample={'case_name':'toy','image':torch.zeros(3,1500,1500)}
        with patch('builtins.print'):
            result=runner.stitch(Fixed(),sample,torch.device('cpu'))
        for name,logit in (('p0',-1.),('b',2.),('p1',1.)):
            np.testing.assert_allclose(result[name],torch.sigmoid(torch.tensor(logit)).item(),rtol=1e-5,atol=1e-6)

    def test_crop_dataset_alignment_reproducibility_and_cache_binding(self):
        class Full:
            image_files=['toy.tif']
            def __len__(self):return 1
            def __getitem__(self,index):
                mask=torch.zeros(1,1500,1500);mask[:,700:800,700:800]=1
                return {'case_name':'toy','image':torch.zeros(3,1500,1500),'mask':mask,'skeleton':mask.clone()}
        full=Full();sample=full[0];root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.candidate_data_') as temp:
            arrays={'missing_points':np.array([750*1500+750]),'missing_offsets':np.array([0,1]),
                    'missing_ids':np.array([1]),'hard_points':np.array([500*1500+500]),
                    'hard_offsets':np.array([0,1]),'hard_ids':np.array([2]),'priority_ids':np.array([2]),
                    'input_sha256':np.asarray(signature(sample))}
            np.savez_compressed(Path(temp)/'toy.npz',**arrays)
            with patch('datasets.direct_candidate_data.full_dataset',return_value=full):
                dataset=CandidateCropDataset('unused',temp,'hard',1234);dataset.set_epoch(4)
                first=dataset[0];second=dataset[0]
            torch.testing.assert_close(first['image'],second['image'],rtol=0,atol=0)
            torch.testing.assert_close(first['mask'],first['skeleton'],rtol=0,atol=0)
            self.assertEqual(first['requested'],second['requested'])
            self.assertEqual(dataset.available_components,{'missing':1,'hard':1})

    def test_one_epoch_head_only_training_and_resume(self):
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__();self.original=nn.Module();self.original.proposal_head=nn.Conv2d(1,1,1)
                self.original.frozen=nn.Conv2d(1,1,1)
            def forward(self,image,capture_diagnostics=False):
                f=self.original.frozen(image);z=self.original.proposal_head(f)
                result={'b_logit':z,'b':z.sigmoid(),'p0':torch.full_like(z,.2),'q':torch.full_like(z,.5)}
                if capture_diagnostics:result['proposal_input']=f
                return result
        class Dataset:
            available_components={'missing':2,'hard':2}
            def __init__(self,*args):pass
            def set_epoch(self,epoch):pass
            def __len__(self):return 2
            def __getitem__(self,i):
                return {'case_name':str(i),'image':torch.ones(1,16,16)*i,'mask':torch.ones(1,16,16),
                        'skeleton':torch.zeros(1,16,16),'requested':'missing','actual':'missing',
                        'fallback':False,'boundary_failed':False,'priority':False,'component':i+1}
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.candidate_train_') as temp:
            out=Path(temp);ref=out/'reference_train';ref.mkdir();(ref/'identity.json').write_text('{}')
            args=SimpleNamespace(output_dir=out,root_path=out,source_sha='abc',checkpoint=out/'source.pth',
                epochs=1,num_workers=0,seed=1234,lr=2e-4,device=torch.device('cpu'),code_sha={},probabilities=(.5,.25,.25))
            model=FakeModel();initial={k:v.clone() for k,v in model.state_dict().items()}
            fixed=torch.ones(1,1,16,16)
            with patch.object(runner,'require_reference',return_value=ref),patch.object(runner,'CandidateCropDataset',Dataset), \
                 patch.object(runner,'fixed_input',return_value=fixed),patch.object(runner,'smoke'), \
                 patch.object(runner,'val_iou',return_value=(.5,{'tp':1,'fp':1,'fn':0})),patch('builtins.print'):
                runner.train(args,model,'random')
                self.assertTrue((out/'random/best.pth').is_file())
                model.load_state_dict(initial);runner.train(args,model,'random')
            record=json.loads((out/'random/epoch_001.json').read_text())
            self.assertEqual(record['sampling']['success_missing'],2)
            self.assertGreater(record['proposal_parameter_change_l2'],0)


if __name__=='__main__':unittest.main()
