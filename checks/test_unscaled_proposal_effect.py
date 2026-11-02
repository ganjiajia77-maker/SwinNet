import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from checks.test_full_h0_joint import arguments,make_source
from networks.full_h0_joint import VERSION,build_h0
from networks.full_h0_proposal_adapt import ProposalAdaptedJoint
from networks.reliable_directional_repair import compose_surface_logits
import diagnostics.diagnose_unscaled_proposal_effect as audit


class ProposalEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_restore_correct_unscaled_formula_and_capture_actual_rates(self):
        torch.manual_seed(123);base,saved=build_h0(arguments(),128);source,q=make_source()
        model=ProposalAdaptedJoint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001).eval()
        ckpt={'version':VERSION,'config':{'variant':'B','h0_args':saved},'model':model.state_dict()}
        restored=audit.build_unscaled(ckpt,torch.device('cpu'),size=128)
        image=torch.randn(1,3,128,128);rates=[]
        hook=restored.original.dynamics.rates.register_forward_hook(lambda m,i,o:rates.append(o.softmax(1)))
        try:
            with torch.no_grad():out=restored(image)
        finally:hook.remove()
        with torch.no_grad():expected=model(image)
        torch.testing.assert_close(out['p1'],expected['p1'],rtol=0,atol=0)
        for step in range(3):
            values=audit.fields_for_step(out,rates,step)
            np.testing.assert_allclose(values['proposal_coefficient'],
                .2*values['A']*values['rB'],rtol=1e-5,atol=1e-7)
            np.testing.assert_allclose(values['net_delta'],values['proposal_delta']+
                values['transport_delta']+values['anchor_delta'],rtol=1e-5,atol=1e-7)

    def test_signed_cancellation_and_small_value_quantiles(self):
        values={key:np.zeros((1,2),np.float32) for key in audit.FIELDS}
        values['B_minus_U'][:]=[.2,-.1]
        values['proposal_delta'][:]=[.0002,-.0001]
        values['positive_proposal'][:]=[.0002,0]
        values['positive_proposal_canceled'][:]=[.0001,0]
        stats=audit.Stats();stats.add(values,np.ones((1,2),bool))
        row=stats.result()
        self.assertEqual(row['cells'],2);self.assertAlmostEqual(row['proposal_delta_mean'],.00005,places=8)
        self.assertAlmostEqual(row['positive_proposal_canceled_amount_ratio'],.5,places=5)
        self.assertLess(abs(row['proposal_delta_p10_approx']),.001)
        clone=audit.Stats.load(stats.dump());combined=audit.Stats();combined.add_stats(clone)
        self.assertEqual(combined.result()['cells'],2)

    def test_regions_identify_missing_edge_and_background(self):
        gt=np.zeros((512,512),bool);gt[252:260,60:450]=True
        skeleton=np.zeros_like(gt);skeleton[256,60:450]=True
        p0=np.full((512,512),.1,np.float32);p0[252:260,60:180]=.9
        p1=p0.copy();p1[240:244,200:204]=.9
        q=np.ones((128,128),np.float32);ref=np.zeros_like(gt);ref[256,200:210]=True
        regions=audit.work_regions(gt,skeleton,p0,p1,q,ref,np.ones((128,128),bool),.45)
        self.assertGreater(int(regions['current_missed_GT_skeleton'].sum()),0)
        self.assertGreater(int(regions['road_edge_GT'].sum()),0)
        self.assertGreater(int(regions['Q_high_background_GT'].sum()),0)
        self.assertGreater(int(regions['fixed_reference_gap'].sum()),0)
        self.assertGreater(int(regions['new_FP_GT'].sum()),0)

    def test_ownership_covers_each_work_center_once(self):
        cover=np.zeros((375,375),np.uint8)
        for top in (0,256,512,768,988):
            for left in (0,256,512,768,988):
                owner=audit.owner_mask(top,left)
                ys=(top+np.arange(128)*4+2)//4;xs=(left+np.arange(128)*4+2)//4
                cover[np.ix_(ys,xs)]+=owner.astype(np.uint8)
        self.assertTrue((cover==1).all())

    def test_main_writes_and_resumes_case_cache(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.unscaled_effect_') as temp:
            pair=Path(temp)/'pair';reference=pair/'reference';reference.mkdir(parents=True)
            run=pair/'B_unscaled_same_initial_20e';(run/'eval').mkdir(parents=True)
            n=1500;gt=np.zeros((n,n),bool);gt[100,100:140]=True
            gap={'id':1,'pixels':[100*n+x for x in range(115,125)],'length_bin':'8-15',
                 'anchors':[[100*n+x for x in range(110,113)],[100*n+x for x in range(128,131)]]}
            manifest={'count':1,'images':{'sample':{'shape':[n,n],'gaps':[gap]}}}
            ref=reference/'fixed_gaps_and_anchors.json';ref.write_text(json.dumps(manifest),encoding='utf-8')
            np.savez_compressed(reference/'sample.npz',gt=gt)
            ckpt=run/'best.pth';torch.save({'epoch':5,'config':{'reference_sha256':audit.digest(ref),
                'root_path':temp,'seed':1234}},ckpt)
            (run/'eval/val_selection.json').write_text(json.dumps({'checkpoint_sha256':audit.digest(ckpt),
                'reference_sha256':audit.digest(ref),'threshold':.45}))
            sample={'case_name':'sample','image':torch.zeros(3,n,n),
                    'mask':torch.from_numpy(gt.astype(np.float32))[None]}
            predictions=np.zeros((2,n,n),np.float32)
            stats={(name,step):audit.Stats() for name in audit.REGIONS for step in range(3)}
            output=Path(temp)/'output'
            argv=['audit','--pair_dir',str(pair),'--output_dir',str(output),'--device','cpu',
                  '--expected_images','1','--expected_gaps','1']
            common=(patch.object(sys,'argv',argv),patch.object(audit,'verify_recorded_code'),
                    patch.object(audit,'verify_adaptation_code'),patch.object(audit,'build_unscaled',return_value=object()),
                    patch.object(audit,'full_dataset',return_value=[sample]),patch('builtins.print'))
            with common[0],common[1],common[2],common[3],common[4],common[5], \
                 patch.object(audit,'inspect_case',return_value=(predictions,stats,{},{})):
                audit.main()
            self.assertTrue(json.loads((output/'summary.json').read_text())['complete'])
            self.assertTrue((output/'fixed_gap_steps.csv').is_file())
            with patch.object(sys,'argv',argv),patch.object(audit,'verify_recorded_code'), \
                 patch.object(audit,'verify_adaptation_code'),patch.object(audit,'build_unscaled',return_value=object()), \
                 patch.object(audit,'full_dataset',return_value=[sample]),patch('builtins.print'), \
                 patch.object(audit,'inspect_case',side_effect=AssertionError('cache not used')):
                audit.main()

    def test_full_image_sliding_forward_and_actual_rate_capture(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.original=nn.Module();self.original.dynamics=nn.Module()
                self.original.dynamics.rates=nn.Conv2d(1,4,1)
                nn.init.zeros_(self.original.dynamics.rates.weight);nn.init.zeros_(self.original.dynamics.rates.bias)
            def forward(self,image):
                z0=image[:,:1];p0=F.avg_pool2d(z0.sigmoid(),4);b=(p0+.1).clamp(max=1);a=p0
                states=[p0];betas=[];proposals=[];anchors=[];transports=[]
                for _ in range(3):
                    rate=self.original.dynamics.rates(states[-1]).softmax(1)
                    beta=.2*a*rate[:,2:3];proposal=beta*(b-states[-1]);anchor=rate[:,3:4]*(p0-states[-1])
                    transport=torch.zeros_like(p0);states.append(states[-1]+proposal+anchor+transport)
                    betas.append(beta);proposals.append(proposal);anchors.append(anchor);transports.append(transport)
                z1,_=compose_surface_logits(z0,p0,states[-1])
                return {'z0':z0,'z1':z1,'p0':z0.sigmoid(),'p1':z1.sigmoid(),'u0':p0,'b':b,'a':a,
                        'q':p0,'positive_support':p0,'delta_z_plus':p0,'delta_z_minus':p0,
                        'states':states,'beta':torch.stack(betas,1),'proposal_delta':torch.stack(proposals,1),
                        'anchor_delta':torch.stack(anchors,1),'transport_delta':torch.stack(transports,1)}
        n=1500;gt=torch.zeros(1,n,n);gt[0,100:108,90:160]=1
        skel=torch.zeros_like(gt);skel[0,104,90:160]=1
        sample={'case_name':'toy','image':torch.zeros(3,n,n),'mask':gt,'skeleton':skel}
        gap={'id':1,'pixels':[104*n+x for x in range(110,125)],'length_bin':'8-15',
             'anchors':[[104*n+x for x in range(100,104)],[104*n+x for x in range(130,134)]]}
        with patch('builtins.print'):
            predictions,stats,gaps,change=audit.inspect_case(Toy().eval(),sample,
                {'shape':[n,n],'gaps':[gap]},.5,torch.device('cpu'))
        self.assertEqual(predictions.shape,(2,n,n))
        self.assertGreater(stats[('fixed_reference_gap',0)].n,0)
        self.assertIn((1,2),gaps)
        self.assertTrue(np.isfinite(predictions).all())


if __name__=='__main__':unittest.main()
