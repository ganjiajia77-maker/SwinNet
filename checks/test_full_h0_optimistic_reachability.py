"""Verify three-step max bounds, coordinate mapping, stitching and model reload."""
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
from diagnostics.diagnose_full_h0_optimistic_reachability import (
    digest,grid_gap,optimistic_sequence,rebuild_model,stitch_case,tile_gap,validate_saved_prediction,work_gap_state)
import diagnostics.diagnose_full_h0_optimistic_reachability as runner
from networks.full_h0_joint import VERSION,FullH0Joint,build_h0
from networks.reliable_directional_repair import compose_surface_logits


class FakeModel(nn.Module):
    def forward(self,image):
        z0=image[:,:1]
        size=image.shape[-1]//4
        p0=F.interpolate(z0.sigmoid(),size=(size,size),mode='bilinear',align_corners=False)
        proposal=(p0+.1).clamp(max=1.)
        states=[p0]
        for _ in range(3):
            mean=F.avg_pool2d(states[-1],3,1,1)
            states.append(.8*states[-1]+.1*mean+.1*proposal)
        z1,_=compose_surface_logits(z0,p0,states[-1])
        return {'z0':z0,'z1':z1,'p1':z1.sigmoid(),'u0':p0,'u3':states[-1],
                'states':states,'b':proposal}


class ReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_three_steps_and_boundaries_and_candidate(self):
        p0=torch.zeros(1,1,9,9);p0[0,0,4,4]=.8
        proposal=torch.zeros_like(p0);proposal[0,0,0,0]=.6
        states=optimistic_sequence(p0,proposal)
        self.assertEqual(len(states),4)
        self.assertAlmostEqual(float(states[1][0,0,0,0]),.6,places=6)
        self.assertEqual(float(states[1][0,0,4,6]),0.)
        self.assertAlmostEqual(float(states[3][0,0,4,7]),.8,places=6)
        self.assertEqual(float(states[3][0,0,4,8]),0.)
        for before,after in zip(states,states[1:]):
            self.assertTrue(bool((after>=before).all()))

    def test_upper_bounds_any_convex_source_and_proposal_updates(self):
        torch.manual_seed(3)
        p0=torch.rand(2,1,16,16);proposal=torch.rand_like(p0)
        upper=optimistic_sequence(p0,proposal)
        state=p0
        for step in range(1,4):
            neighbours=F.avg_pool2d(state,3,1,1)
            state=.45*state+.2*neighbours+.2*proposal+.15*p0
            self.assertTrue(bool((state<=upper[step]+1e-7).all()))
        z0=torch.randn(2,1,64,64)
        actual,_=compose_surface_logits(z0,p0,state)
        bound,_=compose_surface_logits(z0,p0,upper[-1])
        self.assertTrue(bool((actual<=bound+1e-6).all()))

    def test_mapping_flags_collisions_and_incomplete_tiles(self):
        gap={'id':1,'pixels':[8*32+8,8*32+9,8*32+10,8*32+11],
             'anchors':[[8*32+7],[8*32+12]]}
        local=tile_gap(gap,(32,32),0,0,32)
        mapped,reasons=grid_gap(local,32,8)
        self.assertIn('fewer_than_2_unique_gap_cells',reasons)
        self.assertEqual(len(mapped['pixels']),1)
        self.assertIsNone(tile_gap(gap,(32,32),8,8,16))

    def test_work_global_detour_does_not_count_as_local_connection(self):
        prediction=np.zeros((32,32),dtype=np.float32)
        prediction[16,4:28]=1.;prediction[16,15]=0.
        prediction[2:17,4]=1.;prediction[2:17,27]=1.;prediction[2,4:28]=1.
        gap={'pixels':np.asarray([16*32+x for x in range(10,22)]),
             'anchors':[np.asarray([16*32+x for x in range(7,10)]),
                        np.asarray([16*32+x for x in range(22,25)])]}
        labels=runner.component_data(prediction>.5)[1]
        state=work_gap_state(prediction,labels,gap,.5)
        self.assertTrue(state['closed_proxy'])
        self.assertFalse(state['local_closed_proxy'])

    def test_native_stitch_bound_and_per_tile_diagnostics(self):
        torch.manual_seed(5)
        image=torch.randn(1,3,64,64)
        gap={'id':2,'pixels':[20*64+x for x in range(18,26)],
             'anchors':[[20*64+x for x in range(14,17)],[20*64+x for x in range(27,30)]]}
        maps,rows,stats=stitch_case(FakeModel().eval(),image,{'gaps':[gap]},[.45],torch.device('cpu'),tile=32,stride=16)
        self.assertEqual(stats['tiles'],9)
        self.assertEqual(stats['work_actual_above_upper_pixels_by_step'],[0,0,0])
        self.assertGreater(len(rows),0)
        self.assertTrue(np.all(maps['upper3']>=maps['p1']-1e-6))
        np.testing.assert_allclose(maps['p0'],image[:,0].sigmoid().numpy()[0],rtol=1e-5,atol=1e-6)

    def test_invalid_true_dynamics_above_upper_is_rejected(self):
        class InvalidModel(FakeModel):
            def forward(self,image):
                result=super().forward(image)
                result['states'][1]=torch.ones_like(result['u0'])
                return result
        image=torch.zeros(1,3,64,64)
        with self.assertRaisesRegex(RuntimeError,'failed to upper-bound'):
            stitch_case(InvalidModel().eval(),image,{'gaps':[]},[.45],torch.device('cpu'),tile=32,stride=16)

    def test_wrong_saved_B_prediction_is_not_silently_accepted(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.optimistic_saved_') as temp:
            pair=Path(temp);folder=pair/'B/eval/val';folder.mkdir(parents=True)
            p0=np.zeros((4,4),np.float32);gt=np.zeros((4,4),bool)
            np.savez_compressed(folder/'case.npz',gt=gt,current_p0=p0,p1=p0)
            report=validate_saved_prediction(pair,'B','case',gt,{'p0':p0,'p1':p0})
            self.assertTrue(report['available'])
            with self.assertRaisesRegex(ValueError,'differs from saved B'):
                validate_saved_prediction(pair,'B','case',gt,{'p0':p0,'p1':p0+.01})

    def test_complete_joint_model_rebuilt_from_one_checkpoint(self):
        torch.manual_seed(1234)
        base,saved=build_h0(arguments(),128);source,q=make_source()
        original=FullH0Joint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001).eval()
        checkpoint={'version':VERSION,'config':{'variant':'B','h0_args':saved},'model':original.state_dict()}
        restored=rebuild_model(checkpoint,torch.device('cpu'),size=128)
        image=torch.randn(1,3,128,128)
        with torch.no_grad(): left,right=original(image),restored(image)
        for key in ('p0','p1','b','u0','u3'):
            torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)

    def test_full_entrypoint_and_resume_from_saved_case_without_model_forward(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.optimistic_main_') as temp:
            pair=Path(temp)/'pair';reference=pair/'reference';reference.mkdir(parents=True)
            run=pair/'B_transport_0p01_proposal_0p001';run.mkdir()
            n=1500;gt=np.zeros((n,n),dtype=bool);gt[200,100:160]=True
            gap={'id':1,'pixels':[200*n+x for x in range(120,132)],'length_bin':'8-15',
                 'anchors':[[200*n+x for x in range(117,120)],[200*n+x for x in range(132,135)]]}
            manifest={'count':1,'images':{'sample':{'shape':[n,n],'gaps':[gap]}}}
            manifest_path=reference/'fixed_gaps_and_anchors.json'
            manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
            np.savez_compressed(reference/'sample.npz',gt=gt)
            checkpoint={'version':VERSION,'epoch':5,'config':{'variant':'B',
                'reference_sha256':digest(manifest_path),'root_path':temp,'seed':1234},'model':{}}
            torch.save(checkpoint,run/'best.pth')
            sample={'case_name':'sample','image':torch.zeros(3,n,n),
                    'mask':torch.from_numpy(gt.astype(np.float32))[None]}
            output=Path(temp)/'output'
            argv=['audit','--device','cpu','--pair_dir',str(pair),'--output_dir',str(output),
                  '--expected_gaps','1','--expected_images','1']
            with patch.object(sys,'argv',argv),patch.object(runner,'verify_recorded_code',return_value={'B':1}), \
                 patch.object(runner,'rebuild_model',return_value=FakeModel().eval()), \
                 patch.object(runner,'full_dataset',return_value=[sample]),patch('builtins.print'):
                runner.main()
            report=json.loads((output/'summary.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['threshold_results']['0.45']['fixed_gaps'],1)
            self.assertTrue((output/'gap_upper_bound.csv').is_file())
            self.assertTrue((output/'tile_gap_diagnostics.csv').is_file())
            self.assertTrue((output/'cases/sample.npz').is_file())
            with patch.object(sys,'argv',argv),patch.object(runner,'verify_recorded_code',return_value={'B':1}), \
                 patch.object(runner,'rebuild_model',return_value=FakeModel().eval()), \
                 patch.object(runner,'full_dataset',return_value=[sample]), \
                 patch.object(runner,'stitch_case',side_effect=AssertionError('cache not reused')),patch('builtins.print'):
                runner.main()


if __name__=='__main__':unittest.main()
