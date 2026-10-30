"""Read-only matching and same-checkpoint source intervention checks."""
import tempfile
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from checks.test_v9_state_gate import model as fake_model
from networks.v9_multiscale_transport import install
from diagnostics.v9_multiscale_fair_and_relay import choose_points, infer, set_mode
import diagnostics.v9_multiscale_fair_and_relay as diagnostic
from evaluation.road_comparison.road_metrics import prepare_target
import run_v9_multiscale as runner


class FairRelayTests(unittest.TestCase):
    def test_val_matching_and_freeze_uses_multiscale_threshold(self):
        gt=np.array([[1,1,0,0]],bool)
        caches=[{'gt':gt,'local':np.array([[.9,.6,.4,.1]]),
                 'multiscale':np.array([[.8,.7,.6,.2]])}]
        points=choose_points(caches,{'local':.5,'multiscale':.65})
        self.assertEqual(points['matching']['recall_tp_residual'],0)
        self.assertEqual(points['matching']['area_pixels_residual'],0)
        for key in ('own_val','matched_recall','matched_area'):
            self.assertEqual(points[key]['multiscale'],points[key]['multiscale_fixed_sources'])
        self.assertEqual(points['own_val']['local'],.5)

    def test_same_checkpoint_relay_and_gap_trace(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        model=install(fake_model(), 'multiscale').eval()
        state={'modules':{name:getattr(model,name).state_dict() for name in runner.NAMES}}
        sample={'image':torch.randn(3,512,512),'mask':torch.zeros(1,512,512)}
        points=np.array([[250,250],[12,20],[500,500]],np.int64)
        set_mode(model,state,'multiscale')
        with torch.inference_mode():
            # Amplify learned transport without changing the intervention protocol.
            model.dynamics.selector.stay[-1].bias.fill_(-2.)
            model.dynamics.rates[-1].bias[1]=5.
            model.proposal_head[-1].bias[0]=4.
            model.state_gate.net[-1].bias.fill_(2.)
        state={'modules':{name:{k:v.clone() for k,v in getattr(model,name).state_dict().items()} for name in runner.NAMES}}
        set_mode(model,state,'multiscale'); on, a=infer(model,sample,torch.device('cpu'),False,points)
        set_mode(model,state,'multiscale_fixed_sources'); off, b=infer(model,sample,torch.device('cpu'),True,points)
        np.testing.assert_array_equal(on[0],off[0])
        self.assertEqual(a['u'].shape,(3,3))
        self.assertTrue(all(np.isfinite(value).all() for value in a.values()))
        for key in ('u','m','alpha','beta','gamma','transport_delta','proposal_delta','anchor_delta'):
            np.testing.assert_allclose(a[key][0],b[key][0],rtol=0,atol=0)
        # Relay only changes later steps. Distinguish legitimate zero-effect cases.
        self.assertGreater(float(np.abs(a['available'][1:]-b['available'][1:]).max()),0.)

    def test_end_to_end_val_and_reuse_csv(self):
        torch.set_num_threads(1); torch.manual_seed(1234)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp); run=root/'run'; run.mkdir()
            source=root/'source.pth'
            payload={'identity':{},'training_config':{'reference_threshold':.45,'seed':1234,'batch_size':2}}
            torch.save(payload,source)
            sample={'case_name':'case_a','image':torch.randn(3,512,512),'mask':torch.zeros(1,512,512)}
            sample['mask'][:,250:253,20:490]=1.
            gt=sample['mask'][0].numpy()>.5
            model=fake_model().eval(); selections={}; configs={}; p0=None
            for mode in ('local','multiscale'):
                install(model,mode); model.eval()
                config={'mode':mode,'source_sha256':runner.v9.digest(source),'identity':{}}
                configs[mode]=config; out=run/mode; out.mkdir()
                optimizer=runner.configure(model,{name:1e-4 for name in runner.v9.STAGE_MODULES['D']},1e-4)
                runner.save(out/'best.pth',model,payload,config,optimizer,1,.5,{})
                pred,_=infer(model,sample,torch.device('cpu'))
                p0=pred[0]; binary=pred[1]>=.45
                counts={'tp':int((binary&gt).sum()),'fp':int((binary&~gt).sum()),'fn':int((~binary&gt).sum())}
                selection={'selected_on':'val','threshold':.45,'checkpoint_sha256':runner.v9.digest(out/'best.pth'),
                           'config':config,'report':{'p1':counts}}
                runner.write_json(out/'val_selection.json',selection)
            old=run/'old_audit'; old.mkdir()
            finger=hashlib.sha256()
            for value in (b'case_a\0',np.asarray(gt.shape,dtype=np.int64).tobytes(),
                    *(np.packbits(mask,axis=None).tobytes() for mask in
                      (p0>=.45,gt,prepare_target(gt,diagnostic.PROTOCOL)['skeleton']))): finger.update(value)
            runner.write_json(old/'summary.json',{'source_sha256':runner.v9.digest(source),
                'reference_set':{'sha256':finger.hexdigest()},'same_threshold':.45,
                'own_val_thresholds':{'local':.45,'multiscale':.45}})
            keys=['step1_generated_fp_weight_mean','step2_generated_fp_weight_mean','step3_generated_fp_weight_mean',
                  'same_generated_fp_source_used_twice_fraction','background_bridge_proxy','parallel_background_bridge_proxy']
            with (old/'generated_source_reuse_and_bridges.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['case','mode',*keys]); writer.writeheader()
                for mode in ('local','multiscale'): writer.writerow({'case':'case_a','mode':mode,**{k:0 for k in keys}})
            (run/'latest_audit_val.txt').write_text(str(old))
            output=root/'output_val'
            argv=['diagnostic','--run',str(run),'--source',str(source),'--data',str(root),
                  '--output',str(output),'--device','cpu','--split','val']
            with patch.object(sys,'argv',argv), patch.object(diagnostic.experiment,'original',return_value=(model,payload)), \
                 patch.object(diagnostic,'full_dataset',return_value=[sample]):
                diagnostic.main()
            report=json.loads((output/'summary.json').read_text())
            self.assertTrue(report['previous_reuse_csv_loaded'])
            self.assertEqual(len(report['metrics']),9)
            self.assertTrue((output/'long_gap_pixels.csv').is_file())
            self.assertTrue((run/'fair_operating_points_val.json').is_file())
            points=report['points']
            output_test=root/'output_test'
            argv_test=['diagnostic','--run',str(run),'--source',str(source),'--data',str(root),
                       '--output',str(output_test),'--device','cpu','--split','test']
            with patch.object(sys,'argv',argv_test), patch.object(diagnostic.experiment,'original',return_value=(model,payload)), \
                 patch.object(diagnostic,'full_dataset',return_value=[sample]):
                diagnostic.main()
            test_report=json.loads((output_test/'summary.json').read_text())
            self.assertEqual(test_report['points'],points)
            self.assertFalse(test_report['previous_reuse_csv_loaded'])
            # Mismatched val checkpoint IDs must fail before any test inference.
            selection_path=run/'fair_operating_points_val.json'
            selected=json.loads(selection_path.read_text()); selected['checkpoint_sha256']['local']='wrong'
            selection_path.write_text(json.dumps(selected))
            argv_test[argv_test.index(str(output_test))]=str(root/'invalid_test')
            with patch.object(sys,'argv',argv_test), patch.object(diagnostic.experiment,'original',return_value=(model,payload)), \
                 patch.object(diagnostic,'full_dataset',return_value=[sample]), \
                 patch.object(diagnostic,'prediction_pass',side_effect=AssertionError('test inference should not run')):
                with self.assertRaisesRegex(ValueError,'operating points'): diagnostic.main()


if __name__=='__main__': unittest.main()
