import types
import unittest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from checks.test_full_h0_joint import arguments, make_source
from diagnostics.diagnose_full_h0_update_bottleneck import change_choice, replay, MODES
import diagnostics.diagnose_full_h0_update_bottleneck as audit
from networks.full_h0_joint import FullH0Joint, build_h0
from networks.reliable_directional_repair import DIRECTIONS, shifted, compose_surface_logits
from networks.v9_source_selector import SourceSelectedRepair


class UpdateBottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_redistribution_keeps_mass_and_respects_invalid_edges(self):
        choice = torch.tensor([.5, .1, .1, .1, .1, .05, .05, 0., 0.]).reshape(1, 9, 1, 1)
        states = torch.tensor([.2, .3, .4, .5, .6, .7, .8, 1.]).reshape(1, 8, 1, 1)
        valid = torch.ones_like(states, dtype=torch.bool); valid[:, 7] = False
        moved = change_choice(choice, states, valid, 'redistribute')
        self.assertAlmostEqual(float(moved[:, 0]), .5)
        self.assertAlmostEqual(float(moved[:, 7]), .5)
        self.assertEqual(float(moved[:, 8]), 0.)
        no_stay = change_choice(choice, states, valid, 'no_stay')
        self.assertEqual(float(no_stay[:, 0]), 0.)
        self.assertEqual(float(no_stay[:, 7]), 1.)
        invalid = change_choice(choice, states, torch.zeros_like(valid), 'no_stay')
        self.assertEqual(float(invalid[:, 0]), 1.)

    def test_actual_model_replay_and_intervention_budgets(self):
        torch.manual_seed(123)
        base, _ = build_h0(arguments(), 128)
        source, q = make_source()
        model = FullH0Joint(base, q, source, transport_rate_init=.01, proposal_rate_init=.001).eval()
        original = model._source_dynamics
        captured = {}
        def wrap(this, *args):
            actual = original(*args)
            for mode in MODES:
                states, traces = replay(this, args, mode)
                captured[mode] = (states, traces)
                self.assertEqual(len(states), 4)
                self.assertTrue(all(bool(((u >= -1e-6) & (u <= 1.+1e-6)).all()) for u in states))
            for a, b in zip(actual[0], captured['baseline'][0]):
                torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
            return actual
        object.__setattr__(model, '_source_dynamics', types.MethodType(wrap, model))
        with torch.no_grad(): result = model(torch.randn(1, 3, 128, 128))
        self.assertEqual(len(captured), len(MODES))
        torch.testing.assert_close(captured['baseline'][0][-1], result['u3'])
        for row in captured['transport_010'][1]:
            self.assertTrue(bool((row['transport_rate'] >= 0).all()))
        for row in captured['no_anchor'][1]:
            self.assertEqual(float(row['anchor_delta'].abs().max()), 0.)
        for row in captured['baseline'][1]:
            for d in range(8):
                self.assertEqual(float((row[f'source_{d}_weight'] * (1-row[f'source_{d}_valid'])).abs().max()), 0.)

    def test_entrypoint_fixed_gap_and_resume(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root, prefix='.update_audit_') as temp:
            pair = Path(temp)/'pair'; ref = pair/'reference'; ref.mkdir(parents=True)
            run = pair/'B_transport_0p01_proposal_0p001'; run.mkdir()
            n = 1500; gt = np.zeros((n,n), bool); gt[200,100:160] = True
            gap = {'id':1, 'pixels':[200*n+x for x in range(120,132)], 'length_bin':'8-15',
                   'anchors':[[200*n+x for x in range(117,120)], [200*n+x for x in range(132,135)]]}
            manifest = {'count':1, 'images':{'sample':{'shape':[n,n], 'gaps':[gap]}}}
            path = ref/'fixed_gaps_and_anchors.json'; path.write_text(json.dumps(manifest), encoding='utf-8')
            np.savez_compressed(ref/'sample.npz', gt=gt)
            torch.save({'config':{'reference_sha256':audit.digest(path), 'root_path':temp, 'seed':1}}, run/'best.pth')
            sample = {'case_name':'sample', 'image':torch.zeros(3,n,n),
                      'mask':torch.from_numpy(gt.astype(np.float32))[None]}
            p = np.zeros((n,n), np.float32); maps = {key:p for key in ('p0',)+MODES+('upper3','baseline_step1','baseline_step2')}
            output = Path(temp)/'output'
            argv = ['audit','--device','cpu','--pair_dir',str(pair),'--output_dir',str(output),
                    '--expected_gaps','1','--expected_images','1']
            with patch.object(sys,'argv',argv), patch.object(audit,'verify_recorded_code'), \
                 patch.object(audit,'rebuild_model',return_value=object()), \
                 patch.object(audit,'full_dataset',return_value=[sample]), \
                 patch.object(audit,'infer_case',return_value=(maps,[],[])), patch('builtins.print'):
                audit.main()
            report = json.loads((output/'summary.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['actual_upper_intersection_counts']['neither_closed'], 1)
            self.assertTrue((output/'gap_comparison.csv').is_file())
            with patch.object(sys,'argv',argv), patch.object(audit,'verify_recorded_code'), \
                 patch.object(audit,'rebuild_model',return_value=object()), \
                 patch.object(audit,'full_dataset',return_value=[sample]), \
                 patch.object(audit,'infer_case',side_effect=AssertionError('failed to resume')), patch('builtins.print'):
                audit.main()

    def test_tile_trace_and_original_resolution_alignment(self):
        gap = {'id':1, 'pixels':[256*512+x for x in range(240,248)],
               'anchors':[[256*512+x for x in range(228,232)],
                          [256*512+x for x in range(256,260)]]}
        entry = {'shape':[512,512], 'gaps':[gap]}
        p = torch.full((1,1,128,128), .2)
        state = [p, p+.1, p+.2, p+.3]
        fields = {'U':p, 'U_next':p+.1, 'B':p+.4}
        traces = {'baseline':[fields, fields, fields]}
        rows = audit.make_trace_rows('case',entry,0,0,traces,.45)
        self.assertEqual(len(rows), 9)
        self.assertEqual({r['region'] for r in rows}, {'gap','centre_proxy','anchors'})
        result = {'z0':torch.full((1,1,512,512), -1.), 'u0':p}
        topology = audit.make_tile_topology('case',entry,0,0,{'baseline':(state,[])},result,.45)
        self.assertEqual(len(topology), 3)
        self.assertEqual(topology[0]['step'], 1)

    def test_one_full_tile_runs_actual_capture_and_interventions(self):
        class Toy(nn.Module):
            _source_dynamics = SourceSelectedRepair._source_dynamics
            def __init__(self):
                super().__init__()
                rates = nn.Conv2d(39,4,1)
                nn.init.zeros_(rates.weight); nn.init.zeros_(rates.bias)
                self.original = types.SimpleNamespace(dynamics=types.SimpleNamespace(rates=rates))
                self._context = {}
            def selector(self, context, state, support, available, rate):
                valid = torch.cat([shifted(torch.ones_like(state),dy,dx)>.5 for dy,dx in DIRECTIONS],1)
                sources = torch.cat([shifted(state,dy,dx) for dy,dx in DIRECTIONS],1)
                logits = torch.zeros_like(sources)
                choice = torch.cat((torch.zeros_like(state),logits.masked_fill(~valid,-torch.inf)),1).softmax(1)
                return {'choice':choice,'source_states':sources,'valid':valid,
                        'edge_logits':logits,'stay_logit':torch.zeros_like(state)}
            def forward(self,image):
                z0 = image[:,:1]
                p0 = F.interpolate(z0.sigmoid(),size=(128,128),mode='bilinear',align_corners=False)
                q = p0; r = p0*.5; b = (p0+.1).clamp(max=1); a = p0
                weights = torch.full((1,8,128,128),1/8,device=image.device)
                visual = torch.zeros((1,32,128,128),device=image.device)
                states, _, _, _ = self._source_dynamics(visual,p0,q,r,b,a,weights)
                return {'z0':z0,'u0':p0,'b':b,'p1':compose_surface_logits(z0,p0,states[-1])[0].sigmoid()}
        gap = {'id':1, 'pixels':[256*512+x for x in range(240,248)],
               'anchors':[[256*512+x for x in range(228,232)], [256*512+x for x in range(256,260)]]}
        maps, traces, topology = audit.infer_case(Toy(), torch.zeros(1,3,512,512),
                                                   'case', {'shape':[512,512], 'gaps':[gap]}, .45)
        self.assertTrue(np.all(maps['upper3'] >= maps['baseline']-1e-6))
        self.assertEqual(len(traces), 7*3*3)
        self.assertEqual(len(topology), 7*3)


if __name__ == '__main__': unittest.main()
