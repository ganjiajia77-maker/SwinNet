"""Audit checks on actual downstream equations, without original trained weights."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.test_v9_state_gate import model, repair, runner
from diagnostics import diagnose_v9_transport as audit


class TestTransportAudit(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1234)
        self.model = model()
        self.model.requires_grad_(True)
        self.model.frozen.requires_grad_(False)
        self.image = torch.randn(2, 3, 64, 64)
        self.mask = torch.zeros(2, 1, 64, 64)
        self.mask[:, :, 30:33, 4:60] = 1
        self.skeleton = torch.zeros_like(self.mask)
        self.skeleton[:, :, 31, 4:60] = 1
        self.source = dict(lambda_bg=1., lambda_missing=2., lambda_cont=.2, missing_threshold=.45)

    def test_real_gradient_chain_and_no_parameter_updates(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        report, rows = audit.gradient_batch(self.model, self.image, self.mask, self.skeleton, self.source)
        self.assertEqual(report['frozen_parameter_grad_count'], 0)
        self.assertEqual(report['frozen_parameter_requires_grad_count'], 0)
        for name, metrics in report['modules'].items():
            self.assertGreater(metrics['total_l2'], 0, name)
        for k in (1, 2, 3):
            self.assertTrue(report['u_gradient_chain'][f'u{k}']['requires_grad'])
            self.assertGreater(report['u_gradient_chain'][f'u{k}']['loss_gradient']['mean_abs'], 0)
        for k in (1, 2):
            self.assertGreater(report['u_gradient_chain'][f'u{k}_to_u{k+1}']['mean_abs'], 0)
        self.assertEqual(report['compose_surface_logits']['identity_logit_max_error'], 0.)
        self.assertGreater(report['compose_surface_logits']['p1_wrt_u3']['mean_abs'], 0)
        self.assertGreater(report['proposal_saturation']['plus_local_derivative']['mean'], 0)
        self.assertTrue(rows)
        json.dumps(report, allow_nan=False)
        for key, val in self.model.state_dict().items():
            torch.testing.assert_close(val, before[key], rtol=0, atol=0)

    def test_update_formula_checks_and_deliberate_corruption(self):
        rates = []
        handle = self.model.dynamics.rates.register_forward_hook(lambda m, i, out: rates.append(out.softmax(1)))
        with torch.no_grad():
            result = self.model(self.image[:1], capture_diagnostics=True)
            maps = audit.transport_maps(result, rates)
        handle.remove()
        self.assertEqual(len(rates), 3)
        for k in (1, 2, 3):
            self.assertIn(f'V_{k}', maps)
            self.assertIn(f'rate_0_{k}', maps)
            np.testing.assert_allclose(maps[f'transport_{k}'],
                result['transport_delta'][0, k-1, 0].numpy(), atol=1e-7)
        with self.assertRaises(AssertionError):
            audit.transport_maps(result, [torch.ones_like(rate) for rate in rates])

    def test_compose_clamp_derivative_and_interpolation(self):
        u = torch.tensor([[[[-.01, .000001, .2, .8, .999999, 1.01]]]], requires_grad=True)
        p0 = torch.full_like(u, .3)
        z0 = torch.zeros(1, 1, 4, 12)
        z1, delta = repair.compose_surface_logits(z0, p0, u)
        grad, = torch.autograd.grad(z1.sum(), u)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad[0, 0, 0, [0, 1, 4, 5]] == 0).all())
        self.assertTrue((grad[0, 0, 0, [2, 3]] > 0).all())
        self.assertEqual(tuple(z1.shape), tuple(z0.shape))
        self.assertTrue(torch.isfinite(delta).all())

    def test_fine_matching_actual_threshold_counts(self):
        p0 = np.array([[.9, .8, .1, .1]], dtype=np.float32)
        p1 = np.array([[.7, .60123, .60124, .1]], dtype=np.float32)
        caches = [dict(p0=p0, p1=p1, gt=np.array([[1, 1, 0, 0]], dtype=bool))]
        threshold, error = audit.matched_threshold(caches, 'p1', 2, 'tp')
        actual = int(((p1 >= threshold) & caches[0]['gt']).sum())
        self.assertEqual(error, actual-2)
        self.assertEqual(error, 0)
        threshold, error = audit.matched_threshold(caches, 'p1', 2, 'area')
        self.assertEqual(error, int((p1 >= threshold).sum())-2)
        # A 0.0001 grid cannot separate these two near-equal values; residual must be explicit.
        self.assertNotEqual(error, 0)

    def test_histories_and_histograms_are_json_serializable(self):
        with tempfile.TemporaryDirectory(prefix='.v9_audit_test_', dir=ROOT) as tmp:
            root = Path(tmp); (root/'stage_D').mkdir(); output = root/'out'; output.mkdir()
            entry = {'epoch': 1, 'gradient_norms': {n: .001 for n in runner.STAGE_MODULES['D']}}
            (root/'stage_D/epoch_metrics.jsonl').write_text(json.dumps(entry)+'\n')
            result = audit.read_history(root, output, self.model)
            self.assertEqual(result['module_gradients']['state_gate']['finite_positive_epochs'], 1)
            self.assertTrue((output/'historical_gradients.csv').is_file())
        stats = audit.Stats()
        stats.add('all', 'candidate_1', np.array([-.1, 0., .1]))
        row = stats.rows()[0]
        self.assertEqual(row['pixels'], 3)
        self.assertAlmostEqual(row['mean'], 0.)
        self.assertGreater(row['mean_abs'], 0.)
        json.dumps(stats.rows(), allow_nan=False)

    def test_all_images_and_overlapping_windows_count_each_pixel_once(self):
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dynamics = nn.Module()
                self.dynamics.rates = nn.Conv2d(1, 4, 1)

            def forward(self, image, capture_diagnostics=False):
                signal = image[:, :1, ::4, ::4]
                for _ in range(3):
                    self.dynamics.rates(signal)
                return {'z0': image[:, :1] - 2, 'z1': image[:, :1] - 1,
                        'q': signal + .7}

        image = torch.zeros(3, 768, 768)
        mask = torch.zeros(1, 768, 768)
        mask[:, 380:383, 150:600] = 1
        dataset = [{'image': image, 'mask': mask, 'case_name': str(i)} for i in range(14)]
        q3 = SimpleNamespace(full_dataset=lambda root, split: dataset)
        metrics = SimpleNamespace(skeletonize=lambda gt: gt)
        with tempfile.TemporaryDirectory(prefix='.v9_audit_full_', dir=ROOT) as tmp:
            args = SimpleNamespace(root_path=ROOT, output_dir=Path(tmp), device=torch.device('cpu'),
                                   reference_threshold=.45, threshold=.25)
            with patch.dict(sys.modules, {'run_q3_r3_mass': q3,
                    'evaluation.road_comparison.road_metrics': metrics}), \
                 patch.object(audit, 'transport_maps', return_value={'V_1': np.full((128, 128), .25)}):
                caches, rows = audit.collect_validation(args, FakeModel())
            self.assertEqual(len(caches), 14)
            all_v = next(r for r in rows if r['region'] == 'all' and r['quantity'] == 'V_1')
            self.assertEqual(all_v['pixels'], 14 * 768 * 768)
            self.assertAlmostEqual(all_v['mean'], .25)
            self.assertTrue((Path(tmp)/'transport_regions.csv').is_file())

    def test_four_comparisons_keep_the_same_reference_and_validate_counts(self):
        gt = np.zeros((16, 16), bool); gt[7, 2:14] = True
        p0 = np.full(gt.shape, .1, np.float32); p0[gt] = .8; p0[7, 7:9] = .2
        p1 = np.full(gt.shape, .1, np.float32); p1[gt] = .75; p1[7, 7:9] = .7; p1[0, 0] = .8
        caches = [dict(case='toy', p0=p0, p1=p1, gt=gt)]
        metrics_keys = ('cldice', 'apls_gt_to_pred_approx', 'apls_pred_to_gt_approx',
                        'apls_bidirectional_approx', 'short_pred_components')
        def image_metrics(pred, prepared, protocol):
            target = prepared['mask']
            return {'tp': int((pred & target).sum()), 'fp': int((pred & ~target).sum()),
                    'fn': int((~pred & target).sum()), **{k: .5 for k in metrics_keys}}
        def summarize(rows):
            counts = {k: sum(row[k] for row in rows) for k in ('tp', 'fp', 'fn')}
            tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
            return {'images': len(rows), **counts, 'iou': tp/max(tp+fp+fn, 1),
                    'f1': 2*tp/max(2*tp+fp+fn, 1), 'precision': tp/max(tp+fp, 1),
                    'recall': tp/max(tp+fn, 1), **{k: .5 for k in metrics_keys}}
        metrics = SimpleNamespace(prepare_target=lambda target, protocol:
                                  {'mask': target, 'skeleton': target},
                                  image_metrics=image_metrics, summarize=summarize)
        observed_references = []
        def gap_lengths(ref, pred, skeleton, target):
            observed_references.append(ref.copy())
            bins = {name: {'eligible': int(name == '3-7'), 'closed': int(name == '3-7' and pred[7, 7])}
                    for name in ('3-7', '8-15', '16-31', '32+')}
            return bins, 0
        direction = SimpleNamespace(gap_lengths=gap_lengths,
            selection_score=lambda metrics, paired, images: metrics['iou'])
        q3 = SimpleNamespace(PROTOCOL={})
        curves = {}
        for threshold in (.45, .65):
            curves[str(threshold)] = {'p0': summarize([image_metrics(p0 >= threshold, {'mask': gt}, {})]),
                'p1': summarize([image_metrics(p1 >= threshold, {'mask': gt}, {})]), 'selection_score': .5}
        selection = {'threshold_curves': curves, 'threshold': .65}
        with tempfile.TemporaryDirectory(prefix='.v9_audit_cmp_', dir=ROOT) as tmp:
            args = SimpleNamespace(output_dir=Path(tmp), reference_threshold=.45, matching_bins=10000)
            with patch.dict(sys.modules, {'run_q3_r3_mass': q3,
                    'run_reliable_directional_mass': direction,
                    'evaluation.road_comparison.road_metrics': metrics}):
                result = audit.compare_outputs(args, caches, selection)
            self.assertEqual(set(result), {'own_val_thresholds', 'same_threshold',
                                           'matched_recall', 'matched_foreground_area'})
            self.assertTrue(all(np.array_equal(item, p0 >= .45) for item in observed_references))
            self.assertTrue(all(row['paired']['eligible_p0_gaps'] == 1 for row in result.values()))
            self.assertLess(abs(result['matched_recall']['recall_difference']), 1e-8)
            self.assertTrue((Path(tmp)/'operating_points.json').is_file())
            selection['threshold_curves']['0.65']['p1']['tp'] += 1
            with patch.dict(sys.modules, {'run_q3_r3_mass': q3,
                    'run_reliable_directional_mass': direction,
                    'evaluation.road_comparison.road_metrics': metrics}), self.assertRaises(ValueError):
                audit.compare_outputs(args, caches, selection)


if __name__ == '__main__':
    unittest.main()
