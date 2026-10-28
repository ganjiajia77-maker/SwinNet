"""Fixed-weight intervention and train-only gate candidate contract tests."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.test_v9_state_gate import model
from diagnostics.v91_counterfactual import tile_variants, operating_points, evaluate, infer_cases, selection_analysis
from diagnostics.v91_compare_gate import infer as infer_gate_comparison
from networks.v91_gate_candidates import gate_candidates
import run_v91_gate_experiment as experiment


def toy_candidate():
    gate = torch.full((1, 3, 1, 4, 4), .6, requires_grad=True)
    p0 = torch.full((1, 1, 4, 4), .1)
    p0[0, 0, 1, 0] = .8
    mask = torch.zeros_like(p0); mask[0, 0, 1, 1] = 1.
    skeleton = mask.clone()
    action = torch.zeros_like(gate); action[:, :, 0, 1, 1] = .1; action[:, :, 0, 2, 0] = .1
    avail = torch.full_like(action, .3)
    proposal_state = p0[:, None].expand(-1, 3, -1, -1, -1).clone()
    proposal_state[:, :, :, 1, 1] += .1
    proposal_state[:, :, :, 2, 0] += .1
    return {'g_step': gate, 'u0': p0, 'u1': p0, 'u2': p0,
            'p0': p0, 'p1': p0.clone(), 'q': torch.zeros_like(p0),
            'm': proposal_state, 'transport_rate': torch.ones_like(action),
            'transport_delta': action, 'source_available': avail}, mask, skeleton


class TestV91(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(1234)

    def test_interventions_use_original_frozen_and_keep_checkpoint_weights(self):
        net = model().eval()
        with torch.no_grad():
            net.state_gate.net[-1].bias.fill_(1.)
        before = {key: value.clone() for key, value in net.state_dict().items()}
        calls = []
        def hook(module, args, kwargs):
            calls.append({'freeze': kwargs.get('freeze_generated_sources', False),
                          'disable': kwargs.get('disable_transport', False),
                          'state_gate': kwargs.get('state_gate') is not None})
        handle = net.dynamics.register_forward_pre_hook(hook, with_kwargs=True)
        gate = net.state_gate
        variants = tile_variants(net, torch.randn(1, 3, 64, 64))
        handle.remove()
        self.assertIs(net.state_gate, gate)
        self.assertEqual(calls, [{'freeze': False, 'disable': False, 'state_gate': True},
                                 {'freeze': False, 'disable': False, 'state_gate': False},
                                 {'freeze': True, 'disable': False, 'state_gate': True},
                                 {'freeze': False, 'disable': True, 'state_gate': True}])
        for name in variants:
            torch.testing.assert_close(variants[name]['z0'], variants['A_original']['z0'], rtol=0, atol=0)
        self.assertGreater(float((variants['A_original']['z1']-variants['B_state_zero']['z1']).abs().max()), 0.)
        for key, value in net.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_actual_positive_transport_only_and_matching_gradients(self):
        result, mask, skeleton = toy_candidate()
        loss, report = gate_candidates(result, mask, skeleton, include_samples=True)
        self.assertEqual(report['positive_before_matching'], 3)
        self.assertEqual(report['negative_before_matching'], 3)
        self.assertEqual(report['matched_positive'], 3)
        self.assertEqual(report['matched_negative'], 3)
        self.assertEqual(len(report['samples']), 6)
        loss.backward()
        grad = result['g_step'].grad
        self.assertTrue((grad[:, :, 0, 1, 1] < 0).all())
        self.assertTrue((grad[:, :, 0, 2, 0] > 0).all())
        self.assertEqual(int((grad != 0).sum()), 6)
        self.assertEqual(report['mean_absolute_pair_differences']['potential_action'], 0.)
        self.assertEqual(report['mean_absolute_pair_differences']['realized_action'], 0.)
        summary = experiment.aggregate([report])
        self.assertEqual(summary['matched_positive'], 3)
        self.assertAlmostEqual(summary['matched_G_roc_auc_hist_approx'], .5)

    def test_four_interventions_cover_all_images_and_stitch_logits(self):
        image = torch.zeros(3, 768, 768); mask = torch.zeros(1, 768, 768)
        dataset = [{'image': image, 'mask': mask, 'case_name': str(i)} for i in range(14)]
        z0 = torch.full((1, 1, 512, 512), -2.)
        variants = {name: {'z0': z0, 'z1': z0 + i*.1} for i, name in enumerate(
                    ('A_original', 'B_state_zero', 'C_source_fixed', 'D_transport_off'))}
        with patch('diagnostics.v91_counterfactual.tile_variants', return_value=variants) as variant_fn:
            outputs = infer_cases(None, dataset, torch.device('cpu'))
        self.assertEqual(len(outputs), 14)
        self.assertEqual(variant_fn.call_count, 14*4)
        self.assertAlmostEqual(float(outputs[0]['p0'][0, 0]), float(torch.sigmoid(torch.tensor(-2.))), places=6)
        self.assertAlmostEqual(float(outputs[0]['D_transport_off'][700, 700]),
                               float(torch.sigmoid(torch.tensor(-1.7))), places=6)

    def test_three_model_comparison_shares_frozen_features(self):
        class FakeModel:
            def __init__(self, bias): self.bias = bias; self.calls = 0
            def frozen(self, tile, capture_diagnostics=False):
                self.calls += 1; return {'z0': tile[:, :1]-2}
            def __call__(self, image, frozen_override, capture_diagnostics=False):
                z0 = frozen_override['z0']; result = {'z0': z0, 'z1': z0+self.bias}
                if capture_diagnostics:
                    spatial = z0[:, :, ::4, ::4]
                    by_step = torch.stack([spatial*0+.2+self.bias for _ in range(3)], dim=1)
                    result.update(q=spatial*0+.7, g_step=by_step, source_available=by_step,
                                  m=by_step, transport_delta=by_step, proposal_delta=by_step,
                                  **{f'u{k}': spatial*0 for k in range(3)})
                return result
        models = {'A_original': FakeModel(0), 'B_control': FakeModel(.1), 'C_gate': FakeModel(.2)}
        image = torch.zeros(3, 768, 768); mask = torch.zeros(1, 768, 768)
        outputs = infer_gate_comparison(models, [{'image': image, 'mask': mask, 'case_name': 'toy'}],
                                        torch.device('cpu'))
        self.assertEqual(models['A_original'].calls, 4)
        self.assertEqual(models['B_control'].calls, 0)
        self.assertEqual(models['C_gate'].calls, 0)
        self.assertGreater(float(outputs[0]['C_gate'][600, 600]), float(outputs[0]['A_original'][600, 600]))
        mask[:, 300:303, 300:400] = 1
        with tempfile.TemporaryDirectory(prefix='.v91_regions_', dir=ROOT) as tmp:
            context = {'reference_threshold': .45, 'original_threshold': .68, 'output_dir': Path(tmp)}
            with patch.dict(sys.modules, {'evaluation.road_comparison.road_metrics':
                                         SimpleNamespace(skeletonize=lambda x: x)}):
                infer_gate_comparison(models, [{'image': image, 'mask': mask, 'case_name': 'toy'}],
                                      torch.device('cpu'), region_context=context)
            with (Path(tmp)/'same_position_gate_regions.csv').open(encoding='utf-8') as f:
                rows = list(__import__('csv').DictReader(f))
            values = [float(r['mean']) for r in rows if r['region'] == 'missing_gt' and
                      r['quantity'] == 'G' and r['step'] == '1']
            self.assertEqual(len(values), 3)
            np.testing.assert_allclose(values, [.2, .3, .4], atol=1e-6)

    def test_no_valid_or_unmatched_candidate_produces_no_forced_gate_loss(self):
        result, mask, skeleton = toy_candidate()
        result['transport_rate'].zero_()
        loss, report = gate_candidates(result, mask, skeleton)
        self.assertEqual(report['matched_positive'], 0)
        self.assertEqual(report['matched_negative'], 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(int((result['g_step'].grad != 0).sum()), 0)
        result, mask, skeleton = toy_candidate()
        result['source_available'][:, :, 0, 2, 0] = .99
        loss, report = gate_candidates(result, mask, skeleton)
        self.assertEqual(report['positive_before_matching'], 3)
        self.assertEqual(report['negative_before_matching'], 3)
        self.assertEqual(report['matched_positive'], 0)

    def test_low_gate_does_not_hide_useful_pre_gate_transport(self):
        result, mask, skeleton = toy_candidate()
        result['g_step'].data.fill_(1e-4)
        result['transport_delta'].zero_()
        loss, report = gate_candidates(result, mask, skeleton)
        self.assertEqual(report['matched_positive'], 3)
        self.assertEqual(report['matched_negative'], 3)
        loss.backward()
        self.assertTrue((result['g_step'].grad != 0).any())

    def test_valid_matching_reports_actual_residual_and_frozen_gap_set(self):
        gt = np.zeros((8, 8), dtype=bool); gt[3, 1:7] = True
        p0 = np.full((8, 8), .1, np.float32); p0[gt] = .8; p0[3, 3:5] = .2
        a = p0.copy(); a[3, 3:5] = .75; a[0, 0] = .8
        b = a.copy(); b[0, 0] = .1
        cases = [dict(case='toy', gt=gt, p0=p0, A_original=a, B_state_zero=b,
                      C_source_fixed=b, D_transport_off=b)]
        points = operating_points(cases, .68, 10000)
        self.assertIn('matched_recall', points['B_state_zero'])
        def metrics(pred, prepared, protocol):
            target = prepared['mask']
            return {'tp': int((pred & target).sum()), 'fp': int((pred & ~target).sum()),
                'fn': int((~pred & target).sum()), 'cldice': .5,
                'apls_bidirectional_approx': .5, 'short_pred_components': 1.}
        def summarize(rows):
            row = dict(rows[0]); row.pop('case_id'); row['images'] = len(rows)
            row['recall'] = row['tp']/max(row['tp']+row['fn'], 1)
            row['iou'] = row['tp']/max(row['tp']+row['fp']+row['fn'], 1)
            return row
        seen = []
        def gaps(reference, pred, skeleton, mask):
            seen.append(reference.copy())
            return ({k: {'eligible': int(k == '3-7'), 'closed': int(k == '3-7' and pred[3, 3])}
                     for k in ('3-7', '8-15', '16-31', '32+')}, 0)
        mocks = {'evaluation.road_comparison.road_metrics': SimpleNamespace(
                    image_metrics=metrics, prepare_target=lambda x, p: {'mask': x, 'skeleton': x}, summarize=summarize),
                 'run_q3_r3_mass': SimpleNamespace(PROTOCOL={}),
                 'run_reliable_directional_mass': SimpleNamespace(gap_lengths=gaps)}
        with tempfile.TemporaryDirectory(prefix='.v91_test_', dir=ROOT) as tmp:
            with patch.dict(sys.modules, mocks):
                reports = evaluate(cases, points, .45, 1, Path(tmp))
                fingerprint = __import__('json').loads((Path(tmp)/'reference_set.json').read_text())['sha256']
                with self.assertRaises(ValueError):
                    evaluate(cases, points, .45, 206, Path(tmp))
                unchanged_count = len(seen)
                changed = copy.deepcopy(cases); changed[0]['p0'][0, 0] = .9
                evaluate(changed, points, .45, 1, Path(tmp))
                self.assertNotEqual(fingerprint, __import__('json').loads((Path(tmp)/'reference_set.json').read_text())['sha256'])
            self.assertTrue(all(np.array_equal(reference, p0 >= .45) for reference in seen[:unchanged_count]))
            self.assertLess(abs(reports['B_state_zero']['matched_recall']['recall_difference_vs_A']), 1e-8)
            self.assertEqual(reports['B_state_zero']['same_threshold']['paired']['removed_background_vs_original'], 1)

    def test_threshold_penalty_decomposition_does_not_change_selection(self):
        def entry(iou, bridges):
            m = {'iou': iou, 'f1': .7, 'recall': .7, 'cldice': .8,
                 'apls_gt_to_pred_approx': .6, 'apls_bidirectional_approx': .65,
                 'short_pred_components': 5.}
            paired = {'background_bridge_proxy': bridges, 'closed_p0_gaps': 1, 'eligible_p0_gaps': 206}
            score = iou+.15*.8+.1*.6+.05*.65-.001*5-.01*bridges/14
            return {'p1': m, 'paired': paired, 'images': 14, 'selection_score': score}
        selection = {'threshold': .68, 'threshold_curves': {'0.55': entry(.64, 20), '0.68': entry(.635, 0)}}
        with tempfile.TemporaryDirectory(prefix='.v91_score_', dir=ROOT) as tmp:
            result = selection_analysis(selection, Path(tmp))
            self.assertEqual(result['score_winner'], .68)
            self.assertEqual(result['without_bridge_penalty_winner_diagnostic_only'], .55)
            self.assertEqual(selection['threshold'], .68)
            selection['threshold_curves']['0.55']['selection_score'] += .1
            with self.assertRaises(ValueError):
                selection_analysis(selection, Path(tmp))

    def test_control_and_gate_continuation_start_identically_and_save_separately(self):
        # Prime PyTorch's lazy optimizer imports before patch.dict restores sys.modules.
        import torch._dynamo
        initial = model().eval()
        control_model = copy.deepcopy(initial)
        gate_model = copy.deepcopy(initial)
        image = torch.randn(3, 64, 64)
        mask = torch.zeros(1, 64, 64); mask[:, 30:33, 4:60] = 1
        skeleton = torch.zeros_like(mask); skeleton[:, 31, 4:60] = 1
        sample = {'image': image, 'mask': mask, 'skeleton': skeleton, 'case_name': 'toy'}
        class Dataset:
            def __len__(self): return 2
            def __getitem__(self, i): return sample
            def set_epoch(self, i): pass
        original, toy_mask, toy_skeleton = toy_candidate()
        _, report = gate_candidates(original, toy_mask, toy_skeleton)
        def fake_gate(result, batch, source, args, seed, include_samples=False):
            return (result['g_step']-.2).square().mean(), copy.deepcopy(report)
        baseline = {'training_config': {'root_path': str(ROOT), 'batch_size': 2, 'seed': 1234,
                    'reference_threshold': .45}, 'learning_rates': dict(experiment.v9.LR)}
        source = {'lambda_bg': 1., 'lambda_missing': 2., 'lambda_cont': .2, 'missing_threshold': .45}
        with tempfile.TemporaryDirectory(prefix='.v91_train_', dir=ROOT) as tmp:
            root = Path(tmp); checkpoint = root/'original.pth'; torch.save({'fake': True}, checkpoint)
            candidate_report = root/'candidates.json'
            candidate_report.write_text(__import__('json').dumps({
                'source_sha256': experiment.v9.digest(checkpoint), 'split': 'train', 'images': 2,
                'random_crop_seed': 1234, 'min_action': 1e-5, 'near_gt_work_pixels': 8.,
                'max_pairs': 256, 'max_pairs_per_step': 256, 'output_threshold': .68,
                'batch_size': 2, 'matched_positive': 3}))
            args = SimpleNamespace(max_images=0, candidate_report=candidate_report, output_dir=root/'out',
                seed=1234, batch_size=2, min_action=1e-5, near_gt_work_pixels=8., max_pairs=256,
                max_pairs_per_step=256, output_threshold=.68, variant='control', num_workers=0,
                epochs=1, val_interval=1, lambda_gate=.1, root_path=ROOT, device=torch.device('cpu'))
            q3 = SimpleNamespace(full_dataset=lambda *unused: [sample])
            evaluation = SimpleNamespace(evaluate_thresholds=lambda *unused, **kwargs:
                ({.6: {'selection_score': .5}}, None, None, None))
            with patch.dict(sys.modules, {'run_q3_r3_mass': q3, 'run_reliable_directional_mass': evaluation}), \
                 patch.object(experiment, 'training_dataset', return_value=Dataset()), \
                 patch.object(experiment, 'candidate_loss', side_effect=fake_gate):
                experiment.train(args, control_model, source, {'test': True}, baseline, checkpoint)
                args.variant = 'gate'
                experiment.train(args, gate_model, source, {'test': True}, baseline, checkpoint)
            control = torch.load(root/'out/control/best.pth', weights_only=False)
            gate = torch.load(root/'out/gate/best.pth', weights_only=False)
            self.assertEqual(control['v91_initial_checkpoint_sha256'], gate['v91_initial_checkpoint_sha256'])
            self.assertEqual(control['v91_config']['candidate_report_sha256'], gate['v91_config']['candidate_report_sha256'])
            self.assertGreater(float((control['state_gate']['net.2.bias']-gate['state_gate']['net.2.bias']).abs().max()), 0.)
            restored = copy.deepcopy(initial)
            experiment.v9.restore_checkpoint(restored, root/'out/gate/best.pth', {'test': True}, 'D')
            for key, value in gate_model.state_gate.state_dict().items():
                torch.testing.assert_close(restored.state_gate.state_dict()[key], value, rtol=0, atol=0)
            for key, value in initial.frozen.state_dict().items():
                torch.testing.assert_close(control_model.frozen.state_dict()[key], value, rtol=0, atol=0)
                torch.testing.assert_close(gate_model.frozen.state_dict()[key], value, rtol=0, atol=0)
            args.command = 'resume'; args.variant = 'control'; args.epochs = 2
            resumed_model = copy.deepcopy(initial)
            with patch.dict(sys.modules, {'run_q3_r3_mass': q3, 'run_reliable_directional_mass': evaluation}), \
                 patch.object(experiment, 'training_dataset', return_value=Dataset()), \
                 patch.object(experiment, 'candidate_loss', side_effect=fake_gate):
                experiment.train(args, resumed_model, source, {'test': True}, baseline, checkpoint)
            self.assertEqual(torch.load(root/'out/control/last.pth', weights_only=False)['epoch'], 2)


if __name__ == '__main__':
    unittest.main()
