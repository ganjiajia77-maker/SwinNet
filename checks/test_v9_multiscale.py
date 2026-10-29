"""Executable source selection/dynamics checks; no upstream checkpoint required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.test_v9_state_gate import model as original_model, repair
from networks.v9_multiscale_transport import SourceSelector, SOURCES, install, transition, sample_offset
import run_v9_multiscale as runner


class MultiscaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(1234)

    def test_local_exact_and_no_rng_change(self):
        model = original_model().eval(); image = torch.randn(2, 3, 64, 64)
        before = model(image, capture_diagnostics=True)
        rng = torch.get_rng_state().clone()
        install(model, 'local')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = model(image, capture_diagnostics=True)
        for key in before:
            if isinstance(before[key], torch.Tensor):
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        install(model, 'multiscale')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.dynamics.blend = 0.
        mixed = model(image, capture_diagnostics=True)
        for key in ('z1', 'u1', 'u2', 'u3', 'source_available', 'alpha'):
            torch.testing.assert_close(before[key], mixed[key], atol=0, rtol=0)

    def test_sampling_fractional_xy_and_no_wrap(self):
        y, x = torch.meshgrid(torch.arange(5.), torch.arange(7.), indexing='ij')
        value = (10*y+x)[None, None]
        actual = sample_offset(value, .25, .5)
        torch.testing.assert_close(actual[0, 0, :-1, :-1], value[0, 0, :-1, :-1]+3., atol=1e-5, rtol=0)
        self.assertEqual(float(sample_offset(value, 0., 7.)[0, 0, 2, 0]), 0.)

    def test_softmax_convexity_boundary_and_empty(self):
        module = SourceSelector(width=8, embedding=4, recompute=False)
        for h, w in ((1, 1), (3, 5), (9, 10)):
            visual = torch.randn(2, 8, h, w)
            state = torch.rand(2, 1, h, w); q = torch.rand_like(state)
            weights = torch.rand(2, 8, h, w).softmax(1)
            target, available, details = module(visual, state, torch.zeros_like(state), state, q, weights)
            probabilities = details['selector_weights']
            torch.testing.assert_close(probabilities.sum(1, keepdim=True), torch.ones_like(state))
            self.assertTrue(bool((target >= 0).all() & (target <= 1).all()))
            self.assertEqual(float(probabilities[:, 1:].masked_select(~details['selector_valid']).abs().sum()), 0.)
            if h == w == 1:
                torch.testing.assert_close(target, state, atol=0, rtol=0)
                self.assertEqual(float(available.abs().sum()), 0.)
            else:
                self.assertGreater(float(probabilities[:, :1].mean()), .99)

    def test_source_identity_and_corridor_is_not_hard_mask(self):
        selector = SourceSelector(width=8, embedding=4, recompute=False)
        state = torch.zeros(1, 1, 20, 20); state[:, :, 10, 18] = 1.
        support = torch.zeros_like(state); support[:, :, 10, 18] = 1.
        visual = torch.randn(1, 8, 20, 20)
        target, _, details = selector(visual, state, support, torch.zeros_like(state),
                                      torch.zeros_like(state), torch.ones(1, 8, 20, 20)/8)
        index = SOURCES.index((0, 1, 8))
        self.assertEqual(float(details['selector_source_values'][0, index, 10, 10]), 1.)
        self.assertGreater(float(details['selector_weights'][0, index+1, 10, 10]), .01)
        self.assertGreater(float(target[0, 0, 10, 10]), .01)
        # The score head really sees intermediate and transverse visual samples.
        channels = selector.score[0].in_channels
        self.assertEqual(channels, 4*4 + 9*(4+3) + 11)

    def test_gradient_freeze_optimizer_bounds_and_checkpoint(self):
        model = install(original_model(), 'multiscale')
        config = {'test': True}
        rates = {name: 1e-3 for name in runner.v9.STAGE_MODULES['D']}
        optimizer = runner.configure(model, rates, 1e-3)
        model.train(); model.dynamics.blend = 1.
        image = torch.randn(2, 3, 64, 64)
        gt = torch.zeros(2, 1, 64, 64); gt[:, :, 30:34, 5:60] = 1.
        skeleton = torch.zeros_like(gt); skeleton[:, :, 31, 5:60] = 1.
        initial = copy.deepcopy(model.dynamics.selector.state_dict())
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(image, capture_diagnostics=True)
            loss, _ = repair.dynamic_losses(result, gt, skeleton)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward(); runner.check_gradients(model)
            self.assertGreater(runner.v9.grad_norm(model.dynamics.selector), 0)
            self.assertFalse(model.frozen.training)
            for key in ('alpha', 'beta', 'gamma', 'stay'):
                self.assertTrue(bool((result[key] >= -1e-7).all()))
            torch.testing.assert_close(result['alpha']+result['beta']+result['gamma']+result['stay'],
                                       torch.ones_like(result['stay']))
            for key in ('u1', 'u2', 'u3'):
                self.assertTrue(bool((result[key] >= 0).all() & (result[key] <= 1).all()))
            optimizer.step()
        self.assertFalse(torch.equal(initial['score.4.weight'], model.dynamics.selector.state_dict()['score.4.weight']))
        self.assertGreater(runner.v9.grad_norm(model.dynamics.selector.project), 0)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory)/'last.pth'
            runner.save(path, model, {'identity': {'test': 1}}, config, optimizer, 2, .5, {})
            cloned = install(original_model(), 'multiscale')
            runner.restore(cloned, path, config)
            # Frozen provider is deliberately synthetic; copy its parameters too.
            cloned.frozen.load_state_dict(model.frozen.state_dict(), strict=True)
            model.eval(); cloned.eval()
            torch.testing.assert_close(model(image)['z1'], cloned(image)['z1'], atol=0, rtol=0)
            wrong = install(original_model(), 'local')
            with self.assertRaises(RuntimeError):
                runner.restore(wrong, path, config)

    def test_fixed_schedule(self):
        self.assertEqual([transition(k, 3) for k in (1, 2, 3, 4, 5)], [0, 1/3, 2/3, 1, 1])
        self.assertEqual(transition(1, 0), 1.)

    def test_fixed_whole_gap_and_anchor_closure(self):
        from diagnostics.audit_v9_multiscale import fixed_gaps, connected
        gt = np.zeros((64, 64), bool); gt[30:33, 2:62] = True
        skeleton = np.zeros_like(gt); skeleton[31, 2:62] = True
        p0 = gt.copy(); p0[:, 20:42] = False
        gaps, _ = fixed_gaps(p0, skeleton, gt)
        self.assertEqual(len(gaps), 1)
        gap = gaps[0]; self.assertTrue(gap['eligible'])
        self.assertFalse(connected(gap, p0, gt))
        island = p0.copy(); island[31, 29:33] = True
        self.assertFalse(connected(gap, island, gt))
        self.assertTrue(connected(gap, gt, gt))

    def test_training_resume_and_four_return_evaluator(self):
        class SmallDataset(torch.utils.data.Dataset):
            def set_epoch(self, value): self.epoch = value
            def __len__(self): return 2
            def __getitem__(self, index):
                return {'image': torch.zeros(3, 32, 32), 'mask': torch.zeros(1, 32, 32),
                        'skeleton': torch.zeros(1, 32, 32)}
        rates = {name: 1e-4 for name in runner.v9.STAGE_MODULES['D']}
        source_config = {'reference_threshold': .45, 'lambda_bg': 1., 'lambda_missing': 2.,
                         'lambda_cont': .2, 'missing_threshold': .45}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory); source = root/'source.pth'; source.write_bytes(b'source')
            config = {'mode': 'multiscale', 'learning_rates': rates, 'selector_lr': 1e-4,
                      'training': source_config, 'source_sha256': runner.v9.digest(source)}
            args = SimpleNamespace(output_dir=root, transport_mode='multiscale', command='train',
                root_path=root, device=torch.device('cpu'), seed=1234, batch_size=2,
                num_workers=0, epochs=2, transition_epochs=0, val_interval=1, source_checkpoint=source)
            model = install(original_model(), 'multiscale')
            with patch.object(runner, 'dataset', return_value=SmallDataset()), \
                 patch('run_q3_r3_mass.full_dataset', return_value=SmallDataset()), \
                 patch('run_reliable_directional_mass.evaluate_thresholds', return_value=({.60: {'selection_score': .5}}, {}, [], 0.)):
                runner.train(args, model, {'identity': {}}, config)
                args.command = 'resume'
                runner.train(args, model, {'identity': {}}, config)
                self.assertEqual(len((root/'multiscale/train.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
