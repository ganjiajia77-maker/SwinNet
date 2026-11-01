"""Exercise rate startup, metrics and a real reduced-size training update."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from checks.test_full_h0_joint import arguments, make_source
import diagnostics.diagnose_full_h0_rate_startup as runner
from diagnostics.diagnose_full_h0_rate_startup import (differences, evaluate, initialize_rates, write_results)
from networks.full_h0_joint import FullH0Joint, build_h0, original_criterion, parameter_groups
from networks.v9_source_relation_labels import RelationLabels
from run_full_h0_joint import batch_loss


ROOT = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(1234)
        base, cls.saved = build_h0(arguments(), 128)
        source, q = make_source()
        cls.model = FullH0Joint(base, q, source)

    def batch(self):
        image = torch.randn(2, 3, 128, 128)
        mask = torch.zeros(2, 1, 128, 128)
        mask[:, :, 47:79, 4:124] = 1
        skeleton = torch.zeros_like(mask)
        skeleton[:, :, 63, 4:124] = 1
        return {'image': image, 'mask': mask, 'skeleton': skeleton,
                'skeleton_dilate': torch.nn.functional.max_pool2d(skeleton, 7, 1, 3)}

    def test_separate_softmax_budgets_and_invalid_inputs(self):
        init = initialize_rates(self.model, .01, .001)
        head = self.model.original.dynamics.rates[-1]
        rates = head.bias.softmax(0)
        self.assertAlmostEqual(float(rates[1]), .01, places=7)
        self.assertAlmostEqual(float(rates[2]), .001, places=7)
        self.assertEqual(float(head.weight.detach().abs().max()), 0.)
        self.assertEqual(init['transport'], .01)
        for bad in ((0., .001), (.01, 0.), (.9, .2)):
            with self.assertRaises(ValueError):
                initialize_rates(self.model, *bad)

    def test_tp_fp_flips_and_backbone_drift(self):
        p0 = torch.tensor([.1, .1, .9, .9]).view(1, 1, 2, 2)
        p1 = torch.tensor([.9, .9, .1, .1]).view(1, 1, 2, 2)
        gt = torch.tensor([1., 0., 1., 0.]).view(1, 1, 2, 2)
        initial = torch.tensor([.1, .1, .1, .9]).view(1, 1, 2, 2)
        result = differences(p0, p1, gt, initial, .45)
        for key in ('repair_added_tp', 'repair_lost_tp', 'repair_added_fp', 'repair_removed_fp'):
            self.assertEqual(result[key], 1)
        self.assertEqual(result['P0_changed_binary_pixels'], 1)
        self.assertEqual(result['P1_P0_flipped_pixels'], 4)

    def test_fixed_batch_real_update_and_output(self):
        batch = self.batch()
        initialize_rates(self.model, .01, .001)
        before, initial_p0 = evaluate(self.model, batch, None, .45)
        self.assertAlmostEqual(before['transport_rate_mean'], .01, places=7)
        self.assertAlmostEqual(before['proposal_rate_mean'], .001, places=7)
        self.assertEqual(len(before['steps']), 3)
        self.assertGreater(before['transport_abs_mean'], 0.)
        config = {'lambda_relation': .1, 'source_training': {
            'source_losses': {'lambda_cont': .2}, 'labels': asdict(RelationLabels())}}
        args = SimpleNamespace(variant='B', config=config)
        optimizer = parameter_groups(self.model)
        criterion = original_criterion(self.saved, 'cpu')
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        output = self.model(batch['image'])
        loss, _ = batch_loss(args, self.model, criterion, output, batch, 1234)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        after, _ = evaluate(self.model, batch, initial_p0, .45)
        self.assertGreater(after['P0_change_from_initial_abs_mean'], 0.)
        self.assertGreater(after['transport_abs_mean'], 0.)
        before.update(update=0, loss=None)
        after.update(update=1, loss=float(loss.detach()))
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.rate_startup_test_') as folder:
            write_results(Path(folder), {'seed': 1234}, [before, after])
            report = json.loads((Path(folder)/'startup_report.json').read_text())
            self.assertEqual([row['update'] for row in report['history']], [0, 1])
            self.assertIn('repair_added_fp', (Path(folder)/'startup_metrics.csv').read_text())

    def test_entrypoint_uses_one_batch_and_saves_only_diagnostics(self):
        batch = self.batch()
        class FixedDataset(torch.utils.data.Dataset):
            def set_epoch(self, epoch): self.epoch = epoch
            def __len__(self): return 2
            def __getitem__(self, index):
                return {**{key: value[index] for key, value in batch.items()}, 'case_name': str(index)}
        source, q = make_source()
        source['config'] = {'lambda_relation': .1, 'source_losses': {'lambda_cont': .2},
                            'labels': asdict(RelationLabels()), 'optimizer': {}}
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.rate_startup_main_') as folder:
            output = Path(folder)/'result'
            argv = ['audit', '--device', 'cpu', '--steps', '2', '--report_every', '1',
                    '--output_dir', str(output)]
            with patch.object(sys, 'argv', argv), \
                 patch.object(runner, 'load_h0', return_value=(self.model.base, {'epoch': 65}, self.saved)), \
                 patch.object(runner, 'inspect_source', return_value=(source, q, {'sha256': 'fixture'})), \
                 patch.object(runner, 'training_dataset', return_value=FixedDataset()), \
                 patch.object(runner, 'digest', return_value='fixture'), patch('builtins.print'):
                runner.main()
            report = json.loads((output/'startup_report.json').read_text())
            self.assertEqual([x['update'] for x in report['history']], [0, 1, 2])
            self.assertEqual(len(report['metadata']['training_cases']), 2)
            self.assertEqual(list(output.glob('*.pth')), [])


if __name__ == '__main__':
    unittest.main()
