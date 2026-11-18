"""Run with Python directly; no pytest dependency needed."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_h0_stage2_data1 as evaluate


class Checks(unittest.TestCase):
    def test_custom_grid_exact_counts_and_boundary_inclusion(self):
        grid = evaluate.threshold_grid(.2, .65, .001)
        self.assertEqual(len(grid), 451)
        self.assertEqual(grid[0], np.float32(.2))
        self.assertEqual(grid[-1], np.float32(.65))
        p = grid[:12].reshape(3, 4)
        gt = np.arange(12).reshape(3, 4) % 2 == 0
        np.testing.assert_array_equal(evaluate.sweep_counts(p, gt, grid),
            [evaluate.common.counts(p >= t, gt) for t in grid])
        with self.assertRaises(ValueError):
            evaluate.threshold_grid(.65, .2, .001)

    def test_sweep_equality_and_all_background_foreground(self):
        random = np.random.default_rng(1234)
        probability = random.random((64, 64), dtype=np.float32)
        probability[0, :len(evaluate.THRESHOLDS[:64])] = evaluate.THRESHOLDS[:64]
        for gt in (random.random((64, 64)) > .5, np.zeros((64, 64), bool), np.ones((64, 64), bool)):
            actual = evaluate.sweep_counts(probability, gt)
            expected = np.array([evaluate.common.counts(probability >= t, gt) for t in evaluate.THRESHOLDS])
            np.testing.assert_array_equal(actual, expected)

    def test_native1024_nine_tiles_and_logit_fusion(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.starts = []
            def forward(self, tile):
                self.starts.append(int(tile[0, 0, 0, 0]))
                return torch.ones(1, 1, 512, 512),
        image = torch.arange(1024*1024, dtype=torch.float32).reshape(1, 1024, 1024).expand(3, -1, -1)
        model = Constant()
        result = evaluate.infer(model, image, torch.device('cpu'))
        self.assertEqual(model.starts, [y*1024+x for y in (0, 256, 512) for x in (0, 256, 512)])
        np.testing.assert_allclose(result, torch.tensor(1.).sigmoid().item(), rtol=1e-6)

    def test_explicit_ema_strict_restore_and_mismatch_rejection(self):
        model = torch.nn.Linear(2, 1)
        ema = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
        state = dict(args=dict(h0_data1=True, use_ema=True, source_patch_size=1024,
            img_size=512, amp_dtype='none'), epoch=77, model_state_dict=ema,
            ema_state_dict=ema, training_model_state_dict=model.state_dict())
        with patch.object(evaluate.torch, 'load', return_value=state), \
                patch.object(evaluate.common, 'build_model', return_value=model):
            restored, _, epoch = evaluate.load_model(Path('unused.pth'), torch.device('cpu'))
            self.assertEqual(epoch, 77)
            self.assertFalse(restored.training)
            self.assertTrue(all(torch.equal(p, torch.ones_like(p)) for p in restored.parameters()))
            state['model_state_dict'] = {k: v+1 for k, v in ema.items()}
            with self.assertRaisesRegex(ValueError, 'differ from EMA'):
                evaluate.load_model(Path('unused.pth'), torch.device('cpu'))

    def test_validation_selection_fixed_on_test_and_resumable_masks(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            checkpoint = root/'best.pth'
            checkpoint.write_bytes(b'checkpoint')
            args = SimpleNamespace(checkpoint=checkpoint, root_path=root/'data',
                output_dir=root/'report', device='cpu', workers=4,
                threshold_min=.2, threshold_max=.65, threshold_step=.001)
            calls = []
            def sample(data, i):
                return f'{data}_{i}', torch.zeros(3, 2, 2), np.array([[1, 0], [1, 0]], bool)
            def infer(*args):
                calls.append(1)
                return np.array([[.7, .2], [.7, .2]], np.float32)
            with patch.object(evaluate, 'EXPECTED', dict(val=2, test=2)), \
                    patch.object(evaluate, 'dataset', return_value=range(2)), \
                    patch.object(evaluate, 'read_sample', side_effect=sample), \
                    patch.object(evaluate, 'infer', side_effect=infer), \
                    patch.object(evaluate, 'SIZE', 2), \
                    patch.object(evaluate, 'load_model', return_value=(object(), {}, 77)):
                evaluate.validation(args)
                selected = json.loads((args.output_dir/'val_selection.json').read_text())
                self.assertEqual(selected['iou'], 1.)
                self.assertEqual(selected['selected_on'], 'val')
                self.assertEqual(len(json.loads((args.output_dir/'inference_protocol.json').read_text())['thresholds']), 451)
                evaluate.testing(args)
                manifest = json.loads((args.output_dir/'comparison.json').read_text())
                self.assertEqual(manifest['models'][0]['predictions']['test'][0]['threshold'], selected['threshold'])
                self.assertEqual(manifest['protocol']['apls_max_control_nodes'], 150)
                self.assertEqual(manifest['protocol']['topo_seed_density_deg'], .002)
                self.assertEqual(manifest['protocol']['topo_step_deg'], .0002)
                self.assertEqual(manifest['protocol']['tlts_n_paths'], 200)
                count = len(calls)
                evaluate.testing(args)
                self.assertEqual(len(calls), count)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
