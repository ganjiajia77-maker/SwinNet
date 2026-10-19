"""Verify direction-only interventions and the native1500 unified-report pipeline."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_direction as diagnostic
from diffusion_protocol import architecture, build_model
from networks.directional_diffusion import SemanticDirectionalDiffusion


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_unet = nn.Module()
        self.swin_unet.decoder_structure_blocks = nn.ModuleDict({'3': nn.Module()})
        self.swin_unet.decoder_structure_blocks['3'].diffusion = SemanticDirectionalDiffusion(
            channels=4, structure_channels=4, highres_channels=4, latent_channels=4)

    @property
    def diffusion(self):
        return self.swin_unet.decoder_structure_blocks['3'].diffusion

    def forward(self, image):
        small = F.adaptive_avg_pool2d(image, (12, 12))
        x = torch.cat((small, small.mean(1, keepdim=True)), 1)
        direction = torch.zeros_like(x[:, :2]); direction[:, 0] = 1
        output = self.diffusion(x, x.sin(), x.cos(), direction)
        return F.interpolate(output.mean(1, keepdim=True)*.01-10., size=image.shape[-2:], mode='bilinear'), None, None


class DirectionDiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'actual H0 paired inference uses CUDA')
    def test_actual_h0_512_paired_inference(self):
        torch.manual_seed(1234)
        model = build_model(architecture()).cuda().eval()
        diffusion = model.swin_unet.decoder_structure_blocks['3'].diffusion
        with torch.no_grad(): diffusion.out_proj.weight.normal_(std=.01)
        initial = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
        on, off, stats = diagnostic.paired_window(model, diffusion, torch.randn(1, 3, 512, 512, device='cuda'))
        self.assertEqual(on.shape, (1, 1, 512, 512))
        self.assertEqual(off.shape, on.shape)
        self.assertTrue(bool(torch.isfinite(on).all() and torch.isfinite(off).all()))
        self.assertGreater(stats['edge_abs_change_mean_valid'], 0.)
        self.assertGreater(stats['delta_x_abs_change_mean'], 0.)
        self.assertTrue(all(torch.equal(tensor.cpu(), initial[name]) for name, tensor in model.state_dict().items()))

    def test_intervention_only_changes_direction_and_restores_state(self):
        torch.manual_seed(1234)
        model = SmallModel().eval()
        image = torch.randn(1, 3, 16, 16)
        on, off, stats = diagnostic.paired_window(model, model.diffusion, image)
        self.assertGreater(stats['edge_abs_change_mean_valid'], 0.)
        self.assertEqual(stats['delta_x_abs_change_mean'], 0.)
        torch.testing.assert_close(on, off, rtol=0, atol=0)
        with torch.no_grad():
            model.diffusion.out_proj.weight.normal_()
        original = copy.deepcopy(model.state_dict())
        on, off, stats = diagnostic.paired_window(model, model.diffusion, image)
        self.assertGreater(stats['delta_x_abs_change_mean'], 0.)
        model.diffusion.mode = 'semantic_directional'
        torch.testing.assert_close(on, model(image)[0], rtol=0, atol=0)
        model.diffusion.mode = 'semantic'
        torch.testing.assert_close(off, model(image)[0], rtol=0, atol=0)
        model.diffusion.mode = 'semantic_directional'
        self.assertTrue(all(torch.equal(value, original[name]) for name, value in model.state_dict().items()))
        with patch.object(model, 'forward', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError): diagnostic.paired_window(model, model.diffusion, image)
        self.assertEqual(model.diffusion.mode, 'semantic_directional')
        self.assertFalse(model.diffusion.capture_diagnostics)
        self.assertEqual(len(model.diffusion._forward_hooks), 0)

    def test_native1500_pipeline_and_threshold_policies(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as temp:
            temp = Path(temp)
            for split in ('val', 'test'):
                (temp/'data'/split).mkdir(parents=True)
                (temp/'data'/f'{split}_labels').mkdir()
                image = np.zeros((1500, 1500, 3), np.uint8)
                label = np.zeros((1500, 1500), np.uint8)
                cv2.line(label, (400, 750), (1100, 750), 255, 5)
                cv2.imwrite(str(temp/'data'/split/'example.tiff'), image)
                cv2.imwrite(str(temp/'data'/f'{split}_labels'/'example.tif'), label)
            model = SmallModel().eval()
            checkpoint = {'epoch': 80, 'architecture_config': architecture(),
                          'args': {'seed': 1234}, 'model_state_dict': copy.deepcopy(model.state_dict())}
            path = temp/'synthetic.pth'
            torch.save(checkpoint, path)
            def load(_):
                return copy.deepcopy(model), copy.deepcopy(checkpoint)
            for action in ('val', 'test'):
                argv = ['diagnose_direction.py', action, '--model_path', str(path),
                        '--root_path', str(temp/'data'), '--output_dir', str(temp/'report'), '--gpu', 'cpu']
                with patch.object(diagnostic, 'load_checkpoint', side_effect=load), patch.object(sys, 'argv', argv):
                    with contextlib.redirect_stdout(io.StringIO()): diagnostic.main()
            fixed = json.loads((temp/'report/val/val_selection_fixed.json').read_text())
            own = json.loads((temp/'report/val/val_selection.json').read_text())
            for name in diagnostic.VARIANTS:
                self.assertEqual(fixed['models'][name]['identity'], own['models'][name]['identity'])
                self.assertEqual(fixed['models'][name]['threshold'], own['models']['direction_on']['threshold'])
            self.assertTrue((temp/'report/test_fixed/test_comparison.csv').is_file())
            self.assertTrue((temp/'report/test_own_val/test_comparison.csv').is_file())
            self.assertTrue((temp/'report/test_metric_deltas.json').is_file())
            mask = cv2.imread(str(temp/'report/predictions/test/direction_on/threshold_0.05/surface/example_pred.png'), 0)
            self.assertEqual(mask.shape, (1500, 1500))
            # Distinct val optima still become one normal-selected threshold in the primary report.
            selection = copy.deepcopy(own)
            selection['models']['direction_on']['threshold'] = .45
            selection['models']['direction_off']['threshold'] = .5
            scores = [{'name': name, 'threshold': .45, 'iou': .6} for name in diagnostic.VARIANTS]
            result = diagnostic.fixed_selection(selection, scores)
            self.assertEqual(result['models']['direction_off']['threshold'], .45)
            self.assertEqual(result['models']['direction_off']['independently_selected_threshold'], .5)
            self.assertEqual(selection['models']['direction_off']['threshold'], .5)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
