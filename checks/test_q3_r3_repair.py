import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from networks.q3_r3_repair import Q3R3Repair, make_q_target, repair_losses
from run_q3_r3_mass import gap_proxy


class FakeStage3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Conv2d(3, 12, 1)
        self.structure_branch = [SimpleNamespace(block=[nn.Conv2d(12, 12, 1)])]

    def forward(self, x):
        return self.input_proj(x), None


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.decoder_structure_blocks = nn.ModuleDict({'2': nn.Module(), '3': FakeStage3()})
        inner.highres_structure_channels = 8
        inner.highres_proj = nn.Conv2d(3, 8, 1)
        inner.surface = nn.Conv2d(3, 1, 1)
        inner.skeleton = nn.Conv2d(3, 1, 1)
        inner.connectivity = nn.Conv2d(3, 8, 1)
        inner.last_stage_features = {}
        self.swin_unet = inner

    def forward(self, image):
        x = F.avg_pool2d(image, 4)
        self.swin_unet.decoder_structure_blocks['3'](x)
        self.swin_unet.last_stage_features['H2'] = self.swin_unet.highres_proj(x)
        structure = [{'stage': 2, 'skeleton': self.swin_unet.skeleton(x),
                      'connectivity': self.swin_unet.connectivity(x)}]
        return self.swin_unet.surface(image), None, None, None, structure


class RepairTests(unittest.TestCase):
    def test_initial_identity_and_trainable_gradient_path(self):
        torch.manual_seed(7)
        model = Q3R3Repair(FakeBase())
        model.train()
        self.assertFalse(model.base.training)
        image = torch.randn(2, 3, 32, 32)
        mask = (torch.rand(2, 1, 32, 32) > .5).float()
        skeleton = (torch.rand(2, 1, 32, 32) > .8).float()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
        result = model(image)
        self.assertEqual(tuple(result['q'].shape), (2, 1, 8, 8))
        torch.testing.assert_close(result['z0'], result['z1'])
        loss, _ = repair_losses(result, mask, skeleton, lambda_q=.1)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(float(model.q_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.r_head.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        result = model(image)
        self.assertGreater(float((result['z1'] - result['z0']).detach().abs().sum()), 0)
        surface_only, _ = repair_losses(result, mask, skeleton, lambda_q=0)
        surface_only.backward()
        self.assertGreater(float(model.q_head.weight.grad.abs().sum()), 0)

    def test_q_target_marks_missed_centerline(self):
        skeleton = torch.zeros((1, 1, 32, 32))
        skeleton[0, 0, 16, 8:24] = 1
        z0 = torch.full_like(skeleton, -8.)
        target = make_q_target(skeleton, z0, (8, 8))
        self.assertGreater(float(target.sum()), 0)

    def test_gap_proxy_detects_closed_reference_gap(self):
        gt = np.zeros((24, 24), dtype=bool)
        gt[12, 2:22] = True
        p0 = np.zeros_like(gt)
        p0[12, 2:5] = True
        p0[12, 15:22] = True
        p1 = gt.copy()
        eligible, closed, background_bridges = gap_proxy(p0, p1, gt, gt)
        self.assertGreaterEqual(eligible, 1)
        self.assertGreaterEqual(closed, 1)
        self.assertEqual(background_bridges, 0)


if __name__ == '__main__':
    unittest.main()
