"""Small bridge-route checks without the full training configuration stack."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from networks.reliable_directional_repair import ReliableDirectionalRepair, dynamic_losses


class FrozenConditions(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        stage3 = nn.Module()
        stage3.structure_branch = [SimpleNamespace(block=[nn.Conv2d(12, 12, 1)])]
        inner = nn.Module()
        inner.decoder_structure_blocks = nn.ModuleDict({'3': stage3})
        inner.highres_structure_channels = 8
        self.base.swin_unet = inner
        self.q_only = True
        self.surface = nn.Conv2d(3, 1, 1)
        self.h = nn.Conv2d(3, 8, 1)
        self.stage = nn.Conv2d(3, 12, 1)

    def forward(self, image, capture_diagnostics=False):
        low = F.avg_pool2d(image, 4)
        return {'z0': self.surface(image), 'q': low[:, :1].sigmoid(),
                'highres_h': self.h(low), 'stage3_h': self.stage(low)}


class BridgeRouteTests(unittest.TestCase):
    def test_forward_and_gradients(self):
        torch.manual_seed(12)
        model = ReliableDirectionalRepair(FrozenConditions()).train()
        model.set_accept_floor(.15)
        image = torch.randn(1, 3, 32, 32)
        result = model(image, capture_diagnostics=True)
        self.assertEqual(result['w'].shape, (1, 8, 8, 8))
        self.assertTrue(bool(torch.isfinite(result['p1']).all()))
        self.assertTrue(bool((result['c_support'] >= 0).all()))
        self.assertTrue(bool((result['c_support'] <= 1).all()))
        mask = (torch.rand_like(result['p0']) > .5).float()
        skeleton = mask.clone()
        loss, terms = dynamic_losses(result, mask, skeleton)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(torch.allclose(loss, terms['seg'] + .2 * terms['cont']))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.frozen.parameters()))
        for name in ('relations', 'proposal_head', 'accept_head', 'dynamics'):
            total = sum(float(p.grad.abs().sum()) for p in getattr(model, name).parameters()
                        if p.grad is not None)
            self.assertGreater(total, 0., name)
        for step in range(3):
            delta = sum(result[key][:, step] for key in
                        ('transport_delta', 'proposal_delta', 'anchor_delta'))
            self.assertTrue(torch.allclose(result[f'u{step+1}'] - result[f'u{step}'],
                                           delta, atol=1e-6))

    def test_w_affects_proposal_acceptance_and_surface(self):
        torch.manual_seed(13)
        model = ReliableDirectionalRepair(FrozenConditions()).eval()
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            original = model(image)
            disabled = model(image, weights_override=torch.zeros_like(original['w']))
        for key in ('b', 'a', 'p1'):
            self.assertGreater(float((original[key] - disabled[key]).abs().sum()), 0., key)

    def test_legacy_head_shapes_remain_available(self):
        model = ReliableDirectionalRepair(FrozenConditions(), bridge_route=False)
        self.assertEqual(model.proposal_head[0].in_channels, 39)
        self.assertEqual(model.proposal_head[-1].out_channels, 1)
        self.assertEqual(model.accept_head[0].in_channels, 40)

    def test_missing_path_continuity_reaches_direction_weights(self):
        torch.manual_seed(14)
        model = ReliableDirectionalRepair(FrozenConditions()).train()
        model.set_accept_floor(.15)
        result = model(torch.randn(1, 3, 32, 32))
        result['p0'] = torch.zeros_like(result['p0'])
        skeleton = torch.zeros_like(result['p0'])
        skeleton[:, :, 16, 4:28] = 1.
        _, terms = dynamic_losses(result, skeleton, skeleton, return_live_terms=True)
        self.assertEqual(terms['paths'], 1)
        gradient = torch.autograd.grad(
            terms['cont_live'], model.relations.direction_head.weight)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
