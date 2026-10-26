"""Small bridge-route checks without the full training configuration stack."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from networks.reliable_directional_repair import (DirectionalDynamics,
                                                  MultiScaleGateRelation,
                                                  ReliableDirectionalRepair,
                                                  dynamic_losses)


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
    def test_v8_zero_residual_preserves_v7_and_freezes_public_path(self):
        torch.manual_seed(19)
        model = ReliableDirectionalRepair(FrozenConditions(), transport_gate_mode='v7',
                                          gate_relation_mode='full').eval()
        model.requires_grad_(False)
        model.gate_relation.requires_grad_(True)
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            v8 = model(image, capture_diagnostics=True)
            relation = model.gate_relation
            model.gate_relation = None
            try:
                v7 = model(image)
            finally:
                model.gate_relation = relation
        self.assertTrue(torch.equal(v8['g_transport'], v7['g_transport']))
        self.assertTrue(torch.equal(v8['p1'], v7['p1']))
        with torch.no_grad():
            model.gate_relation.delta[-1].bias.add_(.5)
            changed = model(image)
            model.gate_relation.delta[-1].bias.sub_(.5)
        for key in ('r', 'w', 'b', 'a'):
            self.assertTrue(torch.equal(changed[key], v7[key]), key)
        self.assertGreater(float((changed['g_transport'] - v7['g_transport']).abs().max()), 0.)
        self.assertGreater(float((changed['p1'] - v7['p1']).abs().max()), 0.)
        self.assertEqual(v8['delta_g_logit'].shape, (1, 1, 8, 8))
        self.assertEqual(v8['gate_source_coverage'].shape, (1, 1, 8, 8))
        loss = model(image)['p1'].mean()
        loss.backward()
        self.assertGreater(float(model.gate_relation.delta[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for name, module in model.named_children()
                            if name != 'gate_relation' for p in module.parameters()))

    def test_v8_reliable_sampling_does_not_wrap_and_modes_are_finite(self):
        relation = MultiScaleGateRelation(8, 12)
        feature = torch.zeros(1, 32, 8, 8)
        feature[..., 0, 0] = 1.
        reliable = torch.zeros(1, 1, 8, 8)
        reliable[..., 0, 0] = 1.
        far, coverage = relation._source(feature, reliable, -1, 0)
        self.assertEqual(float(coverage[..., 7, 0].max()), 0.)
        self.assertEqual(float(far[..., 7, 0].abs().max()), 0.)
        highres = torch.randn(1, 8, 8, 8)
        decoder = torch.randn(1, 12, 8, 8)
        p0 = torch.rand(1, 1, 8, 8)
        q = torch.rand_like(p0)
        w = torch.full((1, 8, 8, 8), .125)
        two = torch.rand(1, 4, 8, 8)
        one = torch.rand_like(two)
        for mode in ('local', 'bilateral', 'directional', 'full'):
            delta, cover = relation(highres, decoder, feature, p0, q,
                                    reliable, w, two, one, mode)
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
            self.assertTrue(torch.isfinite(cover).all())
            self.assertTrue(((cover >= 0) & (cover <= 1)).all())

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
        model = ReliableDirectionalRepair(FrozenConditions(), bridge_route=False,
                                          conditional_transport=False)
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
        weight_gradient, bias_gradient = torch.autograd.grad(
            terms['cont_live'], (model.relations.direction_head.weight,
                                 model.relations.direction_head.bias))
        self.assertGreater(float(weight_gradient.abs().sum()), 0.)
        self.assertGreater(float(bias_gradient[0].abs()), 0.)

    def test_conditional_weights_and_gate_are_separate(self):
        torch.manual_seed(15)
        model = ReliableDirectionalRepair(FrozenConditions()).eval()
        with torch.no_grad():
            result = model(torch.randn(1, 3, 32, 32), capture_diagnostics=True)
        self.assertTrue(torch.allclose(result['w'].sum(1),
                                       torch.ones_like(result['w'][:, 0]), atol=1e-6))
        self.assertTrue(bool((result['g_transport'] >= 0).all()))
        self.assertTrue(bool((result['g_transport'] <= 1).all()))
        self.assertGreater(float(result['source_available'].mean()), 0.)
        self.assertTrue(bool(torch.isfinite(result['transport_delta']).all()))

    def test_v7_changes_gate_evidence_without_changing_w_or_parameters(self):
        torch.manual_seed(151)
        v6 = ReliableDirectionalRepair(FrozenConditions(),
                                       transport_gate_mode='v6').eval()
        v7 = ReliableDirectionalRepair(FrozenConditions(),
                                       transport_gate_mode='v7').eval()
        v7.load_state_dict(v6.state_dict(), strict=True)
        image = torch.randn(1, 3, 40, 40)
        with torch.no_grad():
            result_v6 = v6(image, capture_diagnostics=True)
            result_v7 = v7(image, capture_diagnostics=True)
        self.assertTrue(torch.equal(result_v6['w'], result_v7['w']))
        self.assertTrue(torch.equal(result_v6['b'], result_v7['b']))
        self.assertTrue(torch.equal(result_v6['a'], result_v7['a']))
        self.assertFalse(torch.allclose(result_v6['g_transport'],
                                        result_v7['g_transport']))
        self.assertTrue(torch.equal(result_v7['gate_source_evidence'],
                                    result_v7['path_continuity_evidence']))
        self.assertTrue(bool(torch.isfinite(result_v6['path_continuity_evidence']).all()))

    def test_no_reliable_source_closes_permission(self):
        torch.manual_seed(16)
        model = ReliableDirectionalRepair(FrozenConditions()).eval()
        visual = torch.randn(1, 32, 8, 8)
        empty = torch.zeros(1, 1, 8, 8)
        with torch.no_grad():
            directions, _, _, permission = model.relations(visual, empty, empty)
        self.assertTrue(torch.allclose(directions.sum(1),
                                       torch.ones_like(empty[:, 0]), atol=1e-6))
        self.assertLess(float(permission.max()), 1e-3)

    def test_v5_parameter_shapes_load_strictly(self):
        old = ReliableDirectionalRepair(FrozenConditions(), conditional_transport=False)
        restored = ReliableDirectionalRepair(FrozenConditions(), conditional_transport=False)
        for name in ('visual', 'reliable_head', 'relations', 'proposal_head',
                     'accept_head', 'dynamics'):
            getattr(restored, name).load_state_dict(getattr(old, name).state_dict(),
                                                    strict=True)

    def test_newly_gained_road_can_become_next_step_source(self):
        dynamics = DirectionalDynamics(32)
        visual = torch.zeros(1, 32, 8, 8)
        p0 = torch.zeros(1, 1, 8, 8)
        p0[:, :, 4, 1] = .9
        reliable = p0.clone()
        proposal = torch.full_like(p0, .6)
        weights = torch.zeros(1, 8, 8, 8)
        weights[:, 0] = 1.
        states, _, available = dynamics(
            visual, p0, p0, reliable, proposal, torch.ones_like(p0), weights,
            transport_gate=torch.ones_like(p0))
        self.assertEqual(float(available[0][0, 0, 4, 3]), 0.)
        self.assertGreater(float(states[1][0, 0, 4, 2].detach()), 0.)
        self.assertGreater(float(available[1][0, 0, 4, 3].detach()), 0.)


if __name__ == '__main__':
    unittest.main()
