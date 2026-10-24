import unittest

import numpy as np
import torch

from checks.test_q3_r3_repair import FakeBase
from networks.q3_r3_repair import Q3R3Repair
from networks.reliable_directional_repair import (
    DirectionalDynamics, ReliableDirectionalRepair, dynamic_losses, shifted)
from run_reliable_directional_mass import gap_lengths


class DirectionalRepairTests(unittest.TestCase):
    def test_shift_has_no_wraparound(self):
        tensor = torch.zeros(1, 1, 8, 8)
        tensor[0, 0, 0, 0] = 1.
        moved = shifted(tensor, -1, 0)
        self.assertEqual(float(moved[0, 0, 1, 0]), 1.)
        self.assertEqual(float(moved[0, 0, 7, 0]), 0.)

    def test_no_source_closes_transport(self):
        dynamics = DirectionalDynamics(32)
        p0 = torch.full((1, 1, 8, 8), .4)
        visual = torch.zeros((1, 32, 8, 8))
        weights = torch.full((1, 8, 8, 8), .1)
        states, masses, avail = dynamics(visual, p0, p0, torch.zeros_like(p0),
                                         p0, torch.zeros_like(p0), weights)
        self.assertEqual(len(states), 4)
        self.assertTrue(all(float(item.detach().max()) == 0. for item in masses + avail))
        self.assertTrue(all(torch.allclose(item, p0, atol=1e-6) for item in states))

    def test_length_binned_gap_closure_uses_fixed_p0(self):
        ground = np.zeros((64, 64), dtype=bool)
        ground[32, 5:59] = True
        p0 = np.zeros_like(ground)
        p0[32, 5:13] = True
        p0[32, 49:59] = True
        bins, bridges = gap_lengths(p0, ground, ground, ground)
        self.assertEqual(sum(value['eligible'] for value in bins.values()), 1)
        self.assertEqual(sum(value['closed'] for value in bins.values()), 1)
        self.assertEqual(bridges, 0)

    def test_frozen_conditions_three_steps_and_gradients(self):
        torch.manual_seed(17)
        old = Q3R3Repair(FakeBase())
        model = ReliableDirectionalRepair(old)
        model.train()
        image = torch.randn(2, 3, 32, 32)
        mask = (torch.rand(2, 1, 32, 32) > .5).float()
        skeleton = (torch.rand(2, 1, 32, 32) > .8).float()
        result = model(image, capture_diagnostics=True)
        self.assertEqual(tuple(result['u3'].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(result['p1'].shape), (2, 1, 32, 32))
        for key in ('r', 'w', 'b', 'a', 'u0', 'u1', 'u2', 'u3', 'p1'):
            self.assertTrue(bool(torch.isfinite(result[key]).all()), key)
            self.assertGreaterEqual(float(result[key].detach().min()), 0., key)
            self.assertLessEqual(float(result[key].detach().max()), 1., key)
        self.assertTrue(bool((result['w'].sum(1) <= 1.000001).all()))
        self.assertLess(float((result['p1'] - result['p0']).detach().abs().mean()), .02)
        without_w, _, _ = model.dynamics(
            result['visual'], result['u0'], result['q'], result['r'],
            result['b'], result['a'], torch.zeros_like(result['w']))
        self.assertGreater(float((without_w[-1] - result['u3']).detach().abs().mean()), 0.)
        loss, stats = dynamic_losses(result, mask, skeleton)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(stats['paths'], 0)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.frozen.parameters()))
        for name in ('reliable_head', 'relations', 'proposal_head', 'accept_head',
                     'dynamics', 'detail_head'):
            module = getattr(model, name)
            total = sum(float(p.grad.abs().sum()) for p in module.parameters()
                        if p.grad is not None)
            self.assertGreater(total, 0., name)

    def test_diagnostic_overrides_preserve_baseline_and_expose_bottleneck(self):
        torch.manual_seed(23)
        model = ReliableDirectionalRepair(Q3R3Repair(FakeBase())).eval()
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            baseline = model(image, capture_diagnostics=True)
            repeated = model(image, diagnostic_override={})
            final_open = model(image, diagnostic_override={'final_a_one': True})
            internal_open = model(image, diagnostic_override={'internal_a_one': True})
            proposal = model(image, diagnostic_override={'proposal_shift': .5})
            rotated = model(image, diagnostic_override={'rotate_w': True})
        self.assertTrue(torch.equal(baseline['p1'], repeated['p1']))
        self.assertGreater(float((final_open['p1'] - baseline['p1']).abs().mean()), 0.)
        self.assertGreater(float((internal_open['u3'] - baseline['u3']).abs().mean()), 0.)
        self.assertGreater(float((proposal['u3'] - baseline['u3']).abs().mean()), 0.)
        self.assertGreater(float((rotated['u3'] - baseline['u3']).abs().mean()), 0.)
        self.assertEqual(tuple(baseline['rate_coefficients'].shape), (1, 3, 4, 8, 8))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_512_batch_two_forward_backward(self):
        model = ReliableDirectionalRepair(Q3R3Repair(FakeBase())).cuda()
        image = torch.randn(2, 3, 512, 512, device='cuda')
        mask = torch.zeros(2, 1, 512, 512, device='cuda')
        mask[:, :, 250:258, 40:472] = 1.
        skeleton = torch.zeros_like(mask)
        skeleton[:, :, 254, 40:472] = 1.
        torch.cuda.reset_peak_memory_stats()
        result = model(image)
        loss, _ = dynamic_losses(result, mask, skeleton)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool((result['p1'] >= 0).all()))
        self.assertTrue(bool((result['p1'] <= 1).all()))
        self.assertGreater(float(model.relations.direction_head.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.frozen.parameters()))
        print(f'fake_base_cuda_peak_mib={torch.cuda.max_memory_allocated()/2**20:.1f}')


if __name__ == '__main__':
    unittest.main()
