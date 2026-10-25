import unittest

import numpy as np
import torch

from checks.test_q3_r3_repair import FakeBase
from networks.q3_r3_repair import Q3R3Repair
from networks.reliable_directional_repair import (
    DirectionalDynamics, ReliableDirectionalRepair, compose_surface_logits,
    dynamic_losses, missed_skeleton_segments, shifted)
from run_reliable_directional_mass import accept_floor_at_epoch, gap_lengths


class DirectionalRepairTests(unittest.TestCase):
    def test_logit_update_is_identity_for_unchanged_working_grid(self):
        z0 = torch.randn(1, 1, 32, 32)
        p0_work = torch.sigmoid(torch.randn(1, 1, 8, 8))
        unchanged, delta = compose_surface_logits(z0, p0_work, p0_work)
        self.assertTrue(torch.equal(unchanged, z0))
        self.assertTrue(bool((delta == 0).all()))
        updated, _ = compose_surface_logits(z0, p0_work, (p0_work + .1).clamp_max(1.))
        self.assertGreater(float((updated.sigmoid() - z0.sigmoid()).abs().mean()), .01)

    def test_accept_floor_anneals_to_zero(self):
        self.assertAlmostEqual(accept_floor_at_epoch(1), .15)
        self.assertAlmostEqual(accept_floor_at_epoch(5), .15 * 5 / 9)
        self.assertEqual(accept_floor_at_epoch(10), 0.)
        self.assertEqual(accept_floor_at_epoch(20), 0.)

    def test_continuity_paths_only_cover_missing_skeleton(self):
        skeleton = torch.zeros(1, 1, 32, 32)
        skeleton[0, 0, 16, 3:29] = 1.
        p0 = torch.zeros_like(skeleton)
        p0[0, 0, 16, 3:11] = 1.
        p0[0, 0, 16, 21:29] = 1.
        paths = missed_skeleton_segments(skeleton, p0)
        self.assertEqual(len(paths[0]), 1)
        self.assertGreaterEqual(len(paths[0][0][0]), 3)
        self.assertTrue(all(12 <= x <= 19 for x in paths[0][0][1]))

    def test_missing_path_continuity_reaches_direction_weights(self):
        torch.manual_seed(31)
        model = ReliableDirectionalRepair(Q3R3Repair(FakeBase())).train()
        model.set_accept_floor(.15)
        result = model(torch.randn(1, 3, 32, 32))
        result['p0'] = torch.zeros_like(result['p0'])
        skeleton = torch.zeros_like(result['p0'])
        skeleton[0, 0, 16, 4:28] = 1.
        loss, terms = dynamic_losses(result, skeleton, skeleton,
                                     return_live_terms=True)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(terms['missing_segments'], 1)
        self.assertEqual(terms['paths'], 1)
        self.assertTrue(np.isfinite(terms['path_soft_min_mean']))
        direction_grad = torch.autograd.grad(
            terms['cont_live'], model.relations.direction_head.weight,
            retain_graph=True)[0]
        self.assertGreater(float(direction_grad.abs().sum()), 0.)

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

    def test_dynamics_reports_effective_rates_and_transport_target(self):
        dynamics = DirectionalDynamics(32)
        p0 = torch.rand(1, 1, 8, 8)
        visual = torch.randn(1, 32, 8, 8)
        weights = torch.full((1, 8, 8, 8), .08)
        states, masses, available, detail = dynamics(
            visual, p0, p0, p0, p0, torch.full_like(p0, .5), weights,
            capture_diagnostics=True)
        self.assertEqual(detail['alpha'].shape, (1, 3, 1, 8, 8))
        for step in range(3):
            reconstructed = (detail['stay'][:, step] * states[step] +
                             detail['alpha'][:, step] * detail['m'][:, step] +
                             detail['beta'][:, step] * p0 +
                             detail['gamma'][:, step] * p0)
            self.assertTrue(torch.allclose(states[step+1], reconstructed, atol=1e-6))
            self.assertTrue(bool((available[step] <= 1).all()))

    def test_direction_reroute_changes_proposal_and_accept_only_in_v3(self):
        torch.manual_seed(37)
        model = ReliableDirectionalRepair(Q3R3Repair(FakeBase())).eval()
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            baseline = model(image)
            rotated = baseline['w'].roll(2, dims=1)
            changed = model(image, weights_override=rotated)
            self.assertGreater(float((changed['b'] - baseline['b']).abs().sum()), 0.)
            self.assertGreater(float((changed['a'] - baseline['a']).abs().sum()), 0.)
            model.legacy_route = True
            legacy = model(image)
            legacy_changed = model(image, weights_override=legacy['w'].roll(2, dims=1))
            self.assertTrue(torch.allclose(legacy['b'], legacy_changed['b'], atol=1e-6))
            self.assertTrue(torch.allclose(legacy['a'], legacy_changed['a'], atol=1e-6))

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
        model.set_accept_floor(.15)
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
        self.assertTrue(torch.allclose(result['a'],
            .15 + .85 * result['a_raw'], atol=1e-6))
        self.assertGreater(float((result['p1'] - result['p0']).detach().abs().mean()), 0.)
        without_w, _, _ = model.dynamics(
            result['visual'], result['u0'], result['q'], result['r'],
            result['b'], result['a'], torch.zeros_like(result['w']))
        self.assertGreater(float((without_w[-1] - result['u3']).detach().abs().mean()), 0.)
        no_w_z1, _ = compose_surface_logits(result['z0'], result['u0'], without_w[-1])
        self.assertGreater(float((no_w_z1.sigmoid() - result['p1']).abs().mean()), 0.)
        loss, stats = dynamic_losses(result, mask, skeleton)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(stats['paths'], 0)
        self.assertTrue(torch.allclose(loss, stats['seg'] + .2 * stats['cont'], atol=1e-6))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.frozen.parameters()))
        for name in ('reliable_head', 'relations', 'proposal_head', 'accept_head',
                     'dynamics'):
            module = getattr(model, name)
            total = sum(float(p.grad.abs().sum()) for p in module.parameters()
                        if p.grad is not None)
            self.assertGreater(total, 0., name)
        model.eval()
        with torch.no_grad():
            inference = model(image)
        self.assertTrue(torch.equal(inference['a'], inference['a_raw']))

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
