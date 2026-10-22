"""Run with: python -m unittest discover -s checks -v (CPU is sufficient)."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch

from local_gap_config import LocalGapConfig
from networks.local_gap_connectivity import (LocalConnectivityPropagation, GapConnectivity,
                                              path_coordinates, sample_at, rasterize_evidence)
from networks.dca_fpn_lite import DCAFPNLite
from gap_candidates import generate_candidates
from losses.gap_losses import label_edges, gap_auxiliary_loss
from losses.road_losses import build_stage_connectivity_target, SurfaceStructureLoss

torch.set_num_threads(4)


def road_probability(size=32):
    result = np.zeros((1, size, size), np.float32) + .01
    result[0, size//2, 4:12] = .99
    result[0, size//2, 20:28] = .99
    return result


class LocalGapTests(unittest.TestCase):
    def test_fine_edges_do_not_invent_parallel_connection(self):
        s = torch.zeros(1, 1, 16, 16)
        s[0, 0, 4, 2:14] = 1
        s[0, 0, 11, 2:14] = 1
        old = build_stage_connectivity_target(s, (4, 4), 'pooled_adjacency')
        new = build_stage_connectivity_target(s, (4, 4))
        self.assertEqual(old[0, 4, 1, 1], 1)
        self.assertEqual(new[0, 4, 1, 1], 0)
        self.assertEqual(new[0, 2, 1, 1], 1)
        self.assertEqual(new[0, 6, 1, 2], 1)

    def test_fine_edges_preserve_gap_and_diagonal_mapping(self):
        s = torch.zeros(1, 1, 16, 16)
        s[0, 0, 5, 2:4] = 1
        s[0, 0, 5, 7:11] = 1
        new = build_stage_connectivity_target(s, (4, 4))
        self.assertEqual(new[0, 2, 1, 0], 0)
        s.zero_(); s[0, 0, 3, 3] = s[0, 0, 4, 4] = 1
        new = build_stage_connectivity_target(s, (4, 4))
        self.assertEqual(new[0, 3, 0, 0], 1)
        self.assertEqual(new[0, 7, 1, 1], 1)

    def test_local_reads_senders_reverse_direction_and_receives_without_own_s(self):
        module = LocalConnectivityPropagation(1)
        module.projection.weight.data.fill_(1); module.beta.data.fill_(1)
        x = torch.zeros(1, 1, 5, 5); x[0, 0, 2, 3] = 4
        s = torch.zeros_like(x); s[0, 0, 2, 3] = 1
        c = torch.zeros(1, 8, 5, 5); c[0, 6, 2, 3] = 1
        self.assertEqual(module(x, s, c)[0, 0, 2, 2], 2)
        c.zero_(); c[0, 2, 2, 3] = 1
        self.assertEqual(module(x, s, c)[0, 0, 2, 2], 0)

    def test_local_no_wrap_and_zero_confidence_identity(self):
        module = LocalConnectivityPropagation(1)
        module.projection.weight.data.fill_(1); module.beta.data.fill_(1)
        x = torch.zeros(1, 1, 5, 5); x[..., 0] = 8
        s = torch.ones_like(x); c = torch.zeros(1, 8, 5, 5); c[:, 6, :, 0] = 1
        self.assertTrue(torch.equal(module(x, s, c), x))
        self.assertTrue(torch.equal(module(x, s*0, c), x))

    def test_local_surface_gradient_reaches_s_and_c(self):
        module = LocalConnectivityPropagation(2); module.beta.data.fill_(.1)
        x = torch.randn(1, 2, 5, 5, requires_grad=True)
        s = torch.randn(1, 1, 5, 5, requires_grad=True)
        c = torch.randn(1, 8, 5, 5, requires_grad=True)
        module(x, s.sigmoid(), c.sigmoid()).square().sum().backward()
        self.assertGreater(float(s.grad.abs().sum()), 0)
        self.assertGreater(float(c.grad.abs().sum()), 0)

    def test_dca_zero_offset_samples_exact_centers(self):
        module = DCAFPNLite(4).eval()
        for parameter in module.offset_conv.parameters(): parameter.data.zero_()
        module.align_shallow.weight.data.zero_(); module.align_shallow.bias.data.zero_()
        for i in range(4): module.align_shallow.weight.data[i, i] = 1
        seen = []
        hook = module.fusion_conv.register_forward_pre_hook(lambda m, a: seen.append(a[0]))
        x = torch.arange(64*64, dtype=torch.float32).reshape(1, 1, 64, 64).repeat(1, 4, 1, 1)
        module(x, x); hook.remove()
        torch.testing.assert_close(seen[0], torch.cat([x]*4, 1), atol=.002, rtol=0)

    def test_proposals_are_bounded_and_tangent_filtered(self):
        cfg = LocalGapConfig()
        result = generate_candidates(road_probability(), cfg)[0]
        self.assertEqual(len(result['edges']), 1)
        self.assertLessEqual(len(result['edges']), cfg.max_edges)
        np.testing.assert_array_equal(result['edges'][0, :, 1], [11, 20])
        vertical = np.zeros((1, 32, 32), np.float32)
        vertical[0, 4:28, 10] = 1; vertical[0, 4:28, 20] = 1
        self.assertEqual(len(generate_candidates(vertical, cfg)[0]['edges']), 0)

    def test_labels_local_path_negative_and_bent_ignore(self):
        cfg = LocalGapConfig()
        edge = np.array([[[16, 11], [16, 20]]], np.float32)
        skel = np.zeros((128, 128), bool); skel[66, 16:112] = True
        labels, _ = label_edges(skel, skel, edge, (32, 32), cfg)
        self.assertEqual(labels.tolist(), [1])
        skel[66, 58:72] = False
        labels, _ = label_edges(skel, skel, edge, (32, 32), cfg)
        self.assertEqual(labels.tolist(), [0])
        skel[40:67, 46] = True; skel[40, 46:83] = True; skel[40:67, 82] = True
        labels, _ = label_edges(skel, skel, edge, (32, 32), cfg)
        self.assertEqual(labels.tolist(), [-1])

    def test_sampling_orientation_and_center_alignment(self):
        image = torch.arange(32.).view(1, 1, 1, 32).expand(1, 1, 32, 32)
        edges = torch.tensor([[[16., 11.], [16., 20.]]])
        a = sample_at(image, path_coordinates(edges, 32, 1))
        b = sample_at(image, path_coordinates(edges.flip(1), 32, 1))
        torch.testing.assert_close(a, b.flip((1, 2)))
        torch.testing.assert_close(a[0, 1, [0, -1], 0], torch.tensor([11., 20.]))

    def test_gap_symmetric_gradient_and_zero_evidence_identity(self):
        cfg = LocalGapConfig()
        module = GapConnectivity(4, 3, cfg)
        module.gamma.data.fill_(.1); module.set_epoch(20)
        x = torch.randn(1, 4, 32, 32, requires_grad=True)
        h = torch.randn(1, 3, 32, 32)
        p2 = torch.tensor(road_probability()).unsqueeze(1).logit().requires_grad_()
        output, record = module(x, h, p2, p2.sigmoid())
        self.assertEqual(len(record['gap_logits'][0]), 1)
        proposal = generate_candidates(road_probability(), cfg)[0]
        reverse = dict(proposal, edges=proposal['edges'][:, ::-1].copy(), geometry=proposal['geometry'].copy())
        reverse['geometry'][:, 1:3] = proposal['geometry'][:, [2, 1]]
        reverse['geometry'][:, 4:6] *= -1
        with patch('networks.local_gap_connectivity.generate_candidates', return_value=[reverse]):
            _, swapped = module(x, h, p2, p2.sigmoid())
        torch.testing.assert_close(record['gap_logits'][0], swapped['gap_logits'][0])
        output.square().mean().backward()
        self.assertGreater(float(module.edge_mlp[-1].weight.grad.abs().sum()), 0)
        blank, empty = module(x.detach(), h, torch.full_like(p2, -10), p2.detach().sigmoid())
        self.assertTrue(torch.equal(blank, x.detach()))
        self.assertEqual(len(empty['gap_edges'][0]), 0)

    def test_overlap_normalization_preserves_confidence(self):
        edge = torch.tensor([[[16., 5.], [16., 25.]]])
        a = rasterize_evidence(edge, torch.tensor([.2]), 32, 32, 1.5)
        b = rasterize_evidence(edge.repeat(2, 1, 1), torch.tensor([.2, .2]), 32, 32, 1.5)
        self.assertAlmostEqual(float(a[0,0,16,15]), .2, places=6)
        torch.testing.assert_close(a[0,0,16,5:26], b[0,0,16,5:26])
        self.assertLessEqual(float(b.max()), .200001)

    def test_schedule_persists_on_eval_checkpoint(self):
        a = GapConnectivity(4, 3); a.set_epoch(9)
        b = GapConnectivity(4, 3); b.load_state_dict(a.state_dict()); b.eval()
        self.assertEqual(float(b.writeback_scale), 0)
        a.set_epoch(12); b.load_state_dict(a.state_dict())
        self.assertAlmostEqual(float(b.writeback_scale), .6, places=6)


if __name__ == '__main__':
    unittest.main()
