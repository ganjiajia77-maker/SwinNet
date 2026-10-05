import tempfile
import unittest

import numpy as np
import torch

from datasets.anchor_reachability import ReachabilityCache, build_graph, lookup_distance
from losses.anchor_connectivity import connection_targets, supervised_connection_loss
from networks.supervised_anchor_topology import SupervisedAnchorTopology


class AnchorTests(unittest.TestCase):
    def test_exact_local_reachability_and_unconnected_parallel_roads(self):
        mask = np.zeros((48, 48), np.float32)
        mask[8, 3:44] = 1
        mask[18, 3:44] = 1
        # Joined only via a long U-shaped detour: same component is not enough.
        mask[8:19, 43] = 1
        graph = build_graph(mask)
        coords = torch.tensor([[[8, 5], [8, 12], [18, 5], [0, 0]]])
        pairs = torch.tensor([[[[0, 1], [0, 2], [0, 3]]]])
        output = dict(anchor_coords=coords, edge_pairs=pairs,
                      edge_valid=torch.ones(1, 1, 3, dtype=torch.bool),
                      edge_logits=torch.zeros(1, 1, 3, requires_grad=True))
        labels, valid = connection_targets(output, [graph])
        self.assertEqual(labels[0, 0, :2].tolist(), [1, 0])
        self.assertEqual(valid.tolist(), [[[True, True, False]]])
        loss, stats = supervised_connection_loss(output, [graph])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['connection_positive'].item(), 1)
        self.assertEqual(stats['connection_negative'].item(), 1)
        self.assertEqual(output['edge_logits'].grad[0, 0, 2].item(), 0)

    def test_empty_gt_cache_and_exact_cache_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ReachabilityCache(directory)
            blank = cache.get(np.zeros((24, 24)))
            self.assertEqual(int(blank['node_count']), 0)
            self.assertTrue(np.isinf(lookup_distance(blank, [0], [0])).all())
            mask = np.zeros((24, 24))
            mask[12, 2:22] = 1
            first = cache.get(mask)
            second = ReachabilityCache(directory).get(mask)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])

    def test_bounds_symmetry_gradients_and_saved_ramp(self):
        torch.manual_seed(17)
        module = SupervisedAnchorTopology(16, 8, max_nodes=8, heads=4,
                                          neighbours=4, hidden=16, max_distance=64)
        module.enable_global_topology = True
        feature = torch.randn(2, 16, 32, 32, requires_grad=True)
        structure = torch.randn(2, 8, 8, 8, requires_grad=True)
        surface = torch.rand(2, 1, 32, 32)
        connectivity = torch.rand(2, 8, 8, 8)
        output = module.forward_feature_anchors(feature, structure, surface, connectivity)
        torch.testing.assert_close(output, feature)  # zero-initialized residual scale
        graph = module.last_output
        self.assertLessEqual(graph['edge_valid'].sum().item(), 2 * 8 * 4)
        for batch in range(2):
            seen = {}
            for pair, logit, valid in zip(graph['edge_pairs'][batch].reshape(-1, 2),
                                          graph['edge_logits'][batch].flatten(),
                                          graph['edge_valid'][batch].flatten()):
                if valid:
                    key = tuple(pair.tolist())
                    if key in seen:
                        torch.testing.assert_close(logit, seen[key])
                    seen[key] = logit
        loss = graph['edge_logits'].square().mean() + output.square().mean()
        loss.backward()
        self.assertGreater(module.connection_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(feature.grad).all())
        module.set_epoch(9)
        self.assertEqual(module.prior_strength.item(), 0)
        module.set_epoch(14)
        self.assertEqual(module.prior_strength.item(), 1)
        clone = SupervisedAnchorTopology(16, 8, max_nodes=8, heads=4,
                                         neighbours=4, hidden=16, max_distance=64)
        clone.load_state_dict(module.state_dict(), strict=True)
        self.assertEqual(clone.prior_strength.item(), 1)

    def test_bilinear_writeback_location_confidence_and_identity(self):
        module = SupervisedAnchorTopology(4, 4, max_nodes=4, heads=1, hidden=4, neighbours=2)
        module.raw_alpha.data.fill_(0.5)
        feature = torch.randn(1, 4, 8, 8)
        positions = torch.tensor([[[2.0, 3.0]]])
        values = torch.ones(1, 1, 4)
        low, _ = module._writeback(feature, positions, values, torch.tensor([[[0.1]]]))
        high, _ = module._writeback(feature, positions, values, torch.tensor([[[0.9]]]))
        difference = high - feature
        self.assertEqual(torch.count_nonzero(difference).item(), 4)
        self.assertGreater((high - feature).abs().sum(), (low - feature).abs().sum())
        unchanged, _ = module._writeback(feature, positions, values, torch.zeros(1, 1, 1))
        torch.testing.assert_close(unchanged, feature)

    def test_active_residual_has_attention_gate_and_scatter_gradients(self):
        torch.manual_seed(7)
        module = SupervisedAnchorTopology(8, 8, max_nodes=8, neighbours=4,
                                          hidden=8, heads=2, enabled=True)
        module.raw_alpha.data.fill_(0.3)
        module.set_epoch(14)
        feature = torch.randn(1, 8, 16, 16, requires_grad=True)
        output = module.forward_feature_anchors(feature, torch.randn(1, 8, 4, 4),
                                                torch.rand(1, 1, 16, 16))
        output.square().mean().backward()
        for layer in (module.write_projection, module.write_gate, module.qkv,
                      module.connection_head[-1]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_no_valid_candidates_has_no_nan_or_writeback(self):
        module = SupervisedAnchorTopology(8, 8, max_nodes=8, neighbours=4, hidden=8, heads=2, enabled=True)
        feature = torch.randn(1, 8, 16, 16)
        output = module.forward_feature_anchors(feature, torch.zeros(1, 8, 4, 4),
                                                torch.zeros(1, 1, 16, 16))
        torch.testing.assert_close(output, feature)
        self.assertEqual(module.last_output['edge_valid'].sum().item(), 0)
        self.assertTrue(torch.isfinite(output).all())

    def test_ambiguous_parallel_branch_snap_is_ignored(self):
        mask = np.zeros((24, 24), np.float32)
        mask[8, 2:22] = 1
        mask[12, 2:22] = 1
        graph = build_graph(mask)
        self.assertTrue(graph['ambiguous'][10, 10])
        output = dict(anchor_coords=torch.tensor([[[10, 10], [8, 16]]]),
                      edge_pairs=torch.tensor([[[[0, 1]]]]),
                      edge_valid=torch.ones(1, 1, 1, dtype=torch.bool),
                      edge_logits=torch.zeros(1, 1, 1, requires_grad=True))
        loss, stats = supervised_connection_loss(output, [graph])
        self.assertEqual(stats['connection_valid'].item(), 0)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(output['edge_logits'].grad.abs().sum().item(), 0)

    def test_incomplete_connection_checkpoint_is_rejected(self):
        from networks.vision_transformer import load_topology_checkpoint_state
        wrapper = torch.nn.Module()
        wrapper.swin_unet = torch.nn.Module()
        wrapper.swin_unet.global_topology_mode = 'supervised_anchors'
        wrapper.swin_unet.global_topology = SupervisedAnchorTopology(8, 8, max_nodes=8,
                                                                     neighbours=4, hidden=8, heads=2)
        state = wrapper.state_dict()
        del state['swin_unet.global_topology.connection_head.2.weight']
        with self.assertRaisesRegex(RuntimeError, 'complete trained'):
            load_topology_checkpoint_state(wrapper, state, 'stage-topology-roadness-gap-v1')


if __name__ == '__main__':
    unittest.main()
