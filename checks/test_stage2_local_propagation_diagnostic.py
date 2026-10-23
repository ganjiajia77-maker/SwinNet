import unittest

import torch

from diagnose_stage2_local_propagation import propagation_weight_maps, set_propagation_beta


class Stage2PropagationDiagnosticTests(unittest.TestCase):
    def test_beta_can_be_disabled_and_restored_without_tracking_gradients(self):
        class Propagation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.beta = torch.nn.Parameter(torch.tensor(.25))

        propagation = Propagation()
        saved_beta = propagation.beta.detach().clone()
        set_propagation_beta(propagation, 0.)
        self.assertEqual(float(propagation.beta.detach()), 0.)
        set_propagation_beta(propagation, saved_beta)
        self.assertEqual(float(propagation.beta.detach()), .25)
        self.assertTrue(propagation.beta.requires_grad)

    def test_effective_weights_match_local_propagation_denominator(self):
        skeleton = torch.ones((1, 1, 3, 3))
        connectivity = torch.full((1, 8, 3, 3), .5)
        raw, effective = propagation_weight_maps(skeleton, connectivity)
        self.assertEqual(tuple(raw.shape), (1, 8, 3, 3))
        self.assertEqual(tuple(effective.shape), (1, 8, 3, 3))
        torch.testing.assert_close(effective[0, :, 1, 1], torch.full((8,), .1))

    def test_no_skeleton_or_no_connectivity_means_no_propagation_weight(self):
        skeleton = torch.zeros((1, 1, 4, 4))
        connectivity = torch.ones((1, 8, 4, 4))
        raw, effective = propagation_weight_maps(skeleton, connectivity)
        self.assertEqual(float(raw.abs().sum()), 0.)
        self.assertEqual(float(effective.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
