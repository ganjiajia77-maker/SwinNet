import unittest

import torch
from torch import nn

from diagnose_zero_val import set_probe_mode


class CIWM(nn.Sequential):
    pass


class MSWD(nn.Sequential):
    pass


class ProbeTest(unittest.TestCase):
    def make_model(self):
        model = nn.Module()
        model.encoder1 = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout())
        model.ciwm = CIWM(nn.BatchNorm2d(3), nn.Dropout())
        model.mswd = MSWD(nn.BatchNorm2d(3), nn.Dropout())
        model.other = nn.BatchNorm2d(3)
        return model

    def test_bn_groups_do_not_enable_dropout(self):
        for mode, active in (("eval", set()), ("bn_all", {"encoder1.0", "ciwm.0", "mswd.0", "other"}),
                             ("bn_encoder", {"encoder1.0"}), ("bn_ciwm", {"ciwm.0"}),
                             ("bn_mswd", {"mswd.0"})):
            model = self.make_model()
            set_probe_mode(model, mode)
            self.assertFalse(model.training)
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.assertEqual(module.training, name in active)
                elif isinstance(module, nn.Dropout):
                    self.assertFalse(module.training)

    def test_checkpoint_restores_running_stats(self):
        model = self.make_model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        set_probe_mode(model, "bn_all")
        model.encoder1(torch.ones(2, 3, 8, 8))
        self.assertFalse(torch.equal(model.encoder1[0].running_mean, state["encoder1.0.running_mean"]))
        model.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(model.encoder1[0].running_mean, state["encoder1.0.running_mean"]))


if __name__ == "__main__":
    unittest.main()
