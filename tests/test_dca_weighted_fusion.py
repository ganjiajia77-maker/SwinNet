import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.road_comparison.build_surface_dca_weighted_mass_manifest import EXPECTED, verify_checkpoint
from networks.dca_fpn_lite import DCAFPNLite


ROOT = Path(__file__).resolve().parents[1]


def legacy_dca_class():
    source = subprocess.check_output(
        ["git", "show", "eb2da24d768b7f643e4fb288e3736693862c1fe8:networks/dca_fpn_lite.py"],
        cwd=ROOT).decode("utf-8")
    namespace = {"__name__": "legacy_dca"}
    exec(compile(source, "legacy_dca.py", "exec"), namespace)
    return namespace["DCAFPNLite"]


class WeightedDCATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_class = legacy_dca_class()

    def setUp(self):
        torch.manual_seed(1234)
        self.original = self.legacy_class(8)
        torch.manual_seed(1234)
        self.weighted = DCAFPNLite(8)
        self.deep = torch.randn(2, 8, 8, 8)
        self.shallow = torch.randn(2, 8, 8, 8)

    def test_uniform_start_matches_immutable_baseline(self):
        original_state = self.original.state_dict()
        for name, value in self.weighted.state_dict().items():
            if name == "weighted_fusion_version" or name.startswith("weight_conv.2."):
                continue
            self.assertTrue(torch.equal(value, original_state[name]), name)
        torch.testing.assert_close(
            self.weighted(self.deep, self.shallow), self.original(self.deep, self.shallow), rtol=0, atol=0)

    def test_weights_affect_output_and_train_both_predictor_layers(self):
        initial = self.weighted(self.deep, self.shallow).detach()
        optimizer = torch.optim.SGD(self.weighted.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            self.weighted(self.deep, self.shallow).square().mean().backward()
            for parameter in self.weighted.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(self.weighted.weight_conv[-1].weight.grad.abs().sum()), 0.0)
            optimizer.step()
        self.assertGreater(float(self.weighted.weight_conv[0].weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            self.weighted.weight_conv[-1].bias.copy_(torch.tensor([8.0, -8.0, -8.0, -8.0]))
        self.assertGreater(float((self.weighted(self.deep, self.shallow).detach() - initial).abs().max()), 0.0)

    def test_checkpoint_protocol_rejects_unweighted_baseline(self):
        saved = dict(EXPECTED, seed=1234)
        state = {f"swin_unet.dca_blocks.{index}.weighted_fusion_version": torch.tensor(1)
                 for index in (0, 1)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(dict(args=saved, epoch=1, model_state_dict=state), path)
            self.assertEqual(verify_checkpoint(path)[1], 1)
            del state["swin_unet.dca_blocks.1.weighted_fusion_version"]
            torch.save(dict(args=saved, epoch=1, model_state_dict=state), path)
            with self.assertRaisesRegex(ValueError, "weighted-DCA checkpoint"):
                verify_checkpoint(path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
