import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from bn_recovery import METHOD, load_checkpoint_model, split_mswd_stats
from calibrate_mswd_bn import ChannelMoments, calibrate_layerwise
from networks.WeavingUnet import MSWD
import prepare_manifest


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder4 = MSWD(4, 3, 3)

    def forward(self, x):
        return self.decoder4(x)


class RecoveryTest(unittest.TestCase):
    def test_moments_weight_unequal_batches(self):
        x = torch.arange(3 * 2 * 4 * 4).reshape(3, 2, 4, 4).float()
        moments = ChannelMoments()
        moments.update(x[:2])
        moments.update(x[2:])
        bn = nn.BatchNorm2d(2)
        moments.apply(bn)
        self.assertTrue(torch.allclose(bn.running_mean, x.mean((0, 2, 3))))
        self.assertTrue(torch.allclose(bn.running_var, x.var((0, 2, 3), correction=1)))

    def test_nonfinite_rejected(self):
        with self.assertRaises(FloatingPointError):
            ChannelMoments().update(torch.full((2, 3, 4, 4), float("nan")))

    def test_split_preserves_parameters_and_loads_old_checkpoint(self):
        model = TinyDecoder()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        original_parameters = sum(p.numel() for p in model.parameters())
        load_checkpoint_model(model, {"model_state_dict": state})
        targets = split_mswd_stats(model)
        self.assertEqual(targets, ["decoder4.norm", "decoder4.norm_second"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), original_parameters)
        self.assertIs(model.decoder4.norm.weight, model.decoder4.norm_second.weight)
        self.assertIsNot(model.decoder4.norm.running_mean, model.decoder4.norm_second.running_mean)
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        data = torch.randn(5, 3, 8, 8)
        loader = DataLoader(data, batch_size=2)
        calibrate_layerwise(model, loader, targets, torch.device("cpu"))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameters[name], parameter))
        self.assertFalse(model.training)
        self.assertFalse(model.decoder4.norm.training)
        self.assertFalse(model.decoder4.norm_second.training)
        self.assertFalse(torch.equal(model.decoder4.norm.running_mean, model.decoder4.norm_second.running_mean))
        with torch.inference_mode():
            together = model(data)
            alone = torch.cat([model(image[None]) for image in data])
        self.assertTrue(torch.allclose(together, alone, atol=1e-5))
        recovered = {"model_state_dict": model.state_dict(), "bn_recovery": {"method": METHOD}}
        loaded = TinyDecoder().eval()
        load_checkpoint_model(loaded, recovered)
        loaded.eval()
        with torch.inference_mode():
            self.assertTrue(torch.allclose(model(data), loaded(data)))

    def test_explicit_checkpoint_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            val = run / "unified_val_predictions"
            (val / "threshold_0.2" / "surface").mkdir(parents=True)
            checkpoint = run / "recovered.pth"
            checkpoint.touch()
            report = {"split": "val", "checkpoint": str(checkpoint), "tile_size": 512,
                      "overlap_stride": 256, "threshold_scores": {"0.2": {}}, "checkpoint_epoch": 80,
                      "pretrained_source": "ImageNet", "train_micro_batch": 2, "train_accumulation_steps": 2,
                      "bn_recovery": {"method": METHOD, "calibration_split": "train"}}
            (val / "inference_report.json").write_text(json.dumps(report), encoding="utf-8")
            with patch.object(sys, "argv", ["prepare_manifest.py", "--run_dir", str(run),
                                             "--data_root", str(run), "--checkpoint", str(checkpoint)]):
                prepare_manifest.main()
            manifest = json.loads((run / "unified_manifest.json").read_text())
            provenance = manifest["models"][0]["provenance"]
            self.assertEqual(provenance["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(provenance["bn_recovery"]["calibration_split"], "train")


if __name__ == "__main__":
    unittest.main()
