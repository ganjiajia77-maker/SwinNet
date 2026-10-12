"""Mass crop schedule and encoder road-prior handoff regressions."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.dataset_road_skeleton import mass_crop_position
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class RecordingStage(nn.Module):
    def __init__(self, prior=None):
        super().__init__()
        self.last_road_attention = prior
        self.received = None

    def forward(self, x, road_prior=None):
        self.received = road_prior
        return x


class EncoderRoadPriorAblationTests(unittest.TestCase):
    def test_mass_crop_schedule_does_not_repeat_in_80_epochs(self):
        positions = [mass_crop_position(1234, 2, 0, epoch, 1500, 1500, 512)
                     for epoch in range(80)]
        self.assertEqual(len(set(positions)), 80)
        self.assertEqual(positions[0], mass_crop_position(1234, 2, 0, 0, 1500, 1500, 512))
        self.assertTrue(all(0 <= top <= 988 and 0 <= left <= 988
                            for top, left in positions))

    def test_stage3_gets_no_a1_a2_bias_when_disabled(self):
        for enabled in (False, True):
            model = SwinTransformerSys.__new__(SwinTransformerSys)
            nn.Module.__init__(model)
            model.ape = False
            model.patch_embed = nn.Identity()
            model.pos_drop = nn.Identity()
            model.bottleneck_context_fusion = nn.Identity()
            model.bottleneck_swin_block = nn.Identity()
            model.norm = nn.Identity()
            model.enable_encoder_road_prior = enabled
            priors = [torch.rand(1, 1, 8, 8), torch.rand(1, 1, 4, 4)]
            model.layers = nn.ModuleList([RecordingStage(priors[0]),
                                          RecordingStage(priors[1]),
                                          RecordingStage(), RecordingStage()])
            model.forward_features(torch.rand(1, 4, 8))
            self.assertIsNone(model.layers[0].received)
            self.assertIsNone(model.layers[1].received)
            if enabled:
                self.assertIs(model.layers[2].received[0], priors[0])
                self.assertIs(model.layers[2].received[1], priors[1])
            else:
                self.assertIsNone(model.layers[2].received)


if __name__ == '__main__':
    unittest.main()
