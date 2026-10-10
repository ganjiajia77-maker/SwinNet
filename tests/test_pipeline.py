import unittest

import torch

from data1 import case_id, index_pairs
from inference import counts, metrics, positions, predict_full
from train_data1 import criterion


class ConstantModel(torch.nn.Module):
    def forward(self, image):
        return image[:, :1].sigmoid()


class PipelineTest(unittest.TestCase):
    def test_case_matching(self):
        self.assertEqual(case_id("123_sat.jpg"), case_id("123_mask.png"))

    def test_positions_cover_edge(self):
        self.assertEqual(positions(1024, 512, 256), [0, 256, 512])
        self.assertEqual(positions(1100, 512, 256), [0, 256, 512, 588])

    def test_weighted_stitch_preserves_constant(self):
        image = torch.full((1, 3, 16, 16), 0.4)
        result = predict_full(ConstantModel(), image, tile_size=8, stride=4)
        self.assertEqual(result.shape, (1, 1, 16, 16))
        self.assertTrue(torch.allclose(result, torch.sigmoid(image[:, :1]), atol=1e-6))

    def test_empty_mask_loss_finite(self):
        prediction = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)
        loss = criterion(prediction, torch.zeros_like(prediction))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(prediction.grad).all().item())

    def test_global_metrics(self):
        prediction = torch.tensor([[0.8, 0.8], [0.1, 0.1]])
        target = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        self.assertEqual(counts(prediction, target, 0.5), (1, 1, 1))
        self.assertAlmostEqual(metrics(1, 1, 1)["iou"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
