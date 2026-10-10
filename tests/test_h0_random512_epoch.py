"""Regression: persistent spawn workers must observe each new crop epoch."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.dataset_road_skeleton import RoadSkeletonDataset


class SharedEpochTests(unittest.TestCase):
    def test_persistent_spawn_workers_change_crops_and_repeat_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks = root / 'train/image', root / 'train/mask'
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            for index in range(4):
                cv2.imwrite(str(images / f'{index}_sat.png'), np.zeros((1024, 1024, 3), np.uint8))
                cv2.imwrite(str(masks / f'{index}_mask.png'), np.zeros((1024, 1024), np.uint8))
            dataset = RoadSkeletonDataset(str(root), image_size=512, tile_size=512,
                                          augment=False, random_crop_train=True,
                                          random_crops_per_image=1, random_crop_seed=1234)
            self.assertEqual(len(dataset), 4)
            loader = DataLoader(dataset, batch_size=1, num_workers=2,
                                persistent_workers=True, multiprocessing_context='spawn')
            try:
                results = []
                for epoch in (0, 1, 0):
                    dataset.set_epoch(epoch)
                    positions = []
                    for index, sample in enumerate(loader):
                        self.assertEqual(tuple(sample['image'].shape), (1, 3, 512, 512))
                        position = (int(sample['tile_top']), int(sample['tile_left']))
                        rng = np.random.RandomState(1234 + epoch * 1000003 + index * 9176)
                        self.assertEqual(position, (rng.randint(513), rng.randint(513)))
                        positions.append(position)
                    results.append(positions)
                self.assertNotEqual(results[0], results[1])
                self.assertEqual(results[0], results[2])
            finally:
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()


if __name__ == '__main__':
    unittest.main()
