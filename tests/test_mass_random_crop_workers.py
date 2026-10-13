"""Catch repeating crops when persistent workers cannot see epoch updates."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from datasets.dataset_road_skeleton import RoadSkeletonDataset


class MassCropWorkerTests(unittest.TestCase):
    def test_persistent_spawn_worker_observes_new_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train').mkdir()
            (root / 'train_labels').mkdir()
            image = np.zeros((1500, 1500, 3), dtype=np.uint8)
            mask = np.zeros((1500, 1500), dtype=np.uint8)
            mask[749:752, :] = 255
            cv2.imwrite(str(root / 'train/case.tiff'), image)
            cv2.imwrite(str(root / 'train_labels/case.tif'), mask)
            dataset = RoadSkeletonDataset(
                str(root), image_size=512, source_patch_size=1500,
                random_crop_train=True, random_crops_per_image=1, augment=False,
            )
            self.assertEqual(len(dataset), 1)
            loader = DataLoader(dataset, batch_size=1, num_workers=2,
                                persistent_workers=True, multiprocessing_context='spawn')
            try:
                positions = []
                for epoch in (0, 1):
                    dataset.set_epoch(epoch)
                    batch = next(iter(loader))
                    self.assertEqual(tuple(batch['image'].shape), (1, 3, 512, 512))
                    positions.append((int(batch['tile_top'][0]), int(batch['tile_left'][0])))
                self.assertNotEqual(positions[0], positions[1])
                dataset.set_epoch(1)
                direct = dataset[0]
                self.assertEqual(positions[1], (direct['tile_top'], direct['tile_left']))
            finally:
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()


if __name__ == '__main__':
    unittest.main()
