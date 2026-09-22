import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Data1RoadDataset(Dataset):
    """Swin-Unet data1 adapter for the raster road branch of SAM-Road.

    Training uses one deterministic 512 crop per 1024 source image per epoch,
    matching Swin-Unet's random_crop_train pipeline. Validation/test return the
    full 1024 source patch for identical 512/256-overlap evaluation.
    """

    def __init__(self, root, split, crop_size=512, source_size=1024,
                 random_crop=False, augment=False, seed=1234):
        self.root = root
        self.split = split
        self.crop_size = int(crop_size)
        self.source_size = int(source_size)
        self.random_crop = bool(random_crop and split == "train")
        self.augment = bool(augment and split == "train")
        self.seed = int(seed)
        self.epoch = 0

        self.image_dir = os.path.join(root, split, "image")
        self.label_dir = os.path.join(root, split, "label")
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.label_dir}"
            )
        self.image_files = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        )
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _center_crop_or_pad(array, size):
        h, w = array.shape[:2]
        if h > size:
            top = (h - size) // 2
            array = array[top:top + size]
        elif h < size:
            pad0 = (size - h) // 2
            pad1 = size - h - pad0
            pad = ((pad0, pad1), (0, 0))
            if array.ndim == 3:
                pad += ((0, 0),)
            array = np.pad(array, pad, mode="constant")
        h, w = array.shape[:2]
        if w > size:
            left = (w - size) // 2
            array = array[:, left:left + size]
        elif w < size:
            pad0 = (size - w) // 2
            pad1 = size - w - pad0
            pad = ((0, 0), (pad0, pad1))
            if array.ndim == 3:
                pad += ((0, 0),)
            array = np.pad(array, pad, mode="constant")
        return array

    @staticmethod
    def _geometry(image, mask, rng):
        if rng.rand() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if rng.rand() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        rotations = rng.randint(0, 4)
        if rotations:
            image = np.rot90(image, rotations, axes=(0, 1))
            mask = np.rot90(mask, rotations, axes=(0, 1))
        return image.copy(), mask.copy()

    @staticmethod
    def _color_degrade(image, rng):
        image = image.astype(np.float32)
        image *= rng.uniform(0.9, 1.1)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * rng.uniform(0.9, 1.1) + mean
        gray = (0.299 * image[..., 0:1] + 0.587 * image[..., 1:2]
                + 0.114 * image[..., 2:3])
        image = gray + (image - gray) * rng.uniform(0.9, 1.1)
        if rng.rand() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.6)
        if rng.rand() < 0.15:
            image = image + rng.normal(0.0, rng.uniform(3.0, 8.0), image.shape)
        return np.clip(image, 0.0, 255.0).astype(np.float32)

    def __getitem__(self, index):
        name = self.image_files[index]
        stem = os.path.splitext(name)[0]
        if stem.endswith("_sat"): 
            label_name = stem.replace("_sat", "_mask") + ".png"
        else:
            label_name = stem + ".png"
        image = cv2.imread(os.path.join(self.image_dir, name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.label_dir, label_name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image/label for {name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._center_crop_or_pad(image, self.source_size)
        mask = self._center_crop_or_pad(mask, self.source_size)

        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + index * 9176)
        if self.augment:
            image, mask = self._geometry(image, mask, rng)
            image = self._color_degrade(image, rng)

        if self.random_crop:
            max_top = self.source_size - self.crop_size
            max_left = self.source_size - self.crop_size
            top = rng.randint(0, max_top + 1) if max_top > 0 else 0
            left = rng.randint(0, max_left + 1) if max_left > 0 else 0
            image = image[top:top + self.crop_size, left:left + self.crop_size]
            mask = mask[top:top + self.crop_size, left:left + self.crop_size]

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return {"image": image, "mask": mask, "name": name}
