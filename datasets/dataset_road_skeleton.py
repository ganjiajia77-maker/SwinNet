import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RoadSkeletonDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=512, source_patch_size=1024, transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.source_patch_size = source_patch_size
        self.transform = transform

        self.image_dir = os.path.join(root_dir, split, "image")
        self.mask_dir = self._resolve_label_dir(root_dir, split, ("mask", "label"))

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(exts)
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def _resolve_label_dir(root_dir, split, candidates):
        for dirname in candidates:
            path = os.path.join(root_dir, split, dirname)
            if os.path.exists(path):
                return path
        expected = ", ".join(os.path.join(root_dir, split, name) for name in candidates)
        raise FileNotFoundError(f"Mask/label dir not found. Expected one of: {expected}")

    def _find_label_name(self, image_name, label_dir):
        base = os.path.splitext(image_name)[0]

        candidates = [
            image_name,
            base + ".png",
            base.replace("_sat", "_mask") + ".png",
            base.replace("_image", "_mask") + ".png",
            base.replace("_img", "_mask") + ".png",
        ]

        for name in candidates:
            if os.path.exists(os.path.join(label_dir, name)):
                return name

        raise FileNotFoundError(f"Cannot find label for image: {image_name} in {label_dir}")

    @staticmethod
    def _center_crop_or_pad(array, target_size):
        height, width = array.shape[:2]
        target_height, target_width = target_size

        if height > target_height:
            top = (height - target_height) // 2
            array = array[top:top + target_height, ...]
        elif height < target_height:
            pad_top = (target_height - height) // 2
            pad_bottom = target_height - height - pad_top
            pad_width = ((pad_top, pad_bottom), (0, 0)) if array.ndim == 2 else ((pad_top, pad_bottom), (0, 0), (0, 0))
            array = np.pad(array, pad_width, mode='constant', constant_values=0)

        if array.shape[1] > target_width:
            left = (array.shape[1] - target_width) // 2
            array = array[:, left:left + target_width, ...]
        elif array.shape[1] < target_width:
            pad_left = (target_width - array.shape[1]) // 2
            pad_right = target_width - array.shape[1] - pad_left
            pad_width = ((0, 0), (pad_left, pad_right)) if array.ndim == 2 else ((0, 0), (pad_left, pad_right), (0, 0))
            array = np.pad(array, pad_width, mode='constant', constant_values=0)

        return array

    @staticmethod
    def _skeletonize_binary(mask):
        binary = ((mask > 127).astype(np.uint8)) * 255
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while cv2.countNonZero(binary) > 0:
            eroded = cv2.erode(binary, element)
            opened = cv2.dilate(eroded, element)
            residue = cv2.subtract(binary, opened)
            skeleton = cv2.bitwise_or(skeleton, residue)
            binary = eroded

        return skeleton

    @staticmethod
    def _dilate_skeleton(skeleton, iterations=1):
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.dilate(skeleton, kernel, iterations=iterations)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label_name = self._find_label_name(image_name, self.mask_dir)

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, label_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        if self.source_patch_size is not None:
            source_size = (self.source_patch_size, self.source_patch_size)
            image = self._center_crop_or_pad(image, source_size)
            mask = self._center_crop_or_pad(mask, source_size)

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        mask = (mask > 127).astype(np.float32)
        skeleton_hard = self._skeletonize_binary((mask * 255).astype(np.uint8))
        skeleton_dilate = self._dilate_skeleton(skeleton_hard, iterations=1)
        skeleton_hard = (skeleton_hard > 127).astype(np.float32)
        skeleton_dilate = (skeleton_dilate > 127).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        skeleton_hard = np.expand_dims(skeleton_hard, axis=0)
        skeleton_dilate = np.expand_dims(skeleton_dilate, axis=0)

        sample = {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "skeleton": torch.from_numpy(skeleton_hard).float(),
            "skeleton_dilate": torch.from_numpy(skeleton_dilate).float(),
            "image_name": image_name,
            "case_name": os.path.splitext(image_name)[0].replace("_sat", ""),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
