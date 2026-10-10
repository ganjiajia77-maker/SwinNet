"""Data1 images and one random native-resolution crop per training image."""

from pathlib import Path
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def case_id(path):
    return re.sub(r"(?:_(?:sat|image|img|mask|label))+$", "", Path(path).stem)


def index_pairs(root, split):
    base = Path(root) / split
    image_dir = base / "image"
    label_dir = base / "label"
    if not label_dir.is_dir():
        label_dir = base / "mask"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected image and label/mask under {base}")
    images = {case_id(p): p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
    labels = {case_id(p): p for p in label_dir.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff"}}
    if not images or images.keys() != labels.keys():
        raise ValueError(f"Image/label case mismatch in {base}: {len(images)} vs {len(labels)}")
    return [(case, images[case], labels[case]) for case in sorted(images)]


def read_pair(image_path, label_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if image is None or label is None or image.shape[:2] != label.shape:
        raise ValueError(f"Unreadable or mismatched image/label: {image_path}, {label_path}")
    return image, label > 0


def normalize(image):
    # Preserve the BGR input scaling used by the published WeavingUnet loader.
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255).mul_(3.2).sub_(1.6)


class RandomCropData1(Dataset):
    def __init__(self, root, crop_size=512):
        self.pairs = index_pairs(root, "train")
        self.crop_size = crop_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        _, image_path, label_path = self.pairs[index]
        image, label = read_pair(image_path, label_path)
        height, width = label.shape
        size = self.crop_size
        if height < size or width < size:
            raise ValueError(f"Image smaller than crop {size}: {image_path}")
        top, left = random.randint(0, height - size), random.randint(0, width - size)
        image = image[top:top + size, left:left + size]
        label = label[top:top + size, left:left + size]
        if random.random() < 0.5:
            image, label = image[::-1], label[::-1]
        if random.random() < 0.5:
            image, label = image[:, ::-1], label[:, ::-1]
        k = random.randrange(4)
        image, label = np.rot90(image, k), np.rot90(label, k)
        return normalize(image), torch.from_numpy(np.ascontiguousarray(label[None])).float()


class FullImageData1(Dataset):
    def __init__(self, root, split):
        self.pairs = index_pairs(root, split)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        case, image_path, label_path = self.pairs[index]
        image, label = read_pair(image_path, label_path)
        return case, normalize(image), torch.from_numpy(label.copy())
