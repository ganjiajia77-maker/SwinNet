"""Shared data1 preprocessing and full-image inference for OARENet."""

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from networks.testNet import SwinT_OAM


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def image_names(root, split):
    directory = os.path.join(root, split, 'image')
    names = sorted(name for name in os.listdir(directory)
                   if name.lower().endswith(IMAGE_EXTENSIONS))
    if not names:
        raise RuntimeError('No images found: ' + directory)
    return names


def mask_path(root, split, image_name):
    directory = os.path.join(root, split, 'label')
    if not os.path.isdir(directory):
        directory = os.path.join(root, split, 'mask')
    stem = os.path.splitext(image_name)[0]
    candidates = (image_name, stem + '.png', stem.replace('_sat', '_mask') + '.png',
                  stem.replace('_image', '_mask') + '.png')
    for candidate in candidates:
        path = os.path.join(directory, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError('No label for ' + image_name + ' in ' + directory)


def center_crop_or_pad(array, size):
    height, width = array.shape[:2]
    top, left = max((height - size) // 2, 0), max((width - size) // 2, 0)
    array = array[top:top + min(size, height), left:left + min(size, width)]
    height, width = array.shape[:2]
    pad = ((size - height) // 2, size - height - (size - height) // 2)
    pad_w = ((size - width) // 2, size - width - (size - width) // 2)
    widths = (pad, pad_w) + (((0, 0),) if array.ndim == 3 else ())
    return np.pad(array, widths, mode='constant')


def load_case(root, split, name, source_size):
    with Image.open(os.path.join(root, split, 'image', name)) as handle:
        image = np.asarray(handle.convert('RGB'))
    with Image.open(mask_path(root, split, name)) as handle:
        mask = np.asarray(handle.convert('L'))
    return center_crop_or_pad(image, source_size), center_crop_or_pad(mask, source_size) >= 128


def image_tensor(image):
    array = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Data1Train(Dataset):
    def __init__(self, root, seed=1234, source_size=1024, crop_size=512):
        if crop_size > source_size:
            raise ValueError('crop_size must be <= source_size')
        self.root, self.seed = root, int(seed)
        self.source_size, self.crop_size = source_size, crop_size
        self.names = image_names(root, 'train')
        for name in self.names:
            mask_path(root, 'train', name)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image, mask = load_case(self.root, 'train', self.names[index], self.source_size)
        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + index * 9176)
        if rng.rand() < 0.5:
            image, mask = np.flip(image, 1), np.flip(mask, 1)
        if rng.rand() < 0.5:
            image, mask = np.flip(image, 0), np.flip(mask, 0)
        turns = rng.randint(0, 4)
        if turns:
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        image = image.copy().astype(np.float32)
        mask = mask.copy()
        image *= rng.uniform(0.9, 1.1)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * rng.uniform(0.9, 1.1) + mean
        gray = 0.299 * image[..., :1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]
        image = gray + (image - gray) * rng.uniform(0.9, 1.1)
        image = np.clip(image, 0, 255).astype(np.uint8)
        if rng.rand() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.6)
        if rng.rand() < 0.15:
            noise = rng.normal(0, rng.uniform(3, 8), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        max_offset = self.source_size - self.crop_size
        top = rng.randint(0, max_offset + 1) if max_offset else 0
        left = rng.randint(0, max_offset + 1) if max_offset else 0
        size = self.crop_size
        image = image[top:top + size, left:left + size]
        mask = mask[top:top + size, left:left + size]
        return {'image': image_tensor(image),
                'mask': torch.from_numpy(mask.copy()).unsqueeze(0).float(),
                'crop_epoch': self.epoch, 'crop_index': index,
                'crop_top': top, 'crop_left': left}


def positions(length, tile, stride):
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.no_grad()
def predict_full(model, image, device, tile=512, stride=256, tta=True):
    height, width = image.shape[:2]
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    flips = ((), (2,), (3,), (2, 3)) if tta else ((),)
    for top in positions(height, tile, stride):
        for left in positions(width, tile, stride):
            crop = image[top:top + tile, left:left + tile]
            if crop.shape[:2] != (tile, tile):
                crop = center_crop_or_pad(crop, tile)
            tensor = image_tensor(crop).unsqueeze(0).to(device)
            views = []
            for dims in flips:
                result = model(torch.flip(tensor, dims) if dims else tensor)
                views.append(torch.flip(result, dims) if dims else result)
            probability = torch.stack(views).mean(0)[0, 0].float().cpu().numpy()
            h, w = min(tile, height - top), min(tile, width - left)
            total[top:top + h, left:left + w] += probability[:h, :w]
            count[top:top + h, left:left + w] += 1
    return total / np.maximum(count, 1)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in checkpoint:
        raise KeyError('Not a data1 training checkpoint: ' + checkpoint_path)
    model = SwinT_OAM(pretrained_backbone=None)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    print('Loaded checkpoint epoch={} weights={}'.format(
        checkpoint.get('epoch'), checkpoint.get('eval_weights', 'raw')), flush=True)
    return model


def pixel_counts(prediction, target):
    return (int((prediction & target).sum()), int((prediction & ~target).sum()),
            int((~prediction & target).sum()))


def metrics(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {'iou': tp / max(tp + fp + fn, 1),
            'f1': 2 * precision * recall / max(precision + recall, 1e-12),
            'precision': precision, 'recall': recall}
