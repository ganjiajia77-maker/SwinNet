import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders import custom_transforms as tr


class Data1Random512(Dataset):
    """Data1 loader aligned with Swin-Unet random512 training."""

    NUM_CLASSES = 1

    def __init__(self, args, split='train'):
        self.args = args
        self.split = split
        self.root = args.data_root
        self.crop_size = int(args.crop_size)
        self.source_size = int(args.base_size)
        self.random_crop = split == 'train' and bool(args.random_crop_train)
        self.seed = int(args.seed)
        self.epoch = 0
        self.image_dir = os.path.join(self.root, split, 'image')
        self.mask_dir = self._find_mask_dir(split)
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(self.image_dir)
        self.images = sorted(name for name in os.listdir(self.image_dir)
                             if name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')))
        if not self.images:
            raise RuntimeError('No images found in ' + self.image_dir)
        for name in self.images:
            self._resolve_mask(name)
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def _find_mask_dir(self, split):
        for dirname in ('mask', 'label'):
            path = os.path.join(self.root, split, dirname)
            if os.path.isdir(path):
                return path
        sibling = os.path.join(self.root, split + '_labels')
        if os.path.isdir(sibling):
            return sibling
        raise FileNotFoundError('No mask/label directory for ' + split)

    def _resolve_mask(self, image_name):
        base, _ = os.path.splitext(image_name)
        candidates = [image_name, base + '.png', base + '.tif', base + '.tiff',
                      base.replace('_sat', '_mask') + '.png',
                      base.replace('_sat', '_mask') + '.tif',
                      base.replace('_image', '_mask') + '.png']
        for name in candidates:
            path = os.path.join(self.mask_dir, name)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError('Cannot find mask for ' + image_name)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _augment_geometry(image, mask, rng):
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
    def _augment_color_degrade(image, rng):
        image = image.astype(np.float32)
        image *= rng.uniform(0.9, 1.1)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * rng.uniform(0.9, 1.1) + mean
        gray = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]
        image = gray + (image - gray) * rng.uniform(0.9, 1.1)
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        if rng.rand() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.6)
        if rng.rand() < 0.15:
            noise = rng.normal(0.0, rng.uniform(3.0, 8.0), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return image

    @staticmethod
    def _connectivity(mask, distance):
        binary = (mask > 0).astype(np.float32)
        padded = np.pad(binary, distance, mode='constant')
        channels = []
        for dy in (-distance, 0, distance):
            for dx in (-distance, 0, distance):
                y0 = distance + dy
                x0 = distance + dx
                channels.append(padded[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] * binary)
        return np.stack(channels, axis=0).astype(np.float32)

    def _make_sample(self, index):
        image_name = self.images[index]
        image = np.asarray(Image.open(os.path.join(self.image_dir, image_name)).convert('RGB'))
        mask = np.asarray(Image.open(self._resolve_mask(image_name)).convert('L'))
        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + index * 9176)

        if self.split == 'train':
            image, mask = self._augment_geometry(image, mask, rng)
            image = self._augment_color_degrade(image, rng)

        height, width = mask.shape[:2]
        size = self.crop_size
        if self.random_crop:
            max_top = max(height - size, 0)
            max_left = max(width - size, 0)
            top = rng.randint(0, max_top + 1) if max_top else 0
            left = rng.randint(0, max_left + 1) if max_left else 0
        else:
            top = max((height - size) // 2, 0)
            left = max((width - size) // 2, 0)
        image = image[top:top + size, left:left + size]
        mask = mask[top:top + size, left:left + size]
        if image.shape[0] != size or image.shape[1] != size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        mask_binary = (mask >= 128).astype(np.float32)
        sample = {'image': Image.fromarray(image), 'label': Image.fromarray((mask_binary * 255).astype(np.uint8))}
        con1 = self._connectivity(mask_binary, 1)
        con3 = self._connectivity(mask_binary, 3)
        for group, array in (('connect', con1), ('connect_d1', con3)):
            for channel_index, channel in enumerate(np.array_split(array, 3, axis=0)):
                key = group + ('_' if group == 'connect_d1' else '') + str(channel_index)
                sample[key] = Image.fromarray((channel.transpose(1, 2, 0) * 255).astype(np.uint8))
        return sample, image_name

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        sample, image_name = self._make_sample(index)
        sample = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
        ])(sample)
        if self.split == 'train':
            return sample
        return sample, image_name