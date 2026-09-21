import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders import custom_transforms as tr


class Data1Random512(Dataset):
    """Data1 loader matching the Swin-Unet 1024 -> random 512 protocol.

    Connectivity labels are generated after cropping from the cropped GT mask,
    so image, segmentation target, and all 9 CoANet neighbor channels remain
    in the same coordinate system.
    """

    NUM_CLASSES = 1

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.root = args.data_root
        self.crop_size = int(args.crop_size)
        self.source_size = int(args.base_size)
        self.random_crop = split == "train" and bool(args.random_crop_train)
        self.seed = int(args.seed)
        self.image_dir = os.path.join(self.root, split, "image")
        self.mask_dir = self._find_mask_dir(split)
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(self.image_dir)
        self.images = sorted(
            name for name in os.listdir(self.image_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        )
        if not self.images:
            raise RuntimeError("No images found in " + self.image_dir)
        for name in self.images:
            self._resolve_mask(name)
        print("Number of images in {}: {:d}".format(split, len(self.images)))

    def _find_mask_dir(self, split):
        for dirname in ("mask", "label"):
            path = os.path.join(self.root, split, dirname)
            if os.path.isdir(path):
                return path
        sibling = os.path.join(self.root, split + "_labels")
        if os.path.isdir(sibling):
            return sibling
        raise FileNotFoundError("No mask/label directory for " + split)

    @staticmethod
    def _mask_name(image_name):
        base, _ = os.path.splitext(image_name)
        candidates = [
            image_name,
            base + ".png",
            base + ".tif",
            base.replace("_sat", "_mask") + ".png",
            base.replace("_image", "_mask") + ".png",
        ]
        return candidates[0]

    def _resolve_mask(self, image_name):
        base, _ = os.path.splitext(image_name)
        candidates = [
            image_name,
            base + ".png",
            base + ".tif",
            base + ".tiff",
            base.replace("_sat", "_mask") + ".png",
            base.replace("_sat", "_mask") + ".tif",
            base.replace("_image", "_mask") + ".png",
        ]
        for name in candidates:
            path = os.path.join(self.mask_dir, name)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError("Cannot find mask for " + image_name)

    def __len__(self):
        return len(self.images)

    def _crop_box(self, index, width, height):
        size = self.crop_size
        if width < size or height < size:
            return 0, 0, min(width, size), min(height, size)
        if self.random_crop:
            rng = random.Random(self.seed + index + getattr(self, "epoch", 0) * 1000003)
            left = rng.randint(0, width - size)
            top = rng.randint(0, height - size)
        else:
            left = max(0, (width - size) // 2)
            top = max(0, (height - size) // 2)
        return left, top, left + size, top + size

    @staticmethod
    def _connectivity(mask, distance):
        binary = (mask > 0).astype(np.float32)
        padded = np.pad(binary, distance, mode="constant")
        channels = []
        for dy in (-distance, 0, distance):
            for dx in (-distance, 0, distance):
                y0 = distance + dy
                x0 = distance + dx
                channels.append(padded[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] * binary)
        return np.stack(channels, axis=0).astype(np.float32)

    def _make_sample(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        mask = Image.open(self._resolve_mask(image_name)).convert("L")
        left, top, right, bottom = self._crop_box(index, image.width, image.height)
        image = image.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))
        if image.size != (self.crop_size, self.crop_size):
            image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
        mask_np = (np.asarray(mask) >= 128).astype(np.float32)
        sample = {
            "image": image,
            "label": mask,
        }
        con1 = self._connectivity(mask_np, 1)
        con3 = self._connectivity(mask_np, 3)
        for group, array in (("connect", con1), ("connect_d1", con3)):
            for index, channel in enumerate(np.array_split(array, 3, axis=0)):
                sample[group + "_" + str(index)] = Image.fromarray((channel.transpose(1, 2, 0) * 255).astype(np.uint8))
        return sample, image_name

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        sample, image_name = self._make_sample(index)
        if self.split == "train":
            sample = transforms.Compose([
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ])(sample)
            return sample
        sample = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
        ])(sample)
        return sample, image_name
