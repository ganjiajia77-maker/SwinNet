import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from losses.road_losses import build_boundary_target, build_connectivity_target
from direction_target_utils import build_continuous_direction_target


class RoadSkeletonDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=512,
        source_patch_size=1024,
        transform=None,
        tile_size=None,
        tile_stride=None,
        augment=False,
        return_full_image=False,
        random_crop_train=False,
        random_crops_per_image=1,
        random_crop_seed=1234,
        crop_list_path="",
        return_dense_targets=False,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.source_patch_size = source_patch_size
        self.transform = transform
        self.tile_size = tile_size
        self.tile_stride = tile_stride if tile_stride is not None else tile_size
        self.augment = bool(augment and split == "train")
        self.return_full_image = return_full_image
        self.random_crop_train = bool(random_crop_train and split == "train")
        self.random_crops_per_image = max(1, int(random_crops_per_image))
        self.random_crop_seed = int(random_crop_seed)
        self.return_dense_targets = bool(return_dense_targets)
        self.epoch = 0

        self.image_dir = self._resolve_image_dir(root_dir, split)
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

        self.fixed_crop_items = self._load_crop_list(crop_list_path)
        self.tile_positions = None if self.fixed_crop_items else self._build_tile_positions()

    def __len__(self):
        fixed_crop_items = getattr(self, "fixed_crop_items", [])
        if fixed_crop_items:
            return len(fixed_crop_items)
        if self.random_crop_train:
            return len(self.image_files) * self.random_crops_per_image
        if self.tile_positions is None:
            return len(self.image_files)
        return len(self.image_files) * len(self.tile_positions)

    @staticmethod
    def sliding_positions(length, tile_size, stride):
        if length <= tile_size:
            return [0]
        positions = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _load_crop_list(self, crop_list_path):
        if not crop_list_path:
            return []
        if not os.path.exists(crop_list_path):
            raise FileNotFoundError(f"Crop list not found: {crop_list_path}")

        image_to_index = {
            os.path.splitext(name)[0]: index for index, name in enumerate(self.image_files)
        }
        image_to_index.update({name: index for index, name in enumerate(self.image_files)})
        items = []
        with open(crop_list_path, "r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    raise ValueError(
                        f"Invalid crop list line {line_number}: expected 'image: x=..., y=...'"
                    )
                image_key, rest = line.split(":", 1)
                fields = {}
                for part in rest.replace(",", " ").split():
                    if "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    fields[key.strip().lower()] = int(value)
                if "x" not in fields or "y" not in fields:
                    raise ValueError(
                        f"Invalid crop list line {line_number}: missing x/y coordinates"
                    )
                image_key = image_key.strip()
                lookup_keys = [image_key, os.path.splitext(image_key)[0]]
                image_index = next(
                    (image_to_index[key] for key in lookup_keys if key in image_to_index),
                    None,
                )
                if image_index is None:
                    raise ValueError(
                        f"Crop list line {line_number} references unknown image: {image_key}"
                    )
                items.append((image_index, int(fields["y"]), int(fields["x"])))
        if not items:
            raise ValueError(f"Crop list is empty: {crop_list_path}")
        return items

    def _build_tile_positions(self):
        if self.tile_size is None or self.return_full_image:
            return None
        source_size = self.source_patch_size or self.tile_size
        rows = self.sliding_positions(source_size, self.tile_size, self.tile_stride)
        cols = self.sliding_positions(source_size, self.tile_size, self.tile_stride)
        return [(top, left) for top in rows for left in cols]

    @staticmethod
    def _resolve_image_dir(root_dir, split):
        nested = os.path.join(root_dir, split, "image")
        if os.path.exists(nested):
            return nested
        flat = os.path.join(root_dir, split)
        if os.path.exists(flat):
            return flat
        raise FileNotFoundError(
            f"Image dir not found. Expected one of: {nested}, {flat}"
        )

    @staticmethod
    def _resolve_label_dir(root_dir, split, candidates):
        for dirname in candidates:
            path = os.path.join(root_dir, split, dirname)
            if os.path.exists(path):
                return path
        sibling = os.path.join(root_dir, f"{split}_labels")
        if os.path.exists(sibling):
            return sibling
        expected_dirs = [os.path.join(root_dir, split, name) for name in candidates]
        expected_dirs.append(sibling)
        expected = ", ".join(expected_dirs)
        raise FileNotFoundError(f"Mask/label dir not found. Expected one of: {expected}")

    def _find_label_name(self, image_name, label_dir):
        base = os.path.splitext(image_name)[0]

        candidates = [
            image_name,
            base + ".png",
            base + ".tif",
            base + ".tiff",
            base.replace("_sat", "_mask") + ".png",
            base.replace("_sat", "_mask") + ".tif",
            base.replace("_sat", "_mask") + ".tiff",
            base.replace("_image", "_mask") + ".png",
            base.replace("_image", "_mask") + ".tif",
            base.replace("_image", "_mask") + ".tiff",
            base.replace("_img", "_mask") + ".png",
            base.replace("_img", "_mask") + ".tif",
            base.replace("_img", "_mask") + ".tiff",
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
    def _zhang_suen_fallback(mask):
        image = (mask > 0).astype(np.uint8)

        while True:
            changed = False
            for step in (0, 1):
                padded = np.pad(image, 1, mode="constant")
                p2 = padded[:-2, 1:-1]
                p3 = padded[:-2, 2:]
                p4 = padded[1:-1, 2:]
                p5 = padded[2:, 2:]
                p6 = padded[2:, 1:-1]
                p7 = padded[2:, :-2]
                p8 = padded[1:-1, :-2]
                p9 = padded[:-2, :-2]

                neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                transitions = np.stack(
                    [
                        (p2 == 0) & (p3 == 1),
                        (p3 == 0) & (p4 == 1),
                        (p4 == 0) & (p5 == 1),
                        (p5 == 0) & (p6 == 1),
                        (p6 == 0) & (p7 == 1),
                        (p7 == 0) & (p8 == 1),
                        (p8 == 0) & (p9 == 1),
                        (p9 == 0) & (p2 == 1),
                    ],
                    axis=0,
                ).sum(axis=0)

                common = (
                    (image == 1)
                    & (neighbors >= 2)
                    & (neighbors <= 6)
                    & (transitions == 1)
                )
                if step == 0:
                    removable = common & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
                else:
                    removable = common & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

                if removable.any():
                    image[removable] = 0
                    changed = True

            if not changed:
                break

        return image * 255

    @classmethod
    def _skeletonize_binary(cls, mask):
        binary = ((mask > 0).astype(np.uint8)) * 255
        if (
            hasattr(cv2, "ximgproc")
            and hasattr(cv2.ximgproc, "thinning")
            and hasattr(cv2.ximgproc, "THINNING_ZHANGSUEN")
        ):
            return cv2.ximgproc.thinning(
                binary,
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
            )
        return cls._zhang_suen_fallback(binary)

    @staticmethod
    def _dilate_skeleton(skeleton, iterations=1):
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.dilate(skeleton, kernel, iterations=iterations)

    @staticmethod
    def _augment_geometry(image, mask, rng=None):
        rng = rng if rng is not None else np.random
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
    def _augment_color_degrade(image, rng=None):
        rng = rng if rng is not None else np.random
        image = image.astype(np.float32)
        image *= rng.uniform(0.9, 1.1)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * rng.uniform(0.9, 1.1) + mean
        gray = (
            0.299 * image[..., 0:1]
            + 0.587 * image[..., 1:2]
            + 0.114 * image[..., 2:3]
        )
        image = gray + (image - gray) * rng.uniform(0.9, 1.1)
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        if rng.rand() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.6)
        if rng.rand() < 0.15:
            noise = rng.normal(0.0, rng.uniform(3.0, 8.0), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return image

    @staticmethod
    def _build_direction_target(skeleton):
        return build_continuous_direction_target(skeleton, radius=3)

    def __getitem__(self, idx):
        rng = None
        fixed_crop_items = getattr(self, "fixed_crop_items", [])
        if fixed_crop_items:
            image_index, top, left = fixed_crop_items[idx]
            tile_position = (top, left)
        elif self.random_crop_train:
            image_index = idx // self.random_crops_per_image
            crop_ordinal = idx % self.random_crops_per_image
            rng_seed = (
                self.random_crop_seed
                + self.epoch * 1000003
                + image_index * 9176
                + crop_ordinal * 101
            )
            rng = np.random.RandomState(rng_seed)
            tile_position = None
        elif self.tile_positions is None:
            image_index = idx
            tile_position = None
        else:
            image_index = idx // len(self.tile_positions)
            tile_position = self.tile_positions[idx % len(self.tile_positions)]
        image_name = self.image_files[image_index]
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

        if self.augment:
            image, mask = self._augment_geometry(image, mask, rng=rng)
            image = self._augment_color_degrade(image, rng=rng)

        mask = (mask > 127).astype(np.float32)
        skeleton_hard = (self._skeletonize_binary(mask) > 127).astype(np.float32)
        skeleton_dilate = (self._dilate_skeleton(skeleton_hard * 255, iterations=1) > 127).astype(np.float32)
        full_skeleton = torch.from_numpy(skeleton_hard).float().unsqueeze(0).unsqueeze(0)
        valid_mask = torch.ones_like(full_skeleton).squeeze(0)
        connectivity_gt = direction_gt = boundary_gt = None
        if self.return_dense_targets:
            full_surface = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            connectivity_gt = build_connectivity_target(full_skeleton).squeeze(0)
            direction_gt = self._build_direction_target(full_skeleton).squeeze(0)
            boundary_gt = build_boundary_target(full_surface).squeeze(0)

        if self.random_crop_train:
            crop_size = self.tile_size or self.image_size
            if crop_size is None:
                raise ValueError("random_crop_train requires tile_size or image_size.")
            height, width = image.shape[:2]
            max_top = max(height - crop_size, 0)
            max_left = max(width - crop_size, 0)
            top = rng.randint(0, max_top + 1) if max_top > 0 else 0
            left = rng.randint(0, max_left + 1) if max_left > 0 else 0
            tile_position = (top, left)

        if tile_position is not None:
            top, left = tile_position
            crop_size = self.tile_size or self.image_size
            bottom = top + crop_size
            right = left + crop_size
            image = image[top:bottom, left:right]
            mask = mask[top:bottom, left:right]
            skeleton_hard = skeleton_hard[top:bottom, left:right]
            skeleton_dilate = skeleton_dilate[top:bottom, left:right]
            if connectivity_gt is not None:
                connectivity_gt = connectivity_gt[:, top:bottom, left:right]
            if direction_gt is not None:
                direction_gt = direction_gt[:, top:bottom, left:right]
            if boundary_gt is not None:
                boundary_gt = boundary_gt[:, top:bottom, left:right]
            valid_mask = valid_mask[:, top:bottom, left:right]

        if self.image_size is not None and image.shape[:2] != (self.image_size, self.image_size):
            size = (self.image_size, self.image_size)
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        skeleton_hard = np.expand_dims(skeleton_hard, axis=0)
        skeleton_dilate = np.expand_dims(skeleton_dilate, axis=0)

        sample = {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "skeleton": torch.from_numpy(skeleton_hard).float(),
            "skeleton_dilate": torch.from_numpy(skeleton_dilate).float(),
            "valid_mask": valid_mask.float(),
            "image_name": image_name,
            "case_name": os.path.splitext(image_name)[0].replace("_sat", ""),
        }
        if connectivity_gt is not None:
            sample["connectivity_gt"] = connectivity_gt.float()
        if direction_gt is not None:
            sample["direction_gt"] = direction_gt.float()
        if boundary_gt is not None:
            sample["boundary_gt"] = boundary_gt.float()

        if tile_position is not None:
            sample["tile_top"] = tile_position[0]
            sample["tile_left"] = tile_position[1]

        if self.transform:
            sample = self.transform(sample)

        return sample
