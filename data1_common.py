"""Data1 input, Swin-Unet setup, and 512-window inference for the baseline."""

import argparse
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import get_config
from networks.vision_transformer import SwinUnet


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def case_id(path):
    return re.sub(r"(?:_(?:surface_pred|mask_pred|pred|sat|image|img|mask|label))+$",
                  "", Path(path).stem)


def _index_directory(directory, extensions):
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = case_id(path)
        if key in result:
            raise ValueError(f"Duplicate case {key}: {result[key]} and {path}")
        result[key] = path
    if not result:
        raise ValueError(f"No supported files under {directory}")
    return result


def _read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] not in (3, 4) or not all(
                np.array_equal(mask[..., 0], mask[..., channel]) for channel in (1, 2)):
            raise ValueError(f"Expected a replicated black/white GT mask: {path}")
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel binary GT mask: {path}")
    values = np.unique(mask)
    if not np.isin(values, (0, 1, 255)).all() or (1 in values and 255 in values):
        raise ValueError(f"Expected a 0/1 or 0/255 binary GT mask: {path}")
    return (mask > 0).astype(np.uint8)


def _normalise(image):
    image = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class Data1RoadDataset(Dataset):
    """Exactly one seeded 512 crop per 1024 source image in each train epoch."""

    def __init__(self, root, split, seed=1234, augment=False):
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.augment = bool(augment and split == "train")
        image_dir = self.root / split / "image"
        label_dir = next((self.root / split / name for name in ("mask", "label")
                          if (self.root / split / name).is_dir()), None)
        if not image_dir.is_dir() or label_dir is None:
            raise FileNotFoundError(f"Expected {root}/{split}/image and mask or label")
        images = _index_directory(image_dir, IMAGE_EXTENSIONS)
        labels = _index_directory(label_dir, IMAGE_EXTENSIONS - {".jpg", ".jpeg"})
        if set(images) != set(labels):
            missing = sorted(set(images) - set(labels))[:8]
            extra = sorted(set(labels) - set(images))[:8]
            raise ValueError(f"{split} image/GT mismatch: missing={missing}; extra={extra}")
        self.items = [(key, images[key], labels[key]) for key in sorted(images)]

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _crop_position(self, image_index, max_y, max_x, epoch):
        total = (max_y + 1) * (max_x + 1)
        if total == 1:
            return 0, 0
        rng = np.random.default_rng(self.seed + image_index * 9176)
        offset = int(rng.integers(0, total))
        step = int(rng.integers(1, total))
        while math.gcd(step, total) != 1:
            step = (step + 1) % total or 1
        # A seeded permutation of crop locations: no repeated location within
        # the first `total` epochs, including with multiple loader workers.
        return divmod((offset + epoch * step) % total, max_x + 1)

    def __getitem__(self, index):
        key, image_path, label_path = self.items[index]
        image = _read_rgb(image_path)
        mask = _read_mask(label_path)
        if image.shape[:2] != (1024, 1024) or mask.shape != (1024, 1024):
            raise ValueError(f"Data1 case {key} must have native 1024x1024 image and GT")
        crop_top = crop_left = 0
        if self.split == "train":
            crop_top, crop_left = self._crop_position(index, 512, 512, self.epoch)
            image = image[crop_top:crop_top + 512, crop_left:crop_left + 512]
            mask = mask[crop_top:crop_top + 512, crop_left:crop_left + 512]
            if self.augment:
                rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index * 9176 + 37)
                if rng.integers(2):
                    image, mask = np.flip(image, 0), np.flip(mask, 0)
                if rng.integers(2):
                    image, mask = np.flip(image, 1), np.flip(mask, 1)
                turns = int(rng.integers(4))
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        return {"image": _normalise(image),
                "mask": torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64))),
                "case_id": key, "crop_top": crop_top, "crop_left": crop_left}


def build_config(cfg_path, img_size=512):
    args = argparse.Namespace(cfg=str(cfg_path), opts=None, batch_size=None,
                              zip=False, cache_mode=None, resume=None,
                              accumulation_steps=None, use_checkpoint=False,
                              amp_opt_level=None, tag=None, eval=False, throughput=False)
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = int(img_size)
    # 512 / patch4 -> 128, 64, 32, 16 tokens; all divide evenly into 8x8 windows.
    config.MODEL.SWIN.WINDOW_SIZE = 8
    config.freeze()
    return config


def build_model(cfg_path):
    return SwinUnet(build_config(cfg_path), img_size=512, num_classes=2)


def load_swin_tiny_pretrain(model, path):
    """Load the original Swin-T encoder and mirrored decoder, resizing 7->8 biases."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Swin-T pretrained checkpoint missing: {path}")
    payload = torch.load(str(path), map_location="cpu")
    source = payload.get("model", payload.get("state_dict", payload))
    target = model.swin_unet.state_dict()
    loaded = {}
    encoder_count = decoder_count = interpolated = 0
    for raw_name, value in source.items():
        if not torch.is_tensor(value):
            continue
        name = raw_name.removeprefix("module.").removeprefix("swin_unet.")
        if name.startswith("head.") or "relative_position_index" in name or "attn_mask" in name:
            continue
        names = [name]
        if name.startswith("layers."):
            parts = name.split(".")
            if parts[1].isdigit():
                names.append("layers_up." + str(3 - int(parts[1])) + "." + ".".join(parts[2:]))
        for candidate in names:
            if candidate not in target:
                continue
            tensor = value
            expected = target[candidate]
            if tensor.shape != expected.shape and candidate.endswith("relative_position_bias_table"):
                old_side = math.isqrt(tensor.shape[0])
                new_side = math.isqrt(expected.shape[0])
                if (old_side ** 2 == tensor.shape[0] and
                        new_side ** 2 == expected.shape[0] and
                        tensor.shape[1] == expected.shape[1]):
                    tensor = F.interpolate(
                        tensor.float().T.reshape(1, tensor.shape[1], old_side, old_side),
                        (new_side, new_side), mode="bicubic", align_corners=False,
                    ).reshape(tensor.shape[1], -1).T
                    interpolated += 1
            if tensor.shape == expected.shape:
                loaded[candidate] = tensor.to(dtype=expected.dtype)
                if candidate.startswith("layers_up."):
                    decoder_count += 1
                else:
                    encoder_count += 1
    if encoder_count < 50:
        raise RuntimeError(f"Only {encoder_count} encoder tensors matched {path}; check Swin-T V1 weights")
    model.swin_unet.load_state_dict(loaded, strict=False)
    print(f"Pretrained tensors: encoder={encoder_count}, decoder={decoder_count}, "
          f"window_bias_interpolated={interpolated}", flush=True)
    return {"encoder_tensors": encoder_count, "decoder_tensors": decoder_count,
            "window_bias_interpolated": interpolated}


def _positions(length, tile, stride):
    positions = list(range(0, max(length - tile + 1, 1), stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def sliding_road_logits(model, image, device, tile_size=512, stride=256):
    """Taper-weighted logit stitching on the native 1024x1024 image."""
    if image.shape[-2:] != (1024, 1024) or tile_size != 512 or stride != 256:
        raise ValueError("Unified Data1 protocol requires native 1024, tile 512, stride 256")
    height, width = image.shape[-2:]
    canvas = torch.zeros((height, width), device=device, dtype=torch.float32)
    denominator = torch.zeros_like(canvas)
    ramp = (1 - torch.linspace(-1, 1, tile_size, device=device).abs()).clamp_min(0.05)
    weight = torch.outer(ramp, ramp)
    for top in _positions(height, tile_size, stride):
        for left in _positions(width, tile_size, stride):
            tile = image[:, top:top + tile_size, left:left + tile_size].unsqueeze(0).to(device)
            logits = model(tile)[0]
            road_logit = (logits[1] - logits[0]).float()
            canvas[top:top + tile_size, left:left + tile_size] += road_logit * weight
            denominator[top:top + tile_size, left:left + tile_size] += weight
    if bool((denominator <= 0).any()):
        raise RuntimeError("Sliding windows did not cover the full image")
    return canvas / denominator


def counts_from_logits(logits, target, threshold):
    prediction = torch.sigmoid(logits) >= float(threshold)
    truth = target.to(logits.device) > 0
    return (int((prediction & truth).sum()), int((prediction & ~truth).sum()),
            int((~prediction & truth).sum()))


def segmentation_from_counts(tp, fp, fn):
    return {"iou": tp / max(tp + fp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1)}
