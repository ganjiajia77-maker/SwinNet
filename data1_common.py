"""Data1 1024-image protocol and the original DARENet model loader."""

import importlib.machinery
import importlib.util
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def case_id(path):
    return re.sub(r"(?:_(?:surface_pred|mask_pred|pred|sat|image|img|mask|label))+$",
                  "", Path(path).stem)


def _index(directory, extensions):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            key = case_id(path)
            if key in result:
                raise ValueError(f"Duplicate Data1 case {key}: {result[key]} and {path}")
            result[key] = path
    if not result:
        raise ValueError(f"No supported files in {directory}")
    return result


class Data1RoadDataset(Dataset):
    """One distinct 512 crop per native 1024 training image and epoch."""

    def __init__(self, root, split, seed=1234, augment=False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.split, self.seed, self.epoch = split, int(seed), 0
        self.augment = bool(augment and split == "train")
        base = Path(root) / split
        image_dir = base / "image"
        label_dir = next((base / name for name in ("mask", "label")
                          if (base / name).is_dir()), None)
        if not image_dir.is_dir() or label_dir is None:
            raise FileNotFoundError(f"Expected {base}/image and mask or label")
        images = _index(image_dir, EXTENSIONS)
        labels = _index(label_dir, EXTENSIONS - {".jpg", ".jpeg"})
        if set(images) != set(labels):
            raise ValueError(f"{split} image/GT mismatch: "
                             f"missing={sorted(set(images) - set(labels))[:8]}, "
                             f"extra={sorted(set(labels) - set(images))[:8]}")
        self.items = [(key, images[key], labels[key]) for key in sorted(images)]

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _crop_position(self, index):
        width = 1024 - 512 + 1
        total = width * width
        rng = np.random.default_rng(self.seed + index * 9176)
        offset = int(rng.integers(total))
        step = int(rng.integers(1, total))
        while math.gcd(step, total) != 1:
            step = (step + 1) % total or 1
        return divmod((offset + self.epoch * step) % total, width)

    def __getitem__(self, index):
        key, image_path, mask_path = self.items[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # DARENet uses BGR.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f"Cannot read image/GT for {key}")
        if mask.ndim == 3:
            if mask.shape[2] not in (3, 4) or not all(
                    np.array_equal(mask[..., 0], mask[..., channel]) for channel in (1, 2)):
                raise ValueError(f"Expected replicated binary mask: {mask_path}")
            mask = mask[..., 0]
        if image.shape != (1024, 1024, 3) or mask.shape != (1024, 1024):
            raise ValueError(f"Data1 case {key} must be native 1024x1024")
        values = np.unique(mask)
        if not np.isin(values, (0, 1, 255)).all() or (1 in values and 255 in values):
            raise ValueError(f"Expected binary 0/1 or 0/255 GT: {mask_path}")
        mask = (mask > 0).astype(np.uint8)
        if self.split == "train":
            top, left = self._crop_position(index)
            image = image[top:top + 512, left:left + 512]
            mask = mask[top:top + 512, left:left + 512]
            if self.augment:
                rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index * 9176 + 37)
                if rng.integers(2):
                    image, mask = np.flip(image, 0), np.flip(mask, 0)
                if rng.integers(2):
                    image, mask = np.flip(image, 1), np.flip(mask, 1)
                turns = int(rng.integers(4))
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        # The upstream loader uses BGR and x/255*3.2-1.6.
        image = image.astype(np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        return {"image": torch.from_numpy(np.ascontiguousarray(image)),
                "mask": torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32))),
                "case_id": key}


def build_model(original_repo):
    """Load upstream networks.lmz from source, or its tracked CPython 3.10 bytecode."""
    repo = Path(original_repo).resolve()
    source = repo / "networks" / "lmz.py"
    bytecode = repo / "networks" / "__pycache__" / "lmz.cpython-310.pyc"
    if not source.is_file() and not bytecode.is_file():
        raise FileNotFoundError(f"DARENet model missing: {source} or {bytecode}")
    if not source.is_file() and sys.version_info[:2] != (3, 10):
        raise RuntimeError("Upstream only provides CPython 3.10 DARENet bytecode; use Python 3.10")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import networks  # noqa: F401 - establish the upstream package/namespace
    path = source if source.is_file() else bytecode
    if source.is_file():
        spec = importlib.util.spec_from_file_location("networks.lmz", path)
    else:
        loader = importlib.machinery.SourcelessFileLoader("networks.lmz", str(path))
        spec = importlib.util.spec_from_loader("networks.lmz", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.DARENet(img_size=512)
    return model, str(path)


def _positions():
    return (0, 256, 512)


@torch.no_grad()
def sliding_logits(model, image, device):
    """Taper-weighted 512/256 logits merged on the native 1024 canvas."""
    if image.shape != (3, 1024, 1024):
        raise ValueError(f"Expected CHW 1024 image, got {tuple(image.shape)}")
    canvas = torch.zeros((1024, 1024), dtype=torch.float32, device=device)
    denominator = torch.zeros_like(canvas)
    ramp = (1 - torch.linspace(-1, 1, 512, device=device).abs()).clamp_min(0.05)
    weight = torch.outer(ramp, ramp)
    for top in _positions():
        for left in _positions():
            tile = image[:, top:top + 512, left:left + 512].unsqueeze(0).to(device)
            output = model(tile)
            if not torch.is_tensor(output):
                raise ValueError("DARENet must return a single logits tensor")
            if output.shape == (1, 1, 512, 512):
                output = output[0, 0]
            elif output.shape == (1, 512, 512):
                output = output[0]
            else:
                raise ValueError(f"Unexpected DARENet output: {tuple(output.shape)}")
            canvas[top:top + 512, left:left + 512] += output.float() * weight
            denominator[top:top + 512, left:left + 512] += weight
    return canvas / denominator


def counts(logits, mask, threshold):
    predicted = torch.sigmoid(logits) >= threshold
    target = mask.to(logits.device) > 0
    return (int((predicted & target).sum()), int((predicted & ~target).sum()),
            int((~predicted & target).sum()))


def metrics(tp, fp, fn):
    return {"iou": tp / max(tp + fp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
