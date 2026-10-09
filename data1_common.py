"""Data1 crops, SegRoadv2 losses, and native-resolution sliding inference."""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
UPSTREAM_COMMIT = 'ab6924ecedbe4bef36713e9b3be6d5272dec1986'


def case_id(path):
    return re.sub(r'(?:_(?:surface_pred|mask_pred|pred|sat|image|img|mask|label))+$',
                  '', Path(path).stem)


def split_index(root, split):
    root = Path(root)
    image_dir = root / split / 'image'
    label_dir = root / split / 'mask'
    if not label_dir.is_dir():
        label_dir = root / split / 'label'
    indexed = []
    for directory in (image_dir, label_dir):
        cases = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                key = case_id(path)
                if key in cases:
                    raise ValueError('Duplicate case {} in {}'.format(key, directory))
                cases[key] = path
        if not cases:
            raise ValueError('No data in ' + str(directory))
        indexed.append(cases)
    images, labels = indexed
    if images.keys() != labels.keys():
        raise ValueError('Image/GT pairing mismatch: missing={}, extra={}'.format(
            sorted(images.keys() - labels.keys())[:8],
            sorted(labels.keys() - images.keys())[:8]))
    return [(images[key], labels[key]) for key in sorted(images)]


def load_case(paths, source_size=1024):
    with Image.open(paths[0]) as handle:
        image = np.asarray(handle.convert('RGB'))
    with Image.open(paths[1]) as handle:
        target = np.asarray(handle)
    if target.ndim == 3 and target.shape[2] == 3:
        if not (np.array_equal(target[..., 0], target[..., 1]) and
                np.array_equal(target[..., 0], target[..., 2])):
            raise ValueError('GT RGB channels differ: ' + str(paths[1]))
        target = target[..., 0]
    expected = (source_size, source_size)
    if image.shape[:2] != expected or target.shape != expected:
        raise ValueError('Native {} frame required; no resize/crop/pad: {} {}, {} {}'.format(
            expected, paths[0], image.shape, paths[1], target.shape))
    values = np.unique(target)
    if not np.isin(values, [0, 1, 255]).all() or (1 in values and 255 in values):
        raise ValueError('GT must be binary 0/1 or 0/255: ' + str(paths[1]))
    return image, target > 0


def image_tensor(image):
    array = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Data1Train(Dataset):
    """One deterministic random crop per image, with epoch fixed in each loader."""

    def __init__(self, root, seed=1234, source_size=1024, crop_size=512, epoch=0,
                 index=None):
        if source_size != 1024 or crop_size != 512:
            raise ValueError('This comparison requires native 1024 -> one random 512 crop')
        self.index = split_index(root, 'train') if index is None else index
        self.seed, self.epoch = int(seed), int(epoch)
        self.source_size, self.crop_size = source_size, crop_size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        image, mask = load_case(self.index[index], self.source_size)
        rng = np.random.RandomState((self.seed + self.epoch * 1000003 + index * 9176) % 2**32)
        if rng.rand() < 0.5:
            image, mask = np.flip(image, 1), np.flip(mask, 1)
        if rng.rand() < 0.5:
            image, mask = np.flip(image, 0), np.flip(mask, 0)
        turns = rng.randint(0, 4)
        if turns:
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        image = image.copy().astype(np.float32)
        mask = mask.copy()
        # Match the data1 OARENet/CoANet comparison augmentation.
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
        top = int(rng.randint(0, self.source_size - self.crop_size + 1))
        left = int(rng.randint(0, self.source_size - self.crop_size + 1))
        size = self.crop_size
        return {'image': image_tensor(image[top:top + size, left:left + size]),
                'mask': torch.from_numpy(mask[top:top + size, left:left + size].copy()).long(),
                'name': self.index[index][0].name, 'index': index,
                'crop_epoch': self.epoch, 'crop_top': top, 'crop_left': left}


def connectivity_targets(mask, spacing):
    """Source row-major 3x3 neighborhood, including center; no boundary wrap."""
    height, width = mask.shape[-2:]
    padded = F.pad(mask.float(), (spacing, spacing, spacing, spacing))
    neighbors = [padded[..., spacing + dy:spacing + dy + height,
                        spacing + dx:spacing + dx + width]
                 for dy in (-spacing, 0, spacing) for dx in (-spacing, 0, spacing)]
    return torch.stack(neighbors, dim=1) * mask.unsqueeze(1)


def source_loss(outputs, mask, class_weights, target_mode='source_fixed'):
    surface, near, far = outputs
    near_target = connectivity_targets(mask, 2)
    far_target = connectivity_targets(mask, 4)
    if target_mode == 'source_legacy':
        # Exact original get_con_3 channel-overwrite bug, for explicit reproduction only.
        legacy = torch.zeros_like(far_target)
        legacy[:, :3] = far_target[:, 6:9]
        far_target = legacy
    seg_loss = F.cross_entropy(surface, mask, weight=class_weights)
    near_loss = F.binary_cross_entropy_with_logits(near, near_target)
    far_loss = F.binary_cross_entropy_with_logits(far, far_target)
    total = seg_loss + 0.4 * (0.4 * near_loss + 0.6 * far_loss)
    return total, (seg_loss, near_loss, far_loss)


def create_model(phi='b2', pretrain_ckpt=None):
    from nets.segformer import SegFormer
    from nets.segformer_training import weights_init
    model = SegFormer(num_classes=2, phi=phi, pretrained=False)
    if pretrain_ckpt:
        weights_init(model)
        checkpoint = torch.load(pretrain_ckpt, map_location='cpu', weights_only=False)
        for wrapper in ('state_dict', 'model'):
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get(wrapper), dict):
                checkpoint = checkpoint[wrapper]
        expected = model.backbone.state_dict()
        loaded, ignored = {}, []
        for name, value in checkpoint.items():
            key = name.removeprefix('module.').removeprefix('backbone.')
            if key in expected and isinstance(value, torch.Tensor) and value.shape == expected[key].shape:
                loaded[key] = value
            else:
                ignored.append(name)
        # MiT has no SegRoadv2 added DCN or deformable-query offset parameters.
        required = [name for name in expected if '.dcn.' not in name and '.q_offset.' not in name]
        missing = sorted(set(required) - loaded.keys())
        if missing:
            raise ValueError('Wrong/incomplete {} MiT backbone checkpoint; missing {}: {}'.format(
                phi, len(missing), missing[:12]))
        model.backbone.load_state_dict(loaded, strict=False)
        print('PRETRAIN loaded {}/{} backbone tensors; new DCN/offset tensors={}; ignored={}'.format(
            len(loaded), len(expected), len(expected) - len(loaded), ignored[:10]), flush=True)
    return model


def load_model(path, device, weights='ema'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if weights == 'ema':
        state = checkpoint.get('ema_state_dict')
        if state is None:
            raise ValueError('EMA weights required but absent: ' + str(path))
    else:
        state = checkpoint['training_model_state_dict']
    model = create_model(config['phi'])
    model.load_state_dict(state, strict=True)
    print('Loaded SegRoadv2 {} epoch={} weights={}'.format(
        config['phi'], checkpoint['epoch'], weights), flush=True)
    del checkpoint
    return model.to(device).eval(), config


def positions(length, tile, stride):
    if not 0 < stride <= tile <= length:
        raise ValueError('Require 0 < stride <= tile <= native image size')
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.no_grad()
def predict_full(model, image, device, tile=512, stride=256, tta=True):
    height, width = image.shape[:2]
    sums = [np.zeros((height, width), dtype=np.float32) for _ in range(3)]
    count = np.zeros((height, width), dtype=np.float32)
    flips = ((), (2,), (3,), (2, 3)) if tta else ((),)
    for top in positions(height, tile, stride):
        for left in positions(width, tile, stride):
            tensor = image_tensor(image[top:top + tile, left:left + tile]).unsqueeze(0).to(device)
            views = [[], [], []]
            for dims in flips:
                surface, near, far = model(torch.flip(tensor, dims) if dims else tensor)
                heads = (surface.softmax(dim=1)[:, 1:2],
                         near.sum(dim=1, keepdim=True), far.sum(dim=1, keepdim=True))
                for values, head in zip(views, heads):
                    restored = torch.flip(head, dims) if dims else head
                    if not torch.isfinite(restored).all():
                        raise RuntimeError('Non-finite SegRoadv2 prediction')
                    values.append(restored)
            for total, values in zip(sums, views):
                value = torch.stack(values).mean(dim=0)[0, 0].cpu().numpy()
                total[top:top + tile, left:left + tile] += value
            count[top:top + tile, left:left + tile] += 1.0
    if not (count > 0).all():
        raise RuntimeError('Sliding inference left uncovered pixels')
    return tuple(value / count for value in sums)


def binary_prediction(components, threshold, mode='source_fusion'):
    surface, near, far = components
    result = surface >= threshold
    if mode == 'source_fusion':
        # Original segformer.py combines raw connectivity-logit sums with surface.
        result = result | (near >= 3.0) | (far >= 1.5)
    elif mode != 'surface':
        raise ValueError('Unknown prediction mode: ' + mode)
    return result


def pixel_counts(prediction, target):
    return (int((prediction & target).sum()), int((prediction & ~target).sum()),
            int((~prediction & target).sum()))


def metrics(tp, fp, fn):
    return {'iou': tp / max(tp + fp + fn, 1), 'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1)}
