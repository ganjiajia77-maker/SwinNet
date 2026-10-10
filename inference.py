"""Full-resolution, tapered overlap inference for WeavingUnet probabilities."""

import torch


def positions(length, tile, stride):
    if length < tile or not 0 < stride <= tile:
        raise ValueError("Image must cover a tile and stride must be in (0, tile]")
    return sorted(set(range(0, length - tile + 1, stride)) | {length - tile})


@torch.inference_mode()
def predict_full(model, image, tile_size=512, stride=256):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected one image in BCHW format")
    height, width = image.shape[-2:]
    edge = (1 - torch.linspace(-1, 1, tile_size, device=image.device).abs()).clamp_min(0.1)
    weight = (edge[:, None] * edge[None, :])[None, None]
    numerator = image.new_zeros((1, 1, height, width))
    denominator = image.new_zeros((1, 1, height, width))
    for top in positions(height, tile_size, stride):
        for left in positions(width, tile_size, stride):
            tile = image[:, :, top:top + tile_size, left:left + tile_size]
            probability = model(tile)
            if probability.shape != (1, 1, tile_size, tile_size) or not torch.isfinite(probability).all():
                raise ValueError(f"Invalid model output for tile at {top}, {left}")
            numerator[:, :, top:top + tile_size, left:left + tile_size] += probability * weight
            denominator[:, :, top:top + tile_size, left:left + tile_size] += weight
    if (denominator <= 0).any():
        raise ValueError("Tiling left uncovered pixels")
    return numerator / denominator


def counts(probability, target, threshold):
    pred = probability >= threshold
    gt = target.bool()
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def metrics(tp, fp, fn):
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(iou=iou, f1=f1, precision=precision, recall=recall)
