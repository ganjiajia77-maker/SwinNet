"""Full-resolution sliding-window evaluation for the data1 road head."""

import torch

from train_data1_road import road_logits


@torch.no_grad()
def sliding_probability(model, image, device, tile=512, stride=256, use_amp=True):
    if image.shape[0] != 1:
        raise ValueError("Sliding evaluation expects one full source image")
    _, height, width, _ = image.shape
    if height < tile or width < tile or (height - tile) % stride or (width - tile) % stride:
        raise ValueError(f"Image shape {(height, width)} is incompatible with tile={tile}, stride={stride}")

    # Accumulate float32 logits to avoid fp16 rounding in overlapping regions.
    canvas = torch.zeros((1, 1, height, width), device=device)
    weights = torch.zeros_like(canvas)
    ramp = (1 - torch.linspace(-1, 1, tile, device=device).abs()).clamp_min(0.1)
    tile_weight = (ramp[:, None] * ramp[None, :]).view(1, 1, tile, tile)
    for top in range(0, height - tile + 1, stride):
        for left in range(0, width - tile + 1, stride):
            patch = image[:, top:top + tile, left:left + tile].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp and device.type == "cuda"):
                logits = road_logits(model, patch)
            canvas[:, :, top:top + tile, left:left + tile] += logits.float() * tile_weight
            weights[:, :, top:top + tile, left:left + tile] += tile_weight
    if weights.min().item() <= 0:
        raise RuntimeError("Sliding evaluation left uncovered pixels")
    return torch.sigmoid(canvas / weights)


def confusion_counts(probability, target, threshold):
    pred = probability >= threshold
    gt = target > 0.5
    return (
        int((pred & gt).sum().item()),
        int((pred & ~gt).sum().item()),
        int((~pred & gt).sum().item()),
    )


def counts_to_metrics(counts):
    tp, fp, fn = counts
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "iou": tp / (tp + fp + fn + 1e-8),
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
        "precision": precision,
        "recall": recall,
    }
