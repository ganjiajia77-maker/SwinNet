"""Lightweight directional Pixel Shape Index descriptors for road imagery."""

import math

import torch
import torch.nn.functional as F


# Eight unique, undirected lattice directions. These are primitive offsets,
# rather than rounded samples from an angle grid, so no direction is repeated.
PSI_DIRECTIONS = (
    (0, 1),
    (1, 2),
    (1, 1),
    (2, 1),
    (1, 0),
    (2, -1),
    (1, -1),
    (1, -2),
)


def _shift_rgb(image, dy, dx):
    """Return image[y+dy, x+dx] and an in-bounds mask, without wraparound."""
    batch, channels, height, width = image.shape
    yy = torch.arange(height, device=image.device) + int(dy)
    xx = torch.arange(width, device=image.device) + int(dx)
    valid = (yy[:, None] >= 0) & (yy[:, None] < height)
    valid = valid & (xx[None, :] >= 0) & (xx[None, :] < width)
    shifted = image[:, :, yy.clamp(0, height - 1)[:, None], xx.clamp(0, width - 1)[None, :]]
    return shifted, valid.view(1, 1, height, width)


def psi_directional_descriptor(
    normalized_rgb,
    max_extension=4,
    color_threshold=40.0,
    output_size=None,
):
    """Compute per-direction similarity-line lengths plus shape statistics.

    The input follows the dataset's ImageNet normalization. Geometry and the
    existing appearance augmentation are already applied by the dataset, so
    descriptors remain pixel-aligned with the image and labels.

    Returns D directional channels followed by normalized PSI sum, max, min,
    and log elongation. Color comparisons use RGB values in [0, 255].
    """
    if normalized_rgb.ndim != 4 or normalized_rgb.shape[1] != 3:
        raise ValueError("PSI expects an RGB tensor with shape [B, 3, H, W].")

    image = normalized_rgb.float()
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    image = ((image * std + mean) * 255.0).clamp(0.0, 255.0)
    center = image
    lengths = []
    extension = max(1, int(max_extension))
    threshold = float(color_threshold)

    with torch.no_grad():
        for dy, dx in PSI_DIRECTIONS:
            positive = torch.ones_like(center[:, :1], dtype=torch.bool)
            negative = torch.ones_like(center[:, :1], dtype=torch.bool)
            positive_length = torch.zeros_like(center[:, :1])
            negative_length = torch.zeros_like(center[:, :1])
            for step in range(1, extension + 1):
                neighbor, valid = _shift_rgb(image, dy * step, dx * step)
                similar = ((neighbor - center).square().sum(1, keepdim=True).sqrt() <= threshold) & valid
                positive = positive & similar
                positive_length = positive_length + positive.float()

                neighbor, valid = _shift_rgb(image, -dy * step, -dx * step)
                similar = ((neighbor - center).square().sum(1, keepdim=True).sqrt() <= threshold) & valid
                negative = negative & similar
                negative_length = negative_length + negative.float()
            lengths.append((1.0 + positive_length + negative_length) / float(2 * extension + 1))

        directional = torch.cat(lengths, dim=1)
        length_sum = directional.mean(dim=1, keepdim=True)
        length_max = directional.amax(dim=1, keepdim=True)
        length_min = directional.amin(dim=1, keepdim=True)
        elongation = torch.log1p(
            (length_max / length_min.clamp_min(1.0 / float(2 * extension + 1))).clamp_min(1.0)
        ) / math.log(1.0 + float(2 * extension + 1))
        descriptor = torch.cat(
            [directional, length_sum, length_max, length_min, elongation], dim=1
        )
        if output_size is not None and descriptor.shape[-2:] != tuple(output_size):
            descriptor = F.adaptive_avg_pool2d(descriptor, output_size=output_size)
    return descriptor
