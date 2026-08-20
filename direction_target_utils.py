import math

import numpy as np
import torch


def build_continuous_direction_target(skeleton, radius=3):
    if skeleton.dim() == 3:
        skeleton = skeleton.unsqueeze(1)
    if skeleton.dim() != 4:
        raise ValueError(
            f"Expected skeleton shape [B,1,H,W] or [B,H,W], got {tuple(skeleton.shape)}"
        )

    device = skeleton.device
    dtype = skeleton.dtype
    batch_targets = []

    for b in range(skeleton.shape[0]):
        skel = (skeleton[b, 0] > 0.5).detach().cpu().numpy().astype(np.float32)
        height, width = skel.shape
        target = np.zeros((2, height, width), dtype=np.float32)
        ys, xs = np.where(skel > 0.5)

        for y, x in zip(ys.tolist(), xs.tolist()):
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            patch = np.argwhere(skel[y0:y1, x0:x1] > 0.5)
            if patch.shape[0] < 2:
                continue

            coords = patch.astype(np.float32)
            coords[:, 0] += y0
            coords[:, 1] += x0
            coords[:, 0] -= coords[:, 0].mean()
            coords[:, 1] -= coords[:, 1].mean()

            cov = np.matmul(coords.T, coords) / float(coords.shape[0])
            if not np.isfinite(cov).all():
                continue

            evals, evecs = np.linalg.eigh(cov)
            tangent = evecs[:, int(np.argmax(evals))]
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-8:
                continue
            tangent = tangent / norm
            theta = math.atan2(float(tangent[0]), float(tangent[1]))
            target[0, y, x] = math.cos(2.0 * theta)
            target[1, y, x] = math.sin(2.0 * theta)

        batch_targets.append(target)

    return torch.from_numpy(np.stack(batch_targets, axis=0)).to(device=device, dtype=dtype)
