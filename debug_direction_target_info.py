import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from topology_direction_constants import CONNECTIVITY_DIR_NAMES, connectivity_double_angle_basis

DIR_NAMES = list(CONNECTIVITY_DIR_NAMES)
DIR_OFFSETS = connectivity_double_angle_basis()


def angle_from_vec(vec):
    dy, dx = float(vec[0]), float(vec[1])
    return math.atan2(dy, dx) % math.pi


def axial_diff(theta1, theta2):
    diff = abs(theta1 - theta2) % math.pi
    return min(diff, math.pi - diff)


def vec_from_angle(theta):
    return np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)


def rounded_vec_key(vec, decimals=3):
    return tuple(np.round(np.asarray(vec, dtype=np.float32), decimals=decimals).tolist())


def draw_line_skeleton(kind, size=33):
    s = torch.zeros(1, 1, size, size, dtype=torch.float32)
    c = size // 2
    coords = []
    if kind == "horizontal":
        coords = [(c, x) for x in range(size)]
    elif kind == "vertical":
        coords = [(y, c) for y in range(size)]
    else:
        angle_deg = {
            "45": 45.0,
            "30": 30.0,
            "60": 60.0,
            "135": 135.0,
        }[kind]
        theta = math.radians(angle_deg)
        dy = math.sin(theta)
        dx = math.cos(theta)
        half = size // 2
        for t in range(-half, half + 1):
            y = int(round(c + t * dy))
            x = int(round(c + t * dx))
            if 0 <= y < size and 0 <= x < size:
                coords.append((y, x))
    for y, x in coords:
        s[0, 0, y, x] = 1.0
    return s


def old_direction_target(skeleton):
    return RoadSkeletonDataset._build_direction_target(skeleton)


def estimate_local_tangent_from_neighbors(skeleton_2d, y, x, radius=5):
    h, w = skeleton_2d.shape
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    patch = skeleton_2d[y0:y1, x0:x1] > 0.5
    pts = np.argwhere(patch)
    if pts.shape[0] < 2:
        return np.array([0.0, 1.0], dtype=np.float32)
    pts = pts.astype(np.float32)
    pts[:, 0] += y0 - y
    pts[:, 1] += x0 - x
    cov = np.cov(pts.T)
    if not np.isfinite(cov).all():
        return np.array([0.0, 1.0], dtype=np.float32)
    evals, evecs = np.linalg.eigh(cov)
    tangent = evecs[:, int(np.argmax(evals))]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-8:
        return np.array([0.0, 1.0], dtype=np.float32)
    tangent = tangent / norm
    # tangent is (dy, dx)
    return tangent.astype(np.float32)


def continuous_doubled_angle_target(skeleton, radius=5):
    if skeleton.dim() == 3:
        skeleton = skeleton.unsqueeze(0)
    if skeleton.dim() != 4:
        raise ValueError(f"Expected skeleton shape [B,1,H,W] or [B,H,W], got {tuple(skeleton.shape)}")
    batch = []
    for b in range(skeleton.shape[0]):
        skel_2d = skeleton[b, 0].detach().cpu().numpy().astype(np.float32)
        h, w = skel_2d.shape
        out = np.zeros((2, h, w), dtype=np.float32)
        ys, xs = np.where(skel_2d > 0.5)
        for y, x in zip(ys.tolist(), xs.tolist()):
            tangent = estimate_local_tangent_from_neighbors(skel_2d, y, x, radius=radius)
            theta = math.atan2(float(tangent[0]), float(tangent[1]))
            out[0, y, x] = math.cos(2.0 * theta)
            out[1, y, x] = math.sin(2.0 * theta)
        batch.append(out)
    return torch.from_numpy(np.stack(batch, axis=0)).float()


def make_dir_hist_from_vec_map(vec_map, mask):
    offsets = F.normalize(DIR_OFFSETS.to(vec_map.device, vec_map.dtype), dim=1, eps=1e-6)
    vec = F.normalize(vec_map, dim=1, eps=1e-6)
    score = torch.einsum("bchw,kc->bkhw", vec, offsets)
    idx = score.argmax(dim=1)
    hist = torch.bincount(idx[mask.squeeze(1)], minlength=8)
    return hist


def vec_to_theta_map(vec_map):
    if vec_map.dim() == 4:
        dy = vec_map[:, 0]
        dx = vec_map[:, 1]
    else:
        dy = vec_map[0]
        dx = vec_map[1]
    return torch.atan2(dy, dx) % math.pi


def report_artificial_cases():
    cases = [
        ("horizontal", "horizontal"),
        ("vertical", "vertical"),
        ("45", "45"),
        ("30", "30"),
        ("60", "60"),
        ("135", "135"),
    ]
    print("=" * 80)
    print("Artificial skeleton sanity check")
    print("=" * 80)
    for label, kind in cases:
        skel = draw_line_skeleton(kind)
        old = old_direction_target(skel)
        cont = continuous_doubled_angle_target(skel, radius=5)
        mask = skel > 0.5
        cy = skel.shape[-2] // 2
        cx = skel.shape[-1] // 2

        old_vecs = old[0, :, mask[0, 0]].T.detach().cpu().numpy()
        cont_vecs = cont[0, :, mask[0, 0]].T.detach().cpu().numpy()
        old_unique = sorted({rounded_vec_key(v) for v in old_vecs})
        cont_mean = cont_vecs.mean(axis=0)
        print(f"\n{label}")
        print(f"  old unique vectors: {old_unique}")
        print(f"  old center vector  : {old[0, :, cy, cx].tolist()}")
        print(f"  continuous mean vec: {cont_mean.tolist()}")
        print(f"  continuous center  : {cont[0, :, cy, cx].tolist()}")


def sample_real_skeletons(args):
    ds = RoadSkeletonDataset(
        root_dir=args.root_path,
        split="test",
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path="",
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rng = random.Random(args.seed)
    sampled = []
    old_vector_keys = set()
    cont_vector_keys = set()
    old_hist = torch.zeros(8, dtype=torch.long)
    cont_angle_hist = torch.zeros(args.angle_bins, dtype=torch.long)
    all_old_axial = []
    all_cont_axial = []
    all_old_angles = []
    all_cont_angles = []

    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.max_batches:
            break

        skeleton = batch["skeleton"].float()
        old = old_direction_target(skeleton)
        mask = skeleton > 0.5

        old_hist += make_dir_hist_from_vec_map(old, mask)

        for b in range(skeleton.shape[0]):
            ys, xs = torch.where(mask[b, 0])
            coords = list(zip(ys.tolist(), xs.tolist()))
            if not coords:
                continue
            skel_2d = skeleton[b, 0].detach().cpu().numpy().astype(np.float32)
            rng.shuffle(coords)
            coords = coords[: min(args.max_pixels_per_image, len(coords))]
            for y, x in coords:
                old_vec = old[b, :, y, x].detach().cpu().numpy()
                tangent = estimate_local_tangent_from_neighbors(
                    skel_2d,
                    y,
                    x,
                    radius=args.neighbor_radius,
                )
                theta = math.atan2(float(tangent[0]), float(tangent[1]))
                cont_vec = np.array(
                    [math.cos(2.0 * theta), math.sin(2.0 * theta)],
                    dtype=np.float32,
                )
                old_vector_keys.add(rounded_vec_key(old_vec))
                cont_vector_keys.add(rounded_vec_key(cont_vec))
                theta_old = angle_from_vec(old_vec)
                theta_cont = angle_from_vec(cont_vec)
                all_old_angles.append(theta_old)
                all_cont_angles.append(theta_cont)
                all_old_axial.append(axial_diff(theta_old, theta_cont))
                all_cont_axial.append(axial_diff(theta_cont, theta_old))
                bin_idx = int((theta_cont / math.pi) * args.angle_bins) % args.angle_bins
                cont_angle_hist[bin_idx] += 1
                sampled.append((old_vec, cont_vec))

    return {
        "old_hist": old_hist,
        "cont_angle_hist": cont_angle_hist,
        "old_unique_count": len(old_vector_keys),
        "cont_unique_count": len(cont_vector_keys),
        "old_mean_axial_diff": float(np.mean(all_old_axial)) if all_old_axial else float("nan"),
        "cont_mean_axial_diff": float(np.mean(all_cont_axial)) if all_cont_axial else float("nan"),
        "old_angles": np.array(all_old_angles, dtype=np.float32),
        "cont_angles": np.array(all_cont_angles, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=r"D:\Code\Swin-Unet-main\data1")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--max_pixels_per_image", type=int, default=200)
    parser.add_argument("--neighbor_radius", type=int, default=5)
    parser.add_argument("--angle_bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    report_artificial_cases()

    print("\n" + "=" * 80)
    print("Real test skeleton sample")
    print("=" * 80)
    stats = sample_real_skeletons(args)
    print(f"old unique vector count      = {stats['old_unique_count']}")
    print(f"continuous unique vector count= {stats['cont_unique_count']}")
    print(f"old direction histogram      = {dict(zip(DIR_NAMES, stats['old_hist'].tolist()))}")
    print(f"continuous angle histogram   = {stats['cont_angle_hist'].tolist()}")
    print(f"old mean axial difference    = {stats['old_mean_axial_diff']:.6f}")
    print(f"continuous mean axial diff   = {stats['cont_mean_axial_diff']:.6f}")

    print("\n" + "=" * 80)
    print("Simple report")
    print("=" * 80)
    print("Old target:")
    print(f"  unique directions: {stats['old_unique_count']}")
    print(f"  mean axial difference: {stats['old_mean_axial_diff']:.6f}")
    print("Continuous target:")
    print(f"  unique directions: {stats['cont_unique_count']}")
    print(f"  mean axial difference: {stats['cont_mean_axial_diff']:.6f}")


if __name__ == "__main__":
    main()
