import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from topology_direction_constants import AXIAL_DIR_NAMES, axial_double_angle_basis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=r"D:\Code\Swin-Unet-main\data1")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    args = parser.parse_args()

    ds = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
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

    first = next(iter(dl))
    print(f"image shape      = {tuple(first['image'].shape)}")
    print(f"mask shape       = {tuple(first['mask'].shape)}")
    print(f"skeleton shape    = {tuple(first['skeleton'].shape)}")
    print(f"direction_gt shape= {tuple(first['direction_gt'].shape)}")
    print(f"direction_gt dtype= {first['direction_gt'].dtype}")

    offsets = axial_double_angle_basis()
    counts = torch.zeros(4, dtype=torch.long)
    total_skel = 0
    total_px = 0
    junction_like = 0
    endpoint_like = 0
    ignore_like = 0
    shapes = set()

    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.max_batches:
            break
        direction_gt = batch["direction_gt"].float()
        skeleton = batch["skeleton"].float()
        shapes.add(tuple(direction_gt.shape))

        vec = F.normalize(direction_gt, dim=1, eps=1e-6)
        score = torch.einsum("bchw,kc->bkhw", vec, offsets)
        idx = score.argmax(dim=1)
        mask = skeleton > 0.5

        total_skel += int(mask.sum().item())
        total_px += int(mask.numel())
        for k in range(4):
            counts[k] += int((idx[mask.squeeze(1)] == k).sum().item())

        magnitude = torch.linalg.vector_norm(direction_gt, dim=1)
        ignore_like += int(((magnitude < 1e-6) & mask.squeeze(1)).sum().item())
        junction_like += int(((skeleton > 0.5).sum(dim=1, keepdim=True) >= 2).sum().item())
        endpoint_like += int(((skeleton > 0.5).sum(dim=1, keepdim=True) == 1).sum().item())

    print(f"batch direction_gt shapes = {sorted(shapes)}")
    print(f"valid skeleton pixels     = {total_skel}")
    print(f"skeleton pixel ratio      = {total_skel / max(total_px, 1):.6f}")
    print("direction axis counts     =", dict(zip(AXIAL_DIR_NAMES, counts.tolist())))
    print(
        "direction axis ratios     =",
        {n: round(c / max(total_skel, 1), 4) for n, c in zip(AXIAL_DIR_NAMES, counts.tolist())},
    )
    print(f"zero-vector-like pixels   = {ignore_like}")
    print(f"junction-like count        = {junction_like}")
    print(f"endpoint-like count        = {endpoint_like}")


if __name__ == "__main__":
    main()
