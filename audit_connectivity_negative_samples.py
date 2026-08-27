import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.skeleton_guided_head import PairwiseConnectivityHead
from topology_direction_constants import CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=4)
    args = parser.parse_args()

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    pos = torch.zeros(8, dtype=torch.float64)
    neg = torch.zeros(8, dtype=torch.float64)
    neg_p_skel_q_bg = torch.zeros(8, dtype=torch.float64)
    neg_p_skel_q_skel = torch.zeros(8, dtype=torch.float64)
    neg_p_bg = torch.zeros(8, dtype=torch.float64)

    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        skeleton = batch["skeleton"].float()
        connectivity = batch["connectivity_gt"].float()
        p_skel = skeleton > 0.5
        for idx, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            q_skel = PairwiseConnectivityHead._shift_feature(skeleton, dy, dx) > 0.5
            target = connectivity[:, idx:idx + 1] > 0.5
            pos[idx] += float((target & p_skel).sum().item())
            neg_mask = ~target
            neg[idx] += float(neg_mask.sum().item())
            neg_p_skel_q_bg[idx] += float((neg_mask & p_skel & (~q_skel)).sum().item())
            neg_p_skel_q_skel[idx] += float((neg_mask & p_skel & q_skel).sum().item())
            neg_p_bg[idx] += float((neg_mask & (~p_skel)).sum().item())

    print("Connectivity negative sample composition")
    print(f"split={args.split}, max_batches={args.max_batches}, batch_size={args.batch_size}")
    print("dir pos neg pos_ratio neg_p_skel_q_bg neg_p_skel_q_skel neg_p_bg")
    for idx, name in enumerate(CONNECTIVITY_DIR_NAMES):
        total = pos[idx] + neg[idx]
        print(
            f"{name:2s} "
            f"{int(pos[idx].item())} "
            f"{int(neg[idx].item())} "
            f"{float(pos[idx] / total.clamp_min(1.0)):.6f} "
            f"{int(neg_p_skel_q_bg[idx].item())} "
            f"{int(neg_p_skel_q_skel[idx].item())} "
            f"{int(neg_p_bg[idx].item())}"
        )

    print(
        "\nNote: stage training with --masked_connectivity_center_experiment masks loss to p_skeleton pixels, "
        "so neg_p_bg is excluded from the stage connectivity loss even if it exists in the raw label tensor."
    )


if __name__ == "__main__":
    main()

