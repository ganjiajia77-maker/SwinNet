import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model, resize_like, select_stage_outputs
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_connectivity_target, build_stage_skeleton_target
from networks.skeleton_guided_head import PairwiseConnectivityHead
from topology_direction_constants import CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS


def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_stage_stats(s_stage, c_train):
    c_ske = build_connectivity_target(s_stage)
    stats = []
    for idx, (name, (dy, dx)) in enumerate(zip(CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS)):
        train = c_train[:, idx:idx + 1] > 0.5
        regen = c_ske[:, idx:idx + 1] > 0.5
        q_skel = PairwiseConnectivityHead._shift_feature(s_stage, dy, dx) > 0.5
        p_skel = s_stage > 0.5

        total = train.numel()
        train_pos = int(train.sum().item())
        regen_pos = int(regen.sum().item())
        tp = int((train & regen).sum().item())
        fp = int((~train & regen).sum().item())
        fn = int((train & ~regen).sum().item())
        agree = int((train == regen).sum().item())
        q_given_train = int((train & q_skel).sum().item())
        p_given_train = int((train & p_skel).sum().item())
        stats.append(
            {
                "dir": name,
                "total": total,
                "train_pos": train_pos,
                "regen_pos": regen_pos,
                "agreement": agree / max(total, 1),
                "positive_agreement": tp / max(train_pos, 1),
                "false_positive": fp,
                "false_positive_rate": fp / max(total - train_pos, 1),
                "false_negative": fn,
                "false_negative_rate": fn / max(train_pos, 1),
                "p_skel_given_c": p_given_train / max(train_pos, 1),
                "q_skel_given_c": q_given_train / max(train_pos, 1),
            }
        )
    return stats


def add_stats(acc, rows):
    for row in rows:
        name = row["dir"]
        if name not in acc:
            acc[name] = {
                "total": 0,
                "train_pos": 0,
                "regen_pos": 0,
                "agree": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "p_given": 0,
                "q_given": 0,
            }
        item = acc[name]
        total = row["total"]
        train_pos = row["train_pos"]
        item["total"] += total
        item["train_pos"] += train_pos
        item["regen_pos"] += row["regen_pos"]
        item["agree"] += int(round(row["agreement"] * total))
        item["tp"] += int(round(row["positive_agreement"] * max(train_pos, 1))) if train_pos else 0
        item["fp"] += row["false_positive"]
        item["fn"] += row["false_negative"]
        item["p_given"] += int(round(row["p_skel_given_c"] * max(train_pos, 1))) if train_pos else 0
        item["q_given"] += int(round(row["q_skel_given_c"] * max(train_pos, 1))) if train_pos else 0


def print_stage(stage_name, acc):
    print(f"\n{stage_name}")
    print(
        "dir agreement positive_agreement false_positive false_negative "
        "P(S(p)=1|C=1) P(S(q_d)=1|C=1) train_pos regen_pos"
    )
    for name in CONNECTIVITY_DIR_NAMES:
        item = acc.get(name)
        if not item:
            continue
        total = item["total"]
        train_pos = item["train_pos"]
        train_neg = total - train_pos
        print(
            f"{name:2s} "
            f"{item['agree'] / max(total, 1):.6f} "
            f"{item['tp'] / max(train_pos, 1):.6f} "
            f"{item['fp']}({item['fp'] / max(train_neg, 1):.6f}) "
            f"{item['fn']}({item['fn'] / max(train_pos, 1):.6f}) "
            f"{item['p_given'] / max(train_pos, 1):.6f} "
            f"{item['q_given'] / max(train_pos, 1):.6f} "
            f"{train_pos} "
            f"{item['regen_pos']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--stage", type=str, default="both", choices=["both", "stage2_refine", "stage3_refine"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=["auto", "standard", "selective"])
    args = parser.parse_args()

    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

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
        pin_memory=True,
    )

    wanted = ["stage2_refine", "stage3_refine"] if args.stage == "both" else [args.stage]
    accumulators = {stage: {} for stage in wanted}
    batches = 0
    images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Stage target audit")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batches += 1
            images += int(batch["image"].shape[0])
            images_tensor = batch["image"].to(device)
            skeleton_gt = batch["skeleton"].to(device).float()

            outputs = model(images_tensor, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            selected = select_stage_outputs(outputs[-1])
            for stage in wanted:
                if stage not in selected:
                    raise RuntimeError(f"Stage {stage} not present in structure outputs.")
                c_logits = selected[stage]["connectivity"]
                s_stage = build_stage_skeleton_target(skeleton_gt, c_logits.shape[-2:]).to(device)
                c_train = build_connectivity_target(s_stage)
                add_stats(accumulators[stage], compute_stage_stats(s_stage, c_train))

    print(f"\nmodel_path={args.model_path}")
    print(f"split={args.split}, batches={batches}, images={images}")
    print("C_train: build_connectivity_target(S_stage), matching the updated training code")
    print("S_stage: full skeleton occupancy/max-pooled to stage connectivity resolution")
    print("C_ske: build_connectivity_target(S_stage); agreement should be exactly 1.0")
    for stage in wanted:
        print_stage(stage, accumulators[stage])


if __name__ == "__main__":
    main()
