import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_frozen_feature_probe import (
    apply_checkpoint_args,
    attach_stage3_feature_hooks,
    build_model,
    resize_feature,
)
from losses.road_losses import build_connectivity_target


class ConnectivityProbe(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, x):
        return self.net(x)


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_valid_pixels(feature, target, valid, max_pixels, generator):
    # feature: [B,C,H,W], target: [B,8,H,W], valid: [B,1,H,W]
    valid_flat = valid.squeeze(1).reshape(-1).bool()
    idx = valid_flat.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None, None
    if max_pixels > 0 and idx.numel() > max_pixels:
        perm = torch.randperm(idx.numel(), generator=generator)[:max_pixels].to(idx.device)
        idx = idx[perm]

    feat_flat = feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, 8)
    return feat_flat[idx].float().cpu(), target_flat[idx].float().cpu()


def collect_probe_data(args, model, swin, batches):
    block = attach_stage3_feature_hooks(swin)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    xs = []
    ys = []
    road_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for batch in tqdm(batches, desc="Collect Con hidden"):
            images = batch["image"].to(args.device).float()
            skeleton_gt = batch["skeleton"].to(args.device).float()
            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0]

            con_hidden = block._probe_hidden
            if con_hidden is None:
                raise RuntimeError("Stage3 structure_branch hook did not capture Con hidden.")
            con_hidden = con_hidden.to(args.device)
            con_hidden = resize_feature(con_hidden, surface_logits.shape[-2:]).to(args.device)

            skeleton_gt = F.interpolate(
                skeleton_gt,
                size=surface_logits.shape[-2:],
                mode="nearest",
            )
            connectivity_gt = build_connectivity_target(skeleton_gt, erode_kernel_size=args.erode_kernel_size)
            connectivity_gt = F.interpolate(
                connectivity_gt,
                size=con_hidden.shape[-2:],
                mode="nearest",
            )
            valid = F.interpolate(
                skeleton_gt,
                size=con_hidden.shape[-2:],
                mode="nearest",
            ) > 0.5

            road_pixels += int(valid.sum().item())
            total_pixels += int(valid.numel())
            x, y = sample_valid_pixels(
                con_hidden.detach(),
                connectivity_gt.detach(),
                valid.detach(),
                args.samples_per_batch,
                generator,
            )
            if x is not None:
                xs.append(x)
                ys.append(y)

    if not xs:
        raise RuntimeError("No valid road pixels were collected for the connectivity probe.")
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    if args.max_samples > 0 and x_all.shape[0] > args.max_samples:
        idx = torch.randperm(x_all.shape[0], generator=generator)[:args.max_samples]
        x_all = x_all[idx]
        y_all = y_all[idx]
    return x_all, y_all, road_pixels, total_pixels


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true, y_score):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def train_probe(args, x, y):
    indices = np.arange(x.shape[0])
    stratify = (y.sum(dim=1) > 0).numpy().astype(np.int64)
    if np.unique(stratify).size < 2:
        stratify = None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    device = args.device
    model = ConnectivityProbe(x.shape[1], args.hidden_dim).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)

    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(max=args.max_pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch_size = min(args.probe_batch_size, x_train.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    model.train()
    for epoch in range(args.probe_epochs):
        perm = torch.randperm(x_train.shape[0], generator=generator)
        total_loss = 0.0
        count = 0
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size].to(device)
            logits = model(x_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * idx.numel()
            count += idx.numel()
        if (epoch + 1) in {1, args.probe_epochs}:
            print(f"probe epoch {epoch + 1}/{args.probe_epochs}: loss={total_loss / max(count, 1):.6f}")

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_test)).cpu().numpy()
    y_np = y_test.cpu().numpy().astype(np.int32)
    return y_np, prob, train_idx, test_idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=r"D:\Code\Swin-Unet-main\data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--samples_per_batch", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=160000)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--probe_batch_size", type=int, default=8192)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--erode_kernel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
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
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--structure_profile", type=str, default="full")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23", choices=["stage2", "stage3", "stage23"])
    parser.add_argument(
        "--highres_structure_fusion_mode",
        type=str,
        default="stage23",
        choices=["stage23", "final_correction", "stage23_final_correction", "post_refine_interaction", "none"],
    )
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_deterministic(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    apply_checkpoint_args(args, checkpoint)
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batches = []
    for idx, batch in enumerate(loader):
        if args.max_batches > 0 and idx >= args.max_batches:
            break
        batches.append(batch)

    model = build_model(args, checkpoint)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    x, y, road_pixels, total_pixels = collect_probe_data(args, model, swin, batches)
    y_np, prob, train_idx, test_idx = train_probe(args, x, y)

    flat_auc = safe_auc(y_np.reshape(-1), prob.reshape(-1))
    flat_auprc = safe_auprc(y_np.reshape(-1), prob.reshape(-1))
    per_dir_auc = [safe_auc(y_np[:, i], prob[:, i]) for i in range(8)]
    per_dir_auprc = [safe_auprc(y_np[:, i], prob[:, i]) for i in range(8)]
    pos_rate = y_np.mean(axis=0)

    print("\n========== Con hidden -> 8 connectivity MLP probe ==========")
    print(f"Checkpoint: {args.model_path}")
    print(f"split={args.split}, batches={len(batches)}, images={sum(batch['image'].shape[0] for batch in batches)}")
    print(f"Con hidden samples: total={x.shape[0]}, train={len(train_idx)}, test={len(test_idx)}, channels={x.shape[1]}")
    print(f"Valid skeleton pixels seen: {road_pixels}/{total_pixels} ({road_pixels / max(total_pixels, 1):.6f})")
    print("Target: build_connectivity_target(skeleton_gt), valid_mask=skeleton_gt")
    print(f"Overall AUROC: {flat_auc:.4f}")
    print(f"Overall AUPRC: {flat_auprc:.4f}")
    print("\nPer-direction:")
    print(f"{'dir':>3} {'pos_rate':>10} {'AUROC':>10} {'AUPRC':>10}")
    for i, (auc, auprc) in enumerate(zip(per_dir_auc, per_dir_auprc)):
        print(f"{i:>3} {pos_rate[i]:>10.6f} {auc:>10.4f} {auprc:>10.4f}")
    print("\nInterpretation:")
    print("  AUROC > 0.8: Con hidden contains decodable connectivity information; inspect/fix connectivity head.")
    print("  AUROC ~ 0.5: Con hidden does not carry connectivity; supervision/representation needs redesign.")


if __name__ == "__main__":
    main()
