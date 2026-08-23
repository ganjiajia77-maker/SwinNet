import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import CONNECTIVITY_DIRECTIONS
from networks.vision_transformer_selective_fusion import SwinUnet as ViT_seg


def make_config_args(saved_args, overrides):
    values = dict(saved_args or {})
    defaults = {
        "cfg": "./configs/swin_tiny_patch4_window7_224_lite.yaml",
        "batch_size": 4,
        "img_size": 256,
        "opts": None,
        "zip": False,
        "cache_mode": "",
        "resume": "",
        "accumulation_steps": 0,
        "use_checkpoint": False,
        "amp_opt_level": "",
        "tag": "",
        "eval": False,
        "throughput": False,
    }
    defaults.update(values)
    defaults.update({key: value for key, value in overrides.items() if value is not None})
    return SimpleNamespace(**defaults)


def build_model(config, args, device):
    return ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=getattr(args, "num_classes", 2),
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=getattr(args, "bottleneck_type", "global_local"),
        final_topology_eta_init=getattr(args, "final_topology_eta_init", 0.0),
        final_gap_rho_init=getattr(args, "final_gap_rho_init", 0.0),
        stage_topology_stages=getattr(args, "stage_topology_stages", "none"),
        stage_topology_alpha_max=getattr(args, "stage_topology_alpha_max", 1.0),
        stage_topology_alpha_init=getattr(args, "stage_topology_alpha_init", 0.1),
        stage_topology_bias_mode=getattr(args, "stage_topology_bias_mode", "pairwise_skeleton"),
        stage_topology_ratio=getattr(args, "stage_topology_ratio", 0.08),
        stage_topology_topo_clip=getattr(args, "stage_topology_topo_clip", 4.0),
        structure_profile=getattr(args, "structure_profile", "full"),
        enable_final_graph_prop=getattr(args, "enable_graph_prop", False),
        use_msfe_skip=not getattr(args, "disable_msfe_skip", False),
        stage2_skeleton_gradient_ratio=getattr(args, "stage2_skeleton_gradient_ratio", 0.5),
        stage3_skeleton_gradient_ratio=getattr(args, "stage3_skeleton_gradient_ratio", 0.5),
        final_skeleton_gradient_ratio=getattr(args, "final_skeleton_gradient_ratio", 0.0),
        enable_highres_structure_stream=getattr(args, "enable_highres_structure_stream", False),
        highres_structure_channels=getattr(args, "highres_structure_channels", 64),
        highres_structure_fuse_stages=getattr(args, "highres_structure_fuse_stages", "stage23"),
        highres_structure_fusion_mode=getattr(args, "highres_structure_fusion_mode", "stage23"),
        enable_post_refine_structure_interaction=getattr(
            args,
            "enable_post_refine_structure_interaction",
            False,
        ),
    ).to(device)


def shift_map(x, dy, dx):
    _, _, height, width = x.shape
    pad_left = max(-dx, 0)
    pad_right = max(dx, 0)
    pad_top = max(-dy, 0)
    pad_bottom = max(dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    y0 = max(dy, 0)
    x0 = max(dx, 0)
    return padded[:, :, y0:y0 + height, x0:x0 + width]


def boundary_valid_like(reference):
    batch, _, height, width = reference.shape
    valid = torch.ones((batch, 8, height, width), device=reference.device, dtype=torch.bool)
    for index, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        if dy < 0:
            valid[:, index, 0, :] = False
        if dy > 0:
            valid[:, index, -1, :] = False
        if dx < 0:
            valid[:, index, :, 0] = False
        if dx > 0:
            valid[:, index, :, -1] = False
    return valid


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class EdgeProbe(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class Reservoir:
    def __init__(self, max_samples, seed):
        self.max_samples = int(max_samples)
        self.rng = np.random.default_rng(seed)
        self.features = []
        self.labels = []
        self.count = 0

    def add(self, features, labels):
        if features.numel() == 0:
            return
        features = features.detach().cpu()
        labels = labels.detach().cpu().float()
        if self.count < self.max_samples:
            remaining = self.max_samples - self.count
            take = min(remaining, features.shape[0])
            self.features.append(features[:take])
            self.labels.append(labels[:take])
            self.count += take
            features = features[take:]
            labels = labels[take:]
        if features.shape[0] == 0:
            return
        existing_features = torch.cat(self.features, dim=0)
        existing_labels = torch.cat(self.labels, dim=0)
        for row_index in range(features.shape[0]):
            seen_index = self.count + row_index + 1
            replace_index = int(self.rng.integers(0, seen_index))
            if replace_index < self.max_samples:
                existing_features[replace_index] = features[row_index]
                existing_labels[replace_index] = labels[row_index]
        self.features = [existing_features]
        self.labels = [existing_labels]
        self.count += features.shape[0]

    def tensors(self):
        if not self.features:
            return None, None
        return torch.cat(self.features, dim=0), torch.cat(self.labels, dim=0)


def sample_edges_from_feature(feature, connectivity_gt, skeleton_mask, per_head_samples):
    if connectivity_gt.shape[-2:] != feature.shape[-2:]:
        connectivity_gt = F.interpolate(connectivity_gt.float(), size=feature.shape[-2:], mode="nearest")
        skeleton_mask = F.interpolate(skeleton_mask.float(), size=feature.shape[-2:], mode="nearest") > 0.5
    valid = boundary_valid_like(connectivity_gt) & skeleton_mask.expand_as(connectivity_gt)
    edge_features = []
    edge_labels = []
    for direction_index, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        direction_valid = valid[:, direction_index:direction_index + 1]
        if not direction_valid.any():
            continue
        center = feature
        neighbor = shift_map(feature, dy, dx)
        pair = torch.cat([center, neighbor], dim=1)
        pair = pair.permute(0, 2, 3, 1)[direction_valid.squeeze(1)]
        label = connectivity_gt[:, direction_index:direction_index + 1][direction_valid]
        edge_features.append(pair)
        edge_labels.append(label)
    if not edge_features:
        return None, None
    edge_features = torch.cat(edge_features, dim=0)
    edge_labels = torch.cat(edge_labels, dim=0)
    if edge_features.shape[0] > per_head_samples:
        index = torch.randperm(edge_features.shape[0], device=edge_features.device)[:per_head_samples]
        edge_features = edge_features[index]
        edge_labels = edge_labels[index]
    return edge_features, edge_labels


def register_feature_hooks(model):
    captured = {}
    handles = []

    def should_hook(name, module):
        if not name.endswith("connectivity_head"):
            return False
        if hasattr(module, "connectivity_channels"):
            return True
        return isinstance(module, nn.Conv2d) and module.out_channels == 8

    for name, module in model.named_modules():
        if not should_hook(name, module):
            continue

        def hook(_module, inputs, _output, key=name):
            if inputs:
                captured[key] = inputs[0].detach()

        handles.append(module.register_forward_hook(hook))
    return captured, handles


def collect_split_samples(model, loader, split_name, max_batches, per_batch_samples, max_samples, device, seed):
    captured, handles = register_feature_hooks(model)
    reservoirs = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            captured.clear()
            images = batch["image"].to(device)
            skeleton = batch["skeleton"].to(device) > 0.5
            connectivity_gt = batch["connectivity_gt"].to(device).float()
            _ = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            for head_name, feature in captured.items():
                edge_features, edge_labels = sample_edges_from_feature(
                    feature,
                    connectivity_gt,
                    skeleton,
                    per_batch_samples,
                )
                if edge_features is None:
                    continue
                if head_name not in reservoirs:
                    reservoirs[head_name] = Reservoir(max_samples, seed + len(reservoirs))
                reservoirs[head_name].add(edge_features, edge_labels)
            if (batch_index + 1) % 20 == 0:
                print(f"[{split_name}] sampled {batch_index + 1} batches", flush=True)
    for handle in handles:
        handle.remove()
    samples = {}
    for head_name, reservoir in reservoirs.items():
        features, labels = reservoir.tensors()
        if features is not None:
            samples[head_name] = (features, labels)
    return samples


def train_and_eval_probe(head_name, train_data, val_data, args, device):
    x_train, y_train = train_data
    x_val, y_val = val_data
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=0,
    )
    probe = EdgeProbe(x_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    n_pos = float(y_train.sum().item())
    n_neg = float(y_train.numel() - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    probe.train()
    for _epoch in range(args.probe_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = loss_fn(probe(batch_x), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, x_val.shape[0], args.probe_batch_size):
            batch_x = x_val[start:start + args.probe_batch_size].to(device)
            scores.append(torch.sigmoid(probe(batch_x)).cpu())
    scores = torch.cat(scores).numpy()
    labels = y_val.numpy().astype(np.uint8)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    row = {
        "head": head_name,
        "feature_dim": int(x_train.shape[1] // 2),
        "train_edges": int(y_train.numel()),
        "val_edges": int(y_val.numel()),
        "train_positive_ratio": float(y_train.mean().item()),
        "val_positive_ratio": float(y_val.mean().item()),
        "auroc": rank_auc(scores, labels),
        "pos_prob_mean": float(pos_scores.mean()) if pos_scores.size else float("nan"),
        "neg_prob_mean": float(neg_scores.mean()) if neg_scores.size else float("nan"),
        "prob_gap": (
            float(pos_scores.mean() - neg_scores.mean())
            if pos_scores.size and neg_scores.size
            else float("nan")
        ),
        "pos_prob_p50": float(np.quantile(pos_scores, 0.5)) if pos_scores.size else float("nan"),
        "neg_prob_p50": float(np.quantile(neg_scores, 0.5)) if neg_scores.size else float("nan"),
    }
    return row


def build_loader(config_args, split, batch_size, num_workers):
    dataset = RoadSkeletonDataset(
        root_dir=getattr(config_args, "root_path", "./data1"),
        split=split,
        image_size=getattr(config_args, "img_size", 256),
        source_patch_size=getattr(config_args, "source_patch_size", 1024),
        tile_size=None,
        tile_stride=getattr(config_args, "overlap_stride", 256),
        augment=False,
        return_full_image=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_batches", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--per_batch_samples", type=int, default=4096)
    parser.add_argument("--max_samples_per_split", type=int, default=200000)
    parser.add_argument("--probe_epochs", type=int, default=5)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    config_args = make_config_args(
        saved_args,
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "resume": "",
        },
    )
    config = get_config(config_args)
    device = torch.device(args.device)
    model = build_model(config, config_args, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    batch_size = args.batch_size or getattr(config_args, "batch_size", 4)
    train_loader = build_loader(config_args, "train", batch_size, args.num_workers)
    val_loader = build_loader(config_args, "val", batch_size, args.num_workers)

    train_samples = collect_split_samples(
        model,
        train_loader,
        "train",
        args.train_batches,
        args.per_batch_samples,
        args.max_samples_per_split,
        device,
        args.seed,
    )
    val_samples = collect_split_samples(
        model,
        val_loader,
        "val",
        args.val_batches,
        args.per_batch_samples,
        args.max_samples_per_split,
        device,
        args.seed + 1000,
    )

    rows = []
    for head_name in sorted(train_samples):
        if head_name not in val_samples:
            continue
        row = train_and_eval_probe(head_name, train_samples[head_name], val_samples[head_name], args, device)
        rows.append(row)
        print(
            "{head}: AUROC={auroc:.4f}, pos_mean={pos_prob_mean:.4f}, "
            "neg_mean={neg_prob_mean:.4f}, gap={prob_gap:.4f}, "
            "val_pos_ratio={val_positive_ratio:.4f}, dim={feature_dim}, "
            "train_edges={train_edges}, val_edges={val_edges}".format(**row),
            flush=True,
        )

    if args.output_csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output_csv}", flush=True)

    if not rows:
        print("No connectivity head features were captured.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
