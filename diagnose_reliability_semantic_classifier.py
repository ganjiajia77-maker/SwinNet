import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_reliability_delta_quadrants import reliability_modules, reliability_off
from diagnose_structure_delta_quadrants import extract_structure_features


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect FP->TN and TP->FN transition pixels and test whether "
            "stage semantic features can classify correct FP removals versus "
            "damaged true roads with a tiny offline classifier."
        )
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples_per_class", type=int, default=20000)
    parser.add_argument("--classifier_epochs", type=int, default=120)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_npz", type=str, default="")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
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
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--final_topology_eta_init", type=float, default=0.0)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.0)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=("auto", "standard", "selective"))
    return parser.parse_args()


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_capture_feature_tensors(model, enabled=True):
    for _name, module in reliability_modules(model):
        if hasattr(module, "capture_feature_tensors"):
            module.capture_feature_tensors = bool(enabled)


def resize_like(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def sample_transition_features(feature_map, transition_mask, max_needed):
    if feature_map is None or max_needed <= 0:
        return None
    if feature_map.shape[-2:] != transition_mask.shape[-2:]:
        feature_map = resize_like(feature_map, transition_mask.float(), mode="bilinear")
    flat_mask = transition_mask.reshape(-1)
    indices = flat_mask.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    if indices.numel() > max_needed:
        perm = torch.randperm(indices.numel(), device=indices.device)[:max_needed]
        indices = indices[perm]
    channels = feature_map.shape[1]
    features = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
    return features.index_select(0, indices).detach().cpu()


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLPProbe(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_probe(name, model, x_train, y_train, x_test, y_test, epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = min(2048, max(128, x_train.shape[0]))
    for _epoch in range(epochs):
        perm = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            logits = model(x_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx].float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = model(x_test)
        prob = torch.sigmoid(logits)
        pred = prob >= 0.5
        acc = (pred == y_test.bool()).float().mean().item()
        auc = rank_auc(prob.detach().cpu().numpy(), y_test.detach().cpu().numpy())
    print(f"  {name:<10} AUROC={auc:.6f} accuracy={acc:.6f}")
    return auc, acc


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    model.eval()
    set_capture_feature_tensors(model, True)

    beta_snapshot = [
        (name, float(module.reliability_beta.detach().cpu()))
        for name, module in reliability_modules(model)
    ]
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
        pin_memory=torch.cuda.is_available() and args.num_workers > 0,
    )

    fp_to_tn_chunks = []
    tp_to_fn_chunks = []
    fp_to_tn_total = 0
    tp_to_fn_total = 0
    batches = 0
    images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Collect semantic transitions")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if (
                fp_to_tn_total >= args.max_samples_per_class
                and tp_to_fn_total >= args.max_samples_per_class
            ):
                break
            batches += 1
            images += int(batch["image"].shape[0])
            image = batch["image"].to(device)
            mask = (batch["mask"].to(device) > 0.5).float()
            skeleton = batch["skeleton"].to(device).float()

            with reliability_off(model):
                off_outputs = model(
                    image,
                    gt_skeleton=skeleton,
                    topology_alpha_scale=1.0,
                    teacher_forcing_ratio=0.0,
                )
            on_outputs = model(
                image,
                gt_skeleton=skeleton,
                topology_alpha_scale=1.0,
                teacher_forcing_ratio=0.0,
            )
            off_logits = off_outputs[0]
            on_logits = resize_like(on_outputs[0], off_logits, mode="bilinear")
            gt = resize_like(mask, off_logits, mode="nearest") > 0.5
            off_pred = torch.sigmoid(off_logits) >= args.threshold
            on_pred = torch.sigmoid(on_logits) >= args.threshold

            fp_off = (~gt) & off_pred
            tp_off = gt & off_pred
            fp_to_tn = fp_off & (~on_pred)
            tp_to_fn = tp_off & (~on_pred)

            stage_outputs = on_outputs[4] if isinstance(on_outputs, tuple) and len(on_outputs) > 4 else []
            feature_maps, _ = extract_structure_features(stage_outputs, off_logits)
            semantic_feature = feature_maps.get("semantic_feature")
            if semantic_feature is None:
                raise RuntimeError("No semantic_feature found. Did capture_feature_tensors get enabled?")

            need_fp = args.max_samples_per_class - fp_to_tn_total
            need_tp = args.max_samples_per_class - tp_to_fn_total
            fp_features = sample_transition_features(semantic_feature, fp_to_tn, need_fp)
            tp_features = sample_transition_features(semantic_feature, tp_to_fn, need_tp)
            if fp_features is not None:
                fp_to_tn_chunks.append(fp_features)
                fp_to_tn_total += int(fp_features.shape[0])
            if tp_features is not None:
                tp_to_fn_chunks.append(tp_features)
                tp_to_fn_total += int(tp_features.shape[0])

    print("\nRELIABILITY SEMANTIC TRANSITION CLASSIFIER")
    print(f"split={args.split} batches={batches} images={images} threshold={args.threshold:.3f}")
    print(f"model_path={args.model_path}")
    print("Reliability beta checkpoint values")
    for name, beta in beta_snapshot:
        print(f"  {name}: {beta:.8f}")
    print("")
    print(f"Collected FP_to_TN samples: {fp_to_tn_total}")
    print(f"Collected TP_to_FN samples: {tp_to_fn_total}")
    if fp_to_tn_total < 20 or tp_to_fn_total < 20:
        print("Not enough transition samples for a meaningful classifier.")
        return

    x_pos = torch.cat(fp_to_tn_chunks, dim=0).float()
    x_neg = torch.cat(tp_to_fn_chunks, dim=0).float()
    n = min(x_pos.shape[0], x_neg.shape[0])
    x_pos = x_pos[torch.randperm(x_pos.shape[0])[:n]]
    x_neg = x_neg[torch.randperm(x_neg.shape[0])[:n]]
    x = torch.cat([x_neg, x_pos], dim=0)
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)], dim=0)
    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            features=x.numpy(),
            labels=y.numpy(),
            label_0="TP_to_FN_damaged_true_road",
            label_1="FP_to_TN_correct_fp_removal",
            threshold=np.array([args.threshold], dtype=np.float32),
        )
        print(f"Saved semantic transition features: {args.save_npz}")

    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]

    train_n = max(2, int(0.7 * x.shape[0]))
    x_train = x[:train_n]
    y_train = y[:train_n]
    x_test = x[train_n:]
    y_test = y[train_n:]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    print("")
    print(
        "Classifier target: label=1 is FP_to_TN correct FP removal; "
        "label=0 is TP_to_FN damaged true road."
    )
    print(
        f"Balanced samples per class={n}, train={x_train.shape[0]}, "
        f"test={x_test.shape[0]}, feature_dim={x_train.shape[1]}"
    )
    train_probe(
        "Linear",
        LinearProbe(x_train.shape[1]),
        x_train,
        y_train,
        x_test,
        y_test,
        args.classifier_epochs,
        args.lr,
    )
    train_probe(
        "2-layer MLP",
        MLPProbe(x_train.shape[1], args.hidden_dim),
        x_train,
        y_train,
        x_test,
        y_test,
        args.classifier_epochs,
        args.lr,
    )


if __name__ == "__main__":
    main()
