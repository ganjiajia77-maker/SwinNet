"""Compare trained skeleton heads under training-consistent targets and probe frozen z_struct."""

import argparse
import csv
import gc
import os
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import (
    build_soft_stage_skeleton_target,
    build_stage_skeleton_target,
)
from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--root_path", required=True)
    p.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--source_patch_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--fixed_iou_threshold", type=float, default=None)
    p.add_argument("--probe_epochs", type=int, default=10)
    p.add_argument("--probe_batch_size", type=int, default=8)
    p.add_argument("--probe_lr", type=float, default=1e-2)
    p.add_argument("--probe_weight_decay", type=float, default=1e-4)
    p.add_argument("--probe_pos_weight_cap", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--n_class", type=int, default=2)
    p.add_argument("--dataset", default="ImageData")
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    p.add_argument("--zip", action="store_true")
    p.add_argument("--cache_mode", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--accumulation_steps", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--amp_opt_level", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--throughput", action="store_true")
    return p.parse_args()


def load_model(args, device):
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    saved = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_args = SimpleNamespace(**vars(args))
    for name in (
        "structure_profile", "bottleneck_type", "final_topology_eta_init",
        "final_gap_rho_init", "stage2_skeleton_gradient_ratio",
        "stage3_skeleton_gradient_ratio", "final_skeleton_gradient_ratio",
        "enable_highres_structure_stream", "highres_structure_channels",
        "highres_structure_fuse_stages", "highres_structure_fusion_mode",
        "enable_post_refine_structure_interaction", "enable_global_topology",
        "global_topology_max_nodes", "global_topology_heads",
        "global_topology_alpha_max",
    ):
        if isinstance(saved, dict) and name in saved:
            setattr(model_args, name, saved[name])
    if checkpoint.get("structure_profile"):
        model_args.structure_profile = checkpoint["structure_profile"]
    model_args.num_classes = 1

    config = get_config(model_args)
    model = SwinUnet(
        config=config,
        img_size=model_args.img_size,
        num_classes=1,
        return_skeleton=True,
        bottleneck_type=getattr(model_args, "bottleneck_type", "global_local"),
        final_topology_eta_init=getattr(model_args, "final_topology_eta_init", 0.0),
        final_gap_rho_init=getattr(model_args, "final_gap_rho_init", 0.0),
        structure_profile=getattr(model_args, "structure_profile", "stage23_boundary_0626"),
        stage2_skeleton_gradient_ratio=getattr(model_args, "stage2_skeleton_gradient_ratio", 0.5),
        stage3_skeleton_gradient_ratio=getattr(model_args, "stage3_skeleton_gradient_ratio", 0.5),
        final_skeleton_gradient_ratio=getattr(model_args, "final_skeleton_gradient_ratio", 0.0),
        enable_highres_structure_stream=getattr(model_args, "enable_highres_structure_stream", False),
        highres_structure_channels=getattr(model_args, "highres_structure_channels", 64),
        highres_structure_fuse_stages=getattr(model_args, "highres_structure_fuse_stages", "stage23"),
        highres_structure_fusion_mode=getattr(model_args, "highres_structure_fusion_mode", "stage23"),
        enable_post_refine_structure_interaction=getattr(
            model_args, "enable_post_refine_structure_interaction", False
        ),
        enable_global_topology=getattr(model_args, "enable_global_topology", False),
        global_topology_max_nodes=getattr(model_args, "global_topology_max_nodes", 32),
        global_topology_heads=getattr(model_args, "global_topology_heads", 4),
        global_topology_alpha_max=getattr(model_args, "global_topology_alpha_max", 0.05),
    ).to(device)
    adapt_connectivity_modules_for_checkpoint(model, state, "standard")
    load_topology_checkpoint_state(
        model,
        state,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=(getattr(model_args, "bottleneck_type", "global_local") == "global_local"),
    )
    return model.eval()


def collate_targets(batch, size, device):
    skeleton = batch["skeleton"].to(device=device, dtype=torch.float32)
    hard, soft = build_soft_stage_skeleton_target(skeleton, size)
    return hard, soft


def name_from_batch(batch, index):
    names = batch.get("image_name", [])
    if isinstance(names, (list, tuple)):
        value = names[index]
    else:
        value = names
    return os.path.splitext(os.path.basename(str(value)))[0]


def add_head_sample(store, name, logits, hard_target, train_soft_target=None):
    probability = torch.sigmoid(logits.detach().float()).cpu().numpy()
    hard = hard_target.detach().float().cpu().numpy()
    soft = (
        train_soft_target.detach().float().cpu().numpy()
        if train_soft_target is not None
        else hard
    )
    for b in range(probability.shape[0]):
        store.append({"name": name[b], "prob": probability[b, 0], "target": hard[b, 0], "soft": soft[b, 0]})


def best_threshold_metrics(records, fixed_threshold=None):
    probs = np.concatenate([r["prob"].reshape(-1) for r in records]).astype(np.float32)
    target = np.concatenate([r["target"].reshape(-1) for r in records]).astype(np.uint8)
    auprc = float(average_precision_score(target, probs)) if target.any() else float("nan")
    thresholds = (
        [float(fixed_threshold)]
        if fixed_threshold is not None
        else np.linspace(0.01, 0.99, 99).tolist()
    )
    best_iou = (-1.0, 0.5)
    best_f1 = (-1.0, 0.5)
    for threshold in thresholds:
        pred = probs >= threshold
        tp = np.logical_and(pred, target == 1).sum(dtype=np.int64)
        fp = np.logical_and(pred, target == 0).sum(dtype=np.int64)
        fn = np.logical_and(~pred, target == 1).sum(dtype=np.int64)
        iou = float(tp / max(int(tp + fp + fn), 1))
        precision = float(tp / max(int(tp + fp), 1))
        recall = float(tp / max(int(tp + fn), 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
        if iou > best_iou[0]:
            best_iou = (iou, float(threshold))
        if f1 > best_f1[0]:
            best_f1 = (f1, float(threshold))
    return {
        "auprc": auprc,
        "best_iou": best_iou[0],
        "best_iou_threshold": best_iou[1],
        "best_f1": best_f1[0],
        "best_f1_threshold": best_f1[1],
        "positive_fraction": float(target.mean()),
        "pixels": int(target.size),
    }


def make_panel(image_tensor, target, probability, threshold, output_path):
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
    target = cv2.resize(target.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    probability = cv2.resize(probability.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = probability >= threshold
    heat = cv2.applyColorMap(np.clip(probability * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    binary = np.repeat((pred.astype(np.uint8) * 255)[..., None], 3, axis=2)
    gt = np.repeat((target * 255)[..., None], 3, axis=2)
    panel = np.concatenate((image, gt, heat, binary), axis=1)
    cv2.imwrite(output_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


class PixelProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.readout = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        return self.readout(x)


def pooled_surface_target(mask, size):
    return F.adaptive_max_pool2d(mask.float(), size).clamp_(0.0, 1.0)


def train_probe(name, train_x, train_y, val_x, val_y, mean, std, args, device):
    channels = int(train_x.shape[1])
    probe = PixelProbe(channels).to(device)
    # Targets are stored as FP16 to keep the probe cache small. Summing in
    # FP16 over the full training set can overflow at 65504 and corrupt
    # pos_weight, so accumulate in FP64.
    positive = float(train_y.sum(dtype=torch.float64).item())
    total = float(train_y.numel())
    pos_weight = min((total - positive) / max(positive, 1.0), args.probe_pos_weight_cap)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay
    )
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=0,
    )
    for epoch in range(args.probe_epochs):
        probe.train()
        total_loss = 0.0
        for features, labels in loader:
            features = (features.to(device=device, dtype=torch.float32) - mean) / std
            labels = labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(probe(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch == 0 or epoch + 1 == args.probe_epochs:
            print(f"{name} probe epoch {epoch + 1}/{args.probe_epochs}: loss={total_loss / max(len(loader), 1):.5f}", flush=True)

    probe.eval()
    val_records = []
    with torch.no_grad():
        for start in range(0, val_x.shape[0], args.batch_size):
            features = (val_x[start:start + args.batch_size].to(device=device, dtype=torch.float32) - mean) / std
            logits = probe(features)
            target = val_y[start:start + args.batch_size]
            add_head_sample(val_records, [f"probe_{start + i}" for i in range(logits.shape[0])], logits, target)
    return probe, val_records


def dataset_for(args, split):
    return RoadSkeletonDataset(
        root_dir=args.root_path,
        split=split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        tile_size=None,
        augment=(split == "train"),
    )


def extract_split(model, dataset, args, device, collect_probes=False, split="val", cache_dir=None):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    swin = model.swin_unet
    head_records = {"highres": [], "stage2": [], "stage3": []}
    feature_store = skeleton_store = surface_store = None
    channel_sum = channel_sq_sum = None
    feature_count = 0
    write_offset = 0
    image_panels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"extract {split}"):
            record_offsets = {name: len(records) for name, records in head_records.items()}
            images = batch["image"].to(device, non_blocking=True)
            skeleton = batch["skeleton"].to(device=device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = model(images)
            structure_outputs = outputs[-1] if isinstance(outputs, tuple) else []
            highres_logits = getattr(swin, "last_highres_structure_skeleton", None)
            z_struct = getattr(swin, "last_highres_z_struct", None)
            if highres_logits is not None:
                hard, _ = build_soft_stage_skeleton_target(skeleton, highres_logits.shape[-2:])
                names = [name_from_batch(batch, i) for i in range(images.shape[0])]
                add_head_sample(head_records["highres"], names, highres_logits, hard)
            if collect_probes:
                if z_struct is None:
                    raise RuntimeError("This checkpoint did not produce z_struct.")
                if cache_dir is None:
                    raise ValueError("cache_dir is required when collecting z_struct probes")
                size = z_struct.shape[-2:]
                z_cpu = z_struct.detach().to(device="cpu", dtype=torch.float16)
                skeleton_target = build_stage_skeleton_target(skeleton, size).cpu().to(torch.float16)
                surface_target = pooled_surface_target(masks, size).cpu().to(torch.float16)
                if feature_store is None:
                    os.makedirs(cache_dir, exist_ok=True)
                    feature_store = np.lib.format.open_memmap(
                        os.path.join(cache_dir, f"{split}_features.npy"),
                        mode="w+",
                        dtype=np.float16,
                        shape=(len(dataset), *z_cpu.shape[1:]),
                    )
                    skeleton_store = np.lib.format.open_memmap(
                        os.path.join(cache_dir, f"{split}_skeleton_targets.npy"),
                        mode="w+",
                        dtype=np.float16,
                        shape=(len(dataset), *skeleton_target.shape[1:]),
                    )
                    surface_store = np.lib.format.open_memmap(
                        os.path.join(cache_dir, f"{split}_surface_targets.npy"),
                        mode="w+",
                        dtype=np.float16,
                        shape=(len(dataset), *surface_target.shape[1:]),
                    )
                write_end = write_offset + z_cpu.shape[0]
                feature_store[write_offset:write_end] = z_cpu.numpy()
                skeleton_store[write_offset:write_end] = skeleton_target.numpy()
                surface_store[write_offset:write_end] = surface_target.numpy()
                write_offset = write_end
                z_float = z_struct.detach().float()
                current_sum = z_float.sum(dim=(0, 2, 3)).cpu()
                current_sq = z_float.square().sum(dim=(0, 2, 3)).cpu()
                channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
                channel_sq_sum = current_sq if channel_sq_sum is None else channel_sq_sum + current_sq
                feature_count += int(z_float.shape[0] * z_float.shape[2] * z_float.shape[3])

            by_stage = {}
            for item in structure_outputs:
                stage = item.get("stage")
                if stage in (2, 3) and int(item.get("refinement_step", 1)) == 1:
                    by_stage[stage] = item.get("skeleton")
            for stage, name in ((2, "stage2"), (3, "stage3")):
                logits = by_stage.get(stage)
                if logits is None:
                    continue
                hard, soft = build_soft_stage_skeleton_target(skeleton, logits.shape[-2:])
                names = [name_from_batch(batch, i) for i in range(images.shape[0])]
                add_head_sample(head_records[name], names, logits, hard, soft)
            if split == "val":
                names = [name_from_batch(batch, i) for i in range(images.shape[0])]
                current_probabilities = {}
                for head, records in head_records.items():
                    for record in records[record_offsets[head]:]:
                        current_probabilities[record["name"]] = (head, record["prob"])
                for i, case_name in enumerate(names):
                    maps = {
                        head: probability
                        for record_name, (head, probability) in current_probabilities.items()
                        if record_name == case_name
                    }
                    image_panels.append((case_name, images[i].detach().cpu(), maps))
    if collect_probes and write_offset != len(dataset):
        raise RuntimeError(
            f"Collected {write_offset} {split} feature rows, expected {len(dataset)}."
        )
    for store in (feature_store, skeleton_store, surface_store):
        if store is not None:
            store.flush()
    return {
        "heads": head_records,
        # torch.from_numpy keeps these as disk-backed tensors instead of
        # materializing several GB of z_struct activations in system RAM.
        "features": torch.from_numpy(feature_store) if feature_store is not None else None,
        "skeleton_targets": torch.from_numpy(skeleton_store) if skeleton_store is not None else None,
        "surface_targets": torch.from_numpy(surface_store) if surface_store is not None else None,
        "feature_mean": channel_sum / max(feature_count, 1) if channel_sum is not None else None,
        "feature_std": (
            (channel_sq_sum / max(feature_count, 1) - (channel_sum / max(feature_count, 1)).square())
            .clamp_min(1e-6)
            .sqrt()
            if channel_sum is not None else None
        ),
        "panels": image_panels,
    }


def write_metrics(records_by_head, output_dir, fixed_threshold=None):
    rows = []
    selected_thresholds = {}
    for name, records in records_by_head.items():
        if not records:
            print(f"[WARN] No logits collected for {name}", flush=True)
            continue
        metrics = best_threshold_metrics(records, fixed_threshold)
        selected_thresholds[name] = metrics["best_iou_threshold"]
        row = {"source": name, "images": len(records), **metrics}
        rows.append(row)
        soft_means = [float(r["soft"].mean()) for r in records]
        row["training_target_mean"] = float(np.mean(soft_means))
        print(
            f"{name:8s} AUPRC={row['auprc']:.5f} bestIoU={row['best_iou']:.5f} "
            f"@{row['best_iou_threshold']:.2f} bestF1={row['best_f1']:.5f} "
            f"positive={row['positive_fraction']:.5f} train_target_mean={row['training_target_mean']:.5f}",
            flush=True,
        )
    path = os.path.join(output_dir, "skeleton_head_metrics.csv")
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return selected_thresholds


def save_head_panels(panels, records_by_head, thresholds, output_dir):
    record_index = {
        (head, record["name"]): record
        for head, records in records_by_head.items()
        for record in records
    }
    for head in ("highres", "stage2", "stage3"):
        head_dir = os.path.join(output_dir, "predictions", head)
        os.makedirs(head_dir, exist_ok=True)
        threshold = thresholds.get(head, 0.5)
        for case_name, image, maps in panels:
            if head not in maps or (head, case_name) not in record_index:
                continue
            target = record_index[(head, case_name)]["target"]
            make_panel(
                image,
                target,
                maps[head],
                threshold,
                os.path.join(head_dir, f"{case_name}_panel.png"),
            )
            prob = cv2.resize(maps[head], (image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(
                os.path.join(head_dir, f"{case_name}_prob.png"),
                np.clip(prob * 255.0, 0, 255).astype(np.uint8),
            )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    model.eval()
    cache_dir = os.path.join(args.output_dir, ".zstruct_probe_cache")
    print(f"Device: {device}; checkpoint: {args.model_path}", flush=True)

    val_result = extract_split(
        model, dataset_for(args, "val"), args, device,
        collect_probes=True, split="val", cache_dir=cache_dir,
    )
    thresholds = write_metrics(
        val_result["heads"], args.output_dir, fixed_threshold=args.fixed_iou_threshold
    )
    save_head_panels(val_result["panels"], val_result["heads"], thresholds, args.output_dir)

    train_result = extract_split(
        model, dataset_for(args, "train"), args, device,
        collect_probes=True, split="train", cache_dir=cache_dir,
    )
    train_x = train_result["features"]
    val_x = val_result["features"]
    mean = train_result["feature_mean"].view(1, -1, 1, 1).to(device=device, dtype=torch.float32)
    std = train_result["feature_std"].view(1, -1, 1, 1).to(device=device, dtype=torch.float32)
    probe_rows = []
    for name, train_y, val_y in (
        ("zstruct_skeleton_probe", train_result["skeleton_targets"], val_result["skeleton_targets"]),
        ("zstruct_road_probe", train_result["surface_targets"], val_result["surface_targets"]),
    ):
        probe, records = train_probe(
            name, train_x, train_y, val_x, val_y, mean, std, args, device
        )
        metrics = best_threshold_metrics(records, args.fixed_iou_threshold)
        row = {"source": name, "images": int(val_y.shape[0]), **metrics}
        probe_rows.append(row)
        print(
            f"{name:24s} AUPRC={row['auprc']:.5f} bestIoU={row['best_iou']:.5f} "
            f"@{row['best_iou_threshold']:.2f} bestF1={row['best_f1']:.5f} "
            f"positive={row['positive_fraction']:.5f}",
            flush=True,
        )
        del probe
    probe_path = os.path.join(args.output_dir, "zstruct_probe_metrics.csv")
    with open(probe_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(probe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(probe_rows)
    del train_x, val_x, train_result, val_result, train_y, val_y, mean, std
    gc.collect()
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"Saved full validation head metrics: {os.path.join(args.output_dir, 'skeleton_head_metrics.csv')}")
    print(f"Saved probe metrics: {probe_path}")
    print(f"Saved head prediction panels: {os.path.join(args.output_dir, 'predictions')}")


if __name__ == "__main__":
    main()
