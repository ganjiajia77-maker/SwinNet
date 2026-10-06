"""SAM-Road ViT-B road-head training on data1, using the official 512 recipe."""

import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data1_road_512_official_common import (
    confusion_counts, counts_to_metrics, sliding_probability,
)
from data1_road_dataset import Data1RoadDataset
from model import SAMRoad
from train_data1_road import road_logits, seed_everything, seed_worker
from train_data1_road_256_official import build_optimizer, road_bce
from utils import load_config


@torch.no_grad()
def evaluate(model, loader, device, config, use_amp):
    model.eval()
    counts = [0, 0, 0]
    loss_sum = 0.0
    for batch in tqdm(loader, total=len(loader), desc="Validation", leave=False):
        probability = sliding_probability(
            model, batch["image"], device,
            tile=int(config.EVAL_TILE), stride=int(config.EVAL_STRIDE),
            use_amp=use_amp,
        ).cpu()
        target = batch["mask"]
        loss_sum += F.binary_cross_entropy(probability, target).item()
        for i, value in enumerate(confusion_counts(
            probability, target, float(config.VAL_THRESHOLD)
        )):
            counts[i] += value
    return {"loss": loss_sum / max(len(loader), 1), **counts_to_metrics(counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data1_road_vitb_512_official.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sam_ckpt", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--precision", choices=["16", "32"], default="16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.sam_ckpt:
        config.SAM_CKPT_PATH = args.sam_ckpt
    if args.epochs is not None:
        config.TRAIN_EPOCHS = args.epochs
    if args.workers is not None:
        config.DATA_WORKER_NUM = args.workers
    if args.micro_batch_size < 1 or args.grad_accum < 1:
        parser.error("--micro_batch_size and --grad_accum must be positive")
    if args.micro_batch_size * args.grad_accum != int(config.BATCH_SIZE):
        parser.error("micro_batch_size * grad_accum must equal official BATCH_SIZE=16")
    if int(config.PATCH_SIZE) != 512 or int(config.SOURCE_SIZE) != 1024:
        parser.error("This pipeline requires PATCH_SIZE=512 and SOURCE_SIZE=1024")
    if not os.path.isfile(config.SAM_CKPT_PATH):
        parser.error(f"SAM checkpoint missing: {config.SAM_CKPT_PATH}")
    if os.path.exists(os.path.join(args.output_dir, "last.pth")) and not args.resume:
        parser.error("Output already contains last.pth; use --resume or a new output_dir")

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.precision == "16"
    train_ds = Data1RoadDataset(
        args.data_root, "train", crop_size=512, source_size=1024,
        random_crop=bool(config.RANDOM_CROP),
        rotation_augment=bool(config.ROTATION_AUGMENT), seed=args.seed,
    )
    val_ds = Data1RoadDataset(
        args.data_root, "val", source_size=1024, seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch_size, shuffle=True,
        num_workers=int(config.DATA_WORKER_NUM), pin_memory=True,
        worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=int(config.DATA_WORKER_NUM), pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = SAMRoad(config).to(device)
    optimizer, scheduler = build_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    best_f1 = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["training_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if use_amp and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_f1 = float(checkpoint["best_val_f1"])

    with open(os.path.join(args.output_dir, "config_used.json"), "w", encoding="utf-8") as file:
        json.dump({
            "args": vars(args), "config": config.to_dict(),
            "effective_batch_size": args.micro_batch_size * args.grad_accum,
            "precision": "16-mixed" if use_amp else "32",
            "train": "one 512 random crop per 1024 image per epoch; random 90-degree rotation",
            "evaluation": "512 window, stride 256, weighted-logit stitching to 1024",
            "supervision": "road BCE only; data1 has no graph labels for TopoNet",
            "source_config": "htcr/sam_road config/toponet_vitb_512_cityscale.yaml",
        }, file, indent=2, ensure_ascii=False)

    loss_path = os.path.join(args.output_dir, "epoch_losses.csv")
    append = bool(args.resume and os.path.isfile(loss_path))
    with open(loss_path, "a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not append:
            writer.writerow([
                "epoch", "lr_encoder", "lr_decoder", "train_loss", "val_loss",
                "val_iou", "val_f1", "val_precision", "val_recall",
            ])
            file.flush()

        for epoch in range(start_epoch, int(config.TRAIN_EPOCHS)):
            train_ds.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            lr_encoder = optimizer.param_groups[0]["lr"]
            lr_decoder = optimizer.param_groups[-1]["lr"]
            progress = tqdm(
                train_loader, total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{config.TRAIN_EPOCHS}", dynamic_ncols=True,
            )
            for batch_index, batch in enumerate(progress):
                images = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                group_start = (batch_index // args.grad_accum) * args.grad_accum
                group_samples = min(
                    int(config.BATCH_SIZE),
                    len(train_ds) - group_start * args.micro_batch_size,
                )
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    loss = road_bce(road_logits(model, images), target)
                scaler.scale(loss * images.shape[0] / group_samples).backward()
                if (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += loss.item() * images.shape[0]
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{loss_sum / min((batch_index + 1) * args.micro_batch_size, len(train_ds)):.4f}",
                    lr=f"{lr_decoder:.2e}",
                )

            metrics = evaluate(model, val_loader, device, config, use_amp)
            train_loss = loss_sum / len(train_ds)
            print(
                f"epoch={epoch + 1}/{config.TRAIN_EPOCHS} "
                f"lr_enc={lr_encoder:.3g} lr_dec={lr_decoder:.3g} "
                f"train={train_loss:.5f} val_iou={metrics['iou']:.5f} "
                f"val_f1={metrics['f1']:.5f}", flush=True,
            )
            improved = metrics["f1"] > best_f1
            if improved:
                best_f1 = metrics["f1"]
            scheduler.step()
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "training_model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "best_val_f1": best_f1,
                "val_f1": metrics["f1"],
                "val_iou": metrics["iou"],
                "args": vars(args),
                "config": config.to_dict(),
            }
            writer.writerow([
                epoch + 1, lr_encoder, lr_decoder, train_loss, metrics["loss"],
                metrics["iou"], metrics["f1"], metrics["precision"], metrics["recall"],
            ])
            file.flush()
            torch.save(checkpoint, os.path.join(args.output_dir, "last.pth"))
            if improved:
                torch.save(checkpoint, os.path.join(args.output_dir, "best.th"))
                torch.save(checkpoint, os.path.join(args.output_dir, "best.pth"))


if __name__ == "__main__":
    main()
