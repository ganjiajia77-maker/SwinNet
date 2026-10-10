"""Train WeavingUnet on one native 512 crop per Data1 image per epoch."""

import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data1 import FullImageData1, RandomCropData1
from inference import counts, metrics, predict_full
from networks.WeavingUnet import WeavingUnet


def criterion(probability, target):
    bce = F.binary_cross_entropy(probability, target)
    overlap = 2 * (probability * target).sum() + 1e-6
    union = probability.sum() + target.sum() + 1e-6
    return bce + 1 - overlap / union


@torch.inference_mode()
def validate(model, loader, device, tile, stride, threshold):
    model.eval()
    tp = fp = fn = 0
    for _, image, label in loader:
        probability = predict_full(model, image.to(device), tile, stride).cpu()
        a, b, c = counts(probability[0, 0], label[0], threshold)
        tp, fp, fn = tp + a, fp + b, fn + c
    return metrics(tp, fp, fn)


def save(path, model, optimizer, scheduler, epoch, best_iou, args):
    state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
             "best_val_iou": best_iou, "args": vars(args)}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--encoder_ckpt", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--val_threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1 or args.tile_size != 512:
        parser.error("Require positive epochs/batch/accumulation and tile_size=512")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} (logical cuda:0)", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = DataLoader(RandomCropData1(args.data_root, args.tile_size), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val = DataLoader(FullImageData1(args.data_root, "val"), batch_size=1,
                     num_workers=args.workers, pin_memory=device.type == "cuda")
    model = WeavingUnet(pretrained_encoder=args.resume is None and args.encoder_ckpt is None,
                        encoder_ckpt=args.encoder_ckpt if args.resume is None else None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch, best_iou = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best_iou = checkpoint["epoch"], checkpoint["best_val_iou"]
        print(f"Resuming after epoch {start_epoch}; best val IoU {best_iou:.6f}", flush=True)
    csv_path = args.output_dir / "epoch_metrics.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(["epoch", "lr", "train_loss", "val_iou", "val_f1", "val_precision", "val_recall"])
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (image, target) in enumerate(train):
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            probability = model(image)
            loss = criterion(probability, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}; training stopped")
            group_start = (batch_index // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(train) - group_start)
            (loss / group_size).backward()
            if (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        result = None
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            result = validate(model, val, device, args.tile_size, args.stride, args.val_threshold)
            if result["iou"] > best_iou:
                best_iou = result["iou"]
                save(args.output_dir / "best.pth", model, optimizer, scheduler, epoch, best_iou, args)
        save(args.output_dir / "last.pth", model, optimizer, scheduler, epoch, best_iou, args)
        row = [epoch, current_lr, total / len(train)] + ([result[k] for k in ("iou", "f1", "precision", "recall")] if result else [""] * 4)
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(row)
        print(f"epoch={epoch}/{args.epochs} loss={row[2]:.6f} val={result} best_iou={best_iou:.6f}", flush=True)


if __name__ == "__main__":
    main()
