"""Train original DARENet on Data1 with one random 512 crop per 1024 image."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data1_common import Data1RoadDataset, build_model, counts, metrics, sliding_logits


def bce_dice(logits, target):
    if logits.shape != target.shape:
        raise ValueError(f"DARENet logits {tuple(logits.shape)} != target {tuple(target.shape)}")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dice = 1 - 2 * (probability * target).sum() / (probability.sum() + target.sum())
    return bce + dice


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        logits = sliding_logits(model, batch["image"][0], device)
        a, b, c = counts(logits, batch["mask"][0, 0], 0.5)
        tp, fp, fn = tp + a, fp + b, fn + c
    return metrics(tp, fp, fn)


def save(path, epoch, model, optimizer, best_iou, args, model_file):
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(), "best_iou": best_iou,
                  "args": vars(args), "model_file": model_file,
                  "architecture": "upstream_darenet_lmz_512_binary"}
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_repo", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if (args.max_epochs < 1 or args.val_interval < 1 or args.batch_size < 1 or
            args.accumulation_steps < 1 or args.num_workers < 0 or args.lr <= 0 or
            not 0 <= args.min_lr <= args.lr or not 0 <= args.warmup_epochs < args.max_epochs):
        parser.error("Invalid epochs, batch, workers, or learning rate configuration")
    if not torch.cuda.is_available():
        parser.error("CUDA required for this server training entry")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=2 maps physical GPU2 here.
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n",
                                               encoding="utf-8")
    train_set = Data1RoadDataset(args.root_path, "train", args.seed, args.augment)
    val_set = Data1RoadDataset(args.root_path, "val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=args.num_workers,
                              persistent_workers=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=min(args.num_workers, 2), pin_memory=True)
    model, model_file = build_model(args.original_repo)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    start_epoch, best_iou = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != "upstream_darenet_lmz_512_binary":
            raise ValueError("Resume checkpoint is not the Data1 DARENet baseline")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint["best_iou"])
    log_path = run_dir / "epochs.csv"
    if not log_path.exists():
        log_path.write_text("epoch,lr,train_loss,val_iou,val_f1,val_precision,val_recall\n",
                            encoding="utf-8")
    print(f"DARENet model={model_file}; train={len(train_set)}, val={len(val_set)}; "
          "one 512 crop/image/epoch; persistent_workers=False; 512/256 val; FP32; no EMA",
          flush=True)
    for epoch in range(start_epoch, args.max_epochs):
        train_set.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if epoch < args.warmup_epochs:
            lr = args.min_lr + (args.lr - args.min_lr) * (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(args.max_epochs - args.warmup_epochs - 1, 1)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            logits = model(image)
            loss = bce_dice(logits, target)
            total_loss += float(loss.detach())
            group_start = (batch_index // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(train_loader) - group_start)
            (loss / group_size).backward()
            if (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        result = None
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.max_epochs:
            result = validate(model, val_loader, device)
            if result["iou"] > best_iou:
                best_iou = result["iou"]
                save(run_dir / "best.pth", epoch, model, optimizer, best_iou, args, model_file)
                print(f"[BEST] epoch={epoch + 1}, val IoU@0.5={best_iou:.5f}", flush=True)
        save(run_dir / "last.pth", epoch, model, optimizer, best_iou, args, model_file)
        row = [epoch + 1, lr, total_loss / len(train_loader)]
        row += ([result[key] for key in ("iou", "f1", "precision", "recall")]
                if result else ["", "", "", ""])
        with log_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(row)
        print(f"epoch={epoch + 1}/{args.max_epochs} loss={row[2]:.5f} "
              f"val_iou={result['iou'] if result else float('nan'):.5f} lr={lr:.6g}",
              flush=True)
    print(f"Best checkpoint: {run_dir / 'best.pth'}", flush=True)


if __name__ == "__main__":
    main()
