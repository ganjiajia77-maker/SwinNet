"""Train the original Swin-Unet architecture on Data1 road masks."""

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

from data1_common import (Data1RoadDataset, build_model, counts_from_logits,
                          load_swin_tiny_pretrain, segmentation_from_counts,
                          sliding_road_logits)


def original_ce_dice(logits, mask):
    """Original Swin-Unet's 0.4 CE + 0.6 squared-denominator Dice, two classes."""
    ce = F.cross_entropy(logits, mask)
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(mask, num_classes=2).permute(0, 3, 1, 2).float()
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = probs.square().sum(dim=(0, 2, 3)) + one_hot.square().sum(dim=(0, 2, 3))
    dice = (1 - (2 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return 0.4 * ce + 0.6 * dice


@torch.no_grad()
def validate(model, loader, device, threshold):
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        logits = sliding_road_logits(model, batch["image"][0], device)
        counts = counts_from_logits(logits, batch["mask"][0], threshold)
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
    return segmentation_from_counts(tp, fp, fn)


def _save(path, epoch, model, optimizer, best_iou, args, pretrain_info):
    payload = {"epoch": epoch, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "best_iou": best_iou,
               "args": vars(args), "pretrain_info": pretrain_info,
               "architecture": "original_swin_unet_swin_t_window8_512_two_class"}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--cfg", default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--pretrain_ckpt", default="")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--selection_threshold", type=float, default=0.5)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        parser.error("max_epochs, batch_size, and accumulation_steps must be positive")
    if args.val_interval < 1 or args.num_workers < 0 or args.base_lr <= 0:
        parser.error("val_interval and base_lr must be positive; num_workers must be nonnegative")
    if not 0 < args.selection_threshold < 1:
        parser.error("selection_threshold must be between 0 and 1")
    if not args.resume and not args.no_pretrain and not args.pretrain_ckpt:
        parser.error("Provide the original Swin-T V1 --pretrain_ckpt or explicitly use --no_pretrain")
    if args.no_pretrain and args.pretrain_ckpt:
        parser.error("Choose --pretrain_ckpt or --no_pretrain")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n",
                                               encoding="utf-8")
    train_set = Data1RoadDataset(args.root_path, "train", args.seed, augment=args.augment)
    val_set = Data1RoadDataset(args.root_path, "val")
    train_generator = torch.Generator().manual_seed(args.seed)
    # Workers are recreated on every epoch. They receive the updated dataset
    # epoch; persistent worker copies would otherwise repeat seeded crops.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              generator=train_generator, num_workers=args.num_workers,
                              persistent_workers=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=min(args.num_workers, 2), pin_memory=True)
    model = build_model(args.cfg)
    pretrain_info = None
    if not args.resume and args.pretrain_ckpt:
        pretrain_info = load_swin_tiny_pretrain(model, args.pretrain_ckpt)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
                                momentum=0.9, weight_decay=1e-4)
    start_epoch = 0
    best_iou = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint["best_iou"])
        pretrain_info = checkpoint.get("pretrain_info")
        print(f"Resumed at epoch {start_epoch + 1}; best IoU={best_iou:.5f}", flush=True)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    max_updates = max(1, args.max_epochs * updates_per_epoch)
    log_path = run_dir / "epochs.csv"
    if not log_path.exists():
        log_path.write_text("epoch,lr,train_loss,val_iou,val_f1,val_precision,val_recall\n",
                            encoding="utf-8")
    print(f"train={len(train_set)}, val={len(val_set)}, device={device}, "
          f"crop=one 512 per image/epoch, val=512/256 sliding, "
          f"persistent_workers=False, FP32, no EMA", flush=True)
    for epoch in range(start_epoch, args.max_epochs):
        train_set.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        learning_rate = args.base_lr
        for batch_index, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            loss = original_ce_dice(logits, masks)
            running_loss += float(loss.detach())
            group_start = (batch_index // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(train_loader) - group_start)
            (loss / group_size).backward()
            if (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                update = epoch * updates_per_epoch + batch_index // args.accumulation_steps
                learning_rate = args.base_lr * (1 - update / max_updates) ** 0.9
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = None
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.max_epochs:
            metrics = validate(model, val_loader, device, args.selection_threshold)
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                _save(run_dir / "best.pth", epoch, model, optimizer, best_iou, args, pretrain_info)
                print(f"[BEST] epoch={epoch + 1}, val IoU={best_iou:.5f}", flush=True)
        _save(run_dir / "last.pth", epoch, model, optimizer, best_iou, args, pretrain_info)
        row = [epoch + 1, learning_rate, running_loss / len(train_loader)]
        row += ([metrics[key] for key in ("iou", "f1", "precision", "recall")]
                if metrics else ["", "", "", ""])
        with log_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(row)
        print(f"epoch={epoch + 1}/{args.max_epochs} train_loss={row[2]:.5f} "
              f"val_iou={metrics['iou'] if metrics else float('nan'):.5f} lr={learning_rate:.6g}",
              flush=True)
    print(f"Best checkpoint: {run_dir / 'best.pth'}", flush=True)


if __name__ == "__main__":
    main()
