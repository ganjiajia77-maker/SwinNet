import argparse
import csv
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data1_road_dataset import Data1RoadDataset
from model import SAMRoad
from utils import load_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def cosine_warmup(epoch, total, base, floor, warmup):
    if epoch < warmup:
        return base * float(epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, total - warmup - 1)
    return floor + 0.5 * (base - floor) * (1.0 + np.cos(np.pi * min(1.0, t)))


def road_logits(model, images):
    x = images.permute(0, 3, 1, 2)
    x = (x - model.pixel_mean) / model.pixel_std
    embeddings = model.image_encoder(x)
    logits = model.map_decoder(embeddings)
    return logits[:, 1:2]


def bce_dice(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    union = prob.sum(dims) + target.sum(dims)
    dice = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()
    return bce + 0.5 * dice


def update_ema(ema, model, decay):
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[key].copy_(value)


@torch.no_grad()
def evaluate(model, loader, device, tile=512, stride=256):
    model.eval()
    weight_1d = torch.linspace(-1.0, 1.0, steps=tile, device=device).abs()
    weight_1d = (1.0 - weight_1d).clamp_min(0.1)
    tile_weight = (weight_1d[:, None] * weight_1d[None, :]).view(1, 1, tile, tile)
    tp = fp = fn = 0
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        _, _, h, w = image.shape
        canvas = torch.zeros((1, 1, h, w), device=device)
        weights = torch.zeros_like(canvas)
        for top in range(0, h - tile + 1, stride):
            for left in range(0, w - tile + 1, stride):
                patch = image[:, top:top + tile, left:left + tile]
                logits = road_logits(model, patch)
                canvas[:, :, top:top + tile, left:left + tile] += logits * tile_weight
                weights[:, :, top:top + tile, left:left + tile] += tile_weight
        logits = canvas / weights.clamp_min(1.0)
        total_loss += bce_dice(logits, target).item()
        pred = torch.sigmoid(logits) >= 0.2
        tp += int((pred & (target > 0.5)).sum())
        fp += int((pred & (target <= 0.5)).sum())
        fn += int(((~pred) & (target > 0.5)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"loss": total_loss / max(1, len(loader)), "iou": iou,
            "f1": f1, "precision": precision, "recall": recall}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/data1_road_vitb_512.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sam_ckpt", default="")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", default="")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    config = load_config(args.config)
    config.SAM_CKPT_PATH = args.sam_ckpt or config.SAM_CKPT_PATH
    config.PATCH_SIZE = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Data1RoadDataset(args.data_root, "train", random_crop=True,
                                augment=True, seed=args.seed)
    val_ds = Data1RoadDataset(args.data_root, "val", random_crop=False,
                              augment=False, seed=args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            worker_init_fn=seed_worker)
    model = SAMRoad(config).to(device)

    loaded = {n for n, p in model.named_parameters()
              if p.requires_grad and n.startswith("image_encoder.")
              and n in model.matched_param_names}
    enc_params = [p for n, p in model.named_parameters() if n in loaded]
    new_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith("topo_net.")
                  and n not in loaded]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": 5e-5, "group_name": "encoder"},
        {"params": new_params, "lr": 2e-4, "group_name": "decoder"},
    ], weight_decay=1e-4)

    start_epoch = 0
    best_f1 = -1.0
    ema = copy.deepcopy(model.state_dict()) if not args.no_ema else None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("training_model_state_dict", ckpt["model_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_f1 = float(ckpt.get("val_f1", -1.0))
        if ema is not None and ckpt.get("ema_state_dict"):
            ema = ckpt["ema_state_dict"]

    loss_csv = os.path.join(args.output_dir, "epoch_losses.csv")
    if args.resume and os.path.isfile(loss_csv):
        csv_mode = "a"
        write_header = False
    else:
        csv_mode = "w"
        write_header = True
    loss_file = open(loss_csv, csv_mode, newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_file)
    if write_header:
        loss_writer.writerow(["epoch", "lr_encoder", "lr_decoder", "train_loss", "val_loss", "val_iou", "val_f1", "val_precision", "val_recall"])
        loss_file.flush()
    with open(os.path.join(args.output_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        lr_enc = cosine_warmup(epoch, args.epochs, 5e-5, 5e-6, 10)
        lr_dec = cosine_warmup(epoch, args.epochs, 2e-4, 1e-5, 10)
        optimizer.param_groups[0]["lr"] = lr_enc
        optimizer.param_groups[1]["lr"] = lr_dec
        running = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_dice(road_logits(model, images), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                update_ema(ema, model, args.ema_decay)
            running += loss.item()

        eval_model = model
        backup = None
        if ema is not None:
            backup = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema)
        metrics = evaluate(eval_model, val_loader, device)
        if backup is not None:
            model.load_state_dict(backup)
        print(f"epoch={epoch + 1}/{args.epochs} lr_enc={lr_enc:.3g} lr_dec={lr_dec:.3g} "
              f"train={running / max(1, len(train_loader)):.5f} "
              f"val_iou={metrics['iou']:.5f} val_f1={metrics['f1']:.5f}", flush=True)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": ema if ema is not None else model.state_dict(),
            "training_model_state_dict": model.state_dict(),
            "ema_state_dict": ema,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": metrics["f1"], "val_iou": metrics["iou"],
            "args": vars(args),
        }
        loss_writer.writerow([epoch + 1, lr_enc, lr_dec, running / max(1, len(train_loader)), metrics["loss"], metrics["iou"], metrics["f1"], metrics["precision"], metrics["recall"]])
        loss_file.flush()
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pth"))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pth"))
    loss_file.close()

if __name__ == "__main__":
    main()
