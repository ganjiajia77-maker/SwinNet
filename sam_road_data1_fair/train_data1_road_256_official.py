import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data1_road_dataset import Data1RoadDataset
from model import SAMRoad
from train_data1_road import (
    road_logits,
    seed_everything,
    seed_worker,
)
from utils import load_config


def road_bce(logits, target):
    # Official SAM-Road uses BCEWithLogitsLoss when FOCAL_LOSS is False.
    return F.binary_cross_entropy_with_logits(logits, target)


def build_optimizer(model, config):
    encoder_params = []
    if not config.FREEZE_ENCODER and not config.ENCODER_LORA:
        encoder_params = [
            p for name, p in model.named_parameters()
            if name in model.matched_param_names and name.startswith("image_encoder.")
            and p.requires_grad
        ]
    decoder_params = [
        p for p in model.map_decoder.parameters() if p.requires_grad
    ]
    groups = []
    if encoder_params:
        groups.append({
            "params": encoder_params,
            "lr": float(config.BASE_LR) * float(config.ENCODER_LR_FACTOR),
            "group_name": "encoder",
        })
    groups.append({
        "params": decoder_params,
        "lr": float(config.BASE_LR),
        "group_name": "decoder",
    })
    optimizer = torch.optim.Adam(groups, lr=float(config.BASE_LR))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[9], gamma=0.1
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    tp = fp = fn = 0
    total_loss = 0.0
    for batch in tqdm(loader, total=len(loader), desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        logits = road_logits(model, images)
        total_loss += road_bce(logits, target).item()
        pred = torch.sigmoid(logits) >= threshold
        gt = target > 0.5
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "loss": total_loss / max(1, len(loader)),
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data1_road_vitb_256_official.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam_ckpt", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    config = load_config(args.config)
    if args.sam_ckpt:
        config.SAM_CKPT_PATH = args.sam_ckpt
    if args.epochs is not None:
        config.TRAIN_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.workers is not None:
        config.DATA_WORKER_NUM = args.workers
    config.PATCH_SIZE = int(config.RESIZE_INPUT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_size = int(config.SOURCE_SIZE)
    input_size = int(config.RESIZE_INPUT)
    train_ds = Data1RoadDataset(
        args.data_root, "train", source_size=source_size,
        resize_size=input_size,
        rotation_augment=bool(config.ROTATION_AUGMENT),
        augment=bool(config.COLOR_AUGMENT), seed=args.seed,
    )
    val_ds = Data1RoadDataset(
        args.data_root, "val", source_size=source_size,
        resize_size=input_size, seed=args.seed,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=int(config.BATCH_SIZE), shuffle=True,
        num_workers=int(config.DATA_WORKER_NUM), pin_memory=True,
        worker_init_fn=seed_worker, generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(config.INFER_BATCH_SIZE), shuffle=False,
        num_workers=int(config.DATA_WORKER_NUM), pin_memory=True,
        worker_init_fn=seed_worker,
    )
    model = SAMRoad(config).to(device)
    optimizer, scheduler = build_optimizer(model, config)
    grad_accum = max(1, int(args.grad_accum))

    start_epoch = 0
    best_f1 = -1.0
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        state = checkpoint.get(
            "training_model_state_dict",
            checkpoint.get("model_state_dict", checkpoint.get("state_dict")),
        )
        if state is None:
            raise KeyError(f"Checkpoint keys: {list(checkpoint)[:20]}")
        model.load_state_dict(state, strict=False)
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_f1 = float(checkpoint.get("val_f1", -1.0))

    config_used = {
        "script": "train_data1_road_256_official.py",
        "args": vars(args),
        "config": config.to_dict(),
        "device": str(device),
        "effective_batch_size": int(config.BATCH_SIZE) * grad_accum,
        "loss": "BCEWithLogitsLoss on road channel",
        "data_note": "1024x1024 source -> one 256x256 input",
        "toponet_note": "not optimized: data1 provides raster masks, not graph labels",
    }
    with open(os.path.join(args.output_dir, "config_used.json"), "w",
              encoding="utf-8") as file:
        json.dump(config_used, file, indent=2, ensure_ascii=False)

    loss_path = os.path.join(args.output_dir, "epoch_losses.csv")
    append = bool(args.resume and os.path.isfile(loss_path))
    loss_file = open(
        loss_path, "a" if append else "w", newline="", encoding="utf-8"
    )
    loss_writer = csv.writer(loss_file)
    if not append:
        loss_writer.writerow([
            "epoch", "lr_encoder", "lr_decoder", "train_loss",
            "val_loss", "val_iou", "val_f1", "val_precision", "val_recall",
        ])
        loss_file.flush()

    for epoch in range(start_epoch, int(config.TRAIN_EPOCHS)):
        train_ds.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        progress = tqdm(
            train_loader, total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.TRAIN_EPOCHS}",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            loss = road_bce(road_logits(model, images), target)
            (loss / grad_accum).backward()
            if (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{running / max(1, progress.n):.4f}",
                lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
            )

        metrics = evaluate(
            model, val_loader, device, float(config.VAL_THRESHOLD)
        )
        lr_encoder = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0
        lr_decoder = optimizer.param_groups[-1]["lr"]
        train_loss = running / max(1, len(train_loader))
        print(
            f"epoch={epoch + 1}/{config.TRAIN_EPOCHS} "
            f"lr_enc={lr_encoder:.3g} lr_dec={lr_decoder:.3g} "
            f"train={train_loss:.5f} val_loss={metrics['loss']:.5f} "
            f"val_iou={metrics['iou']:.5f} val_f1={metrics['f1']:.5f}",
            flush=True,
        )
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "training_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_f1": metrics["f1"],
            "val_iou": metrics["iou"],
            "args": vars(args),
            "config": config.to_dict(),
        }
        loss_writer.writerow([
            epoch + 1, lr_encoder, lr_decoder, train_loss,
            metrics["loss"], metrics["iou"], metrics["f1"],
            metrics["precision"], metrics["recall"],
        ])
        loss_file.flush()
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pth"))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(checkpoint, os.path.join(args.output_dir, "best.th"))
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pth"))
        scheduler.step()
    loss_file.close()


if __name__ == "__main__":
    main()
