"""Train the original SwinT_OAM architecture on data1 random 512 crops."""

import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data1_common import Data1Train, image_names, load_case, metrics, pixel_counts, predict_full
from loss import dice_bce_loss
from networks.testNet import SwinT_OAM


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--pretrain_ckpt', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--overlap_stride', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--val_threshold', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--no_val_tta', action='store_true')
    args = parser.parse_args()
    if (args.source_patch_size, args.img_size) != (1024, 512):
        parser.error('This comparison requires 1024 source and 512 model input')
    if args.batch_size < 1 or args.num_workers < 0 or args.val_interval < 1:
        parser.error('Invalid batch size, worker count, or validation interval')
    if not args.resume and not os.path.isfile(args.pretrain_ckpt):
        parser.error('Official Swin-T ImageNet-22K checkpoint is required: --pretrain_ckpt')
    return args


@torch.no_grad()
def update_ema(ema, model, decay):
    source = model.state_dict()
    for name, value in ema.state_dict().items():
        if torch.is_floating_point(value):
            value.mul_(decay).add_(source[name].to(value.dtype), alpha=1.0 - decay)
        else:
            value.copy_(source[name])


@torch.no_grad()
def validate(model, args, device):
    model.eval()
    tp = fp = fn = 0
    names = image_names(args.root_path, 'val')
    for index, name in enumerate(names, 1):
        image, target = load_case(args.root_path, 'val', name, args.source_patch_size)
        probability = predict_full(model, image, device, args.img_size,
                                   args.overlap_stride, tta=not args.no_val_tta)
        a, b, c = pixel_counts(probability >= args.val_threshold, target)
        tp += a
        fp += b
        fn += c
        if index % 50 == 0 or index == len(names):
            print('Validation {}/{}'.format(index, len(names)), flush=True)
    return metrics(tp, fp, fn)


def save_checkpoint(path, epoch, model, ema, optimizer, best_iou, args):
    evaluation = ema if ema is not None else model
    torch.save({'epoch': epoch, 'model_state_dict': evaluation.state_dict(),
                'training_model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict() if ema is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_iou, 'eval_weights': 'ema' if ema is not None else 'raw',
                'config': vars(args)}, path)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; set CUDA_VISIBLE_DEVICES=0 on the server')
    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = os.path.join(args.output_dir, args.run_name)
    os.makedirs(out, exist_ok=True)
    dataset = Data1Train(args.root_path, args.seed, args.source_patch_size, args.img_size)
    print('Train images: {}; val images: {}; persistent_workers=False'.format(
        len(dataset), len(image_names(args.root_path, 'val'))), flush=True)
    model = SwinT_OAM(pretrained_backbone=None if args.resume else args.pretrain_ckpt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = copy.deepcopy(model).eval() if args.use_ema else None
    criterion = dice_bce_loss()
    first_epoch, best_iou = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['training_model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = int(checkpoint['epoch'])
        best_iou = float(checkpoint['best_val_iou'])
        if (checkpoint.get('ema_state_dict') is not None) != args.use_ema:
            raise ValueError('Resume --use_ema must match the saved run')
        if ema is not None:
            ema.load_state_dict(checkpoint['ema_state_dict'], strict=True)
        print('Resuming at epoch {}'.format(first_epoch + 1), flush=True)
    history_path = os.path.join(out, 'epoch_metrics.csv')
    write_header = not os.path.isfile(history_path)
    with open(history_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('epoch', 'train_loss', 'lr', 'val_iou',
                                                     'val_f1', 'val_precision', 'val_recall',
                                                     'elapsed_seconds'))
        if write_header:
            writer.writeheader()
        for epoch_index in range(first_epoch, args.max_epochs):
            started = time.monotonic()
            dataset.set_epoch(epoch_index)
            fixed_audit = dataset[0]
            print('CROP-AUDIT fixed_index=0 epoch={} top={} left={}'.format(
                epoch_index + 1, fixed_audit['crop_top'], fixed_audit['crop_left']),
                flush=True)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=False,
                                generator=torch.Generator().manual_seed(args.seed + epoch_index))
            learning_rate = args.lr / (5 ** sum(epoch_index >= step for step in (50, 65, 80)))
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            model.train()
            total_loss = 0.0
            for batch_index, batch in enumerate(loader):
                if batch_index == 0:
                    print('CROP-AUDIT epoch={} worker_epoch={} indices={} top={} left={}'.format(
                        epoch_index + 1, batch['crop_epoch'].tolist(),
                        batch['crop_index'].tolist(), batch['crop_top'].tolist(),
                        batch['crop_left'].tolist()), flush=True)
                image = batch['image'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(mask, model(image))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss at epoch {}, batch {}'.format(
                        epoch_index + 1, batch_index + 1))
                loss.backward()
                optimizer.step()
                if ema is not None:
                    update_ema(ema, model, args.ema_decay)
                total_loss += float(loss.detach())
            train_loss = total_loss / len(loader)
            values = {}
            if (epoch_index + 1) % args.val_interval == 0 or epoch_index + 1 == args.max_epochs:
                values = validate(ema if ema is not None else model, args, device)
                if values['iou'] > best_iou:
                    best_iou = values['iou']
                    save_checkpoint(os.path.join(out, 'best.pth'), epoch_index + 1,
                                    model, ema, optimizer, best_iou, args)
            save_checkpoint(os.path.join(out, 'last.pth'), epoch_index + 1,
                            model, ema, optimizer, best_iou, args)
            row = {'epoch': epoch_index + 1, 'train_loss': train_loss,
                   'lr': learning_rate, 'elapsed_seconds': round(time.monotonic() - started, 1)}
            row.update({'val_' + key: value for key, value in values.items()})
            writer.writerow(row)
            handle.flush()
            print('Epoch {}/{} loss={:.6f} lr={:.8g} val={} best_iou={:.6f}'.format(
                epoch_index + 1, args.max_epochs, train_loss, learning_rate,
                values if values else 'skipped', best_iou), flush=True)


if __name__ == '__main__':
    main()
