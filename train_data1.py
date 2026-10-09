"""SegRoadv2 data1 comparison: FP32, one random512 crop/image/epoch, optional EMA."""

import argparse
import copy
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data1_common import (Data1Train, UPSTREAM_COMMIT, binary_prediction, create_model,
                          load_case, metrics, pixel_counts, predict_full, source_loss, split_index)
from nets.segformer_training import get_lr_scheduler


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--pretrain_ckpt')
    parser.add_argument('--phi', choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'], default='b2')
    parser.add_argument('--source_patch_size', type=int, choices=[1024], default=1024)
    parser.add_argument('--img_size', type=int, choices=[512], default=512)
    parser.add_argument('--overlap_stride', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--val_threshold', type=float, default=0.5)
    parser.add_argument('--prediction_mode', choices=['source_fusion', 'surface'], default='source_fusion')
    parser.add_argument('--no_val_tta', action='store_true')
    parser.add_argument('--connectivity_targets', choices=['source_fixed', 'source_legacy'], default='source_fixed')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if not args.resume and (not args.pretrain_ckpt or not Path(args.pretrain_ckpt).is_file()):
        parser.error('--pretrain_ckpt must point to the matching MiT backbone checkpoint')
    if min(args.batch_size, args.max_epochs, args.val_interval) < 1 or args.num_workers < 0:
        parser.error('Invalid batch/epoch/worker count')
    if not 0 <= args.ema_decay < 1 or not 0 < args.min_lr <= args.lr:
        parser.error('Invalid EMA decay or learning rates')
    return args


@torch.no_grad()
def update_ema(ema, model, decay):
    source_state = model.state_dict()
    for name, value in ema.state_dict().items():
        source = source_state[name]
        if value.is_floating_point():
            value.mul_(decay).add_(source, alpha=1 - decay)
        else:
            value.copy_(source)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def save_checkpoint(path, epoch, model, ema, optimizer, best_iou, args, commit):
    state = {'epoch': epoch, 'model_state_dict': (ema if ema is not None else model).state_dict(),
             'training_model_state_dict': model.state_dict(),
             'ema_state_dict': ema.state_dict() if ema is not None else None,
             'optimizer_state_dict': optimizer.state_dict(), 'best_val_iou': best_iou,
             'eval_weights': 'ema' if ema is not None else 'raw', 'config': vars(args),
             'rng_state': rng_state(), 'training_code_commit': commit,
             'upstream_commit': UPSTREAM_COMMIT}
    temporary = str(path) + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(model, index, args, device):
    model.eval()
    totals = [0, 0, 0]
    for number, paths in enumerate(index, 1):
        image, target = load_case(paths, args.source_patch_size)
        components = predict_full(model, image, device, args.img_size, args.overlap_stride,
                                  tta=not args.no_val_tta)
        prediction = binary_prediction(components, args.val_threshold, args.prediction_mode)
        totals = [a + b for a, b in zip(totals, pixel_counts(prediction, target))]
        if number == 1 or number % 25 == 0 or number == len(index):
            print('Validation {}/{}'.format(number, len(index)), flush=True)
    return metrics(*totals)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this FP32 comparison')
    device = torch.device('cuda:0')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_index, val_index = split_index(args.root_path, 'train'), split_index(args.root_path, 'val')
    print('Train={} val={} precision=FP32 persistent_workers=False'.format(
        len(train_index), len(val_index)), flush=True)
    model = create_model(args.phi, None if args.resume else args.pretrain_ckpt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    ema = copy.deepcopy(model).eval() if args.use_ema else None
    if ema is not None:
        ema.requires_grad_(False)
    first_epoch, best_iou = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        keys = ('phi', 'source_patch_size', 'img_size', 'use_ema', 'ema_decay', 'max_epochs',
                'lr', 'min_lr', 'weight_decay', 'seed', 'connectivity_targets', 'batch_size',
                'val_threshold', 'prediction_mode', 'overlap_stride', 'no_val_tta')
        differences = {key: (checkpoint['config'].get(key), getattr(args, key)) for key in keys
                       if checkpoint['config'].get(key) != getattr(args, key)}
        if differences:
            raise ValueError('Resume settings differ from saved run: ' + str(differences))
        model.load_state_dict(checkpoint['training_model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if ema is not None:
            ema.load_state_dict(checkpoint['ema_state_dict'], strict=True)
        first_epoch, best_iou = int(checkpoint['epoch']), float(checkpoint['best_val_iou'])
        state = checkpoint['rng_state']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        del checkpoint
        print('Resume next epoch={}'.format(first_epoch + 1), flush=True)
    out = Path(args.output_dir) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / 'last.pth').exists():
        raise ValueError('Run already has a checkpoint; use --resume or a new run_name')
    (Path(args.output_dir) / 'latest_segroadv2_run.txt').write_text(args.run_name + '\n', encoding='utf-8')
    (out / 'training_config.json').write_text(json.dumps(vars(args), indent=2) + '\n', encoding='utf-8')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        version = Path(__file__).with_name('SOURCE_REVISION.txt')
        commit = version.read_text().strip() if version.exists() else 'archive'
    schedule = get_lr_scheduler('cos', args.lr, args.min_lr, args.max_epochs)
    weights = torch.tensor([1.0, 3.0], device=device)
    history = out / 'epoch_metrics.csv'
    fresh = not history.exists()
    with history.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_loss', 'seg_loss', 'con_near_loss',
                                                    'con_far_loss', 'lr', 'val_iou', 'val_f1',
                                                    'val_precision', 'val_recall', 'elapsed_seconds'])
        if fresh:
            writer.writeheader()
        for epoch in range(first_epoch, args.max_epochs):
            started = time.monotonic()
            dataset = Data1Train(args.root_path, args.seed, args.source_patch_size, args.img_size,
                                 epoch=epoch, index=train_index)
            audit = dataset[0]
            print('CROP-AUDIT fixed_index=0 epoch={} top={} left={}'.format(
                epoch + 1, audit['crop_top'], audit['crop_left']), flush=True)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, persistent_workers=False,
                                generator=torch.Generator().manual_seed(args.seed + epoch))
            lr = schedule(epoch)
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            losses = np.zeros(4, dtype=np.float64)
            examples = 0
            for number, batch in enumerate(loader, 1):
                if not (batch['crop_epoch'] == epoch).all():
                    raise RuntimeError('Worker epoch mismatch; random crop state is stale')
                if number == 1:
                    print('CROP-AUDIT epoch={} worker_epoch={} indices={} top={} left={}'.format(
                        epoch + 1, batch['crop_epoch'].tolist(), batch['index'].tolist(),
                        batch['crop_top'].tolist(), batch['crop_left'].tolist()), flush=True)
                image, mask = batch['image'].to(device), batch['mask'].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, terms = source_loss(model(image), mask, weights, args.connectivity_targets)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss epoch={} batch={} images={}'.format(
                        epoch + 1, number, batch['name']))
                loss.backward()
                optimizer.step()
                if ema is not None:
                    update_ema(ema, model, args.ema_decay)
                size = image.shape[0]
                losses += np.asarray([float(value.detach()) for value in (loss, *terms)]) * size
                examples += size
                if number == 1 or number % 100 == 0 or number == len(loader):
                    print('Epoch {}/{} batch {}/{} loss={:.6f} lr={:.8g}'.format(
                        epoch + 1, args.max_epochs, number, len(loader), float(loss.detach()), lr), flush=True)
            if examples != len(dataset):
                raise RuntimeError('Not every training image was used once')
            values = {}
            if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.max_epochs:
                values = validate(ema if ema is not None else model, val_index, args, device)
                if values['iou'] > best_iou:
                    best_iou = values['iou']
                    save_checkpoint(out / 'best.pth', epoch + 1, model, ema, optimizer, best_iou, args, commit)
            save_checkpoint(out / 'last.pth', epoch + 1, model, ema, optimizer, best_iou, args, commit)
            mean = losses / examples
            row = dict(zip(['train_loss', 'seg_loss', 'con_near_loss', 'con_far_loss'], mean))
            row.update(epoch=epoch + 1, lr=lr, elapsed_seconds=round(time.monotonic() - started, 1))
            row.update({'val_' + key: value for key, value in values.items()})
            writer.writerow(row)
            handle.flush()
            print('Epoch {} val={} best_iou={:.6f}; saved {}'.format(
                epoch + 1, values or 'skipped', best_iou, out / 'last.pth'), flush=True)


if __name__ == '__main__':
    main()
