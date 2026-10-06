"""Evaluate OARENet on data1 test using a validation-selected threshold."""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from data1_common import image_names, load_case, load_model, metrics, pixel_counts, predict_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=('test',), default='test')
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap_stride', type=int, default=256)
    parser.add_argument('--no_tta', action='store_true')
    args = parser.parse_args()
    mask_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    names = image_names(args.root_path, 'test')
    rows = []
    for index, name in enumerate(names, 1):
        image, target = load_case(args.root_path, 'test', name, args.source_patch_size)
        probability = predict_full(model, image, device, args.tile_size,
                                   args.overlap_stride, tta=not args.no_tta)
        prediction = probability >= args.threshold
        Image.fromarray(prediction.astype(np.uint8) * 255).save(
            os.path.join(mask_dir, os.path.splitext(name)[0] + '.png'))
        tp, fp, fn = pixel_counts(prediction, target)
        rows.append({'image_id': name, 'tp': tp, 'fp': fp, 'fn': fn,
                     **metrics(tp, fp, fn)})
        print('[{}/{}] {}'.format(index, len(names), name), flush=True)
    tp, fp, fn = (sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn'))
    summary = {'threshold': args.threshold, 'tta': not args.no_tta,
               'images': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
               **metrics(tp, fp, fn)}
    with open(os.path.join(args.output_dir, 'test_metrics.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    with open(os.path.join(args.output_dir, 'test_metrics_per_image.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary)


if __name__ == '__main__':
    main()
