"""Choose an OARENet surface threshold using data1 validation images only."""

import argparse
import csv
import os

import torch

from data1_common import image_names, load_case, load_model, metrics, pixel_counts, predict_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=('val',), default='val')
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap_stride', type=int, default=256)
    parser.add_argument('--thresholds', default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--no_tta', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    thresholds = [float(value) for value in args.thresholds.split(',')]
    counts = {value: [0, 0, 0] for value in thresholds}
    names = image_names(args.root_path, 'val')
    for index, name in enumerate(names, 1):
        image, target = load_case(args.root_path, 'val', name, args.source_patch_size)
        probability = predict_full(model, image, device, args.tile_size,
                                   args.overlap_stride, tta=not args.no_tta)
        for threshold in thresholds:
            result = pixel_counts(probability >= threshold, target)
            for position, value in enumerate(result):
                counts[threshold][position] += value
        print('[{}/{}] {}'.format(index, len(names), name), flush=True)
    rows = [{'threshold': value, 'tta': not args.no_tta,
             **metrics(*counts[value]), 'tp': counts[value][0],
             'fp': counts[value][1], 'fn': counts[value][2]} for value in thresholds]
    with open(os.path.join(args.output_dir, 'threshold_sweep_val.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row['iou'])
    with open(os.path.join(args.output_dir, 'best_threshold.txt'), 'w') as handle:
        handle.write('{:.6f}\n'.format(best['threshold']))
    print('Best VAL threshold: {:.2f}; IoU={:.6f} F1={:.6f} P={:.6f} R={:.6f}'.format(
        best['threshold'], best['iou'], best['f1'], best['precision'], best['recall']))


if __name__ == '__main__':
    main()
