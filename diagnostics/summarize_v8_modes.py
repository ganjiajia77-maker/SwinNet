"""Summarize incremental v8 gains from paired val comparisons."""

import argparse
import json
from pathlib import Path


METRICS = ('iou', 'f1', 'precision', 'recall', 'cldice',
           'apls_gt_to_pred_approx', 'apls_pred_to_gt_approx',
           'apls_bidirectional_approx', 'short_pred_components')
COUNTS = ('closed_gaps', 'background_bridge_proxy',
          'new_gt_pixels_vs_p0', 'new_background_pixels_vs_p0')


def values(entry):
    return {**{name: entry['metrics'][name] for name in METRICS},
            **{name: entry['paired'][name] for name in COUNTS},
            'threshold': entry['threshold'],
            'foreground_area': entry['predicted_foreground_area']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ('local', 'bilateral', 'directional', 'full'):
        parser.add_argument(f'--{mode}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {mode: json.loads(getattr(args, mode).read_text(encoding='utf-8'))
               for mode in ('local', 'bilateral', 'directional', 'full')}
    if any(report['split'] != 'val' for report in reports.values()):
        raise ValueError('Only validation comparisons may select v8 architecture')
    if len({report['images'] for report in reports.values()}) != 1:
        raise ValueError('All modes must use the same validation images')
    reference = values(reports['local']['operating_points']['v7_reference'])
    for mode, report in reports.items():
        observed = values(report['operating_points']['v7_reference'])
        if any(abs(observed[key] - reference[key]) > 1e-8 for key in observed):
            raise ValueError(f'v7 reference differs in {mode}')
    lines = ['# v8 relation-stage comparison', '',
             f'Validation images: {reports["local"]["images"]}.', '',
             'Each row is a separately trained model initialized from identical v7 weights.',
             'Increments are between models, not a causal decomposition of one trained model.', '']
    points = ('v8_val_selected', 'v8_match_recall', 'v8_match_area')
    for point in points:
        lines += [f'## {point}', '',
                  '| Stage | IoU | clDice | APLS GT->Pred | APLS Pred->GT | '
                  'APLS bidir | Closed gaps | Bridge proxy | Short comps | Delta IoU | '
                  'Delta APLS bidir |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        previous = reference
        for mode in ('local', 'bilateral', 'directional', 'full'):
            item = values(reports[mode]['operating_points'][point])
            lines.append(f'| {mode} | {item["iou"]:.4f} | {item["cldice"]:.4f} | '
                         f'{item["apls_gt_to_pred_approx"]:.4f} | '
                         f'{item["apls_pred_to_gt_approx"]:.4f} | '
                         f'{item["apls_bidirectional_approx"]:.4f} | '
                         f'{item["closed_gaps"]} | {item["background_bridge_proxy"]} | '
                         f'{item["short_pred_components"]:.2f} | '
                         f'{item["iou"]-previous["iou"]:+.4f} | '
                         f'{item["apls_bidirectional_approx"]-previous["apls_bidirectional_approx"]:+.4f} |')
            previous = item
        lines.append('')
    lines.append('Matched Recall/area thresholds and all architecture choices come from val; '
                 'do not reselect them on test. APLS and bridge counts are raster proxies.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
