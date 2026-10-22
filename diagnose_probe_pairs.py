"""Pair existing native-size probe masks; no model inference or training."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from probe_experiment import ALL_NAMES, native_cases
from evaluation.road_comparison.road_metrics import skeletonize

PAIRS = [('E2', 'E3'), ('H_E2', 'H_E3'), ('E3', 'F2'),
         ('H_E3', 'F2'), ('F2', 'F3'), ('F3', 'surface_prehead'),
         ('surface_prehead', 'original_surface')]


def analyze(masks, target):
    skeleton = skeletonize(target)
    missing = skeleton & ~masks['original_surface']
    rows = []
    for a, b in PAIRS:
        row = {'from': a, 'to': b}
        for prefix, region in [('all_gt_skeleton', skeleton),
                               ('final_missed_gt_skeleton', missing),
                               ('background', ~target)]:
            row[prefix + '_total'] = int(region.sum())
            for label, condition in [('both_positive', masks[a] & masks[b]),
                                     ('from_only', masks[a] & ~masks[b]),
                                     ('to_only', ~masks[a] & masks[b]),
                                     ('both_negative', ~masks[a] & ~masks[b])]:
                row[prefix + '_' + label] = int((region & condition).sum())
        rows.append(row)
    names = [n for n in ALL_NAMES if n != 'original_surface']
    codes = np.zeros(target.shape, dtype=np.uint8)
    for bit, name in enumerate(names):
        codes |= masks[name].astype(np.uint8) << bit
    values, counts = np.unique(codes[missing], return_counts=True)
    return rows, Counter(dict(zip(map(int, values), map(int, counts)))), names


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    args = parser.parse_args()
    prediction_root = args.run_dir / 'test_predictions'
    selection = json.loads((args.run_dir / 'val_selection.json').read_text(encoding='utf-8'))
    files = {n: {p.name for p in (prediction_root / n / 'surface').glob('*_pred.png')}
             for n in ALL_NAMES}
    per_image, totals, patterns, expected = [], {}, Counter(), set()
    for case, _, target in native_cases(args.root_path, 'test'):
        filename = case + '_pred.png'
        if filename in expected:
            raise ValueError('Duplicate case: ' + case)
        expected.add(filename)
        masks = {}
        for name in ALL_NAMES:
            path = prediction_root / name / 'surface' / filename
            array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if array is None or array.shape != target.shape:
                raise ValueError('Missing or invalid native mask: ' + str(path))
            if not set(np.unique(array)).issubset({0, 255}):
                raise ValueError('Expected binary 0/255 mask: ' + str(path))
            masks[name] = array > 127
        rows, counts, names = analyze(masks, target)
        patterns.update(counts)
        for row in rows:
            per_image.append({'case': case, **row})
            key = (row['from'], row['to'])
            if key not in totals:
                totals[key] = {'from': key[0], 'to': key[1]}
            for field, value in row.items():
                if field not in ('from', 'to'):
                    totals[key][field] = totals[key].get(field, 0) + value
        print(f'Paired {len(expected)}: {case}', flush=True)
    if not expected:
        raise ValueError('Empty test split')
    for name in ALL_NAMES:
        if files[name] != expected:
            raise ValueError(f'Incomplete pairing {name}: missing={expected-files[name]}, extra={files[name]-expected}')
    output = args.run_dir / 'paired_stage_diagnosis'
    output.mkdir(exist_ok=True)
    save_csv(output / 'paired_transitions.csv', list(totals.values()))
    save_csv(output / 'per_image_transitions.csv', per_image)
    pattern_rows = []
    denominator = sum(patterns.values())
    for code, count in patterns.most_common():
        pattern_rows.append({**{n: int(bool(code & (1 << bit))) for bit, n in enumerate(names)},
                             'pixels': count, 'fraction_of_final_misses': count / denominator})
    if pattern_rows:
        save_csv(output / 'final_miss_patterns.csv', pattern_rows)
    summary = {'images': len(expected), 'final_missed_gt_skeleton_pixels': denominator,
               'all_seven_probes_negative_pixels': patterns[0],
               'thresholds': {n: selection['models'][n]['threshold'] for n in ALL_NAMES},
               'interpretation': 'Paired binary readouts at individual validation-selected thresholds. '
               'From-only means positive in the earlier readout and negative in the later readout; '
               'it does not prove irreversible feature loss. Thresholds, resolutions and probe convergence differ. '
               'Background counts are false-positive pixels, not false road connections. '
               'Probability and matched-FP analyses require probability exports.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('Output:', output, flush=True)


if __name__ == '__main__':
    main()
