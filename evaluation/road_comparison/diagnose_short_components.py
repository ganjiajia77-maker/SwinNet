"""Classify new short predicted regions using paired native-size masks and GT.

This is a diagnostic for saved surface masks. Its categories describe GT support;
they do not by themselves prove that a particular network module caused a change.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from compare_predictions import case_id, index_files, label_index, read_binary
from road_metrics import confusion, segmentation

FIELDS = (
    'case_id', 'component_id', 'image_size', 'x', 'y', 'width', 'height',
    'center_x', 'center_y', 'classification', 'origin', 'area', 'gt_pixels',
    'gt_fraction', 'within_2px_gt_fraction', 'old_overlap_pixels',
    'old_overlap_fraction', 'old_matched_component_area',
    'gt_matched_component_area',
)


def component_labels(mask):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def surface_dir(directory):
    directory = directory / 'surface' if (directory / 'surface').is_dir() else directory
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def classify_short_component(component, gt, gt_labels, gt_stats, near_gt,
                             old_labels, old_stats, short_area):
    area = int(component.sum())
    gt_hits = int((component & gt).sum())
    near_hits = int((component & near_gt).sum())
    old_ids, old_hits = np.unique(old_labels[component], return_counts=True)
    old_pairs = [(int(i), int(n)) for i, n in zip(old_ids, old_hits) if i != 0]
    old_id, old_hit = max(old_pairs, key=lambda item: item[1], default=(0, 0))
    old_area = int(old_stats[old_id, cv2.CC_STAT_AREA]) if old_id else 0
    gt_ids, gt_counts = np.unique(gt_labels[component & gt], return_counts=True)
    gt_id = int(gt_ids[np.argmax(gt_counts)]) if gt_ids.size else 0
    gt_area = int(gt_stats[gt_id, cv2.CC_STAT_AREA]) if gt_id else 0
    if gt_hits / area >= 0.5:
        label = 'gt_road_fragment' if gt_area >= short_area else 'gt_tiny_road'
    elif near_hits / area < 0.1:
        label = 'background_island'
    else:
        label = 'near_road_review'
    if old_hit == 0:
        origin = 'new_island'
    elif old_area >= short_area:
        origin = 'split_from_old_large_component'
    elif old_hit == area and old_area == area:
        origin = 'old_short_component'
    else:
        origin = 'changed_short_component'
    return {
        'classification': label, 'origin': origin, 'area': area,
        'gt_pixels': gt_hits, 'gt_fraction': gt_hits / area,
        'within_2px_gt_fraction': near_hits / area,
        'old_overlap_pixels': old_hit, 'old_overlap_fraction': old_hit / area,
        'old_matched_component_area': old_area,
        'gt_matched_component_area': gt_area,
    }


def render_example(record, data_root, old_paths, new_paths, gt_paths, output):
    case = record['case_id']
    old = read_binary(old_paths[case], record['image_size'])
    new = read_binary(new_paths[case], record['image_size'])
    gt = read_binary(gt_paths[case], record['image_size'], allow_replicated_rgb=True)
    _, labels, _, _ = component_labels(new)
    component = labels == int(record['component_id'])
    cy, cx = int(record['center_y']), int(record['center_x'])
    radius = 80
    top, bottom = max(cy - radius, 0), min(cy + radius, new.shape[0])
    left, right = max(cx - radius, 0), min(cx + radius, new.shape[1])
    image_dir = data_root / 'test' / 'image'
    if not image_dir.is_dir():
        image_dir = data_root / 'test'
    source = next((p for p in image_dir.iterdir() if p.is_file() and
                   case_id(p) == case), None) if image_dir.is_dir() else None
    image = cv2.imread(str(source), cv2.IMREAD_COLOR) if source else None
    if image is None or image.shape[:2] != new.shape:
        image = np.full((*new.shape, 3), 64, dtype=np.uint8)
    patch = image[top:bottom, left:right]
    panels = []
    for title, mask, color in (
        ('RGB', None, None), ('GT', gt, (0, 200, 0)),
        ('H0', old, (255, 180, 0)), ('NEW', new, (0, 0, 255)),
    ):
        panel = patch.copy()
        if mask is not None:
            selected = mask[top:bottom, left:right]
            panel[selected] = (panel[selected].astype(np.float32) * 0.35 +
                               np.asarray(color, np.float32) * 0.65).astype(np.uint8)
        border = component[top:bottom, left:right].astype(np.uint8)
        outline = cv2.morphologyEx(border, cv2.MORPH_GRADIENT,
                                   np.ones((3, 3), np.uint8)).astype(bool)
        panel[outline] = (0, 255, 255)
        cv2.putText(panel, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    mosaic = cv2.hconcat(panels)
    destination = output / 'examples' / (
        f"{record['classification']}_{case}_{int(record['component_id']):04d}.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), mosaic):
        raise OSError(f'Could not save {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--baseline_dir', type=Path, required=True,
                        help='Original H0 test surface mask directory')
    parser.add_argument('--candidate_dir', type=Path, required=True,
                        help='Encoder-enhanced H0 test surface mask directory')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--image_size', type=int, default=1500)
    parser.add_argument('--short_area', type=int, default=20)
    parser.add_argument('--example_count', type=int, default=24)
    args = parser.parse_args()
    if args.short_area < 2 or args.image_size < 2 or args.example_count < 0:
        parser.error('Invalid size or example count')
    gt_paths = label_index(args.data_root, 'test')
    old_paths = index_files(surface_dir(args.baseline_dir))
    new_paths = index_files(surface_dir(args.candidate_dir))
    if set(gt_paths) != set(old_paths) or set(gt_paths) != set(new_paths):
        parser.error('GT, original H0 and new H0 must contain exactly the same test cases')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    records, old_totals, new_totals = [], Counter(), Counter()
    old_short_total = new_short_total = 0
    new_pixel_tp = new_pixel_fp = removed_pixel_tp = removed_pixel_fp = 0
    for case in sorted(gt_paths):
        gt = read_binary(gt_paths[case], args.image_size, allow_replicated_rgb=True)
        old = read_binary(old_paths[case], args.image_size)
        new = read_binary(new_paths[case], args.image_size)
        old_totals.update(confusion(old, gt))
        new_totals.update(confusion(new, gt))
        added, removed = new & ~old, old & ~new
        new_pixel_tp += int((added & gt).sum())
        new_pixel_fp += int((added & ~gt).sum())
        removed_pixel_tp += int((removed & gt).sum())
        removed_pixel_fp += int((removed & ~gt).sum())
        _, old_labels, old_stats, _ = component_labels(old)
        new_count, new_labels, new_stats, centers = component_labels(new)
        _, gt_labels, gt_stats, _ = component_labels(gt)
        old_short_total += int((old_stats[1:, cv2.CC_STAT_AREA] < args.short_area).sum())
        new_short_total += int((new_stats[1:, cv2.CC_STAT_AREA] < args.short_area).sum())
        near_gt = cv2.dilate(gt.astype(np.uint8), kernel).astype(bool)
        for component_id in range(1, new_count):
            area = int(new_stats[component_id, cv2.CC_STAT_AREA])
            if area >= args.short_area:
                continue
            x, y, width, height, _ = map(int, new_stats[component_id])
            box = (slice(y, y + height), slice(x, x + width))
            component = new_labels[box] == component_id
            result = classify_short_component(
                component, gt[box], gt_labels[box], gt_stats,
                near_gt[box], old_labels[box], old_stats, args.short_area)
            record = {'case_id': case, 'component_id': component_id,
                      'image_size': args.image_size, 'x': x, 'y': y,
                      'width': width, 'height': height,
                      'center_x': round(float(centers[component_id, 0]), 1),
                      'center_y': round(float(centers[component_id, 1]), 1),
                      **result}
            records.append(record)
    with (args.output_dir / 'short_components.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    changed = [r for r in records if r['origin'] != 'old_short_component']
    categories = Counter(r['classification'] for r in changed)
    category_pixels = Counter()
    for record in changed:
        category_pixels[record['classification']] += record['area']
    before, after = segmentation(old_totals), segmentation(new_totals)
    summary = {
        'cases': len(gt_paths), 'native_size': args.image_size,
        'short_area_lt': args.short_area,
        'old_short_components': old_short_total,
        'new_short_components': new_short_total,
        'new_or_changed_short_components': len(changed),
        'new_or_changed_by_gt_support': dict(categories),
        'new_or_changed_pixels_by_gt_support': dict(category_pixels),
        'gt_support_rule': 'GT overlap >= 0.5; GT component area distinguishes road fragment from tiny road',
        'background_rule': 'fewer than 10% of component pixels within 2 px of GT road',
        'old_h0': before, 'encoder_h0': after,
        'delta': {key: after[key] - before[key] for key in before},
        'added_surface_pixels': {'true_road': new_pixel_tp, 'background': new_pixel_fp},
        'removed_surface_pixels': {'true_road': removed_pixel_tp,
                                   'background': removed_pixel_fp},
        'interpretation_limit': 'GT support and paired prediction changes are observable; module-specific causal benefit requires matched retraining ablations',
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    priority = {'gt_road_fragment': 0, 'background_island': 1,
                'gt_tiny_road': 2, 'near_road_review': 3}
    examples = []
    per_class = max(1, (args.example_count + len(priority) - 1) // len(priority))
    for label in priority:
        examples.extend(sorted(
            (r for r in changed if r['classification'] == label),
            key=lambda r: (-r['area'], r['case_id']))[:per_class])
    examples = examples[:args.example_count]
    (args.output_dir / 'examples').mkdir(parents=True, exist_ok=True)
    for record in examples:
        render_example(record, args.data_root, old_paths, new_paths,
                       gt_paths, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Rows: {args.output_dir / 'short_components.csv'}", flush=True)
    print(f"Examples: {args.output_dir / 'examples'}", flush=True)


if __name__ == '__main__':
    main()
