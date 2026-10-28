"""Pixel and fixed-GT-skeleton gap comparisons for H0 paired inference."""
import cv2
import numpy as np

MODES = ('on', 'global_off', 'h_direct_off')


def pixel_changes(gt, skeleton, predictions):
    """Treat full output as the candidate and each intervention as reference."""
    rows = []
    full = predictions['on']
    for mode in MODES[1:]:
        off = predictions[mode]
        masks = {
            'repaired_fn': gt & full & ~off,
            'added_fn': gt & ~full & off,
            'removed_fp': ~gt & ~full & off,
            'added_fp': ~gt & full & ~off,
        }
        row = {'reference': mode,
               'upward_crossings': int((full & ~off).sum()),
               'downward_crossings': int((~full & off).sum())}
        for name, mask in masks.items():
            row[name] = int(mask.sum())
            row[name + '_gt_skeleton'] = int((mask & skeleton).sum())
        assert row['upward_crossings'] == row['repaired_fn'] + row['added_fp']
        assert row['downward_crossings'] == row['added_fn'] + row['removed_fp']
        rows.append(row)
    return rows


def _labels(mask):
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[1]


def _connected(labels, anchor_a, anchor_b):
    a = set(np.unique(labels[anchor_a]).tolist()) - {0}
    b = set(np.unique(labels[anchor_b]).tolist()) - {0}
    return bool(a & b)


def fixed_gap_audit(skeleton, predictions, gt, pad=32, min_gap_pixels=1, gt_tolerance=2):
    """Identify shared GT-skeleton missed runs with exactly two common anchors.

    Anchors lie on GT skeleton and are predicted road in all modes. Connectivity
    uses binary surface components, with and without a padded local crop. A
    connection may still be a false shortcut and is not proof of correct routing.
    """
    missing = skeleton & ~(predictions['on'] & predictions['global_off'] & predictions['h_direct_off'])
    common = skeleton & ~missing
    n, components, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
    full_labels = {mode: _labels(predictions[mode]) for mode in MODES}
    road_support = cv2.dilate(gt.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*gt_tolerance+1, 2*gt_tolerance+1))) > 0
    image_h, image_w = skeleton.shape
    kernel = np.ones((3, 3), np.uint8)
    rows = []
    excluded = {'too_short': 0, 'anchor_groups_not_two': 0}
    for i in range(1, n):
        x, y, w, h, area = (int(z) for z in stats[i])
        if area < min_gap_pixels:
            excluded['too_short'] += 1
            continue
        y0, y1 = max(0, y-1), min(image_h, y+h+1)
        x0, x1 = max(0, x-1), min(image_w, x+w+1)
        piece = components[y0:y1, x0:x1] == i
        adjacent = (cv2.dilate(piece.astype(np.uint8), kernel) > 0) & common[y0:y1, x0:x1]
        nanchors, anchor_labels = cv2.connectedComponents(adjacent.astype(np.uint8), connectivity=8)
        if nanchors != 3:
            excluded['anchor_groups_not_two'] += 1
            continue
        ay0, ay1 = max(0, y-pad-1), min(image_h, y+h+pad+1)
        ax0, ax1 = max(0, x-pad-1), min(image_w, x+w+pad+1)
        anchor_a = np.zeros((ay1-ay0, ax1-ax0), dtype=bool)
        anchor_b = np.zeros_like(anchor_a)
        anchor_a[y0-ay0:y1-ay0, x0-ax0:x1-ax0] = anchor_labels == 1
        anchor_b[y0-ay0:y1-ay0, x0-ax0:x1-ax0] = anchor_labels == 2
        row = {'gap_id': i, 'x': x, 'y': y, 'width': w, 'height': h,
               'gt_skeleton_pixels': area, 'anchor_a_pixels': int(anchor_a.sum()),
               'anchor_b_pixels': int(anchor_b.sum())}
        for mode in MODES:
            full_slice = full_labels[mode][ay0:ay1, ax0:ax1]
            local = _labels(predictions[mode][ay0:ay1, ax0:ax1])
            supported = _labels(predictions[mode][ay0:ay1, ax0:ax1] &
                                road_support[ay0:ay1, ax0:ax1])
            row[mode+'_global_connected'] = _connected(full_slice, anchor_a, anchor_b)
            row[mode+'_local_connected'] = _connected(local, anchor_a, anchor_b)
            row[mode+'_local_gt_supported_connected'] = _connected(supported, anchor_a, anchor_b)
        rows.append(row)
    return rows, {'all_missed_components': n-1, 'eligible': len(rows), **excluded,
                  'missed_union_pixels': int(missing.sum())}


def gap_changes(rows):
    out = {}
    for mode in MODES[1:]:
        out[mode] = {}
        for kind in ('local', 'global', 'local_gt_supported'):
            on = 'on_'+kind+'_connected'
            off = mode+'_'+kind+'_connected'
            out[mode][kind] = {
                'off_to_full_connected': sum(not r[off] and r[on] for r in rows),
                'off_to_full_disconnected': sum(r[off] and not r[on] for r in rows),
                'both_connected': sum(r[off] and r[on] for r in rows),
                'both_disconnected': sum(not r[off] and not r[on] for r in rows),
            }
    return out
