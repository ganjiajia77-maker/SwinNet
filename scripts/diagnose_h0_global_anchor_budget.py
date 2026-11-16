"""Compare 32/64 Stage2 global anchors on the original fixed validation gaps."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnostics.global_anchor_gap_geometry import bridge_proxy, gap_geometry, local_status, summarize_gap
from h0_phase_training import file_digest
from scripts.diagnose_h0_global_anchor_gaps import capture_image, state_digest, write_csv, write_json
from scripts.evaluate_h0_encoder_phase import TOPOLOGY_PROTOCOL, infer, load_model, save_mask

BUDGETS = (32, 64)
HOME = Path('/home/gjj')
DEFAULT_GAPS = HOME/'SwinNet-roadbias/model_out/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json'
DEFAULT_TOOL = HOME/'unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py'


def nearest_road_segments(skeleton):
    """Assign pixels within 8 px of the GT skeleton a segment ID when unambiguous.

    A segment is an 8-connected skeleton component after removing junction pixels.
    An anchor whose nearest skeleton pixel is a junction receives ID zero.
    """
    if not skeleton.any():
        return np.zeros(skeleton.shape, dtype=np.int32)
    pixels = skeleton.astype(np.uint8)
    neighbors = cv2.filter2D(pixels, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    _, labels = cv2.connectedComponents((skeleton & (neighbors <= 3)).astype(np.uint8), connectivity=8)
    distances, indices = distance_transform_edt(~skeleton, return_indices=True)
    return np.where(distances <= 8, labels[tuple(indices)], 0)


def compare_nodes(old, new, near_road, segments):
    """Check the intervention and describe where the added FPS nodes land."""
    for key in ('top', 'left', 'candidate_coords', 'candidate_valid', 'candidate_scores'):
        if not np.array_equal(old[key], new[key]):
            raise RuntimeError(f'The two budgets changed the candidate pool or window: {key}')
    old_ids = old['selected_candidate_indices'][old['selected_valid']]
    new_ids = new['selected_candidate_indices'][new['selected_valid']]
    if not np.array_equal(old_ids, new_ids[:len(old_ids)]):
        raise RuntimeError('64-node FPS did not preserve the 32-node selection prefix')

    def global_coords(record):
        return record['selected_coords'][record['selected_valid']] + [record['top'], record['left']]

    prior = global_coords(old)
    added = global_coords(new)[len(old_ids):]
    old_cells = set(map(tuple, prior[near_road[prior[:, 0], prior[:, 1]]] // 32))
    added_road = added[near_road[added[:, 0], added[:, 1]]]
    new_cells = set(map(tuple, added_road // 32)) - old_cells
    seen_segments = set(segments[prior[:, 0], prior[:, 1]]) - {0}
    added_segments = set(segments[added[:, 0], added[:, 1]]) - {0}
    return dict(candidates=int(old['candidate_valid'].sum()), selected32=len(old_ids),
                selected64=len(new_ids), added_nodes=len(added),
                added_on_or_near_GT_road=len(added_road),
                newly_covered_road_cells_32px=len(new_cells),
                newly_covered_GT_segments=len(added_segments - seen_segments),
                added_coords_yx=added.tolist())


def measure(model, sample, device, budget):
    """Time one image after a full-image warm-up, without capture hooks."""
    model.swin_unet.global_topology.max_nodes = budget
    infer(model, sample, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    infer(model, sample, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return dict(nodes=budget, case=sample['case_name'], seconds=time.perf_counter()-start,
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20
                if device.type == 'cuda' else None)


def compare_gap(case, gap, geometry, records, masks):
    row = dict(case=case, gap_id=gap['id'], GT_sides_valid=geometry['valid'],
               reason=geometry['reason'], length=gap.get('length', len(gap['pixels'])),
               length_bin=gap.get('length_bin', 'unknown'))
    for budget in BUDGETS:
        status = local_status(masks[budget], gap)
        coverage, windows = summarize_gap(records[budget], geometry)
        row.update({f'{key}_{budget}': value for key, value in {**status, **coverage}.items()})
        row[f'windows_missing_one_correct_side_{budget}'] = sum(
            window['candidates_both_correct'] and
            (bool(window['correct_selected_side0']) != bool(window['correct_selected_side1']))
            for window in windows)
    return row


def summarize_gaps(rows, cohort):
    valid = [row for row in rows if row['GT_sides_valid']]
    blocked = [r for r in valid if r['candidates_both_correct_32'] and not r['selected_correct_pair_32']]
    return dict(cohort=cohort, gaps=len(rows), eligible_for_side_check=len(valid),
                candidates_cover_both=sum(r['candidates_both_correct_32'] for r in valid),
                eligible_but_no_pair_at_32=len(blocked),
                candidate_both_selected_one_side_at_32=sum(
                    r['windows_missing_one_correct_side_32'] > 0 for r in valid),
                of_blocked_recovered_pair_at_64=sum(r['selected_correct_pair_64'] for r in blocked),
                selected_pair_32=sum(r['selected_correct_pair_32'] for r in valid),
                selected_pair_64=sum(r['selected_correct_pair_64'] for r in valid),
                connected_32=sum(r['anchor_connected_local_32'] for r in rows),
                connected_64=sum(r['anchor_connected_local_64'] for r in rows),
                connections_gained=sum(r['anchor_connected_local_64'] and not r['anchor_connected_local_32'] for r in rows),
                connections_lost=sum(r['anchor_connected_local_32'] and not r['anchor_connected_local_64'] for r in rows),
                closures_gained=sum(r['closed_proxy_64'] and not r['closed_proxy_32'] for r in rows),
                closures_lost=sum(r['closed_proxy_32'] and not r['closed_proxy_64'] for r in rows))


def run(args):
    checkpoint = args.checkpoint.resolve()
    sha = file_digest(checkpoint)
    selection = json.loads((args.evaluation_dir/'val_selection.json').read_text(encoding='utf-8'))
    protocol = json.loads((args.evaluation_dir/'protocol.json').read_text(encoding='utf-8'))
    if (selection.get('selected_on') != 'val' or selection.get('criterion') != 'global_iou'
            or selection.get('checkpoint_sha256') != sha or protocol.get('checkpoint_sha256') != sha
            or protocol.get('global_topology_location') != 'after_decoder_stage2'
            or Path(protocol['data_root']).resolve() != args.root_path.resolve()
            or tuple(protocol.get(k) for k in ('image_size', 'window', 'stride', 'ramp_min')) != (1500, 512, 256, .1)):
        raise ValueError('Expected the original 32-node evaluation directory for this checkpoint and dataset')
    threshold = float(selection['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Invalid validation threshold')
    reference = json.loads(args.fixed_gaps.read_text(encoding='utf-8'))
    entries = reference['images']
    ids = [(case, gap['id']) for case, entry in entries.items() for gap in entry['gaps']]
    if (reference.get('count') != 198 or reference.get('p0_reference_threshold') != .45
            or len(entries) != 14 or len(ids) != 198 or len(set(ids)) != 198
            or any(entry['shape'] != [1500, 1500] for entry in entries.values())):
        raise ValueError('Expected the fixed198 validation gaps with native coordinates')

    device = torch.device(args.device)
    model, saved, epoch = load_model(checkpoint, device)
    core = model.swin_unet
    module = core.global_topology
    if (saved.get('no_s') or saved.get('h_road_loss_weight') or saved.get('img_size') != 512
            or saved.get('amp_dtype') != 'none' or not core.enable_global_topology or module.max_nodes != 32):
        raise ValueError('Expected the trained 32-node Stage2-global checkpoint with S and FP32')
    data = RoadSkeletonDataset(str(args.root_path), 'val', image_size=None, source_patch_size=1500,
                               augment=False, return_full_image=True)
    if {Path(name).stem.removesuffix('_sat') for name in data.image_files} != set(entries):
        raise ValueError('Validation images do not match fixed198')

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    masks_dir = {n: out/f'val_masks_{n}' for n in BUDGETS}
    for path in masks_dir.values():
        path.mkdir()
    write_json(out/'protocol.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha,
        checkpoint_epoch=epoch, threshold=threshold, source_eval=str(args.evaluation_dir.resolve()),
        budgets=BUDGETS, candidate_limit=256, fixed_gaps_sha256=file_digest(args.fixed_gaps),
        gap_side_tolerance_px=args.tolerance, near_road_dilation_radius_px=3,
        road_cell_size_px=32, segment_distance_px=8,
        segment_rule='nearest GT skeleton pixel after splitting segments at 8-neighbor junctions',
        topology=TOPOLOGY_PROTOCOL))
    model_hash = state_digest(model)
    gap_rows, windows_rows, image_rows = [], [], []
    try:
        write_csv(out/'timing.csv', [measure(model, data[0], device, n) for n in BUDGETS])
        for i in range(len(data)):
            sample = data[i]
            case = sample['case_name']
            gt = sample['mask'][0].numpy().astype(bool)
            records, masks, probabilities = {}, {}, {}
            for budget in BUDGETS:
                module.max_nodes = budget
                prob, records[budget], _ = capture_image(model, sample['image'], device, check_first=i == 0)
                probabilities[budget], masks[budget] = prob, prob >= threshold
                save_mask(masks_dir[budget]/f'{case}.png', masks[budget])
            cached = np.load(args.evaluation_dir/'val_probabilities'/f'{case}.npy', allow_pickle=False)
            if cached.shape != gt.shape or not np.isfinite(cached).all():
                raise ValueError(f'Invalid original 32-node prediction cache: {case}')
            cache_diff = float(np.max(np.abs(cached-probabilities[32])))
            near_road = cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            segments = nearest_road_segments(sample['skeleton'][0].numpy() > .5)
            if len(records[32]) != len(records[64]):
                raise RuntimeError('Window counts differ')
            for old, new in zip(records[32], records[64]):
                windows_rows.append(dict(case=case, top=old['top'], left=old['left'],
                                         **compare_nodes(old, new, near_road, segments)))
            for gap in entries[case]['gaps']:
                geometry = gap_geometry(gt, gap, args.tolerance)
                gap_rows.append(compare_gap(case, gap, geometry, records, masks))
            image_rows.append(dict(case=case, cache32_probability_max_abs=cache_diff,
                cache32_mask_diff_pixels=int(np.count_nonzero((cached >= threshold) != masks[32])),
                probability_mae=float(np.abs(probabilities[32]-probabilities[64]).mean()),
                changed_pixels=int(np.count_nonzero(masks[32] != masks[64])),
                bridge_proxy_added=bridge_proxy(masks[32], masks[64], gt),
                bridge_proxy_removed=bridge_proxy(masks[64], masks[32], gt)))
            print(f'[VAL {i+1}/{len(data)}] {case} cache32 max error={cache_diff:.3g}', flush=True)
    finally:
        module.max_nodes = 32
    if state_digest(model) != model_hash:
        raise RuntimeError('Model parameters or buffers changed during diagnosis')

    if len(gap_rows) != 198:
        raise RuntimeError('Expected all 198 fixed gaps in the output')
    write_csv(out/'fixed_gap_changes.csv', gap_rows)
    write_csv(out/'anchor_windows.csv', windows_rows)
    write_csv(out/'per_image.csv', image_rows)
    disconnected = [r for r in gap_rows if not r['anchor_connected_local_32']]
    write_csv(out/'gap_summary.csv', [summarize_gaps(gap_rows, 'all_fixed198'),
                                      summarize_gaps(disconnected, 'disconnected_at_32')])
    totals = {key: sum(row[key] for row in windows_rows)
              for key in ('selected32', 'selected64', 'added_nodes',
                          'added_on_or_near_GT_road', 'newly_covered_road_cells_32px', 'newly_covered_GT_segments')}
    totals.update({key: sum(row[key] for row in image_rows)
                   for key in ('changed_pixels', 'bridge_proxy_added', 'bridge_proxy_removed',
                               'cache32_mask_diff_pixels')})
    write_csv(out/'output_summary.csv', [totals])
    manifest = dict(data_root=str(args.root_path.resolve()), protocol=TOPOLOGY_PROTOCOL,
        models=[dict(name=f'anchors{n}', provenance=dict(checkpoint=str(checkpoint),
                     sha256=sha, max_nodes=n), predictions={
                         'val': [dict(path=str(masks_dir[n]), threshold=threshold)]})
                for n in BUDGETS])
    write_json(out/'manifest.json', manifest)
    print('[INFERENCE DONE]', out, flush=True)


def topology(args):
    out = args.output_dir.resolve()
    subprocess.run([sys.executable, '-u', str(args.topo_tool.resolve()),
        '--manifest', str(out/'manifest.json'), '--split', 'val',
        '--output_dir', str(out/'topology'), '--workers', str(args.workers)], check=True)


def show(out):
    for name, fields in (
        ('gap_summary.csv', None), ('output_summary.csv', None), ('timing.csv', None),
        ('topology/val_comparison.csv', ('name', 'threshold', 'iou', 'cldice', 'topo_precision',
          'topo_recall', 'topo_f1', 'apls_bidirectional', 'tlts_infeasible')),
    ):
        path = out/name
        print(f'\n{path}', flush=True)
        if not path.is_file():
            print('尚未生成')
            continue
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            fields = fields or tuple(rows[0])
            values = [fields, *(tuple(r[k] for k in fields) for r in rows)]
            widths = [max(map(len, col)) for col in zip(*values)]
            for row in values:
                print('  '.join(value.ljust(width) for value, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', choices=('all', 'infer', 'topology', 'show'), default='all')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluation_dir', type=Path, help='Existing evaluation for the same 32-node checkpoint')
    parser.add_argument('--root_path', type=Path, default=HOME/'Swin-Unet-main/Massachusetts/tiff')
    parser.add_argument('--fixed_gaps', type=Path, default=DEFAULT_GAPS)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tolerance', type=float, default=12.)
    parser.add_argument('--topo_tool', type=Path, default=DEFAULT_TOOL)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command in ('all', 'topology') and (not args.topo_tool.is_file() or args.workers < 1):
        parser.error('A valid --topo_tool file and positive --workers are required')
    if args.command in ('all', 'infer'):
        if args.checkpoint is None or args.evaluation_dir is None or not 0 < args.tolerance <= 64:
            parser.error('Inference needs --checkpoint, --evaluation_dir, and tolerance in (0, 64]')
        run(args)
    if args.command in ('all', 'topology'):
        topology(args)
    show(args.output_dir)
    if args.command != 'show':
        print('[ALL DONE]', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
