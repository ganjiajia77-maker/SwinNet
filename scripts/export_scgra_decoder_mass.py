"""Read-only SCGRA figure-data export for the original decoder-control Massachusetts run.

Same-checkpoint ON/OFF uses normal alpha versus alpha=0; no training, no TTA,
no test-threshold search. Only a small fixed subset gets spatial artifacts;
all selected images get three-method anchor statistics and resumable records.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'evaluation/road_comparison'))
from diagnostics import scgra_decoder_mass as evaluate
from diagnostics.scgra_export import METHODS, grid_to_pixels, paired_image, point_metrics, prepare_gt
from diagnostics.scgra_decoder_mass import file_digest, tensor_digests, select_validation_record
from road_metrics import skeletonize

VERSION = 'scgra-decoder-mass-readonly-v1'
METRIC_FIELDS = ('road_hit_rate_exact', 'road_hit_rate_tolerant',
    'nearest_neighbor_mean_px', 'nearest_neighbor_min_px', 'convex_hull_area_fraction',
    'component_coverage', 'skeleton_length_weighted_component_coverage',
    'anchor_events', 'unique_anchor_locations')


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_id(name):
    return Path(name).stem


def source_hashes():
    files = ('scripts/export_scgra_decoder_mass.py', 'diagnostics/scgra_export.py',
        'diagnostics/scgra_decoder_mass.py', 'h0_location.py', 'mass_protocol.py',
        'networks/h0_location_contract.py', 'BASE_MODEL_SHA256.json',
        'PROTOCOL_SOURCE_SHA256.json', 'config.py',
        'configs/swin_tiny_patch4_window7_224_lite.yaml',
        'datasets/dataset_road_skeleton.py', 'evaluation/road_comparison/road_metrics.py')
    files = sorted(set(files) | {str(p.relative_to(ROOT)).replace('\\', '/')
                                for p in (ROOT/'networks').glob('*.py')})
    return {name: file_digest(ROOT/name) for name in files}


def native_rgb(image):
    rgb = image.cpu().numpy().transpose(1, 2, 0)
    return np.rint(np.clip(rgb*np.array([.229, .224, .225])+np.array([.485, .456, .406]), 0, 1)*255).astype(np.uint8)


def save_png(path, value):
    temporary = path.with_suffix('.tmp.png')
    if not cv2.imwrite(str(temporary), value):
        raise IOError(str(path))
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def effect_metrics(on, off, gt, threshold):
    pred_on = on >= threshold
    tp, fp, fn = evaluate.counts(pred_on, gt)
    result = dict(paired=off is not None, threshold=threshold, on_tp=tp, on_fp=fp, on_fn=fn,
                  **{'on_'+k: v for k, v in evaluate.metrics(tp, fp, fn).items()})
    if off is None:
        return result
    pred_off = off >= threshold
    a, b, c = evaluate.counts(pred_off, gt)
    result.update(off_tp=a, off_fp=b, off_fn=c,
        **{'off_'+k: v for k, v in evaluate.metrics(a, b, c).items()},
        new_true_positive_pixels=int((pred_on & ~pred_off & gt).sum()),
        new_false_positive_pixels=int((pred_on & ~pred_off & ~gt).sum()),
        lost_true_positive_pixels=int((~pred_on & pred_off & gt).sum()),
        removed_false_positive_pixels=int((~pred_on & pred_off & ~gt).sum()),
        mean_gt_road_probability_delta=float((on-off)[gt].mean()) if gt.any() else None,
        mean_gt_background_probability_delta=float((on-off)[~gt].mean()) if (~gt).any() else None,
        max_abs_probability_delta=float(np.abs(on-off).max()))
    return result


def process_image(args, model, data, index, threshold, exported, verify):
    case, image, gt = evaluate.read_sample(data, index)
    labels, lengths, distance, skeleton = prepare_gt(gt, skeletonize)
    paired = args.paired_all or exported
    points = {(method, repeat): [] for method in METHODS
              for repeat in (range(args.random_repeats) if method == 'random' else [0])}
    rows, windows = [], []
    case_dir = args.output_dir/'cases'/case
    if exported:
        (case_dir/'windows').mkdir(parents=True, exist_ok=True)

    def record_window(wi, top, left, tile, selection, trace, controls):
        grid_hw = trace['feature_hw']
        if grid_hw != (512, 512) or tile != 512:
            raise RuntimeError(f'Expected original decoder global 512x512 feature grid: {grid_hw}')
        actual, probability, random, probability_selection = controls
        window_gt = gt[top:top+tile, left:left+tile]
        window_labels = labels[top:top+tile, left:left+tile]
        window_skeleton = skeleton[top:top+tile, left:left+tile]
        window_lengths = np.bincount(window_labels[window_skeleton], minlength=len(lengths))
        window_distance = distance[top:top+tile, left:left+tile]
        present = np.unique(window_labels)
        present = present[present > 0]
        sequences = [('structure_fps', 0, actual), ('probability_fps', 0, probability)]
        sequences += [('random', r, xy) for r, xy in enumerate(random)]
        for method, repeat, xy in sequences:
            local = grid_to_pixels(xy, grid_hw, (tile, tile))
            native = grid_to_pixels(xy, grid_hw, (tile, tile), (top, left))
            points[(method, repeat)].append(native)
            values = point_metrics(local, window_gt, window_labels, window_lengths,
                window_distance, args.hit_tolerance_px, present)
            rows.append(dict(case_id=case, window_index=wi, top=top, left=left,
                method=method, repeat=repeat, **values))
        info = dict(window_index=wi, top=top, left=left, tile_size=tile, grid_hw=list(grid_hw),
            actual_anchor_count=len(actual),
            candidate_valid_count=int(selection['candidate_valid'][0].sum()),
            candidate_nominal_count=int(selection['candidate_valid'].shape[1]),
            alpha_global=float(trace['alpha_global']),
            effective_residual_rms_mean=float(trace['effective_residual_rms'].mean()),
            effective_residual_rms_max=float(trace['effective_residual_rms'].max()))
        windows.append(info)
        if exported:
            payload = {key: value[0].numpy().copy() for key, value in selection.items() if key != 'anchor_hw'}
            for key, value in trace.items():
                if torch.is_tensor(value):
                    array = value.numpy()
                    payload[key] = array[0] if array.ndim and array.shape[0] == 1 else array
            for key in ('road_prior', 'structure_score', 'anchor_score'):
                payload[key] = trace[key][0, 0].numpy()
            payload['node_attention_mean'] = trace['node_attention'][0].mean(0).numpy()
            payload['selected_native_yx'] = grid_to_pixels(selection['selected_coords'][0].numpy(),
                grid_hw, (tile, tile), (top, left))
            payload['candidate_native_yx'] = grid_to_pixels(selection['candidate_coords'][0].numpy(),
                grid_hw, (tile, tile), (top, left))
            payload['probability_only_selected_grid_yx'] = probability
            payload['probability_only_candidate_grid_yx'] = probability_selection['candidate_coords'][0].numpy()
            payload['random_selected_grid_yx'] = np.asarray(random, dtype=np.int64)
            payload['window_top_left'] = np.array([top, left], np.int64)
            payload['feature_hw'] = np.array(grid_hw, np.int64)
            save_npz(case_dir/'windows'/f'y{top:04d}_x{left:04d}.npz', **payload)

    on, off, num_windows, checked = paired_image(model, image, args.device, case,
        args.seed, args.random_repeats, record_window, paired=paired, verify_capture=verify)
    present = np.unique(labels)
    present = present[present > 0]
    image_rows = []
    for (method, repeat), blocks in points.items():
        coords = np.concatenate(blocks, axis=0)
        image_rows.append(dict(case_id=case, method=method, repeat=repeat,
            **point_metrics(coords, gt, labels, lengths, distance, args.hit_tolerance_px, present)))
    effect = dict(case_id=case, exported=exported, windows=num_windows,
        alpha_global=windows[0]['alpha_global'],
        effective_residual_rms_window_mean=float(np.mean([w['effective_residual_rms_mean'] for w in windows])),
        **effect_metrics(on, off, gt, threshold))
    artifacts = {}
    if exported:
        save_png(case_dir/'image.png', cv2.cvtColor(native_rgb(image), cv2.COLOR_RGB2BGR))
        save_png(case_dir/'gt.png', gt.astype(np.uint8)*255)
        save_png(case_dir/'prediction_on.png', (on >= threshold).astype(np.uint8)*255)
        save_png(case_dir/'prediction_off.png', (off >= threshold).astype(np.uint8)*255)
        save_npz(case_dir/'prediction_on_off.npz', probability_on=on, probability_off=off,
                 probability_delta=on-off, threshold=np.array(threshold, np.float64))
        # Numeric categories, not a rendered figure. Colouring is a later plotting step.
        po, pf = on >= threshold, off >= threshold
        change = np.zeros(gt.shape, np.uint8)
        change[po & ~pf & gt] = 1
        change[po & ~pf & ~gt] = 2
        change[~po & pf & gt] = 3
        change[~po & pf & ~gt] = 4
        save_png(case_dir/'change_categories.png', change)
        dump(case_dir/'window_info.json', dict(case_id=case, windows=windows,
            coordinate_order='y,x', road_prior_source='original final-feature surface preview; no new Stage2 head', coordinate_rule='offset + (grid_index+0.5)*512/512',
            change_categories={'0': 'unchanged', '1': 'new_true_positive', '2': 'new_false_positive',
                               '3': 'lost_true_positive', '4': 'removed_false_positive'}))
        artifacts = {str(path.relative_to(args.output_dir)): file_digest(path)
                     for path in case_dir.rglob('*') if path.is_file()}
    return dict(case_id=case, capture_verified=checked, image_rows=image_rows,
                window_rows=rows, effects=effect, artifacts=artifacts)


def aggregate(records):
    image_rows = [row for record in records for row in record['image_rows']]
    summary = []
    for method in METHODS:
        cases = sorted({r['case_id'] for r in image_rows if r['method'] == method})
        output = dict(method=method, images=len(cases))
        for field in METRIC_FIELDS:
            means, repeat_sds = [], []
            for case in cases:
                values = [r[field] for r in image_rows
                          if r['case_id'] == case and r['method'] == method and r[field] is not None]
                if values:
                    means.append(float(np.mean(values)))
                    repeat_sds.append(float(np.std(values)))
            output['mean_'+field] = float(np.mean(means)) if means else None
            output['sd_between_images_'+field] = float(np.std(means, ddof=1)) if len(means) > 1 else None
            output['valid_images_'+field] = len(means)
            if method == 'random':
                output['mean_repeat_sd_'+field] = float(np.mean(repeat_sds)) if repeat_sds else None
        summary.append(output)
    return image_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument('--checkpoint', type=Path)
    checkpoint_group.add_argument('--checkpoint_root', type=Path,
        help='Find exactly one original decoder-control run under this root, without reading other model weights')
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path, help='Identity-bound decoder-control validation selection; autodetected when unambiguous')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_images', type=int, default=0, help='0=all; otherwise first N sorted images, recorded explicitly')
    parser.add_argument('--export_count', type=int, default=3, help='Number of prediction-independent hash-selected image exports')
    parser.add_argument('--export_cases', nargs='*', help='Explicit spatial-export case IDs; must belong to the chosen cohort')
    parser.add_argument('--random_repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--hit_tolerance_px', type=float, default=2.)
    parser.add_argument('--paired_all', action='store_true', help='Run alpha=0 for all cohort images; otherwise only spatial-export images')
    args = parser.parse_args()
    if (args.max_images < 0 or args.export_count < 0 or args.random_repeats < 1 or args.seed < 0
            or not np.isfinite(args.hit_tolerance_px) or args.hit_tolerance_px < 0):
        parser.error('Invalid sampling/export settings')
    if args.checkpoint_root is not None:
        args.checkpoint = evaluate.discover_checkpoint(args.checkpoint_root)
        print('[CHECKPOINT]', args.checkpoint, flush=True)
    args.checkpoint, args.root_path, args.output_dir = (p.resolve() for p in
        (args.checkpoint, args.root_path, args.output_dir))
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable in the selected Python environment')
    checkpoint_sha = file_digest(args.checkpoint)
    selection_path, selection, threshold = select_validation_record(args.checkpoint, checkpoint_sha, args.selection)
    data = evaluate.dataset(args.root_path, args.split)
    indices = list(range(len(data)))[:args.max_images or None]
    cases = [case_id(data.image_files[i]) for i in indices]
    if not cases or len(cases) != len(set(cases)):
        parser.error('Empty cohort or duplicate image IDs')
    if args.export_cases is not None:
        export_cases = sorted(set(case_id(c) for c in args.export_cases))
        if not set(export_cases) <= set(cases):
            parser.error('--export_cases must be inside the evaluated cohort')
    else:
        export_cases = sorted(sorted(cases, key=lambda c: digest([args.seed, c]))[:args.export_count])
    meta = dict(version=VERSION, checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha,
        root_path=str(args.root_path), split=args.split, total_split_images=len(data), evaluated_case_ids=cases,
        export_case_ids=export_cases, threshold=threshold, threshold_selected_on='val', model_name=evaluate.NAME, base_code_commit='892e5d017c19bbf017ae28e7bccef03423f7329f',
        validation_selection=str(selection_path), validation_selection_sha256=file_digest(selection_path),
        weights='model_state_dict (raw, EMA disabled)', precision='FP32', TTA=False,
        native_size=1500, tile_size=512, stride=256, blend='triangular weights, each ramp min0.1; logits then sigmoid',
        global_location='original up_x4, after final patch expansion and before H3/surface fusion', off_intervention='same checkpoint; raw_alpha=0 only; restored after every window',
        paired_all=args.paired_all, paired_case_ids=cases if args.paired_all else export_cases,
        controls='same effective K in each window; probability-only uses same top-k/FPS; uniform random without replacement',
        random_repeats=args.random_repeats, seed=args.seed, random_seed_rule='sha256(seed|case|top|left|repeat), first8 little-endian',
        grid_to_image='window offset + (grid_index+0.5)*512/512; y,x; floor only for GT lookup',
        hit_tolerance_px=args.hit_tolerance_px, exact_hit_reported=True,
        component_protocol='8-neighbour binary GT mask; strict anchor hits; no small-component filtering; length weights use fixed Zhang-Suen GT skeleton',
        spatial_protocol='image-level exact duplicate coordinates removed; before/after counts reported; nearest-neighbour Euclidean pixels and convex-hull area fraction',
        aggregation='per-image mean; average random repeats within each image first; SD is image variation, not training uncertainty',
        source_sha256=source_hashes(), environment=dict(python=platform.python_version(), torch=torch.__version__,
            numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir/'metadata.json'
    if meta_path.is_file():
        if load(meta_path) != meta:
            raise ValueError('Existing output has a different protocol/checkpoint/source/cohort; choose a new output_dir')
    elif any(args.output_dir.iterdir()):
        raise ValueError('Output directory contains unrecognized files; choose a new output_dir')
    else:
        dump(meta_path, meta)
    partial = args.output_dir/'partial'
    partial.mkdir(exist_ok=True)
    model, saved, epoch = evaluate.load_model(args.checkpoint, torch.device(args.device))
    module = model.swin_unet.global_topology
    if not module.enable_global_topology or saved.get('stage2_structure_location') != 'decoder':
        raise ValueError('Expected active original decoder-control global writeback')
    before = digest(tensor_digests(model))
    records, verified = [], False
    started = time.perf_counter()
    for n, index in enumerate(indices, 1):
        case = cases[n-1]
        image_path = Path(data.image_dir)/data.image_files[index]
        gt_path = Path(data.mask_dir)/data.label_files[index]
        inputs = dict(image_sha256=file_digest(image_path), gt_sha256=file_digest(gt_path))
        record_path = partial/(case+'.json')
        record, cached = None, False
        if record_path.is_file():
            previous = load(record_path)
            if previous.get('inputs') != inputs:
                raise ValueError('Source image/GT changed: '+case)
            if all((args.output_dir/p).is_file() and file_digest(args.output_dir/p) == sha
                   for p, sha in previous['artifacts'].items()):
                record, cached = previous, True
        if record is None:
            record = process_image(args, model, data, index, threshold, case in export_cases, not verified)
            record['inputs'] = inputs
            dump(record_path, record)
        verified = verified or record['capture_verified']
        records.append(record)
        elapsed = time.perf_counter()-started
        print(f'[SCGRA] {n}/{len(cases)} {case}; cached={cached}; paired={record["effects"]["paired"]}; '
              f'elapsed={elapsed:.1f}s', flush=True)
    if not verified:
        raise RuntimeError('No exact capture-vs-normal forward verification was completed')
    after = digest(tensor_digests(model))
    if before != after or file_digest(args.checkpoint) != checkpoint_sha:
        raise RuntimeError('Diagnostic modified model state or checkpoint')
    image_rows, summary = aggregate(records)
    write_csv(args.output_dir/'anchor_statistics_per_image.csv', image_rows)
    write_csv(args.output_dir/'anchor_statistics_per_window.csv', [r for item in records for r in item['window_rows']])
    write_csv(args.output_dir/'anchor_summary.csv', summary)
    write_csv(args.output_dir/'writeback_effects_per_image.csv', [r['effects'] for r in records])
    dump(args.output_dir/'summary.json', dict(version=VERSION, split=args.split, images=len(cases),
        checkpoint_epoch=epoch, threshold=threshold, export_cases=export_cases,
        paired_images=sum(r['effects']['paired'] for r in records),
        trained_alpha_global=float(module.alpha_global.detach()), anchor_summary=summary,
        capture_does_not_change_forward=True, model_state_unchanged=True, checkpoint_unchanged=True,
        model_state_sha256=after, elapsed_seconds=time.perf_counter()-started,
        interpretation='Anchor placement/attention/residual magnitude do not alone prove gap repair or module benefit. '
          'Probability-only and random groups are selection diagnostics, not retrained model ablations.'))
    dump(args.output_dir/'export_complete.json', dict(complete=True, metadata_sha256=file_digest(meta_path), images=len(cases)))
    print('[DONE]', args.output_dir, '\nSpatial cases:', ', '.join(export_cases), flush=True)


if __name__ == '__main__':
    main()
