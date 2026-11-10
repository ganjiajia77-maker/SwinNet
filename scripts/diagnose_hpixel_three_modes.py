"""One frozen pixel-reassembly checkpoint, three interventions, one val threshold.

No loss, optimizer, test-set selection, retraining, or GT in model forward.
The external pinned six-metric tool scores exported native validation masks.
"""
import argparse
from collections import Counter
import csv
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from evaluate_native_h0_pair import digest, load_model
from evaluation.road_comparison.road_metrics import skeletonize
from hpixel_diagnostic_utils import (MODES, PixelProbe, difficult_background,
                                     distribution, local_closed, owner_mask, state_digest)
from mass_protocol import blend_weight, sliding_positions
from selector_transport_reference import bridge_proxy
from topology_direction_constants import CONNECTIVITY_DIRECTIONS


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_counts(pred, gt):
    return Counter(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()))


def metrics(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    return dict(iou=tp/max(tp+fp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1),
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1))


def cldice(pred, gt, gt_skeleton):
    skel = skeletonize(pred)
    precision = (skel & gt).sum()/max(skel.sum(), 1)
    recall = (gt_skeleton & pred).sum()/max(gt_skeleton.sum(), 1)
    return float(2*precision*recall/max(precision+recall, 1e-12))


def selected_threshold(args, checkpoint_sha):
    if args.threshold_selection:
        value = json.loads(args.threshold_selection.read_text())
        if value.get('selected_on') != 'val' or value.get('checkpoint_sha256') != checkpoint_sha:
            raise ValueError('Threshold evidence must be val-selected for this exact checkpoint')
        threshold = value['threshold']
        evidence = dict(kind='existing_val_selection', path=str(args.threshold_selection.resolve()),
                        sha256=digest(args.threshold_selection))
    else:
        threshold = args.threshold
        evidence = dict(kind='explicit_fixed_diagnostic_threshold')
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be strictly between zero and one')
    return float(threshold), evidence


def topology_manifest(root_path, output_dir, checkpoint, sha, threshold):
    return dict(data_root=str(root_path.resolve()), protocol=dict(
        image_size=1500, apls_max_control_nodes=150, apls_control_sampling_seed=1234,
        topo_seed_density_deg=.002, topo_step_deg=.0002, tlts_n_paths=200),
        models=[dict(name=mode, provenance=dict(checkpoint=str(checkpoint.resolve()), sha256=sha,
            intervention=mode, note='same frozen checkpoint; same fixed val threshold'),
            predictions=dict(val=[dict(path=str((output_dir/'masks'/mode).resolve()),
                                       threshold=threshold)])) for mode in MODES])


@torch.inference_mode()
def infer_three(model, sample, device, probe, window_stats, case, identity_check, threshold):
    """Full native logit stitching; ownership only applies to feature statistics."""
    gt = sample['mask'][0].numpy() > .5
    image = sample['image'][None].to(device)
    size = gt.shape[0]
    if gt.shape != (1500, 1500):
        raise ValueError('Expected original native1500 validation images')
    canvases = {name: torch.zeros((1, 1, size, size), device=device)
                for name in (*MODES, 'bias_only_tail')}
    divisor = torch.zeros_like(canvases['normal'])
    weight = blend_weight(512, device)
    maps = {mode: dict(weights=np.zeros((9, size, size), np.float32),
                       residual_abs=np.zeros((size, size), np.float32),
                       weighted_component_abs=np.zeros((size, size), np.float32),
                       residual_change_abs=np.zeros((size, size), np.float32))
            for mode in MODES[:2]}
    difficult = np.zeros((size, size), bool)
    ownership = np.zeros((size, size), np.uint8)
    upstream_checks = 0
    positions = sliding_positions(size, 512, 256)
    for top in positions:
        for left in positions:
            probe.reset()
            tile = image[:, :, top:top+512, left:left+512]
            logits = {}
            for mode in MODES:
                probe.set_mode(mode)
                outputs = model(tile)
                logits[mode] = outputs[0]
                if mode == 'normal' and identity_check is not None:
                    if not torch.equal(logits[mode], identity_check):
                        raise RuntimeError('Instrumentation changed the original normal prediction')
                    identity_check = None
                if mode == 'residual_off':
                    c3 = next(row['connectivity'] for row in outputs[4] if row.get('stage') == 3)
                    if not torch.equal(c3, probe.inputs[3]):
                        raise RuntimeError('C3 changed when disabling only the new residual')
            logits['bias_only_tail'] = probe.bias_only_tail(outputs[4])
            for mode, value in logits.items():
                if value.shape != (1, 1, 512, 512) or not torch.isfinite(value).all():
                    raise ValueError('Invalid native surface logits')
                canvases[mode][:, :, top:top+512, left:left+512] += value.float()*weight
            divisor[:, :, top:top+512, left:left+512] += weight
            own = owner_mask(top, left)
            region = np.s_[top:top+512, left:left+512]
            ownership[region] += own
            road = gt[region]
            difficult[region] |= difficult_background(road) & own
            for mode in MODES[:2]:
                record = probe.records[mode]
                for key in ('residual_abs', 'weighted_component_abs', 'residual_change_abs'):
                    maps[mode][key][region][own] = record[key][own]
                maps[mode]['weights'][:, top:top+512, left:left+512][:, own] = record['weights'][:, own]
                window_stats.append(dict(case=case, top=top, left=left, mode=mode,
                    weighted_component_spatial_std_mean=record['weighted_component_spatial_std_mean']))
            upstream_checks += probe.upstream_checks
            if top == positions[-1] and left == positions[-1]:
                last_tile = tile.detach().clone()
                last_normal = logits['normal'].detach().clone()
    if not np.all(ownership == 1):
        raise RuntimeError('Diagnostic tile ownership is not a disjoint image partition')
    stitched, native_masks = {}, {}
    for name, value in canvases.items():
        native = (value/divisor)[0, 0]
        stitched[name] = native.cpu().numpy()
        if name in MODES:
            # Match the original evaluator's device, sigmoid and >= convention.
            native_masks[name] = (native.sigmoid() >= threshold).cpu().numpy()
    return stitched, maps, difficult, upstream_checks, last_tile, last_normal, native_masks


def analyse_case(case, gt, logits, maps, difficult, entry, threshold, bias_abs_mean, native_masks=None):
    pred = native_masks if native_masks is not None else {
        name: torch.from_numpy(value).sigmoid().numpy() >= threshold
        for name, value in logits.items() if name in MODES}
    gap_mask = np.zeros_like(gt)
    for gap in entry['gaps']:
        gap_mask.ravel()[gap['pixels']] = True
    regions = dict(all=np.ones_like(gt), fixed_reference_gap=gap_mask & gt,
                   normal_road=gt & pred['normal'] & ~gap_mask,
                   road_road_negative_background=difficult)
    gt_skeleton = skeletonize(gt)
    comparison = []
    fixed = []
    normal_closed = {gap['id']: local_closed(pred['normal'], gap) for gap in entry['gaps']}
    for mode in MODES:
        current, baseline = pred[mode], pred['normal']
        counts = metric_counts(current, gt)
        statuses = {gap['id']: local_closed(current, gap) for gap in entry['gaps']}
        comparison.append(dict(case=case, mode=mode, threshold=threshold, **counts, **metrics(counts),
            cldice=cldice(current, gt, gt_skeleton), closed=sum(statuses.values()),
            newly_closed_vs_normal=sum(v and not normal_closed[k] for k, v in statuses.items()),
            lost_closed_vs_normal=sum(not v and normal_closed[k] for k, v in statuses.items()),
            new_bridge_proxy_vs_normal=bridge_proxy(baseline, current, gt),
            removed_bridge_proxy_vs_normal=bridge_proxy(current, baseline, gt),
            added_tp=int((current & ~baseline & gt).sum()), lost_tp=int((~current & baseline & gt).sum()),
            added_fp=int((current & ~baseline & ~gt).sum()), removed_fp=int((~current & baseline & ~gt).sum())))
        for gap in entry['gaps']:
            before, after = normal_closed[gap['id']], statuses[gap['id']]
            fixed.append(dict(case=case, gap_id=gap['id'], length=gap['length'],
                length_bin=gap['length_bin'], mode=mode, threshold=threshold,
                normal_closed=before, mode_closed=after, newly_closed=after and not before,
                lost_closed=before and not after))
    logit_rows, source_rows, residual_rows = [], [], []
    differences = {
        'normal_minus_residual_off': logits['normal']-logits['residual_off'],
        'c_prior_one_minus_normal': logits['c_prior_one']-logits['normal'],
        'residual_off_minus_normal': logits['residual_off']-logits['normal'],
        'bias_only_minus_residual_off': logits['bias_only_tail']-logits['residual_off'],
        'normal_minus_bias_only': logits['normal']-logits['bias_only_tail'],
    }
    sources = [('self', 0, 0)] + [(f'direction_{i}', dy, dx) for i, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS)]
    normal_w, uniform_w = (maps[m]['weights'] for m in MODES[:2])
    for region, mask in regions.items():
        count = int(mask.sum())
        for name, value in differences.items():
            logit_rows.append(dict(case=case, region=region, comparison=name, **distribution(value, mask)))
        normal_selected = normal_w[:, mask]
        uniform_selected = uniform_w[:, mask]
        all_l1 = float(np.abs(uniform_selected-normal_selected).sum(0).mean()) if count else None
        changed_sources = int((uniform_selected.argmax(0) != normal_selected.argmax(0)).sum()) if count else 0
        for i, (source, dy, dx) in enumerate(sources):
            absolute_change = np.abs(uniform_selected[i]-normal_selected[i])
            source_rows.append(dict(case=case, region=region, source=source, dy=dy, dx=dx, pixels=count,
                normal_mean=float(normal_selected[i].mean()) if count else None,
                c_prior_one_mean=float(uniform_selected[i].mean()) if count else None,
                abs_change_mean=float(absolute_change.mean()) if count else None,
                abs_change_p90=float(np.quantile(absolute_change, .9)) if count else None,
                abs_change_max=float(absolute_change.max()) if count else None,
                all_sources_l1_change_mean=all_l1, argmax_changed_pixels=changed_sources))
        for mode in MODES[:2]:
            residual_rows.append(dict(case=case, region=region, mode=mode, pixels=count,
                residual_abs_mean=float(maps[mode]['residual_abs'][mask].mean()) if count else None,
                weighted_component_abs_mean=float(maps[mode]['weighted_component_abs'][mask].mean()) if count else None,
                residual_change_from_normal_abs_mean=float(maps[mode]['residual_change_abs'][mask].mean()) if count else None,
                projection_bias_abs_mean=bias_abs_mean))
    return pred, comparison, fixed, logit_rows, source_rows, residual_rows


def summarize_comparison(rows):
    result = []
    totals = ('closed', 'newly_closed_vs_normal', 'lost_closed_vs_normal',
              'new_bridge_proxy_vs_normal', 'removed_bridge_proxy_vs_normal',
              'added_tp', 'lost_tp', 'added_fp', 'removed_fp')
    for mode in MODES:
        selected = [row for row in rows if row['mode'] == mode]
        counts = {key: sum(row[key] for row in selected) for key in ('tp', 'fp', 'fn')}
        result.append(dict(mode=mode, threshold=selected[0]['threshold'], images=len(selected),
            **counts, **metrics(counts), cldice=float(np.mean([r['cldice'] for r in selected])),
            **{key: sum(row[key] for row in selected) for key in totals}))
    return result


def run_topology(args):
    if not args.topo_tool.is_file():
        raise FileNotFoundError(args.topo_tool)
    command = [sys.executable, '-u', str(args.topo_tool), '--manifest',
            str(args.output_dir/'manifest.json'), '--split', 'val', '--output_dir',
            str(args.output_dir/'topology'), '--workers', str(args.workers),
            '--progress_every', '1']
    if (args.output_dir/'topology'/'run_state.json').is_file():
        command.append('--resume')
    print('[TOPOLOGY] Same fixed threshold, 14 val images, 3 modes; see topology.log', flush=True)
    with (args.output_dir/'topology.log').open('a') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--root_path', type=Path, default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    p.add_argument('--fixed_gaps', type=Path, required=True)
    t = p.add_mutually_exclusive_group(required=True)
    t.add_argument('--threshold_selection', type=Path)
    t.add_argument('--threshold', type=float)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--command', choices=['all', 'infer', 'topology'], default='all')
    p.add_argument('--topo_tool', type=Path, default=Path('/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py'))
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.command == 'topology':
        protocol = json.loads((a.output_dir/'protocol.json').read_text())
        if protocol['checkpoint_sha256'] != digest(a.checkpoint):
            raise ValueError('Existing predictions came from a different checkpoint')
        threshold, _ = selected_threshold(a, protocol['checkpoint_sha256'])
        if threshold != protocol['threshold']:
            raise ValueError('Existing predictions use a different threshold')
        run_topology(a)
    else:
        sha = digest(a.checkpoint)
        threshold, evidence = selected_threshold(a, sha)
        reference = json.loads(a.fixed_gaps.read_text())
        if reference.get('count') != 198 or len(reference.get('images', {})) != 14:
            raise ValueError('Expected the fixed 198-gap, 14-image reference')
        if sum(len(entry['gaps']) for entry in reference['images'].values()) != 198:
            raise ValueError('Actual fixed gap list count differs from the declared 198')
        if a.output_dir.exists():
            raise ValueError('Use a new diagnostic output directory')
        model, config = load_model(a.checkpoint, torch.device(a.device), True)
        model.requires_grad_(False)
        if not config.get('h_pixel_reassembly') or config.get('h_reassembly_prior', 'predicted') != 'predicted':
            raise ValueError('Expected a predicted-C pixel-reassembly checkpoint')
        if config.get('local_cs_mode', 'off') != 'off' or config.get('path_training_mode', 'off') != 'off':
            raise ValueError('This diagnostic excludes local-CS and path-supervised runs')
        data = RoadSkeletonDataset(str(a.root_path), 'val', image_size=None, source_patch_size=1500,
                                   augment=False, return_full_image=True)
        if len(data) != 14:
            raise ValueError('Expected 14 native validation images')
        a.output_dir.mkdir(parents=True)
        for mode in MODES:
            (a.output_dir/'masks'/mode).mkdir(parents=True)
        protocol = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha,
            threshold=threshold, threshold_evidence=evidence, split='val', modes=MODES,
            image_size=1500, window=512, stride=256, blend='original triangular weighted logits',
            no_TTA=True, fixed_gaps=str(a.fixed_gaps.resolve()), fixed_gaps_sha256=digest(a.fixed_gaps),
            gap_margin=16, gap_coverage=.8, topology_apls_control_cap=150,
            topology_seed_density=.002, topology_step=.0002,
            normal_road='GT road correctly predicted by normal mode, excluding fixed gap pixels',
            difficult_background='GT background in cells incident to valid GT-negative road/road C edges; proxy',
            feature_scope='disjoint native window ownership; not interpolated or blended source weights',
            bias_scope='bias-only final-head replay; nonlinear conditional decomposition, not a fourth full forward',
            limitations=['Fixed-gap closure and false-bridge counts are proxies, not manual confirmation.',
                        'Checkpoint interventions do not replace retrained ablations.',
                        'Local feature changes alone do not demonstrate whole-road connectivity.'])
        dump(a.output_dir/'protocol.json', protocol)
        dump(a.output_dir/'manifest.json', topology_manifest(a.root_path, a.output_dir, a.checkpoint, sha, threshold))
        before = state_digest(model)
        bias = model.swin_unet.h_pixel_reassembly.out.bias.detach().cpu().numpy()
        write_csv(a.output_dir/'projection_bias.csv', [dict(channel=i, bias=float(v)) for i, v in enumerate(bias)])
        all_rows = {key: [] for key in ('per_image', 'fixed_gap_changes', 'logit_changes', 'source_weights', 'residual_components', 'feature_window_stats')}
        total_upstream = 0
        seen_cases = set()
        started = time.perf_counter()
        with torch.inference_mode():
            for index in range(len(data)):
                sample = data[index]
                case = sample['case_name']
                if case in seen_cases:
                    raise ValueError('Repeated validation image ID: '+case)
                seen_cases.add(case)
                entry = reference['images'][case]
                if entry['shape'] != [1500, 1500]:
                    raise ValueError('Fixed reference shape mismatch')
                expected = None
                if index == 0:
                    expected = model(sample['image'][None, :, :512, :512].to(a.device))[0].detach().clone()
                with PixelProbe(model) as probe:
                    logits, maps, difficult, checked, last_tile, last_normal, native_masks = infer_three(
                        model, sample, a.device, probe, all_rows['feature_window_stats'], case, expected, threshold)
                restored = model(last_tile)[0]
                if not torch.equal(restored, last_normal):
                    raise RuntimeError('Normal output not restored after diagnostic intervention')
                total_upstream += checked
                gt = sample['mask'][0].numpy() > .5
                analysed = analyse_case(case, gt, logits, maps, difficult, entry, threshold,
                                       float(np.abs(bias).mean()), native_masks)
                masks, *tables = analysed
                for key, rows in zip(('per_image', 'fixed_gap_changes', 'logit_changes', 'source_weights', 'residual_components'), tables):
                    all_rows[key].extend(rows)
                for mode in MODES:
                    if not cv2.imwrite(str(a.output_dir/'masks'/mode/(case+'.png')), masks[mode].astype(np.uint8)*255):
                        raise IOError('Cannot save diagnostic mask')
                for key, rows in all_rows.items():
                    write_csv(a.output_dir/(key+'.csv'), rows)
                write_csv(a.output_dir/'comparison.csv', summarize_comparison(all_rows['per_image']))
                print(f'[VAL] {index+1}/14 {case}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
                del logits, maps, sample, restored, last_tile, last_normal, analysed, expected
        if seen_cases != set(reference['images']):
            raise ValueError('Validation image IDs differ from the frozen reference')
        after = state_digest(model)
        checks = dict(model_unchanged=before == after, before=before, after=after,
            normal_instrumentation_bitwise_identical=True, normal_restored_after_each_image=True,
            upstream_equal_checks=total_upstream, images=len(data), fixed_gaps=198,
            all_parameters_frozen=not any(p.requires_grad for p in model.parameters()),
            any_parameter_has_grad=any(p.grad is not None for p in model.parameters()),
            elapsed_seconds=time.perf_counter()-started)
        dump(a.output_dir/'checks.json', checks)
        if before != after:
            raise RuntimeError('Model parameters or buffers changed')
        dump(a.output_dir/'inference_complete.json', dict(complete=True, images=14, checkpoint_sha256=sha))
        if a.command == 'all':
            run_topology(a)
    from show_hpixel_three_modes import show
    rendered = StringIO()
    with redirect_stdout(rendered):
        show(a.output_dir)
    (a.output_dir/'TABLES.txt').write_text(rendered.getvalue(), encoding='utf-8')
    print(rendered.getvalue(), flush=True)
    print('[ALL DONE]', a.output_dir, flush=True)


if __name__ == '__main__':
    main()
