"""Read-only ccc7992 global-anchor coverage on the original fixed198 validation gaps."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnostics.global_anchor_gap_geometry import gap_geometry, local_status, summarize_gap
from h0_phase_training import SAMPLING_PROTOCOL, file_digest, tensor_digests
from scripts.evaluate_h0_encoder_phase import blend_weight, load_model, positions


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def state_digest(model):
    return hashlib.sha256(json.dumps(tensor_digests(model), sort_keys=True).encode()).hexdigest()


class AnchorCapture:
    def __init__(self, module):
        self.module = module
        self.calls = 0

    def __enter__(self):
        self.saved = {key: (key in self.module.__dict__, getattr(self.module, key, None))
                      for key in ('capture_anchor_selection', 'last_anchor_selection', '_extract_fps_anchors')}
        self.original = self.module._extract_fps_anchors
        def observed(score):
            self.calls += 1
            return self.original(score)
        self.module._extract_fps_anchors = observed
        self.module.capture_anchor_selection = True
        self.module.last_anchor_selection = None
        return self

    def __exit__(self, *unused):
        for key, (existed, value) in self.saved.items():
            if existed:
                setattr(self.module, key, value)
            elif key in self.module.__dict__:
                delattr(self.module, key)


@torch.inference_mode()
def capture_image(model, image, device, check_first=False, tile=512, stride=256):
    """Only RGB enters the model. GT/gap definitions never enter this function."""
    image = image[None].to(device)
    size = image.shape[-1]
    if image.shape[-2:] != (size, size) or size < tile:
        raise ValueError('Expected a native square RGB image')
    canvas = torch.zeros((1, 1, size, size), device=device)
    divisor = torch.zeros_like(canvas)
    if tile == 512:
        weight = blend_weight(device)
    else:
        ramp = (1-torch.linspace(-1., 1., tile, device=device).abs()).clamp_min(.1)
        weight = torch.outer(ramp, ramp)[None, None]
    module = model.swin_unet.global_topology
    records = []
    identical = None
    for top in positions(size, tile, stride):
        for left in positions(size, tile, stride):
            rgb = image[:, :, top:top+tile, left:left+tile]
            ordinary = model(rgb)[0].detach().clone() if check_first and not records else None
            with AnchorCapture(module) as observer:
                outputs = model(rgb)
                if observer.calls != 1 or module.last_anchor_selection is None:
                    raise RuntimeError('Expected exactly one actual global anchor selection per window')
                trace = module.last_anchor_selection
                if trace['anchor_hw'] != (tile, tile):
                    raise RuntimeError('Global anchor coordinates are not native window pixels')
                record = {key: value[0].numpy().copy() for key, value in trace.items() if key != 'anchor_hw'}
                record.update(top=top, left=left, tile=tile, alpha_global=float(module.alpha_global))
            logits = outputs[0]
            if ordinary is not None:
                identical = bool(torch.equal(ordinary, logits))
                if not identical:
                    raise RuntimeError('Anchor recording changed the normal model output')
            if logits.shape != (1, 1, tile, tile) or not torch.isfinite(logits).all():
                raise RuntimeError('Invalid native surface logits')
            region = (slice(None), slice(None), slice(top, top+tile), slice(left, left+tile))
            canvas[region] += logits.float()*weight
            divisor[region] += weight
            records.append(record)
    if not bool((divisor > 0).all()):
        raise RuntimeError('Uncovered image pixels')
    probability = (canvas/divisor.clamp_min(1e-8)).sigmoid()[0, 0].cpu().numpy()
    return probability, records, identical


def aggregate(rows, title):
    valid = [row for row in rows if row['side_GT_valid']]
    result = dict(cohort=title, gaps=len(rows), valid_two_side_gaps=len(valid),
                  excluded=len(rows)-len(valid))
    for name in ('candidates_both_geometric', 'selected_both_geometric',
                 'candidates_both_correct', 'selected_correct_pair', 'both_correct_only_across_windows'):
        result[name] = sum(bool(row[name]) for row in valid)
        result[name+'_rate'] = result[name]/len(valid) if valid else None
    result['correct_candidates_but_no_selected_pair'] = sum(
        bool(row['candidates_both_correct']) and not row['selected_correct_pair'] for row in valid)
    result['no_joint_window'] = sum(bool(row['no_joint_window']) for row in valid)
    return result


def draw_case(sample, pred, gap, geometry, row, records, output):
    image = sample['image'].numpy().transpose(1, 2, 0)
    image = np.clip((image*np.array([.229, .224, .225])+np.array([.485, .456, .406]))*255, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = geometry['bbox']
    y0, x0 = max(y0-20, 0), max(x0-20, 0)
    y1, x1 = min(y1+20, pred.shape[0]), min(x1+20, pred.shape[1])
    base = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
    panels = []
    for title, name in (('GT + anchors', None), ('prediction + nodes', 'prediction')):
        panel = base.copy()
        target = sample['mask'][0].numpy().astype(bool) if name is None else pred
        cropped = target[y0:y1, x0:x1]
        panel[cropped] = (.6*panel[cropped]+.4*np.array([0, 180, 0])).astype(np.uint8)
        palette = ((255, 100, 0), (0, 120, 255), (180, 80, 180), (180, 180, 80))
        for group_index, group in enumerate(geometry['anchors']):
            color = palette[group_index % len(palette)]
            for y, x in group:
                if y0 <= y < y1 and x0 <= x < x1:
                    panel[y-y0, x-x0] = color
        for y, x in geometry['pixels']:
            panel[y-y0, x-x0] = (255, 0, 255)
        if name:
            # Display a single best window; never make cross-window pairs look real.
            candidates = []
            for index, record in enumerate(records):
                result, detail = summarize_gap([record], geometry)
                d = detail[0]
                candidates.append((bool(result['selected_correct_pair']),
                    bool(result['candidates_both_correct']), bool(result['selected_both_geometric']),
                    d['candidate_side0']+d['candidate_side1'], index))
            *_, selected_window = max(candidates)
            record = records[selected_window]
            offset = np.array([record['top'], record['left']])
            for key, valid_key, color, radius in (
                ('candidate_coords', 'candidate_valid', (0, 220, 220), 1),
                ('selected_coords', 'selected_valid', (255, 255, 0), 3)):
                for y, x in record[key][record[valid_key]]+offset:
                    if y0 <= y < y1 and x0 <= x < x1:
                        cv2.circle(panel, (int(x-x0), int(y-y0)), radius, color, 1)
            _, best = summarize_gap([record], geometry)
            for label, (y, x) in zip(('A', 'B'), best[0]['pair_nodes']):
                cv2.circle(panel, (int(x-x0), int(y-y0)), 5, (0, 255, 0), 1)
                cv2.putText(panel, label, (int(x-x0)+3, int(y-y0)-3), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
            title += f' W{selected_window}'
        panel = cv2.copyMakeBorder(panel, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, title, (3, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        panels.append(panel)
    cv2.imwrite(str(output), np.concatenate(panels, axis=1))


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a new diagnostic output directory')
    selection = json.loads((args.evaluation_dir/'val_selection.json').read_text())
    cached_protocol = json.loads((args.evaluation_dir/'protocol.json').read_text())
    checkpoint_sha = file_digest(args.checkpoint)
    if selection.get('selected_on') != 'val' or selection.get('criterion') != 'global_iou':
        raise ValueError('Use the existing validation-selected threshold; no threshold fitting here')
    if selection['checkpoint_sha256'] != checkpoint_sha or cached_protocol['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Checkpoint differs from the existing evaluation prediction cache')
    if cached_protocol.get('window') != 512 or cached_protocol.get('stride') != 256 or cached_protocol.get('ramp_min') != .1:
        raise ValueError('Existing evaluation uses a different stitching protocol')
    if Path(cached_protocol['data_root']).resolve() != args.root_path.resolve():
        raise ValueError('Data root differs from the original evaluation')
    reference = json.loads(args.fixed_gaps.read_text())
    if (reference.get('count') != 198 or len(reference.get('images', {})) != 14
            or reference.get('p0_reference_threshold') != .45):
        raise ValueError('Use the original fixed198/14-image reference at initial P0 threshold0.45')
    keys = [(case, gap['id']) for case, entry in reference['images'].items() for gap in entry['gaps']]
    if len(keys) != 198 or len(set(keys)) != 198:
        raise ValueError('Fixed gap IDs/counts differ from the original reference')
    device = torch.device(args.device)
    model, saved, epoch = load_model(args.checkpoint, device)
    if saved.get('sampling_protocol') != SAMPLING_PROTOCOL or saved.get('complete_no_h_ablation'):
        raise ValueError('Expected the crop-repaired ccc7992 checkpoint with encoder H/global enabled')
    if saved.get('img_size') != 512 or saved.get('amp_dtype') != 'none':
        raise ValueError('Expected native512 FP32 experiment')
    core = model.swin_unet
    if not core.enable_global_topology or core.global_topology is None:
        raise ValueError('Global topology must be active')
    data = RoadSkeletonDataset(str(args.root_path), 'val', image_size=None, source_patch_size=1500,
                               augment=False, return_full_image=True)
    cases = [Path(name).stem.replace('_sat', '') for name in data.image_files]
    if len(cases) != 14 or set(cases) != set(reference['images']):
        raise ValueError('Validation cases differ from the original fixed-gap image set')
    for folder in ('val_probabilities', 'val_masks'):
        actual = {path.stem for path in (args.evaluation_dir/folder).glob('*') if path.is_file()}
        if actual != set(cases):
            raise ValueError('Missing/extra cases in existing '+folder)
    threshold = float(selection['threshold'])
    out = args.output_dir
    out.mkdir(parents=True)
    for name in ('nodes', 'cases_visual'):
        (out/name).mkdir()
    protocol = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
        checkpoint_epoch=epoch, evaluation_dir=str(args.evaluation_dir.resolve()),
        root_path=str(args.root_path.resolve()), fixed_gaps=str(args.fixed_gaps.resolve()),
        fixed_gaps_sha256=file_digest(args.fixed_gaps), declared_reference_fingerprint=reference.get('sha256'),
        fixed_reference_origin=reference.get('origin'), threshold=threshold,
        reference_P0_threshold=.45, tolerance_native_pixels=args.tolerance,
        side_validation='GT-local eight-neighbor geodesic within tolerance; closer to own anchors than other side or gap',
        closure_margin=16, closure_coverage=.8, primary_cohort='locally disconnected anchors at current final surface threshold',
        image_size=1500, window=512, stride=256, ramp_min=.1, max_nodes=core.global_topology.max_nodes,
        candidate_rule='actual topk min(max(256,max_nodes),window_pixels); score>0; actual FPS',
        pair_rule='two distinct valid nodes in the SAME inference window; all selected tokens fully interact',
        no_training=True, GT_in_model_forward=False, source_hashes={
            str(path.relative_to(ROOT)): file_digest(path) for path in (
                ROOT/'networks/keypoint_global_topology.py', ROOT/'diagnostics/global_anchor_gap_geometry.py', Path(__file__).resolve())})
    write_json(out/'protocol.json', protocol)
    before = state_digest(model)
    gap_rows, window_rows, per_image, node_rows = [], [], [], []
    cache_checks = []
    first_equal = None
    for index in range(len(data)):
        sample = data[index]
        case = sample['case_name']
        gt = sample['mask'][0].numpy().astype(bool)
        entry = reference['images'][case]
        if entry['shape'] != [1500, 1500]:
            raise ValueError('Reference coordinates are not native1500')
        cached = np.load(args.evaluation_dir/'val_probabilities'/(case+'.npy'), allow_pickle=False)
        mask_file = args.evaluation_dir/'val_masks'/(case+'.png')
        cached_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if (cached.shape != gt.shape or not np.isfinite(cached).all() or cached.min() < 0 or cached.max() > 1
                or cached_mask is None or not np.array_equal(cached_mask > 0, cached >= threshold)):
            raise ValueError('Invalid or inconsistent original prediction cache: '+case)
        probability, records, same = capture_image(model, sample['image'], device, check_first=index == 0)
        if same is not None:
            first_equal = same
        difference = float(np.abs(probability-cached).max())
        if difference > 1e-6 or not np.array_equal(probability >= threshold, cached_mask > 0):
            raise ValueError(f'Current forward differs from cached predictions: {case} max_abs={difference}')
        pred = cached_mask > 0
        cache_checks.append(dict(case=case, probability_max_abs=difference,
            cached_probability_sha256=file_digest(args.evaluation_dir/'val_probabilities'/(case+'.npy')),
            cached_mask_sha256=file_digest(mask_file), GT_sha256=hashlib.sha256(gt.tobytes()).hexdigest()))
        arrays = {key: np.stack([record[key] for record in records]) for key in records[0]}
        np.savez_compressed(out/'nodes'/(case+'.npz'), **arrays)
        for wi, record in enumerate(records):
            for kind in ('candidate', 'selected'):
                valid = record[kind+'_valid']
                for ni in np.flatnonzero(valid):
                    y, x = record[kind+'_coords'][ni]
                    node_rows.append(dict(case=case, window=wi, window_top=record['top'], window_left=record['left'],
                        kind=kind, index=int(ni), y=int(y)+record['top'], x=int(x)+record['left'],
                        score=float(record[kind+'_scores'][ni]),
                        selected_candidate_index=int(record['selected_candidate_indices'][ni]) if kind == 'selected' else int(ni)))
        case_rows = []
        for gap in entry['gaps']:
            geometry = gap_geometry(gt, gap, args.tolerance)
            status = local_status(pred, gap)
            coverage, details = summarize_gap(records, geometry)
            row = dict(case=case, gap_id=gap['id'], length=gap.get('length', len(gap['pixels'])),
                length_bin=gap.get('length_bin', 'unknown'), anchor_groups=len(gap['anchors']),
                side_GT_valid=geometry['valid'], exclusion_reason=geometry['reason'], **status, **coverage)
            gap_rows.append(row); case_rows.append(row)
            for detail in details:
                window_rows.append(dict(case=case, gap_id=gap['id'], side_GT_valid=geometry['valid'], **detail))
            if not args.no_visuals:
                draw_case(sample, pred, gap, geometry, row, records,
                          out/'cases_visual'/f'{case}_gap_{gap["id"]}.png')
        per_image.append(dict(case=case, windows=len(records), alpha_global=records[0]['alpha_global'],
            candidates=sum(int(r['candidate_valid'].sum()) for r in records),
            selected=sum(int(r['selected_valid'].sum()) for r in records),
            **{key: value for key, value in aggregate([r for r in case_rows if r['still_disconnected']], 'disconnected').items()
               if key != 'cohort'}))
        write_csv(out/'fixed_gap_node_coverage.csv', gap_rows)
        print(f'[VAL {index+1}/14] {case} gaps={len(case_rows)} disconnected={sum(r["still_disconnected"] for r in case_rows)} windows={len(records)} cache_max_abs={difference:.3g}', flush=True)
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Model weights/buffers/gradients changed')
    cohorts = [aggregate(gap_rows, 'all198'),
        aggregate([r for r in gap_rows if r['still_disconnected']], 'still_disconnected'),
        aggregate([r for r in gap_rows if not r['closed_proxy']], 'legacy_unclosed_proxy'),
        aggregate([r for r in gap_rows if r['closed_proxy']], 'closed_control')]
    summary = dict(complete=True, images=len(data), fixed_gaps=len(gap_rows), threshold=threshold,
        tolerance=args.tolerance, cohorts=cohorts,
        exclusion_reasons=dict(Counter(r['exclusion_reason'] for r in gap_rows if not r['side_GT_valid'])),
        note='Availability is not attention strength, actual use, or confirmed road repair. GT-rejected geometric hits are not automatically parallel roads.')
    checks = dict(model_unchanged=before == after, state_before=before, state_after=after,
        first_window_capture_bitwise_equal=first_equal, cached_predictions_equal=True,
        no_training=True, GT_in_forward=False, cache_checks=cache_checks)
    write_csv(out/'summary.csv', cohorts)
    write_csv(out/'per_image.csv', per_image)
    write_csv(out/'gap_windows.csv', window_rows)
    write_csv(out/'nodes.csv', node_rows)
    write_json(out/'checks.json', checks)
    write_json(out/'summary.json', summary)
    write_report(out, summary)
    show(out)
    print('[ALL DONE]', out, flush=True)


def write_report(out, summary):
    lines = ['# 全局节点对固定缺口的覆盖诊断', '',
        f"固定缺口{summary['fixed_gaps']}；阈值{summary['threshold']:.6f}；原图容差{summary['tolerance']}像素。", '',
        '| 分组 | 缺口数 | 可核验两侧 | 候选覆盖两侧 | 最终节点覆盖两侧 | 最终正确道路节点对 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['cohorts']:
        lines.append(f"| {row['cohort']} | {row['gaps']} | {row['valid_two_side_gaps']} | {row['candidates_both_geometric']} | {row['selected_both_geometric']} | {row['selected_correct_pair']} |")
    lines += ['', '主分析使用still_disconnected，按局部锚点未连通确定。legacy_unclosed_proxy另含覆盖不足。',
        '候选/最终两侧几何命中不保证落在正确道路；candidates_both_correct单列GT核验结果。',
        '所有主计数必须在同一个512推理窗口成立；跨窗合并仅作辅助统计。',
        '超过两组锚点、GT局部路径或侧别不明确者列为排除，不计作失败。',
        '局部GT道路距离不能区分标注中已连成一体的立交/交叉；高分案例仍需检查图像。',
        '可用节点对不等于高注意力权重、有效回写或实际补通。可视化每幅图只显示一个窗口的节点。',
        'GT仅用于离线核验；模型权重与缓冲区不变，输出与原评价缓存一致。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def show(out):
    summary = json.loads((out/'summary.json').read_text())
    print('\ncohort                         gaps valid candidates_both selected_both correct_pair')
    for row in summary['cohorts']:
        print(f"{row['cohort']:30} {row['gaps']:4d} {row['valid_two_side_gaps']:5d} {row['candidates_both_geometric']:15d} {row['selected_both_geometric']:13d} {row['selected_correct_pair']:12d}")
    print('排除原因：', json.dumps(summary['exclusion_reasons'], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluation_dir', type=Path)
    parser.add_argument('--root_path', type=Path, default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    parser.add_argument('--fixed_gaps', type=Path, default=Path('/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json'))
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=12.)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_visuals', action='store_true')
    parser.add_argument('--command', choices=('run', 'show'), default='run')
    args = parser.parse_args()
    if args.command == 'show':
        show(args.output_dir)
        return
    if args.checkpoint is None or args.evaluation_dir is None or not 0 < args.tolerance <= 64:
        parser.error('Specify checkpoint, existing evaluation directory, and tolerance in (0,64]')
    run(args)


if __name__ == '__main__':
    main()
