"""Export paired relation-ablation tables and coordinate-resolved road diagnostics."""
import argparse
from contextlib import contextmanager
import html
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation/road_comparison')]
from diagnostics import scgra_decoder_mass as mass
from diagnostics.scgra_location_probes import gt_paths, check_path, unsupported_segments
from mass_protocol import stitch_logits
from road_metrics import skeletonize
from scripts.export_scgra_decoder_mass import dump, load, write_csv, native_rgb, save_png, source_hashes


MODES = {
    'full': (),
    'no_connectivity_bias': ('connectivity_topology_bias_scale',),
    'no_spatial_bias': ('token_relation_scale',),
    'no_both_biases': ('connectivity_topology_bias_scale', 'token_relation_scale'),
    'alpha_zero': ('raw_alpha',),
}
GRAPH_MODES = ('full', 'alpha_zero', 'retrained_scgra_off')
PATH_STATUSES = ('matched', 'gap_disconnected', 'local_disconnected', 'endpoint_unmatched',
                 'endpoints_merged', 'path_length_mismatch')


@contextmanager
def zero_parameters(module, names):
    saved = [(getattr(module, name), getattr(module, name).detach().clone()) for name in names]
    with torch.no_grad():
        for parameter, _ in saved:
            parameter.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, original in saved:
                parameter.copy_(original)


def summarize_attention(captured, reference=None):
    rows = []
    for index, (heads, valid, coords) in enumerate(captured):
        valid = valid[0].numpy()
        matrix = heads[0].numpy().astype(np.float64)[:, valid][:, :, valid]
        k = matrix.shape[-1]
        if k < 2:
            continue
        if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(-1), 1., atol=2e-6):
            raise ValueError('Invalid attention weights')
        baseline = None
        if reference is not None:
            original, baseline_valid, baseline_coords = reference[index]
            if not np.array_equal(valid, baseline_valid[0].numpy()) or not torch.equal(coords, baseline_coords):
                raise RuntimeError('Bias intervention changed the selected anchors')
            baseline = original[0].numpy().astype(np.float64)[:, valid][:, :, valid]
        for head in range(len(matrix)):
            weights = matrix[head]
            row = dict(window=index, head=head, valid_anchors=k,
                       entropy=float((-weights * np.log(weights.clip(1e-30))).sum(-1).mean() / np.log(k)),
                       dominant_key_mass=float(weights.mean(0).max()),
                       modal_argmax_share=float(np.bincount(weights.argmax(-1)).max() / k))
            if baseline is not None:
                row['attention_abs_change'] = float(abs(weights - baseline[head]).mean())
            rows.append(row)
    return rows


@torch.inference_mode()
def infer(model, image, mode, *, on_relation_terms=None):
    module = model.swin_unet.global_topology
    captured = []
    collect = mode in MODES and mode != 'alpha_zero'

    def forward(tile):
        result = model(tile)
        if collect:
            captured.append(module.last_relation_attention)
            module.last_relation_attention = None
        if on_relation_terms is not None:
            on_relation_terms({key: (val[0] if val.ndim else val).numpy()
                               for key, val in module.last_relation_terms.items()})
            del module.last_relation_terms
        return result[0]

    module.capture_relation_attention = collect
    previous_term_capture = getattr(module, 'capture_relation_terms', False)
    if on_relation_terms is not None:
        module.capture_relation_terms = True
    try:
        with zero_parameters(module, MODES.get(mode, ())):
            logits, _ = stitch_logits(forward, image)
    finally:
        module.capture_relation_attention = False
        if on_relation_terms is not None:
            module.capture_relation_terms = previous_term_capture
            if hasattr(module, 'last_relation_terms'):
                del module.last_relation_terms
    if collect and (len(captured) != 25 or any(x is None for x in captured)):
        raise RuntimeError('Expected one attention matrix in each of the 25 windows')
    return logits.sigmoid()[0, 0].cpu().numpy(), captured


def pixel_row(case, mode, threshold, probability, full_probability, gt, full_prediction):
    prediction = probability >= threshold
    delta = probability - full_probability
    added = prediction & ~full_prediction
    removed = full_prediction & ~prediction
    tp, fp, fn = mass.counts(prediction, gt)
    return dict(case_id=case, mode=mode, threshold=threshold,
                probability_mean_abs_change_vs_full=float(abs(delta).mean()),
                added_tp_vs_full=int((added & gt).sum()), added_fp_vs_full=int((added & ~gt).sum()),
                lost_tp_vs_full=int((removed & gt).sum()), removed_fp_vs_full=int((removed & ~gt).sum()),
                tp=tp, fp=fp, fn=fn, **mass.metrics(tp, fp, fn))


def write_location_evidence(case, rgb, gt, masks, args, directory):
    probes = gt_paths(gt, args.seed_spacing, args.path_length)
    gt_distance = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
    rows, segments = [], []
    case_dir = directory / 'maps' / case
    case_dir.mkdir(parents=True, exist_ok=True)
    save_png(case_dir/'rgb.png', rgb)
    save_png(case_dir/'gt.png', gt.astype(np.uint8)*255)
    for mode in GRAPH_MODES:
        if mode not in masks:
            continue
        pred = masks[mode]
        skel = skeletonize(pred)
        unsupported, unsupported_mask = unsupported_segments(skel, gt_distance, args.tolerance, args.min_branch)
        path_rows = []
        overlay = (rgb.astype(np.float32) * .65).astype(np.uint8)
        # A thicker marking is only for display. Coordinates and masks remain native.
        overlay[cv2.dilate(unsupported_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (170, 62, 211)
        for index, (path, length) in enumerate(probes):
            path_rows.append(dict(case_id=case, mode=mode, path_id=index,
                                  start_y=int(path[0, 0]), start_x=int(path[0, 1]),
                                  end_y=int(path[-1, 0]), end_x=int(path[-1, 1]),
                                  gt_path_px=length, **check_path(path, length, skel, args.tolerance, args.min_gap)))
        for row in path_rows:
            if row['status'] != 'matched':
                y, x = row['missing_run_center_y'], row['missing_run_center_x']
                if y is None:
                    path = probes[row['path_id']][0]
                    y, x = path[len(path)//2]
                cv2.circle(overlay, (int(x), int(y)), 4, (40, 175, 255), 1)
        save_png(case_dir/f'{mode}_mask.png', pred.astype(np.uint8)*255)
        save_png(case_dir/f'{mode}_location.png', overlay)
        rows.extend(path_rows)
        segments.extend(dict(case_id=case, mode=mode, **s) for s in unsupported)
    comparisons = []
    by_mode_path = {(r['mode'], r['path_id']): r for r in rows}
    for control in GRAPH_MODES[1:]:
        if control not in masks:
            continue
        overlay = (rgb.astype(np.float32) * .65).astype(np.uint8)
        candidates = []
        for index, (path, length) in enumerate(probes):
            full = by_mode_path['full', index]['status']
            off = by_mode_path[control, index]['status']
            outcome = ('gap_recovered' if off == 'gap_disconnected' and full == 'matched'
                       else 'gap_lost' if full == 'gap_disconnected' and off == 'matched'
                       else 'other_gain' if off != 'matched' and full == 'matched'
                       else 'other_loss' if full != 'matched' and off == 'matched'
                       else 'same_match_status')
            comparisons.append(dict(case_id=case, control=control, path_id=index, full=full,
                                    control_status=off, outcome=outcome))
            if outcome in {'gap_recovered', 'gap_lost'}:
                color = (220, 207, 0) if outcome == 'gap_recovered' else (34, 98, 255)
                cv2.polylines(overlay, [path[:, ::-1].astype(np.int32)], False, color, 3)
                candidates.append(dict(path_id=index, outcome=outcome, gt_yx=path.tolist()))
        save_png(case_dir/f'{control}_gap_comparison.png', overlay)
        dump(case_dir/f'{control}_gap_paths.json', candidates)
    return rows, segments, comparisons


def aggregate_pixels(rows, attention):
    result = []
    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        tp, fp, fn = [sum(r[k] for r in subset) for k in ('tp', 'fp', 'fn')]
        state = dict(mode=mode, images=len(subset), threshold=subset[0]['threshold'],
                     tp=tp, fp=fp, fn=fn, **mass.metrics(tp, fp, fn))
        for field in ('added_tp_vs_full', 'added_fp_vs_full', 'lost_tp_vs_full', 'removed_fp_vs_full'):
            state[field] = sum(r[field] for r in subset)
        state['probability_mean_abs_change_vs_full'] = float(np.mean([r['probability_mean_abs_change_vs_full'] for r in subset]))
        for key in ('entropy', 'dominant_key_mass', 'modal_argmax_share', 'attention_abs_change'):
            by_image = {}
            for r in attention:
                if r['mode'] == mode and key in r:
                    by_image.setdefault(r['case_id'], []).append(r[key])
            if by_image:
                state[key] = float(np.mean([np.mean(v) for v in by_image.values()]))
        result.append(state)
    return result


def graph_counts(mode, paths, segments):
    counts = {status: sum(row['status'] == status for row in paths) for status in PATH_STATUSES}
    return dict(mode=mode, probes=len(paths), match_rate=counts['matched']/max(len(paths), 1), **counts,
                **{kind: sum(row['kind'] == kind for row in segments) for kind in (
                   'attached_unsupported', 'isolated_unsupported')})


def make_report(output, pixel, graph, comparisons, cases, metadata):
    def table(rows, fields):
        def display(value):
            return f'{value:.5f}' if isinstance(value, float) else html.escape(str(value))
        return '<table><tr>'+''.join(f'<th>{k}</th>' for k in fields)+'</tr>'+''.join(
            '<tr>'+''.join(f'<td>{display(r.get(k, ""))}</td>' for k in fields)+'</tr>' for r in rows)+'</table>'

    links = []
    for case in cases:
        for control in GRAPH_MODES[1:]:
            filename = f'maps/{case}/{control}_gap_comparison.png'
            if (output / filename).is_file():
                links.append(f'<a href="{filename}">{case} · {control}</a>')
        for mode in GRAPH_MODES:
            filename = f'maps/{case}/{mode}_location.png'
            if (output / filename).is_file():
                links.append(f'<a href="{filename}">{case} · {mode} 位置图</a>')
    (output/'report.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>SCGRA evidence</title>'
        '<style>body{font:15px sans-serif;margin:30px;color:#223}td,th{padding:7px;border:1px solid #ddd}table{border-collapse:collapse}a{display:inline-block;margin:5px}</style>'
        f'<h1>SCGRA 偏置与道路位置诊断</h1><p>Massachusetts {metadata["split"]}，{len(cases)} 张。</p>'
        '<h2>偏置消融与分割结果</h2><p>分割指标先汇总 TP/FP/FN 再计算。注意力各窗各头在图内平均，再按影像平均。熵越低、主导键质量及最大权重键共享率越高，表示越集中。</p>'
        +table(pixel, ['mode','images','threshold','entropy','dominant_key_mass','modal_argmax_share','attention_abs_change','iou','f1','precision','recall','added_tp_vs_full','added_fp_vs_full','lost_tp_vs_full','removed_fp_vs_full'])
        +'<h2>道路位置级对照</h2><p>以下为固定的 GT 局部骨架路径探针。重叠探针可能指向同一物理断口，计数单位为探针。</p>'
        +table(graph, ['mode','probes','matched','match_rate','gap_disconnected','local_disconnected','endpoint_unmatched','endpoints_merged','path_length_mismatch','attached_unsupported','isolated_unsupported'])
        +table(comparisons, ['control','probes','gap_recovered','gap_lost','other_gain','other_loss','same_match_status'])
        +'<p>对照图青色为完整模型恢复的 GT 路径，橙色为完整模型丢失的 GT 路径。各组位置图橙色圆点标记最长缺失段中点，无缺失段时标记探针中点。品红为超出 GT 容差的预测骨架段。全部坐标见 CSV，恢复和丢失路径的像素见对应 JSON。</p>'
        +''.join(links)
        +'<h2>数据表</h2>'
        +''.join(f'<a href="{p.name}">{p.stem}</a>' for p in sorted(output.glob('*.csv')))
        +'<p>固定 checkpoint 的偏置消融描述推理依赖。重训对照按相同训练预算从相同预训练来源开始。探针匹配为本实验定义的局部诊断。</p>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--retrained_checkpoint', type=Path)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--graph_only', action='store_true', help='Run only full and SCGRA-off conditions')
    parser.add_argument('--seed_spacing', type=int, default=48)
    parser.add_argument('--path_length', type=int, default=64)
    parser.add_argument('--tolerance', type=int, default=5)
    parser.add_argument('--min_gap', type=int, default=3)
    parser.add_argument('--min_branch', type=int, default=10)
    args = parser.parse_args()
    if args.max_images < 0 or min(args.seed_spacing, args.path_length, args.min_gap, args.min_branch, args.tolerance) < 1:
        parser.error('Invalid diagnostic settings')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Check CUDA_VISIBLE_DEVICES and the active Python environment')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.checkpoint = args.checkpoint.resolve()
    full_sha = mass.file_digest(args.checkpoint)
    selection_path, _, threshold = mass.select_validation_record(args.checkpoint, full_sha, args.selection)
    full, saved, epoch = mass.load_model(args.checkpoint, device)
    if saved.get('freeze_scgra_off'):
        parser.error('--checkpoint must be the original full decoder model')
    control = None
    control_sha = None
    control_epoch = None
    if args.retrained_checkpoint:
        control_sha = mass.file_digest(args.retrained_checkpoint)
        control, control_saved, control_epoch = mass.load_model(args.retrained_checkpoint, device)
        if not control_saved.get('freeze_scgra_off') or control.swin_unet.global_topology.raw_alpha.item() != 0:
            parser.error('--retrained_checkpoint must come from --freeze_scgra_off training')
        if any(control_saved.get(k) != saved.get(k) for k in ('seed', 'pretrain_ckpt', 'root_path', 'num_workers', 'deterministic')):
            parser.error('Full and retrained control disagree on seed, pretraining path, data root, workers or determinism')
    modes = ['full', 'alpha_zero'] if args.graph_only else list(MODES)
    if control is not None:
        modes.append('retrained_scgra_off')
    data = mass.dataset(args.root_path, args.split)
    cases = [Path(name).stem for name in data.image_files][:args.max_images or None]
    output = args.output_dir
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=full_sha, full_epoch=epoch,
                    retrained_checkpoint=str(args.retrained_checkpoint) if control else None,
                    retrained_sha256=control_sha, retrained_epoch=control_epoch, split=args.split,
                    image_ids=cases, data_root=str(args.root_path.resolve()), modes=modes, device=str(device),
                    threshold=threshold, threshold_policy='shared original full-model validation threshold',
                    selection=str(selection_path), selection_sha256=mass.file_digest(selection_path),
                    seed_spacing=args.seed_spacing, path_length=args.path_length, tolerance=args.tolerance,
                    min_gap=args.min_gap, min_branch=args.min_branch, window_size=512, stride=256,
                    source_sha256={**source_hashes(), **{
                        str(p.relative_to(ROOT)): mass.file_digest(p) for p in (
                            Path(__file__).resolve(), ROOT/'diagnostics/scgra_location_probes.py')}})
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output/'metadata.json'
    if meta_path.exists():
        if load(meta_path) != metadata:
            parser.error('Existing output has different settings; select a new output directory')
    elif any(output.iterdir()):
        parser.error('Use a fresh output directory')
    dump(meta_path, metadata)
    (output/'partial').mkdir(exist_ok=True)
    weights_before = mass.tensor_digests(full)
    control_before = mass.tensor_digests(control) if control is not None else None
    pixel, attention, paths, segments, changes = [], [], [], [], []
    for i, case in enumerate(cases):
        cached = output/'partial'/f'{case}.json'
        inputs = {name: mass.file_digest(Path(directory)/filenames[i]) for name, directory, filenames in (
            ('rgb', data.image_dir, data.image_files), ('gt', data.mask_dir, data.label_files))}
        previous = load(cached) if cached.is_file() else None
        if previous is not None and previous.get('inputs') != inputs:
            raise ValueError(f'Image or GT content changed: {case}; choose a fresh output directory')
        if previous is not None and all((output/name).is_file() for name in previous['artifacts']):
            result = previous
        else:
            actual, image, gt = mass.read_sample(data, i)
            assert actual == case
            rgb = cv2.cvtColor(native_rgb(image), cv2.COLOR_RGB2BGR)
            image = image[None].to(device)
            probabilities, reference = infer(full, image, 'full' if not args.graph_only else 'plain')
            pred_full = probabilities >= threshold
            result = dict(pixel=[], attention=[])
            masks = {}
            for mode in modes:
                if mode == 'full':
                    prob, trace = probabilities, reference
                else:
                    prob, trace = infer(control if mode == 'retrained_scgra_off' else full, image, mode)
                result['pixel'].append(pixel_row(case, mode, threshold, prob, probabilities, gt, pred_full))
                result['attention'].extend(dict(case_id=case, mode=mode, **row) for row in summarize_attention(trace, reference))
                if mode in GRAPH_MODES:
                    masks[mode] = prob >= threshold
            result['paths'], result['segments'], result['changes'] = write_location_evidence(case, rgb, gt, masks, args, output)
            result['inputs'] = inputs
            result['artifacts'] = [str(p.relative_to(output)) for p in sorted((output/'maps'/case).iterdir())]
            dump(cached, result)
        pixel.extend(result['pixel'])
        attention.extend(result['attention'])
        paths.extend(result['paths'])
        segments.extend(result['segments'])
        changes.extend(result['changes'])
        print(f'[SCGRA] {i+1}/{len(cases)} {case}', flush=True)
    if weights_before != mass.tensor_digests(full):
        raise RuntimeError('Diagnosis did not restore all full-model weights and buffers')
    if control is not None and control_before != mass.tensor_digests(control):
        raise RuntimeError('Diagnosis changed the retrained model weights or buffers')
    pixel_summary = aggregate_pixels(pixel, attention)
    graph_summary, graph_per_image = [], []
    for mode in GRAPH_MODES:
        if mode not in modes:
            continue
        mode_paths = [r for r in paths if r['mode'] == mode]
        mode_segments = [r for r in segments if r['mode'] == mode]
        graph_summary.append(graph_counts(mode, mode_paths, mode_segments))
        for case in cases:
            graph_per_image.append(dict(case_id=case, **graph_counts(
                mode, [r for r in mode_paths if r['case_id'] == case],
                [r for r in mode_segments if r['case_id'] == case])))
    contrast = []
    for control_name in GRAPH_MODES[1:]:
        subset = [r for r in changes if r['control'] == control_name]
        if control_name in modes:
            contrast.append(dict(control=control_name, probes=len(subset),
                **{outcome: sum(r['outcome'] == outcome for r in subset) for outcome in (
                    'gap_recovered', 'gap_lost', 'other_gain', 'other_loss', 'same_match_status')}))
    for name, rows in (
        ('relation_and_segmentation_summary', pixel_summary), ('segmentation_per_image', pixel),
        ('attention_per_window_head', attention), ('local_path_probes', paths),
        ('unsupported_segments', segments), ('paired_path_changes', changes),
        ('graph_summary', graph_summary), ('graph_per_image', graph_per_image),
        ('paired_graph_summary', contrast),
    ):
        if rows:
            write_csv(output/f'{name}.csv', rows)
    make_report(output, pixel_summary, graph_summary, contrast, cases, metadata)
    print(f'[DONE] {output / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
