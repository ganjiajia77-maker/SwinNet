"""Supplement an existing three-mode diagnostic with bias-only whole-image masks.

The old script retained bias-only regional summaries but no masks/logits. One
new val pass is required; the three cached modes and their topology are reused.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_hpixel_three_modes import (cldice, dump, metric_counts, metrics,
                                         summarize_comparison, write_csv)
from evaluate_native_h0_pair import digest, load_model
from evaluation.road_comparison.road_metrics import skeletonize
from hpixel_diagnostic_utils import MODES, PixelProbe, local_closed, state_digest
from mass_protocol import blend_weight, sliding_positions
from selector_transport_reference import bridge_proxy
from show_hpixel_three_modes import fmt, read, render

ALL_MODES = (*MODES, 'bias_only')


class BiasOnly:
    """Return the trained feature bias; skip all Q/K/V and source matching."""
    def __init__(self, model):
        self.core = getattr(model, 'swin_unet', model)
        self.module = self.core.h_pixel_reassembly

    def __enter__(self):
        self.had_forward = 'forward' in self.module.__dict__
        self.old_forward = self.module.__dict__.get('forward')
        self.had_disable = hasattr(self.core, 'disable_h_pixel_reassembly')
        self.old_disable = getattr(self.core, 'disable_h_pixel_reassembly', False)
        def forward(module, x, h, g, c, return_weights=False):
            if return_weights:
                raise ValueError('Bias-only has no source weights')
            return module.out.bias[None, :, None, None].expand_as(x)
        self.module.forward = types.MethodType(forward, self.module)
        self.core.disable_h_pixel_reassembly = False
        return self

    def __exit__(self, *unused):
        if self.had_forward:
            self.module.forward = self.old_forward
        else:
            del self.module.forward
        if self.had_disable:
            self.core.disable_h_pixel_reassembly = self.old_disable
        else:
            del self.core.disable_h_pixel_reassembly


def read_mask(path):
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.shape != (1500, 1500) or not np.isin(value, [0, 255]).all():
        raise ValueError('Expected original native1500 binary mask: '+str(path))
    return value > 0


def case_metrics(case, gt, baseline, bias, entry, threshold):
    counts = metric_counts(bias, gt)
    initial = {gap['id']: local_closed(baseline, gap) for gap in entry['gaps']}
    current = {gap['id']: local_closed(bias, gap) for gap in entry['gaps']}
    row = dict(case=case, mode='bias_only', threshold=threshold, **counts, **metrics(counts),
        cldice=cldice(bias, gt, skeletonize(gt)), closed=sum(current.values()),
        newly_closed_vs_normal=sum(v and not initial[k] for k, v in current.items()),
        lost_closed_vs_normal=sum(not v and initial[k] for k, v in current.items()),
        new_bridge_proxy_vs_normal=bridge_proxy(baseline, bias, gt),
        removed_bridge_proxy_vs_normal=bridge_proxy(bias, baseline, gt),
        added_tp=int((bias & ~baseline & gt).sum()), lost_tp=int((~bias & baseline & gt).sum()),
        added_fp=int((bias & ~baseline & ~gt).sum()), removed_fp=int((~bias & baseline & ~gt).sum()))
    fixed = [dict(case=case, gap_id=gap['id'], length=gap['length'], length_bin=gap['length_bin'],
        mode='bias_only', threshold=threshold, normal_closed=initial[gap['id']], mode_closed=current[gap['id']],
        newly_closed=current[gap['id']] and not initial[gap['id']],
        lost_closed=initial[gap['id']] and not current[gap['id']]) for gap in entry['gaps']]
    return row, fixed


def overlap_rows(fixed):
    closed = {mode: {(r['case'], int(r['gap_id'])) for r in fixed
                    if r['mode'] == mode and r['mode_closed']} for mode in ALL_MODES}
    normal = closed['normal']
    return [dict(mode=mode, normal_closed=len(normal), mode_closed=len(closed[mode]),
        common_closed=len(normal & closed[mode]), normal_only=len(normal-closed[mode]),
        mode_only=len(closed[mode]-normal),
        normal_only_ids=json.dumps(sorted(normal-closed[mode])),
        mode_only_ids=json.dumps(sorted(closed[mode]-normal))) for mode in ALL_MODES]


def cache_rows(source):
    per_image = read(source/'per_image.csv')
    floating = {'threshold', 'iou', 'f1', 'precision', 'recall', 'cldice'}
    for row in per_image:
        for key in row:
            if key not in ('case', 'mode'):
                row[key] = float(row[key]) if key in floating else int(row[key])
    fixed = read(source/'fixed_gap_changes.csv')
    for row in fixed:
        for key in ('gap_id', 'length'):
            row[key] = int(row[key])
        row['threshold'] = float(row['threshold'])
        for key in ('normal_closed', 'mode_closed', 'newly_closed', 'lost_closed'):
            if row[key] not in ('True', 'False'):
                raise ValueError('Invalid cached gap status')
            row[key] = row[key] == 'True'
    if len(per_image) != 14*3 or len(fixed) != 198*3:
        raise ValueError('Expected complete original three-mode per-image and 198-gap tables')
    return per_image, fixed


def packed_digest(chunks):
    value = hashlib.sha256()
    for case in sorted(chunks):
        value.update(case.encode()+b'\0'+chunks[case])
    return value.hexdigest()


def merge_topology(source, out):
    bias_report = out/'topology_fixed_val'/'val_report.json'
    if not bias_report.is_file():
        return False
    new = json.loads(bias_report.read_text())
    pointer = source/'topology_result_dir.txt'
    old_dir = Path(pointer.read_text().strip()) if pointer.is_file() else source/'topology'
    old_path = old_dir/'val_report.json'
    lookup = {r['name']: r for r in new['summaries']}
    if old_path.is_file():
        old = json.loads(old_path.read_text())
        for key in ('protocol', 'metric_implementation_sha256', 'case_ids_sha256'):
            if old[key] != new[key]:
                raise ValueError('Cannot merge cached topology with a different '+key)
        cache = json.loads((out/'checks.json').read_text())['cached_binary_content_sha256']
        for name in ('gt', *MODES):
            if old['binary_content_sha256'][name] != cache[name]:
                raise ValueError('Cached masks/GT differ from original topology input: '+name)
        if new['binary_content_sha256']['gt'] != cache['gt']:
            raise ValueError('Topology GT differs from bias replay GT')
        lookup.update({r['name']: r for r in old['summaries']})
    rows = []
    for mode in ALL_MODES:
        row = lookup.get(mode, {})
        rows.append(dict(mode=mode, **{k: row.get(k) for k in ('iou', 'cldice', 'topo_precision',
            'topo_recall', 'topo_f1', 'apls_bidirectional', 'tlts_infeasible')}))
    write_csv(out/'topology_comparison.csv', rows)
    dump(out/'topology_merge.json', dict(original_report=str(old_path),
        original_available=old_path.is_file(), bias_report=str(bias_report),
        old_modes_recomputed=False, protocol_checked=old_path.is_file()))
    return True


def show(out):
    protocol = json.loads((out/'protocol.json').read_text())
    print(f"目录：{out}\n固定阈值：{protocol['threshold']}；同一权重、同一14张验证图")
    rows = read(out/'comparison.csv')
    fields = ['mode', 'iou', 'cldice', 'closed', 'newly_closed_vs_normal', 'lost_closed_vs_normal',
              'new_bridge_proxy_vs_normal', 'added_tp', 'lost_tp', 'added_fp', 'removed_fp']
    render('四模式结果；所有变化相对正常模式',
        ['模式', 'IoU', 'clDice', '闭合', '新增闭合', '丢失闭合', '新增桥接代理', '新增TP', '丢失TP', '新增FP', '消除FP'],
        [[fmt(r[k]) if k in ('iou', 'cldice') else r[k] for k in fields] for r in rows])
    rows = read(out/'gap_overlap.csv')
    fields = ['mode', 'normal_closed', 'mode_closed', 'common_closed', 'normal_only', 'mode_only']
    render('相同缺口ID的闭合交集', ['模式', '正常闭合', '本模式闭合', '共同闭合', '仅正常闭合', '仅本模式闭合'],
           [[r[k] for k in fields] for r in rows])
    rows = read(out/'topology_comparison.csv')
    render('六项拓扑；150控制点，TOPO间隔0.002、步长0.0002',
        ['模式', 'clDice', 'TOPO-P', 'TOPO-R', 'TOPO-F1', 'APLS', 'TLTS-Infeasible'],
        [[r['mode']]+[fmt(r[k]) if r[k] else '尚未生成' for k in ('cldice', 'topo_precision',
            'topo_recall', 'topo_f1', 'apls_bidirectional', 'tlts_infeasible')] for r in rows])
    print('\n旧三模式预测和指标复用；只新增bias_only回放及其拓扑。')
    print('bias_only保留学习到的特征偏置，经过原道路头；不等同于手工平移道路阈值。')
    print('闭合数接近还需核对相同ID、误检和桥接；固定checkpoint干预不能替代重训消融。')


def topology(args):
    command = [sys.executable, '-u', str(ROOT/'scripts/evaluate_fixed_val_topology.py'),
        '--diagnostic_dir', str(args.output_dir), '--topo_tool', str(args.topo_tool),
        '--workers', str(args.workers), '--models', 'bias_only']
    with (args.output_dir/'topology.log').open('a') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    merge_topology(args.source_dir, args.output_dir)


@torch.inference_mode()
def infer(args):
    source = args.source_dir.resolve(); out = args.output_dir.resolve()
    original = json.loads((source/'protocol.json').read_text())
    done = json.loads((source/'inference_complete.json').read_text())
    frozen = json.loads((source/'checks.json').read_text())
    if not (done.get('complete') and done.get('images') == 14 and frozen.get('model_unchanged')):
        raise ValueError('Original diagnostic inference must be complete and frozen')
    if done['checkpoint_sha256'] != original['checkpoint_sha256']:
        raise ValueError('Original diagnostic provenance mismatch')
    if original.get('gap_margin') != 16 or original.get('gap_coverage') != .8:
        raise ValueError('Gap criterion differs from this diagnostic')
    checkpoint = args.checkpoint or Path(original['checkpoint'])
    fixed_path = args.fixed_gaps or Path(original['fixed_gaps'])
    if digest(checkpoint) != original['checkpoint_sha256'] or digest(fixed_path) != original['fixed_gaps_sha256']:
        raise ValueError('Checkpoint or frozen gap list differs from the original diagnostic')
    reference = json.loads(fixed_path.read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    root = Path(manifest['data_root'])
    if not root.is_absolute():root = source/root
    mask_dirs = {}
    for model in manifest['models']:
        path = Path(model['predictions']['val'][0]['path'])
        mask_dirs[model['name']] = path if path.is_absolute() else source/path
    if set(mask_dirs) != set(MODES):raise ValueError('Expected the original three cached modes')
    rows, fixed = cache_rows(source)
    model, config = load_model(checkpoint, torch.device(args.device), True)
    model.requires_grad_(False)
    if not config.get('h_pixel_reassembly'):
        raise ValueError('Expected the same H-pixel reassembly checkpoint')
    before = state_digest(model)
    if before != frozen['before']:
        raise ValueError('Loaded model state differs from the original frozen diagnostic')
    data = RoadSkeletonDataset(str(root), 'val', image_size=None, source_patch_size=1500,
                               augment=False, return_full_image=True)
    if len(data) != 14:raise ValueError('Expected 14 validation images')
    if out.exists():raise ValueError('Use a new bias-only output directory')
    (out/'masks'/'bias_only').mkdir(parents=True)
    protocol = copy.deepcopy(original)
    protocol.update(source_diagnostic_dir=str(source), modes=list(ALL_MODES),
        added_mode='bias_only', old_modes_recomputed=False,
        actual_checkpoint_loaded=str(checkpoint.resolve()),
        note='Same trained feature bias; Q/K/V skipped; no training or threshold selection')
    dump(out/'protocol.json', protocol)
    for item in manifest['models']:
        item['predictions']['val'][0]['path'] = str(mask_dirs[item['name']].resolve())
    manifest['data_root'] = str(root.resolve())
    added = copy.deepcopy(manifest['models'][0])
    added.update(name='bias_only')
    added['provenance']['intervention'] = 'bias_only'
    added['predictions'] = dict(val=[dict(path=str(out/'masks'/'bias_only'), threshold=original['threshold'])])
    manifest['models'].append(added)
    dump(out/'manifest.json', manifest)
    chunks = {name: {} for name in ('gt', *MODES)}
    identities = []; started = time.perf_counter(); threshold = original['threshold']
    weight = blend_weight(512, args.device)
    for index in range(len(data)):
        sample = data[index]; case = sample['case_name']; identities.append(case)
        gt = sample['mask'][0].numpy() > .5
        entry = reference['images'][case]
        cached = {mode: read_mask(mask_dirs[mode]/(case+'.png')) for mode in MODES}
        for mode in MODES:
            prior = [r for r in rows if r['case'] == case and r['mode'] == mode]
            if len(prior) != 1 or any(prior[0][k] != v for k, v in metric_counts(cached[mode], gt).items()):
                raise ValueError('Cached masks/GT differ from original table: '+case+' '+mode)
            prior_fixed = {(r['case'], r['gap_id']): r for r in fixed if r['mode'] == mode}
            for gap in entry['gaps']:
                if local_closed(cached[mode], gap) != prior_fixed[(case, gap['id'])]['mode_closed']:
                    raise ValueError('Cached fixed-gap status differs from original diagnostic')
            chunks[mode][case] = np.packbits(cached[mode]).tobytes()
        chunks['gt'][case] = np.packbits(gt).tobytes()
        image = sample['image'][None].to(args.device)
        if index == 0:
            tile = image[:, :, :512, :512]
            normal_before = model(tile)[0].clone()
            with PixelProbe(model) as probe:
                probe.set_mode('normal'); model(tile)
                probe.set_mode('residual_off'); outputs = model(tile)
                expected = probe.bias_only_tail(outputs[4]).clone()
        canvas = torch.zeros((1, 1, 1500, 1500), device=args.device)
        divisor = torch.zeros_like(canvas)
        with BiasOnly(model):
            for top in sliding_positions(1500, 512, 256):
                for left in sliding_positions(1500, 512, 256):
                    logits = model(image[:, :, top:top+512, left:left+512])[0]
                    if index == 0 and top == left == 0 and not torch.equal(logits, expected):
                        raise RuntimeError('Bias-only forward differs from the existing tail replay')
                    if not torch.isfinite(logits).all():raise ValueError('Invalid bias-only logits')
                    canvas[:, :, top:top+512, left:left+512] += logits.float()*weight
                    divisor[:, :, top:top+512, left:left+512] += weight
        if index == 0 and not torch.equal(model(tile)[0], normal_before):
            raise RuntimeError('Normal forward not restored after bias-only intervention')
        bias = ((canvas/divisor).sigmoid() >= threshold)[0, 0].cpu().numpy()
        if not cv2.imwrite(str(out/'masks'/'bias_only'/(case+'.png')), bias.astype(np.uint8)*255):
            raise IOError('Cannot save bias mask')
        row, changed = case_metrics(case, gt, cached['normal'], bias, entry, threshold)
        rows.append(row); fixed.extend(changed)
        write_csv(out/'per_image.csv', rows)
        write_csv(out/'fixed_gap_changes.csv', fixed)
        print(f'[BIAS ONLY] {index+1}/14 {case}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    if len(set(identities)) != 14 or set(identities) != set(reference['images']):
        raise ValueError('Validation image IDs differ from the fixed reference')
    after = state_digest(model)
    checks = dict(model_unchanged=before == after, before=before, after=after,
        bias_forward_matches_existing_tail_replay=True, replay_identity_scope='first 512 window',
        normal_forward_restored=True,
        old_modes_recomputed=False, cached_masks_and_gap_tables_verified=True,
        cached_binary_content_sha256={k: packed_digest(v) for k, v in chunks.items()},
        any_parameter_has_grad=any(p.grad is not None for p in model.parameters()), images=14,
        elapsed_seconds=time.perf_counter()-started)
    dump(out/'checks.json', checks)
    if before != after:raise RuntimeError('Model parameters or buffers changed')
    write_csv(out/'comparison.csv', summarize_comparison(rows, ALL_MODES))
    write_csv(out/'gap_overlap.csv', overlap_rows(fixed))
    dump(out/'inference_complete.json', dict(complete=True, images=14, checkpoint_sha256=original['checkpoint_sha256']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_dir', type=Path)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--fixed_gaps', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--command', choices=['all', 'infer', 'topology', 'show'], default='all')
    p.add_argument('--topo_tool', type=Path, default=Path('/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py'))
    args = p.parse_args()
    if args.command in ('all', 'infer'):
        if args.source_dir is None:p.error('Specify the existing three-mode source_dir')
        infer(args)
    else:
        args.source_dir = Path(json.loads((args.output_dir/'protocol.json').read_text())['source_diagnostic_dir'])
    if args.command in ('all', 'topology'):topology(args)
    if args.command == 'show':merge_topology(args.source_dir, args.output_dir)
    from contextlib import redirect_stdout
    from io import StringIO
    buffer = StringIO()
    with redirect_stdout(buffer):show(args.output_dir)
    (args.output_dir/'TABLES.txt').write_text(buffer.getvalue(), encoding='utf-8')
    print(buffer.getvalue(), flush=True)
    print('[ALL DONE]', args.output_dir, flush=True)


if __name__ == '__main__':main()
