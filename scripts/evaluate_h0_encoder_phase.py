"""Strict Stage2-global checkpoint loading, native val sweep, fixed test, pinned six metrics.

No training. Same native1500/512/256 weighted-logit protocol as the recent
H-pixel evaluations, including a 0.1 lower bound on each triangular ramp.
"""
import argparse
import csv
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from h0_phase_training import file_digest, load_stage2_weights
from networks.encoder_h_postmerge_phase import install_encoder_h_postmerge_phase
from networks.vision_transformer import SwinUnet

NAME = 'h0_encoder_postmerge_global_stage2'
THRESHOLDS = np.round(np.arange(.05, .951, .01), 2).astype(np.float32)
TOPOLOGY_PROTOCOL = dict(image_size=1500, apls_max_control_nodes=150,
    apls_control_sampling_seed=1234, topo_seed_density_deg=.002, topo_step_deg=.0002,
    tlts_n_paths=200, tlts_seed=1234)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metrics(tp, fp, fn):
    return dict(iou=tp/max(tp+fp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1),
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1))


def counts(pred, gt):
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def build_model(saved):
    if not saved.get('encoder_h_postmerge_phase') or saved.get('use_ema'):
        raise ValueError('Expected the EMA-off original-H0 post-merge phase experiment')
    if not saved.get('global_topology_stage2') or saved.get('native_relation_experiment') or saved.get('h_pixel_reassembly'):
        raise ValueError('Expected a Stage2-global checkpoint from this experiment')
    size = saved.get('img_size', 512)
    options = SimpleNamespace(cfg=str(ROOT/'configs/swin_tiny_patch4_window7_224_lite.yaml'),
        opts=None, batch_size=2, img_size=size, zip=False, cache_mode=None, resume=None,
        accumulation_steps=1, use_checkpoint=False, amp_opt_level=None, tag=None, eval=True, throughput=False)
    config = get_config(options)
    keys = inspect.signature(SwinUnet).parameters
    kwargs = {key: value for key, value in saved.items()
              if key in keys and key not in ('config', 'img_size', 'num_classes', 'return_skeleton')}
    model = SwinUnet(config, img_size=size, num_classes=1, return_skeleton=True, **kwargs)
    install_encoder_h_postmerge_phase(model, saved.get('h_phase_initial_strength', .05))
    return model


def load_model(checkpoint, device):
    if not checkpoint.is_file():
        raise FileNotFoundError('Specify the actual best.pth file: '+str(checkpoint))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = state['args']
    model = build_model(saved)
    load_stage2_weights(model, state['model_state_dict'])
    model.requires_grad_(False)
    return model.to(device).eval(), saved, state.get('epoch')


def positions(length, tile=512, stride=256):
    starts = list(range(0, length-tile+1, stride))
    if not starts or starts[-1] != length-tile:
        starts.append(length-tile)
    return starts


def blend_weight(device):
    ramp = (1-torch.linspace(-1., 1., 512, device=device).abs()).clamp_min(.1)
    return torch.outer(ramp, ramp)[None, None]


@torch.inference_mode()
def infer(model, sample, device):
    image = sample['image'][None].to(device)
    gt = sample['mask'][0].numpy() > .5
    if image.shape != (1, 3, 1500, 1500) or gt.shape != (1500, 1500):
        raise ValueError('Expected native1500 RGB/GT without resizing')
    canvas = torch.zeros((1, 1, 1500, 1500), device=device)
    divisor = torch.zeros_like(canvas); weight = blend_weight(device)
    for top in positions(1500):
        for left in positions(1500):
            result = model(image[:, :, top:top+512, left:left+512])
            logits = result[0] if isinstance(result, tuple) else result
            if logits.shape != (1, 1, 512, 512) or not torch.isfinite(logits).all():
                raise ValueError('Invalid native surface output')
            canvas[:, :, top:top+512, left:left+512] += logits.float()*weight
            divisor[:, :, top:top+512, left:left+512] += weight
    if not (divisor > 0).all():raise RuntimeError('Uncovered native image pixels')
    probability = (canvas/divisor.clamp_min(1e-8)).sigmoid()[0, 0].cpu().numpy()
    return probability, gt


def select_threshold(curve):
    return max(curve, key=lambda row: (row['iou'], -row['threshold']))


def manifest(root, out, checkpoint, sha, threshold, name=NAME):
    return dict(data_root=str(root.resolve()), protocol=TOPOLOGY_PROTOCOL,
        models=[dict(name=name, provenance=dict(checkpoint=str(checkpoint.resolve()), sha256=sha,
            protocol='native1500 FP32; 512/256 triangular logit fusion, ramp min0.1; no EMA/TTA'),
            validation_selection=dict(selected_on='val', criterion='global_iou', threshold=threshold,
                                      evidence=str((out/'val_selection.json').resolve())),
            predictions={split: [dict(path=str((out/(split+'_masks')).resolve()), threshold=threshold)]
                         for split in ('val', 'test')})])


def save_mask(path, mask):
    if not cv2.imwrite(str(path), mask.astype(np.uint8)*255):raise IOError(path)


def inference(args):
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError('Specify --checkpoint /absolute/run/best.pth')
    if args.output_dir.exists():raise ValueError('Use a new inference output directory')
    device = torch.device(args.device)
    model, saved, epoch = load_model(args.checkpoint, device)
    name = NAME
    if saved.get('img_size') != 512 or saved.get('amp_dtype') != 'none':
        raise ValueError('Expected the FP32/512 experiment checkpoint')
    sha = file_digest(args.checkpoint)
    out = args.output_dir; out.mkdir(parents=True)
    for name in ('val_probabilities', 'val_masks', 'test_masks'):(out/name).mkdir()
    dump(out/'protocol.json', dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha,
        checkpoint_epoch=epoch, data_root=str(args.root_path.resolve()), image_size=1500,
        window=512, stride=256, ramp_min=.1, stitch='weighted logits then sigmoid',
        amp=False, EMA=False, TTA=False, threshold_selected_on='val only; global IoU',
        global_topology_location='after_decoder_stage2',
        thresholds=[float(t) for t in THRESHOLDS], topology=TOPOLOGY_PROTOCOL))
    val = RoadSkeletonDataset(str(args.root_path), 'val', image_size=None, source_patch_size=1500,
                              augment=False, return_full_image=True)
    if len(val) != 14:raise ValueError('Expected 14 validation images')
    aggregate = np.zeros((len(THRESHOLDS), 3), np.int64); cases = []
    for index in range(len(val)):
        sample = val[index]; probability, gt = infer(model, sample, device); case = sample['case_name']
        if case in cases:raise ValueError('Duplicate validation image ID')
        cases.append(case);np.save(out/'val_probabilities'/(case+'.npy'), probability.astype(np.float32))
        for j, threshold in enumerate(THRESHOLDS):aggregate[j] += counts(probability >= threshold, gt)
        print(f'[VAL] {index+1}/14 {case}', flush=True)
    curve = [dict(threshold=float(threshold), tp=int(c[0]), fp=int(c[1]), fn=int(c[2]),
                  **metrics(*map(int, c))) for threshold, c in zip(THRESHOLDS, aggregate)]
    write_csv(out/'val_threshold_scores.csv', curve)
    selected = select_threshold(curve); threshold = selected['threshold']
    dump(out/'val_selection.json', dict(selected_on='val', criterion='global_iou',
         checkpoint_sha256=sha, checkpoint_epoch=epoch, **selected))
    for case in cases:
        save_mask(out/'val_masks'/(case+'.png'), np.load(out/'val_probabilities'/(case+'.npy')) >= threshold)
    print(f'[VAL SELECTED] threshold={threshold:.8f} IoU={selected["iou"]:.6f}', flush=True)
    test = RoadSkeletonDataset(str(args.root_path), 'test', image_size=None, source_patch_size=1500,
                               augment=False, return_full_image=True)
    if len(test) != 49:raise ValueError('Expected 49 test images')
    rows = []; test_cases = set()
    for index in range(len(test)):
        sample = test[index]; probability, gt = infer(model, sample, device); case = sample['case_name']
        if case in test_cases:raise ValueError('Duplicate test image ID')
        test_cases.add(case); pred = probability >= threshold
        save_mask(out/'test_masks'/(case+'.png'), pred)
        tp, fp, fn = counts(pred, gt)
        rows.append(dict(case=case, threshold=threshold, tp=tp, fp=fp, fn=fn, **metrics(tp, fp, fn)))
        write_csv(out/'test_per_image.csv', rows)
        print(f'[TEST] {index+1}/49 {case}', flush=True)
    total = [sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn')]
    write_csv(out/'test_segmentation.csv', [dict(model=name, threshold=threshold, **metrics(*total))])
    dump(out/'manifest.json', manifest(args.root_path, out, args.checkpoint, sha, threshold, name))
    dump(out/'inference_complete.json', dict(complete=True, val_images=14, test_images=49,
         checkpoint_sha256=sha, threshold=threshold))
    print('[INFERENCE DONE]', out, flush=True)


def topology(args):
    out = args.output_dir
    complete = json.loads((out/'inference_complete.json').read_text())
    if not (complete.get('complete') and complete['val_images'] == 14 and complete['test_images'] == 49):
        raise ValueError('Complete val/test inference before computing topology')
    if not args.topo_tool.is_file():raise FileNotFoundError(args.topo_tool)
    command = [sys.executable, '-u', str(args.topo_tool), '--manifest', str(out/'manifest.json'),
        '--split', 'test', '--import_val_selection', '--output_dir', str(out/'topology'),
        '--workers', str(args.workers), '--progress_every', '1']
    if (out/'topology').is_dir() and any((out/'topology').iterdir()):command.append('--resume')
    print('[TOPOLOGY] test49; APLS150; TOPO0.002/0.0002; see topology.log', flush=True)
    with (out/'topology.log').open('a') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def show(out):
    files = [('验证集选定阈值', out/'val_selection.json'),
             ('测试分割指标', out/'test_segmentation.csv'),
             ('测试六项拓扑及其他指标', out/'topology'/'test_comparison.csv')]
    for title, path in files:
        print('\n========== '+title+' ==========')
        if not path.is_file():print('尚未生成：', path);continue
        if path.suffix == '.json':
            value = json.loads(path.read_text())
            print(' | '.join(f'{key}={value[key]}' for key in ('threshold', 'iou', 'f1', 'precision', 'recall')))
        else:
            with path.open(newline='', encoding='utf-8-sig') as stream:
                for row in csv.DictReader(stream):
                    fields = ('model', 'threshold', 'iou', 'f1', 'precision', 'recall') if path.name == 'test_segmentation.csv' else (
                        'name', 'cldice', 'topo_precision', 'topo_recall', 'topo_f1', 'apls_bidirectional', 'tlts_infeasible')
                    print(' | '.join(f'{key}={row.get(key, "缺失")}' for key in fields))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--root_path', type=Path, default=Path('/home/gjj/Swin-Unet-main/Massachusetts/tiff'))
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--command', choices=['all', 'infer', 'topology', 'show'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--topo_tool', type=Path, default=Path(
        '/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py'))
    args = parser.parse_args()
    if args.workers < 1:parser.error('workers must be positive')
    if args.command in ('all', 'infer'):inference(args)
    if args.command in ('all', 'topology'):topology(args)
    show(args.output_dir)
    if args.command != 'show':print('[ALL DONE]', args.output_dir, flush=True)


if __name__ == '__main__':main()
