"""Paired inference intervention on one directional-diffusion checkpoint; never trains."""
import os
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

from diffusion_experiment import dataset
from diffusion_protocol import ROOT, load_checkpoint, write_json
from mass_protocol import blend_weight, sliding_positions
from networks.directional_diffusion import neighbor
from topology_direction_constants import CONNECTIVITY_DIRECTIONS

VARIANTS = ('direction_on', 'direction_off')
GRID = [round(i / 100, 2) for i in range(5, 100, 5)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def paired_window(model, diffusion, image):
    """Toggle only the direction preference, using identical X/G/H/D and weights."""
    previous_mode, capture = diffusion.mode, diffusion.capture_diagnostics
    saved = {}

    def observe(module, inputs, result):
        saved['x'] = inputs[0].detach()
        saved['inputs'] = tuple(tensor.detach() for tensor in inputs)
        saved['delta_x'] = (result - inputs[0]).float().detach()

    hook = diffusion.register_forward_hook(observe)
    try:
        diffusion.capture_diagnostics = True
        diffusion.mode = 'semantic_directional'
        output = model(image)
        on = output[0] if isinstance(output, tuple) else output
        first = dict(saved, **diffusion.last_diagnostics)
        diffusion.mode = 'semantic'  # beta=0 at edge_weights; D3 itself is unchanged.
        output = model(image)
        off = output[0] if isinstance(output, tuple) else output
        second = dict(saved, **diffusion.last_diagnostics)
        if not all(torch.equal(on_input, off_input) for on_input, off_input in
                   zip(first['inputs'], second['inputs'])):
            raise RuntimeError('X3/G3/H2/D3 changed: this is not a direction-only intervention')
        valid = torch.cat([neighbor(torch.ones_like(first['edges'][:, :1]), dy, dx)
                           for dy, dx in CONNECTIVITY_DIRECTIONS], 1)
        edge_difference = (first['edges'] - second['edges']).abs()
        delta_difference = (first['delta_x'] - second['delta_x']).abs()
        stats = {
            'edge_abs_change_mean_valid': float((edge_difference * valid).sum() / valid.sum()),
            'delta_z_abs_change_mean': float((first['delta_z'] - second['delta_z']).abs().mean()),
            'delta_x_abs_change_mean': float(delta_difference.mean()),
            'delta_x_on_abs_mean': float(first['delta_x'].abs().mean()),
            'delta_x_off_abs_mean': float(second['delta_x'].abs().mean()),
            'x_abs_mean': float(first['x'].float().abs().mean()),
            'delta_x_abs_change_max': float(delta_difference.max()),
        }
        return on, off, stats
    finally:
        hook.remove()
        diffusion.mode = previous_mode
        diffusion.capture_diagnostics = capture
        diffusion.last_diagnostics = None


@torch.inference_mode()
def paired_stitch(model, diffusion, image):
    if image.shape != (1, 3, 1500, 1500):
        raise ValueError('Expected one native1500 RGB image')
    weight = blend_weight(512, image.device)
    divisor = torch.zeros((1, 1, 1500, 1500), device=image.device)
    canvases = [torch.zeros_like(divisor), torch.zeros_like(divisor)]
    measurements = []
    for top in sliding_positions(1500, 512, 256):
        for left in sliding_positions(1500, 512, 256):
            on, off, stats = paired_window(model, diffusion, image[:, :, top:top+512, left:left+512])
            measurements.append(stats)
            region = (..., slice(top, top+512), slice(left, left+512))
            for canvas, logits in zip(canvases, (on, off)):
                if logits.shape != (1, 1, 512, 512) or not bool(torch.isfinite(logits).all()):
                    raise ValueError('Invalid window logits')
                canvas[region] += logits.float() * weight
            divisor[region] += weight
    stats = {key: (max(row[key] for row in measurements) if key.endswith('_max') else
                   float(np.mean([row[key] for row in measurements]))) for key in measurements[0]}
    stats['delta_x_change_over_x_abs_mean'] = stats['delta_x_abs_change_mean'] / max(stats['x_abs_mean'], 1e-12)
    return [canvas / divisor for canvas in canvases], stats


def build_manifest(args, metadata, config):
    root = Path(args.output_dir)
    models = []
    for name in VARIANTS:
        sources = {split: [{'threshold': float(p.parent.name.split('_', 1)[1]), 'path': str(p.resolve())}
                           for p in sorted((root/'predictions'/split/name).glob('threshold_*/surface'))]
                   for split in ('val', 'test')}
        models.append({'name': name, 'provenance': {
            'checkpoint_sha256': config['checkpoint_sha256'], 'checkpoint': config['checkpoint'],
            'epoch': metadata['epoch'], 'architecture': metadata['architecture_config'],
            'seed': metadata['args']['seed'], 'source_sha256': config['source_sha256'],
            'experiment': 'same-checkpoint inference intervention; not a retrained semantic model',
            'effective_direction_beta': config['beta'] if name == 'direction_on' else 0.,
            'inference': 'native1500; 512/stride256 weighted logits; FP32; no EMA/TTA/postprocess',
        }, 'predictions': sources})
    path = root/'direction_manifest.json'
    write_json(path, {'data_root': config['data_root'], 'protocol': {
        'image_size': 1500, 'short_area_threshold': 20, 'apls_max_nodes': 64, 'apls_snap_radius': 5.},
        'models': models})
    return path


def compare(args, manifest, split, report_name, selection=None):
    command = [sys.executable, '-u', str(ROOT/'evaluation/road_comparison/compare_predictions.py'),
               '--manifest', str(manifest), '--split', split, '--output_dir',
               str(Path(args.output_dir)/report_name), '--progress_every', '10']
    if split == 'val':
        command.append('--select_threshold_only')
    else:
        command += ['--selection', str(selection)]
    subprocess.run(command, check=True)


def fixed_selection(selection, scores):
    fixed = copy.deepcopy(selection)
    threshold = selection['models']['direction_on']['threshold']
    fixed['criterion'] = 'direction_on_validation_global_iou_threshold_fixed_for_both'
    for name in VARIANTS:
        entry = fixed['models'][name]
        entry['independently_selected_threshold'] = entry['threshold']
        entry['threshold'] = threshold
        entry['val_iou'] = next(float(row['iou']) for row in scores
                               if row['name'] == name and abs(float(row['threshold'])-threshold) < 1e-8)
        entry['note'] = 'Threshold selected on direction_on val, then held fixed for both interventions'
    return fixed


def export(args, model, diffusion, thresholds, fixed_threshold):
    root = Path(args.output_dir)
    split = args.action
    directories = {(name, t): root/'predictions'/split/name/f'threshold_{t:.2f}'/'surface'
                   for name in VARIANTS for t in thresholds[name]}
    if (root/f'{split}_influence.csv').exists() or any(p.exists() for p in directories.values()):
        raise FileExistsError('Partial/existing export: use a fresh diagnostic output directory')
    for path in directories.values():
        path.mkdir(parents=True)
    data = dataset(args, split)
    rows = []
    for index in range(len(data)):
        sample = data[index]
        logits, stats = paired_stitch(model, diffusion, sample['image'][None].to(args.device))
        probabilities = [tensor.sigmoid()[0, 0].cpu().numpy() for tensor in logits]
        for name, probability in zip(VARIANTS, probabilities):
            for threshold in thresholds[name]:
                path = directories[(name, threshold)]/(sample['case_name']+'_pred.png')
                if not cv2.imwrite(str(path), (probability >= threshold).astype(np.uint8)*255):
                    raise OSError(path)
        logit_change = (logits[0]-logits[1]).abs()
        probability_change = np.abs(probabilities[0]-probabilities[1])
        row = {'case_id': sample['case_name'], **stats,
               'surface_logit_abs_change_mean': float(logit_change.mean()),
               'surface_logit_abs_change_max': float(logit_change.max()),
               'surface_probability_abs_change_mean': float(probability_change.mean()),
               'surface_probability_abs_change_max': float(probability_change.max())}
        # Val is not allowed to select a threshold until every val case is exported.
        # Its fixed-threshold flips are filled from saved masks below, after selection.
        if fixed_threshold is not None:
            row['fixed_threshold'] = fixed_threshold
            row['mask_flip_fraction'] = float(np.mean((probabilities[0] >= fixed_threshold) !=
                                                      (probabilities[1] >= fixed_threshold)))
        rows.append(row)
        write_csv(root/f'{split}_influence.csv', rows)
        print(f'[{split}] paired inference {index+1}/{len(data)}: {sample["case_name"]}', flush=True)
    return rows


def summarize(args, rows, threshold):
    root = Path(args.output_dir)
    if args.action == 'val':
        for row in rows:
            masks = [cv2.imread(str(root/'predictions'/'val'/name/f'threshold_{threshold:.2f}'/'surface'/
                                   (row['case_id']+'_pred.png')), cv2.IMREAD_GRAYSCALE) for name in VARIANTS]
            row['fixed_threshold'] = threshold
            row['mask_flip_fraction'] = float(np.mean(masks[0] != masks[1]))
        write_csv(root/'val_influence.csv', rows)
    result = {'images': len(rows), 'fixed_threshold_selected_on_direction_on_val': threshold,
              'checkpoint_parameters_and_buffers_unchanged': True,
              'device': torch.cuda.get_device_name() if args.device.type == 'cuda' else 'CPU',
              'torch_version': torch.__version__,
              'interpretation': 'Changes measure direction dependence; metric deltas measure benefit in this checkpoint, not retraining benefit.',
              'aggregation': 'each image has 25 windows; edge/feature means average windows including overlaps; surface means use stitched native1500 pixels',
              'means': {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'case_id'}}
    write_json(root/f'{args.action}_influence_summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def metric_deltas(root):
    reports = {}
    for report in ('test_fixed', 'test_own_val'):
        with (root/report/'test_comparison.csv').open() as stream:
            rows = {row['name']: row for row in csv.DictReader(stream)}
        reports[report] = {'direction_on_minus_off': {
            key: float(rows['direction_on'][key])-float(rows['direction_off'][key])
            for key in rows['direction_on'] if key not in ('name', 'threshold')},
            'thresholds': {name: float(row['threshold']) for name, row in rows.items()}}
    reports['interpretation'] = ('For IoU/F1/clDice/APLS positive deltas favor direction_on; '
        'for break_rate negative deltas favor direction_on. Fewer fragments alone do not prove better roads: '
        'also inspect recall/clDice/APLS. This is an inference intervention, not retraining ablation.')
    write_json(root/'test_metric_deltas.json', reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['val', 'test'])
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--root_path', default='/home/gjj/Swin-Unet-main/Massachusetts/tiff')
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu' if args.gpu == 'cpu' else 'cuda')
    args.root_path = str(Path(args.root_path).resolve())
    root = Path(args.output_dir)
    if args.action == 'val' and root.exists() and any(root.iterdir()):
        parser.error('Use a fresh output_dir for val')
    root.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(args.model_path)
    arc = checkpoint['architecture_config']
    if arc['diffusion_mode'] != 'semantic_directional' or arc['diffusion_beta'] <= 0:
        parser.error('Select the already-trained semantic_directional checkpoint, not semantic or identity')
    args.seed = checkpoint['args']['seed']
    model = model.to(args.device).eval()
    diffusion = model.swin_unet.decoder_structure_blocks['3'].diffusion
    config = {'checkpoint': str(Path(args.model_path).resolve()), 'checkpoint_sha256': sha256(args.model_path),
              'data_root': args.root_path, 'beta': arc['diffusion_beta'],
              'source_sha256': {name: sha256(ROOT/name) for name in (
                  'diagnose_direction.py', 'diffusion_experiment.py', 'diffusion_protocol.py', 'networks/directional_diffusion.py',
                  'networks/vision_transformer.py', 'networks/skeleton_guided_head.py',
                  'networks/swin_transformer_unet_skip_expand_decoder_sys.py',
                  'mass_protocol.py', 'datasets/dataset_road_skeleton.py',
                  'evaluation/road_comparison/compare_predictions.py', 'evaluation/road_comparison/road_metrics.py')}}
    if args.action == 'test':
        if config != json.loads((root/'diagnostic_config.json').read_text()):
            parser.error('Checkpoint/data/code changed since val; use the same files or re-run val in a fresh directory')
        selection = json.loads((root/'val/val_selection.json').read_text())
        fixed_threshold = selection['models']['direction_on']['threshold']
        thresholds = {name: sorted({fixed_threshold, selection['models'][name]['threshold']}) for name in VARIANTS}
    else:
        write_json(root/'diagnostic_config.json', config)
        fixed_threshold = None
        thresholds = {name: GRID for name in VARIANTS}
    # Optimizer state is never used; discard it before inference.
    checkpoint.pop('optimizer_state_dict', None)
    original_state = checkpoint.pop('model_state_dict')
    rows = export(args, model, diffusion, thresholds, fixed_threshold)
    if any(not torch.equal(tensor.cpu(), original_state[name]) for name, tensor in model.state_dict().items()):
        raise RuntimeError('Checkpoint parameters/buffers changed during diagnosis')
    manifest = build_manifest(args, checkpoint, config)
    if args.action == 'val':
        compare(args, manifest, 'val', 'val')
        selection = json.loads((root/'val/val_selection.json').read_text())
        with (root/'val/val_threshold_scores.csv').open() as stream:
            scores = list(csv.DictReader(stream))
        fixed = fixed_selection(selection, scores)
        write_json(root/'val/val_selection_fixed.json', fixed)
        fixed_threshold = fixed['models']['direction_on']['threshold']
    else:
        compare(args, manifest, 'test', 'test_fixed', root/'val/val_selection_fixed.json')
        compare(args, manifest, 'test', 'test_own_val', root/'val/val_selection.json')
        for report in ('test_fixed', 'test_own_val'):
            print(f'\n{report}:\n'+(root/report/'test_comparison.csv').read_text(), flush=True)
        metric_deltas(root)
    summarize(args, rows, fixed_threshold)


if __name__ == '__main__':
    main()
