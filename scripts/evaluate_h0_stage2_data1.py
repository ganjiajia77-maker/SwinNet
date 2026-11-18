"""Data1 Stage2-global EMA evaluation: validation sweep, fixed test, six metrics."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_h0_encoder_phase as common

NAME = 'data1_h0_global_stage2_random512_fp32_ema80'
SIZE = 1024
EXPECTED = {'val': 500, 'test': 1246}
THRESHOLDS = common.THRESHOLDS
PROTOCOL = dict(image_size=SIZE, short_area_threshold=20,
    topo_seed_density_deg=.002, topo_step_deg=.0002, apls_max_control_nodes=150,
    apls_control_sampling_seed=1234, apls_snap_radius=5., apls_min_path_px=10.,
    tlts_n_paths=200, tlts_seed=1234)


def threshold_grid(lower=.20, upper=.65, step=.01):
    lo, hi, stride = (Decimal(str(value)) for value in (lower, upper, step))
    if not all(v.is_finite() for v in (lo, hi, stride)):
        raise ValueError('Threshold range and step must be finite')
    if not (0 <= lo <= hi <= 1 and stride > 0):
        raise ValueError('Require 0 <= threshold_min <= threshold_max <= 1 and step > 0')
    count = int((hi-lo)//stride)+1
    if count > 100001:
        raise ValueError('Threshold grid is too dense; use at most 100001 values')
    values = [lo+i*stride for i in range(count)]
    if values[-1] != hi:
        values.append(hi)
    grid = np.array([float(value) for value in values], dtype=np.float32)
    if len(np.unique(grid)) != len(grid):
        raise ValueError('Threshold step is smaller than float32 resolution')
    return grid


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = state['args']
    if not (saved.get('h0_data1') and saved.get('use_ema') and
            saved.get('source_patch_size') == SIZE and saved.get('img_size') == 512 and
            saved.get('amp_dtype') == 'none'):
        raise ValueError('Expected the original Data1 FP32/EMA Stage2-global checkpoint')
    weights = state.get('ema_state_dict')
    evaluation = state.get('model_state_dict')
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Checkpoint has no explicit EMA weights')
    if (not isinstance(evaluation, dict) or weights.keys() != evaluation.keys() or
            any(not torch.equal(weights[k], evaluation[k]) for k in weights)):
        raise ValueError('Saved evaluation weights differ from EMA; refusing implicit weight selection')
    epoch = state.get('epoch')
    del state
    # The shared factory's EMA-off guard concerns its old Mass entry point only;
    # EMA does not alter the architecture. All architecture flags are inherited.
    model = common.build_model(dict(saved, use_ema=False))
    common.load_stage2_weights(model, weights)
    del weights, evaluation
    model.requires_grad_(False)
    print(f'[RESTORE] strict=True; weights=ema_state_dict; epoch={epoch}', flush=True)
    return model.to(device).eval(), saved, epoch


def dataset(root, split):
    data = common.RoadSkeletonDataset(str(root), split, image_size=None,
        source_patch_size=SIZE, augment=False, return_full_image=True, original_data1=True)
    if len(data) != EXPECTED[split]:
        raise ValueError(f'Expected {EXPECTED[split]} {split} images, found {len(data)}')
    return data


def read_sample(data, index):
    # Same RGB/ImageNet normalization as training, without unneeded skeleton GT.
    name = data.image_files[index]
    image = cv2.imread(str(Path(data.image_dir)/name), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(Path(data.mask_dir)/data.label_files[index]), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise IOError(f'Cannot read image/GT: {name}')
    if image.shape != (SIZE, SIZE, 3) or mask.shape != (SIZE, SIZE):
        raise ValueError(f'Expected native {SIZE} image and GT: {name}')
    if not np.isin(np.unique(mask), (0, 255)).all():
        raise ValueError(f'Expected original Data1 0/255 GT: {name}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    image = (image-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    case = Path(name).stem.replace('_sat', '')
    return case, torch.from_numpy(image.transpose(2, 0, 1).copy()), mask > 127


@torch.inference_mode()
def infer(model, image, device):
    image = image[None].to(device)
    canvas = torch.zeros((1, 1, SIZE, SIZE), device=device)
    divisor = torch.zeros_like(canvas)
    weight = common.blend_weight(device)
    for top in common.positions(SIZE):
        for left in common.positions(SIZE):
            outputs = model(image[:, :, top:top+512, left:left+512])
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if logits.shape != (1, 1, 512, 512) or not torch.isfinite(logits).all():
                raise ValueError('Non-finite or incorrectly sized surface logits')
            canvas[:, :, top:top+512, left:left+512] += logits.float()*weight
            divisor[:, :, top:top+512, left:left+512] += weight
    if not (divisor > 0).all():
        raise RuntimeError('Sliding windows left uncovered pixels')
    return (canvas/divisor.clamp_min(1e-8)).sigmoid()[0, 0].cpu().numpy()


def sweep_counts(probability, gt, thresholds=THRESHOLDS):
    # Exactly counts probability >= threshold, including equality, without
    # materializing 91 separate masks or storing floating-point prediction maps.
    bins = np.searchsorted(thresholds, probability, side='right')
    pos = np.bincount(bins[gt], minlength=len(thresholds)+1)
    neg = np.bincount(bins[~gt], minlength=len(thresholds)+1)
    tp = np.cumsum(pos[::-1], dtype=np.int64)[::-1][1:]
    fp = np.cumsum(neg[::-1], dtype=np.int64)[::-1][1:]
    return np.stack((tp, fp, gt.sum()-tp), axis=1)


def run_identity(args, thresholds=None):
    if thresholds is None:
        record = args.output_dir/'inference_protocol.json'
        thresholds = (json.loads(record.read_text(encoding='utf-8'))['thresholds']
                      if record.is_file() else THRESHOLDS)
    return dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=common.file_digest(args.checkpoint),
        data_root=str(args.root_path.resolve()), image_size=SIZE,
        window=512, stride=256, ramp_min=.1, stitch='weighted logits then sigmoid',
        precision='FP32', weights='ema_state_dict', TTA=False,
        global_topology_location='after_decoder_stage2', topology=PROTOCOL,
        thresholds=[float(t) for t in thresholds])


def require_same(path, expected):
    if json.loads(path.read_text(encoding='utf-8')) != expected:
        raise ValueError(f'Existing output belongs to another evaluation: {path}')


def validation(args):
    out = args.output_dir
    thresholds = threshold_grid(getattr(args, 'threshold_min', .20),
        getattr(args, 'threshold_max', .65), getattr(args, 'threshold_step', .01))
    identity = run_identity(args, thresholds)
    out.mkdir(parents=True, exist_ok=True)
    record = out/'inference_protocol.json'
    if record.is_file():
        require_same(record, identity)
    elif any(out.iterdir()):
        raise ValueError('Use a new evaluation output directory')
    else:
        dump(record, identity)
    if (out/'val_complete.json').is_file():
        print('[VAL] completed previously; reusing saved selection', flush=True)
        return
    model, _, epoch = load_model(args.checkpoint, torch.device(args.device))
    data = dataset(args.root_path, 'val')
    print(f'[SWEEP] {thresholds[0]:.6f}..{thresholds[-1]:.6f}; {len(thresholds)} thresholds; one inference per image', flush=True)
    total = np.zeros((len(thresholds), 3), np.int64)
    seen = set()
    for i in range(len(data)):
        case, image, gt = read_sample(data, i)
        if case in seen:
            raise ValueError(f'Duplicate case: {case}')
        seen.add(case)
        total += sweep_counts(infer(model, image, torch.device(args.device)), gt, thresholds)
        print(f'[VAL] {i+1}/{len(data)} {case}', flush=True)
    curve = [dict(threshold=float(t), tp=int(c[0]), fp=int(c[1]), fn=int(c[2]),
                  **common.metrics(*map(int, c))) for t, c in zip(thresholds, total)]
    common.write_csv(out/'val_threshold_scores.csv', curve)
    selected = common.select_threshold(curve)
    dump(out/'val_selection.json', dict(selected_on='val', criterion='global_iou',
        checkpoint_sha256=identity['checkpoint_sha256'], checkpoint_epoch=epoch,
        eval_weights='ema', val_images=len(data),
        search_range=[float(thresholds[0]), float(thresholds[-1])],
        at_search_boundary=selected['threshold'] in (float(thresholds[0]), float(thresholds[-1])),
        **selected))
    dump(out/'val_complete.json', dict(complete=True, images=len(data)))
    print('[VAL SELECTED]', selected, flush=True)
    if selected['threshold'] in (float(thresholds[0]), float(thresholds[-1])):
        print('[SWEEP] Best threshold is at the search boundary; expand the range to check outside it.', flush=True)


def testing(args):
    out = args.output_dir
    identity = run_identity(args)
    require_same(out/'inference_protocol.json', identity)
    completed = json.loads((out/'val_complete.json').read_text())
    selection = json.loads((out/'val_selection.json').read_text())
    if (completed != dict(complete=True, images=EXPECTED['val']) or
            selection.get('selected_on') != 'val' or
            selection.get('checkpoint_sha256') != identity['checkpoint_sha256']):
        raise ValueError('Run the matching validation sweep first')
    threshold = selection['threshold']
    test_identity = dict(identity, threshold=threshold)
    record = out/'test_protocol.json'
    if record.is_file():
        require_same(record, test_identity)
    else:
        dump(record, test_identity)
    masks = out/'test_masks'
    masks.mkdir(exist_ok=True)
    data = dataset(args.root_path, 'test')
    model = None
    rows, seen = [], set()
    for i in range(len(data)):
        case, image, gt = read_sample(data, i)
        if case in seen:
            raise ValueError(f'Duplicate case: {case}')
        seen.add(case)
        path = masks/(case+'.png')
        pred = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if pred is None:
            if model is None:
                model, _, _ = load_model(args.checkpoint, torch.device(args.device))
            pred = infer(model, image, torch.device(args.device)) >= threshold
            common.save_mask(path, pred)
        else:
            if pred.shape != (SIZE, SIZE) or not np.isin(np.unique(pred), (0, 255)).all():
                raise ValueError(f'Invalid existing mask: {path}')
            pred = pred > 0
        tp, fp, fn = common.counts(pred, gt)
        rows.append(dict(case=case, threshold=threshold, tp=tp, fp=fp, fn=fn,
                         **common.metrics(tp, fp, fn)))
        print(f'[TEST] {i+1}/{len(data)} {case}', flush=True)
    if {p.stem for p in masks.glob('*.png')} != seen:
        raise ValueError('Test output contains extra image IDs')
    common.write_csv(out/'test_per_image.csv', rows)
    total = [sum(row[k] for row in rows) for k in ('tp', 'fp', 'fn')]
    common.write_csv(out/'test_segmentation.csv', [dict(name=NAME, threshold=threshold,
        images=len(rows), **dict(zip(('tp', 'fp', 'fn'), total)), **common.metrics(*total))])
    manifest = dict(data_root=str(args.root_path.resolve()), protocol=PROTOCOL,
        models=[dict(name=NAME, provenance=identity,
            validation_selection=dict(selected_on='val', criterion='global_iou',
                threshold=threshold, evidence=str((out/'val_selection.json').resolve())),
            predictions=dict(test=[dict(path=str(masks.resolve()), threshold=threshold)]))])
    dump(out/'comparison.json', manifest)
    dump(out/'test_complete.json', dict(complete=True, images=len(rows), threshold=threshold))
    print('[TEST DONE]', out/'test_segmentation.csv', flush=True)


def topology(args):
    out = args.output_dir
    complete = json.loads((out/'test_complete.json').read_text())
    if complete.get('images') != EXPECTED['test'] or not complete.get('complete'):
        raise ValueError('Finish test inference first')
    command = [sys.executable, '-u', str(ROOT/'evaluation/road_comparison/compare_predictions.py'),
        '--manifest', str(out/'comparison.json'), '--split', 'test', '--import_val_selection',
        '--output_dir', str(out/'topology'), '--workers', str(args.workers), '--progress_every', '1']
    if (out/'topology').is_dir() and any((out/'topology').iterdir()):
        command.append('--resume')
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', choices=('val', 'test', 'topology'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threshold_min', type=float, default=.20, help='Validation sweep lower bound')
    parser.add_argument('--threshold_max', type=float, default=.65, help='Validation sweep upper bound')
    parser.add_argument('--threshold_step', type=float, default=.01, help='Validation sweep interval')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    {'val': validation, 'test': testing, 'topology': topology}[args.command](args)


if __name__ == '__main__':
    main()
