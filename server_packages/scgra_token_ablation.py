"""Fixed-checkpoint ablation of SCGRA anchor projection inputs on Massachusetts."""

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


MODES = {'full': (), 'no_h': ('h',), 'no_c': ('c',), 'no_hc': ('h', 'c')}
CHECKPOINT_SHA = '75145caf960def5eddc56ea622ab1fe79459d63f9a484bf75f7a7b68ce815c17'
PROTOCOL = dict(image_size=1500, short_area_threshold=20, topo_seed_density_deg=.002,
                topo_step_deg=.0002, apls_max_control_nodes=150, apls_snap_radius=5.,
                apls_control_sampling_seed=1234, apls_min_path_px=10., tlts_n_paths=200, tlts_seed=1234)
DELTAS = ('mean_signed_probability_delta', 'mean_signed_gt_road_delta',
          'mean_signed_gt_background_delta', 'max_abs_probability_delta', 'changed_pixels')


def infer(model, image):
    import torch
    from mass_protocol import stitch_logits

    module = model.swin_unet.global_topology
    extract = module._extract_fps_anchors
    bias_methods = ('_local_topology_bias', '_relative_topology_bias')
    original_biases = {name: getattr(module, name) for name in bias_methods}
    anchors, tokens = [], []
    biases = {name: [] for name in bias_methods}
    h_end = module.struct_channels
    c_start = h_end + module.channels
    channels = {'h': slice(0, h_end), 'c': slice(c_start, c_start + module.connectivity_channels)}
    if module.node_projection.in_features != channels['c'].stop + module.node_type_embedding.embedding_dim + 2:
        raise RuntimeError('Unexpected SCGRA token layout')

    def run():
        logits, _ = stitch_logits(lambda tile: model(tile)[0], image)
        return logits.sigmoid()[0, 0].cpu().numpy()

    def capture_anchors(score):
        result = extract(score)
        coords, valid, scores, count = result
        anchors.append((coords.clone(), valid.clone(), scores.clone(), count))
        return result

    def capture_tokens(_projection, args):
        tokens.append(args[0].clone())

    def capture_bias(name):
        def capture(*args):
            value = original_biases[name](*args)
            biases[name].append(value.clone())
            return value
        return capture

    with torch.inference_mode(), ExitStack() as stack:
        stack.enter_context(patch.object(module, '_extract_fps_anchors', capture_anchors))
        for name in bias_methods:
            stack.enter_context(patch.object(module, name, capture_bias(name)))
        stack.callback(module.node_projection.register_forward_pre_hook(capture_tokens).remove)
        predictions = {'full': run()}

    runs = [anchors, tokens, *biases.values()]
    if any(len(values) != 25 for values in runs):
        raise RuntimeError(f'Expected one SCGRA call per patch for all 25 patches, got {[len(v) for v in runs]}')
    audit = dict(windows=len(anchors), valid_anchors=sum(int(record[1].sum()) for record in anchors),
                 verified_interventions={})
    for mode, cleared in MODES.items():
        if not cleared:
            continue
        anchor_items, token_items = iter(anchors), iter(tokens)
        bias_items = {name: iter(items) for name, items in biases.items()}

        def zero_inputs(_projection, args):
            original = args[0]
            if not torch.equal(original, next(token_items)):
                raise RuntimeError(f'Upstream token inputs differ from Full in {mode}')
            masked = original.clone()
            for part in cleared:
                masked[..., channels[part]] = 0
            return (masked,)

        def check_bias(name):
            def check(*args):
                value = original_biases[name](*args)
                if not torch.equal(value, next(bias_items[name])):
                    raise RuntimeError(f'{name} differs from Full in {mode}')
                return value
            return check

        with torch.inference_mode(), ExitStack() as stack:
            stack.enter_context(patch.object(module, '_extract_fps_anchors', lambda _score: next(anchor_items)))
            for name in bias_methods:
                stack.enter_context(patch.object(module, name, check_bias(name)))
            stack.callback(module.node_projection.register_forward_pre_hook(zero_inputs).remove)
            predictions[mode] = run()
        if any(next(items, None) is not None for items in (anchor_items, token_items, *bias_items.values())):
            raise RuntimeError(f'{mode} skipped a Full SCGRA window')
        audit['verified_interventions'][mode] = len(anchors)
    return predictions, audit


def infer_split(args):
    sys.path.insert(0, str(args.repo.resolve(strict=True)))
    import cv2
    import torch
    from diagnostics import scgra_decoder_mass as mass
    from scripts.diagnose_scgra_evidence import aggregate_pixels, pixel_row
    from scripts.export_scgra_decoder_mass import dump, load, save_png, source_hashes, write_csv

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check the selected GPU and Python environment.')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = args.checkpoint.resolve(strict=True)
    sha = mass.file_digest(checkpoint)
    if sha != CHECKPOINT_SHA:
        raise ValueError('Expected the original decoder-control checkpoint used for the SCGRA test49 analyses')
    selection, _, threshold = mass.select_validation_record(checkpoint, sha, args.selection)
    if abs(threshold - .45) > 1e-7:
        raise ValueError('Use the original Full val-selected threshold of 0.45')
    root = args.root_path.resolve(strict=True)
    data = mass.dataset(root, args.split)
    cases = [Path(name).stem for name in data.image_files]
    if len(set(cases)) != len(cases):
        raise ValueError(f'Duplicate Massachusetts {args.split} image identifiers')
    config = dict(split=args.split, case_ids=cases, data_root=str(root), checkpoint=str(checkpoint),
                  checkpoint_sha256=sha, threshold=threshold, threshold_source=str(selection),
                  threshold_sha256=mass.file_digest(selection), modes={m:list(s) for m,s in MODES.items()},
                  tile_size=512, stride=256, anchor_policy='replay Full selector outputs per patch',
                  bias_policy='preserve and compare both explicit biases per patch',
                  source_sha256={**source_hashes(), 'token_ablation':mass.file_digest(__file__)})
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = out/'metadata.json'
    if meta.is_file() and load(meta) != config:
        raise ValueError('Existing output uses different settings. Choose a fresh output directory.')
    if not meta.is_file():
        if any(out.iterdir()):
            raise ValueError('Use a fresh output directory')
        dump(meta, config)
    (out/'partial').mkdir(exist_ok=True)
    for mode in MODES:
        (out/'masks'/mode).mkdir(parents=True, exist_ok=True)

    model, _, epoch = mass.load_model(checkpoint, device)
    before = mass.tensor_digests(model)
    rows, audits = [], []
    for i, case in enumerate(cases):
        progress = out/'partial'/f'{case}.json'
        inputs = [mass.file_digest(Path(folder)/name) for folder, name in
                  ((data.image_dir, data.image_files[i]), (data.mask_dir, data.label_files[i]))]
        record = load(progress) if progress.is_file() else None
        if record is not None and record['inputs'] != inputs:
            raise ValueError(f'Input image or label changed: {case}')
        cached = record is not None and all((out/name).is_file() and mass.file_digest(out/name) == digest
                                            for name, digest in record['masks'].items())
        if not cached:
            _, image, gt = mass.read_sample(data, i)
            probabilities, audit = infer(model, image[None].to(device))
            full = probabilities['full']
            full_mask = full >= threshold
            record = dict(inputs=inputs, audit=audit, masks={}, rows=[])
            for mode, probability in probabilities.items():
                path = Path('masks')/mode/f'{case}.png'
                mask = probability >= threshold
                save_png(out/path, mask.astype(np.uint8)*255)
                record['masks'][path.as_posix()] = mass.file_digest(out/path)
                delta = probability - full
                row = pixel_row(case, mode, threshold, probability, full, gt, full_mask)
                row.update(mean_signed_probability_delta=float(delta.mean()),
                           mean_signed_gt_road_delta=float(delta[gt].mean()) if gt.any() else None,
                           mean_signed_gt_background_delta=float(delta[~gt].mean()) if (~gt).any() else None,
                           max_abs_probability_delta=float(np.abs(delta).max()),
                           changed_pixels=int((mask != full_mask).sum()))
                record['rows'].append(row)
            dump(progress, record)
        audits.append(dict(case_id=case, **record['audit']))
        rows.extend(record['rows'])
        print(f'[SCGRA token {args.split}] {i+1}/{len(cases)} {case} cached={cached}', flush=True)

    if before != mass.tensor_digests(model) or mass.file_digest(checkpoint) != sha:
        raise RuntimeError('Model weights or checkpoint were modified')
    summary = aggregate_pixels(rows, [])
    for item in summary:
        group = [row for row in rows if row['mode'] == item['mode']]
        for field in DELTAS:
            values = [row[field] for row in group if row[field] is not None]
            if field == 'changed_pixels':
                item[field] = sum(values)
            elif field == 'max_abs_probability_delta':
                item[field] = max(values)
            else:
                item[field] = float(np.mean(values)) if values else None
    write_csv(out/'segmentation_per_image.csv', rows)
    write_csv(out/'segmentation_summary.csv', summary)
    dump(out/'anchor_bias_audit.json', audits)
    models = [dict(name=mode,
                   provenance=dict(checkpoint_sha256=sha, checkpoint_epoch=epoch, zeroed_token_fields=list(parts)),
                   validation_selection=dict(selected_on='val', criterion='global_iou', threshold=threshold,
                                             evidence=str(selection)),
                   predictions={args.split:[dict(path=str(out/'masks'/mode), threshold=threshold)]})
              for mode, parts in MODES.items()]
    dump(out/'comparison.json', dict(data_root=str(root), protocol=PROTOCOL, models=models))
    print(f'[INFERENCE DONE] {out}', flush=True)


def evaluate_split(args):
    sys.path.insert(0, str(args.tool.resolve(strict=True)))
    import compare_predictions as cp

    out = args.output_dir.resolve(strict=True)
    metadata = json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    split = metadata['split']
    if split not in ('val', 'test'):
        raise ValueError(f'Unsupported inference split: {split}')
    manifest = json.loads((out/'comparison.json').read_text(encoding='utf-8'))
    protocol = cp.validate_manifest(manifest)
    selection = cp.import_declared_selection(manifest, out, protocol)
    topology = out/'topology'
    topology.mkdir(exist_ok=True)
    cp.write_json(topology/'imported_val_selection.json', selection)
    report = cp.evaluate(manifest, out, protocol, split, selection, topology, 1,
                         workers=args.workers, resume=True)
    metrics = {row['name']:row for row in report['summaries']}
    with (out/'segmentation_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        values = metrics[row['mode']]
        if any(int(row[k]) != values[k] for k in ('images', 'tp', 'fp', 'fn')):
            raise ValueError(f'Inference and evaluator pixel counts disagree for {row["mode"]}')
        row.update(topo_recall=values['topo_recall'], apls_bidirectional=values['apls_bidirectional'])
    cp.write_csv(out/'ablation_summary.csv', rows)
    print(f'[DONE] {out / "ablation_summary.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest='phase', required=True)
    inference = phases.add_parser('infer')
    for key in ('repo', 'checkpoint', 'selection', 'root_path', 'output_dir'):
        inference.add_argument('--'+key, type=Path, required=True)
    inference.add_argument('--split', choices=('val', 'test'), default='val',
                           help='Inference split. Both use the original Full validation threshold.')
    inference.add_argument('--device', default='cuda:0')
    inference.set_defaults(run=infer_split)
    topology = phases.add_parser('topology')
    topology.add_argument('--tool', type=Path, required=True)
    topology.add_argument('--output_dir', type=Path, required=True)
    topology.add_argument('--workers', type=int, default=4)
    topology.set_defaults(run=evaluate_split)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
