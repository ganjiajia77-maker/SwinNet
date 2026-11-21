"""Measure SCGRA logit scales and export matched full/ablation test masks."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics import scgra_decoder_mass as mass
from diagnostics.scgra_relation_stats import window_statistics
from scripts.diagnose_scgra_evidence import aggregate_pixels, infer, pixel_row, summarize_attention
from scripts.export_scgra_decoder_mass import dump, load, save_png, source_hashes, write_csv


MODES = ('full', 'no_connectivity_bias', 'no_spatial_bias')
PROTOCOL = dict(image_size=1500, short_area_threshold=20, topo_seed_density_deg=.002,
                topo_step_deg=.0002, apls_max_control_nodes=150, apls_snap_radius=5.,
                apls_control_sampling_seed=1234, apls_min_path_px=10., tlts_n_paths=200, tlts_seed=1234)


def scale_summary(records):
    fields = ('qk_row_centered_rms','conn_row_centered_rms','pos_row_centered_rms',
              'qk_top1_margin_median','base_normalized_entropy','conn_normalized_entropy',
              'pos_normalized_entropy','full_normalized_entropy')
    result = []
    for key in fields:
        values = [row[key] for row in records]
        result.append(dict(metric=key, mean=float(np.mean(values)), median=float(np.median(values)),
                           min=float(np.min(values)), max=float(np.max(values)), window_heads=len(values)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--root_path', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_images', type=int, default=0, help='0 evaluates all 49; positive values are pilot only')
    args = parser.parse_args()
    if not 0 <= args.max_images <= 49:
        parser.error('--max_images must be 0–49')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; check CUDA_VISIBLE_DEVICES')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    checkpoint = args.checkpoint.resolve(strict=True)
    data_root = args.root_path.resolve(strict=True)
    output = args.output_dir.resolve()
    checkpoint_sha = mass.file_digest(checkpoint)
    selection_path, _, threshold = mass.select_validation_record(checkpoint, checkpoint_sha, args.selection)
    data = mass.dataset(data_root, 'test')
    cases = [Path(name).stem for name in data.image_files][:args.max_images or None]
    if len(set(cases)) != len(cases):
        raise ValueError('Duplicate test case identities')
    metadata = dict(split='test', image_ids=cases, checkpoint=str(checkpoint),
                    checkpoint_sha256=checkpoint_sha, threshold=threshold,
                    selection=str(selection_path), selection_sha256=mass.file_digest(selection_path),
                    data_root=str(data_root), device=str(device), modes=list(MODES), tile=512, stride=256,
                    topology_protocol=PROTOCOL, source_sha256={**source_hashes(), **{
                        name:mass.file_digest(ROOT/name) for name in (
                            'scripts/benchmark_scgra_relation_bias.py',
                            'scripts/diagnose_scgra_evidence.py', 'diagnostics/scgra_relation_stats.py')}})
    output.mkdir(parents=True,exist_ok=True)
    meta_path = output/'metadata.json'
    if meta_path.is_file():
        if load(meta_path) != metadata:
            parser.error('Existing results use different settings; select a new output directory')
    elif any(output.iterdir()):
        parser.error('Use a fresh output directory')
    dump(meta_path,metadata)
    (output/'partial').mkdir(exist_ok=True)
    for mode in MODES:
        (output/'masks'/mode).mkdir(parents=True,exist_ok=True)
    model, saved, epoch = mass.load_model(checkpoint,device)
    if saved.get('freeze_scgra_off'):
        parser.error('Use the original trained full SCGRA checkpoint')
    state_before = mass.tensor_digests(model)

    pixels, attention, scales = [], [], []
    for i,case in enumerate(cases):
        partial_path = output/'partial'/f'{case}.json'
        inputs = {key:mass.file_digest(Path(folder)/name) for key,folder,name in (
            ('image',data.image_dir,data.image_files[i]), ('gt',data.mask_dir,data.label_files[i]))}
        record = load(partial_path) if partial_path.is_file() else None
        if record is not None and record['inputs'] != inputs:
            raise ValueError(f'Test input changed: {case}')
        if record is None or not all((output/file).is_file() and mass.file_digest(output/file)==fingerprint
                                     for file,fingerprint in record['masks'].items()):
            actual, img, gt = mass.read_sample(data,i)
            if actual != case:
                raise ValueError(f'Unexpected case identity: {actual}')
            img = img[None].to(device)
            record = dict(inputs=inputs, pixels=[], attention=[], scales=[], masks={})
            window = 0

            def collect_terms(terms):
                nonlocal window
                record['scales'].extend(dict(case_id=case,window=window,**row) for row in window_statistics(terms))
                window += 1

            full, reference = infer(model,img,'full',on_relation_terms=collect_terms)
            if window != 25:
                raise RuntimeError(f'Expected 25 windows, captured {window}')
            full_mask = full >= threshold
            for mode in MODES:
                probability, trace = (full,reference) if mode == 'full' else infer(model,img,mode)
                mask_path = output/'masks'/mode/f'{case}.png'
                save_png(mask_path,(probability >= threshold).astype(np.uint8)*255)
                record['masks'][str(mask_path.relative_to(output))] = mass.file_digest(mask_path)
                record['pixels'].append(pixel_row(case,mode,threshold,probability,full,gt,full_mask))
                record['attention'].extend(dict(case_id=case,mode=mode,**row)
                                           for row in summarize_attention(trace,reference))
            dump(partial_path,record)
        pixels.extend(record['pixels'])
        attention.extend(record['attention'])
        scales.extend(record['scales'])
        print(f'[SCGRA] {i+1}/{len(cases)} {case}',flush=True)

    if state_before != mass.tensor_digests(model) or mass.file_digest(checkpoint) != checkpoint_sha:
        raise RuntimeError('The checkpoint or model state changed during the interventions')
    write_csv(output/'segmentation_per_image.csv',pixels)
    write_csv(output/'segmentation_summary.csv',aggregate_pixels(pixels,attention))
    write_csv(output/'attention_per_window_head.csv',attention)
    write_csv(output/'scale_per_window_head.csv',scales)
    if scales:
        write_csv(output/'scale_summary.csv',scale_summary(scales))
    if len(cases) == 49:
        models = [{
            'name':mode,
            'provenance':dict(checkpoint_sha256=checkpoint_sha, checkpoint_epoch=epoch, intervention=mode,
                              threshold_policy='shared original full-model validation threshold'),
            'validation_selection':dict(selected_on='val',criterion='global_iou',
                                        threshold=threshold,evidence=str(selection_path)),
            'predictions':dict(val=[],test=[dict(threshold=threshold,path=str(output/'masks'/mode))]),
        } for mode in MODES]
        dump(output/'comparison.json',dict(data_root=str(data_root),protocol=PROTOCOL,models=models))
    dump(output/'export_complete.json',dict(images=len(cases), checkpoint_epoch=epoch, threshold=threshold))
    print(f'[DONE] {output}',flush=True)


if __name__ == '__main__':
    main()
