"""Verify the H0 random512 EMA checkpoint and describe its exported masks."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


NAME = 'h0_599a410_random512_fp32_ema80'
BASE = '599a410de8d341850e1810417307e4b3bbdfc82d'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--data_root', required=True, type=Path)
    p.add_argument('--predictions_root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--check_only', action='store_true')
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint.get('args', {})
    expected = dict(img_size=512, source_patch_size=1024, random_crop_train=True,
                    random_crops_per_image=1, direct_resize_train=False,
                    amp_dtype='none', use_ema=True, ema_decay=0.999, max_epochs=80,
                    structure_profile='stage23_boundary_0626',
                    enable_highres_structure_stream=True, enable_global_topology=True,
                    stage_skeleton_mode='direct', enable_h3_surface_fusion=True,
                    remove_stage2_pre_topology_source=True)
    mismatches = {key: [saved.get(key), value] for key, value in expected.items()
                  if saved.get(key) != value}
    if mismatches or checkpoint.get('ema_state_dict') is None:
        p.error(f'Unexpected checkpoint args: {mismatches}; '
                f'EMA present={checkpoint.get("ema_state_dict") is not None}')
    epoch = checkpoint.get('epoch')
    del checkpoint
    print(f'Verified H0 random512 FP32 EMA 80e checkpoint, saved epoch={epoch}', flush=True)
    if args.check_only:
        return
    candidates = []
    for directory in sorted((args.predictions_root / 'val').iterdir()):
        match = re.fullmatch(r'threshold_(\d+\.\d{2})', directory.name)
        if match and (directory / 'surface').is_dir():
            candidates.append(dict(threshold=float(match.group(1)), path=str(directory / 'surface')))
    if not candidates:
        p.error('No exported validation threshold masks')
    test = []
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding='utf-8'))
        if selection.get('selected_on') != 'val' or NAME not in selection.get('models', {}):
            p.error('Selection is not a validation selection for this H0 model')
        threshold = float(selection['models'][NAME]['threshold'])
        directory = args.predictions_root / 'test' / f'threshold_{threshold:.2f}' / 'surface'
        if not directory.is_dir():
            p.error(f'Missing selected test masks: {directory}')
        test = [dict(threshold=threshold, path=str(directory))]
    root = Path(__file__).resolve().parents[2]
    source_files = ['train_image.py', 'test_image.py', 'datasets/dataset_road_skeleton.py',
                    'networks/vision_transformer.py',
                    'networks/swin_transformer_unet_skip_expand_decoder_sys.py',
                    'networks/skeleton_guided_head.py', 'losses/road_losses.py',
                    'evaluation/road_comparison/compare_predictions.py',
                    'evaluation/road_comparison/road_metrics.py']
    manifest = dict(data_root=str(args.data_root),
                    protocol=dict(image_size=1024, short_area_threshold=20,
                                  apls_max_nodes=64, apls_snap_radius=5.0),
                    models=[dict(name=NAME, provenance=dict(
                        checkpoint=str(args.checkpoint), checkpoint_epoch=epoch,
                        checkpoint_args_verified=True, base_model_commit=BASE,
                        training_crop='one random512 crop per native1024 image each epoch; shared epoch',
                        training_precision='FP32', training_max_epochs=80,
                        weights='EMA', ema_decay=0.999,
                        inference_tile_size=512, overlap_stride=256,
                        merge='taper-weighted logits, divisor clamp1e-8, sigmoid',
                        tta='none', postprocessing='surface threshold only',
                        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                       for name in source_files}),
                        predictions=dict(val=candidates, test=test))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}: {len(candidates)} val thresholds, {len(test)} test threshold')


if __name__ == '__main__':
    main()
