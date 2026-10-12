"""Verify an H0 random512 checkpoint and describe its exported masks."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


NAME_FULL = 'h0_599a410_mass1500_random512_fp32_ema80'
NAME_NO_ROAD = 'h0_599a410_no_encoder_road_prior_mass1500_random512_fp32_ema80'
NAME_A = 'h0_a_no_highres_mass1500_random512_fp32_ema80'
NAME_B = 'h0_b_static_highres_mass1500_random512_fp32_ema80'
NAME_DATA1 = 'h0_599a410_random512_fp32_ema80'
BASE = '599a410de8d341850e1810417307e4b3bbdfc82d'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--data_root', required=True, type=Path)
    p.add_argument('--predictions_root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--check_only', action='store_true')
    p.add_argument('--expected_accumulation_steps', type=int, default=2,
                   help='Expected training accumulation; use 1 for the original H0 Mass protocol')
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint.get('args', {})
    source_size = int(saved.get('source_patch_size', 0))
    if source_size not in (1024, 1500):
        p.error(f'Unsupported source_patch_size: {source_size}')
    use_ema = saved.get('use_ema')
    if type(use_ema) is not bool:
        p.error('Checkpoint must record use_ema as a boolean')
    if source_size == 1024 and not use_ema:
        p.error('The legacy Data1 comparison expects EMA weights')
    ablated = bool(saved.get('ablate_h0_stream', False))
    static = bool(saved.get('static_h0_stage_fusion', False))
    encoder_road_prior = saved.get('enable_encoder_road_prior', True)
    if type(encoder_road_prior) is not bool:
        p.error('enable_encoder_road_prior must be a boolean')
    if ablated and static:
        p.error('A and B ablation flags cannot be combined')
    if source_size == 1024 and (ablated or static):
        p.error('A and B ablations are only configured for native-1500 Massachusetts data')
    if source_size == 1500 and not encoder_road_prior and (ablated or static):
        p.error('The encoder road prior ablation requires the full H0 model')
    name = (
        NAME_A if ablated else NAME_B if static else
        NAME_NO_ROAD if not encoder_road_prior else NAME_FULL
    ) if source_size == 1500 else NAME_DATA1
    if source_size == 1500 and not use_ema:
        name = name.removesuffix('ema80') + 'noema80'
    expected = dict(img_size=512, source_patch_size=source_size, random_crop_train=True,
                    random_crops_per_image=1, direct_resize_train=False,
                    amp_dtype='none', use_ema=use_ema, ema_decay=0.999, max_epochs=80,
                    structure_profile='stage23_boundary_0626',
                    enable_highres_structure_stream=not ablated, enable_global_topology=True,
                    stage_skeleton_mode='direct', enable_h3_surface_fusion=True,
                    remove_stage2_pre_topology_source=True)
    if source_size == 1500:
        expected.update(ablate_h0_stream=ablated, batch_size=2,
                        accumulation_steps=args.expected_accumulation_steps, overlap_stride=256,
                        road_attention_weight=0.0)
        if not encoder_road_prior:
            expected.update(enable_encoder_road_prior=False, static_h0_stage_fusion=False,
                            threshold=0.5, trend_val_every_steps=0, val_interval=5,
                            base_lr=5e-4, pretrained_lr=5e-5, new_lr=2e-4,
                            stage_skeleton_loss_factor=1.0,
                            stage_skeleton_only_loss_factor=1.0)
        if 'enable_encoder_road_prior' in saved:
            expected['enable_encoder_road_prior'] = encoder_road_prior
        if static:
            expected['static_h0_stage_fusion'] = True
        if not isinstance(saved.get('seed'), int):
            p.error('Massachusetts checkpoint must record its random seed')
    mismatches = {key: [saved.get(key), value] for key, value in expected.items()
                  if saved.get(key) != value}
    ema_present = checkpoint.get('ema_state_dict') is not None
    if mismatches or ema_present != use_ema:
        p.error(f'Unexpected checkpoint args: {mismatches}; '
                f'EMA present={ema_present}; use_ema={use_ema}')
    epoch = checkpoint.get('epoch')
    del checkpoint
    print(f'Verified {name} checkpoint, saved epoch={epoch}', flush=True)
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
        if selection.get('selected_on') != 'val' or name not in selection.get('models', {}):
            p.error('Selection is not a validation selection for this H0 model')
        threshold = float(selection['models'][name]['threshold'])
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
                    protocol=dict(image_size=source_size, short_area_threshold=20,
                                  apls_max_nodes=64, apls_snap_radius=5.0),
                    models=[dict(name=name, provenance=dict(
                        checkpoint=str(args.checkpoint), checkpoint_epoch=epoch,
                        checkpoint_args_verified=True, base_model_commit=BASE,
                        ablate_h0_stream=ablated,
                        enable_encoder_road_prior=encoder_road_prior,
                        static_h0_stage_fusion=static,
                        seed=saved.get('seed'),
                        global_topology_structure_source=(
                            'projected Stage 3 G3' if ablated else
                            'initial high-resolution H0' if static else
                            'updated high-resolution H2'
                        ),
                        training_crop=f'one random512 crop per native{source_size} image each epoch; shared epoch',
                        crop_schedule=(
                            'seeded non-repeating coordinate permutation, matching Mass H0'
                            if source_size == 1500 else 'epoch-seeded random crop'
                        ),
                        physical_batch_size=saved.get('batch_size'),
                        accumulation_steps=saved.get('accumulation_steps'),
                        training_precision='FP32', training_max_epochs=80,
                        weights='EMA' if use_ema else 'model_state_dict',
                        ema_decay=0.999 if use_ema else None,
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
