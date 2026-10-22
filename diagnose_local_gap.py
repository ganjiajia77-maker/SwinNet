"""Paired frozen-checkpoint Gap writeback on/off; no retraining or threshold selection."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch

from config import _C
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state
from mass_protocol import MODEL_NAME, stitch_logits
from mass_h0 import verify_checkpoint
from gap_diagnostic_metrics import verified_gaps, gap_metrics, adjacent_gt_joins
from evaluation.road_comparison.road_metrics import DEFAULT_PROTOCOL, prepare_target, image_metrics, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--selection', required=True, help='Original validation val_selection.json')
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    info = verify_checkpoint(args.model_path)
    selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    if selection.get('selected_on') != 'val':
        parser.error('Threshold must be selected on validation')
    threshold = float(selection['models'][MODEL_NAME]['threshold'])
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    saved = info['args']
    config = _C.clone()
    config.merge_from_file(str(Path(__file__).parent / 'configs/swin_tiny_patch4_window7_224_lite.yaml'))
    config.DATA.IMG_SIZE = 512
    model = SwinUnet(config, img_size=512, num_classes=1, return_skeleton=True,
                    structure_profile=saved['structure_profile'],
                    enable_highres_structure_stream=True, enable_h3_surface_fusion=True,
                    stage_skeleton_mode='direct', remove_stage2_pre_topology_source=True,
                    local_gap_config=saved['local_gap_config']).to(args.device).eval()
    load_topology_checkpoint_state(model, checkpoint['model_state_dict'], checkpoint['topology_attention_version'])
    del checkpoint
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ('gap_off', 'gap_on'):
        (args.output_dir/name).mkdir()
    dataset = RoadSkeletonDataset(args.root_path, args.split, image_size=None,
                                  source_patch_size=1500, return_full_image=True)
    scale = model.swin_unet.gap_connection.writeback_scale.clone()
    protocol = dict(DEFAULT_PROTOCOL, image_size=1500)
    rows = {'gap_off': [], 'gap_on': []}
    repairs, per_image = [], []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            image = sample['image'][None].to(args.device)
            mask = sample['mask'][0].numpy() > .5
            target = prepare_target(mask, protocol)
            model.swin_unet.gap_connection.writeback_scale.zero_()
            off = stitch_logits(model, image)[0][0,0].cpu().numpy()
            model.swin_unet.gap_connection.writeback_scale.copy_(scale)
            on = stitch_logits(model, image)[0][0,0].cpu().numpy()
            if not np.isfinite(off).all() or not np.isfinite(on).all():
                raise RuntimeError('Non-finite stitched diagnostic logits')
            off_mask, on_mask = off >= np.log(threshold/(1-threshold)), on >= np.log(threshold/(1-threshold))
            key = Path(sample['image_name']).stem
            for name, prediction in (('gap_off', off_mask), ('gap_on', on_mask)):
                rows[name].append(image_metrics(prediction, target, protocol))
                success = cv2.imwrite(str(args.output_dir/name/(key+'.png')), prediction.astype(np.uint8)*255)
                if not success:
                    raise OSError('Could not write diagnostic mask')
            from evaluation.road_comparison.road_metrics import skeletonize
            gaps, _ = verified_gaps(skeletonize(mask), off_mask, 2, 20, 3)
            current = gap_metrics(gaps, on_mask, off, on, 2)
            repairs.extend(dict(image=key, **r) for r in current)
            _, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            old_joins = adjacent_gt_joins(off_mask, gt_labels, gt_stats, 20)
            new_joins = adjacent_gt_joins(on_mask, gt_labels, gt_stats, 20)
            per_image.append({'image': key, 'verified_gaps': len(gaps),
                              'closed_gaps': sum(r['complete_closure'] for r in current),
                              'connected_flanks': sum(r['connected_flanks'] for r in current),
                              'new_tp': int((on_mask & ~off_mask & mask).sum()),
                              'new_fp': int((on_mask & ~off_mask & ~mask).sum()),
                              'lost_tp': int((off_mask & ~on_mask & mask).sum()),
                              'removed_fp': int((off_mask & ~on_mask & ~mask).sum()),
                              'new_nearby_gt_joins_proxy': len(new_joins-old_joins),
                              'resolved_nearby_gt_joins_proxy': len(old_joins-new_joins)})
            print(f'[GAP-DIAG] {index+1}/{len(dataset)} {key}: gaps={len(gaps)} closed={per_image[-1]["closed_gaps"]}', flush=True)
    for name, data in (('verified_gaps.csv', repairs), ('per_image.csv', per_image)):
        if data:
            with (args.output_dir/name).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=data[0]); writer.writeheader(); writer.writerows(data)
    summary = {'protocol': protocol, 'checkpoint': args.model_path, 'split': args.split,
               'threshold': threshold, 'selection': args.selection,
               'gap_gamma': float(model.swin_unet.gap_connection.gamma),
               'saved_writeback_scale': float(scale), 'weights': 'raw; no EMA',
               'gap_off': summarize(rows['gap_off']), 'gap_on': summarize(rows['gap_on']),
               'diagnostic': {k: sum(row[k] for row in per_image) for k in per_image[0] if k != 'image'},
               'scope': 'frozen inference intervention; nearby GT joins are a screening proxy, not all false connections'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
