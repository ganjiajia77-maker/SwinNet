"""Compare exact final samples/labels and CPU pipeline throughput; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from datasets.native_relation_batch import legacy_native_batch, reduce_native_stats, TARGET_KEYS


def fingerprint(batch, targets, stats):
    result = {}
    for key in ('image', 'mask', 'skeleton', 'skeleton_dilate', 'tile_top', 'tile_left'):
        result[key] = hashlib.sha256(batch[key].contiguous().numpy().tobytes()).hexdigest()
    for key in TARGET_KEYS:
        result['native_'+key] = hashlib.sha256(targets[key].contiguous().numpy().tobytes()).hexdigest()
    result['cases'] = list(batch['case_name'])
    result['statistics'] = stats
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root_path', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--epochs', type=int, default=2)
    a = p.parse_args()
    if a.samples < 4 or a.workers < 0 or a.epochs < 1:
        p.error('Need >=4 samples, >=0 workers and >=1 epochs')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    reference = {}; timing = []
    for mode in ('legacy_main_process', 'loader_workers'):
        dataset = RoadSkeletonDataset(
            str(a.root_path), 'train', image_size=512, source_patch_size=1500,
            random_crop_train=True, random_crops_per_image=1,
            random_crop_seed=a.seed, augment=True,
            native_relations=mode == 'loader_workers')
        subset = Subset(dataset, list(range(min(a.samples, len(dataset)))))
        for epoch in range(a.epochs):
            dataset.set_epoch(epoch)
            loader = DataLoader(subset, batch_size=2, shuffle=True,
                                num_workers=a.workers,
                                generator=torch.Generator().manual_seed(a.seed+epoch),
                                **({'prefetch_factor': 2, 'persistent_workers': False}
                                   if a.workers else {}))
            waits = []; cpu_generation = []
            started = time.perf_counter(); iterator = iter(loader); i = 0
            while True:
                batch_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if mode == 'legacy_main_process':
                    target_started = time.perf_counter()
                    targets, stats = legacy_native_batch(batch['mask'], batch['skeleton'])
                    cpu_generation.append(time.perf_counter()-target_started)
                else:
                    targets = batch['native_relations']
                    stats = reduce_native_stats(batch['native_relation_stats'])
                    cpu_generation.append(float(batch['native_target_seconds'].sum()))
                waits.append(time.perf_counter()-batch_started)
                signature = fingerprint(batch, targets, stats)
                if mode == 'legacy_main_process':
                    reference[(epoch, i)] = signature
                elif signature != reference[(epoch, i)]:
                    raise RuntimeError(f'Element-byte/input/statistics mismatch at epoch={epoch}, batch={i}')
                i += 1
                print(f'[{mode}] epoch={epoch+1} batch={i} equality=OK', flush=True)
            timing.append(dict(mode=mode, epoch=epoch+1, batches=i,
                               wall_seconds=time.perf_counter()-started,
                               steady_data_and_target_ms=1000*sum(waits[2:])/max(len(waits)-2, 1),
                               target_cpu_seconds_sum=sum(cpu_generation)))
    report = dict(passed=True, samples_per_epoch=len(subset), epochs=a.epochs,
                  batches_verified=len(reference), workers=a.workers, timings=timing,
                  note='CPU data/label throughput only, not GPU training speed. Worker CPU seconds overlap across processes.',
                  unchanged='RGB, road, skeleton, dilation, crop coordinates, order, all four native tensors and statistics')
    (a.output_dir/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
