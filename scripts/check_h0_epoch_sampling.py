"""CPU preflight: original Mass H0 sources, unique crops, worker/direct equality."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from mass_protocol import crop_position

ORIGINAL_SOURCES = {
    'datasets/dataset_road_skeleton.py': 'd2a8b42801d7458cbe59ed5d45dee378bc89da554a82fba960fd2805cf63b1bb',
    'mass_protocol.py': '1cfdc7cf14ddf498970ce61dea86cc6ea268a4222993ed1f6141ab147e386128',
}
TENSORS = ('image', 'mask', 'skeleton', 'skeleton_dilate')


def check_original_sources():
    actual = {name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
              for name in ORIGINAL_SOURCES}
    if actual != ORIGINAL_SOURCES:
        raise ValueError('Dataset/crop implementation differs from original Mass H0 fe3b730')
    return actual


def fingerprint(sample):
    value = hashlib.sha256()
    for key in TENSORS:
        value.update(key.encode())
        value.update(sample[key].contiguous().numpy().tobytes())
    return value.hexdigest()


def audit_workers(dataset, indices, workers=4, epochs=(0, 1, 79), persistent=False, context=None):
    """Consume every iterator before changing epoch, including prefetched data."""
    options = dict(num_workers=workers, batch_size=1, shuffle=False,
                   generator=torch.Generator().manual_seed(dataset.random_crop_seed))
    if workers:
        options['persistent_workers'] = persistent
        if context:
            options['multiprocessing_context'] = context
    loader = DataLoader(Subset(dataset, indices), **options)
    rows = []
    try:
        for epoch in epochs:
            dataset.set_epoch(epoch)
            expected = [dataset[index] for index in indices]
            observed = list(loader)
            if len(observed) != len(expected):
                raise AssertionError('Worker sample count changed')
            for index, direct, batch in zip(indices, expected, observed):
                for key in TENSORS:
                    if not torch.equal(direct[key], batch[key][0]):
                        raise AssertionError(f'Worker/direct mismatch: epoch={epoch} index={index} {key}')
                top, left = int(batch['tile_top'][0]), int(batch['tile_left'][0])
                if (top, left) != (direct['tile_top'], direct['tile_left']):
                    raise AssertionError('Worker crop coordinates differ')
                rows.append(dict(epoch_index=epoch, index=index, image=direct['image_name'],
                                 top=top, left=left, tensors_sha256=fingerprint(direct)))
    finally:
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
    for index in indices:
        locations = [(row['top'], row['left']) for row in rows if row['index'] == index]
        if len(set(locations)) != len(epochs):
            raise AssertionError('Crop position repeated across checked epochs')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = check_original_sources()
    dataset = RoadSkeletonDataset(str(args.root_path), 'train', image_size=512,
        source_patch_size=1500, tile_size=512, tile_stride=256, augment=True,
        random_crop_train=True, random_crops_per_image=1, random_crop_seed=args.seed)
    # Check all image/crop indices across the complete planned training budget.
    for index in range(len(dataset)):
        image, ordinal = divmod(index, dataset.random_crops_per_image)
        positions = {crop_position(args.seed, image, ordinal, epoch, 1500, 1500, 512)
                     for epoch in range(80)}
        if len(positions) != 80:
            raise AssertionError(f'Non-unique crop sequence for image index {image}')
    print(f'[SAMPLING] {len(dataset)} images: all 80 crop positions differ', flush=True)
    indices = sorted(set((0, len(dataset)//2, len(dataset)-1)))
    rows = audit_workers(dataset, indices, args.num_workers)
    report = dict(passed=True, optimizer_steps=0, reference_source_commit='fe3b730',
        data_root=str(args.root_path.resolve()),
        sources_sha256_lf=sources, seed=args.seed, num_workers=args.num_workers,
        persistent_workers=False, train_images=len(dataset.image_files),
        crops_per_image=dataset.random_crops_per_image, checked_unique_epochs=80,
        worker_direct_equal=True, checked_epochs=[0, 1, 79], samples=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('[SAMPLING PASSED] worker RGB/GT/skeleton equal direct original-H0 samples', flush=True)
    for row in rows:
        print(f"epoch={row['epoch_index']+1:2d} image={row['image']} crop=({row['top']},{row['left']})", flush=True)


if __name__ == '__main__':
    main()
