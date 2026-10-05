"""One-off train GT cache preparation, including all flip/90-degree variants."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from datasets.anchor_reachability import ReachabilityCache
from datasets.dataset_road_skeleton import RoadSkeletonDataset


def prepare_case(job):
    path, cache_dir, source_size, size, max_geodesic = job
    cv2.setNumThreads(1)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = RoadSkeletonDataset._center_crop_or_pad(mask, (source_size, source_size))
    cache = ReachabilityCache(cache_dir, max_geodesic)
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for rotation in range(4):
            transformed = np.ascontiguousarray(np.rot90(base, rotation))
            resized = cv2.resize((transformed > 127).astype(np.float32),
                                 (size, size), interpolation=cv2.INTER_NEAREST)
            cache.get(resized)
    return os.path.basename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--cache_dir', default='')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--source_patch_size', type=int, default=1024)
    parser.add_argument('--max_geodesic', type=float, default=96.0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cache_dir = args.cache_dir or os.path.join(args.root_path, '.anchor_reachability_cache')
    dataset = RoadSkeletonDataset(args.root_path, split='train', image_size=args.img_size,
                                  source_patch_size=args.source_patch_size)
    jobs = [(os.path.join(dataset.mask_dir, dataset._find_label_name(name, dataset.mask_dir)),
             cache_dir, args.source_patch_size, args.img_size, args.max_geodesic)
            for name in dataset.image_files]
    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _ in tqdm(pool.map(prepare_case, jobs), total=len(jobs), desc='GT cache (8 variants/case)'):
                pass
    else:
        for job in tqdm(jobs, desc='GT cache (8 variants/case)'):
            prepare_case(job)
    print('GT cache ready:', cache_dir, flush=True)


if __name__ == '__main__':
    main()
