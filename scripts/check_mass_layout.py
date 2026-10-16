"""Check Massachusetts split pairing, native size and label encoding before training."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.dataset_road_skeleton import RoadSkeletonDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--source_size", type=int, default=1500)
    args = parser.parse_args()
    split_names = {}
    for split in ("train", "val", "test"):
        dataset = RoadSkeletonDataset(
            root_dir=str(args.root_path), split=split,
            image_size=512, source_patch_size=args.source_size,
            return_full_image=True,
        )
        names = set(dataset.image_files)
        if not names:
            parser.error(f"Empty split: {split}")
        split_names[split] = names
        positive = 0
        for name in dataset.image_files:
            image_path = Path(dataset.image_dir) / name
            label_path = Path(dataset.mask_dir) / dataset._find_label_name(name, dataset.mask_dir)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if image is None or label is None or image.shape[:2] != (args.source_size,) * 2 or label.shape != image.shape[:2]:
                parser.error(f"Expected paired {args.source_size}x{args.source_size} image and label: {image_path}, {label_path}")
            values = np.unique(label)
            if not np.isin(values, (0, 255)).all():
                parser.error(f"Expected binary 0/255 label for training and unified metrics: {label_path}; values={values[:12]}")
            if label.max() > 127:
                positive += 1
        if not positive:
            parser.error(f"All {split} labels are empty")
        print(f"{split}: {len(names)} paired {args.source_size}x{args.source_size} images; {positive} labels contain roads", flush=True)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_names[left] & split_names[right]
        if overlap:
            parser.error(f"Overlapping filenames between {left} and {right}: {sorted(overlap)[:5]}")


if __name__ == "__main__":
    main()
