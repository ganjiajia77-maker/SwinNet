import argparse
import os

import cv2
import numpy as np


def resolve_dir(root_dir, split, names):
    for name in names:
        path = os.path.join(root_dir, split, name)
        if os.path.exists(path):
            return path
    flat = os.path.join(root_dir, split)
    if "image" in names and os.path.exists(flat):
        return flat
    sibling = os.path.join(root_dir, f"{split}_labels")
    if "label" in names and os.path.exists(sibling):
        return sibling
    raise FileNotFoundError(f"Cannot resolve {split} directory under {root_dir}")


def find_label_name(image_name, label_dir):
    base = os.path.splitext(image_name)[0]
    candidates = [
        image_name,
        base + ".png",
        base + ".tif",
        base + ".tiff",
        base.replace("_sat", "_mask") + ".png",
        base.replace("_sat", "_mask") + ".tif",
        base.replace("_sat", "_mask") + ".tiff",
        base.replace("_image", "_mask") + ".png",
        base.replace("_image", "_mask") + ".tif",
        base.replace("_image", "_mask") + ".tiff",
        base.replace("_img", "_mask") + ".png",
        base.replace("_img", "_mask") + ".tif",
        base.replace("_img", "_mask") + ".tiff",
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(label_dir, candidate)):
            return candidate
    raise FileNotFoundError(f"Cannot find label for {image_name} in {label_dir}")


def center_crop_or_pad(array, target_size):
    height, width = array.shape[:2]
    target_height, target_width = target_size

    if height > target_height:
        top = (height - target_height) // 2
        array = array[top:top + target_height, ...]
    elif height < target_height:
        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        pad_width = ((pad_top, pad_bottom), (0, 0))
        array = np.pad(array, pad_width, mode="constant", constant_values=0)

    if array.shape[1] > target_width:
        left = (array.shape[1] - target_width) // 2
        array = array[:, left:left + target_width, ...]
    elif array.shape[1] < target_width:
        pad_left = (target_width - array.shape[1]) // 2
        pad_right = target_width - array.shape[1] - pad_left
        pad_width = ((0, 0), (pad_left, pad_right))
        array = np.pad(array, pad_width, mode="constant", constant_values=0)
    return array


def sample_crop(mask, rng, crop_size, min_road_pixels, max_tries):
    height, width = mask.shape[:2]
    max_top = max(height - crop_size, 0)
    max_left = max(width - crop_size, 0)
    best = (0, 0)
    best_pixels = -1
    for _ in range(max_tries):
        top = int(rng.randint(0, max_top + 1)) if max_top > 0 else 0
        left = int(rng.randint(0, max_left + 1)) if max_left > 0 else 0
        road_pixels = int((mask[top:top + crop_size, left:left + crop_size] > 127).sum())
        if road_pixels > best_pixels:
            best = (top, left)
            best_pixels = road_pixels
        if road_pixels >= min_road_pixels:
            return top, left
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--source_patch_size", type=int, default=1500)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--crops_per_image", type=int, default=4)
    parser.add_argument("--min_road_pixels", type=int, default=0)
    parser.add_argument("--max_tries", type=int, default=50)
    args = parser.parse_args()

    image_dir = resolve_dir(args.root_path, args.split, ("image",))
    label_dir = resolve_dir(args.root_path, args.split, ("mask", "label"))
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = sorted(name for name in os.listdir(image_dir) if name.lower().endswith(exts))
    if not image_files:
        raise RuntimeError(f"No images found in {image_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(f"# root_path={args.root_path}\n")
        handle.write(f"# split={args.split}\n")
        handle.write(f"# seed={args.seed}\n")
        handle.write(f"# source_patch_size={args.source_patch_size}\n")
        handle.write(f"# crop_size={args.crop_size}\n")
        handle.write(f"# crops_per_image={args.crops_per_image}\n")
        for image_index, image_name in enumerate(image_files):
            label_name = find_label_name(image_name, label_dir)
            mask = cv2.imread(os.path.join(label_dir, label_name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot read label: {label_name}")
            if args.source_patch_size:
                mask = center_crop_or_pad(mask, (args.source_patch_size, args.source_patch_size))
            for crop_index in range(args.crops_per_image):
                rng = np.random.RandomState(
                    args.seed + image_index * 9176 + crop_index * 101
                )
                top, left = sample_crop(
                    mask,
                    rng,
                    args.crop_size,
                    args.min_road_pixels,
                    args.max_tries,
                )
                handle.write(f"{image_name}: x={left}, y={top}\n")
    print(f"Wrote {len(image_files) * args.crops_per_image} crops to {args.output}")


if __name__ == "__main__":
    main()
