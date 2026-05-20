import os
import argparse

import cv2
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm


def generate_one_skeleton(mask_path: str, save_path: str) -> None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    binary = mask > 127
    skel = skeletonize(binary)
    skel_img = skel.astype(np.uint8) * 255

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, skel_img)


def generate_skeleton_folder(mask_dir: str, skeleton_dir: str) -> None:
    os.makedirs(skeleton_dir, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith(exts)]

    if len(files) == 0:
        print(f"[Warning] No mask files found in {mask_dir}")
        return

    for name in tqdm(files, desc=f"Skeletonizing {mask_dir}"):
        mask_path = os.path.join(mask_dir, name)
        save_path = os.path.join(skeleton_dir, name)
        generate_one_skeleton(mask_path, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=r"D:\Code\DeepGlobe_512")
    parser.add_argument("--label_dir_name", type=str, default="", help="mask or label; auto-detect when empty")
    args = parser.parse_args()

    dataset_root = args.dataset_root

    for split in ["train", "val", "test"]:
        if args.label_dir_name:
            mask_dir = os.path.join(dataset_root, split, args.label_dir_name)
        else:
            mask_dir = os.path.join(dataset_root, split, "mask")
            if not os.path.exists(mask_dir):
                mask_dir = os.path.join(dataset_root, split, "label")
        skeleton_dir = os.path.join(dataset_root, split, "skeleton")

        if not os.path.exists(mask_dir):
            print(f"[Skip] {mask_dir} not found.")
            continue

        generate_skeleton_folder(mask_dir, skeleton_dir)

    print("Done.")


if __name__ == "__main__":
    main()
