import os
import argparse
import numpy as np
import imageio
from skimage.morphology import skeletonize


def process_folder(mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(mask_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    for fname in files:
        mask_path = os.path.join(mask_dir, fname)
        img = imageio.imread(mask_path)
        if img.ndim == 3:
            img = img[..., 0]
        bw = (img > 127).astype(np.uint8)
        skel = skeletonize(bw > 0).astype(np.uint8) * 255
        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.png')
        imageio.imwrite(out_path, skel)
        print(f'Wrote {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='root dataset dir')
    parser.add_argument('--split', type=str, default='train', help='split: train/val/test')
    args = parser.parse_args()

    mask_dir = os.path.join(args.root, args.split, 'mask')
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(args.root, args.split, 'label')
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f'Cannot find mask/label dir in {args.root}/{args.split}')

    out_dir = os.path.join(args.root, args.split, 'skeleton')
    process_folder(mask_dir, out_dir)


if __name__ == '__main__':
    main()
