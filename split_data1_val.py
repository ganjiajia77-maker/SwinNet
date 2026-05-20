import os
import random
import shutil

ROOT = r'd:\Code\Swin-Unet-main\data1'
TRAIN_IMAGE_DIR = os.path.join(ROOT, 'train', 'image')
TRAIN_LABEL_DIR = os.path.join(ROOT, 'train', 'label')
VAL_IMAGE_DIR = os.path.join(ROOT, 'val', 'image')
VAL_LABEL_DIR = os.path.join(ROOT, 'val', 'label')

VAL_COUNT = 500
SEED = 42


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(VAL_IMAGE_DIR)
    ensure_dir(VAL_LABEL_DIR)

    image_files = sorted([f for f in os.listdir(TRAIN_IMAGE_DIR) if f.endswith('.jpg')])
    if len(image_files) < VAL_COUNT:
        raise RuntimeError(f'Not enough training images: {len(image_files)} < {VAL_COUNT}')

    random.seed(SEED)
    val_files = sorted(random.sample(image_files, VAL_COUNT))

    for image_name in val_files:
        label_name = image_name.replace('_sat.jpg', '_mask.png')
        src_image = os.path.join(TRAIN_IMAGE_DIR, image_name)
        src_label = os.path.join(TRAIN_LABEL_DIR, label_name)
        dst_image = os.path.join(VAL_IMAGE_DIR, image_name)
        dst_label = os.path.join(VAL_LABEL_DIR, label_name)

        if not os.path.exists(src_label):
            raise FileNotFoundError(f'Missing label for {image_name}: {src_label}')

        shutil.move(src_image, dst_image)
        shutil.move(src_label, dst_label)

    print(f'Moved {len(val_files)} samples to data1/val')
    print(f'Train images remaining: {len([f for f in os.listdir(TRAIN_IMAGE_DIR) if f.endswith(".jpg")])}')
    print(f'Val images: {len([f for f in os.listdir(VAL_IMAGE_DIR) if f.endswith(".jpg")])}')


if __name__ == '__main__':
    main()
