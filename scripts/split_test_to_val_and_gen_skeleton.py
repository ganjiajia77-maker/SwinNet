#!/usr/bin/env python3
"""
从 test 分割出前 100 张到 val，然后为所有 split 生成 skeleton。
最终：train 1500、val 100、test 300。
"""
import os
import shutil
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import imageio.v2 as imageio


def split_test_to_val(root_dir, num_val=100):
    """从 test 中分出前 num_val 张到 val."""
    test_image_dir = os.path.join(root_dir, 'test', 'image')
    test_label_dir = os.path.join(root_dir, 'test', 'label')
    
    val_image_dir = os.path.join(root_dir, 'val', 'image')
    val_label_dir = os.path.join(root_dir, 'val', 'label')
    
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 列出 test 中的图像文件（假设为 _sat.jpg）
    test_images = sorted([f for f in os.listdir(test_image_dir) if f.endswith('_sat.jpg')])
    val_images = test_images[:num_val]
    
    print(f"将前 {num_val} 张图像从 test 移到 val...")
    for img_file in val_images:
        # 图像：{id}_sat.jpg -> {id}_mask.png
        img_id = img_file.replace('_sat.jpg', '')
        label_file = f"{img_id}_mask.png"
        
        # 移动图像
        src_img = os.path.join(test_image_dir, img_file)
        dst_img = os.path.join(val_image_dir, img_file)
        shutil.move(src_img, dst_img)
        
        # 移动标签
        src_label = os.path.join(test_label_dir, label_file)
        dst_label = os.path.join(val_label_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        
        print(f"  Moved {img_file} -> val/")
    
    print(f"分割完成：val 中现在有 {len(val_images)} 张图像")
    remaining_test = len(test_images) - num_val
    print(f"test 中剩余 {remaining_test} 张图像")


def generate_skeletons(root_dir, split):
    """为指定 split 生成 skeleton."""
    label_dir = os.path.join(root_dir, split, 'label')
    skeleton_dir = os.path.join(root_dir, split, 'skeleton')
    
    os.makedirs(skeleton_dir, exist_ok=True)
    
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('_mask.png')])
    
    print(f"\n为 {split} 生成 skeleton ({len(label_files)} 张图像)...")
    for i, label_file in enumerate(label_files):
        label_path = os.path.join(label_dir, label_file)
        label_img = imageio.imread(label_path)
        
        # 二值化
        label_binary = (label_img > 127).astype(np.uint8)
        
        # 生成 skeleton
        skeleton = skeletonize(label_binary)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        
        # 保存
        skeleton_file = label_file.replace('_mask.png', '_mask.png')
        skeleton_path = os.path.join(skeleton_dir, skeleton_file)
        imageio.imwrite(skeleton_path, skeleton_uint8)
        
        if (i + 1) % 100 == 0 or i == len(label_files) - 1:
            print(f"  已处理 {i + 1}/{len(label_files)}")
    
    print(f"{split} skeleton 生成完成，保存到 {skeleton_dir}")


if __name__ == '__main__':
    root_dir = './data'
    
    # 1. 分割 test -> val (前 100 张)
    split_test_to_val(root_dir, num_val=100)
    
    # 2. 为所有 split 生成 skeleton
    for split in ['train', 'val', 'test']:
        generate_skeletons(root_dir, split)
    
    print("\n✓ 数据分割和 skeleton 生成完成！")
    print("最终数据布局：")
    train_imgs = len(os.listdir(os.path.join(root_dir, 'train', 'image')))
    val_imgs = len(os.listdir(os.path.join(root_dir, 'val', 'image')))
    test_imgs = len(os.listdir(os.path.join(root_dir, 'test', 'image')))
    print(f"  train: {train_imgs}")
    print(f"  val: {val_imgs}")
    print(f"  test: {test_imgs}")
