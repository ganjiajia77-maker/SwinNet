#!/usr/bin/env python3
"""
重新分割：train 1500、val 100、test 300（总共 1900，原始 test 400）。
"""
import os
import shutil
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import imageio.v2 as imageio


def reset_data_split(root_dir):
    """重新组织数据分割。"""
    print("重新整理数据分割...")
    
    # 将现有 val 移回到 test（原本是从 test 分出来的 100 张）
    val_image_dir = os.path.join(root_dir, 'val', 'image')
    val_label_dir = os.path.join(root_dir, 'val', 'label')
    test_image_dir = os.path.join(root_dir, 'test', 'image')
    test_label_dir = os.path.join(root_dir, 'test', 'label')
    
    if os.path.exists(val_image_dir) and len(os.listdir(val_image_dir)) > 0:
        print("  将现有 val 移回 test...")
        for img_file in os.listdir(val_image_dir):
            if img_file.endswith('_sat.jpg'):
                src_img = os.path.join(val_image_dir, img_file)
                dst_img = os.path.join(test_image_dir, img_file)
                shutil.move(src_img, dst_img)
                
                img_id = img_file.replace('_sat.jpg', '')
                label_file = f"{img_id}_mask.png"
                src_label = os.path.join(val_label_dir, label_file)
                dst_label = os.path.join(test_label_dir, label_file)
                if os.path.exists(src_label):
                    shutil.move(src_label, dst_label)
        
        # 清空 val 目录
        for d in [val_image_dir, val_label_dir]:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except:
                    pass
    
    # 现在 test 应该有 400 张
    test_images = sorted([f for f in os.listdir(test_image_dir) if f.endswith('_sat.jpg')])
    print(f"  test 现在有 {len(test_images)} 张图像")
    
    # 从 test 中分出前 100 张到 val
    val_image_dir = os.path.join(root_dir, 'val', 'image')
    val_label_dir = os.path.join(root_dir, 'val', 'label')
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    val_images = test_images[:100]
    
    print(f"\n将前 100 张从 test 移到 val...")
    for img_file in val_images:
        img_id = img_file.replace('_sat.jpg', '')
        label_file = f"{img_id}_mask.png"
        
        src_img = os.path.join(test_image_dir, img_file)
        dst_img = os.path.join(val_image_dir, img_file)
        shutil.move(src_img, dst_img)
        
        src_label = os.path.join(test_label_dir, label_file)
        dst_label = os.path.join(val_label_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
    
    print(f"✓ 分割完成：train 1500、val 100、test 300")


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
    
    print(f"{split} skeleton 生成完成")


if __name__ == '__main__':
    root_dir = './data'
    
    # 1. 重新整理数据分割
    reset_data_split(root_dir)
    
    # 2. 重新生成所有 skeleton（清除旧的）
    for split in ['train', 'val', 'test']:
        skeleton_dir = os.path.join(root_dir, split, 'skeleton')
        if os.path.exists(skeleton_dir):
            try:
                shutil.rmtree(skeleton_dir)
            except:
                pass
        generate_skeletons(root_dir, split)
    
    print("\n✓ 数据整理和 skeleton 生成完成！")
    print("最终数据布局：")
    train_imgs = len(os.listdir(os.path.join(root_dir, 'train', 'image')))
    val_imgs = len(os.listdir(os.path.join(root_dir, 'val', 'image')))
    test_imgs = len(os.listdir(os.path.join(root_dir, 'test', 'image')))
    print(f"  train: {train_imgs}")
    print(f"  val: {val_imgs}")
    print(f"  test: {test_imgs}")
