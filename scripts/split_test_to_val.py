"""
Split first 100 images from test into val folder.
Original: train 1500, test 400
After split: train 1500, val 100, test 300
"""
import os
import shutil
from pathlib import Path


def split_test_to_val(root_dir='./data'):
    """Split first 100 images from test into val."""
    
    # Paths
    test_image_dir = os.path.join(root_dir, 'test', 'image')
    test_label_dir = os.path.join(root_dir, 'test', 'label')
    val_image_dir = os.path.join(root_dir, 'val', 'image')
    val_label_dir = os.path.join(root_dir, 'val', 'label')
    
    # Create val directories
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Get all image files sorted (could be .jpg, .png, .jpeg)
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Total test images: {len(image_files)}")
    print(f"Will split first 100 to val, keep last {len(image_files) - 100} in test")
    
    # Split first 100
    val_image_files = image_files[:100]
    
    # Build label file mapping (extract ID from image filename)
    # Image: {id}_sat.jpg, Label: {id}_mask.png
    label_files = os.listdir(test_label_dir)
    label_map = {}
    for label_file in label_files:
        # Label files are like "100712_mask.png", get the ID before "_mask"
        label_id = label_file.replace('_mask.png', '')
        label_map[label_id] = label_file
    
    print(f"\nMoving {len(val_image_files)} images to val/image...")
    for image_file in val_image_files:
        src = os.path.join(test_image_dir, image_file)
        dst = os.path.join(val_image_dir, image_file)
        shutil.move(src, dst)
        print(f"  Moved {image_file}")
    
    print(f"\nMoving {len(val_image_files)} labels to val/label...")
    for image_file in val_image_files:
        # Extract ID from image filename: "123456_sat.jpg" -> "123456"
        image_id = image_file.split('_sat')[0]
        if image_id in label_map:
            label_file = label_map[image_id]
            src = os.path.join(test_label_dir, label_file)
            dst = os.path.join(val_label_dir, label_file)
            shutil.move(src, dst)
            print(f"  Moved {label_file}")
        else:
            print(f"  WARNING: No label found for {image_file}")
    
    # Verify final counts
    final_test_images = len(os.listdir(test_image_dir))
    final_val_images = len(os.listdir(val_image_dir))
    
    print(f"\n=== Final Split ===")
    print(f"Train: 1500 (unchanged)")
    print(f"Val: {final_val_images}")
    print(f"Test: {final_test_images}")
    print(f"Total: {1500 + final_val_images + final_test_images}")


if __name__ == '__main__':
    split_test_to_val()
