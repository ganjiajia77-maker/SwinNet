"""
Move missing labels from test to val based on val/image files.
"""
import os
import shutil


def fix_val_labels(root_dir='./data'):
    """Move corresponding labels from test to val."""
    
    val_image_dir = os.path.join(root_dir, 'val', 'image')
    val_label_dir = os.path.join(root_dir, 'val', 'label')
    test_label_dir = os.path.join(root_dir, 'test', 'label')
    
    # Get all image files in val
    val_images = sorted([f for f in os.listdir(val_image_dir)])
    print(f"Val images: {len(val_images)}")
    
    # Get label files in test
    test_labels = os.listdir(test_label_dir)
    label_map = {}
    for label_file in test_labels:
        # Label: "123456_mask.png" -> id "123456"
        label_id = label_file.replace('_mask.png', '')
        label_map[label_id] = label_file
    
    print(f"Test labels available: {len(label_map)}")
    
    # Move labels
    print(f"\nMoving labels from test to val...")
    moved = 0
    missing = 0
    
    for image_file in val_images:
        # Extract ID: "123456_sat.jpg" -> "123456"
        image_id = image_file.split('_sat')[0]
        
        if image_id in label_map:
            label_file = label_map[image_id]
            src = os.path.join(test_label_dir, label_file)
            dst = os.path.join(val_label_dir, label_file)
            
            if os.path.exists(src):
                shutil.move(src, dst)
                moved += 1
            else:
                print(f"  ERROR: Source label not found: {src}")
                missing += 1
        else:
            print(f"  WARNING: No label mapping for {image_file}")
            missing += 1
    
    print(f"\nMoved: {moved}")
    print(f"Missing/Failed: {missing}")
    
    # Verify
    final_val_labels = len(os.listdir(val_label_dir))
    final_test_labels = len(os.listdir(test_label_dir))
    
    print(f"\n=== Final Status ===")
    print(f"Val labels: {final_val_labels}")
    print(f"Test labels: {final_test_labels}")


if __name__ == '__main__':
    fix_val_labels()
