#!/usr/bin/env python3
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from PIL import Image
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from datasets.dataset_synapse import ImageDataset
import argparse
import cv2
from datetime import datetime

# ResizeToTensor transform
class ResizeToTensor:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = image.transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image * std + mean) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(image, mode='RGB').resize(self.output_size, Image.BILINEAR)
        
        label_255 = (label * 255).astype(np.uint8)
        label = Image.fromarray(label_255).resize(self.output_size, Image.NEAREST)
        
        image = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)
        
        label = np.array(label, dtype=np.float32)
        label = (label > 127).astype(np.float32)
        
        sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}
        return sample

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml')
parser.add_argument('--root_path', type=str, default='./data1')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--threshold', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
parser.add_argument('--zip', action='store_true')
parser.add_argument('--cache_mode', type=str, default='')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--accumulation_steps', type=int, default=0)
parser.add_argument('--use_checkpoint', action='store_true')
parser.add_argument('--amp_opt_level', type=str, default='')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--throughput', action='store_true')
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dataset', type=str, default='ImageData')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')
parser.add_argument('--output_dir', type=str, default='./model_out')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--print_freq', type=int, default=10)

args = parser.parse_args([])

config = get_config(args)
model = ViT_seg(config=config, img_size=args.img_size, num_classes=args.num_classes, use_asterisk=True).cuda()

checkpoint = torch.load('./model_out/checkpoints/epoch_100.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_image_dir = f'{args.root_path}/test/image'
test_label_dir = f'{args.root_path}/test/label'

test_dataset = ImageDataset(
    image_dir=test_image_dir,
    label_dir=test_label_dir,
    transform=ResizeToTensor(output_size=[args.img_size, args.img_size])
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

print("\n" + "="*80)
print("诊断模型输出形状和概率分布 - 检查前3个样本")
print("="*80)

model.eval()
# 创建带时间戳的诊断输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
diagnose_dir = f'./diagnose_output_{timestamp}'
os.makedirs(diagnose_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 3:
            break
        
        images = batch['image'].cuda()
        labels = batch['label'].long().cuda()
        case_name = batch['case_name'][0]
        
        outputs = model(images)
        
        print(f"\n【样本 {batch_idx+1}: {case_name}】")
        print(f"  原图形状: {images.shape} (B, C, H, W)")
        print(f"  输出形状: {outputs.shape} (B, num_classes, H, W)")
        print(f"  标签形状: {labels.shape} (B, H, W)")
        
        # 检查是否需要上采样
        if outputs.shape[-2:] != labels.shape[-2:]:
            print(f"  ⚠️ 警告: 输出尺寸 {outputs.shape[-2:]} != 标签尺寸 {labels.shape[-2:]}")
            print(f"  ⚠️ 进行上采样...")
            outputs = torch.nn.functional.interpolate(
                outputs,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            print(f"  上采样后: {outputs.shape}")
        
        # 计算概率
        prob = torch.softmax(outputs, dim=1)[:, 1]  # [B, H, W]
        
        print(f"\n  概率统计 (softmax第1类):")
        print(f"    min: {prob.min():.6f}")
        print(f"    max: {prob.max():.6f}")
        print(f"    mean: {prob.mean():.6f}")
        print(f"    median: {torch.median(prob):.6f}")
        print(f"    std: {prob.std():.6f}")
        
        # 不同阈值的统计
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        print(f"\n  不同阈值的预测点数:")
        for th in thresholds:
            pred_count = (prob >= th).sum().item()
            total_pixels = prob.numel()
            percentage = 100 * pred_count / total_pixels
            print(f"    threshold={th}: {pred_count:6d} pixels ({percentage:5.1f}%)")
        
        # 保存概率图和不同阈值的预测
        prob_np = prob[0].detach().cpu().numpy()
        
        # 保存概率图（热力图）
        prob_img = (prob_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(diagnose_dir, f'{batch_idx+1}_prob.png'), prob_img)
        
        # 保存不同阈值的二值图
        for th in thresholds:
            pred = (prob_np > th).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(diagnose_dir, f'{batch_idx+1}_pred_th_{th}.png'), pred)
        
        # 保存GT标签
        gt_np = labels[0].detach().cpu().numpy()
        gt_img = (gt_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(diagnose_dir, f'{batch_idx+1}_gt.png'), gt_img)
        
        # 保存原始图片
        orig_img = images[0].detach().cpu().numpy()  # [3, H, W], normalized
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        orig_img = (orig_img * std + mean) * 255
        orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)
        orig_img = np.transpose(orig_img, (1, 2, 0))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(diagnose_dir, f'{batch_idx+1}_image.png'), orig_img)
        
        print(f"  已保存诊断图到: {diagnose_dir}")

print("\n" + "="*80)
print(f"诊断完成！已保存以下文件到 {diagnose_dir}:")
print("  - X_image.png: 原始输入图像")
print("  - X_prob.png: 模型输出的概率热力图")
print("  - X_pred_th_0.X.png: 不同阈值的二值预测")
print("  - X_gt.png: 真实标签")
print("="*80)
