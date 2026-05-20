#!/usr/bin/env python3
import torch
import sys
from torch.utils.data import DataLoader
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from datasets.dataset_synapse import ImageDataset
import argparse
import numpy as np
from PIL import Image

# 定义 ResizeToTensor transform
class ResizeToTensor:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image: [3, H, W], label: [H, W] (已二值化为 0 或 1)
        
        # 转换为 [H, W, 3] 便于 PIL 处理
        image = image.transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        
        # 转换为 uint8 进行 PIL 操作（反去归一化和标准化）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image * std + mean) * 255  # 反归一化
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(image, mode='RGB').resize(self.output_size, Image.BILINEAR)
        
        # 标签已经二值化为 0.0 或 1.0，需要转换为 0-255 再进行 resize
        label_255 = (label * 255).astype(np.uint8)  # 转换回 0-255 用于 resize
        label = Image.fromarray(label_255).resize(self.output_size, Image.NEAREST)
        
        # 转回 float32 并重新归一化
        image = np.array(image, dtype=np.float32) / 255.0
        
        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        
        label = np.array(label, dtype=np.float32)
        label = (label > 127).astype(np.float32)  # 再次二值化 0-255 -> 0-1
        
        sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}
        return sample

# 准备参数
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data1', help='root dir for data')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', help='config path')
parser.add_argument('--img_size', type=int, default=224, help='input size')
parser.add_argument('--num_classes', type=int, default=2, help='num classes')
parser.add_argument('--threshold', type=float, default=0.2, help='threshold')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='num workers')
# 添加其他必需的参数
parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[], help='opts')
parser.add_argument('--zip', action='store_true', help='zip')
parser.add_argument('--cache_mode', type=str, default='', help='cache_mode')
parser.add_argument('--resume', type=str, default='', help='resume')
parser.add_argument('--accumulation_steps', type=int, default=0, help='accumulation_steps')
parser.add_argument('--use_checkpoint', action='store_true', help='use_checkpoint')
parser.add_argument('--amp_opt_level', type=str, default='', help='amp_opt_level')
parser.add_argument('--tag', type=str, default='', help='tag')
parser.add_argument('--eval', action='store_true', help='eval')
parser.add_argument('--throughput', action='store_true', help='throughput')
parser.add_argument('--base_lr', type=float, default=0.001, help='base_lr')
parser.add_argument('--max_epochs', type=int, default=100, help='max_epochs')
parser.add_argument('--n_gpu', type=int, default=1, help='n_gpu')
parser.add_argument('--deterministic', type=int, default=1, help='deterministic')
parser.add_argument('--seed', type=int, default=1234, help='seed')
parser.add_argument('--dataset', type=str, default='ImageData', help='dataset')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list_dir')
parser.add_argument('--output_dir', type=str, default='./model_out', help='output_dir')
parser.add_argument('--n_class', type=int, default=2, help='n_class')
parser.add_argument('--print_freq', type=int, default=10, help='print_freq')

args = parser.parse_args([])  # 使用默认值

print("加载配置...")
config = get_config(args)

print("创建模型...")
model = ViT_seg(config=config, img_size=args.img_size, 
                num_classes=args.num_classes, use_asterisk=True).cuda()

print("加载检查点...")
checkpoint = torch.load('./model_out/checkpoints/epoch_100.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print("加载验证数据...")
val_dataset = ImageDataset(
    image_dir=f'{args.root_path}/val/image',
    label_dir=f'{args.root_path}/val/label',
    transform=ResizeToTensor((args.img_size, args.img_size))
)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print(f"验证集大小: {len(val_dataset)}, batch数: {len(val_loader)}")

model.eval()

print("\n" + "="*80)
print("验证测试 - 只检查前2个batch")
print("="*80)

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 2:
            break
        
        images = batch['image'].cuda()
        labels = batch['label'].long().cuda()
        outputs = model(images)
        
        print(f"\n【Batch {batch_idx}】")
        print(f"  输出形状: {outputs.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签唯一值: {torch.unique(labels).cpu().tolist()}")
        print(f"  标签中类1的像素数: {(labels == 1).sum().item()}")
        print(f"  标签中类0的像素数: {(labels == 0).sum().item()}")
        
        # Softmax
        prob = torch.softmax(outputs, dim=1)[:, 1]  # 取类1的概率
        print(f"\n  概率统计:")
        print(f"    形状: {prob.shape}")
        print(f"    min: {prob.min():.6f}, max: {prob.max():.6f}, mean: {prob.mean():.6f}")
        print(f"    中位数: {torch.median(prob):.6f}")
        print(f"    <0.2的像素数: {(prob < 0.2).sum().item()}")
        print(f"    0.2-0.8的像素数: {((prob >= 0.2) & (prob <= 0.8)).sum().item()}")
        print(f"    >0.8的像素数: {(prob > 0.8).sum().item()}")
        
        # 二值化预测
        pred = (prob >= args.threshold).long()
        print(f"\n  预测统计 (阈值={args.threshold}):")
        print(f"    形状: {pred.shape}")
        print(f"    唯一值: {torch.unique(pred).cpu().tolist()}")
        print(f"    预测为类1的像素数: {(pred == 1).sum().item()}")
        print(f"    预测为类0的像素数: {(pred == 0).sum().item()}")
        
        # 计算TP/FP/FN
        tp = int(((pred == 1) & (labels == 1)).sum().item())
        fp = int(((pred == 1) & (labels == 0)).sum().item())
        fn = int(((pred == 0) & (labels == 1)).sum().item())
        tn = int(((pred == 0) & (labels == 0)).sum().item())
        
        print(f"\n  混淆矩阵统计:")
        print(f"    TP (预测1,实际1): {tp}")
        print(f"    FP (预测1,实际0): {fp}")
        print(f"    FN (预测0,实际1): {fn}")
        print(f"    TN (预测0,实际0): {tn}")
        
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            print(f"  指标:")
            print(f"    IoU: {iou:.6f}")
            print(f"    Precision: {precision:.6f}")
            print(f"    Recall: {recall:.6f}")
            print(f"    F1: {f1:.6f}")
        else:
            print(f"  指标: 无法计算 (TP+FP+FN=0)")

print("\n" + "="*80)
