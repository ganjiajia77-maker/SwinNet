import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from datetime import datetime

from networks.vision_transformer import SwinUnet as ViT_seg
from datasets.dataset_synapse import ImageDataset, RandomGenerator
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceSkeletonLoss
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data1', help='root dir for data')
parser.add_argument('--dataset', type=str, default='ImageData', help='dataset name')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='./predictions', help='output dir')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--overlap_infer', action='store_true', help='use overlapping tile inference')
parser.add_argument('--threshold', type=float, default=0.2, help='binary threshold for predictions')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml',
                    help='path to config file')
parser.add_argument('--n_class', default=2, type=int)
parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--num_workers', default=4, type=int)
# Options expected by config.update_config
parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None, help='modify config options using the command-line')
parser.add_argument('--zip', action='store_true', help='use zipped dataset')
parser.add_argument('--cache_mode', type=str, default='', help='cache mode for dataset')
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
parser.add_argument('--accumulation_steps', type=int, default=0, help='gradient accumulation steps')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing')
parser.add_argument('--amp_opt_level', type=str, default='', help='AMP opt level')
parser.add_argument('--tag', type=str, default='', help='experiment tag')
parser.add_argument('--eval', action='store_true', help='evaluation only')
parser.add_argument('--throughput', action='store_true', help='test throughput only')

args = parser.parse_args()

def make_unique_dir(base_dir, run_name):
    run_dir = os.path.join(base_dir, run_name)
    if not os.path.exists(run_dir):
        return run_dir

    index = 2
    while True:
        candidate = f"{run_dir}_{index}"
        if not os.path.exists(candidate):
            return candidate
        index += 1

# 模块级别定义
class ResizeToTensor:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image: [3, H, W], label: [H, W] (已二值化�?0 �?1)
        
        # 转换�?[H, W, 3] 便于 PIL 处理
        image = image.transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        
        # 转换�?uint8 进行 PIL 操作（反去归一化和标准化）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image * std + mean) * 255  # 反归一�?
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(image, mode='RGB').resize(self.output_size, Image.BILINEAR)
        
        # 标签已经二值化�?0.0 �?1.0，需要转换为 0-255 再进�?resize
        label_255 = (label * 255).astype(np.uint8)  # 转换�?0-255 用于 resize
        label = Image.fromarray(label_255).resize(self.output_size, Image.NEAREST)
        
        # 转回 float32 并重新归一�?
        image = np.array(image, dtype=np.float32) / 255.0
        
        # ImageNet 标准�?
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        
        label = np.array(label, dtype=np.float32)
        label = (label > 127).astype(np.float32)  # 再次二值化 0-255 -> 0-1
        
        sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}
        return sample

def dice_loss(pred, target, threshold=0.5):
    """
    计算 Dice Loss
    pred: [B, 2, H, W] - 模型输出（logits�?
    target: [B, H, W] - 目标标签
    threshold: 二值化阈�?
    """
    # 如果�?[B, 2, H, W]，取�?1 类的概率
    if pred.dim() == 4 and pred.size(1) == 2:
        pred_prob = torch.softmax(pred, dim=1)[:, 1]  # [B, H, W]
    else:
        pred_prob = pred
    
    pred_binary = (pred_prob >= threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    
    dice_score = 2 * intersection / (union + 1e-8)
    return 1 - dice_score

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = get_config(args)
    
    # 创建网络
    args.num_classes = 1
    model = ViT_seg(config=config, img_size=args.img_size,
                    num_classes=args.num_classes, use_asterisk=True,
                    return_skeleton=True).to(device)
    
    # 加载模型
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"已加载模�? {args.model_path}")
    else:
        print(f"错误: 模型文件不存�?{args.model_path}")
        exit(1)
    
    # 加载测试数据
    test_image_dir = os.path.join(args.root_path, 'test', 'image')
    test_label_dir = os.path.join(args.root_path, 'test', 'mask')
    if not os.path.exists(test_label_dir):
        test_label_dir = os.path.join(args.root_path, 'test', 'label')
    
    print(f"加载测试数据...")
    print(f"  Image目录: {test_image_dir}")
    print(f"  Label目录: {test_label_dir}")
    print(f"  阈�? {args.threshold}")
    
    test_dataset = RoadSkeletonDataset(root_dir=args.root_path, split='test', image_size=args.img_size)
    
    print(f"测试集大�? {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 评估
    print("\n开始推�?..")
    model.eval()
    
    total_loss = 0
    tp = fp = fn = 0
    total_samples = 0
    
    # 创建预测结果目录（包含模型名和时间戳，保留历史结果）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用模型文件名作为标识，方便对比不同checkpoint的结�?
    model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    pred_dir = make_unique_dir(args.output_dir, f'{model_basename}_{timestamp}')
    os.makedirs(pred_dir, exist_ok=True)
    surface_dir = os.path.join(pred_dir, 'surface')
    skeleton_dir = os.path.join(pred_dir, 'skeleton')
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)
    
    criterion = SurfaceSkeletonLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=0.3,
    ).to(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            skeletons = batch['skeleton'].to(device)
            
            surface_logits, skeleton_logits, skeleton_attn = model(images)
            
            # 计算损失 - CE Loss + Dice Loss
            loss, _ = criterion(surface_logits, skeleton_logits, masks, skeletons)
            total_loss += loss.item()
            
            pred = (torch.sigmoid(surface_logits) >= args.threshold).float()
            skeleton_pred = (torch.sigmoid(skeleton_logits) >= args.threshold).float()
            masks = (masks > 0.5).float()
            
            tp += int((pred * masks).sum().item())
            fp += int((pred * (1.0 - masks)).sum().item())
            fn += int(((1.0 - pred) * masks).sum().item())
            
            total_samples += 1
            
            # 自动保存预测结果（NPY 格式用于计算，PNG 格式便于查看�?
            case_name = batch['case_name'][0]
            
            # 保存�?NPY（二进制，用于后续处理）
            npy_save_path = os.path.join(surface_dir, f'{case_name}_pred.npy')
            np.save(npy_save_path, pred.squeeze(0).squeeze(0).cpu().numpy())
            skeleton_npy_save_path = os.path.join(skeleton_dir, f'{case_name}_skeleton_pred.npy')
            np.save(skeleton_npy_save_path, skeleton_pred.squeeze(0).squeeze(0).cpu().numpy())
            
            # 保存�?PNG（可视化�?
            pred_numpy = (pred.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            pred_img = Image.fromarray(pred_numpy, mode='L')
            png_save_path = os.path.join(surface_dir, f'{case_name}_pred.png')
            pred_img.save(png_save_path)
            skeleton_numpy = (skeleton_pred.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            skeleton_img = Image.fromarray(skeleton_numpy, mode='L')
            skeleton_png_save_path = os.path.join(skeleton_dir, f'{case_name}_skeleton_pred.png')
            skeleton_img.save(skeleton_png_save_path)
    
    print(f"预测结果已保存到: {pred_dir}")
    
    avg_loss = total_loss / len(test_loader)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # 打印到控制台�?txt 日志
    result_lines = []
    result_lines.append(f"\n测试完成!")
    result_lines.append(f"  平均损失: {avg_loss:.4f}")
    result_lines.append(f"  IoU: {iou:.4f}")
    result_lines.append(f"  F1: {f1:.4f}")
    result_lines.append(f"  Precision: {precision:.4f}")
    result_lines.append(f"  Recall: {recall:.4f}")
    result_lines.append(f"  总测试样�? {len(test_loader)}")
    result_lines.append(f"  预测掩码保存位置: {pred_dir}")
    
    result_lines.append(f"  Surface mask save dir: {surface_dir}")
    result_lines.append(f"  Skeleton pred save dir: {skeleton_dir}")

    for line in result_lines:
        print(line, flush=True)
    
    # 保存测试结果�?txt 文件（保存到带时间戳的文件夹下）
    test_log_path = os.path.join(pred_dir, 'test_results.txt')
    with open(test_log_path, 'w', encoding='utf-8') as log_f:
        log_f.write("="*80 + "\n")
        log_f.write("测试结果\n")
        log_f.write("="*80 + "\n")
        log_f.write(f"时间�? {timestamp}\n")
        log_f.write(f"模型路径: {args.model_path}\n")
        log_f.write(f"阈�? {args.threshold}\n")
        log_f.write(f"输入图像大小: {args.img_size}\n")
        log_f.write(f"测试数据路径: {args.root_path}\n")
        log_f.write("="*80 + "\n")
        for line in result_lines[1:]:  # 跳过第一�?测试完成"
            log_f.write(line + "\n")
        log_f.write("="*80 + "\n")
    
    print(f"测试日志已保存到: {test_log_path}", flush=True)
    
    if args.is_savenii:
        print(f"  预测结果保存�? {args.output_dir}")

