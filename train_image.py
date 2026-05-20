import argparse
import csv
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader

from networks.vision_transformer import SwinUnet as ViT_seg
from datasets.dataset_synapse import ImageDataset, RandomGenerator
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceSkeletonLoss
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data1', help='root dir for data')
parser.add_argument('--dataset', type=str, default='ImageData', help='dataset name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='./model_out', help='output dir')
parser.add_argument('--run_name', type=str, default='', help='optional run folder name under output_dir')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', 
                    help='path to config file')
parser.add_argument('--n_class', default=2, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--print_freq', default=10, type=int, help='print loss every N batches')
parser.add_argument('--threshold', default=0.2, type=float, help='binary threshold for validation')
# Options expected by the original config updater
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

# 模块级别定义，以便被 DataLoader worker 进程使用
class ResizeToTensor:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image: [3, H, W], label: [H, W] (已二值化为 0 或 1)
        
        # 转换为 [H, W, 3] 便于 PIL 处理
        image = image.transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        
        # 转换为 uint8 进行 PIL 操作（反去归一化和标准化）
        # 由于已经做过标准化，这里简单还原逻辑：先乘 std，再加 mean，最后乘 255
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

def load_split(split_name, train_mode=True, root_path='.', img_size=224, num_workers=4):
    image_dir = os.path.join(root_path, split_name, 'image')
    label_dir = os.path.join(root_path, split_name, 'label')
    print(f"加载{split_name}数据...")
    print(f"  Image目录: {image_dir}")
    print(f"  Label目录: {label_dir}")
    transform = RandomGenerator(output_size=[img_size, img_size]) if train_mode else ResizeToTensor((img_size, img_size))
    dataset = ImageDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transform
    )
    print(f"{split_name}集大小: {len(dataset)}")
    return dataset

def dice_loss(pred, target, threshold=0.5):
    """
    计算 Dice Loss
    pred: [B, 2, H, W] - 模型输出（logits）
    target: [B, H, W] - 目标标签
    threshold: 二值化阈值
    """
    # 如果是 [B, 2, H, W]，取第 1 类的概率
    if pred.dim() == 4 and pred.size(1) == 2:
        pred_prob = torch.softmax(pred, dim=1)[:, 1]  # [B, H, W]
    else:
        pred_prob = pred
    
    pred_binary = (pred_prob >= threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    
    dice_score = 2 * intersection / (union + 1e-8)
    return 1 - dice_score

def evaluate(model, loader, criterion, threshold=0.2):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = 0
    debug_printed = False  # 仅打印第一个batch的调试信息
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].cuda()
            labels = batch['label'].long().cuda()

            outputs = model(images)
            # model now returns (surface_logits, skeleton_logits, skeleton_attn) when return_skeleton
            if isinstance(outputs, tuple) and len(outputs) == 3:
                surface_logits, skeleton_logits, _ = outputs
            else:
                surface_logits = outputs

            # CE Loss
            ce_loss = criterion(surface_logits, labels)

            # Dice Loss
            dice_l = dice_loss(surface_logits, labels.float(), threshold)
            
            # 合并损失 (Dice权重从1.0降到0.5)
            loss = ce_loss + 0.5 * dice_l
            total_loss += loss.item()

            prob = torch.softmax(outputs, dim=1)[:, 1]
            pred = (prob >= threshold).long()
            
            # 调试：打印第一个batch的信息
            if not debug_printed:
                print(f"\n[验证调试] 第一个Batch:")
                print(f"  Label形状: {labels.shape}, 唯一值: {torch.unique(labels)}")
                print(f"  Pred形状: {pred.shape}, 唯一值: {torch.unique(pred)}")
                print(f"  Label类1的数量: {(labels == 1).sum().item()}")
                print(f"  Pred类1的数量: {(pred == 1).sum().item()}")
                print(f"  Prob min/max/mean: {prob.min():.4f}/{prob.max():.4f}/{prob.mean():.4f}")
                debug_printed = True
            
            tp += int(((pred == 1) & (labels == 1)).sum().item())
            fp += int(((pred == 1) & (labels == 0)).sum().item())
            fn += int(((pred == 0) & (labels == 1)).sum().item())

    avg_loss = total_loss / max(len(loader), 1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return avg_loss, iou, f1, precision, recall

def evaluate_skeleton(model, loader, criterion, threshold=0.2):
    model.eval()
    total_loss = 0.0
    surface_tp = surface_fp = surface_fn = 0
    skeleton_tp = skeleton_fp = skeleton_fn = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].cuda()
            masks = batch['mask'].cuda()
            skeletons = batch['skeleton'].cuda()

            outputs = model(images)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                surface_logits, skeleton_logits, _ = outputs
            else:
                surface_logits = outputs
                skeleton_logits = outputs

            loss, _ = criterion(surface_logits, skeleton_logits, masks, skeletons)
            total_loss += loss.item()

            surface_pred = (torch.sigmoid(surface_logits) >= threshold).float()
            skeleton_pred = (torch.sigmoid(skeleton_logits) >= threshold).float()
            masks = (masks > 0.5).float()
            skeletons = (skeletons > 0.5).float()

            surface_tp += int((surface_pred * masks).sum().item())
            surface_fp += int((surface_pred * (1.0 - masks)).sum().item())
            surface_fn += int(((1.0 - surface_pred) * masks).sum().item())

            skeleton_tp += int((skeleton_pred * skeletons).sum().item())
            skeleton_fp += int((skeleton_pred * (1.0 - skeletons)).sum().item())
            skeleton_fn += int(((1.0 - skeleton_pred) * skeletons).sum().item())

    avg_loss = total_loss / max(len(loader), 1)

    surface_precision = surface_tp / (surface_tp + surface_fp + 1e-8)
    surface_recall = surface_tp / (surface_tp + surface_fn + 1e-8)
    surface_f1 = 2 * surface_precision * surface_recall / (surface_precision + surface_recall + 1e-8)
    surface_iou = surface_tp / (surface_tp + surface_fp + surface_fn + 1e-8)

    skeleton_precision = skeleton_tp / (skeleton_tp + skeleton_fp + 1e-8)
    skeleton_recall = skeleton_tp / (skeleton_tp + skeleton_fn + 1e-8)
    skeleton_f1 = 2 * skeleton_precision * skeleton_recall / (skeleton_precision + skeleton_recall + 1e-8)
    skeleton_iou = skeleton_tp / (skeleton_tp + skeleton_fp + skeleton_fn + 1e-8)

    return {
        'loss': avg_loss,
        'surface_iou': surface_iou,
        'surface_f1': surface_f1,
        'surface_precision': surface_precision,
        'surface_recall': surface_recall,
        'skeleton_iou': skeleton_iou,
        'skeleton_f1': skeleton_f1,
        'skeleton_precision': skeleton_precision,
        'skeleton_recall': skeleton_recall,
    }

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 设置训练参数
    base_output_dir = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() if args.run_name.strip() else f"train_skeleton_{timestamp}"
    args.output_dir = make_unique_dir(base_output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    loss_log_path = os.path.join(args.output_dir, 'epoch_losses.csv')
    batch_loss_log_path = os.path.join(args.output_dir, 'batch_losses.csv')
    training_log_path = os.path.join(args.output_dir, 'training_log.txt')  # 统一日志文件
    
    # 加载配置
    config = get_config(args)
    
    # 创建网络（启用 DilatedAsterisk 空间特征增强）
    args.num_classes = 1
    model = ViT_seg(config=config, img_size=args.img_size,
                    num_classes=args.num_classes, use_asterisk=True,
                    return_skeleton=True).cuda()

    # 加载数据
    train_dataset = RoadSkeletonDataset(root_dir=args.root_path, split='train', image_size=args.img_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_dataset = RoadSkeletonDataset(root_dir=args.root_path, split='val', image_size=args.img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.0001)

    # 损失函数
    criterion = SurfaceSkeletonLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=0.3
    ).cuda()

    # 检查是否恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cuda')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"从epoch {start_epoch} 继续训练...")
        else:
            print(f"Warning: checkpoint未找到 {args.resume}")

    # 训练循环
    print("\n开始训练...")
    print(f"配置:")
    print(f"  最大轮数: {args.max_epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.base_lr}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  Checkpoints目录: {checkpoints_dir}")
    print(f"  验证阈值: {args.threshold}")
    if args.resume:
        print(f"  从checkpoint恢复: {args.resume}")
        print(f"  起始epoch: {start_epoch}")

    best_val_f1 = -1.0
    
    # 写入训练日志头
    with open(training_log_path, 'w', encoding='utf-8') as log_f:
        log_f.write("="*100 + "\n")
        log_f.write("Swin-UNet 训练日志\n")
        log_f.write("="*100 + "\n")
        log_f.write(f"训练配置:\n")
        log_f.write(f"  最大轮数: {args.max_epochs}\n")
        log_f.write(f"  批大小: {args.batch_size}\n")
        log_f.write(f"  学习率: {args.base_lr}\n")
        log_f.write(f"  验证阈值: {args.threshold}\n")
        log_f.write(f"  数据目录: {args.root_path}\n")
        log_f.write("="*100 + "\n\n")
    
    with open(loss_log_path, 'w', newline='', encoding='utf-8') as loss_log_file, \
         open(batch_loss_log_path, 'w', newline='', encoding='utf-8') as batch_loss_log_file:
        loss_writer = csv.writer(loss_log_file)
        batch_loss_writer = csv.writer(batch_loss_log_file)
        loss_writer.writerow([
            'epoch', 'train_avg_loss', 'val_loss',
            'surface_iou', 'surface_f1', 'surface_precision', 'surface_recall',
            'skeleton_iou', 'skeleton_f1', 'skeleton_precision', 'skeleton_recall'
        ])
        batch_loss_writer.writerow(['epoch', 'batch', 'loss'])

        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            total_loss = 0

            for i, batch in enumerate(train_loader):
                images = batch['image'].cuda()
                masks = batch['mask'].cuda()
                skeletons = batch['skeleton'].cuda()

                outputs = model(images)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    surface_logits, skeleton_logits, _ = outputs
                else:
                    surface_logits = outputs
                
                # CE Loss + Dice Loss (权重调整: Dice从1.0降到0.5)
                loss, loss_dict = criterion(surface_logits, skeleton_logits, masks, skeletons)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_loss_writer.writerow([epoch + 1, i + 1, f'{loss.item():.6f}'])
                batch_loss_log_file.flush()

                if (i + 1) % args.print_freq == 0 or i == 0:
                    print(
                        f"Epoch [{epoch+1}/{args.max_epochs}], Batch [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}, Surface: {loss_dict['surface_loss'].item():.4f}, "
                        f"Skeleton: {loss_dict['skeleton_loss'].item():.4f}",
                        flush=True
                    )

            train_avg_loss = total_loss / len(train_loader)
            val_metrics = evaluate_skeleton(model, val_loader, criterion, args.threshold)
            val_loss = val_metrics['loss']
            val_iou = val_metrics['surface_iou']
            val_f1 = val_metrics['surface_f1']
            val_precision = val_metrics['surface_precision']
            val_recall = val_metrics['surface_recall']
            
            # 打印到控制台
            epoch_msg = (
                f"Epoch {epoch+1}/{args.max_epochs}, Train Loss: {train_avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Surface IoU: {val_iou:.4f}, Surface F1: {val_f1:.4f}, "
                f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                f"Skeleton IoU: {val_metrics['skeleton_iou']:.4f}, Skeleton F1: {val_metrics['skeleton_f1']:.4f}"
            )
            print(epoch_msg, flush=True)
            
            # 写入 CSV
            loss_writer.writerow([
                epoch + 1, f'{train_avg_loss:.6f}', f'{val_loss:.6f}',
                f'{val_iou:.6f}', f'{val_f1:.6f}', f'{val_precision:.6f}', f'{val_recall:.6f}',
                f"{val_metrics['skeleton_iou']:.6f}", f"{val_metrics['skeleton_f1']:.6f}",
                f"{val_metrics['skeleton_precision']:.6f}", f"{val_metrics['skeleton_recall']:.6f}"
            ])
            loss_log_file.flush()
            
            # 写入 txt 日志
            with open(training_log_path, 'a', encoding='utf-8') as log_f:
                log_f.write(epoch_msg + "\n")
                log_f.write(f"  Train Avg Loss: {train_avg_loss:.6f}\n")
                log_f.write(f"  Val Loss: {val_loss:.6f}\n")
                log_f.write(f"  Val IoU: {val_iou:.6f}\n")
                log_f.write(f"  Val F1: {val_f1:.6f}\n")
                log_f.write(f"  Val Precision: {val_precision:.6f}\n")
                log_f.write(f"  Val Recall: {val_recall:.6f}\n")
                log_f.write(f"  Skeleton IoU: {val_metrics['skeleton_iou']:.6f}\n")
                log_f.write(f"  Skeleton F1: {val_metrics['skeleton_f1']:.6f}\n")
                log_f.write(f"  Skeleton Precision: {val_metrics['skeleton_precision']:.6f}\n")
                log_f.write(f"  Skeleton Recall: {val_metrics['skeleton_recall']:.6f}\n")
                log_f.write("-"*100 + "\n")

            epoch_checkpoint = os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_avg_loss': train_avg_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'skeleton_iou': val_metrics['skeleton_iou'],
                'skeleton_f1': val_metrics['skeleton_f1'],
                'skeleton_precision': val_metrics['skeleton_precision'],
                'skeleton_recall': val_metrics['skeleton_recall'],
                'args': vars(args),
            }
            torch.save(checkpoint, epoch_checkpoint)
            torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))
            print(f"模型已保存到: {epoch_checkpoint}", flush=True)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_path = os.path.join(args.output_dir, 'best.pth')
                torch.save(checkpoint, best_path)
                print(f"当前最优模型已保存到: {best_path}", flush=True)

    # 训练完成，写总结日志
    with open(training_log_path, 'a', encoding='utf-8') as log_f:
        log_f.write("\n" + "="*100 + "\n")
        log_f.write("训练完成总结\n")
        log_f.write("="*100 + "\n")
        log_f.write(f"总 Epochs: {args.max_epochs}\n")
        log_f.write(f"最佳 F1 分数: {best_val_f1:.6f}\n")
        log_f.write(f"最佳模型位置: {os.path.join(args.output_dir, 'best.pth')}\n")
        log_f.write(f"最新模型位置: {os.path.join(args.output_dir, 'latest.pth')}\n")
        log_f.write(f"所有 checkpoints 位置: {checkpoints_dir}\n")
        log_f.write("="*100 + "\n")
    
    print("\n训练完成!")
    print(f"Training log: {training_log_path}")
