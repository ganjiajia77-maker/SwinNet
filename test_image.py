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

from networks.vision_transformer import (
    TOPOLOGY_ATTENTION_VERSION,
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    format_topology_coefficients,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data1', help='root dir for data')
parser.add_argument('--dataset', type=str, default='ImageData', help='dataset name')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='./predictions', help='output dir')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='network input size after downsampling from a 1024 patch')
parser.add_argument('--source_patch_size', type=int, default=1024, help='source patch size before resizing to img_size')
parser.add_argument('--overlap_infer', action='store_true', help='use overlapping tile inference')
parser.add_argument('--threshold', type=float, default=0.2, help='binary threshold for predictions')
parser.add_argument('--skeleton_threshold', type=float, default=0.5, help='binary threshold for final skeleton')
parser.add_argument('--final_topology_eta_init', type=float, default=0.005, help='initial final topology repair coefficient')
parser.add_argument('--final_gap_rho_init', type=float, default=0.005, help='initial localized gap-repair coefficient')
parser.add_argument(
    '--stage_topology_stages',
    type=str,
    default='none',
    choices=['none', 'stage3', 'stage23'],
)
parser.add_argument('--stage_topology_alpha_max', type=float, default=1.0)
parser.add_argument('--stage_topology_alpha_init', type=float, default=0.1)
parser.add_argument(
    '--stage_topology_bias_mode',
    type=str,
    default='pairwise_skeleton',
    choices=['pairwise_skeleton', 'gap_query'],
)
parser.add_argument('--stage_topology_ratio', type=float, default=0.08)
parser.add_argument('--stage_topology_topo_clip', type=float, default=4.0)
parser.add_argument('--stage2_skeleton_gradient_ratio', type=float, default=0.5)
parser.add_argument('--stage3_skeleton_gradient_ratio', type=float, default=0.5)
parser.add_argument('--final_skeleton_gradient_ratio', type=float, default=0.0)
parser.add_argument(
    '--bottleneck_type',
    type=str,
    default='global_local',
    choices=['global_local', 'legacy_global_local', 'g2l2'],
    help='choose bottleneck implementation',
)
parser.add_argument(
    '--structure_profile',
    type=str,
    default=STRUCTURE_PROFILE_FULL,
    choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626],
)
parser.add_argument(
    '--enable_graph_prop',
    action='store_true',
    help='enable final soft skeleton graph propagation (auto-read from checkpoint when omitted)',
)
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = get_config(args)
    
    # 创建网络
    args.num_classes = 1
    checkpoint = None
    enable_graph_prop = args.enable_graph_prop
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            saved_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
            saved_profile = checkpoint.get("structure_profile")
            if saved_profile:
                args.structure_profile = saved_profile
            elif saved_args:
                args.structure_profile = saved_args.get(
                    "structure_profile",
                    args.structure_profile,
                )
            for name in (
                "stage_topology_stages",
                "stage_topology_alpha_max",
                "stage_topology_alpha_init",
                "stage_topology_bias_mode",
                "stage_topology_ratio",
                "stage_topology_topo_clip",
                "stage2_skeleton_gradient_ratio",
                "stage3_skeleton_gradient_ratio",
                "final_skeleton_gradient_ratio",
            ):
                if name in saved_args:
                    setattr(args, name, saved_args[name])
            if not enable_graph_prop and saved_args:
                enable_graph_prop = bool(
                    saved_args.get("enable_graph_prop", False)
                )

    model = ViT_seg(config=config, img_size=args.img_size,
                    num_classes=args.num_classes, use_asterisk=True,
                    return_skeleton=True, bottleneck_type=args.bottleneck_type,
                    final_topology_eta_init=args.final_topology_eta_init,
                    final_gap_rho_init=args.final_gap_rho_init,
                    stage_topology_stages=args.stage_topology_stages,
                    stage_topology_alpha_max=args.stage_topology_alpha_max,
                    stage_topology_alpha_init=args.stage_topology_alpha_init,
                    stage_topology_bias_mode=args.stage_topology_bias_mode,
                    stage_topology_ratio=args.stage_topology_ratio,
                    stage_topology_topo_clip=args.stage_topology_topo_clip,
                    structure_profile=args.structure_profile,
                    enable_final_graph_prop=enable_graph_prop,
                    stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
                    stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
                    final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio).cuda()
    
    # 加载模型
    if checkpoint is not None:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            load_topology_checkpoint_state(
                model,
                checkpoint['model_state_dict'],
                checkpoint.get("topology_attention_version", "legacy-unrecorded"),
                strict=(args.bottleneck_type == 'global_local'),
            )
        else:
            model.load_state_dict(checkpoint, strict=(args.bottleneck_type == 'global_local'))
        print(f"已加载模型: {args.model_path}")
        print("Using topology attention constrained version", flush=True)
        print_topology_coefficients(model)
        if isinstance(checkpoint, dict):
            saved_version = checkpoint.get(
                "topology_attention_version",
                "legacy-unrecorded",
            )
            print(
                f"[TOPOLOGY] checkpoint_version={saved_version}; "
                f"runtime_version={TOPOLOGY_ATTENTION_VERSION}",
                flush=True,
            )
    else:
        print(f"错误: 模型文件不存在 {args.model_path}")
        exit(1)
    test_image_dir = os.path.join(args.root_path, 'test', 'image')
    test_label_dir = os.path.join(args.root_path, 'test', 'mask')
    if not os.path.exists(test_label_dir):
        test_label_dir = os.path.join(args.root_path, 'test', 'label')
    
    print(f"加载测试数据...")
    print(f"  Image目录: {test_image_dir}")
    print(f"  Label目录: {test_label_dir}")
    print(f"  阈值: {args.threshold}")
    
    # 评估
    print("\n开始推理...")
    model.eval()

    total_loss = 0
    tp = fp = fn = 0
    total_samples = 0

    if args.overlap_infer:
        # Overlap-tile inference on original images
        import cv2

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
        pred_dir = make_unique_dir(args.output_dir, f'{model_basename}_{timestamp}_overlap')
        os.makedirs(pred_dir, exist_ok=True)
        surface_dir = os.path.join(pred_dir, 'surface')
        os.makedirs(surface_dir, exist_ok=True)

        stride = args.img_size // 2
        image_list = sorted([f for f in os.listdir(test_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        print(f"使用滑窗推理: tile={args.img_size}, stride={stride}, cases={len(image_list)}")

        with torch.no_grad():
            for image_name in tqdm(image_list):
                img_path = os.path.join(test_image_dir, image_name)
                img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape

                # Accumulate logits, then apply sigmoid once on the full image.
                logit_canvas = np.zeros((h, w), dtype=np.float32)
                count_canvas = np.zeros((h, w), dtype=np.float32)

                # sliding
                for y in range(0, max(1, h - args.img_size + 1), stride):
                    for x in range(0, max(1, w - args.img_size + 1), stride):
                        y1 = y
                        x1 = x
                        y2 = min(y1 + args.img_size, h)
                        x2 = min(x1 + args.img_size, w)
                        crop = img[y1:y2, x1:x2]

                        # pad if needed
                        ph = args.img_size - crop.shape[0]
                        pw = args.img_size - crop.shape[1]
                        if ph > 0 or pw > 0:
                            crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode='constant', constant_values=0)

                        # preprocess
                        crop_f = crop.astype(np.float32) / 255.0
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        crop_f = (crop_f - mean) / std
                        crop_f = crop_f.transpose(2, 0, 1)
                        inp = torch.from_numpy(crop_f).unsqueeze(0).cuda()

                        outputs = model(inp)
                        surface_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        tile_logits = surface_logits[0, 0].cpu().numpy()

                        # crop to original size if padded
                        tile_logits = tile_logits[:(y2 - y1), :(x2 - x1)]

                        logit_canvas[y1:y2, x1:x2] += tile_logits
                        count_canvas[y1:y2, x1:x2] += 1.0

                full_logits = logit_canvas / np.maximum(count_canvas, 1.0)
                full_prob = 1.0 / (1.0 + np.exp(-full_logits))
                pred = (full_prob >= args.threshold).astype(np.uint8) * 255

                # save
                case_name = os.path.splitext(image_name)[0]
                png_save_path = os.path.join(surface_dir, f'{case_name}_pred.png')
                cv2.imwrite(png_save_path, pred)

                total_samples += 1

        print(f"滑窗推理完成，结果保存在: {pred_dir}")
        # exit after sliding infer
        sys.exit(0)

    # 非滑窗情况：使用 Dataset + DataLoader（会对图像做 resize 到 args.img_size）
    test_dataset = RoadSkeletonDataset(root_dir=args.root_path, split='test', image_size=args.img_size, source_patch_size=args.source_patch_size)

    print(f"测试集大小: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建预测结果目录（包含模型名和时间戳，保留历史结果）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用模型文件名作为标识，方便对比不同checkpoint的结果
    model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    pred_dir = make_unique_dir(args.output_dir, f'{model_basename}_{timestamp}')
    os.makedirs(pred_dir, exist_ok=True)
    surface_dir = os.path.join(pred_dir, 'surface')
    skeleton_dir = os.path.join(pred_dir, 'skeleton')
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)
    
    criterion = SurfaceStructureLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=0.02,
        connectivity_weight=0.03,
        connectivity_erode_kernel_size=1,
        skeleton_cldice_weight=0.01,
        skeleton_cldice_iterations=10,
        boundary_weight=0.03,
        boundary_radius=1,
        stage_structure_weights=(0.0, 0.0, 0.0, 0.0),
        stage_connectivity_factor=0.5,
        stage_distill_weights=(0.0, 0.0),
        stage_distill_connectivity_factor=0.5,
    ).cuda()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            images = batch['image'].cuda()
            masks = batch['mask'].cuda()
            skeletons = batch['skeleton'].cuda()
            skeletons_dilate = batch['skeleton_dilate'].cuda()
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                (
                    surface_logits,
                    boundary_logits,
                    skeleton_logits,
                    connectivity_logits,
                    stage_outputs,
                ) = outputs[:5]
            else:
                raise RuntimeError("Structure-guided test requires auxiliary model outputs.")
            
            loss, _ = criterion(
                surface_logits,
                surface_gt=masks,
                skeleton_gt=skeletons,
                skeleton_dilate_gt=skeletons_dilate,
                stage_outputs=stage_outputs,
                boundary_logits=boundary_logits,
                skeleton_logits=skeleton_logits,
                connectivity_logits=connectivity_logits,
            )
            total_loss += loss.item()
            
            pred = (torch.sigmoid(surface_logits) >= args.threshold).float()
            masks = (masks > 0.5).float()
            
            case_tp = int((pred * masks).sum().item())
            case_fp = int((pred * (1.0 - masks)).sum().item())
            case_fn = int(((1.0 - pred) * masks).sum().item())
            tp += case_tp
            fp += case_fp
            fn += case_fn
            
            total_samples += 1
            
            # 自动保存预测结果（NPY 格式用于计算，PNG 格式便于查看）
            case_name = batch['case_name'][0]
            
            # 仅保存 PNG：最多保留 30 张代表性预测（按前 30 个 case）
            prob_numpy = torch.sigmoid(surface_logits).squeeze(0).squeeze(0).cpu().numpy()
            if skeleton_logits is not None:
                skeleton_prob = (
                    torch.sigmoid(skeleton_logits)
                    .squeeze(0)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            else:
                skeleton_prob = None
            if args.source_patch_size and args.source_patch_size != args.img_size:
                prob_resized = np.array(
                    Image.fromarray(prob_numpy.astype(np.float32), mode='F').resize(
                        (args.source_patch_size, args.source_patch_size),
                        Image.BILINEAR,
                    ),
                    dtype=np.float32,
                )
                pred_numpy = (prob_resized >= args.threshold).astype(np.uint8) * 255
                if skeleton_prob is not None:
                    skeleton_prob_resized = np.array(
                        Image.fromarray(skeleton_prob.astype(np.float32), mode='F').resize(
                            (args.source_patch_size, args.source_patch_size),
                            Image.BILINEAR,
                        ),
                        dtype=np.float32,
                    )
                else:
                    skeleton_prob_resized = None
            else:
                pred_numpy = (prob_numpy >= args.threshold).astype(np.uint8) * 255
                skeleton_prob_resized = skeleton_prob
            pred_img = Image.fromarray(pred_numpy, mode='L')
            # 只保存前 30 张到 surface_dir
            if total_samples <= 30:
                png_save_path = os.path.join(surface_dir, f'{case_name}_pred.png')
                pred_img.save(png_save_path)
                if skeleton_prob_resized is not None:
                    skeleton_pred_img = Image.fromarray(
                        (skeleton_prob_resized >= args.skeleton_threshold).astype(np.uint8) * 255,
                        mode='L',
                    )
                    skeleton_pred_path = os.path.join(
                        skeleton_dir,
                        f'{case_name}_skeleton_pred.png',
                    )
                    skeleton_pred_img.save(skeleton_pred_path)
    
    print(f"预测结果已保存到: {pred_dir}")
    
    avg_loss = total_loss / len(test_loader)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # 打印到控制台和 txt 日志
    result_lines = []
    result_lines.append(f"\n测试完成!")
    result_lines.append(f"  平均损失: {avg_loss:.4f}")
    result_lines.append(f"  IoU: {iou:.4f}")
    result_lines.append(f"  F1: {f1:.4f}")
    result_lines.append(f"  Precision: {precision:.4f}")
    result_lines.append(f"  Recall: {recall:.4f}")
    result_lines.append(f"  Final skeleton threshold: {args.skeleton_threshold:.2f}")
    result_lines.append(f"  总测试样本: {len(test_loader)}")
    result_lines.append(f"  预测掩码保存位置: {pred_dir}")
    
    result_lines.append(f"  Surface mask save dir: {surface_dir}")
    result_lines.append(f"  Final skeleton save dir: {skeleton_dir}")
    result_lines.append(
        f"  Topology attention version: {TOPOLOGY_ATTENTION_VERSION}"
    )
    result_lines.append(f"  {format_topology_coefficients(model)}")

    for line in result_lines:
        print(line, flush=True)
    
    # 保存测试结果到 txt 文件（保存到带时间戳的文件夹下）
    test_log_path = os.path.join(pred_dir, 'test_results.txt')
    with open(test_log_path, 'w', encoding='utf-8') as log_f:
        log_f.write("="*80 + "\n")
        log_f.write("测试结果\n")
        log_f.write("="*80 + "\n")
        log_f.write(f"时间戳: {timestamp}\n")
        log_f.write(f"模型路径: {args.model_path}\n")
        log_f.write(f"阈值: {args.threshold}\n")
        log_f.write(f"Final skeleton threshold: {args.skeleton_threshold}\n")
        log_f.write(f"输入图像大小: {args.img_size}\n")
        log_f.write(f"测试数据路径: {args.root_path}\n")
        log_f.write("="*80 + "\n")
        for line in result_lines[1:]:  # 跳过第一行"测试完成"
            log_f.write(line + "\n")
        log_f.write("="*80 + "\n")
    
    print(f"测试日志已保存到: {test_log_path}", flush=True)
    
    if args.is_savenii:
        print(f"  预测结果保存到: {args.output_dir}")
