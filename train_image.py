import argparse
import csv
import math
import os
import random
import sys
import tempfile
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader

from networks.vision_transformer import (
    TOPOLOGY_ATTENTION_VERSION,
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    freeze_backbone_train_graph_only,
    get_topology_coefficients,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)
from datasets.dataset_synapse import ImageDataset, RandomGenerator
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data1', help='root dir for data')
parser.add_argument('--dataset', type=str, default='ImageData', help='dataset name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='./model_out', help='output dir')
parser.add_argument('--run_name', type=str, default='', help='optional run folder name under output_dir')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=5e-4, help='segmentation network learning rate')
parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate for cosine decay')
parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup epochs before cosine decay')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='network input size after downsampling from a 1024 patch')
parser.add_argument('--source_patch_size', type=int, default=1024, help='source patch size before resizing to img_size')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', 
                    help='path to config file')
parser.add_argument('--n_class', default=2, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--print_freq', default=10, type=int, help='print loss every N batches')
parser.add_argument('--threshold', default=0.2, type=float, help='binary threshold for validation')
parser.add_argument('--skeleton_threshold', default=0.5, type=float, help='final 256x256 skeleton threshold for validation')
parser.add_argument('--final_topology_eta_init', default=0.005, type=float, help='initial final 256 topology repair coefficient')
parser.add_argument('--final_gap_rho_init', default=0.005, type=float, help='initial localized gap-repair coefficient')
parser.add_argument(
    '--stage_topology_stages',
    type=str,
    default='none',
    choices=['none', 'stage3', 'stage23'],
    help='which decoder stages get topology attention',
)
parser.add_argument('--stage_topology_alpha_max', type=float, default=1.0)
parser.add_argument('--stage_topology_alpha_init', type=float, default=0.1)
parser.add_argument(
    '--stage_topology_bias_mode',
    type=str,
    default='pairwise_skeleton',
    choices=['pairwise_skeleton', 'gap_query'],
    help='stage topology bias construction mode',
)
parser.add_argument('--stage_topology_ratio', type=float, default=0.08)
parser.add_argument('--stage_topology_topo_clip', type=float, default=4.0)
parser.add_argument('--stage_topology_warmup_epochs', type=int, default=5)
parser.add_argument(
    '--stage_topology_teacher_forcing_end',
    type=int,
    default=15,
    help='epoch at which teacher forcing ratio drops to 0',
)
parser.add_argument('--stage3_skeleton_weight', type=float, default=0.005)
parser.add_argument('--stage3_roadness_weight', type=float, default=0.003)
parser.add_argument('--stage2_skeleton_weight', type=float, default=0.0)
parser.add_argument('--road_attention_weight', type=float, default=0.003)
parser.add_argument('--max_train_batches', type=int, default=0, help='limit training to the first N batches per epoch; 0 means no limit')
parser.add_argument('--no_pretrain', action='store_true', help='do not load pretrained weights')
parser.add_argument('--disable_centerline_loss', action='store_true', help='disable the centerline response term for debugging NaN instability')
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
    help='0626 profile: stage2/3 structure only, final skeleton/connectivity off',
)
parser.add_argument(
    '--enable_graph_prop',
    action='store_true',
    help='final surface delta-logit soft graph propagation (stage2/3 priors)',
)
parser.add_argument(
    '--freeze_0626_backbone',
    action='store_true',
    help='freeze 0626 encoder/decoder/heads; train graph_propagation only',
)
parser.add_argument(
    '--graph_corr_weight',
    type=float,
    default=0.05,
    help='continuous baseline-error correction loss weight',
)
parser.add_argument(
    '--graph_corr_k',
    type=float,
    default=1.0,
    help='scale for target_delta = k * (GT - P_base)',
)
parser.add_argument(
    '--graph_corr_m_pos',
    type=float,
    default=0.15,
    help='max positive target delta for graph residual',
)
parser.add_argument(
    '--graph_corr_m_neg',
    type=float,
    default=0.15,
    help='max negative target delta magnitude for graph residual',
)
parser.add_argument(
    '--graph_fn_push_weight',
    type=float,
    default=0.0,
    help='deprecated; use --graph_corr_weight',
)
parser.add_argument(
    '--graph_fp_suppress_weight',
    type=float,
    default=0.0,
    help='deprecated; use --graph_corr_weight',
)
parser.add_argument(
    '--graph_delta_sparse_weight',
    type=float,
    default=0.0,
    help='deprecated; use --graph_corr_weight',
)
parser.add_argument(
    '--graph_base_lr',
    type=float,
    default=3e-4,
    help='learning rate for graph-only training when backbone is frozen',
)
DEFAULT_0626_CHECKPOINT = (
    './model_out/train_stage23_structure_final_boundary_nw0_20260626/best.pth'
)
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


def _cli_has(flag):
    return any(
        arg == flag or arg.startswith(flag + "=")
        for arg in sys.argv[1:]
    )


def apply_structure_profile_defaults(args):
    if args.structure_profile != STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626:
        return

    args.stage_topology_stages = "none"
    if not _cli_has("--stage2_skeleton_weight"):
        args.stage2_skeleton_weight = 0.004
    if not _cli_has("--stage3_skeleton_weight"):
        args.stage3_skeleton_weight = 0.006
    args.stage3_roadness_weight = 0.0
    args.final_topology_eta_init = 0.0
    args.final_gap_rho_init = 0.0
    if args.warmup_epochs == 3:
        args.warmup_epochs = 10
    if args.max_epochs == 100:
        args.max_epochs = 60


apply_structure_profile_defaults(args)

if args.freeze_0626_backbone and not args.enable_graph_prop:
    parser.error("--freeze_0626_backbone requires --enable_graph_prop")
if args.freeze_0626_backbone and not args.resume:
    args.resume = DEFAULT_0626_CHECKPOINT


def get_final_loss_weights(args):
    if args.structure_profile == STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626:
        return {
            "skeleton_weight": 0.0,
            "connectivity_weight": 0.0,
            "skeleton_cldice_weight": 0.0,
            "boundary_weight": 0.01,
        }
    return {
        "skeleton_weight": 0.02,
        "connectivity_weight": 0.03,
        "skeleton_cldice_weight": 0.01,
        "boundary_weight": 0.03,
    }

def get_graph_outputs_from_model(model):
    module = model.module if hasattr(model, "module") else model
    head = module.swin_unet.guided_head
    return (
        head.last_surface_pre_logits,
        head.last_delta_logit,
        head.last_graph_delta,
    )


def build_criterion(args, loss_weights, device):
    stage2_weight = 0.0 if args.freeze_0626_backbone else args.stage2_skeleton_weight
    stage3_weight = 0.0 if args.freeze_0626_backbone else args.stage3_skeleton_weight
    boundary_weight = 0.0 if args.freeze_0626_backbone else loss_weights["boundary_weight"]
    graph_corr = args.graph_corr_weight if args.enable_graph_prop else 0.0
    if args.freeze_0626_backbone:
        graph_corr = args.graph_corr_weight

    return SurfaceStructureLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=loss_weights["skeleton_weight"],
        connectivity_weight=loss_weights["connectivity_weight"],
        connectivity_erode_kernel_size=1,
        skeleton_cldice_weight=loss_weights["skeleton_cldice_weight"],
        skeleton_cldice_iterations=10,
        boundary_weight=boundary_weight,
        boundary_radius=1,
        stage_structure_weights=(
            0.0,
            0.0,
            stage2_weight,
            stage3_weight,
        ),
        stage_roadness_weights=(0.0, 0.0, 0.0, args.stage3_roadness_weight),
        road_attention_weight=0.0 if args.freeze_0626_backbone else args.road_attention_weight,
        stage_connectivity_factor=0.5,
        stage_distill_weights=(0.0, 0.0),
        stage_distill_connectivity_factor=0.5,
        use_legacy_stage_connectivity_loss=(
            args.structure_profile == STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626
        ),
        graph_corr_weight=graph_corr,
        graph_corr_k=args.graph_corr_k,
        graph_corr_m_pos=args.graph_corr_m_pos,
        graph_corr_m_neg=args.graph_corr_m_neg,
    ).to(device)


def format_training_config_lines(args, loss_weights):
    lines = [
        f"  结构配置: {args.structure_profile}",
    ]
    if args.structure_profile == STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626:
        lines.extend([
            "  Structure head: decoder stage2/stage3 skeleton + top-2 connectivity residual attention",
            "  Final skeleton/connectivity heads: disabled",
            "  Stage2 structure loss weight: {:.3f}".format(args.stage2_skeleton_weight),
            "  Stage3 structure loss weight: {:.3f}".format(args.stage3_skeleton_weight),
            "  Stage loss: skeleton BCE(dilated) + 0.3 Dice(hard) + 0.5 connectivity "
            "(corridor-weighted BCE + edge Dice + symmetry)",
            "  Encoder road attention: stage2 prior -> F*(1+alpha*A), loss weight {:.3f}".format(
                args.road_attention_weight
            ),
            "  Global context calibration: bottleneck GAP -> stage3 structure gate only",
            "  Global context gate strength: 0.03",
            "  Stage topology attention: none",
        ])
        if args.enable_graph_prop:
            lines.extend([
                "  Final graph propagation: stage2/3 priors -> delta-logit residual",
                "  Graph propagation lambda: init=0.05, max=0.10, edge_beta=0.7",
                "  Graph delta_logit: clamp(-0.3, 0.3) before lambda*G residual",
                "  Graph G: learned gate_mlp(P, weak, two_sided_support, near_H, H, S, C)",
                "  Graph support: sqrt(H_l*H_r) * mean(C_l,C_r); candidate=weak*(1-H)",
                "  Graph correction loss: weighted SmoothL1(delta_logit, k*(GT-P_base))",
            ])
            if args.freeze_0626_backbone:
                lines.append(
                    "  Backbone: frozen 0626 (train graph_propagation + lambda only)"
                )
    else:
        lines.extend([
            "  Decoder structure gates: restored 0621 stages 0/1/2/3",
            "  Final 256 structure-only topology refiner: "
            f"eta_init={args.final_topology_eta_init}, eta_max=0.05, tau=4",
            "  Localized gap repair: "
            f"rho_init={args.final_gap_rho_init}, rho_max=0.05, detached M_gap",
            "  Stage topology attention: "
            f"{args.stage_topology_stages}, mode={args.stage_topology_bias_mode}, "
            f"ratio={args.stage_topology_ratio}, topo_clip={args.stage_topology_topo_clip}, "
            f"alpha_init={args.stage_topology_alpha_init}, "
            f"alpha_max={args.stage_topology_alpha_max}, "
            f"warmup={args.stage_topology_warmup_epochs}",
            "  Stage topology teacher forcing: "
            f"epoch 0 -> {args.stage_topology_teacher_forcing_end}",
        ])
    lines.extend([
        "  Stage structure weights: "
        f"stage2={args.stage2_skeleton_weight}, "
        f"stage3={args.stage3_skeleton_weight}, "
        f"stage3_roadness={args.stage3_roadness_weight}, "
        f"road_attention={args.road_attention_weight}",
        "  Surface target resize: 1024 -> 256 nearest-neighbor",
        "  Boundary-aware refinement: enabled",
        "  Boundary loss weight: {:.2f}".format(loss_weights["boundary_weight"]),
        "  Boundary target: dilate(mask, r=1) - erode(mask, k=3)",
        "  Final skeleton loss weight: {:.2f}".format(loss_weights["skeleton_weight"]),
        "  Final connectivity loss weight: {:.2f}".format(loss_weights["connectivity_weight"]),
        "  Final skeleton clDice weight: {:.2f}".format(loss_weights["skeleton_cldice_weight"]),
        "  Edge loss: disabled",
        "  Edge skip enhance: disabled",
    ])
    return lines


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


def save_checkpoint_safely(checkpoint, target_path):
    """Save directly on Windows to avoid orphaned temporary checkpoint files."""
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    torch.save(checkpoint, target_path, _use_new_zipfile_serialization=False)


def model_state_is_finite(model):
    for param in model.state_dict().values():
        if torch.is_tensor(param) and not torch.isfinite(param).all():
            return False
    return True


def get_cosine_warmup_lr(epoch, max_epochs, base_lr, min_lr, warmup_epochs):
    if max_epochs <= 0:
        return base_lr

    warmup_epochs = max(0, warmup_epochs)
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)

    decay_epochs = max(1, max_epochs - warmup_epochs)
    if decay_epochs == 1:
        return min_lr

    progress = min(float(epoch - warmup_epochs) / float(decay_epochs - 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def get_stage_distill_scale(epoch):
    epoch_number = epoch + 1
    if epoch_number <= 5:
        return 0.0
    if epoch_number < 10:
        return (epoch_number - 5) / 5.0
    return 1.0


def get_stage_topology_alpha_scale(epoch, warmup_epochs=5):
    if warmup_epochs <= 0:
        return 1.0
    if epoch < warmup_epochs:
        return 0.0
    return min(1.0, (epoch - warmup_epochs + 1) / warmup_epochs)


def get_teacher_forcing_ratio(epoch, tf_start=0, tf_end=15):
    if tf_end <= tf_start:
        return 0.0 if epoch >= tf_end else 1.0
    if epoch < tf_start:
        return 1.0
    if epoch >= tf_end:
        return 0.0
    return 1.0 - (epoch - tf_start) / (tf_end - tf_start)


def pad_to_window_multiple(x, window_size=7):
    """
    Pad input tensor to be divisible by window_size.
    Returns: (padded_tensor, original_shape)
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return x, (H, W)


def crop_to_shape(x, target_shape):
    """Crop tensor back to target shape."""
    if x is None:
        return None
    H, W = target_shape
    return x[:, :, :H, :W]

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

def load_split(split_name, train_mode=True, root_path='.', img_size=512, num_workers=4):
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
            images = batch['image'].to(device)
            labels = batch['label'].long().to(device)

            outputs = model(images)
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

def evaluate_skeleton(
    model,
    loader,
    criterion,
    threshold=0.2,
    skeleton_threshold=0.5,
    stage_distill_scale=1.0,
    stage_topology_alpha_scale=1.0,
):
    model.eval()
    total_loss = 0.0
    surface_tp = surface_fp = surface_fn = 0
    skeleton_tp = skeleton_fp = skeleton_fn = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            skeletons = batch['skeleton'].to(device)
            skeletons_dilate = batch['skeleton_dilate'].to(device)

            images_padded, orig_shape = pad_to_window_multiple(images, window_size=8)
            masks_padded, _ = pad_to_window_multiple(masks, window_size=8)
            skeletons_padded, _ = pad_to_window_multiple(skeletons, window_size=8)
            skeletons_dilate_padded, _ = pad_to_window_multiple(skeletons_dilate, window_size=8)
            
            outputs = model(
                images_padded,
                topology_alpha_scale=stage_topology_alpha_scale,
                teacher_forcing_ratio=0.0,
            )

            if isinstance(outputs, tuple):
                (
                    surface_logits,
                    boundary_logits,
                    skeleton_logits,
                    connectivity_logits,
                    stage_outputs,
                ) = outputs[:5]
            else:
                raise RuntimeError("Structure-guided training requires auxiliary model outputs.")
            surface_logits = crop_to_shape(surface_logits, orig_shape)
            boundary_logits = crop_to_shape(boundary_logits, orig_shape)
            skeleton_logits = crop_to_shape(skeleton_logits, orig_shape)
            connectivity_logits = crop_to_shape(connectivity_logits, orig_shape)
            
            masks_padded = crop_to_shape(masks_padded, orig_shape)
            skeletons_padded = crop_to_shape(skeletons_padded, orig_shape)
            skeletons_dilate_padded = crop_to_shape(skeletons_dilate_padded, orig_shape)

            graph_base_logits, graph_delta_logit, graph_delta = (
                get_graph_outputs_from_model(model)
            )
            if graph_base_logits is not None:
                graph_base_logits = crop_to_shape(graph_base_logits, orig_shape)
            if graph_delta_logit is not None:
                graph_delta_logit = crop_to_shape(graph_delta_logit, orig_shape)

            loss, _ = criterion(
                surface_logits,
                surface_gt=masks_padded,
                skeleton_gt=skeletons_padded,
                skeleton_dilate_gt=skeletons_dilate_padded,
                stage_outputs=stage_outputs,
                boundary_logits=boundary_logits,
                skeleton_logits=skeleton_logits,
                connectivity_logits=connectivity_logits,
                stage_distill_scale=stage_distill_scale,
                graph_base_logits=graph_base_logits,
                graph_delta_logit=graph_delta_logit,
            )
            total_loss += loss.item()

            surface_pred = (torch.sigmoid(surface_logits) >= threshold).float().squeeze(1)
            masks_bin = (masks_padded > 0.5).float().squeeze(1)

            surface_tp += int((surface_pred * masks_bin).sum().item())
            surface_fp += int((surface_pred * (1.0 - masks_bin)).sum().item())
            surface_fn += int(((1.0 - surface_pred) * masks_bin).sum().item())

            if skeleton_logits is not None:
                skeleton_pred = (
                    torch.sigmoid(skeleton_logits) >= skeleton_threshold
                ).float().squeeze(1)
                skeletons_bin = (skeletons_padded > 0.5).float().squeeze(1)
                skeleton_tp += int((skeleton_pred * skeletons_bin).sum().item())
                skeleton_fp += int((skeleton_pred * (1.0 - skeletons_bin)).sum().item())
                skeleton_fn += int(((1.0 - skeleton_pred) * skeletons_bin).sum().item())

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
    # 自动检测可用设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
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
    
    # 对齐 img_size 到 window_size=8 的倍数（模型初始化时需要）
    window_size = 8
    aligned_img_size = ((args.img_size + window_size - 1) // window_size) * window_size
    if aligned_img_size != args.img_size:
        print(f"[INFO] 对齐 img_size: {args.img_size} -> {aligned_img_size} (window_size={window_size})")
        config.defrost()
        config.DATA.IMG_SIZE = aligned_img_size
        config.freeze()
        args.img_size = aligned_img_size
    
    # 创建网络（启用 DilatedAsterisk 空间特征增强）
    args.num_classes = 1
    loss_weights = get_final_loss_weights(args)
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
                    enable_final_graph_prop=args.enable_graph_prop).to(device)

    # 加载数据
    train_dataset = RoadSkeletonDataset(root_dir=args.root_path, split='train', image_size=args.img_size, source_patch_size=args.source_patch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_dataset = RoadSkeletonDataset(root_dir=args.root_path, split='val', image_size=args.img_size, source_patch_size=args.source_patch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 优化器 / 损失（graph-only 模式在加载 checkpoint 后再建 optimizer）
    criterion = build_criterion(args, loss_weights, device)
    optimizer = None

    # 检查是否恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            if args.freeze_0626_backbone:
                load_topology_checkpoint_state(
                    model,
                    checkpoint["model_state_dict"],
                    checkpoint.get("topology_attention_version", "legacy-unrecorded"),
                )
            else:
                strict_load = args.bottleneck_type == 'global_local'
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=strict_load)
                except RuntimeError as exc:
                    allowed_missing_prefixes = (
                        "swin_unet.stage2_topology_source.",
                        "swin_unet.stage_topology_scales.",
                        "swin_unet.decoder_structure_blocks.3.stage_roadness_head.",
                        "swin_unet.guided_head.graph_propagation.",
                    )
                    result = model.load_state_dict(
                        checkpoint['model_state_dict'],
                        strict=False,
                    )
                    invalid_missing = [
                        key
                        for key in result.missing_keys
                        if not key.startswith(allowed_missing_prefixes)
                    ]
                    if invalid_missing or result.unexpected_keys:
                        raise RuntimeError(
                            "Checkpoint mismatch on resume: "
                            f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
                        ) from exc
                    print(
                        "[WARN] Loaded checkpoint with strict=False; "
                        "some modules use fresh initialization.",
                        flush=True,
                    )
            if args.freeze_0626_backbone:
                trainable = freeze_backbone_train_graph_only(model)
                print(
                    "[INFO] Frozen 0626 backbone; trainable graph params: "
                    f"{len(trainable)} tensors",
                    flush=True,
                )
                # 0626 ckpt epoch 可能 > max_epochs；graph-only 是新实验，从 0 开始
                start_epoch = 0
                print(
                    "[INFO] Graph-only fine-tune: reset start_epoch to 0 "
                    f"(checkpoint had epoch={checkpoint.get('epoch', '?')})",
                    flush=True,
                )
            else:
                start_epoch = checkpoint.get('epoch', 0)
            if not args.freeze_0626_backbone:
                try:
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=args.base_lr,
                        weight_decay=0.0001,
                    )
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except (ValueError, RuntimeError) as exc:
                    print(
                        "[WARN] Optimizer state not restored; "
                        "continuing with a fresh optimizer for current parameters.",
                        flush=True,
                    )
                    print(f"[WARN] Optimizer resume detail: {exc}", flush=True)
            print(f"从epoch {start_epoch} 继续训练...")
        else:
            print(f"Warning: checkpoint未找到 {args.resume}")

    if optimizer is None:
        if args.freeze_0626_backbone:
            graph_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                graph_params,
                lr=args.graph_base_lr,
                weight_decay=0.0001,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.base_lr,
                weight_decay=0.0001,
            )

    # 训练循环
    print("\n开始训练...")
    print(f"配置:")
    print(f"  最大轮数: {args.max_epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.base_lr}")
    print(f"  最小学习率: {args.min_lr}")
    print(f"  Warmup轮数: {args.warmup_epochs}")
    for line in format_training_config_lines(args, loss_weights):
        print(line)
    print(f"  输出目录: {args.output_dir}")
    print(f"  Checkpoints目录: {checkpoints_dir}")
    print(f"  验证阈值: {args.threshold}")
    if args.max_train_batches > 0:
        print(f"  每轮最多训练batch数: {args.max_train_batches}")
    if args.resume:
        print(f"  从checkpoint恢复: {args.resume}")
        print(f"  起始epoch: {start_epoch}")

    print("Using topology attention constrained version", flush=True)
    print_topology_coefficients(model)

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
        log_f.write(f"  最小学习率: {args.min_lr}\n")
        log_f.write(f"  Warmup轮数: {args.warmup_epochs}\n")
        for line in format_training_config_lines(args, loss_weights):
            log_f.write(line + "\n")
        log_f.write(f"  验证阈值: {args.threshold}\n")
        log_f.write(f"  数据目录: {args.root_path}\n")
        if args.max_train_batches > 0:
            log_f.write(f"  每轮最多训练batch数: {args.max_train_batches}\n")
        log_f.write("="*100 + "\n\n")
    
    with open(loss_log_path, 'w', newline='', encoding='utf-8') as loss_log_file, \
         open(batch_loss_log_path, 'w', newline='', encoding='utf-8') as batch_loss_log_file:
        loss_writer = csv.writer(loss_log_file)
        batch_loss_writer = csv.writer(batch_loss_log_file)
        loss_writer.writerow([
            'epoch', 'lr', 'train_avg_loss', 'val_loss',
            'surface_iou', 'surface_f1', 'surface_precision', 'surface_recall',
            'skeleton_iou', 'skeleton_f1', 'skeleton_precision', 'skeleton_recall'
        ])
        batch_loss_writer.writerow(['epoch', 'batch', 'loss'])

        for epoch in range(start_epoch, args.max_epochs):
            current_lr = get_cosine_warmup_lr(
                epoch,
                args.max_epochs,
                args.base_lr,
                args.min_lr,
                args.warmup_epochs,
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            total_loss = 0
            train_batches = 0
            skipped_batches = 0
            stage_distill_scale = get_stage_distill_scale(epoch)
            stage_topology_alpha_scale = get_stage_topology_alpha_scale(
                epoch,
                args.stage_topology_warmup_epochs,
            )
            teacher_forcing_ratio = get_teacher_forcing_ratio(
                epoch,
                tf_start=0,
                tf_end=args.stage_topology_teacher_forcing_end,
            )

            for i, batch in enumerate(train_loader):
                if args.max_train_batches > 0 and train_batches >= args.max_train_batches:
                    break

                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                skeletons = batch['skeleton'].to(device)
                skeletons_dilate = batch['skeleton_dilate'].to(device)

                images_padded, orig_shape = pad_to_window_multiple(images, window_size=8)
                masks_padded, _ = pad_to_window_multiple(masks, window_size=8)
                skeletons_padded, _ = pad_to_window_multiple(skeletons, window_size=8)
                skeletons_dilate_padded, _ = pad_to_window_multiple(skeletons_dilate, window_size=8)

                outputs = model(
                    images_padded,
                    gt_skeleton=skeletons_padded,
                    topology_alpha_scale=stage_topology_alpha_scale,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )

                if isinstance(outputs, tuple):
                    (
                        surface_logits,
                        boundary_logits,
                        skeleton_logits,
                        connectivity_logits,
                        stage_outputs,
                    ) = outputs[:5]
                else:
                    raise RuntimeError("Structure-guided training requires auxiliary model outputs.")
                surface_logits = crop_to_shape(surface_logits, orig_shape)
                boundary_logits = crop_to_shape(boundary_logits, orig_shape)
                skeleton_logits = crop_to_shape(skeleton_logits, orig_shape)
                connectivity_logits = crop_to_shape(connectivity_logits, orig_shape)
                
                masks_padded = crop_to_shape(masks_padded, orig_shape)
                skeletons_padded = crop_to_shape(skeletons_padded, orig_shape)
                skeletons_dilate_padded = crop_to_shape(skeletons_dilate_padded, orig_shape)
                
                graph_base_logits, graph_delta_logit, graph_delta = (
                    get_graph_outputs_from_model(model)
                )
                loss, loss_dict = criterion(
                    surface_logits,
                    surface_gt=masks_padded,
                    skeleton_gt=skeletons_padded,
                    skeleton_dilate_gt=skeletons_dilate_padded,
                    stage_outputs=stage_outputs,
                    boundary_logits=boundary_logits,
                    skeleton_logits=skeleton_logits,
                    connectivity_logits=connectivity_logits,
                    stage_distill_scale=stage_distill_scale,
                    graph_base_logits=graph_base_logits,
                    graph_delta_logit=graph_delta_logit,
                )

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    print(
                        f"[WARN] Non-finite loss skipped at epoch {epoch+1}, batch {i+1}: {loss.item()}",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad()
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    skipped_batches += 1
                    print(
                        f"[WARN] Non-finite gradients skipped at epoch {epoch+1}, batch {i+1}: {grad_norm}",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                total_loss += loss.item()
                train_batches += 1
                batch_loss_writer.writerow([epoch + 1, i + 1, f'{loss.item():.6f}'])
                batch_loss_log_file.flush()

                if (i + 1) % args.print_freq == 0 or i == 0:
                    print(
                        f"Epoch [{epoch+1}/{args.max_epochs}], Batch [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}, Surface: {loss_dict['surface_loss'].item():.4f}, "
                        f"Skeleton: {loss_dict['skeleton_loss'].item():.4f}, "
                        f"Conn: {loss_dict['connectivity_loss'].item():.4f}, "
                        f"ClDice: {loss_dict['skeleton_cldice_loss'].item():.4f}, "
                        f"StageStruct: {loss_dict['stage_structure_loss'].item():.4f}, "
                        f"StageRoad: {loss_dict['stage_roadness_loss'].item():.4f}, "
                        f"RoadAttn: {loss_dict['road_attention_loss'].item():.4f}, "
                        f"StageKD: {loss_dict['stage_distill_loss'].item():.4f}"
                        f"x{stage_distill_scale:.2f}, "
                        f"TopoAlphaScale: {stage_topology_alpha_scale:.2f}, "
                        f"TF: {teacher_forcing_ratio:.2f}, "
                        f"Boundary: {loss_dict['boundary_loss'].item():.4f}, "
                        f"GraphCorr: {loss_dict['graph_corr_loss'].item():.4f}",
                        flush=True
                    )

            train_avg_loss = total_loss / max(train_batches, 1)
            val_metrics = evaluate_skeleton(
                model,
                val_loader,
                criterion,
                args.threshold,
                args.skeleton_threshold,
                stage_distill_scale,
                stage_topology_alpha_scale,
            )
            val_loss = val_metrics['loss']
            val_iou = val_metrics['surface_iou']
            val_f1 = val_metrics['surface_f1']
            val_precision = val_metrics['surface_precision']
            val_recall = val_metrics['surface_recall']
            skeleton_iou = val_metrics['skeleton_iou']
            skeleton_f1 = val_metrics['skeleton_f1']
            skeleton_precision = val_metrics['skeleton_precision']
            skeleton_recall = val_metrics['skeleton_recall']
            
            # 打印到控制台
            epoch_msg = (
                f"Epoch {epoch+1}/{args.max_epochs}, LR: {current_lr:.6g}, Train Loss: {train_avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Surface IoU: {val_iou:.4f}, Surface F1: {val_f1:.4f}, "
                f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                f"Skeleton IoU: {skeleton_iou:.4f}, Skeleton F1: {skeleton_f1:.4f}"
            )
            print(epoch_msg, flush=True)
            topology_msg = print_topology_coefficients(
                model,
                prefix=f"[TOPOLOGY][Epoch {epoch + 1}]",
            )
            if skipped_batches > 0:
                print(f"[WARN] Skipped non-finite batches this epoch: {skipped_batches}", flush=True)
            
            # 写入 CSV
            loss_writer.writerow([
                epoch + 1, f'{current_lr:.8f}', f'{train_avg_loss:.6f}', f'{val_loss:.6f}',
                f'{val_iou:.6f}', f'{val_f1:.6f}', f'{val_precision:.6f}', f'{val_recall:.6f}',
                f'{skeleton_iou:.6f}', f'{skeleton_f1:.6f}', f'{skeleton_precision:.6f}', f'{skeleton_recall:.6f}'
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
                log_f.write(f"  Skeleton IoU: {skeleton_iou:.6f}\n")
                log_f.write(f"  Skeleton F1: {skeleton_f1:.6f}\n")
                log_f.write(f"  Skeleton Precision: {skeleton_precision:.6f}\n")
                log_f.write(f"  Skeleton Recall: {skeleton_recall:.6f}\n")
                log_f.write(
                    f"  Stage topology alpha scale: "
                    f"{stage_topology_alpha_scale:.6f}\n"
                )
                log_f.write(
                    f"  Stage topology teacher forcing ratio: "
                    f"{teacher_forcing_ratio:.6f}\n"
                )
                log_f.write(f"  Skipped non-finite batches: {skipped_batches}\n")
                log_f.write(topology_msg + "\n")
                log_f.write("-"*100 + "\n")

            if not np.isfinite(train_avg_loss) or not np.isfinite(val_loss) or not model_state_is_finite(model):
                print(
                    f"[WARN] Non-finite epoch state at epoch {epoch+1}; checkpoint not saved.",
                    flush=True,
                )
                continue

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
                'skeleton_iou': skeleton_iou,
                'skeleton_f1': skeleton_f1,
                'skeleton_precision': skeleton_precision,
                'skeleton_recall': skeleton_recall,
                'topology_attention_version': TOPOLOGY_ATTENTION_VERSION,
                'structure_profile': args.structure_profile,
                'topology_coefficients': get_topology_coefficients(model),
                'stage_topology_alpha_scale': stage_topology_alpha_scale,
                'stage_topology_teacher_forcing_ratio': teacher_forcing_ratio,
                'args': vars(args),
            }
            # 保存 last.pth（总是覆盖）
            save_checkpoint_safely(checkpoint, os.path.join(args.output_dir, 'last.pth'))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_path = os.path.join(args.output_dir, 'best.pth')
                save_checkpoint_safely(checkpoint, best_path)
                print(f"[BEST] 当前最优模型已保存: best.pth (F1={val_f1:.4f})", flush=True)

    # 训练完成，写总结日志
    with open(training_log_path, 'a', encoding='utf-8') as log_f:
        log_f.write("\n" + "="*100 + "\n")
        log_f.write("训练完成总结\n")
        log_f.write("="*100 + "\n")
        log_f.write(f"总 Epochs: {args.max_epochs}\n")
        log_f.write(f"最佳 F1 分数: {best_val_f1:.6f}\n")
        log_f.write(f"最佳模型: best.pth\n")
        log_f.write(f"最后模型: last.pth\n")
        log_f.write("="*100 + "\n")

    best_path = os.path.join(args.output_dir, 'best.pth')
    if os.path.isfile(best_path):
        best_checkpoint = torch.load(best_path, map_location='cuda')
        model.load_state_dict(best_checkpoint['model_state_dict'], strict=(args.bottleneck_type == 'global_local'))
        best_val_metrics = evaluate_skeleton(
            model,
            val_loader,
            criterion,
            args.threshold,
            args.skeleton_threshold,
            1.0,
        )

        best_eval_msg = (
            f"Best Checkpoint Eval, Val Loss: {best_val_metrics['loss']:.4f}, "
            f"Surface IoU: {best_val_metrics['surface_iou']:.4f}, Surface F1: {best_val_metrics['surface_f1']:.4f}"
        )
        print("\n" + best_eval_msg, flush=True)
        with open(training_log_path, 'a', encoding='utf-8') as log_f:
            log_f.write("\n" + best_eval_msg + "\n")
    
    print("\n训练完成!")
    print(f"   best.pth: {best_path}")
    print(f"   last.pth: {os.path.join(args.output_dir, 'last.pth')}")
    print(f"Training log: {training_log_path}")
