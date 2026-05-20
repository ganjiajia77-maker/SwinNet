#!/usr/bin/env python
# 快速命令参考 - 使用data目录进行训练和测试

# ============================================================
# 1. 训练模型 (使用data目录, 1500个train样本)
# ============================================================

# 默认参数训练（推荐新手使用）
# python train_image.py

# 自定义参数训练示例
# python train_image.py --batch_size 16 --max_epochs 50 --base_lr 0.001

# ============================================================
# 2. 测试模型 (使用data目录, 400个test样本)
# ============================================================

# 基础测试
# python test_image.py --model_path ./model_out/epoch_50.pth

# 保存预测结果
# python test_image.py --model_path ./model_out/epoch_50.pth --is_savenii


# ============================================================
# PowerShell 执行示例
# ============================================================

# 训练命令
# python train_image.py --root_path ./data --batch_size 12 --max_epochs 100 --output_dir ./model_out

# 测试命令  
# python test_image.py --root_path ./data --model_path ./model_out/epoch_100.pth --output_dir ./predictions --is_savenii


# ============================================================
# 其他常用数据集命令
# ============================================================

# 使用data1目录（完整数据集 4980+1246）
# python train_image.py --root_path ./data1

# 使用Synapse原始数据集
# python train.py --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml --root_path ../data/Synapse/train_npz


# ============================================================
# GPU和并行处理设置
# ============================================================

# 指定使用GPU 0
# set CUDA_VISIBLE_DEVICES=0

# 使用多个GPU
# set CUDA_VISIBLE_DEVICES=0,1

# 增加数据加载线程数（加快读取速度）
# python train_image.py --num_workers 16
