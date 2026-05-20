# 使用 data 目录数据集运行 Swin-Unet

## 项目结构
```
D:\Code\Swin-Unet-main\
├── data/                           # 你准备的数据集（1500+400）
│   ├── train/
│   │   ├── image/  (1500个_sat.jpg)
│   │   └── label/  (1500个_mask.png)
│   └── test/
│       ├── image/  (400个_sat.jpg)
│       └── label/  (400个_mask.png)
├── data1/                          # 完整数据集（4980+1246）
│   ├── train/
│   └── test/
├── train_image.py                  # 新的训练脚本（使用JPG/PNG数据）
├── test_image.py                   # 新的测试脚本（使用JPG/PNG数据）
└── ...
```

## 快速开始

### 1. 训练模型

**基础命令:**
```bash
python train_image.py
```

**自定义参数:**
```bash
python train_image.py \
  --root_path ./data \
  --batch_size 12 \
  --max_epochs 100 \
  --base_lr 0.001 \
  --output_dir ./model_out \
  --img_size 224
```

**参数说明:**
- `--root_path`: 数据集根目录（默认：./data）
- `--batch_size`: 批处理大小（默认：12）
- `--max_epochs`: 训练轮数（默认：100）
- `--base_lr`: 学习率（默认：0.001）
- `--output_dir`: 模型输出目录（默认：./model_out）
- `--img_size`: 输入图像大小（默认：224）
- `--num_workers`: 数据加载线程数（默认：4）

### 2. 测试模型

**基础命令:**
```bash
python test_image.py --model_path ./model_out/epoch_100.pth
```

**自定义参数:**
```bash
python test_image.py \
  --root_path ./data \
  --model_path ./model_out/epoch_100.pth \
  --batch_size 24 \
  --output_dir ./predictions \
  --is_savenii
```

**参数说明:**
- `--root_path`: 数据集根目录（默认：./data）
- `--model_path`: 模型检查点路径（必需）
- `--batch_size`: 批处理大小（默认：24）
- `--output_dir`: 预测结果保存目录（默认：./predictions）
- `--is_savenii`: 是否保存预测结果为.npy文件
- `--num_workers`: 数据加载线程数（默认：4）

## 配置修改

### 修改数据路径

在 `train_image.py` 中修改：
```python
parser.add_argument('--root_path', type=str, default='./data', help='root dir for data')
```

改为你的数据路径，例如：
```python
parser.add_argument('--root_path', type=str, default='./data1', help='root dir for data')
```

### 修改批大小和学习率

如果GPU显存不足，可以降低批大小：
```bash
python train_image.py --batch_size 8 --base_lr 0.0005
```

如果训练不稳定，可以调整学习率：
```bash
python train_image.py --base_lr 0.0001
```

## 原始Synapse数据集运行

如果你有原始的Synapse .npz格式数据，继续使用原来的脚本：
```bash
python train.py --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml \
  --root_path ../data/Synapse/train_npz \
  --output_dir ./model_out
```

## 文件说明

- `train_image.py`: 针对JPG/PNG格式图像的训练脚本
- `test_image.py`: 针对JPG/PNG格式图像的测试脚本
- `datasets/dataset_synapse.py`: 已添加 `ImageDataset` 类用于加载JPG/PNG数据

## 常见问题

**Q: 如何使用GPU进行训练?**
A: 脚本已默认使用GPU。如果要指定GPU，在运行前设置：
```bash
set CUDA_VISIBLE_DEVICES=0
```

**Q: 训练速度很慢怎么办?**
A: 尝试以下方案：
1. 增加 `--num_workers` 的值（如32）
2. 增加 `--batch_size`
3. 使用更强的GPU

**Q: 如何恢复中断的训练?**
A: 目前这些脚本不支持恢复功能，但可以通过修改代码实现。

## 后续优化建议

1. 添加验证集评估功能
2. 实现学习率衰减策略
3. 添加早停（Early Stopping）功能
4. 支持模型断点恢复
5. 添加更详细的日志记录
