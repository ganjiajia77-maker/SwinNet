# 代码诊断报告

## 1️⃣ Mask 二值化情况

**当前代码** (dataset_synapse.py line 113-115):
```python
label = Image.open(label_path).convert('L')  # 转为灰度图
label = np.array(label, dtype=np.float32)
label = (label > 0).astype(np.float32)
```

**诊断**:
- ✅ 使用了 `label > 0` 二值化（接近用户建议的 `label > 127`）
- ⚠️ 但没有显式使用 `label > 127`，对于 0~255 的 mask，`> 0` 也能工作
- ✅ 最终 label 范围是 0 或 1（正确）

**潜在问题**：如果 mask 有灰度边缘（例如 128 这样的中间值），`> 0` 会把所有非零值变成 1。

---

## 2️⃣ 图像格式和范围问题

**当前代码** (dataset_synapse.py line 108-110):
```python
image = Image.open(image_path).convert('L')  # 转为灰度图
image = np.array(image, dtype=np.float32)
```

**诊断**:
- ⚠️ **严重问题**：Image 被转为**灰度图** (单通道)
- ❌ 但 Swin-UNet 应该是 RGB 3 通道输入
- ❌ Image 范围是 **0~255**，没有归一化到 0~1
- ❌ 没有使用 ImageNet mean/std 标准化

**ResizeToTensor 转换** (train_image.py):
```python
image = np.array(image, dtype=np.float32)
sample = {'image': torch.from_numpy(image).unsqueeze(0), 'label': ...}
# 最后 image shape: [1, H, W] 单通道，范围 0~255
```

**输出格式**：[1, H, W] 单通道，范围 0~255
**问题**：
- ❌ Swin-UNet 预训练权重期望 RGB [3, H, W]
- ❌ 范围应该是 0~1，且用 ImageNet 标准化

---

## 3️⃣ 损失函数

**当前代码** (train_image.py line 152):
```python
criterion = torch.nn.CrossEntropyLoss()
```

**诊断**：
- 使用 CrossEntropyLoss，适合多分类
- ⚠️ 但你说 Recall=0.5583 很低（模型偏背景）
- ❌ CrossEntropyLoss 不适合极度不平衡数据（道路像素远少于背景）
- 没有样本权重或类权重

---

## 4️⃣ 数据增强

**训练集** (RandomGenerator - dataset_synapse.py):
- ✅ 随机旋转翻转（90°）
- ✅ 缩放
- ❌ **没有**亮度/对比度变化
- ❌ **没有**随机裁剪

**验证/测试集** (ResizeToTensor):
- ✅ 无增强（正确）

---

## 5️⃣ 完整处理流程对比

| 步骤 | 当前代码 | 用户建议 |
|------|---------|--------|
| 图像读取 | 灰度 L 模式 | ❌ 应该 RGB |
| 图像范围 | 0~255 (未归一化) | ❌ 应该 0~1 |
| 标准化 | ❌ 无 | ✅ ImageNet mean/std |
| Mask 二值化 | > 0 | ≈ > 127 (需调整) |
| 损失函数 | CrossEntropyLoss | ❌ 应该 BCE+Dice 或 Focal+Dice |
| 通道数 | 1 通道 | ❌ 应该 3 通道 (RGB) |
| 增强 | 旋转/翻转/缩放 | ✅ 加上亮度/对比度 |

---

## 🔴 核心问题总结

| 问题 | 严重程度 | 原因 |
|------|---------|------|
| 单通道灰度 vs RGB 3通道 | 🔴 严重 | Swin-UNet 预训练权重是 RGB |
| 图像未归一化 (0~255) | 🔴 严重 | 预训练权重期望 0~1 + ImageNet norm |
| 无 ImageNet 标准化 | 🔴 严重 | 会直接破坏预训练特征 |
| Recall 太低 | 🟡 中等 | CrossEntropyLoss 不适合不平衡类 |
| Mask 二值化边界 | 🟢 轻微 | > 0 可以用，但 > 127 更安全 |

---

## ✅ 需要修改的地方

### 待用户确认的方案：

**方案 A：快速修复（最小改动）**
1. 改 image 为 RGB 格式
2. 图像归一化到 0~1
3. 加 ImageNet 标准化
4. 改损失函数为 Dice Loss
5. Mask > 127

**方案 B：完整升级**
- 方案 A 的所有内容 +
- 加强数据增强（亮度/对比度）
- 使用 Focal Loss + Dice Loss
- 优化器学习率调整

---

## 🤔 用户需要确认

1. **是否要改成 RGB？** (目前是灰度)
2. **用 Dice Loss 还是 Focal+Dice？**
3. **图像标准化用 ImageNet mean/std？**
4. **是否加亮度/对比度增强？**
