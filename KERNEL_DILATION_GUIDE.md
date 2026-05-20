# Kernel 和 Dilation 参数说明

## 📚 基础概念

### Receptive Field（感受野）计算
```
receptive_field = kernel + (kernel - 1) × (dilation - 1)
```

### 例子：
```
kernel=3, dilation=1:  RF = 3 + (3-1)×(1-1) = 3 + 0 = 3
kernel=3, dilation=2:  RF = 3 + (3-1)×(2-1) = 3 + 2 = 5 ✓
kernel=5, dilation=1:  RF = 5 + (5-1)×(1-1) = 5 + 0 = 5
kernel=5, dilation=2:  RF = 5 + (5-1)×(2-1) = 5 + 4 = 9 ✓
kernel=5, dilation=3:  RF = 5 + (5-1)×(3-1) = 5 + 8 = 13 ✓
```

---

## 🎯 DilatedAsterisk 中的参数设置

### 默认配置（三个感受野）：

```python
kernel_sizes = [
    {'kernel': 3, 'dilation': 2},  # RF = 5  (小感受野)
    {'kernel': 5, 'dilation': 2},  # RF = 9  (中感受野)
    {'kernel': 5, 'dilation': 3},  # RF = 13 (大感受野)
]
```

这三个感受野可以捕捉**不同尺度的特征**：
- RF=5：局部细节（小物体、边缘）
- RF=9：中等上下文（中等大小的结构）
- RF=13：全局语义（大的区域关系）

### 为什么不用 kernel=1？
- `kernel=1` 没有空间信息，只做通道混合
- 空洞卷积的优势在于**大感受野 + 少参数**
- 用 `kernel=3,5` 而不是 `kernel=1` 是为了保留空间建模能力

---

## 🔧 Padding 计算

```python
padding = dilation × (kernel - 1) // 2
```

**目的**：保持 H/W 尺寸不变（same padding）

### 例子：
```
kernel=3, dilation=2:  padding = 2 × (3-1) // 2 = 2 × 2 // 2 = 2
kernel=5, dilation=2:  padding = 2 × (5-1) // 2 = 2 × 4 // 2 = 4
kernel=5, dilation=3:  padding = 3 × (5-1) // 2 = 3 × 4 // 2 = 6
```

### 验证：
```
输出尺寸 = (输入尺寸 + 2×padding - dilation×(kernel-1) - 1) // stride + 1
        = (H + 2×2 - 2×(3-1) - 1) // 1 + 1
        = (H + 4 - 4 - 1) // 1 + 1
        = H  ✓ (same)
```

---

## 📊 参数对比

### 不同感受野的对比

| 感受野 | kernel | dilation | padding | 参数 | 计算 | 用途 |
|--------|--------|----------|---------|------|------|------|
| RF=5   | 3      | 2        | 2       | 3×3=9 | 多 | 细节 |
| RF=9   | 5      | 2        | 4       | 5×5=25 | 多 | 中等 |
| RF=13  | 5      | 3        | 6       | 5×5=25 | 少 | 全局 |

---

## 🛠️ 如何自定义参数

### 选项 1: 使用默认配置
```python
model = DilatedAsterisk(in_channels=768, out_channels=768)
# 自动使用 RF=[5, 9, 13]
```

### 选项 2: 自定义感受野
```python
# 例如：只用两个感受野（小 + 大）
kernel_sizes = [
    {'kernel': 3, 'dilation': 2},  # RF = 5
    {'kernel': 7, 'dilation': 2},  # RF = 13
]
model = DilatedAsterisk(in_channels=768, out_channels=768, kernel_sizes=kernel_sizes)
```

### 选项 3: 更多感受野（4个）
```python
kernel_sizes = [
    {'kernel': 3, 'dilation': 1},  # RF = 3
    {'kernel': 3, 'dilation': 2},  # RF = 5
    {'kernel': 5, 'dilation': 2},  # RF = 9
    {'kernel': 5, 'dilation': 3},  # RF = 13
]
model = DilatedAsterisk(in_channels=768, out_channels=768, kernel_sizes=kernel_sizes)
```

---

## 💡 推荐参数配置

### 为遥感道路提取的建议

根据道路的特点（细长线性结构），推荐：

```python
# 方案 A: 保守（参数少）
kernel_sizes = [
    {'kernel': 3, 'dilation': 1},  # RF = 3   (超近距离)
    {'kernel': 3, 'dilation': 2},  # RF = 5   (近距离)
    {'kernel': 5, 'dilation': 2},  # RF = 9   (中距离)
]

# 方案 B: 平衡（推荐）
kernel_sizes = [
    {'kernel': 3, 'dilation': 2},  # RF = 5   (局部)
    {'kernel': 5, 'dilation': 2},  # RF = 9   (中等)
    {'kernel': 5, 'dilation': 3},  # RF = 13  (全局)
]

# 方案 C: 激进（参数多，效果可能更好）
kernel_sizes = [
    {'kernel': 3, 'dilation': 1},  # RF = 3
    {'kernel': 3, 'dilation': 2},  # RF = 5
    {'kernel': 5, 'dilation': 2},  # RF = 9
    {'kernel': 5, 'dilation': 3},  # RF = 13
    {'kernel': 7, 'dilation': 2},  # RF = 17 (超大感受野)
]
```

### 为什么这些参数适合道路提取？
- **RF=5**: 捕捉道路边界和转折
- **RF=9**: 捕捉道路段的连通性
- **RF=13**: 捕捉整体道路方向和长线特征
- **RF=17**（可选）: 捕捉道路的全局走向

---

## ⚡ 计算成本对比

### 空间复杂度（以 [B, 768, 32, 32] 为例）

| 配置 | 分支数 | 每分支参数 | 总参数 | 相对于基础 |
|------|--------|-----------|--------|-----------|
| RF=[5,9,13] (默认) | 6 | 平均22 | ~0.5M | 1.0x |
| RF=[3,5,9,13] | 8 | 平均16 | ~0.7M | 1.4x |
| RF=[3,5,9,13,17] | 10 | 平均12 | ~0.9M | 1.8x |

---

## 🎓 原 OARENet 的设置

OARENet 中的 DecoderBlock 对每个感受野都创建一个独立的 Asterisk 模块：

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(DecoderBlock, self).__init__()
        asterisk_size = [5, 9, 13]  # 三个感受野
        self.aster1 = Asterisk(in_channels, filters//2, asterisk_size[0])  # RF=5
        self.aster2 = Asterisk(in_channels, filters//2, asterisk_size[1])  # RF=9
        self.aster3 = Asterisk(in_channels, filters//2, asterisk_size[2])  # RF=13
        ...
```

改进后，我们把三个感受野集成到一个模块中，更加轻量和高效。

---

## 📝 总结

| 项 | 说明 |
|---|---|
| **kernel** | 卷积核大小（3, 5, 7 等） |
| **dilation** | 空洞率（1=普通, 2=间隔1, 3=间隔2） |
| **padding** | 为保持尺寸而自动计算 |
| **receptive_field** | = kernel + (kernel-1)×(dilation-1) |
| **默认配置** | 三个 RF：[5, 9, 13] |
| **推荐用途** | 道路提取中很有效 |
| **可自定义** | 根据场景调整感受野大小和数量 |
