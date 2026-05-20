# Asterisk 模块改造方案

## 📋 原 Asterisk 模块分析

### 当前结构（来自 OARENet）：
```
输入 [B, C, H, W]
  ↓
conv1 (1x1) → C/4
  ↓
四个空洞卷积分支（保留水平、垂直、斜向）：
  ├─ deconv1: (1, kernel) 水平方向 → C/8
  ├─ deconv2: (kernel, 1) 垂直方向 → C/8
  ├─ deconv3: (kernel, 1) + h_transform → C/8 (对角线)
  └─ deconv4: (1, kernel) + v_transform → C/8 (对角线)
  ↓
合并 [C/8*4=C/2]
  ↓
** ConvTranspose2d ** → 【需要删除，改为 stride=1 卷积】
  ↓
conv4 (1x1) → n_filters
  ↓
输出 [B, C, H, W]（但原代码有上采样，改进后保持 H/W 不变）
```

### 问题：
1. ❌ ConvTranspose2d 会改变 H/W（stride=2 上采样）
2. ❌ 输入输出尺寸不一致
3. ❌ 只能用于 decoder 中上采样的位置

---

## 🔧 改造方案（适配 Swin-UNet）

### 核心改动：

#### 1. **移除 ConvTranspose2d**
```python
# 原代码：
self.conv3 = nn.ConvTranspose2d(
    in_channels//4 + in_channels//4, 
    in_channels//4 + in_channels//4, 
    3, stride=2, padding=1, output_padding=1  # ← 上采样！
)

# 改为：
self.conv3 = nn.Conv2d(
    in_channels//4 + in_channels//4,
    in_channels//4 + in_channels//4,
    3, padding=1  # ← 保持 H/W
)
```

#### 2. **改造完整流程**
```
输入 [B, C, H, W]
  ↓
conv1 (1x1) → C/4
  ↓
四个空洞卷积分支：
  ├─ deconv1: (1, kernel) 水平 → C/8
  ├─ deconv2: (kernel, 1) 垂直 → C/8  
  ├─ deconv3: (kernel, 1) + h_transform 斜向1 → C/8
  └─ deconv4: (1, kernel) + v_transform 斜向2 → C/8
  ↓
cat → [B, C/2, H, W]
  ↓
conv3 (3x3, stride=1, padding=1) → [B, C/2, H, W] ✓ 保持 H/W
  ↓
conv4 (1x1) → [B, n_filters, H, W]
  ↓
输出 [B, n_filters, H, W] ✓ 输入输出一致
```

#### 3. **空洞卷积参数（可学习）**
- deconv1 ~ 4 使用的 kernel 和 dilation 都保持不变
- 这些不是"固定随机权重"，而是 nn.Conv2d 的参数
- nn.Conv2d 会在训练时自动学习这些卷积权重 ✓

#### 4. **方向建模保留**
- ✅ 水平方向：(1, kernel) 形状
- ✅ 垂直方向：(kernel, 1) 形状
- ✅ 斜向方向：通过 h_transform/v_transform 的重排操作
- 这些都保留不动

---

## 📝 改造后的新模块特性

| 特性 | 原 Asterisk | 新 Asterisk |
|------|-----------|-----------|
| 输入尺寸 | [B, C, H, W] | [B, C, H, W] |
| 输出尺寸 | [B, C, H/2, W/2]（上采样后改变） | [B, C, H, W] ✓ |
| 用途 | Decoder 上采样块 | Bottleneck 或任意位置 |
| 计算量 | 有上采样，较大 | 无上采样，更轻量 |
| 方向建模 | 水平、垂直、斜向 | 水平、垂直、斜向 ✓ |

---

## 🎯 使用场景（在 Swin-UNet 中）

### 选项 A：插入 Bottleneck（推荐）
```python
class SwinUnet(nn.Module):
    def __init__(self, config, ...):
        ...
        self.bottleneck = ...
        self.asterisk_module = DilatedAsterisk(
            in_channels=768,  # 或根据 Swin-UNet 的 bottleneck 通道数
            out_channels=768
        )
        ...
    
    def forward(self, x):
        ...
        x_bottleneck = self.bottleneck(x)
        x_enhanced = self.asterisk_module(x_bottleneck)  # 空间特征增强
        x_out = self.decoder(x_enhanced)
        ...
```

### 选项 B：插入 Decoder 某一层
```python
# 在 decoder 的中间层加入
decoder_out = self.decoder_block(x)
decoder_out = self.asterisk_module(decoder_out)  # 特征增强
```

---

## ✅ 优势

1. **保持空间尺寸**：可灵活插入任意位置
2. **方向感知**：保留水平、垂直、对角线建模
3. **参数可学习**：空洞卷积权重在训练中优化
4. **轻量高效**：无上采样，计算量小
5. **即插即用**：与 Swin-UNet 兼容

---

## 🚀 后续实现细节

1. 创建新文件：`swin_unet/dilated_asterisk.py`
2. 实现 `DilatedAsterisk` 类（改进后的 Asterisk）
3. 集成到 `networks/vision_transformer.py` 中
4. 可选：在 bottleneck 或 decoder 中使用

---

## 📌 确认项

你同意这个方案吗？需要我调整以下哪些内容吗？

- [ ] 删除 ConvTranspose2d，改为 stride=1 卷积
- [ ] 保留四个方向的空洞卷积和 transform 操作
- [ ] 输入输出通道数一致（可选降维）
- [ ] 单独创建新模块类
- [ ] 放在什么位置（Bottleneck / Decoder / 两者都加）
