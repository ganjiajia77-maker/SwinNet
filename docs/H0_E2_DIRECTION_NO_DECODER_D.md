# E2 对称方向聚合，解码器无 D

## 基准与改动

直接基准 `1af958056f5953f0eb6edbffcdb9f1a9db77ba63`，
原 H0 `599a410de8d341850e1810417307e4b3bbdfc82d`。
架构标识 `h0_encoder_e2_direction_no_decoder_d_v1`。

仅提供这一个训练架构。保留三阶段 E/H 双向交互、E 道路辅助监督、
H 骨架辅助监督、H 向解码器接力、解码 S/C、H3 surface 融合与全局锚点模块。

512 输入的 E2 为 192×64×64。执行顺序为：

1. 原有 skip 在该编码阶段入口采集，位置保持基准行为。
2. E2 Swin blocks。
3. 从此时的 E2 预测双角度方向 D、单通道门控 q、两个半径权重。
4. 半径 1/2，沿预测切线正反双线性采样，对称平均后减去自身。
5. 零初始化 192→192 的 1×1 投影，乘 sigmoid 门控和残差强度，加回 E2。
6. 原 E2/H_E2 同步双向交互。
7. 交互后的 E2 供道路辅助头、原 A2 和下采样读取；更新后的 H 接力到 E3/H_E3。

`D/q/ω` 均读取本阶段交互前、聚合前 E2。原 E/H 交互读取聚合后的 E2。
原 skip 采集逻辑未移动，后续阶段入口自然接收前一阶段更新的特征。

公式：

`M = Σ_r ω_r [(F(p+r*t)+F(p-r*t))/2-F(p)]`

`F' = F + sigmoid(raw_gamma) * sigmoid(q_logit) * projection(M)`

初始 gamma=0.05，q=0.5，距离权重均为0.5。投影权重及偏置为零。
正反任一侧越界时屏蔽该半径，并在有效半径上重归一化。
没有有效半径时，投影后的修正也被屏蔽，防止学习到的偏置修改边界。
双线性采样 `align_corners=False`，使用与之匹配的像素中心坐标。
方向原始向量模长小于1e-4时，用常量水平切线替代 atan2 的输入，防止零点梯度异常。
方向损失仍读取原始预测，因此可通过监督脱离退化零向量。

新增模块在原模块初始化完成后构造，并恢复随机数状态。
删除 D 本身改变参数集合；不宣称整套新模型与原模型全部随机权重逐元素相同。

## 解码 D 审计

基准 `DecoderStructureRefinement.direction_head` 从结构特征预测两通道方向，
方向仅进入辅助损失；本版本移除该头，返回的内部 D 槽为 None，
公开 stage 字典不再保存 D。收到非空 decoder D 时明确报错。
旧 placeholder 也不再生成虚假的零 D 张量。

基准实际情况：

- C 头使用结构特征和预测骨架先验，没有 D 先验；保留。
- gate 使用分割特征、S 和 C 强度，没有 D 幅值修正；保留。
- 全局节点使用 H、分割特征、C、位置/类型，没有 D；保留。
- 全局关系 bias 使用位置/距离和 C，没有 D；保留。
- Stage2 pre-topology source 在此基准配置中本来就是关闭；仍关闭。

未修改 C 的八方向次序、正负权重和连通监督。
底层通用注意力仍有旧的可选方向参数接口，但此架构没有 D 生产者，
实际路径全部传 None，不构成新的 decoder D 分支。

## 唯一新增损失

`L = L_基准去掉解码D + 0.1 * L_E2_direction`

不新增 q 损失，不新增 soft-clDice。q 是由原任务损失学习的聚合门控，
不声称它的值经过方向可靠性校准。

方向标签使用数据集提供的增强后高分辨率硬骨架，与当前道路 crop 对齐。
沿用原数据管线先在增强后的原生图上细化、再同步裁剪骨架的行为，
不重做低分辨率骨架。局部 PCA 在当前512 crop内计算。

默认配置（首轮设置，非已验证最优值）：

| 参数 | 默认 | 含义 |
|---|---:|---|
| pca_radius | 5 | 11×11高分辨率局部窗口，估计局部走向 |
| min_support | 5 | 至少5个骨架点，避免极少点估计 |
| road_radius | 3 | 骨架3像素邻域且属于GT道路，限定方向监督区域 |
| min_coherence | 0.1 | E2单元中双角度向量严重抵消时不监督 |
| lambda_encoder_direction | 0.1 | 方向辅助权重 |

利用相对坐标矩，直接计算局部协方差 `xx,yy,xy`：
双角度主方向为 normalize(`[xx-yy,2xy]`)，
可靠性为 `sqrt((xx-yy)^2+4xy^2)/(xx+yy+eps)`。
这等价于二维PCA的特征值差/和。

有效监督区域内用 `m*q_GT` 加权，先在高分辨率估计方向，
再对双角度向量做64×64区域平均、归一化。
区域内向量抵消比例还会衰减监督权重；过低时置零。
PCA和标签聚合不反向传播。

`L_D = sum(weight * (1-normalize(D_raw)·D_GT)) / max(sum(weight), eps)`。
无有效标签时为零。
训练和推理的采样、门控全部使用预测值，没有GT传播规则。

其他损失保持基准：surface Focal BCE(γ=1)+0.5 Dice；
E road辅助平均权重0.1；H骨架辅助平均权重0.004；
解码骨架0.008×0.8、0.012×0.8，总骨架预算0.020；
C损失仍按原stage权重乘3。边界及其他基准损失配置沿用。

## 训练/评估协议

Massachusetts native1500；每图每轮随机一张512；FP32；无EMA；80epoch；
物理batch2、累计1；默认GPU2。原增强、优化器、预训练加载范围不变。
pretrained lr5e-5/min5e-6；new lr2e-4/min1e-5；warmup10；val间隔5。
新增模块不属于225个ImageNet载入张量，进入new/custom参数组。

正式训练从指定ImageNet预训练开始，不支持把旧实验best当新架构resume。
新实验自己的resume严格恢复模型和优化器；不允许忽略参数不匹配。
检查点保存基准、架构、全部参数及方向标签配置。
评估继承检查点架构配置，显式冲突报错。

验证/测试512窗口、stride256、25窗拼回1500，原taper权重下限0.1，
先融合logits，再sigmoid。验证global IoU选阈值，测试固定该阈值。
扫阈值只计算像素指标，不计算验证集拓扑。

统一工具对测试集报告IoU/F1、clDice、短块面积<20、8邻域碎片数/密度，
以及最多64节点、匹配距离≤5的三种近似APLS。
`break_rate` 为每图GT骨架未被预测道路覆盖的比例，再按图平均，
不是图论断点计数。当前 `topo_f1` 与clDice相同，不作为独立指标。

训练额外日志：`e2_direction.jsonl`，逐批记录方向损失、有效数、权重和、
q均值/标准差/范围、gamma、修正幅度/比例，以及D/q/投影梯度范数。
零投影导致第一步q聚合梯度为零符合预期，投影更新后应获得梯度。

## 使用

下载固定提交后进入解压目录；下面命令在该目录执行。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-roadbias/model_out
PRETRAIN=/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth
RUN="mass_h0_e2dir_no_decoder_d_fp32_s1234_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN" > e2_direction_run_name.txt

python -u h0_encoder_experiment.py train --gpu 2 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN" --seed 1234 --num_workers 4
```

中断恢复（同目录、新终端可执行）：

```bash
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-roadbias/model_out
RUN=$(cat e2_direction_run_name.txt)
python -u h0_encoder_experiment.py resume --gpu 2 \
  --root_path "$DATA" --output_dir "$OUT" \
  --model_path "$OUT/$RUN/last.pth" --num_workers 4
```

训练完成后：

```bash
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-roadbias/model_out
RUN=$(cat e2_direction_run_name.txt)
CKPT="$OUT/$RUN/best.pth"
REPORT="$OUT/$RUN/unified"

python -u h0_encoder_experiment.py sweep --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"

python -u h0_encoder_experiment.py test --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"

python -u h0_encoder_experiment.py topology --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"

cat "$REPORT/val/val_selection.json"
cat "$REPORT/test/test_comparison.csv"
```

test会生成原生1500二值预测并执行统一拓扑；topology复用已生成预测/报告，
需要重新计算时追加 `--recompute`。
预测位于 `unified/predictions/test/threshold_XX/surface`。

可选验证（合成数据，不启动正式训练）：

```bash
CUDA_VISIBLE_DEVICES=2 OPENCV_LOG_LEVEL=ERROR \
python -u scripts/verify_h0_e2_direction.py --device cuda --batch_sizes 1 2 --native
```

实际验证结果见 `H0_E2_DIRECTION_VALIDATION.md`。
