# H0 高分辨率骨架特征流：Massachusetts

独立分支：`codex/h0-skeleton-feature-mass1500`。
直接基于 `599a410de8d341850e1810417307e4b3bbdfc82d`；架构标识为
`h0_skeleton_feature_stream_sdgate_v2`。原工作树和实验记录保留。本次没有启动正式训练。

## 改动与前向流程

输入使用原生裁剪，不把 1500×1500 整图缩小到 512。

1. RGB512 → 原浅层两次 stride2 编码器 → 原 residual adapter → `F_H0`，64×128×128。
2. 主干、bottleneck 上下文及 decoder 按 H0 运行；Stage 2 生成 `G2_base`，
   原 `SkeletonSpatialHead` 的 out 层之前产生 `B2_base`。
3. 从同一组交互前特征同步计算
   `F_H1=F_H0+alpha2*T_BH(B2_base)`、`B2_new=B2_base+beta2*T_HB(F_H0)`；
   原 out 层读出 S2；`G2_new=G2_base+R2(B2_new-B2_base)`。
   C2/D2 和原局部精化使用更新后的 G2；C2 同时读取 S2 与 D2 先验，二者仍 detach。
4. 原 G2→G3 结构传递保留。Stage 3 同样从 `G3_base` 得到 `B3_base`，
   与 F_H1 同步交互，得到 F_H2、B3_new、S3、G3_new、C3/D3。
5. 最终仅一个 64→1 的 1×1 卷积从 F_H2 读出 S_H；没有 S_H0/S_H1 监督头。
6. 在最终 ×4 上采样之前，拼接 F_dec96、G3 投影32、F_H2 投影32，
   用零初始化 160→96 的 1×1 卷积产生 delta，与 F_dec 残差相加。
   随后运行原 ×4 上采样及 surface/boundary 预测。

T_BH/T_HB 使用输入 GroupNorm 和 1×1 投影，R 使用无归一化、无 bias 的 1×1 卷积。
两阶段各自 alpha/beta 初始均为 0.05，可学习；训练日志及 checkpoint 记录其数值。
新建 F_H/B/G 多通道路径不 detach。原概率 detach、梯度缩放、C/D 标签及损失规则保留。

旧的直接 G↔F_H 交换、Stage 2/3 的 HighResStructureFusion 注入和上采样后
surface_h3_fusion 被替换。KeypointGuidedGlobalTopology 不实例化、不运行；
其 surface 预估前向已移除。原 bottleneck 上下文与 GAP→Stage 3 局部 gate 校准保留。

按用户追加要求，S+D→C 和 D→gate 与 Overlap Test 提交
`e2020d9a8df311f9f680cd3637466a674d76688e` 保持一致：

- C 每个方向的先验为中心 S、邻点 S、D 的双角轴向匹配度，先经过3→16先验嵌入，
  再与原像素对特征一起分类；S和D匹配度均detach。
- gate 保留原 S/C gate，再加入 `reliability_beta * reliability_correction([decoder_x, ||D||])`。
  D向量模长detach，reliability_beta从0初始化并学习。它在初始化时为零，后续可学习方向置信度修正。
- 不引入 D→全局关系路径；全局模块仍移除。没有 PSI、P64/P8 mask、骨架硬mask修正或附加损失。

该版本与上一提交6edae61的checkpoint不兼容，使用新的架构标识、运行名前缀和latest指针。
请启动新实验；严格恢复会拒绝上一版checkpoint。

## 损失、训练与 checkpoint

骨架监督只包括 S2、S3、最终 S_H，C/D监督按原设置保留。三个骨架输出统一使用
`build_soft_stage_skeleton_target + soft_stage_skeleton_loss`；真值优先使用
stage_skeleton_gt，否则 skeleton_gt。S_H 仅在独立高分辨率损失里计算一次。

| 项目 | 配置 |
|---|---|
| S2 有效权重 | 0.008 × 1.0 × 0.5 = 0.004 |
| S3 有效权重 | 0.012 × 1.0 × 0.5 = 0.006 |
| 最终 S_H 权重 | 0.010 |
| C/D | 原阶段权重和标签规则，connectivity factor3、direction factor0.2 |
| 数据 | Massachusetts/tiff：train、val、test，原生1500×1500 |
| 训练输入 | 每图每轮随机裁一张512，保留原增强；加载进程共享 epoch |
| 验证/测试 | 512滑窗，stride256，覆盖尾部边缘；原 H0 权重拼接 logits 后 sigmoid |
| 精度/EMA | FP32 / 关闭 EMA |
| 轮数/batch | 80 / 物理2、累积1，有效batch2 |
| ImageNet 参数学习率 | 5e-5 → 5e-6 |
| 新参数学习率 | 2e-4 → 1e-5 |
| 调度 | 原 cosine，warmup10，seed1234，每5轮验证 |

训练记录三份骨架原始/加权 loss、总 loss 和交互强度。每轮保存 last.pth；
验证 surface F1 改善时保存 best.pth，与原 H0 的 checkpoint 选择规则一致。
独立的验证集阈值扫描随后按全局 IoU 选择阈值。
训练期间的滑窗验证保留原入口的一维权重下限0.1；独立扫阈值与最终测试共同使用
原 test 入口的一维权重下限0.05。两个阶段的拼接规则在各自原始入口上保持一致，
最终验证集选阈值与测试使用完全相同的推理入口和设置。

checkpoint 保存完整模型 YAML、架构标识、特征尺寸/通道、全局关闭、融合方式、
有效损失权重、alpha/beta，以及模型、优化器、训练参数。恢复和推理严格加载全部参数；
旧 H0 checkpoint 不被当作完整恢复。显式 warm-start 迁移会列出移除、新建和不匹配参数。
本实验训练指令从 ImageNet 初始化开始，不使用旧 H0 模型做 warm-start。

## 实现自检与计算量

`scripts/check_sdgate_conditioning.py` 核对C头AST与e2020d9参考一致，并验证S与D均能改变C、
D模长能通过非零beta改变gate、S/D先验detach、beta0时gate等于原gate。
`scripts/check_skeleton_feature_stream.py` 检查物理 batch2、512 输入：

- F_H0/F_H1/F_H2 为 2×64×128×128；G2/G3/B2/B3 为 2×96×128×128。
- S2/S3/S_H 各为 2×1×128×128，两个独立阶段读出层，高分辨率读出仅一次。
- S_H 监督可回传至浅层编码器、adapter 和两次交互。
- 零初始化 delta 首次 surface 反向上游投影梯度为零；更新 delta 后 F_H2、投影和编码器梯度非零。
- 一次模型前向仅一次最终 surface head；无全局模块、无 surface 预估前向。
- 骨架权重与损失分解正确，C/D 项没有因骨架减半而变化。
- 严格 checkpoint 重载后的输出逐值一致，旧 checkpoint 被拒绝。
- persistent_workers=True、num_workers=2 时，epoch0/1/2 的裁剪坐标不同，重设 epoch0 可复现。

| 模型，batch1、RGB512 | 参数量 | 前向 FLOPs（主要卷积/矩阵运算） |
|---|---:|---:|
| 原 599a410 H0，开启全局拓扑 | 77,215,744 | 410.806 G |
| 原 H0，关闭并移除全局拓扑模块 | 77,114,449 | 327.829 G |
| 前一版骨架特征流6edae61，未含S+D/gate | 76,499,268 | 299.408 G |
| 当前骨架特征流，含S+D/gate | 76,583,400 | 302.166 G |

当前版本相对同样无全局模块的原 H0，参数减少531,049（约0.69%），上述运算量减少约7.83%；
原 H0 对照不含本次新增的S+D/gate，计算量差异包含这两项。
统计使用 PyTorch2.11 FlopCounterMode，乘加按2 FLOPs；部分归一化、逐元素和插值运算
未计入，不能据此直接换算训练速度。原 H0 统计来自该精确提交的独立只读代码副本。
详细数值在 skeleton_feature_compute_comparison.json 和 skeleton_feature_validation.json。

这些结果仅证明实现正确性，不代表道路精度改善。后续效果比较应使用同样关闭全局拓扑、
同数据和训练设置的对照。

## 服务器下载与运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet

cd /home/gjj
curl -fL --http1.1 --retry 10 --retry-delay 3 \
  --connect-timeout 30 --max-time 900 \
  'https://codeload.github.com/ganjiajia77-maker/SwinNet/tar.gz/refs/heads/codex/h0-skeleton-feature-mass1500' \
  -o h0-skeleton-feature-mass1500.tar.gz &&
mkdir -p /home/gjj/SwinNet-h0-skeleton-feature-mass1500 &&
tar -xzf h0-skeleton-feature-mass1500.tar.gz \
  -C /home/gjj/SwinNet-h0-skeleton-feature-mass1500 --strip-components=1

cd /home/gjj/SwinNet-h0-skeleton-feature-mass1500
export DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
export CUDA_VISIBLE_DEVICES=0
```

训练：

```bash
bash scripts/run_h0_skeleton_feature_mass1500.sh train
```

中断后继续：

```bash
bash scripts/run_h0_skeleton_feature_mass1500.sh resume
```

验证集扫阈值，仅统计 IoU/F1/P/R 并按 IoU 选阈值，不计算拓扑：

```bash
bash scripts/run_h0_skeleton_feature_mass1500.sh sweep
```

按验证集所选阈值在测试集滑窗推理，并用测试 mask 计算统一拓扑指标：

```bash
bash scripts/run_h0_skeleton_feature_mass1500.sh test
```

已有测试 mask，只重算拓扑：

```bash
bash scripts/run_h0_skeleton_feature_mass1500.sh topology
```

默认 DATA 如上，RESULTS=/home/gjj/SwinNet-roadbias/model_out，预训练权重为
/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth。
可通过 DATA、RESULTS、PRETRAIN 环境变量指定实际路径；每次新训练使用唯一 RUN。
后续命令默认读取 latest_h0_skeleton_feature_sdgate_mass1500_run.txt，也可设置 RUN 指定实验。

统一工具直接评估原生1500×1500二值 mask，固定 Zhang–Suen、8邻域、短块面积<20、
APLS最多64节点、匹配距离≤5；报告 IoU/F1/P/R、clDice、碎片/断裂等指标和
GT→Pred、Pred→GT、双向 APLS。APLS 沿用统一工具的栅格近似定义，并非官方矢量 APLS。
验证与测试图像必须完整配对，检查通过才输出结果。test 命令会打印最终 CSV 路径。
