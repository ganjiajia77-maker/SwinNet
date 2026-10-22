# H0 encoder road/skeleton：删除 A1/A2 消融

直接修改基准：`1af958056f5953f0eb6edbffcdb9f1a9db77ba63`。
原分支：`codex/h0-encoder-road-skeleton`。
本分支：`codex/h0-1af9580-no-a1a2`。

## 模型改动

- 删除原来的两个 `RoadAttentionHead`：不生成 A1/A2，不保留其模型参数或 BN buffers。
- 删除 A1/A2 在 PatchMerging 前的 `X * (1 + road_alpha * A)` 调制及其系数。
- 删除 Encoder Stage 3 的 A1/A2 成对 attention bias 及两个可学习系数。
- 同步删除内部先验传递、收集和输出；`forward_features` 的五元组接口保留，第三项是空列表。

这次删除的 A1/A2 是原有浅层道路注意图，不是 encoder exchange 的三个道路辅助预测。
Encoder 的道路/骨架预测头、三个双向 exchange、H0 接力、原 decoder S/C/D、
H2→H3 结构传递、DCA、bottleneck、global topology 和 Surface 路径都沿用指定基准。
本次不移植其他工作树中的 E2 direction 改动，也不增加 Local C / Gap MLP。

PatchMerging 仍使用原来的 2×2 拼接、LayerNorm、linear/MLP reduction。
DCA 坐标、采样融合、C 标签、各项损失与训练参数没有额外修改。

## 初始化与标识

在原 A1/A2 构造位置仅消耗同样的初始化随机数，然后立即释放临时对象。
这些对象不注册到模型，不参与任何前向，也不会写入 checkpoint。
这样在相同种子下，保留的每个参数/缓冲区及模型构造后的 RNG 均与基准逐项相同。

主实验 architecture_id：`h0_encoder_road_skeleton_no_a1a2_v1`。
关闭 encoder exchange 的控制模式标识：`h0_599a410_no_a1a2_control`；该模式同样没有 A1/A2。
源码、训练 resolved_config、checkpoint、测试与统一评估入口使用一致标识。
历史 `base_commit=599a410...` 字段继续表示 H0 架构起源；本次直接 Git 父提交是 `1af9580...`。

## 训练与评估协议

使用基准 `h0_encoder_experiment.py` 的原参数：Mass 原生1500，每图每轮随机裁一张512，
80 epoch、FP32、物理 batch2/accumulation1、无 EMA、seed1234、warmup10。
预训练组 LR 5e-5/min5e-6，新参数组 LR 2e-4/min1e-5。
Encoder road 总权重0.10、encoder skeleton总权重0.004、decoder S factor0.8。
其余增强、损失和原预训练加载策略不变。
裁剪沿用 `1af9580` 的 persistent workers 共享 epoch 修复。

继续从相同的初始预训练开始本组消融；原版已训练 checkpoint 不用于本组断点续训。
新运行默认使用 `mass_h0_encoder_no_a1a2_` 前缀。
原 `train/resume/sweep/test/topology` 入口可用，GPU 仍通过 `--gpu` 指定。
验证/测试仍是512滑窗、stride256、原生1500拼接、无TTA；验证集只按IoU选阈值，
统一测试工具计算分割、clDice和双向近似APLS。

## 核对

`scripts/verify_h0_no_a1a2.py` 从本地Git读取精确基准，不下载或更改基准代码：

```bash
python scripts/verify_h0_no_a1a2.py --device cpu
python scripts/verify_h0_encoder.py --device cpu --batch_sizes 1 2
```

第一项核对保留参数及RNG逐项相同、只删除A1/A2相关张量，并将512输入的所有返回结果
与“只关闭A1/A2路径的精确基准”逐项比较。
第二项核对前向顺序、H0消费位置、编码辅助任务梯度、严格checkpoint恢复及完整损失反传。
完整消融效果需要正式训练结果判断。

本次本地CPU检查结果：

- 999个保留的state张量及构造后的CPU RNG与精确基准逐项相同。
- 仅减少A1/A2相关的207809个参数；新模型总参数量77096555。
- 512输入的全部返回值与仅关闭A1/A2的基准逐项相同。
- batch1/2前向、stem/adapter执行次数和H接力消费位置检查通过。
- 编码道路、编码骨架分别反传、完整损失反传及原S/C/D预算检查通过。
- 严格checkpoint保存/恢复输出相同，空骨架损失有限。
- persistent workers跨epoch裁剪位置变化检查通过。
- `train_command/architecture_flags/infer_command/compare_command` 与基准AST完全相同。

本机未进行CUDA显存检查或正式数据集训练。
