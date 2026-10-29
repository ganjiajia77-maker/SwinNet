# V9 多尺度来源选择器

基准代码提交：`fdba9ec`（原 V9 + 只读 V9.1/V9.2 诊断）。
独立分支：`codex/v9-multiscale-source-selector`。
未带入另一个工作树中尚未提交的 G-benefit 监督实验。

## CUDA 运行兼容修正

旧启动脚本强制 `expandable_segments:True`。服务器在来源选择器的张量
分配处报告 `CUDA driver error: invalid argument`，这与驱动显存映射路径
不兼容的可能性一致，但单独 Python 堆栈不足以证实具体驱动 API。
现将 launcher 与 Python 入口均设为 `backend:native,expandable_segments:False`，
避免继续强制 VMM 分配；不改变来源网络、运输公式、损失或物理 batch2。

同时修正 smoke 显存占用：原/新局部输出只把六项比较张量保存到 CPU，
逐份释放完整 GPU 诊断；训练检查不构造无用诊断堆叠，每步后释放输出。
启动时记录 torch、编译 CUDA、GPU/驱动、allocator 与可用显存。
只在 smoke 开启 CUDA 同步和 C++ 报错栈，正式训练保持异步执行。

已补测 PyTorch2.6 CPU 的9项检查，以及本机 CUDA 的512/batch1、2与
实际入口 smoke。仍不能冒充在服务器 Quadro/驱动环境上复现通过；服务器
真实 checkpoint smoke 必须先成功，才继续正式训练。

## GPU 占用与完整流程恢复

一次服务器 OOM 显示另一进程占用约13.17GiB，当前训练进程约2.24GiB。
这是共享GPU的可用显存不足，不能靠修改运输公式解释，也不应自动结束其他任务。

launcher 现在先用 nvidia-smi 将用户指定的物理索引解析成 GPU UUID，
再将 CUDA_VISIBLE_DEVICES 设为该 UUID。日志中的逻辑 cuda:0 因此可以
明确对应物理GPU2。每个阶段开始前打印显存和同卡进程。
默认空闲不足6144MiB时停止；这是保守预留阈值，不是实测峰值保证，也不是资源锁。
进程列表查询失败时会明确记录，但不会静默改变所选GPU或终止占用进程。
没有修改训练入口文件、模型/损失或 checkpoint 代码指纹，所以已有4020100
训练权重保持严格恢复兼容。

释放占用后，继续已有实验的完整流程：

```bash
export GPU=2
export RUN_DIR=/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407
export MS_ROOT="$(cat "$RUN_DIR/latest_multiscale_root.txt")"
export EPOCHS=20
bash scripts/run_v9_multiscale.sh continue_all
```

`continue_all` 先重新检查，再分别处理local和multiscale：有last则resume，
无last且目录只有配置/日志时将整个目录备份后从原V9重新开始；有其他
checkpoint但缺last时停止，避免静默忽略权重。最后依次扫阈值、验证诊断、
测试、测试诊断。当前正在运行的同一实验不应重复启动。

## 实际结构

保留原 V9 的 visual、P0、Q3、R、W、proposal B、accept A、gate_relation、
共享 state_gate 和 rates。BASE/Q3 始终冻结、eval。原网络文件和原 V9
runner 均未修改，保留 checkpoint 中的代码 SHA256 身份核验。

`local` 直接调用原 `DirectionalDynamics.forward`，不注册 selector 参数。
`multiscale` 只替换概率运输目标和相应的可用程度；三步更新公式不变。

- 每个位置读取 8 方向 × {1,2,4,8} 共 32 个来源的当前 U。
  方向顺序沿用 `DIRECTIONS`；正 dy 向下、正 dx 向右。
- visual 用 1×1 投影至 8 通道；32 个关系共享一个 1×1 MLP。
  输入含当前位置/来源特征、差、乘积，U_p/U_q、来源支持、W、
  P0/Q3 的两端值、半径及单位方向。
- 沿每条连线的 1/4、1/2、3/4，分别在中心和横向 ±0.5 个工作像素采样。
  双线性读取 visual/P0/Q3/有效范围通道。走廊 P0 低不会硬屏蔽关系。
- 来源分数加入 `log(source_support.clamp_min(1e-6))`。source_support 的
  初始 R 与后续新增道路支持规则完全沿用 V9。
- 另有一个 stay 分数，33 路 softmax，越界来源为负无穷。
  source/stay 输出层权重为零，stay 初始 bias=2。
  全部来源越界时 M=U、V=0。低可靠度仍允许保留当前位置。
- M=stay_weight×U+Σsource_weight×U_q；V=1-stay_weight。
  M-U 与 V 送入原 rates/state_gate。alpha=r_transport×G，
  beta=0.2×A×r_proposal，gamma=r_anchor。没有再乘一个 V 开关。
  三项非负、和不超过 1，状态仍是有界凸组合。
- 没有新增 surface 头；仍由原 `compose_surface_logits` 组合最终输出。
- 来源网络的宽输入采用 activation checkpointing，反向时重算，降低
  32 个关系在 128×128 工作网格上的训练内存。

## 等预算训练

两组都从同一个原 V9 `stage_D/best.pth` 加载全部下游权重，然后各自用
新的 AdamW 优化器继续训练 20 轮（这是额外预算，不是重新训练 BASE）。
原模块学习率从 checkpoint 恢复；新增 selector 默认学习率为原 LR 组最大值，
可预先设置 SELECTOR_LR；两组其他 LR 完全相同。

损失只用原 `L_seg+0.2 L_cont`。沿用原 focal/Dice、困难道路与背景加权、
连续性标签及权重；无新增 G/q/关系损失、clDice 或 surface 辅助头。
本次没有已验证 G 监督有效的证据，因此两组都不加 L_G。

FP32、无 EMA、1500 随机裁一张 512、物理 batch2、accumulation1。
共享 epoch 的数据集修复仍在；runner 不使用 persistent workers。
每个 epoch 使用相同 seed 和 DataLoader generator，两组及 resume 的
样本顺序/数据增强一致。新增模块初始化隔离 CPU/CUDA RNG。

multiscale 固定过渡：第1轮 blend=0，第2轮1/3，第3轮2/3，第4轮及以后1。
M/V 同步从 local 过渡到 multiscale。blend 不是可学习 gate。
首轮 selector 不参与最终损失，因此其梯度为零属于预期行为。
验证始终使用 blend=1，不把过渡态用于正式模型选择。

默认每5轮验证（末轮也验证），沿用 V9 在0.60阈值下的综合 checkpoint
分数；扫阈值在验证集0.35–0.75、步长0.01上选同一综合分数。
这里保留原 V9 选择标准，没有改成只按测试 IoU/APLS 选模型。
验证/测试均 512、stride256、原加权 logits 拼回1500，无后处理。

## 服务器命令

固定 SHA 压缩包下载、解压后，在代码目录执行。脚本使用 swinunet 的
绝对 Python 路径，因此当前 shell 是 base 也可以运行。

```bash
export GPU=2
export RUN_DIR=/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407
export MS_ROOT="$RUN_DIR/multiscale_$(date +%Y%m%d_%H%M%S)"
export EPOCHS=20

# 必须先检查真实 checkpoint 和真实训练裁剪，失败即停止。
bash scripts/run_v9_multiscale.sh smoke

# 两组各继续20轮，依次执行，避免同时占用同一张卡。
bash scripts/run_v9_multiscale.sh train_pair

# 各自在验证集选阈值，然后做验证集整段缺口诊断。
bash scripts/run_v9_multiscale.sh sweep
bash scripts/run_v9_multiscale.sh audit

# 设计确定后进行测试，阈值固定为各自验证集选值。
bash scripts/run_v9_multiscale.sh test
bash scripts/run_v9_multiscale.sh audit_test
```

中断恢复使用相同 EPOCHS/TRANSITION_EPOCHS/SELECTOR_LR/SEED/WORKERS 等设置：

```bash
export GPU=2
export RUN_DIR=/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407
export MS_ROOT="$(cat "$RUN_DIR/latest_multiscale_root.txt")"
export EPOCHS=20
bash scripts/run_v9_multiscale.sh resume_local
bash scripts/run_v9_multiscale.sh resume_multiscale
```

也可单独使用 train_local/train_multiscale、sweep_local/sweep_multiscale、
test_local/test_multiscale。评估按 checkpoint 严格加载，架构、source
SHA256、损失、总预算、训练设置和代码身份不一致会报错。

诊断通过后查看：

```bash
MS_ROOT=$(cat /home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407/latest_multiscale_root.txt)
AUDIT=$(cat "$MS_ROOT/latest_audit_val.txt")
cat "$AUDIT/summary.json"
head -n 8 "$AUDIT/gap_center_sources.csv"
head -n 8 "$AUDIT/fixed_gaps.csv"
cat "$MS_ROOT/local/test_comparison.csv"
cat "$MS_ROOT/multiscale/test_comparison.csv"
```

## 输出及解释

- `local/` 与 `multiscale/`：best/last、完整 experiment.json、train.jsonl、
  val_selection.json、测试每图/汇总 CSV，以及原生1500二值 PNG。
- `gap_center_sources.csv`：每个固定缺口中心、每步的最大来源（含stay）、
  最强真实来源方向/距离/权重、来源坐标/U、far权重、潜在更新与实际运输量。
- `fixed_gaps.csv`：固定坐标哈希、长度分组、中心是否收到有效正更新、
  整段运输覆盖、最大无更新连续段、原锚点是否真正由预测局部连通块接通。
  同时报告统一原V9阈值与各自验证集选阈值下的闭合情况。
- `generated_source_reuse_and_bridges.csv`：各步使用新增加 GT背景/道路
  来源的软权重，同一来源关系在至少两步中被反复使用的比例（权重≥0.01），
  新背景桥接与其中几何近似平行的桥接数。
- `summary.json`：重点看16–31、32+的中心有效更新、完整覆盖、实际闭合。
  如只是中心变亮而闭合不增加，不据此宣称修复有效。

原 V9 的 P0/GT/骨架/图像顺序生成固定缺口；验证时核验原模型 TP/FP/FN
和 eligible gap 数与旧验证文件一致。可用 REFERENCE_SET 指向旧
V9.1 reference_set.json，进一步核对坐标数据指纹。不同数据划分不混用。

内部图使用最大融合权重窗口唯一归属，它们不是全图唯一隐状态；
来源argmax不等于因果贡献。缺口长度为原生骨架连通段像素数。
闭合采用原V9≥2 P0锚点与≥80%缺失中心线覆盖规则，是代理指标。
背景桥接/平行道路统计也为代理，不能覆盖所有误连，需结合代表图检查。

评估复用原 V9 的统一 `road_metrics`：固定 Zhang–Suen、8邻域、短块<20、
APLS≤64节点、snap≤5像素。APLS为现有近似实现；topo_f1等于clDice。
break_rate 为逐图 GT 骨架未被预测道路覆盖的比例（1−topo_recall），
不等于断口数/真实道路段数。没有改动指标实现。

此版本只实现三步；不同时增加六步、关系监督或扩大采样范围。
若覆盖增加却误连增加，下一步研究走廊关系监督，并在训练/验证集设计。

## 已执行验证

见 `validation/` 的实际记录：8项CPU功能/流程检查；512输入batch1/2的
真实下游GPU前向反向；local数值一致性；checkpoint严格加载；
原生1500、25滑窗、三组诊断CSV贯通。
本机使用合成冻结特征提供器，不能替代服务器真实 BASE/Q3 和 Mass 数据
检查。因此正式训练前服务器脚本强制要求实际checkpoint的smoke通过。
没有自动启动正式训练，也没有宣称性能收益。
