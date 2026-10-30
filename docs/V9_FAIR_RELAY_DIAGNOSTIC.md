# V9 公平工作点与来源接力诊断

不改变或重新训练任何 checkpoint；基础模型、选择器、G、B/A、rates 和
三步公式均不修改。使用同一 multiscale 权重，打开 `freeze_generated_sources`
后仅将每步的来源支持固定为预测 R，新增概率仍保留在 U 中，其他通路继续更新。
第一步应与普通模式一致；后续差异是该 checkpoint 的推理反事实，非重训消融。

验证集上以 local 各自验证集阈值为基准，分别寻找 multiscale 的匹配 Recall
阈值、匹配前景面积阈值，10,000格的近似搜索后按原浮点概率复算，报告实际
TP/面积残差。冻结接力复用对应 multiscale 阈值。比较还有各自验证阈值；
测试必须先生成验证阈值并固定使用，测试上的召回和面积未必继续匹配。

使用已有512/256原生1500加权logits拼接、相同张量和指标协议。验证时核对两组
原验证扫阈值记录的 TP/FP/FN，严格验证source和两组best的SHA。结果中包含
IoU/F1、clDice、短碎片、碎片密度、近似双向及单向APLS，以及按照每个新阈值
重新计算的背景桥接和近似平行桥接；后两者只是几何代理。

如已有对应split的 `latest_audit_*.txt`，脚本读取其
`generated_source_reuse_and_bridges.csv`，验证 P0/GT/骨架/图像顺序指纹和
验证阈值，并输出原CSV及分组均值。旧CSV只对应原固定阈值、没有记录各组
checkpoint SHA，不能作为匹配阈值结果或证明其使用的是当前各组权重。
不存在旧指针时 summary 标记未读取，不伪造结果。

固定 P0 缺口按原诊断协议提取，以完整缺口所有像素进行统计。对于骨架长度
16–31、32+，`long_gap_components.csv` 和 `long_gap_pixels.csv` 包含每步 U、
M、可用度、G、rate、alpha/beta/gamma、运输、proposal/anchor 更新与原生最终
概率，并记录固定锚点连通、原图最长连续薄弱段。以 `unclosed_on` 过滤原模式
未补通缺口。逐点数据来自融合权重最高的单个滑窗；最终概率来自所有滑窗的
加权拼接。不同量纲不能直接相加解释。

提供 `review_*` 作为需复核的线索，不是确定性原因分类：
可用度均值<0.1；在M-U>0.02的点/步上正运输平均小于M-U的10%；在正运输
>1e-4的同一批点/步上其他负向更新平均超过正运输50%；或相同位置中≥20%
的工作网格最终U3超过阈值但拼接后的原图概率低于阈值。
stay 也影响可用度；原生输出由插值的logit差修正，因此最后一项也只能提示复核。

输出 `fair_operating_points.csv`、`fair_per_image.csv`、
`relay_gap_connectivity.csv`、`long_gap_components.csv`、`long_gap_pixels.csv`、
`previous_fixed_threshold_reuse.csv`（仅旧CSV存在时）及 `summary.json`。
旧数据不被修改，模型权重在结束前后再次校验SHA。验证诊断全部通过才发布
`fair_operating_points_val.json` 供测试使用。

实际数据在服务器；本机测试使用合成冻结特征，运行真实验证后才能得出结果。

已执行 `checks/test_v9_fair_relay.py`：验证和测试两套完整流程、旧CSV指纹、
验证阈值在测试复用、checkpoint不一致时测试推理前停止、匹配阈值残差、
固定来源反事实第一步一致和后续来源支持变化均通过。另在本机 CUDA 上执行
512输入、原/固定来源两次推理和逐点采样。PyTorch2.6 CPU本地测试因机器上
torchvision编译版本与该torch不匹配，测试进程预注册了未使用的nms算子；
服务器实际环境不使用该测试补丁。所有测试都未使用服务器权重或真实数据。
