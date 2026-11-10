# 对既有三模式诊断补充 bias_only

基于a365246。没有训练代码、损失或网络实现改动；只增加固定checkpoint诊断。

旧三模式程序对bias-only做了末端道路头回放和区域logit统计，但没有保存其整图mask/logit。单靠旧CSV无法恢复闭合及图指标，所以需要一次14张验证图回放。原三模式的预测不重跑。

`scripts/diagnose_hpixel_bias_only.py`读取原诊断 `protocol.json`，固定原checkpoint SHA、原验证阈值、原198缺口清单SHA、原数据路径及局部闭合判据。正常模式、C先验=1、关闭残差的mask和逐图/逐缺口表复用。先核对缓存mask的TP/FP/FN和逐缺口状态与旧表一致，后续合并拓扑还核对输入二值内容SHA。

## bias_only定义

只把新增H重组的返回值临时替换为 `out.bias[None,:,None,None]` 的空间广播，经过相同融合和道路头。不计算Q/K/V及来源权重。原H/G、Stage3、全局、encoder-H等本来已训练的模块保留；C在其他路径中的使用保留。不是修改最终道路logit阈值，也不删除偏置训练。

第一窗验证该简化前向与既有 `PixelProbe.bias_only_tail` 逐位一致，并验证退出干预后正常输出恢复。全程eval/inference_mode，参数冻结，前后权重和缓冲区SHA必须一致。

512/256滑窗、原三角加权logit拼接、原生1500、无TTA。只保存新增bias_only的14张mask和小型CSV/JSON。旧源目录不修改。

## 指标与复用

- `comparison.csv`：四模式的IoU/P/R/F1、clDice、固定缺口闭合、新增/丢失连接、TP/FP变化和桥接代理。前三模式指标来自原表，bias_only用同一GT/判据计算。
- `fixed_gap_changes.csv`：四模式逐ID的状态。
- `gap_overlap.csv`：共同闭合、正常模式独有、各模式独有的数量及ID。相同闭合数不保证相同缺口。
- `topology_comparison.csv`：同口径四模式六项拓扑。
- `checks.json`：冻结、回放一致、缓存核对及二值内容SHA。
- `TABLES.txt`：直接查看的完整紧凑表。

只对bias_only计算六项拓扑，复用原三模式的图指标。APLS150控制点、TOPO间隔0.002、步长0.0002、TLTS参数200沿用原manifest。通过已有固定阈值适配器调用原图指标实现，不重新选阈值。

合并原三模式拓扑时要求协议、指标实现SHA、影像清单SHA和实际mask/GT内容SHA全部一致。旧拓扑尚未生成则明确留空；旧报告后来生成后再次 `--command show` 即可仅合并表格。不同协议会报错，不交叉拼表。

```
python scripts/diagnose_hpixel_bias_only.py \
  --source_dir /absolute/existing_three_mode_diagnostic \
  --output_dir /absolute/new_bias_only_diagnostic --command all
```

`infer`只做bias回放与统计；`topology`只算已有bias mask的图指标并合并；`show`只合并/显示表格，不做前向。原checkpoint或缺口清单搬迁时可显式提供同SHA的 `--checkpoint` 或 `--fixed_gaps`。

这回答固定checkpoint中偏置与完整重组的表现差异。偏置经过非线性道路头后可产生空间变化；不能把它自动解释成统一道路阈值平移。闭合、桥接仍是代理，判断需联合相同缺口ID、误检代价和路径指标；不替代重训练消融，不单凭“都闭合26个”证明贡献完全相同。
