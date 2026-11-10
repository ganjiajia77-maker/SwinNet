# 固定 checkpoint 的三模式验证诊断

基于 `0352eef` 新建独立诊断分支。支持 `d485835`、`c77fc30`、`0352eef` 三种架构的严格权重加载。没有编辑任何网络前向、训练入口、数据集或损失文件。

## 三种干预

- `normal`：当前 checkpoint 的预测 C 先验与完整重组残差。
- `c_prior_one`：仅将 H 像素重组的 `prior_mode` 临时设为 `uniform`，即新路径的九来源关系先验为1。保留学习到的语义匹配、位置得分、所有投影与偏置。不会改 C3 数值、Stage3 关系或全局模块中原有的 C 使用。
- `residual_off`：使用原有 `disable_h_pixel_reassembly` 开关绕过整个新增残差，包含 `self.out.bias`。其他参数不变。这不是历史原 B。

每个512窗口对同一 RGB 进行三次完整前向。记录正常/无C先验的上游 x/H/G/C 完全一致；关闭残差仍检查 native x 和 C3。eval + inference_mode，禁用所有参数梯度。首窗核对插桩与原始正常输出逐位一致；每图诊断结束恢复开关，并用末窗复核正常输出逐位一致。最后对全部参数与缓冲区做前后 SHA256。

## 固定协议

14张原生1500验证图；512/256滑窗，最后窗口原点988；原三角权重融合道路logit；无TTA。通过已有 `val_selection.json` 指定阈值，核对 checkpoint SHA256；三种模式共用这个阈值，不各自扫阈值。不使用测试集。

固定198缺口沿用原清单ID、像素、锚点。局部判据为缺口覆盖至少80%，锚点在原坐标包围盒外扩16像素范围内共享一个预测连通分量。明确记录该范围，不能与使用不同局部范围的旧表混算。双道路桥接沿用 `selector_transport_reference.bridge_proxy`。正反向代理计数不保证一一对应，不直接相减当净错误桥接数。

外部工具 `/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py` 读取三组导出的验证mask，同一GT路径复用。APLS上限150、seed1234；TOPO密度0.002、步长0.0002；TLTS参数200。val manifest每组只有一个已经固定的阈值；工具不进行阈值搜索。它输出真正的六项图评价，未调用本仓库旧的近似APLS。

## 偏置与空间修正

重组特征严格拆为 `R = W_out * delta + bias`。记录 `W_out * delta` 的幅度、窗口内空间标准差、偏置各通道和总残差。非偏置项本身也可能近似恒定；不能只因为来自W就称为结构恢复。

额外只重放末端融合与道路头，输入为关闭残差时的相同特征，加上 `bias`，得到 `z_bias_only`；不再运行编码器或训练第四个模式。合成层之后若启用了原有修正，也按原函数回放。

整图logit分解：

```
z_normal - z_off
  = (z_bias_only - z_off) + (z_normal - z_bias_only)
```

这是一种给定偏置的条件分解，不是非线性道路头的唯一因果贡献分解。空间恒定的特征偏置经过非线性后可以产生空间变化的logit。不能预先将其解释成统一阈值平移。

## 区域和保存内容

- `fixed_reference_gap`：旧198缺口像素与GT道路交集，并非当前模型全部漏检。
- `normal_road`：正常预测正确的GT道路，去掉固定缺口像素。
- `road_road_negative_background`：原生GT连接标签有效、两格均含道路但关系为负的两端格子内GT背景；是邻路间隙代理，不是确认的错误桥接。

困难背景直接复用 `native_targets`。不需要D目标，传零骨架跳过方向追踪；C与歧义标签仅依赖道路掩膜，与原函数的非零骨架结果完全一致。

logit统计使用融合后的整图输出。权重和特征统计使用互不重叠的窗口归属分区，每个原生像素只计一次；不将窗口权重或H特征插值成一套虚假的整图权重。多通道H残差不做大规模磁盘缓存；仅保留mask、CSV和JSON。

主要文件：

- `TABLES.txt`：可直接cat的所有紧凑结果表。
- `comparison.csv`、`per_image.csv`：IoU/clDice、闭合、相对normal新增/丢失TP/FP和桥接代理。
- `topology/val_comparison.csv`：同阈值六项拓扑；APLS列名 `apls_bidirectional`。
- `fixed_gap_changes.csv`：原198缺口逐ID、逐模式状态。
- `logit_changes.csv`：实际道路logit各区域变化，含逐图分位数与偏置条件分解。
- `source_weights.csv`：逐区域、逐方向权重和L1变化。
- `residual_components.csv`：残差分解与取消C后的实际残差差异。
- `feature_window_stats.csv`：W加权项的窗口内空间标准差。
- `projection_bias.csv`、`protocol.json`、`checks.json`：偏置、协议、冻结和恢复检查。

默认启动最新encoder-H版本：

```bash
DIAG_VARIANT=encoder_h bash scripts/run_hpixel_three_mode_diagnostic.sh all
```

另两版使用 `DIAG_VARIANT=pixel` 或 `no_a1a2`。优先读取对应训练/评价目录指针；可显式设置 `CHECKPOINT`、`VAL_SELECTION`、`DIAG_OUT`。找不到同权重阈值时直接报错，不猜阈值。所有结果独立保存。只有明确诊断固定阈值时才用 `DIAG_THRESHOLD` 替代证据文件。

`infer`只做三模式前向及表格；`topology`用原mask继续六项计算，兼容同输入同实现的工具断点恢复；`show`用标准库显示表格，无需pandas。所有模式仅用于固定checkpoint诊断，不能替代重训消融，也不能通过测试集调参。
