# H0 B 代码与诊断：阅读入口

本分支基于 `codex/h0-native-relations-599a410`。B 模型训练代码来自提交 `bb26a1725ae80f8d35ef575fb6999870d490bac0`；其后至 `a6aac211eb9f82020860a6b830b40234c680cbb5` 添加了配对诊断脚本。此分享分支另外加入这份入口和 `DIAGNOSTICS_NATIVE_H0_B_ZH.md`，不修改模型。

建议阅读：

1. [中文诊断摘要](DIAGNOSTICS_NATIVE_H0_B_ZH.md)：整体指标、S/C/D、选择器和固定 checkpoint Oracle。
2. [B 版实验说明](NATIVE_H0_EXPERIMENT.md) 和 [配对训练参数](scripts/run_native_h0_pair.sh)。
3. [B 版局部关系及全局上下文实现](networks/native_h0_relations.py)。
4. [原生连接和方向目标](datasets/native_relation_targets.py) 和 [结构损失](losses/road_losses.py)。
5. [主网络与 H 流](networks/swin_transformer_unet_skip_expand_decoder_sys.py) 以及 [模型入口](networks/vision_transformer.py)。

本仓库不包含本次训练 checkpoint、数据集或服务器端的诊断原始结果。中文摘要为用户提供的数据整理，须保留其中注明的评价口径和证据边界。
