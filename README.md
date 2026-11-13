# Massachusetts 道路提取 · 编码器 H

本分支是实验提交 `ccc79929efc1c6c4dd0a83b5818357ca877393d8` 的独立源码快照，对应 Massachusetts 的 `h0_encoder_postmerge_phase`。研究方法以本分支源码和下列入口为准。

| 内容 | 文件与入口 |
| --- | --- |
| 主模型前向、H 初始化和编解码器连接 | [swin_transformer_unet_skip_expand_decoder_sys.py](networks/swin_transformer_unet_skip_expand_decoder_sys.py) 中的 `forward`、`forward_features`、`forward_up_features`、`up_x4` |
| 编码器 H 双向交换 | [encoder_h_postmerge_phase.py](networks/encoder_h_postmerge_phase.py) 中的 `before_merge` 和 `after_merge` |
| 解码器 Stage 2/3 的 S/C/D 和结构精化 | [skeleton_guided_head.py](networks/skeleton_guided_head.py) |
| 稀疏全局结构交互 | [keypoint_global_topology.py](networks/keypoint_global_topology.py) |
| 道路及辅助监督 | [road_losses.py](losses/road_losses.py) 和 [训练入口](train_image.py) |
| 训练配置和评价协议 | [编码器 H](H0_ENCODER_POSTMERGE_PHASE.md)、[采样](H0_EPOCH_SAMPLING_FIX.md)、[评价](H0_ENCODER_PHASE_EVALUATION.md) |

512×512 RGB 图块生成一条 128×128、64 通道的 H 流。前三次 PatchMerging 前，当前编码特征投影并上采样后更新 H。每次合并后，以因子 2、4、8 依次对 H 的副本做像素重排，将对应粗网格的子位置搬到通道维，再投影回写到新编码尺度。同一条 H 流继续传入解码器，与 Stage 2/3 结构特征交互，并参与末端全局上下文建模及道路预测。编码器 A1/A2 路径及其监督已从发布源码移除。Stage 2/3 原有的 S/C/D 预测与监督属于此版本。

### 训练与评价

安装 [requirements.txt](requirements.txt) 中的依赖，另准备 Massachusetts 数据、SwinV2/ImageNet 预训练权重，以及原 H0 配置参考 checkpoint。参考 checkpoint 用于继承配置，训练从 ImageNet 初始化开始。设置 `DATA`、`PRETRAIN`、`H0_REFERENCE` 和 `PHASE_RUN_NAME` 后运行：

```bash
bash scripts/run_h0_encoder_phase.sh check
bash scripts/run_h0_encoder_phase.sh smoke
bash scripts/run_h0_encoder_phase.sh train
python scripts/evaluate_h0_encoder_phase.py --command infer --checkpoint /path/to/best.pth --root_path "$DATA" --output_dir /path/to/eval
```

评价入口兼容历史实验的模型权重，加载前去掉当时保存在 checkpoint 中的一个闲置 PatchMerging 标量，然后严格校验其余权重。要恢复历史实验的优化器状态，请使用原提交 `ccc7992`。计算图拓扑指标时，将 `--topo_tool` 指向 [固定版本的比较工具](https://github.com/ganjiajia77-maker/SwinNet/blob/4b7d37a/evaluation/road_comparison/compare_predictions.py) 的本地副本，并使用 `--command topology`。
