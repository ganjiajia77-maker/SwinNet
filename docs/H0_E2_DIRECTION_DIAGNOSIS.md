# E2 方向聚合诊断

诊断脚本只读取 `h0_encoder_e2_direction_no_decoder_d_v1` 的 checkpoint、训练日志和 Massachusetts 图像。它不训练模型，也不修改 checkpoint。

## 运行

在本分支源码目录执行。先用现有验证集扫阈值得到 `val_selection.json`；此处的测试阈值只读取该文件。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
cd /home/gjj/SwinNet-h0-e2dir-diagnosis

DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-roadbias/model_out
RUN=mass_h0_e2dir_no_decoder_d_fp32_s1234_20261022_233816
CKPT="$OUT/$RUN/best.pth"
SELECT="$OUT/$RUN/unified/val/val_selection.json"
DIAG="$OUT/$RUN/e2_direction_diagnosis_abcd"

if [ ! -f "$SELECT" ]; then
  python -u h0_encoder_experiment.py sweep --gpu 2 \
    --root_path "$DATA" --model_path "$CKPT" \
    --output_dir "$OUT/$RUN/unified"
fi

CUDA_VISIBLE_DEVICES=2 OPENCV_LOG_LEVEL=ERROR \
python -u scripts/diagnose_h0_e2_direction.py checkpoint \
  --checkpoint "$CKPT" --root_path "$DATA" --split test \
  --selection "$SELECT" --output_dir "$DIAG" \
  --device cuda --gradient_samples 8 --feature_images 8

cat "$DIAG/summary.json"
```

只想先看训练曲线时：

```bash
python scripts/diagnose_h0_e2_direction.py logs \
  --log_path "$OUT/$RUN/e2_direction.jsonl" \
  --output "$OUT/$RUN/e2_direction_epochs.json"
```

诊断会对测试集每张原生 1500 图做 25 个 512/256 窗口，并在相同 checkpoint、相同验证阈值下分别运行四种模式：

| 模式 | E2 方向 | q 门控 | 用途 |
|---|---|---|---|
| A | 预测 D | 预测 q | 当前模型参考 |
| B | 预测 D | 预测 q，但残差置零 | 检查整个聚合残差路径 |
| C | 预测 D | 固定 q=1 | 检查空间 q 差异是否有用 |
| D | 每个窗口空间打乱预测 D | 预测 q | 与 A 对比方向是否与位置对齐；打乱种子由图像名和窗口坐标固定 |

四种预测分别保存为 `A/`、`B/`、`C/`、`D/` 目录；`summary.json`、`per_image.csv` 和四份拓扑 CSV 汇总它们。D 对 A 是重点。四次前向计算比普通测试约慢四倍。默认处理完整测试集；`--max_images 8` 只用于预览，报告中的 `complete_split` 会为 false。执行多次时需提供新的 `--output_dir`，避免覆盖结果。

特征探针在均匀抽取的最多 8 张图的中心窗口中，对聚合前后 E2 特征计算三类局部位置对余弦相似度：GT 骨架同连通段的近距离连续点；GT 道路支持的区间内、A 预测概率低且两端概率高的缺口两侧；不同 GT 骨架连通段的近邻点。还报告连续/近邻非连接对、缺口/近邻非连接对的相似度 AUC。规则是诊断采样定义，不是可学习的匹配器。

`--gradient_samples 8` 在均匀抽取的图像中心窗口做额外反向传播，比较 surface 与方向损失对聚合前 E2 的梯度。

## 怎样阅读

- `training_log.epochs`：每轮的 γ、q、修正幅度、方向损失，以及 D/q/投影层梯度范数。梯度大小本身不能判断是否与分割反馈冲突。
- `module.q_road` / `q_background`：按 **E2 单元内 GT 道路占比≥0.5** 划分后的门控均值和分布。这是事后分析；推理传播不使用 GT。道路与背景分布重叠时，不应称 q 为可靠性估计。
- `module.direction_weighted_angle_error_deg`：只在方向 GT 有效单元上按监督权重统计无向角误差。单元在重叠滑窗中会重复计数，不是独立道路实例的角误差。无有效方向标签时为 null。
- `module.delta_relative_abs_window_mean`：每窗口的 `mean(abs(F'−F))/mean(abs(F))`，随后对所有窗口平均。结合 γ、q 和投影层权重范数判断修正是否真正生效。
- `surface_gradient_audit`：冻结 checkpoint，仅在少量真实裁剪上比较 surface 损失与 `0.1×E2方向损失` 对聚合前 E2 的梯度夹角与范数。余弦小于零提示这两个任务**在所抽裁剪的该张量上**方向相反；不能推断整个训练过程冲突。该检查不会更新权重。
- `paired_same_checkpoint_fixed_val_threshold`：相同 checkpoint、相同验证阈值下 A/B/C/D 的像素和统一拓扑指标差值。D 与 A 测量方向场空间对齐对当前权重的影响；它能判断当前模型是否依赖这条路径，不能替代分别重训的架构消融。
- `feature_pair_diagnostic.metrics`：三类 GT 定义位置对在 E2 聚合前后的余弦相似度。支持对比连续关系是否更相似，以及误连对是否维持可区分；类别为空时会报告 null，不能据此下结论。
- `paired_pixel_changes` 和 `per_image.csv`：各模式增加或移除的真实道路/背景像素、A 的遗漏骨架像素在其他模式下恢复数量，以及概率变化。结合 APLS 和误连图像判断方向是否帮助困难道路。

这里的 APLS 是现有统一工具的栅格近似实现。此分支没有 P2 候选或 Gap 修复头，因此“候选召回、Gap AP、回写前后修复幅度”三行诊断矩阵不适用于这版架构。
