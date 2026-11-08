# 独立80轮训练的 local-CS 对照诊断

原脚本默认验证从同一 initial checkpoint 继续微调的短程实验，要求
local_cs_initial.json。独立训练不应伪造这份文件。

新增 `--comparison_kind independent`：`--reference` 指定原 native B best，
候选指定 independently trained c_guided best；严格加载相同网络。
比较双方 initial_state.json 中全部张量指纹、seed和ImageNet SHA，
再核对 checkpoint 中的训练预算/关键配置及 local_cs_mode。
当前仅有原B和c_guided，不生成或假冒surface_only结果。

`--diagnostic_thresholds .2 .46` 在同一次前向中计算两套独立统计；
不重复加载、推理或重训。每个阈值内，参考/候选使用相同阈值。
C支持条件默认固定为 reference C3>=.5，不使用候选C重新挑选关系。
四个不重复方向沿用contact_indices，滑窗按owner区域归属计数。
正负关系和有效掩码由同一份最终图块GT生成，GT不进入模型前向。

relations.csv 统计图块上的真实跨格接触响应(max of min probabilities)，
fixed_gap_changes.csv 和 fixed_gap_summary.csv 统计加权logits拼接后的
原生1500预测与固定198缺口/锚点。两者统计单位与空间范围不同。
字段 initial / initial_closed 在 independent 模式下指参考原B，
不表示当前候选训练的初始权重，也不表示生成198缺口的旧H0输出。

输出 comparison_provenance.json 记录两份权重SHA和初始化核对；
protocol.json 记录全部固定阈值和区域定义。
comparison.csv 的 own_val 为该脚本自行在val选的F1点；
fixed 两行对应旧诊断0.2及已有测试流程验证选定阈值。
已有选阈值JSON通过 --threshold_selection 绑定候选checkpoint SHA。
不重新选择测试阈值、不计算六项拓扑、不改变已完成的测试结果。

```bash
LOCAL_CS_RUN_DIR=/absolute/completed_fresh80_run \
SELECTION_JSON=/absolute/existing_eval/val_selection.json \
DIAG_OUT=/absolute/new_diagnostic \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_local_cs_fresh_diagnostic.sh

python scripts/show_local_cs_diagnostic.py --output_dir /absolute/new_diagnostic
python scripts/show_local_cs_diagnostic.py --output_dir /absolute/new_diagnostic --all_gaps
```

这可以判断local-CS相对原B的恢复/破坏，不能单独归因于C指导。
验证C指导增量仍需要同预算surface_only权重。
