# 原 H0：编码器H在压缩前更新、压缩后按子位置读出

基于原始提交 `599a410de8d341850e1810417307e4b3bbdfc82d` 独立开发。这里的H0是原完整H0，不是 `bb26a17` native B，也不是 `0352eef` C引导像素重组实验。

## 唯一结构实验

删除A1/A2预测头、前两次PatchMerging的A调制、encoder Stage3的A1/A2偏置。第三次merge原本没有使用A，其原参数保留。原decoder、skip采集位置、H/G交互、末端全局拓扑、surface融合与S/C/D头保留。

原RGB结构编码器和adapter只调用一次，先生成 `H_initial [B,64,128,128]`。前三个encoder stage执行：

```
E_pre = stage.blocks(E)
H_next = H + sigmoid(theta_EH) * bilinear(GN(Conv1x1(E_pre)), size=H)
E_down = original_PatchMerging(E_pre)
H_phases = pixel_unshuffle(H_next, factor)
E_next = E_down + sigmoid(theta_HE) * GN(Conv1x1(H_phases))
```

`before_merge`不修改E，`after_merge`才对E做残差更新。Stage4和瓶颈没有新交互。三个阶段因子分别2、4、8，H读出投影输入/输出分别256→192、1024→384、4096→768；两方向六个可学习sigmoid系数初始0.05。常驻H不下采样，读出的只是副本。子位置重排保持明确坐标，图块内无环绕，无GT或道路概率硬筛选。

H_E3直接进入原H0 decoder，不重新调用adapter生成H。原H0末端全局模块的位置和内部公式不变；不安装native relations、C像素重组、路径监督、local-CS、方向卷积或新辅助头。

重排本身可逆；通道投影仍压缩，H语义仍需学习。这提供跨过当前merge的通路，不保证道路细节无损、不保证真实补通。

## 配置继承

`--h0_config_checkpoint`必填有效原完整H0 checkpoint。默认脚本指向：

```
/home/gjj/SwinNet-h0-mass/model_out/mass_h0_599a410_random512_fp32_80e_bs2_20261011_113054/best.pth
```

仅提取 `args`，不加载参考模型、优化器或epoch到新训练。检查动态Stage2/3 H/G交互权重存在，拒绝静态H、native B、像素重组及已改H编码器参考。

训练数据、增强、随机种子、优化器参数、分组学习率、调度、梯度累积、验证频率、阈值及全部原损失设置从参考args继承。显式保留运行路径和本实验开关。特别是 `accumulation_steps` 从参考读取，不自行假定为1或2。继承的显式配置优先于原profile默认值，防止S2/S3预算被默认0.020/0.030覆盖。

要求原协议80完整epoch、batch2、FP32、无EMA、1500原生影像普通512随机裁剪、每图每轮1块、无批数限制、无已训练道路权重warm-start。重用原数据和损失代码，未替换目标生成。

共享模块先按原H0创建、加载同一个SwinV2/ImageNet文件，再删除A并安装新模块。新模块使用独立且可恢复的CPU RNG；不重置CUDA RNG。安装前后逐张量核对共享初始化，只允许A删除键和新增模块。这个检查核对当前构造的原H0初始化，不冒充缺失的历史run初始状态原件。

## 启动与记录

`scripts/run_h0_encoder_phase.sh smoke|train|resume`。

先在单独smoke进程运行真实512/batch2前向、原surface单独梯度检查、原联合loss反向。无优化器step；确认H128始终存在、六个投影从原surface和原总损失均有有效梯度、decoder读取H_E3。通过后才允许正式训练；两进程的完整初始化指纹必须一致。

- `h0_reference_config.json`：原配置、实际配置、参考SHA及预训练SHA。
- `phase_initial_state.json`：共享初始参数指纹、A删除键和新增参数。
- `${run}_smoke/phase_smoke.json`：启动检查、六个投影梯度、显存。
- `encoder_h_phase_training.jsonl`：每轮首批及每50批的H/E更新、系数、原总损失梯度。
- `training_log.txt`、`batch_losses.csv`、`epoch_losses.csv`、`best.pth`、`last.pth`：沿用原训练输出。

正式训练从epoch1开始，smoke不消耗训练预算。`best.pth`仍按原固定验证阈值F1选择。`test_image.py`会从checkpoint args自动重建本实验架构后严格加载。

查看：

```bash
python scripts/show_h0_encoder_phase.py --run_dir /absolute/new_run
```
