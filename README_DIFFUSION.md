# H0 Stage 2 skeleton / Stage 3 directional diffusion

架构：`h0_s2_skeleton_s3_directional_diffusion_v1`。模型底座是原始 H0
`599a410de8d341850e1810417307e4b3bbdfc82d`。包的 Git 父提交 `fe3b730` 仅提供
此前发布的 Massachusetts 数据加载、512滑窗和统一评估适配；其模型源码与599a410一致。
未使用主目录未提交改动、e2020d9、S0 或 S+D/soft-clDice 分支。
完整设计见 [任务书](docs/H0_STAGE2_SKELETON_STAGE3_DIFFUSION_SPEC.md)。

## 实现与参数清单

- Stage 2：原 structure_branch、原 H0↔G2 双向特征交互、S2。语义输出严格返回 X2。
  不构造 C2、D2、connectivity_context、旧 gate、feature_residual、残差系数和直接H→decoder模块。
- Stage 3：原 G2→G3、H1↔G3、S3/D3；保留原H→decoder注入。删除C3、旧gate及其GAP上下文，
  用八步语义/方向扩散回写X3。方向只为偏好提供 detached 双角向量，不使用模长可靠性。
- 扩散：32通道value、G/H各投影16通道；一次预测八邻域导通权重；对称边权、越界无效、
  同步更新、K=8/eta=0.1/beta=0.5、FP32迭代；out_proj零初始化，不再乘零系数。
- 训练模型不实例化末端锚点全局模块，也不计算额外surface prior。主干瓶颈上下文和Swin保留。
- 最终上采样、G3→surface、原surface/boundary残差头保留；无S0三路融合和新H2→surface旁路。
- 损失严格为 surface focal BCE(gamma1)+0.5 Dice +0.008 S2 +0.012 S3 +0.0024 D3。
  S2/S3软骨架、D3双角方向目标及边界有效区域复用原H0代码；C2/D2/C3均为None，不创建C标签。
  高分辨率骨架独立权重0、道路注意力辅助权重0、旧stage skeleton额外权重0、boundary辅助0；
  无soft-clDice、MAE、路径损失或额外扩散监督。CSV将删除项写为disabled。

原 H0 仍由 `architecture_id=h0_original` 独立构造，可继续运行旧 `mass_h0.py` 入口。
新入口为 `diffusion_experiment.py`，推理严格按新checkpoint架构契约和全部state_dict恢复。

## 初始化与证据

默认不加载训练好的H0权重。初始化时临时构造同种子的原H0参考模型，只加载相同ImageNet
SwinV2 Tiny window8/256 encoder预训练，再复制所有共享张量。参考模型用于初始化，不参与训练前向。
逐个torch.equal核对共享张量；不会用strict=False吞掉结构差异。每次运行保存
`initialization_manifest.json`，列出共享、删除、新增张量和共享初始化SHA256。
需要显式迁移训练好的原H0权重时才提供 `--h0_init_ckpt`，它不恢复旧优化器。

本机已验证：854个共享张量一致；删除156个原张量；新增13个扩散张量。
原H0参数77,215,744；新架构75,982,320。三个控制模式的共享权重及新增权重初始化均一致。
512输入、batch1/2尺寸正确；完成一个真实batch2 FP32训练步；严格保存/恢复输出完全一致；
原H0独立路径也完成512前向。常量、边界、边权对称、方向旋转、零方向、恒等、beta0对照、
投影更新后上游梯度、S2交互梯度、D3梯度及损失系数与原损失的数值对齐均通过。

一次实测：RTX 5070 Laptop GPU，PyTorch2.11.0+cu128，FP32，输入[1,3,512,512]，
预热1次后计时1次模型前向约78.11ms。仅为本机单窗口耗时，不代表服务器或整幅1500推理时间。
详细原始证据及迁移张量清单：[VERIFICATION_EVIDENCE.json](VERIFICATION_EVIDENCE.json)。

可在服务器复核：

```bash
CUDA_VISIBLE_DEVICES=2 \
H0_PRETRAIN_CKPT=/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth \
python tests/test_diffusion_spec.py
```

没有启动80轮正式训练。测试中的一个batch训练只用于验证，不产生可报告的模型精度。

## Massachusetts 运行

对齐本会话已发布fe3b730运行配置：1500随机裁一张512、80epoch、物理batch2、acc1、
FP32、关闭EMA、原增强、共享epoch修复、每5轮验证、趋势验证关闭。原学习率为
pretrained5e-5→5e-6、new2e-4→1e-5、warmup10；AdamW weight_decay0.0001、clip1。
best checkpoint按验证阈值0.5的全局F1选择；随后各模型独立扫阈值。这与本会话发布命令一致。
未读取服务器历史checkpoint，不声称本地任务书引用脚本就是其真实args；该脚本的阈值0.2
不是此包的0.5。全部解析args保存到resolved_config.json及checkpoint。

三组共享同一简化架构：identity精确返回X3；semantic令beta0但边权仍可各向异性；
semantic_directional采用beta0.5。先比较这三组，勿将与原H0的整体差值单独归因于扩散。

从包目录执行，先选择模式/种子：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-h0-s2s3-diffusion/model_out
PRETRAIN=/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth
MODE=semantic_directional
SEED=1234
RUN="mass_s2skel_s3diff_${MODE}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN" > "diffusion_run_${MODE}_s${SEED}.txt"

python -u diffusion_experiment.py inspect --gpu 2 \
  --output_dir "$OUT/${RUN}_inspection" --pretrain_ckpt "$PRETRAIN" \
  --diffusion_mode "$MODE" --seed "$SEED"

python -u diffusion_experiment.py train --gpu 2 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN" --diffusion_mode "$MODE" --seed "$SEED" --num_workers 4
```

重新打开终端后设置DATA/OUT/MODE/SEED，读取同一运行：

```bash
RUN=$(cat "diffusion_run_${MODE}_s${SEED}.txt")
CKPT="$OUT/$RUN/best.pth"
REPORT="$OUT/$RUN/unified"

python -u diffusion_experiment.py sweep --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"
python -u diffusion_experiment.py test --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"
python -u diffusion_experiment.py topology --gpu cpu \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"
```

扫阈值导出val各阈值原生1500 mask，仅按全局IoU选阈值；test固定该阈值，512/stride256
滑窗拼回1500；统一工具报告IoU/F1、clDice、断裂率、短碎片数、碎片密度及三个方向近似APLS。
Zhang–Suen、8邻域、短块面积<20、APLS64节点/匹配<=5保持一致；不在test调参。
断裂率是GT骨架未覆盖比例，APLS是既有统一工具栅格图近似值。
报告位于 `$REPORT/test/test_comparison.csv` 及同目录逐图CSV。

更换MODE为identity/semantic重复；种子可选1234/2026/3407并做相同种子的配对比较。
恢复训练使用同一MODE/SEED和 `--resume "$OUT/$RUN/last.pth"`；保持80轮原学习率进度。

已训练的 `semantic_directional` checkpoint 可先用新增的
`diagnose_direction.py` 做同权重开/关方向推理诊断，查看边权、回写和最终预测变化，
并通过统一工具比较固定阈值与各自验证阈值的测试指标。完整指令及解释见
[DIRECTION_DIAGNOSIS.md](DIRECTION_DIAGNOSIS.md)。该诊断不重训、不覆盖 checkpoint。
