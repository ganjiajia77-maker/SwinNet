# 原生 C/D 标签移到 DataLoader

本版本基于原 native B / local-CS 的独立工作树，仅改变标签计算位置。
原生算法 `datasets/native_relation_targets.py` 不变；不改网络、损失定义、
梯度路径、裁剪、增强、样本顺序、优化器、训练轮数或 checkpoint 选取。
没有引入路径监督。

## 数据契约

`RoadSkeletonDataset(native_relations=True)` 在最终 sample transform 之后，
直接从返回的 CPU `mask` 和 `skeleton` 调用原算法。原先的骨架生成和裁剪顺序
完整保留；不重新骨架化，不消耗随机数，也不建立整图粗网格或磁盘缓存。

DataLoader 自动拼成：connectivity[B,8,H/4,W/4] float32、
connectivity_valid 同形 bool、direction[B,2,H/4,W/4] float32、
direction_valid[B,1,H/4,W/4] bool。统计保留在 CPU。
主进程以 non_blocking=True 传递标签，原 C/D 和 local-CS 共用同一份目标。
预计算模式下缺失目标直接报错，不默默退回旧路径。

## 开关与旧接口

原生训练默认 `--native_targets_in_loader`；
`--no-native_targets_in_loader` 恢复主进程损失时计算，供相同环境的计时比较。
非 native 模式不计算这些目标。
DataLoader num_workers=0 时仍在 CPU 数据读取阶段生成，但不会有多进程重叠收益。

验证的原生滑窗仍使用原先逐窗口目标生成：窗口起点不一定是4的倍数，
不可用整图目标切片替换。没有 stage 输出的最终整图损失不生成 native 目标。
未传入预生成目标的旧损失调用继续兼容。

原生训练前 `--native_targets_verify_batches 1` 批次（默认1）显式复算旧路径并逐元素
检查，写 `native_target_verification.jsonl`；0可关闭。仅这一检查有刻意的 GPU→CPU
标签往返。`native_target_timing.jsonl` 写正常批次耗时和 worker 标签 CPU 时间。
worker 时间是多个进程的CPU工作量，不能当作训练额外等待时间。
完整epoch耗时及含数据等待的吞吐优先于仅GPU批次耗时；首批验证/进程启动需排除。

## 检查，不训练

```bash
python -u scripts/check_native_target_workers.py \
  --root_path /home/gjj/Swin-Unet-main/Massachusetts/tiff \
  --workers 4 --samples 32 --epochs 2 --output_dir /absolute/new/check_directory
```

输出 checks.json，比较相同样本、顺序、增强RGB、标签及统计的字节哈希，
并计时旧CPU串行标签处理与worker预取。它不测GPU训练加速。

## 测试

pytest tests/test_native_target_workers.py tests/test_native_relation_targets.py \
  tests/test_native_h0_relations.py tests/test_local_cs_loss.py

覆盖8种旋转/翻转与非4倍数裁剪原点、实际spawn双worker及sample最终变换，
两种loss路径总损失和预测梯度逐位相同。统计原有NaN空集字段按同位置NaN比较。

优化不影响已完成训练的评价结果，不需要为了这项工程优化重新训练80轮。
