# d485835的A1/A2消融：仅删除encoder道路先验链

基于d485835，保留预测C的原生像素H重组。用户确认的“融合模块”指encoder
Stage3中A1/A2注意力偏置，不是PatchEmbed CNN融合，也不是最终H/G3融合。

实际删除A1/A2两个RoadAttentionHead及layer别名，前两次PatchMerging的
road_alpha参数，encoder Stage3各block的road_bias_scale_a1/a2参数。
停止对应前向，不把参数留在优化器中。第三次merge原本未用A1/A2，不改变其结构。
Stage3 Swin、瓶颈上下文、全局关系、H/G3融合、C引导像素重组、S/C/D均保留。

为保持其余初始化一致，先按d485835完整初始化、加载ImageNet并安装相同重组模块，
然后删除所指定模块；删除不消耗RNG。核对原B初始指纹时只排除明确记录的删除键，
所有保留张量必须逐个一致。重组Q/K/V/out初始张量也与d485835相同。
checkpoint args记录no_encoder_a1a2=True，推理工厂按该配置删模块后strict加载。

训练协议、损失及DataLoader目标优化全部沿用d485835；从头80轮、batch2、FP32、
无EMA、每图一张随机512、相同学习率、seed和验证规则。无local-CS和路径损失。
不加载已训练B或d485835权重。GPU2仅是启动设备，不是实验设置差异。

新增入口 scripts/run_hpixel_no_a1a2.sh smoke|train|resume；默认GPU2。
单批检查中的零残差恒等比较对象是同一无A骨干关闭H重组时的输出，
不是要求删A后仍与完整d485835输出相同。
该实验联合删除两条A使用路径，不能分别归因下采样调制与Stage3偏置。
