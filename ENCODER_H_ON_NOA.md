# 无A1/A2＋C像素重组：增加编码H交互

基于c77fc30独立实验，不修改GPU2上原实验。默认GPU0。
延用原H0已探索过的轻量编码交互形式，不把“贯穿H”本身宣称为新方法。
本次检验它在native-B、无A1/A2、C引导原生H重组这个底座上的作用。

## 数据流

512RGB → 原PrePatchStructureEncoder（两次stride2）→ 原adapter一次
→ H_initial[64,128,128]。
每个encoder stage的所有Swin blocks之后、PatchMerging之前，各执行一次：

H_next = H + sigmoid(theta_EH) * bilinear(GN(Conv1x1(E)), size=H)
E_next = E + sigmoid(theta_HE) * GN(Conv1x1(avgpool(H, size=E)))

两路同时读取旧E/H，投影先在各自原生网格上执行；H主状态不做下采样。
E1/E2/E3分别是96×128²、192×64²、384×32²。
H始终64×128²，六个独立sigmoid系数初始0.05，属待验证起始设置。
不从预测道路/骨架概率硬筛选位置，保留整张H特征图。

三次更新后的H_E3直接传入原decoder Stage2，再继续原全局、Stage3交互，
最后参与原H/G3融合和C引导512尺度重组。decoder不再重复调用adapter生成H。
Stage4和瓶颈没有新增交互。原skip采集位置不变。

## 不变配置

无A1/A2及其Stage3 bias；预测C像素重组；DataLoader原生标签优化；
原surface及S/C/D损失和全部系数；无local-CS、无路径损失。
不新增编码道路/骨架头，不重新分配骨架损失预算。
80完整epoch，FP32，batch2/acc1，无EMA，同ImageNet、seed、裁剪/增强、LR和验证规则。
从头初始化，不加载已训练c77fc30/原B/local-CS权重。

新增模块初始化使用独立且可恢复的CPU随机数状态；原c77共享权重与随机流保持。
初始化清单核对时，新增encoder_h_exchanges和h_pixel_reassembly为明确允许项，
原A1/A2删除键为明确排除项，其余原B初始张量须逐个相同。
checkpoint args保存h_encoder_exchange=True，推理工厂可严格重建和加载。

## 首批与诊断

scripts/run_hpixel_noa_encoder_h.sh smoke|train|resume 复用原训练入口。
单批报告要求六个H/E投影有任务梯度、H_E3与decoder实际输入是同一张量，
并保留C像素重组零初始化恒等检查（参照同一encoder-H骨干旁路重组的输出）。
H_E3经非零编码交互后可与c77输出不同，不能要求整个新增架构初始化仍完全恒等。
GPU0的512/batch2峰值显存与吞吐需在服务器单批实测。

日志 h_reassembly_training.jsonl 增加每级E/H增量、残差系数和每50批梯度；
可用原show_h_pixel_reassembly.py查看，训练后原扫阈值/测试/六项工具继续可用。
H0诊断字段保持真正的H_initial，另记录H_E1/E2/E3及H_decoder_input。

## 证据边界

128网格持续存在提供保留机会，并非无损存储。H→E仍使用池化副本，
E→H插值补充的是语义，不会凭空恢复细道路。不能预先宣称减少了断路。
须与同预算c77实验比较固定缺口新增/丢失、误连及六项拓扑，结合耗时判断。
