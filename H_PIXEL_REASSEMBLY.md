# 原B：C引导原生像素H重组

基线提交 bb26a17。未引入local-CS、路径监督、概率动力学或额外任务损失。
原生标签算法不变；仅迁入已核对的DataLoader目标生成位置优化。

## 实际前向

原 G3/H 投影相加→bilinear(align_corners=False)→surface_h3_fusion 顺序完整保留。
在bilinear结果后加一个H来源重组残差。这样W_out=0时甚至保留原浮点计算顺序。
查询来自最终x4上采样后的原生decoder x(96通道)，读取H(64)/G3(96)对齐网格。
q/k/v均16通道；key读取H/G3，value只读取H。位置为格内中心偏移和来源dx/dy。
每个原生像素单独预测自身+八邻居的9个分数，绝不复制128网格的选择权重。

共享打分：dot(Q(x),K(H,G3))/sqrt(16)+MLP(frac_x,frac_y,dx,dy)。
C先验 r=0.1+0.9*sqrt(sigmoid(C_d(p))*sigmoid(C_opposite(q)))；
自身r=1，全部C输入在该路detach。图外来源在softmax前设-inf，自身总有效。
输出残差为 W_out[sum_q a(u,q)*(V_q-V_p)]，W_out权重/偏置均0初始化。
C仍可由原C损失及原B已有特征路径训练；仅新增C→选择先验一条梯度被阻断。

16原生像素共享同一组候选网格，但查询/格内位置各不相同。
q/k/position/value在零W_out首批梯度为0是正常链式法则；W_out更新后激活。
不应为了首批所有参数非零梯度而破坏恒等初始化。

来源逐一gather，标量9路scores可stack，不创建B×9×16×512×512的KV展开。
默认32原生行一组，训练时对每组用非reentrant checkpoint重算，限制保留中间量。
仍会增加计算和显存，不能承诺零开销；服务器512/batch2单批检查报告峰值。

## 训练与初始化

沿用原B：Massachusetts，80完整epoch，随机512一图一裁，FP32，batch2/acc1，
无EMA，warmup10；ImageNet组5e-5→5e-6，新参数2e-4→1e-5，AdamW衰减1e-4；
原surface/S/C/D系数、数据增强、验证每5轮及checkpoint选择规则不变。
训练加载ImageNet；不加载原B或local-CS训练权重。
只读取原B的initial_state.json核对共享张量、seed和ImageNet SHA。
新模块在原B构造及ImageNet加载后安装，fork_rng隔离初始化的随机数消耗。
原B模型结构/参数均保留，新模块参数单独增加。

`H_REASSEMBLY_PRIOR=predicted`为本轮单组训练。
另支持 `uniform`：相同结构/初始化/预算，只将有效来源关系先验设1，
语义选择依然存在。要判断C独立贡献，需这类同结构训练对照；
与原B比较只说明新增整套重组的收益。不会自动启动两组。

## DataLoader目标

目标在最终裁剪/同步增强/sample transform完成后生成，使用原来的骨架张量，
不重算骨架，不改变几何顺序、随机数序列或原生C/D定义。原损失不再为训练
目标逐批同步GPU→CPU。首批刻意复算旧路径逐元素检查。
验证滑窗仍按各窗口原点生成目标，避免非4倍数窗口起点错位。
--no-native_targets_in_loader 可用于工程性能对比。

## 运行

设置 PIXEL_RUN_NAME 独立目录；B_INITIAL_MANIFEST 指向原B initial_state.json。
`bash scripts/run_native_h0_pair.sh pixel_smoke` 做512/batch2单批检查。
`pixel_train` 需要对应smoke成功，然后从头训练80轮。
`pixel_resume` 仅用于已存在本架构last.pth的断点续训。

输出 original_b_initialization_check.json、native_target_verification.jsonl、
native_smoke.json 和 h_reassembly_training.jsonl；后者含前向残差/权重熵、
保持权重、每50批q/k/v/out梯度、批次耗时和含下一批读取的时间间隔。
零初始化保持旧输出，不保证训练后旧连接不丢失。

## 测试与限制

测试覆盖边界无环绕、方向反向通道、每像素选择、C新增路径detach、
两步梯度启动、分块checkpoint与独立完整展开参考的输出/梯度一致、
完整原B输出逐位一致、共享状态及RNG一致、state_dict严格恢复；
还包含DataLoader标签/损失梯度一致性与原native B回归。
这些是正确性检查，不是精度提升证据。
固定缺口补通/丢失、错误桥接及统一六项拓扑仍需训练后评价。

## 评价与查看

scripts/evaluate_h_pixel_reassembly.py --checkpoint /absolute/best.pth
--output_dir /absolute/new_eval --command all：val扫0.05..0.95步长0.01，
按global IoU选阈值，固定该阈值推理49张test，再调用同一六项工具。
--command infer仅做预测；--command topology复用已有manifest/test掩膜，
不重复推理；--command show以文本输出完整曲线与测试表，无pandas依赖。
topology输出目录交给工具创建，不提前填充文件。
