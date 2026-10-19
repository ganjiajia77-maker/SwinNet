# 原H0位置对照：将Stage 2结构学习与反馈单元前移到编码器E1

## 1. 修改目标与基准

本文件是交给实施Codex的任务书。编写本文件时只进行了代码核对，没有修改模型代码或启动训练。

本次要验证：原H0的第一套结构学习与特征反馈，提前参与编码，是否优于放在解码Stage 2。

- 模型基准是原H0提交 `599a410de8d341850e1810417307e4b3bbdfc82d`。
- Massachusetts训练、原生1500加载和512滑窗协议可参照本地 `baselines/h0_mass/` 的 `mass_h0.py`、`mass_protocol.py`。若实际工作树使用了后续仅数据/评估适配的提交，应核对模型与损失源码确实等同599a410，并分别记录模型SHA和协议SHA。
- 不使用主目录未提交改动作为底座；不从H256、S0、e2020d9或 `a668711` 扩散版开始再删除模块。
- 先读取适用AGENTS.md、核对工作树和用户改动，建立独立分支，建议 `codex/h0-stage2-to-encoder-e1`。
- 建议新架构契约：`architecture_id=h0_stage2_location_v1`，配 `stage2_structure_location=decoder|encoder_e1`。旧原H0入口仍可单独构造。
- 仅实现和完成有针对性的单批次验证；没有用户的正式训练安排时，不自动开始80轮训练。

本次不保证增益，不把“改变位置”直接称为论文创新。

## 2. 用户确认的范围

1. 保留原H流：输入512时为64通道、128×128；不改成256或512。
2. 将原Stage 2的结构预测与门控反馈链移到编码器E1本阶段Swin计算结束、第一次PatchMerging之前。
3. 高分辨率流跨越编码和解码，但只有两次H↔G更新：E1处一次，解码Stage 3处一次。不增加E2/E3处H交互。
4. 编码端预测一套S/C/D，解码Stage 3预测一套S/C/D；总共仍是两套，不增加第三套。
5. A1/A2预测头、其PatchMerging调制、编码器E3的A1/A2 attention bias全部按原H0保留。
6. 原解码Stage 3、G特征传递、末端全局节点模块、G3→surface和原最终输出路径保留。
7. 不引入扩散，不改全局选点或道路概率门控，不改DCA采样/融合，不添加新损失。

## 3. 明确搬动边界：原Stage 2完整局部链

代码核对发现，原解码Stage 2在 `layer_up` 后先执行 `_apply_highres_structure_fusion(..., stage=2)`，再执行 `decoder_structure_blocks['2']`。

为避免留下重复的旧Stage 2增强，本任务把这两个串联操作作为同一条Stage 2局部链一起搬动：

```text
原高分辨率直接残差注入 highres_structure_fusion['2']
    → 原 decoder_structure_blocks['2']
       ├─ structure_branch → G
       ├─ H0↔G → H1、G_E
       ├─ skeleton_head / connectivity_context+head / direction_head
       └─ 原S/C条件gate × 原feature_residual → 原gamma残差回写
```

因此这是“Stage 2局部增强链的位置对照”，不应声称只移动了一个骨架头或单独检验gate位置。

- `highres_structure_fusion['2']` 在原输入96通道、128网格上使用，参数和残差初始化保持原样。
- 原 `decoder_structure_blocks['2']` 也保持原结构、参数、梯度比例和原残差系数，不复制一套新模块。
- 可以保留原state_dict键名并更改调用位置，避免无意义的重命名与初始化差异。调用位置不再由模块名称中的decoder字样决定。
- 解码Stage 2位置不再执行上述两个操作；正常上采样、Swin、DCA和skip仍执行。

## 4. 形状与命名（输入512，embed_dim=96）

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| z_struct、H0 | B×64×128×128 | 原RGB浅层侧流及adapter输出 |
| E1 | B×96×128×128 | 第一编码阶段Swin后的语义特征 |
| G_E | B×96×128×128 | 原Stage 2单元在编码位置产生的结构特征 |
| H1 | B×64×128×128 | 编码位置H↔G更新后的高分辨率流 |
| S_E / C_E / D_E | B×1/8/2×128×128 | 原Stage 2的三项预测，仅前向位置变化 |
| 解码X2/X3、G3 | B×96×128×128 | 原解码语义及Stage 3结构接口 |
| H2 | B×64×128×128 | 原解码Stage 3交互后的高分辨率流 |
| S3 / C3 / D3 | B×1/8/2×128×128 | 原Stage 3预测 |

H是多通道特征，不以骨架概率替代。E1与原Stage 2采用相同通道和网格，保留原S/C/D标签尺度和方向约定，不重新解释八邻域的物理间隔。

## 5. 新前向顺序

```text
RGB512
 ├─ 原PrePatchStructureEncoder → z_struct128
 │    → 原HighResSkeletonAdapter一次 → H0
 │
 └─ 原PatchEmbed → 原绝对位置/pos_drop（如启用）
      → 编码E1原Swin blocks
      → 原 highres_structure_fusion['2'](E1,H0)
      → 原 decoder_structure_blocks['2'](E1_injected,H0)
          得到 E1_updated、G_E、H1、S_E/C_E/D_E、gate
      → 原A1头读取E1_updated
      → 原第一次PatchMerging，使用原A1调制
      → 原编码E2、A2、第二次PatchMerging
      → 原编码E3，使用原A1/A2 attention bias
      → 原后续编码、瓶颈上下文与额外Swin、norm
      → 正常decoder及原skip/DCA
      → 解码Stage 2：只做原正常decoder计算，不再执行已搬走的局部链
      → 解码Stage 3：原H1直接注入、结构分支、接收G_E、H1↔G3
          得到G3、H2、S3/C3/D3、原gate和残差回写
      → 原最终上采样、原全局节点模块、原G3→surface、原输出头
```

新架构初始仍采用原gamma/alpha初始化，不为了“加强效果”擅自调大、改为常量或删除门控。

## 6. 编码器插入位置及A1/A2顺序

原 `BasicLayer.forward` 内部先执行Swin，再计算road_attention，最后执行PatchMerging。因此不能在完整layer返回后插入模块，也不能再执行一遍Swin来取得中间特征。

`encoder_e1` 模式中，第一编码阶段顺序固定为：

```text
Swin → 搬入的H直接注入 → 搬入的结构反馈单元 → A1预测 → PatchMerging
```

- A1由结构修正后的E1读取，与原定义“读取下采样前特征”一致。A1输入改变是前移反馈的下游影响，不新增A1专用改法。
- A2仍按原阶段读取其特征；原第三编码阶段依旧使用A1/A2。不要改成S_E或C_E去替代A1/A2。
- 原普通attention、相对位置bias和其他非结构计算不变。
- 保留原skip采集位置与索引：原 `x_downsample.append(x)` 在各编码层入口，不改为缓存各层末尾输出。第一次skip仍是原位置的特征；后续编码/skip自然消费前面更新后的信息。
- 使用显式参数、辅助返回值或等价的可微接口承接E1/H0/G_E/H1；不得依赖脱离计算图的诊断缓存完成训练数据传递。
- `decoder` 控制模式跳过这一编码插入，保留原A1/A2顺序及原解码Stage 2链。

## 7. H与G如何跨阶段传递

### 7.1 H0初始化

原 `PrePatchStructureEncoder` 两次stride2、64通道投影和原refine全部保留。净stride仍为4。

`encoder_e1` 模式下，在主编码器前由原始RGB创建z_struct并执行adapter一次。不要先让RGB通过主干再从瓶颈生成H0，不修改stem卷积stride，不新增骨架监督。

### 7.2 编码结束返回

显式返回或在同一前向的局部对象中传递：

```text
bottleneck_tokens
x_downsample
road_attentions
G_E、H1、原编码位置S/C/D及gate记录
```

G_E/H1是活跃训练张量，不detach、不clone后detach、不转CPU，不经过PatchMerging。它们保持原128网格，仅作为额外持续传递的特征，不继续在E2/E3更新。

### 7.3 解码器入口

原解码入口会清空 `last_stage_features`、从z_struct执行adapter，并将 `stage2_structure_feat` 设为None。新位置模式必须改为：

- 接续已生成的H1，不能再生成H0或再次执行adapter。
- 令将要交给Stage 3的 `stage2_structure_feat=G_E`，不能初始化成None或零张量。
- 诊断记录中的H0/G2/H1若保留，分别对应原始H0、G_E、更新H1；不能被入口清空覆盖。
- `structure_outputs` 接续编码位置的stage=2记录；不要在解码Stage 2再追加一次。

### 7.4 解码Stage 2与Stage 3

- 解码inx=2仍执行原DCA、skip拼接、维度恢复、`layer_up`中的Swin及上采样。仅跳过已搬走的H直接注入和结构反馈链。
- 继续保持 `remove_stage2_pre_topology_source=True`：不恢复上采样前S/C/D源和对应decoder attention bias，也不将S_E/C_E/D_E额外送进decoder attention。
- 跳过原链时直接使用正常decoder结果，不调用会制造零logits的placeholder来冒充第二套stage=2预测。
- 解码Stage 3按原代码执行：先原H1→decoder注入，再结构分支、`previous_structure_fusion(G_E)`、H1↔G3、S3/C3/D3、原gate与残差。
- Stage 3的瓶颈GAP上下文校准维持原位置和用法。编码E1单元不读取尚未产生的瓶颈上下文，不新建替代上下文头。
- 最终全局节点模块仍读取H2与最新的C3信息，原G3→surface仍读取G3。按前向顺序保存记录，不能因编码记录排序把C_E误当最新C3。

## 8. 预测输出与损失路由

编码位置的预测虽然面向用户可命名S_E/C_E/D_E，但沿用原Stage 2的损失槽位：

```text
stage=2
source_location='encoder_e1'
refinement_step=1
stage_loss_scale=1.0
```

不能把它改成stage=1后触发零权重或另一套权重；也不能同时保留stage=2和encoder_stage两份监督。Stage 3仍为stage=3，原scale不变。

参考 `baselines/h0_mass/mass_h0.py` 的有效配置：

```text
L = L_surface
  + 0.008 * (L_S_E + 3.0*L_C_E + 0.2*L_D_E)
  + 0.012 * (L_S3  + 3.0*L_C3  + 0.2*L_D3)
```

因而S/C/D外部有效系数分别为：编码槽位0.008/0.024/0.0016；Stage 3为0.012/0.036/0.0024。C内部的masked-center、focal、正例加权、对比、对称性等保持实际原损失定义，不以新BCE替代。

- 原S/C/D标签、方向轴线约定、骨架目标、有效区域、插值口径保持不变。
- 编码槽位沿用原Stage 2的structure_input梯度比例0.5及原gate拓扑梯度设置；Stage 3沿用原值。不能把D加入原H0未使用D的局部gate。
- 原高分辨率骨架独立权重0、road_attention辅助权重0保持。A1/A2辅助权重0不表示其前向路径应删除。
- 无新增编码辅助任务、额外H骨架loss、soft-clDice、路径loss、遮挡增强、扩散loss或boundary辅助loss。
- 如果实施基准的真实解析参数与以上参考不符，列出差异并对齐用户选择的原H0运行，不能混用a668711中已删除C/D的损失配方。

## 9. 原训练与评估协议

以同一原H0 Massachusetts协议比较 `decoder` 与 `encoder_e1`：原生1500、每图每epoch随机裁一张512、80epoch、物理batch2/acc1、FP32、无EMA、原增强、原预训练与学习率分组/日程、相同种子及实际相同裁剪序列。

沿用验证checkpoint选择和验证集阈值扫描规则；测试采用512/stride256的原weighted-logit拼接、无TTA、原统一指标脚本。测试集不能用于调阈值。

不要因训练位置变化静默调整batch、预训练范围或监督权重。共享张量初始化应逐项相同；相同seed不足以排除模块构造顺序带来的RNG差异。

原参数量预计不增加，因为复用并移动同一套模块。H/G跨更多阶段保持计算图，显存与时延仍需实测，不能声称成本相同。

## 10. 初始化、checkpoint与入口

- 尽量保留原模块键名及参数形状，避免先新建一套encoder结构头再保留旧decoder闲置参数。
- 新契约记录完整模型基准SHA、协议版本SHA、architecture_id、stage2_structure_location、H网格/通道、A1A2/global状态和结构单元执行位置。
- 共享模块初始化应从同种子同预训练的原H0参考模型逐项复制/验证，或采用等价的严格共享策略；报告张量一致性。
- 默认从原ImageNet初始化策略开始新训练，不未经要求用已训练H0 best.pth做warm start。
- 同名同形状的state_dict不能证明前向位置相同。旧checkpoint只可作为明确的权重迁移源，不作为新位置模式的resume；不恢复旧优化器。
- 新checkpoint恢复严格校验location和全部架构参数，防止把encoder_e1权重按decoder模式评估。不能用未说明的strict=False吞差异。
- 同步适配训练、inspect、resume、阈值扫描、test、topology及可视化。原 `BASE_MODEL_SHA256` 只证明旧源码，不能简单删校验后把新模型仍标记成未经修改的599a410；需新增实验契约和清单，保留原模型独立验证。

## 11. 必须交付的针对性验证

1. 对512输入batch1/2检查E1/G_E/H0/H1/G3/H2/S/C/D与surface形状。两处S/C/D均真实输出，损失各计算一次。
2. hook检查执行顺序：E1 Swin → H注入 → Stage2结构单元 → A1 → PatchMerging；第一PM实际消费更新后的E1。
3. `decoder_structure_blocks['2']` 和 `highres_structure_fusion['2']` 在encoder_e1模式中均只执行一次，且位于编码；解码Stage 2不再执行，正常Swin/上采样/DCA仍执行。
4. adapter全前向仅执行一次；验证Stage 3真实读取编码位置输出的G_E和H1，未被None/零张量/重新生成H0覆盖。原skip采集位置不变。
5. S_E/C_E/D_E及Stage 3监督可训练对应参数；surface梯度通过Stage2残差和G_E/H1后续消费者回传。原零初始化残差系数需要在有效更新后再检查gate相关梯度，不能据初始化时零梯度判失效。
6. 通过同输入、eval和受控临时hook，验证改变编码位置残差会影响第一PM输入及后续编码；不将输出变化本身当作精度增益证据。
7. A1/A2头、前两次PM调制、E3道路bias、Stage 3上下文校准、末端全局模块和原G3融合都实际执行。无新增扩散/H256/S0/DCA加权路径。
8. 将同一组构造权重用于 `decoder` 控制模式，与原H0独立模型的eval输出及结构输出对齐；不能将改变多个默认设置的模型称为原H0控制。
9. 新位置模式严格保存/恢复后输出一致，错误location恢复被拒绝；原H0旧入口仍可运行。
10. 报告两模式参数量、FP32 batch2峰值显存和训练步耗时、batch1推理耗时及设备、输入、预热和计时方法。无GPU时如实列未测项。

交付独立分支/提交、调用位置和参数清单、训练与评估命令、初始化核对和上述验证证据。先完成可审查的实现及单批次验证，再由用户安排正式训练。
