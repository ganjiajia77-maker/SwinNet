# H0：Stage 2 骨架交互、Stage 3 方向扩散修改任务书

用途：将本文交给 Codex 实施。本文是设计说明，本轮仅编写说明，尚未修改模型代码、启动训练或验证模块收益。

## 1. 基准与目标

- 从原始 H0 提交 `599a410de8d341850e1810417307e4b3bbdfc82d` 建立独立实现；分支名称使用 `codex/` 前缀。
- 不以当前有未提交修改的主目录作为模型基准，也不以 `e2020d9` 或后来的 `h0_skeleton_feature_sdgate` / S0 版本作为基准。
- 新架构建议标识：`h0_s2_skeleton_s3_directional_diffusion_v1`。原 H0 路径必须仍可单独使用。
- 目标：Stage 2 只承担骨架特征学习与双向交互；Stage 3 保留 S3/D3，用语义与道路方向控制局部特征扩散，替换原门控残差回写；删除 C2/D2/C3，关闭末端节点全局拓扑模块。
- 这是待验证的局部结构传播设计，不宣称已经恢复正确道路拓扑或保证道路连通。

## 2. 先正确理解原始 H0

在 512×512 输入、原 embed_dim=96 的配置下：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| H0/H1/H2 | B×64×128×128 | 初始及两次更新的相对高分辨率结构特征 |
| G2/G3 | B×96×128×128 | Stage 2/3 结构特征 |
| X2/X3 | B×96×128×128 | 对应结构模块入口处的解码语义特征 |
| S2/S3 | B×1×128×128 | 从 G2/G3 读取的骨架 logits |
| D3 | B×2×128×128 | 双角度道路方向输出 |

原始 `BidirectionalSkeletonFeatureFusion` 实际交换的是 H 与 G 特征，使用两个更新前的分支特征同步计算两份输出。它不是后来 S0 版本的 B2/B3 骨架头中间特征交换。本次保留原 H↔G 交互，不迁入 S0 的新骨架特征流。

H0/H1/H2 均保持多通道特征，不能用单通道骨架概率取代。128×128 也不等于输入原生分辨率。

## 3. 总体前向顺序

```text
原 RGB 浅层编码器 → adapter → H0

Stage 2 正常 decoder 输出 X2
  → 原 structure_branch → G2_base
  → 原双向交互 (H0, G2_base) → (H1, G2)
  → 原 skeleton_head(G2) → S2，保留监督
  → 语义返回 X2 本身
  → G2 继续传给 Stage 3；H1 继续作为高分辨率流

Stage 3 正常 decoder 输出 X3
  → 保留该阶段原有 H→decoder 特征注入（若基准配置启用）
  → 原 structure_branch
  → 原 previous_structure_fusion，接收真实 G2
  → 原双向交互，与 H1 交换 → (H2, G3)
  → 原骨架头、方向头 → S3、D3；不预测 C3
  → 语义与方向扩散，读取 X3、G3、H2、D3
  → 将扩散变化量残差回写 X3
  → 原上采样、原 G3→surface 融合、原最终预测头

末端 KeypointGuidedGlobalTopology 不实例化、不执行。
```

图中 Stage 3 的 X3 指该阶段原特征注入完成后、结构残差回写之前的语义特征。不要从已经被旧 gate 更新的输出再执行扩散。

## 4. Stage 2 必须实现的行为

保留原 `structure_branch`、原 `BidirectionalSkeletonFeatureFusion`、原 `SkeletonSpatialHead` 和 S2 损失。保留真实 G2→G3 的多通道传递及 H0→H1 更新；不 detach 这些特征。

新架构下：

1. 不构造/执行 C2 head、D2 head、仅服务 C2 的 connectivity_context、Stage 2 S/C gate、旧 feature_residual 及其残差系数。
2. 不执行 Stage 2 直接 H→decoder 的 `_apply_highres_structure_fusion`。这条旁路会绕过“Stage 2 仅骨架交互”的定义；H 对 Stage 2 的影响保留在 G2/H1 的双向交互中。
3. Stage 2 结构模块的语义返回值严格为入口 X2；不能清零 X2、删除正常 decoder/Swin/skip 计算，或阻断后续语义主干。
4. C2、D2 和旧 gate 输出使用明确的缺省状态（例如 `None`），相关消费者显式跳过。不能填零 logits 冒充已删除预测，因为 sigmoid(0)=0.5。
5. 保留 `stage=2`、骨架 logits 及实际 H1/G2，供损失、阶段传递和诊断使用。
6. 若基准还启用了 Stage 2 前置结构源/注意力偏置，新架构中跳过该旧 C/D 路径；推荐使用明确的 `direct` 骨架模式，并记录这个设置。不要通过恢复被删除的 C/D 头兼容旧路径。

从 S2 损失以及后续 G3/H2/扩散的 surface 损失，仍应能回传到 Stage 2 结构分支。这不是独立的只读骨架辅助头。

## 5. Stage 3 保留项与替换项

### 5.1 保留

- 原 G2→G3 融合及原 H1↔G3 双向交互，得到真实 H2/G3。
- 原 S3、D3 预测与监督。D3 仍采用 H0 原始双角度标签、坐标约定和有效区域。
- 删除原 C3 头、仅服务 C3 的 connectivity_context 及 C3 辅助监督，不构造/执行 C3，也不保留一个无前向消费者的额外 C 辅助任务。
- 新八邻域权重 a 是特征交换的导通系数，不是受道路连通标签监督的概率 C3。八个通道只表示相同的邻居索引，不能据此认定任务相同或继续附加原 C3 损失。
- 不额外迁入 e2020d9 的 S/D→C 条件化、D 模长 gate 或方向关系 bias。

### 5.2 替换

- 关闭旧 Stage 3 `structure_gate * feature_residual` 的语义回写，用下述扩散残差替代。
- 原来仅服务旧 Stage 3 gate 的 `context_to_gate`、GAP 投影等若没有剩余消费者，在新架构中不构造/执行；记录这是旧 gate 的伴随删除。
- 不在旧 gate 后再叠加扩散，不重复执行两份语义回写。
- 扩散更新语义特征 X3；初版不递归重算 S3/D3，不反复执行 H↔G 交互，不另更新 G3/H2。
- C3 返回明确的缺省状态（例如 `None`），相关消费者显式跳过；不能继续计算原连接损失或记录伪造的零 logits。

## 6. 新模块：有限步语义与方向扩散

### 6.1 输入与轻量投影

输入 X=X3、G=G3、H=H2、D=D3，空间尺寸必须一致。保持 B 维，不写死 batch size。

```text
Z0 = value_proj(X)                         # 96→32，1×1 卷积
Gsmall = guide_g_proj(G)                   # 96→16，1×1 卷积
Hsmall = guide_h_proj(H)                   # 64→16，1×1 卷积
guide = concat[Z0, Gsmall, Hsmall]         # 64 通道
ell = guide_head(guide)                   # 64→32 的 3×3 卷积
                                         # →GELU→32→8 的 1×1 卷积
```

不在八次迭代内部使用新的卷积、BN 或随机操作。导通权重由原始引导特征计算一次，整段迭代共享。

G/H/value 特征保留梯度。不用骨架阈值、道路阈值或真值创建前向传播区域。

### 6.2 八邻域与方向偏好

完全复用 `topology_direction_constants.py` 的方向顺序、反向索引和双角度基：N、NE、E、SE、S、SW、W、NW。

每个固定邻居 q=p+(dy,dx) 对应一个导通分数 ell[p,d]。八个输出是八条邻接边的权重，不是八条道路方向。

```text
dhat = normalize(D3.detach().float(), dim=channel, eps=1e-6)
b[p,d] = dot(dhat[p], connectivity_double_angle_basis[d])
```

仅 detach 进入方向偏好的 D3 输出，D3 本身继续由原方向损失训练；不要 detach X/G/H。D3 的模长不当作可靠性概率。零方向输出的偏好应为零且不能产生 NaN。

对一条有效无向邻接边：

```text
l_pair = 0.5 * (ell[p,d] + ell[q,opposite(d)])
b_pair = 0.5 * (b[p,d]   + b[q,opposite(d)])
a[p,q] = sigmoid(l_pair + beta * b_pair)
```

默认 beta=0.5，固定、不学习。对无效越界邻居强制 a=0，越界位置不参与分母。不能使用会首尾相接的裸 `torch.roll`。

这保证同一条有效边的导通权重对称、非负。它允许多方向同时导通，但不保证路口或隐藏道路连接正确。

### 6.3 八步迭代

默认 K=8，eta=0.1：

```text
Z = Z0.float()
repeat K times:
    Zavg[p] = (Z[p] + sum_q a[p,q] * Z[q]) / (1 + sum_q a[p,q])
    Znext[p] = (1-eta) * Zavg[p] + eta * Z0[p]
    Z = Znext
```

求和使用同一轮旧 Z，同步更新所有位置。分母含 self 权重 1；常量特征应保持常量，边界不能因零填充变暗。权重和、迭代和残差减法在 FP32 下完成，最后按接口转回需要的 dtype。

方向和邻接权重不在迭代中重新预测。当前是 128 网格上的局部传播，K=8 的消息依赖范围最多八次邻接跳跃；不能宣称因此获得整幅影像全局推理或保证长缺口恢复。

### 6.4 回写

```text
deltaZ = ZK - Z0
deltaX = out_proj(deltaZ)                 # 32→96，1×1 卷积
Xout = X + deltaX
```

out_proj 的 weight/bias 零初始化，不再串接另一个零初始化残差系数。这样新模块初始为恒等映射，避免两个相乘的零参数让分支无法启动。初始化时其上游暂时没有梯度是预期行为；验证上游梯度须在输出投影完成一次有效更新后进行。

不把 ZK 直接替换 X，也不在回写后再乘旧 S/C gate。

### 6.5 控制模式

同一架构提供 `identity`、`semantic`、`semantic_directional` 三种模式：

- identity：精确返回 X，作为相同 Stage 2/3 简化结构的无扩散对照。
- semantic：使用学习的 ell，但令 beta=0；它仍可能是各向异性传播，不能称“各向同性”。
- semantic_directional：采用上述 beta=0.5 的完整定义。

需要均匀传播诊断时，可额外用所有有效边 a=1 做数值检查，不把它冒充语义传播模式。第一轮不增加可学习邻居偏移、跨尺度扩散或额外 Transformer。

## 7. 删除末端全局拓扑，保留最终 H0 输出路径

- 新架构中 `KeypointGuidedGlobalTopology` 不实例化、不执行，包括候选点筛选、FPS、节点 attention、密集回写及为其额外计算的 surface prior。
- 用户要求删除的是这套末端节点模块。保留正常 encoder/decoder、Swin blocks、skip、DCA 和主干瓶颈上下文模块；不要按名称包含 global 就一并删除。
- 原最终上采样、`enable_h3_surface_fusion` 对应的 G3→surface 路径，以及 surface/boundary head 按原 H0 设置保留。边界残差仍可通过最终道路预测接受 surface 损失，但不增加独立 boundary 辅助损失。
- 注意原名称中的 h3_surface 实际读取 G3，不是高分辨率 H2；不能误接。
- 不迁入 S0 版本的 decoder/G3/FH2 三路最终融合，不额外增加 H2→surface 新旁路或 logit 修补。
- 原最终高分辨率骨架读出头及其权重沿用 H0 训练配置；用户本轮未要求新增高分辨率独立损失。不能因“骨架”命名自动把原来为 0 的损失设成正数。

## 8. 损失与训练边界

沿用选定 H0 训练运行的真实配置，记录解析后的完整 args。不要把早期 e2020d9 的 100 epoch、batch 1/acc4 命令误当作本次 H0 Massachusetts 设置。

损失只列出实际启用项，不使用“其他仍有效的辅助损失”作为未定义兜底：

```text
L = L_surface + lambda_S2 * L_S2 + lambda_S3 * L_S3 + lambda_D3 * L_D3
```

核对来源：本地 `.worktrees/h0-599a410-random512-ema80/scripts/run_h0_mass1500.sh` 配合原 H0 `train_image.py` / `SurfaceStructureLoss`。该脚本使用 `stage23_boundary_0626`，Stage 2/3 基础权重分别为 0.008/0.012、direction factor=0.2，两个 skeleton loss factor 默认均为 1。按此配置，新架构的三个有效辅助系数为 0.008、0.012、0.0024；若选定服务器 checkpoint 的真实 args 与之不同，应显式报告并按该基准对齐，不能把本地脚本当成历史 checkpoint 的验证。

本地上述配置中的未启用项如下：

| 项目 | 实际状态 |
| --- | --- |
| 高分辨率 H2 读出骨架的独立监督 | `highres_structure_skeleton_weight=0.0` |
| A1/A2 道路注意力图辅助监督 | `road_attention_weight=0.0`，但不因此删除其原前向用途 |
| 最终输出端独立 skeleton/connectivity 辅助监督 | 此 profile 的最终权重为 0.0，且对应最终头已移除 |
| 旧版额外 stage skeleton 辅助项 | `skeleton_stage_weight=0.0`；不要与 S2/S3 的 stage_structure_loss 重复计入 |
| 独立 boundary 辅助监督 | 原实现禁用，不在总损失中加入 |
| 原 C 的 BCE/对比/反向一致性子项 | C2/C3 删除后全部不计算，不迁给新传播权重 |
| 末端全局拓扑模块 | 没有需要保留的独立监督项，删除其前向路径即可 |

`L_surface` 沿用原道路预测的 BCE/focal-BCE 与 Dice 组合；S2/S3 沿用原软骨架目标及损失；D3 沿用原双角度方向损失。新扩散通过最终 surface 损失学习，不增加独立扩散损失。

- C2/D2/C3 既不预测也不计算损失；S2/S3/D3 各只计算一次。新扩散权重由最终道路损失经实际传播与回写路径训练，不附加旧 C 标签监督。
- 原损失的 stage 权重、stage_loss_scale、skeleton_loss_factor 等有乘法关系，必须报告最终有效系数，不能只打印一个表面权重。
- `None` 字段由损失、可视化、日志、评估显式支持，不能默默转成伪预测。若某项缺失，日志应写 disabled/null，而不是将其 0 损失解释为预测完美。
- 保留原骨架目标、软标签构造、方向标签与有效区域，保留原结构入口梯度比例。不要为本轮额外重分配骨架总监督预算。
- 首版沿用原训练增强，不自动加入 MAE、人工遮挡、遮挡区域附加损失、soft-clDice、路径损失或伪标签。它们是另一个受控实验因素。
- 不用 14 张验证或 49 张测试的缺口真值生成推理时的传播 mask。阈值仍在验证集选择，不能在测试集调参。
- 先完成架构、单批次训练/推理和对齐验证；不要在没有用户训练指令时自行启动完整训练。

## 9. checkpoint 与配置

- 配置同时记录原 H0 commit、新 architecture_id、Stage 2 head 开关、Stage 3 回写模式、C3=false、global_topology=false、K/eta/beta/latent_channels 和最终融合开关。
- H0 checkpoint 只作为显式的迁移初始化来源。列出共享参数、被删除参数和新参数，禁止用无说明的 `strict=False` 吞掉架构错误。
- 新架构 checkpoint 推理恢复采用严格校验，防止误用原 H0/S0 模型评估。
- 受控对照应共享实际相同的保留参数初始化。删除模块可能改变 RNG 消耗，单纯设置相同 seed 不等于共享权重相同；需要实际张量比较。
- 本设计改变多条前向路径，基于旧 checkpoint 直接删模块的推理结果不能替代重训消融。

## 10. 必须完成的验证与交付

实施 Codex 应围绕本次风险完成以下检查，而不是只检查程序能运行：

1. 512 输入的 X/G/H、S2/S3/D3 和最终输出尺寸正确，C2/D2/C3 均为明确的缺省状态；至少验证 batch 1/2。
2. Stage 2 C2/D2/旧 gate/直接 H→decoder 注入、Stage 3 C3/旧 gate 回写、末端 global_topology 确实没有被执行；正常 decoder 和 G2→G3/H流仍执行。
3. 常量输入经扩散保持不变；边界无首尾串接；有效 a[p,q]=a[q,p]；FP32 输出和梯度有限。
4. 人工水平/竖直 D 下方向偏好正确，旋转 90° 后权重对应交换，反向邻居使用相同轴线偏好；零 D 无 NaN。
5. identity 模式和 out_proj 初始为零时，输出严格符合预期恒等关系；directional 模式的 beta=0 与 semantic 模式一致。
6. S2 损失能训练 Stage 2 骨架及 H↔G 交互；D3 损失能训练方向头；输出投影完成一次更新后，surface 梯度能进入导通引导器、G3/H2 和 value_proj。
7. 总损失中 S2/S3/D3 无重复计入，C2/D2/C3 预测与损失彻底禁用；旧高分辨率损失的实际权重可追溯。
8. 新 checkpoint 保存/严格恢复后输出一致，原 H0 的独立路径仍能构造及运行。

交付内容：独立代码变更、架构/参数变化清单、配置与运行说明、上述验证证据、共享参数迁移清单。训练前先报告参数量及一次推理耗时，并注明测试输入和设备。

第一轮建议只比较 identity、semantic、semantic_directional 三组；均采用相同的新 Stage 2 简化、Stage 3 旧回写删除及末端全局模块删除。这样可以识别传播本身和显式 D 偏好的作用。与原始 H0 的直接比较是多个改动的整体比较，不能归因于扩散一个模块。

## 11. 交给实施 Codex 的简短指令

请按本任务书，从 599a410 的原始 H0 建立独立版本，完成 Stage 2 仅骨架交互、Stage 3 语义/方向扩散替代旧结构回写、删除末端节点全局拓扑的实现。严格遵循任务书中的信息流、八邻域坐标、同步更新、None 输出、损失有效系数与 checkpoint 边界。删除 C2/D2/C3 预测及监督；Stage 3 只保留 S3/D3，新八邻域权重专门控制特征扩散，不增加旧 C3 辅助任务。保留原 H0 最终融合，不迁入 S0/e2020d9 的其他机制。先交付代码与验证结果，等待用户安排正式训练。
