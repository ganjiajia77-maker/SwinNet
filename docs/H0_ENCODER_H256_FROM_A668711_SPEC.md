# 基于 a668711：256 高分辨率编码交互与解码方向扩散修改任务书

## 0. 用户确认、基准与核对限制

用户要交给另一个 Codex 实施，本次仅编写任务书，没有修改模型代码或启动训练。

- 指定基准分支：`codex/h0-s2-skeleton-s3-diffusion`。
- 指定基准提交：`a6687113860b0bf490c5d578889b47093906ff18`。
- 用户确认：512 输入时，H 主流改为 **256×256，保持原来的 64 通道**，不是上一条建议的 32 通道。
- 同时增加三次编码器下采样前的 H↔E 交互，保留已有解码 Stage 2 骨架学习、Stage 3 方向扩散；删除 A1/A2 及其前向用途。
- 本任务书作者的本地 Git 没有该提交对象，也没有该分支工作树。因此没有核验 a668711 的确切 diff。参考材料是本地 `baselines/h0_mass_s2s3_diffusion/`，其 README、模型、扩散、损失和初始化协议已阅读，但**不声称该目录与 a668711 完全一致**。
- 实施 Codex 必须首先核验真实分支、提交与工作区状态，并读取适用 AGENTS.md。以指定提交建立独立的新版本（建议分支 `codex/h0-encoder-h256-s3-diffusion`）；保留用户已有未提交工作。不能静默换成 599a410、主目录 HEAD 或本地参考包。若自己的环境也没有指定提交，应说明缺失并取得正确源码，不能凭本任务书重建一个假定底座。
- 建议新 architecture_id：`h0_encoder_h256_s2_skeleton_s3_diffusion_v1`。旧架构和旧 checkpoint 仍按其原接口构造。

## 1. 本轮架构定义

H 是多通道高分辨率特征流，接受下游骨架任务及道路任务的梯度；不是单通道骨架概率，也不默认已经具有准确骨架含义。编码器不新增逐层骨架输出或辅助损失。

```text
RGB 512
 ├─ 原主干 PatchEmbed → E1(128)
 │    E1 Swin → exchange(E1,H0) → PatchMerging → E2(64)
 │    E2 Swin → exchange(E2,H_E1) → PatchMerging → E3(32)
 │    E3 Swin → exchange(E3,H_E2) → PatchMerging → E4(16)
 │    E4 / 原瓶颈上下文 → 正常 decoder 与原有 skip
 │
 └─ 高分辨率浅层编码器，净 stride=2 → adapter 一次 → H0(64,256,256)
          → H_E1 → H_E2 → H_E3 = H_enc
                                  │
Decoder Stage 2：X2 → G2_base ↔ H_enc → G2、H_D2；G2 → S2
                语义输出仍为 X2；G2 和 H_D2 继续向后传递
Decoder Stage 3：X3 → 保留基准的 H→decoder 注入 → G3_base，接收 G2
                G3_base ↔ H_D2 → G3、H_D3；G3 → S3、D3
                H_D3 的副本缩放到128，仅供原扩散读取
                原扩散更新 X3(128)，继续正常最终上采样
最终输出：保留原 G3→surface 融合；增加一次 H_D3→最终 surface 特征的轻量残差
          → 原道路/边界输出头 → 512 logits
```

新方案默认启用最终 H 残差，这是前一轮讨论“最终预测读取256细节”的具体落点，也是相对基准的一个独立新增路径。应记录开关，不能将总体结果单独归因于编码交互或分辨率。

## 2. 张量约定（512 输入、embed_dim=96）

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| H0 / H_E1 / H_E2 / H_enc | B×64×256×256 | 初始化及三次编码交互输出 |
| E1，第一次 PatchMerging 前 | B×96×128×128 | 本阶段 Swin 完成后的编码特征 |
| E2，第二次 PatchMerging 前 | B×192×64×64 | 同上 |
| E3，第三次 PatchMerging 前 | B×384×32×32 | 同上 |
| E4 / 瓶颈 | B×768×16×16 | 沿用基准，不增加第四次编码交互 |
| H_D2 / H_D3 | B×64×256×256 | 两次解码交互输出 |
| X2 / X3 / G2 / G3 | B×96×128×128 | 沿用基准实际结构接口 |
| S2 / S3 | B×1×128×128 | 原骨架 logits |
| D3 | B×2×128×128 | 原双角方向输出 |
| H_diff | B×64×128×128 | 仅是 H_D3 供扩散消费的副本 |

H 的64通道从头到尾不变。融合内部的16通道投影、扩散32通道 latent 不代表把 H 主流改成16/32通道。

## 3. H0 的来源：净 stride=2，不能先得到128再插值到256

本地参考的 `PrePatchStructureEncoder` 原为：

```text
Conv3×3 3→16 stride2 → Conv3×3 16→32 stride2 → project32→64 → refine
```

新版本建议保留原卷积通道、投影、归一化和 refine 定义，只把第二个空间下采样卷积的 stride 从2改为1：

```text
Conv3×3 3→16 stride2 → Conv3×3 16→32 stride1 → project32→64 → refine
```

得到真实256网格，再执行原 `HighResSkeletonAdapter` 一次，产生 H0。这个选择保持原 stem 的权重形状，但其采样语义已改变，初始化清单应记录。若 a668711 的实际 stem 与上述不同，先列出差异并实现同样的净 stride=2、输出64通道目标。

在主编码器运行前创建 H0。H 不依赖瓶颈反卷积；不在编码器完成后从 RGB 再造一份 H0。主干 PatchEmbed 及可能存在的其他浅层细节路径沿用基准，不能因相似命名把它们一并替换。

## 4. 新编码器交互：精确放在 Swin 后、PatchMerging 前

原 `BasicLayer.forward` 同时包含 Swin 和 PatchMerging。不能在整层返回后插入交互却声称是下采样前。

每个前三阶段执行：

1. 本阶段原 Swin blocks（A1/A2 bias 已关闭）。
2. 取得本阶段原生分辨率 E_i，与当前 H 同步双向更新。
3. 更新后的 E_i 进入原 PatchMerging；更新后的 H 留在256网格继续传递。

不得改变正常 Swin block 数量、PatchMerging 的非 A1/A2 路径、原 skip 缓存位置及索引。参考包的 `x_downsample.append(x)` 在每层入口；不要擅自改成缓存本阶段末尾输出。需要检查 `last_pre_downsample` 等记录的语义，不能为了拿中间特征重复执行一遍 Swin。

### 4.1 首版采用的具体交互

每个编码阶段一套独立投影参数，不跨不同通道数的阶段共享权重。

```text
Hbar = H + R_i(H)
to_H = resize_bilinear(P_EH_i(E), H.shape[-2:], align_corners=False)
to_E = P_HE_i(avg_pool_to_target(Hbar, E.shape[-2:]))

H_next = Hbar + sigmoid(theta_EH_i) * to_H
E_next = E    + sigmoid(theta_HE_i) * to_E
```

- `R_i`：一层64→64的3×3卷积、归一化与激活；沿用基准兼容的归一化风格，不额外堆叠 Swin/Transformer。三阶段各执行一次。
- `P_EH_i`：在 E 原生网格做1×1投影 C_i→64，配归一化，再上采样；不要把384通道整张上采样到256再投影。
- `P_HE_i`：Hbar 的副本先平均池化到目标128/64/32，再1×1投影64→C_i，配归一化。输入512时分别对应池化倍率2/4/8。
- 两个 sigmoid 系数分别初始化到0.05，是待验证的起始设置，不把其数值当作已有收益证据。不要同时串接两个零初始化因子。
- E_next 与 H_next 都读同一轮旧 E/Hbar；不能用 E_next 再计算同轮 H_next。
- 不对 H/E 使用 detach，不用 S、道路概率、GT mask 做硬筛选。小幅交互与持续保留 H 自身路径用于避免直接覆盖，不能据此宣称已经避免信息淹没。
- 记录系数实际值；提供独立的 `encoder_h_exchange` 开关用于受控对照。关闭时跳过交互及 R_i，E/H 原样传递。

本模块设计只是首版候选。不要顺手增加通道注意力、跨尺度全局 attention、骨架一致性或新损失。

## 5. 删除 A1/A2 的范围

新 architecture_id 下删除以下有效前向路径及其专用参数：

- `encoder_stage1_road_attention_head`、`encoder_stage2_road_attention_head`。
- A1/A2 对前两次 PatchMerging 的乘法残差调制或 merge-attention 先验用途。
- 将 A1/A2 打包成 `road_prior` 送入编码器第三阶段的路径。
- 专用于 A1/A2 的 `road_bias_scale_a1/a2`、pair bias 构建和添加；新模型不构造这些闲置可学习系数。
- 专用辅助损失、可视化和指标消费者。旧协议需要的空字段显式 disabled/null，不能输出零 logits 冒充概率。

保留正常 Q/K/V attention、相对位置偏置、正常 PatchMerging 的拼接/归一化/线性投影及其非 A1/A2 的特征残差，保留正常主干瓶颈上下文。不要因名字带 road/global 就删除其他组件。

旧架构若继续使用这些类，按 architecture_id 分流构造，不能直接破坏其旧 checkpoint 兼容性。

## 6. 解码器必须保持 H 的256网格

本地参考 `BidirectionalSkeletonFeatureFusion` 会先将 H resize 到 G 的网格，必须为新架构实现保留两种原生尺寸的版本；不能照搬这个行为。

建议最大程度保留原解码交互的 refine / projection / activation 和参数形状，只调整跨尺度路径：

```text
Hr = highres_refine(H)                          # 64,256,256
Gr = stage_refine(G)                            # 96,128,128
updated_H = GELU(Hr + resize(stage_to_highres(Gr), size=H原生尺寸))
updated_G = GELU(Gr + highres_to_stage(avg_pool_to_target(Hr, G原生尺寸)))
```

同步更新；返回 G、H 的原生尺寸。跨尺度 bilinear 一律 `align_corners=False`，高到低使用明确的平均池化副本。不得通过原地覆盖 H 来取得128版本。

### 6.1 Stage 2

- 正常 decoder / skip / Swin 仍执行。
- structure_branch 与 H_enc 交互，得到 G2 和 H_D2；S2 读取更新后的 G2。
- 语义返回严格仍为 X2。
- G2→G3 的真实多通道路径保留；H_D2 接续传给 Stage 3。
- 继续不构造 C2、D2、旧 gate、旧 feature residual、Stage 2 直接 H→decoder 注入。

### 6.2 Stage 3

- 保留基准已有的 Stage 3 直接 H→decoder 注入；这里允许对 H_D2 的副本做尺度适配，但不能修改持续传递的 H_D2。
- 保留 G2→G3、H_D2↔G3、S3/D3 预测和监督；继续删除 C3、旧 gate、旧 feature residual。
- 得到更新后的 H_D3，保持64×256×256，供扩散和最终融合两处消费者读取。

### 6.3 H_enc 传递接口

编码器显式返回瓶颈、原 skip 列表、H_enc（及必要兼容字段），解码器显式接收 H_enc，不依赖会 detach 的缓存张量取回训练特征。

原解码入口有从 z_struct 再执行 adapter 的行为；新架构中必须跳过这次初始化，直接接续已完成编码交互的 H_enc。adapter 全前向仅执行一次。

## 7. 扩散仍在128网格，算法不改

```text
H_diff = avg_pool_to_target(H_D3, X3.shape[-2:])  # 64,128,128，保留梯度
X3_out = diffusion(X3, G3, H_diff, D3)
```

这样兼容本地参考扩散模块要求四个输入同网格的接口。H_diff 只是供扩散读取的副本，不代替 H_D3。

- 复用基准的 latent32、G/H各投影16、固定八邻域、对称非负边权、越界无效、FP32同步更新。
- 维持 K=8、eta=0.1、beta=0.5，维持 D3 仅在方向偏好路径 detach 的定义。
- 维持 out_proj 零初始化，以及 identity / semantic / semantic_directional 三种模式。
- 不把扩散搬到256，不调整迭代步数，不加C标签、骨架阈值门、路径损失或额外扩散损失。
- 三种模式共享新的 H256/编码交互/A1A2删除/最终H融合设置。identity 只关闭扩散，不能解释为“整个新结构被关闭”。

## 8. 最终预测读取 H256：一次轻量特征残差

本地参考的最终 `up_x4` 从128直接生成512特征，没有现成的256 decoder 层。实施时必须核对 a668711 的真实形状，不能假设存在256层而改动整个上采样器。

首版保留原最终上采样、原 G3→surface 融合和原输出头，在**原 G3 融合之后、原 guided surface/boundary head 之前**插入一次 H 特征残差：

```text
P256 = GELU(Norm(Conv1x1_64_to_16(H_D3)))
Pout = resize_bilinear(P256, X_surface.shape[-2:], align_corners=False)
delta = Conv1x1_16_to_surface_channels(Pout)
X_surface_out = X_surface + delta
surface_outputs = original_guided_head(X_surface_out, ...原有参数...)
```

- 最后投影的 weight/bias 零初始化，不再乘另一个零初始化系数。原模型起始输出不因这条新增路径被立即扰动；上游梯度要在末投影更新后再检查。
- surface_channels 从实际特征读取/构造，不把64误当成主干输出通道。参考包该处通常为96，输出网格为512。
- H 主流继续保持64通道；16通道仅是该消费者的压缩投影。不增加高分辨率3×3大通道卷积或全局attention。
- 提供 `enable_h_surface_residual` 开关，默认true；关闭时不执行该路径。此路径属于特征融合，不是给道路 logits 硬加骨架，也不强制所有骨架响应为道路。
- 保留原 G3→surface 路径，不能将 h3_surface 的 G3 来源误换为 H。
- 不再额外增加第二个 H→logit 修补头，不迁入 S0 版本的其他三路融合、SD gate 或软clDice。
- 若 a668711 与本地参考的 final head 次序不同，按“最终语义特征、原道路预测头之前”定位，并在交付中列出实际插入点；不能暗中变更原输出头定义。

## 9. 损失、训练与评估

保持指定基准实际启用的损失。按已读参考包应为：

```text
L = L_surface + 0.008 L_S2 + 0.012 L_S3 + 0.0024 L_D3
```

- S2/S3 仍位于128网格，复用原目标和有效区域；D3 保留原双角方向标签。
- 不新增编码端骨架头、逐层深监督、骨架一致性、道路覆盖骨架项或新高分辨率骨架损失。
- 原 H 读出头若仍保留，可输出256诊断图，但其独立权重仍为0，不将未被单独监督的读出精度当成有效骨架能力证据。
- C2/D2/C3继续为None，不计算对应损失；末端节点全局模块继续不实例化，保留原主干瓶颈上下文。
- FP32、物理batch2/acc1、80epoch、无EMA、原增强和随机裁剪、学习率、checkpoint选择、验证扫阈值、原生1500的512/stride256滑窗及统一指标口径，均按 a668711 实际协议继承。不得为缓解显存静默改batch、AMP、输入尺寸或学习率。
- 若实际基准系数/协议与参考包不同，先给出源码/配置差异，继承指定基准的实际设置，而非套用旧H0脚本。
- 参数量、单步峰值显存、训练步耗时及推理耗时必须实测；256×64的同类空间卷积约为128×64的4倍，不声称整网训练时间不变。
- 第一轮交付单批次训练/推理验证和运行说明；没有用户的正式训练安排时不自动启动80轮训练。

## 10. 配置、初始化与 checkpoint 契约

新契约至少记录：指定基准完整SHA、实际实现SHA、architecture_id、H通道64、净stride2、H网格256、三次编码交互位置/倍率、encoder_h_exchange、A1A2=false、enable_h_surface_residual、扩散全部配置、C2/D2/C3/global节点关闭状态。

本地参考 `diffusion_protocol.py` 有固定架构字典、严格字段比对、仅允许新增 `.diffusion.` 参数的初始化检查。新架构必须同步扩展模型工厂、训练解析、inspect、恢复训练、阈值扫描、test、topology、初始化清单和严格加载；不能只改网络forward后让旧入口继续误加载。

- 同名同形状参数可以显式迁移，但逐项记录；stem stride变化必须记录，张量相同不等于运算语义相同。
- 列出A1A2删除参数、新编码交互参数、新最终残差参数及任何不可迁移项。
- 默认初始化继承指定基准实际策略；不能未经要求改为从一个已训练 best.pth warm start。
- 消除构造随机数消耗造成的共享权重差异，核对相同保留张量的实际初始化。三种扩散模式共享同样的新架构初始化。
- 旧checkpoint作为显式权重迁移源时不恢复旧优化器；旧架构checkpoint不能直接作为新架构resume。
- 新checkpoint以新ID、完整配置和严格state_dict恢复。不得通过未报告的strict=False忽略架构错误。

## 11. 有针对性的验证与交付

1. batch1/2、512输入形状：H0、H_E1/E2/E3、H_D2/D3全为64×256×256；E保持各自原生网格；X/G/S/D及最终输出匹配基准约定。
2. 运行hook验证每个编码交互在本层Swin后、PatchMerging前，仅执行一次；三次合并实际消费更新后的E。E4、原skip索引与原decoder计算正常。
3. adapter只执行一次，解码器消费的H确为编码器返回的H_enc；禁止重新初始化/重新从RGB生成，保留梯度。
4. 每次双向交互输出各自尺寸；修改E会影响H、修改H会影响E；必要时用确定输入验证同步更新与池化方向，避免仅凭梯度非零认定信息有效。
5. H_diff是128副本，H_D3仍为256；扩散算法原有常量、边界、对称、零方向、旋转、identity/beta0数值检查仍通过。
6. S2/S3/D3损失可训练对应头；surface梯度能通过新编码交互回到H和编码器。零初始化最终H投影/扩散投影更新后，上游梯度有限且非零。
7. 新模型参数清单及运行hook确认A1A2头、PM调制、E3道路bias、旧C/门控/末端节点模块没有被构造或执行，普通attention及相对位置bias仍存在。
8. H最终残差初始为恒等，关闭开关精确跳过；启用且投影非零后改变H会影响最终预测；原G3融合仍执行，不重复注入两份新H残差。
9. 更新所有训练/推理入口，新checkpoint严格保存恢复输出一致；旧architecture_id仍可独立构造。
10. 报告参数数目、FP32 batch2训练峰值显存、训练步时间、batch1推理耗时及设备/输入/预热计时方法，并给出基准对照。没有CUDA条件时如实报告未测项，不编造服务器耗时或显存。

最终交付：独立分支与提交、实际基准核验、变更和参数迁移清单、上述验证证据、新训练/验证/测试命令。该版本同时改变H分辨率、编码交互、A1A2和最终H输出路径，不能将整体增益归因于其中单一模块。
