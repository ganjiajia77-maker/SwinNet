# 原 H0：编码阶段道路与骨架学习、持续高分辨率特征保留

将本文件作为代码修改任务执行。目标是让编码器在下采样前逐阶段学习道路与骨架相关特征，并持续传递高分辨率特征。本文规定一个待验证方案，不承诺性能或连通性提升。

## 1. 基准及改动范围

- 基于原 H0 提交 `599a410de8d341850e1810417307e4b3bbdfc82d`。首先核验该提交及实际结构，不以当前工作区、e2020d9、S0、H256 或方向扩散版代替。
- 先读取适用的 AGENTS.md，保护已有改动。建立独立实现，建议分支 `codex/h0-encoder-road-skeleton`，架构标识 `h0_encoder_road_skeleton_v1`。
- 本轮实施编码器三次双向交互、编码道路辅助监督、编码 H 骨架辅助监督，以及编码到解码的 H 状态传递。
- 保留原主干/decoder 的 Swin 数量、skip、DCA、瓶颈、PatchMerging 非新增部分、Stage 2/3 S/C/D 和反馈、G2→G3、G3→surface、末端全局拓扑。
- A1/A2、其下采样调制和 E3 attention bias 保留原定义。本轮不修复 DCA 加权覆盖问题，不增加扩散、H256、最终 H→surface 新旁路、骨架一致性或 soft-clDice。
- 原版本须能独立构造和加载原 checkpoint。完成实现、必要验证和服务器命令，不自动启动正式 80 轮训练。

## 2. 固定结构：一条 H 流，三次编码交互

512 输入时，保留原 RGB 浅层卷积侧流和 adapter，输出 `H0=B×64×128×128`。不是另建第二条骨架特征流，也不把 PatchEmbed 输出当成原 RGB 侧流。

```text
RGB → 原浅层侧流 → 原 adapter 一次 → H0(64,128,128)
                                         │
主干 E1：原 Swin → exchange(E1,H0)  → E1' → 原 A1/第一次 PatchMerging
                              └──→ H_E1 → 编码骨架预测 S_E1
主干 E2：原 Swin → exchange(E2,H_E1)→ E2' → 原 A2/第二次 PatchMerging
                              └──→ H_E2 → 编码骨架预测 S_E2
主干 E3：原 Swin → exchange(E3,H_E2)→ E3' → 原第三次 PatchMerging
                              └──→ H_E3 → 编码骨架预测 S_E3
主干 E4及瓶颈：沿用原流程

原 decoder Stage 2：接收 H_E3，原交互更新 H_D2 / G2，原 S2/C2/D2
原 decoder Stage 3：接收 H_D2 和 G2，原交互更新 H_D3 / G3，原 S3/C3/D3
原全局拓扑和最终 surface：继续消费对应的最终 H/G 特征
```

- 道路辅助头分别读取交互更新后的 `E1'/E2'/E3'`。
- H 全程保持 64 通道、128×128；E1/E2/E3 分别保持 96×128×128、192×64×64、384×32×32，再进入原下采样。
- H 是受骨架监督的多通道结构特征。S 是从 H 读出的单通道骨架 logits，不用 S 替代 H，也不用 S 阈值或 GT mask 乘法截断特征。
- E4 不新增交互或辅助头。

## 3. 编码交互的精确位置与运算

原 `BasicLayer.forward` 内含 Swin 和 PatchMerging。增加显式接口，在本阶段全部 Swin blocks 完成后、A1/A2 预测及 PatchMerging 前交互；不能放在整层返回后。

三个阶段各一套独立的轻量双向残差投影：

```python
# E: 本阶段 Swin 后的原生网格特征；H: 上一轮的128网格特征
to_H = bilinear(P_EH_i(E), size=(128, 128), align_corners=False)
to_E = P_HE_i(adaptive_avg_pool2d(H, E.shape[-2:]))
H_next = H + sigmoid(theta_EH_i) * to_H
E_next = E + sigmoid(theta_HE_i) * to_E
```

- `P_EH_i`：原生 E 网格上 1×1 Conv，C_i→64，再 GroupNorm(8)。
- `P_HE_i`：先池化 H 的副本，再 1×1 Conv，64→C_i，再 GroupNorm(8)。投影卷积不带 bias；不额外堆 attention 或 Transformer。
- 两个方向同步读取旧 E/H，不能先更新 E 再用 E_next 更新 H。H 主流不能被池化副本覆盖。
- 六个独立 sigmoid 系数初始化到 0.05。这是明确的首版起始配置，需记录，不声称最优。不要再叠加零初始化输出投影导致双重关断。
- E/H 不 detach，不新增梯度缩放；原解码结构入口的梯度比例仍按原 H0。
- E_next 进入 A1/A2（对应阶段）和 PatchMerging。E3 原 road_prior attention bias 仍按原位置执行。
- 保留原 skip 采集时点及索引。原 `x_downsample.append(x)` 在层入口，不在本轮顺手改成阶段末尾。
- 不重复执行 Swin、浅层 stem 或 adapter 来获取中间特征。

## 4. 编码辅助预测：道路约束 E，骨架约束 H

### 道路预测

- 新增三个独立 `Conv2d(C_i,1,1)`，读取 E1'/E2'/E3'，输出 logits。
- 这些是训练监督头，不修改原 A1/A2 前向，不给 PatchMerging 新增一套道路 gate。
- 为复用原 surface 损失，将各道路 logits 双线性上采样到当前512裁剪尺寸，与同一份道路 GT 计算原 surface 损失。沿用原 focal 参数和 Dice 系数，不经过阈值化。
- 原 `RoadAttentionHead.forward` 返回 sigmoid 概率，不能将其输出直接送入 BCEWithLogitsLoss。本轮使用上述独立 logits 头，避免修改 A1/A2 接口。

### 骨架预测

- 新增三个独立 `Conv2d(64,1,1)`，分别读取 H_E1/H_E2/H_E3，输出 128×128 logits。
- 本轮不强制预测头参数共享。原解码骨架头读取96通道 G2/G3，继续保留，不改为读取 H，不修改 C/gate 的骨架来源。
- 编码和解码共用骨架真值来源、目标处理和损失定义；不要求共用预测头参数。
- 复用原 Stage 的 `build_soft_stage_skeleton_target` 与 `soft_stage_skeleton_loss`，包含原骨架最大池化、soft target、正类权重和 Dice 定义。不另写一套降采样/膨胀目标。
- 三个编码骨架头均使用128网格目标。GT 只用于损失，不进入推理前向。
- 编码道路与骨架预测经可微的辅助输出接口返回训练器，供损失和诊断读取；不能只存入 detach 缓存。可视化副本可以 detach。

## 5. 损失配置：明确预算，避免改动 C/D

以下是首版可配置起始值，不是已验证的最优权重。将三阶段分别记为 i=1,2,3：

```text
L_enc_road = (L_R1 + L_R2 + L_R3) / 3
L_enc_skel = (L_SE1 + L_SE2 + L_SE3) / 3

L_total = L_original_H0_adjusted_S
          + 0.10 * L_enc_road
          + 0.004 * L_enc_skel
```

- 编码道路总权重 `encoder_road_aux_weight=0.10`，三个阶段等分。
- 编码骨架总权重 `encoder_skeleton_aux_weight=0.004`，三个阶段等分。
- 原 Stage 2/3 骨架项有效系数为0.008/0.012，总和0.020。本轮将其中0.004分配给编码骨架；解码骨架有效系数改为0.0064/0.0096，总和仍为0.020。
- **不能直接把原 stage2/3_skeleton_weight 改小**：原代码中它同时乘 C/D。保持 stage weight=0.008/0.012 及 C/D factor，仅将 `stage_skeleton_only_loss_factor` 从1.0设为0.8，实现只缩放 S。`stage_skeleton_loss_factor` 保持1.0。
- 原 C/D 的有效损失系数、目标、mask、梯度比例保持不变；最终 surface、边界及其他实际启用项按原 H0。
- 原独立 `highres_structure_skeleton_weight=0.0` 保留，避免与新增三个编码 H 监督重复计损失；解码 H 不额外增加监督项。
- 新编码头只经过其对应任务损失，不复用 decoder stage 输出列表，避免被原 S/C/D loss 分发器重复计算。
- 明确记录各原始损失、加权损失和 S/C/D 的实际系数。骨架总系数一致不代表梯度大小完全相同，需如实记录。

## 6. H 的编码到解码接力：避免被重新初始化

- 原 H0 的浅层侧流和 adapter 在解码入口附近构造；本版将同一套初始化提前到主编码器前，全前向只执行一次。
- 编码器显式返回 `H_enc=H_E3`，解码器显式接收。
- 原解码入口不能再用 z_struct 执行 adapter 重新生成 H0，不能清空/覆盖已更新 H_enc。
- 原 Stage 2 从 H_enc 接续，Stage 3 从 H_D2 接续；原全局模块继续读取最终 H_D3。初始化 H0 与后续各状态使用不同诊断键，防止命名混淆。
- 骨架 logits 不作为新增特征输入传给 decoder；其信息通过受监督的 H 特征传递。原 decoder 骨架预测及其 C/gate 消费关系继续保留。

## 7. 训练、开关和版本记录

- 继承原 H0 Massachusetts 协议：原生1500、每图每轮随机512裁一张、80epoch、物理batch2/acc1、FP32、无EMA、warmup10、相同预训练/学习率分组/增强/裁剪序列。
- 原检查点选取、验证阈值规则、512/stride256加权logits滑窗、无TTA和后处理设置沿用，不改训练曝光量来制造收益。
- 从原预训练初始化策略开始，不默认加载已训练 best.pth，不冻结主干。
- 新增配置：`encoder_structure_exchange`、`encoder_road_aux_weight`、`encoder_skeleton_aux_weight`。权重及交互状态写入 resolved_config/checkpoint/结果元数据。
- 提供原 H0 控制模式：交互关闭、编码辅助权重均0、原解码 S factor恢复1.0。共享权重相同时，该模式应与原 H0 输出对齐。
- 编码骨架权重为0.004时解码 S factor为0.8；如允许调节其预算，用 `decoder_S_factor=(0.020-encoder_skeleton_aux_weight)/0.020`，限制预算在[0,0.020]，明确日志，避免开关关闭后仍遗留0.8。原0.008/0.012系数前提必须核验。
- 新增头与交互的初始化不得改变共享主干、decoder 的原始初始化；隔离新参数的随机数消耗，并逐项核对共享张量。训练数据采样 RNG 同样保持独立。
- 同步模型工厂、训练、inspect、resume、验证、test、阈值扫描及导出入口。新版本使用独立 architecture_id 和源码清单，不绕过原包SHA校验后仍冒充未经修改的599a410。

## 8. 必须验证和交付

1. 512输入、batch1/2：E1/E2/E3原生尺寸正确，所有编码/解码 H 为64×128×128，最终 logits 为512。
2. hook验证真实顺序：Swin→双向交互→对应 A1/A2→PatchMerging；原 skip 采集时点未变。
3. stem/adapter仅执行一次；解码实际消费 H_E3，末端全局实际消费解码更新后的 H。
4. 单批次分别反传编码道路损失、编码骨架损失，确认对应预测头、交互模块和编码主干有有限梯度；原 decoder C/D 有效系数不变，S总预算0.020。
5. 共享权重下，关闭新增路径并恢复原损失预算的控制模式与原 H0 eval 输出对齐。
6. 新 checkpoint 严格保存/恢复及推理对齐；验证/测试不读取骨架或道路 GT 作为模型输入。图像边缘、空骨架裁剪不出现NaN。
7. 报告参数量、同环境单步耗时和峰值显存；保存 H_E1/E2/E3 骨架预测及道路辅助预测的诊断接口，以便检查信息是否在编码过程中丢失。

交付简短改动清单、验证结果、原H0控制命令、新版80epoch训练及评估命令。先交付实现供核对，不自动开始正式训练。
