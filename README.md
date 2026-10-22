# H0 Local C + Gap MLP / Massachusetts

基于599a410的H0 no-global独立版本。协议标识h0-local-gap-mass1500-fp32-noema80-v1。
旧H0 checkpoint不能直接resume/test。Encoder、A1/A2、PatchMerging、global-local
bottleneck、H0高分辨率流、Stage2→3结构传递及H3/H0→Surface路径保留。

## 实际改动

- Stage2：S2/C2控制发送端八邻域传播，自身分母1+sum(w)，一次一格，
  1×1无bias投影，beta初始0；结构入口梯度比例0.5，Surface梯度可达S2/C2。
- Stage3：保留S3/D3。普通C3、两阶段旧Gate/feature_residual、仅服务旧Gate的
  global_context_head移除。anchor global topology关闭，参数不存在。
- P2读取Stage2 layer_up后、H0融合后的96×128×128 decoder特征，并位于Local C传播之前。
  GT为4×4面积平均道路占比，BCE+0.5Dice权重0.05。
- Gap：decoder投影32通道，H0投影16通道，中心及左右各32点有序采样；
  加两端特征、P2/S2可靠性和8维几何，MLP input→128→32→1。
  正反端点顺序logits平均。无一维/空洞/条带卷积和Graph Transformer。
- 概率经有限支撑余弦窄带生成M，分母max(1,sum(K))与预测置信度无关。
  M*correction(X3,H0,M)在norm_up前回写。无候选严格零修正，不直接画道路。
  gamma初始0；前10轮关闭回写，第11～15轮线性启用。比例保存为checkpoint buffer。
- C2默认由原始512骨架8邻域边映射到128；按粗格方向设置跨格边，不凭占据造边。
  同格多道路仍不可区分。c_target=pooled_adjacency用于监督方式对照。
- DCA基准坐标为2*(arange+0.5)/size-1，align_corners=False。
  保留采样特征拼接→卷积。原无效weight_conv在匹配原初始化顺序后移除。

## 候选与GT

候选仅来自预测P2：阈值0.5、固定Zhang–Suen骨架化，去除少于4格的孤立分量。
端点八邻域度数1，向内追踪5步、至少3步有效链估计外向切线；忽略距窗口边界<3格端点。
最多128端点；距离3～32格、两端方向余弦均≥0.5。确定性排序，每端点≤3边、每图≤64边。
直线路径两侧偏移1格。不要求不同预测分量，不用中间高S/H0作为门槛。

GT只在损失/诊断出现，不进入forward。端点在原始GT骨架8像素内匹配，邻近分量歧义忽略。
8像素走廊内GT路径长度≤1.5倍距离+16，且无明确岔路，标正例。
附近弯曲/绕行而直线走廊不支持则忽略；清晰背景端点或无合理局部路径标负。
Gap BCE按有效正负类分别平均再平均，权重0.01；空候选正常训练其他任务。
原S/D/C2损失保留，C3损失删除。第一版无模拟遮挡/GT引导候选。
阈值均是工程起点，尚未验证最优，完整LocalGapConfig保存于args。
use_h0=false只关闭新Gap路径中的H0证据，原H0结构流仍保留。

## 固定协议

Mass原生1500；训练每图每轮随机512一张，80轮，physical batch2、累积1，FP32，无EMA，seed1234。
共享epoch同步persistent workers；预训练5e-5→5e-6、新参数2e-4→1e-5，warmup10。
从相同ImageNet预训练开始。验证每5轮；验证/测试512滑窗步长256，taper加权拼接logits后sigmoid。
无TTA/后处理。统一工具验证扫阈值只算分割，测试冻结阈值并计算拓扑和双向近似APLS。

## 服务器

激活现有swinunet环境，进入本代码目录：

```bash
GPU=0 bash scripts/run_h0_local_gap_mass1500.sh train
GPU=0 bash scripts/run_h0_local_gap_mass1500.sh resume
GPU=0 bash scripts/run_h0_local_gap_mass1500.sh sweep
GPU=0 bash scripts/run_h0_local_gap_mass1500.sh test
GPU=0 bash scripts/run_h0_local_gap_mass1500.sh topology
GPU=0 bash scripts/run_h0_local_gap_mass1500.sh diagnose
```

DATA默认/home/gjj/Swin-Unet-main/Massachusetts/tiff；RESULTS默认/home/gjj/SwinNet-roadbias/model_out。
PRETRAIN默认/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth。
RUN、SEED、GPU、DATA、RESULTS可由环境变量覆盖。LOCAL_GAP_CONFIG传入JSON覆盖用于新训练。
CUDA内存分配器沿用PyTorch默认或调用者环境，不强制expandable_segments。
遇到CUDA驱动错误时可用PYTORCH_CUDA_ALLOC_CONF=backend:native和CUDA_LAUNCH_BLOCKING=1定位。
latest_h0_local_gap_run.txt记录run；run内latest_unified_audit.txt记录扫阈值目录。
resume读取last，test读取best和验证阈值。强制核对架构/协议；优化器恢复失败则报错。

## 诊断与检查

local_gap_audit.jsonl逐batch记录P2/Gap损失、候选正负/忽略、分类TP原始计数、回写强度。
每50次criterion调用审计P2候选覆盖。未审计有明确coverage_audited_images=0。
覆盖目标是P2漏掉的GT中心线、存在局部断开的预测两翼且长度符合搜索范围；
正GT候选覆盖≥80%目标中心线算发现。这是P2候选搜索上限，不是最终修复率。

诊断同一checkpoint关闭/开启Gap回写，复用验证阈值。原生1500上调用统一road_metrics，
报告完整缺口闭合、两翼连接、新增/丢失TP和FP、近邻GT误连代理；保存mask，不保存大量logits。
冻结推理干预不替代训练消融，GT误连代理无法识别所有误连。

```bash
python -m unittest discover -s checks -v
python checks/smoke_full_h0.py --size 64 --backward
python checks/smoke_full_h0.py --size 512
```

检查原始边标签、反向通道、边界、空候选、对称分类、中心采样、置信度保留、warmup加载，
以及实际H0中Surface对Local C和Gap MLP的梯度。
LOCAL_GAP_SOURCE_SHA256.json锁定本实验实现；BASE_MODEL_SHA256.json记录未改动的H0基线源文件。
两份清单都按LF规范化后计算SHA256，Windows与Linux检验结果一致。

能力范围：候选可覆盖的近似直连缺口。整段完全漏检、强弯曲、超出范围、裁剪边界缺口仍有限制。
