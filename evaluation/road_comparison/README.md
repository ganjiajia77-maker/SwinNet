# Random512 三模型统一评估

只读取已经保存的二值道路 mask，不加载模型，不生成或修改预测。适用：每张 1024 图随机裁一张 512 训练，验证/测试在完整 1024 图上滑窗推理的实验。当前 Swin 核对的是 `e2020d9`（retry1 去 P64/PSI 后的 random512），不是 H0/R64。

## 核对结论

**原指标不是全部完全一致。** 分割指标都累计整个数据集的 TP/FP/FN；拓扑指标按图平均。SAM 的“分割指标逐图宏平均”标注已修正，不应再沿用。

| 项目 | Swin random512 | SAM-road | CoANet paper-best random512 |
| --- | --- | --- | --- |
| 已知预训练来源 | ImageNet SwinV2 Tiny，`swinv2_tiny_patch4_window8_256.pth` | SAM ViT-B，`sam_vit_b_01ec64.pth` | ImageNet ResNet101，`resnet101-5d3b4d8f.pth` |
| 核对代码 | `e2020d9:test_image.py`、`evaluate_prediction_topology.py` | 本地 `sam_road-main/eval_data1_road.py`、fair 副本的 `data1_prediction_topology_metrics.py` | `.worktrees/coanet-paper-best-data1-random512`，HEAD `296317b` |
| 滑窗 | tile512，stride256 | 本地 `collect()` 默认 tile512，stride256 | 当前 `test_data1.py` 默认 tile512，stride256；旧入口曾默认512 |
| 拼接 | taper 加权 logits；除以权重和后 sigmoid | taper 加权 logits；但本地分母 `clamp_min(1.0)`，与 Swin 不完全相同 | 均匀平均模型输出；连通头各通道求和 |
| TTA | 核对的 overlap 测试入口没有 | 核对入口没有 | 默认原图/水平/垂直/双向翻转共4次，可关闭 |
| 二值化/后处理 | surface 阈值，无额外形态学处理 | surface 阈值；实际 mask 导出需确认 | 默认 `paper_fusion`：surface≥τ，或 C_d1 通道和≥0.9，或 C_d3 通道和≥2.0；可选 surface-only |
| 旧 APLS 报告 | GT→预测、预测→GT、逐图双向调和平均 | `apls` 为 GT→预测单向近似 | `apls` 为 GT→预测单向近似 |

这是**源码/已有指令核对结果，不是远程实际运行证明**。实际 checkpoint、EMA/raw、stride、TTA、后处理必须从该次日志/配置确认并填入 manifest。预训练文件名相同也不意味着成功加载的参数数量相同。本地 SAM checkout 的 HEAD 为 `800ad78`，但核对的是磁盘上的脚本，不能把 HEAD 自动当作远程运行版本。

SAM 的权重和低于1时，`clamp_min(1.0)` 并非严格加权平均，可能改变边缘概率；这里只记录差异，不改模型脚本、不宣称它造成了目前的分数差异。CoANet 的 TTA 与连通补路融合也不能被“统一计分”消除。若报告方法各自完整推理流程，应明确记录；若要做严格推理配置消融，需要另行用相同流程重新生成预测。

## APLS 是否一致

CoANet 和 SAM 的 Data1 拓扑脚本核心单向实现相同；Swin 的单向部分也使用8邻域像素骨架图、边长1/√2、最多64个控制节点、最短路径长度相似度。已有三个单向数值可作初步对照，但尚非逐位统一：

- Swin 的 KDTree `distance_upper_bound=5` 会拒绝恰好5像素的节点；另外两份接受距离≤5。等距最近节点也可能选不同。
- 三份优先使用 OpenCV Zhang–Suen；缺少 contrib 时 fallback 的图像边界处理不同。
- Swin 的双向 `apls_approx` 不能与 SAM/CoANet 的单向 `apls` 混比。
- SAM 曾报告的 `Raster connectivity proxies` 使用另一套形态学骨架代理，不与后来的 Data1 Zhang–Suen 指标混用。

本工具统一使用**固定 NumPy Zhang–Suen，图像外部补零后细化**，不随 opencv-contrib 是否安装切换算法；统一匹配距离≤5，等距按行优先节点编号选最小者。该边界约定可能让重算结果略不同于旧报告，需要三组一起重算。

APLS 三列名称明确区分：

1. `apls_gt_to_pred_approx`：GT→预测。
2. `apls_pred_to_gt_approx`：预测→GT。
3. `apls_bidirectional_approx`：每张图先算两方向的调和平均，再逐图平均。不是两方向全数据集均值的调和平均。

每张图最多64节点，优先选度数不为2的点，再等间隔补普通节点。所有有限且大于0的参考路径参与；参考图本身不连通的节点对不参与。无法匹配或预测路径不可达记0。匹配到同一点但参考路径非零也得0。某方向没有有效参考路径得0，包括空图。记录有效路径对数量，避免把空图约定误解成道路完整性。

这是 **sampled raster-skeleton APLS proxy**，不能标成官方完整 APLS。官方实现还涉及向道路边插入控制点、图坐标单位、路径筛选等；仅把采样节点和半径设一样不能等价。来源：[CosmiQ APLS 官方实现](https://github.com/CosmiQ/apls)，[路径与控制点代码](https://github.com/CosmiQ/apls/blob/master/apls/apls.py)。

## 统一口径

- 默认在原生 **1024×1024** mask 与 GT 上计算，禁止隐式 resize/裁剪。不使用256评估结果混比。
- mask/GT 必须是单通道0/1或0/255。概率图、彩色可视化、骨架头预测均不接受。
- 三个模型必须覆盖同一 split 的所有 GT case；缺失、多余、重复均报错，不静默取交集。支持 `113_pred.png`、`113_sat_pred.png`、`113.png` 与 `113_mask.png` 等命名。
- IoU/F1/P/R：全数据集累计 TP/FP/FN。clDice、topoP/R、碎片和路径指标：逐图平均。`topo_f1` 是逐图 clDice 的平均，不是 topoP/R 均值的调和平均。
- 连通块8邻域；短块定义面积**严格小于20像素**。碎片密度为每千个预测前景像素的预测连通块数。`frag_idx=预测块数/max(GT块数,1)`。
- 断裂率为 GT 骨架没有被预测道路 mask 覆盖的比例；不额外添加容差。gap 是这些漏检骨架的8邻域连通块。`extra_components` 只是多出的块数，不等于已确认的假阳性数。
- 不新增去小块、闭运算等后处理；若输入已经处理过，必须在 provenance 写明。
- 输出记录图像 ID 列表、二值内容 SHA256、协议和依赖版本。逐图 CSV 每张图写出并刷新，默认逐图显示进度/ETA。
- **不能从已有 PNG 测模型推理时间**；本工具记录的时间仅为 CPU 指标计算时间。

## 配置

复制 `comparison.example.json` 为自己的 `comparison.json`，填写三个模型 checkpoint、实际推理配置和预测目录。模板中的阈值只是占位示例，**不是已经替你选好的阈值**。相对路径相对于 JSON 所在目录；本地 Windows 路径推荐用 `D:/...`。

每个模型的 `predictions.val` / `predictions.test` 是目录列表，每个目录注明当时生成 mask 的阈值。若已有多个验证阈值的预测，可这样填：

```json
"val": [
  {"threshold": 0.40, "path": "/path/to/val_thr040/surface"},
  {"threshold": 0.45, "path": "/path/to/val_thr045/surface"},
  {"threshold": 0.50, "path": "/path/to/val_thr050/surface"}
]
```

`test` 至少需要验证集所选阈值对应的目录。一个二值 mask 目录不能通过修改 JSON 的 threshold 变成另一个阈值的预测。

## 运行

不需要 GPU / torch / 模型仓库。已有 numpy、scipy、cv2 的环境可直接运行。不需要额外安装另一个 OpenCV 包；缺依赖才运行 `python -m pip install -r requirements.txt`。

### A. 有验证集候选 mask：自动选择后测试

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
cd /home/gjj/unified_random512_metrics

python -u compare_predictions.py --manifest comparison.json --split val --output_dir results_val
python -u compare_predictions.py --manifest comparison.json --split test --selection results_val/val_selection.json --output_dir results_test
```

验证只扫描已有二值目录，按 global IoU 最大选择；相同分数选较低阈值。只有一个候选时，只能说评估了该候选，不能说完成了阈值寻优。扫描阶段只算分割计数，然后只为选中候选计算拓扑，避免每个阈值重复 APLS。

### B. 已经在验证集选过阈值：直接导入记录

在每个模型的 `validation_selection` 填实际阈值、`selected_on: "val"`、准则及验证集扫描日志路径，然后运行：

```bash
python -u compare_predictions.py --manifest comparison.json --split test --import_val_selection --output_dir results_test_imported
```

该模式检查日志存在并保存 SHA256，但不自动解析任意格式日志；输出明确标记为用户声明的验证选择，不能替你证明日志内容。**没有验证集选择记录时，不应使用测试集最优阈值冒充验证选择。** 没有对应 test 二值图时，必须补生成预测，不能重设已有 PNG 的阈值。

本地 PowerShell：

```powershell
cd D:\Code\Swin-Unet-main\evaluation\road_comparison
& D:\torch-cu128\Scripts\python.exe -u .\compare_predictions.py --manifest .\comparison.json --split test --import_val_selection --output_dir .\results_test_imported
```

输出：`test_comparison.csv` 总表、`test_report.json` 完整审计记录、每模型 `*_test_per_image.csv`。若需统计差异可信度，可在逐图结果上做配对 bootstrap；本脚本不把一次总体均值差异当成统计显著。

## 验证

```bash
python -m unittest discover -s . -p test_unified_metrics.py -v
```

覆盖完美/断路 mask、斜边长度、半径恰好5、等距匹配、空图、全局汇总、边界细化、尺寸/缺图/重复错误，以及三模型的 val 选择不能被 test 更优阈值替换。没有实际预测目录前，不能声称已完成1246张真实预测重算。
