# 7d2bc9b 统一验证、测试和六项拓扑

`scripts/evaluate_h0_encoder_phase.py`仅增加评价入口，不改训练、网络、数据或损失文件。

1. 从checkpoint args重建原始H0、删除A1/A2、安装压缩后H子位置读取。`strict=True`加载全部权重；拒绝EMA或native-B/H-pixel权重。使用训练时已选定的best.pth，不在测试集挑checkpoint。
2. 14张验证原生1500图，512窗口/256步长，末窗988。RGB使用原数据集的验证预处理；关闭增强、EMA、AMP和TTA。原三角权重的单轴下限0.1，加权融合logit后sigmoid。这与最近三版统一评价相同；不使用旧test_image.py中下限0.05的另一条入口。
3. 扫0.05到0.95、步长0.01。使用全图累计TP/FP/FN的global IoU选阈值，平分时选较小阈值。阈值文件记录checkpoint SHA及epoch。
4. 同一checkpoint推理49张测试图，固定上述阈值，不在测试集调阈值。保存预测PNG、逐图和全局分割指标。
5. 调用 `/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py` 的test入口，导入验证阈值证据；避免旧工具val入口写文件冲突。输出目录由工具创建，不预先塞文件；已有同输入状态可断点恢复。

固定图协议：APLS上限150个GT控制点及150个预测控制点、seed1234；TOPO间隔0.002、步长0.0002；TLTS参数200、seed1234。其余图参数沿用该固定工具版本。六项为clDice、TOPO-P/R/F1、APLS、TLTS-Infeasible；APLS CSV字段是`apls_bidirectional`。

`--command all`顺序执行全部；`infer`只做验证/测试；`topology`复用已保存的49张测试预测；`show`只显示验证阈值、测试分割与六项指标。无需pandas。

输出：`protocol.json`、`val_selection.json`、`val_threshold_scores.csv`、`test_segmentation.csv`、`test_per_image.csv`、`topology/test_comparison.csv`。val概率缓存约126MB，不保存测试浮点特征。
