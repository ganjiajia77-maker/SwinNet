#!/usr/bin/env python3
"""Audit the original v8 computation graph using source only, without model weights."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'networks/reliable_directional_repair.py': '6b71c441a79583696f863db2b4f88016ac6c8ec53e205757f37cb4adbd95a082',
    'run_reliable_v8_mass.py': '1c910e7a27f6a504746a7eb2d0b27fd8b04331928bbf47444c72435db478f419',
}
TARGETS = {
    'networks/reliable_directional_repair.py': [
        ('compose_surface_logits', None), ('DirectionRelations', 'forward'),
        ('MultiScaleGateRelation', '__init__'), ('MultiScaleGateRelation', '_source'),
        ('MultiScaleGateRelation', 'forward'), ('DirectionalDynamics', 'forward'),
        ('ReliableDirectionalRepair', 'forward'), ('dynamic_losses', None),
    ],
    'run_reliable_v8_mass.py': [('load_experiment', None), ('train', None)],
}


def function_node(tree, owner, method):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == owner:
            if method is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method:
                        return child
    raise ValueError(f'Cannot find {owner}.{method or ""}')


def read_sources():
    sources, fingerprints = {}, {}
    for name, expected in EXPECTED.items():
        path = ROOT / name
        data = path.read_bytes().replace(b'\r\n', b'\n')
        actual = hashlib.sha256(data).hexdigest()
        fingerprints[name] = actual
        if actual != expected:
            raise ValueError(f'{name} differs from original v8 source: expected {expected}, got {actual}; '
                             'use the diagnostic archive for this revision.')
        sources[name] = data.decode('utf-8')
    return sources, fingerprints


REPORT = r'''# 原始 V8 信息流：源码静态审计

本报告针对已校验哈希的原始 v8、`relation_mode=full`、`width=32`。只读取源码，没有加载 checkpoint、真实图像或 GT，不能给出实际分布、梯度大小或因果结论。源函数原文及行号见 `source_excerpts.md`。

## 1. gate_relation 输入与通道

| 位置 | 实际拼接 | 通道 |
|---|---|---:|
| local.0 | H 的 1x1 投影 16 + H3 的 1x1 投影 16 + 冻结 visual 32 | 64 |
| pair.0 | center、left、right、abs(left-right)、left*right 各 32 + geometry 8 | 168 |
| delta.0 | center 32 + context 32 + 八方向加权 relation 32 + P0、Q3、coverage 各 1 | 99 |

H/H3 也先经冻结 VisualProjection 得到 visual。context 是 center 经 dilation=4 的深度卷积及 1x1 卷积。双侧来源从 center+context 沿每个轴方向、半径 2/4/8 采样，用 R 加权平均；coverage 是来源 R 的平均。geometry 依次为左右 coverage、该轴 two、one、P0、Q3、方向 dy、dx。

P0/Q3 进入 pair geometry 和 delta；R 用于来源采样与 coverage；W 在 pair 特征算出之后加权其输出与 coverage，不是 pair.0 的直接输入。原始 two/one 来自冻结 DirectionRelations。

## 2. 八方向关系与 G

每个方向独立经过共享 pair 网络得到 32 通道 embedding，再计算 `relation = sum_d W_d * pair_d`。delta 读取这 32 通道及 center、context、P0、Q3、coverage，预测单通道 DeltaG_logit。逐方向 pair 仅在 capture_diagnostics 返回，融合后没有独立方向通道继续传入 delta。两条中心/上下文特征仍单独进入 delta，因此不能单凭单通道 G 判定信息过度压缩。

旧 W 由方向头 logits 与方向证据计算：`W_dir = softmax(logits_dir + log(directional + 0.05))`。v7 的 `path_continuity = sum(W_dir * (two_by_direction + one_by_direction))`；`old_G_logit = logits_gate + log(clamp(path_continuity, min=1e-6))`；`G_v8 = sigmoid(old_G_logit + DeltaG_logit)`。方向顺序是左、右、上、下、左上、右下、右上、左下。

## 3. 动力学实际计算

在原始 v8 gate 路径，令 U0=P0_work：

```
gained_k = clamp((U_k - P0)_+ / clamp(1-P0, min=1e-5), max=1)
SourceSupport_k = R + (1-R) * gained_k
mass_k = sum_d W_d(p) * SourceSupport_k(p+direction_d)
V_k = clamp(mass_k, 0, 1)
M_k = sum_d [W_d * shifted(SourceSupport_k) * shifted(U_k)] / mass_k
      若 mass_k <= 1e-6 则 M_k = U_k
c = softmax(rates(visual, P0, Q3, R, B, A, U_k, V_k))
transport_rate_k = c[1]
alpha_k = c[1] * G_v8
beta_k = 0.2 * A * c[2]
gamma_k = c[3]
U_{k+1} = U_k + alpha_k*(M_k-U_k) + beta_k*(B-U_k) + gamma_k*(P0-U_k)
```

上述是原始 v8 公式；V 进入 rates 网络，但 alpha 没有显式乘 V。G 增大不能保证补路：c[1] 可能小，M-U 可能接近零或为负，提议及锚定项也会影响结果；无来源时 M=U。质量很低但非零的来源在归一化后可能影响 M，高 G 有可能放大这类运输；是否造成实际误检需要前向验证。

三步后 `delta_z=logit(clamp(U3))-logit(clamp(P0_work))`，上采样后加到全分辨率 Z0；最终 P1=sigmoid(Z1)。

## 4. 训练与监督

加载 v7 后先冻结原模型，然后新建可训练 gate_relation；optimizer 仅接收 gate_relation 参数。H/H3/P0/Q3 来自冻结路径，旧 visual、R、W、B、A 和动力学模块冻结；新投影、local、context、pair、delta 可训练。

总损失为 `L_seg + lambda_cont * L_cont`，由最终 P1 经 U3、alpha、G 间接监督 gate_relation。L_seg 使用加权 focal BCE 和 Dice；L_cont 使用 GT 骨架和冻结 P0 建立漏检段，监督 P1 的路径概率。没有独立的连接对正负标签、方向级关系标签或关系排序损失。

delta 末层零初始化，初始 DeltaG 为 0；首次反传时末层之前的关系层通过该末层收到的梯度为零，后续是否产生足够梯度需要实际训练数据验证。

## 源码事实与尚未验证的假设

**事实：**方向 embedding 在 delta 前按 W 求和；训练仅间接通过 P1/连续性损失指导关系；G 路径的 alpha 不显式乘 V；上游特征冻结。

**假设，不能仅凭源码确认：**W 融合稀释了有效方向、H/H3 没有被利用、旧 G 饱和、间接监督不足以学习连接关系、低质量来源导致误连。需要原始 BASE/Q3 及 v7/v8 权重，或含所需中间张量的完整缓存，才能做对应前向/梯度干预。

目前无法依据静态结构确定哪一个环节是实际失败的首要原因。

## 实际计算图

```
冻结底座/Q3 → H、H3、P0、Q3 → 冻结 visual/R/W/旧G
                     │             │
                     └→ gate_relation: 投影 → center/context
                                      → R 条件采样 → 方向 pair
                                      → W 加权融合 → DeltaG_logit
                                                    │
                                 G_v8 = sigmoid(旧G_logit + DeltaG_logit)
                                                    │
                     冻结 R/W/B/A → 三步 U 更新 ← alpha = G_v8*c[1]
                                                    │
                                  上采样 U3 的 logit 增量 → P1
```

## 精简职责表

| 模块 | 职责 | v8 训练 |
|---|---|---|
| 基础模型/Q3 | 产生 Z0、P0、Q3、H、H3 | 冻结 |
| visual/R/DirectionRelations | 产生视觉特征、可靠来源、W、旧 G | 冻结 |
| gate_relation | 计算 DeltaG_logit | 可训练 |
| B/A | 提议道路状态和接受度 | 冻结 |
| 三步动力学 | 运输、提议及 P0 锚定 | 参数冻结，允许梯度传回 gate |
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    sources, hashes = read_sources()
    sections, locations = [], {}
    for name, requested in TARGETS.items():
        tree = ast.parse(sources[name], filename=name)
        lines = sources[name].splitlines()
        for owner, method in requested:
            node = function_node(tree, owner, method)
            label = owner + ('.' + method if method else '')
            locations[label] = {'file': name, 'start_line': node.lineno, 'end_line': node.end_lineno}
            excerpt = '\n'.join(f'{i}: {lines[i-1]}' for i in range(node.lineno, node.end_lineno + 1))
            sections.append(f'## {name}: {label}\n\n```python\n{excerpt}\n```\n')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'REPORT.md').write_text(REPORT, encoding='utf-8')
    (args.output_dir / 'source_excerpts.md').write_text('\n'.join(sections), encoding='utf-8')
    (args.output_dir / 'source_metadata.json').write_text(
        json.dumps({'source_hashes_sha256_lf': hashes, 'locations': locations,
                    'checkpoint_loaded': False, 'model_forward_run': False}, indent=2),
        encoding='utf-8')
    print(f'源码静态诊断完成：{args.output_dir.resolve()}')
    print('查看 REPORT.md 和 source_excerpts.md；本次未加载模型权重。')


if __name__ == '__main__':
    main()
