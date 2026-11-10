"""Print compact diagnostic tables with only the Python standard library."""
import argparse
import csv
import json
from pathlib import Path
import unicodedata


def read(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def render(title, headers, rows):
    print('\n========== ' + title + ' ==========')
    if not rows:
        print('尚未生成')
        return
    cells = [list(map(str, row)) for row in [headers] + rows]
    width = lambda value: sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in value)
    sizes = [max(width(row[i]) for row in cells) for i in range(len(headers))]
    for row in cells:
        print(' | '.join(value + ' '*(sizes[i]-width(value)) for i, value in enumerate(row)))


def fmt(value):
    return f'{float(value):.6f}' if value not in (None, '') else '无样本'


def pooled(rows, key):
    usable = [r for r in rows if r.get(key) not in (None, '') and int(r['pixels']) > 0]
    count = sum(int(r['pixels']) for r in usable)
    return sum(float(r[key])*int(r['pixels']) for r in usable)/count if count else None


def show(out):
    out = Path(out)
    if (out/'protocol.json').is_file():
        protocol = json.loads((out/'protocol.json').read_text())
        print(f"目录：{out}\n固定阈值：{protocol['threshold']}；验证集14张；同一checkpoint三模式")
    comparison = read(out/'comparison.csv')
    fields = ['mode', 'iou', 'cldice', 'closed', 'newly_closed_vs_normal', 'lost_closed_vs_normal',
              'new_bridge_proxy_vs_normal', 'removed_bridge_proxy_vs_normal', 'added_tp', 'lost_tp', 'added_fp', 'removed_fp']
    render('同阈值结果，所有变化均相对normal',
           ['模式', 'IoU', 'clDice', '闭合', '新增闭合', '丢失闭合', '新增桥接代理', '反向桥接代理', '新增TP', '丢失TP', '新增FP', '消除FP'],
           [[fmt(r[k]) if k in ('iou', 'cldice') else r[k] for k in fields] for r in comparison])
    path_rows = read(out/'topology'/'val_comparison.csv')
    render('同口径六项拓扑：APLS上限150，TOPO间隔0.002，步长0.0002',
           ['模式', 'clDice', 'TOPO-P', 'TOPO-R', 'TOPO-F1', 'APLS', 'TLTS-Infeasible'],
           [[r.get('name', r.get('model', '?'))] + [fmt(r.get(k)) for k in
             ('cldice', 'topo_precision', 'topo_recall', 'topo_f1', 'apls_bidirectional', 'tlts_infeasible')]
            for r in path_rows])
    logit = read(out/'logit_changes.csv')
    values = []
    for region in dict.fromkeys(r['region'] for r in logit):
        for comparison_name in dict.fromkeys(r['comparison'] for r in logit):
            selected = [r for r in logit if r['region'] == region and r['comparison'] == comparison_name]
            values.append([region, comparison_name, sum(int(r['pixels']) for r in selected),
                           fmt(pooled(selected, 'signed_mean')), fmt(pooled(selected, 'abs_mean'))])
    render('整图融合后的实际道路logit变化', ['区域', '差值定义', '像素', '有符号均值', '绝对均值'], values)
    weights = read(out/'source_weights.csv')
    values = []
    for region in dict.fromkeys(r['region'] for r in weights):
        selected = [r for r in weights if r['region'] == region and r['source'] == 'self']
        values.append([region, fmt(pooled(selected, 'normal_mean')), fmt(pooled(selected, 'c_prior_one_mean')),
                       fmt(pooled(selected, 'all_sources_l1_change_mean')),
                       sum(int(r['argmax_changed_pixels']) for r in selected)])
    render('取消C先验的来源选择变化', ['区域', '正常保持权重', 'C=1保持权重', '九来源L1变化', '最大来源改变像素'], values)
    residual = read(out/'residual_components.csv')
    values = []
    for mode in ('normal', 'c_prior_one'):
        for region in dict.fromkeys(r['region'] for r in residual):
            selected = [r for r in residual if r['mode'] == mode and r['region'] == region]
            values.append([mode, region] + [fmt(pooled(selected, k)) for k in
                ('residual_abs_mean', 'weighted_component_abs_mean', 'projection_bias_abs_mean',
                 'residual_change_from_normal_abs_mean')])
    render('特征修正与投影偏置：不是道路概率', ['模式', '区域', '|总残差|', '|W·delta|', '|bias|', '|残差-正常残差|'], values)
    checks = out/'checks.json'
    if checks.is_file():
        data = json.loads(checks.read_text())
        print('\n冻结与恢复检查：' + json.dumps(data, ensure_ascii=False))
    print('\n注意：bias-only只是末端道路头回放；其logit影响可随位置变化。')
    print('桥接为代理计数；正向与反向计数不保证一一对应，不能直接相减当作净错误桥接数。')
    print('完整逐图、逐缺口、逐方向结果保存在同目录CSV；当前关闭分支不等于历史原B。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    show(parser.parse_args().output_dir)
