# -*- coding: utf-8 -*-
"""Generate P0I pilot HTML report with charts."""
import json, statistics, math
from pathlib import Path

base = Path('D:/training')
h2_eval = json.loads((base / 'P0H2_V2_P0H5_EVAL.json').read_text())
n150 = json.loads((base / 'matched_eval_results_n150.json').read_text())
m1_reset = json.loads((base / 'checkpoints/p0i_pilot_m1_reset_s42/train_log.json').read_text())
m1_persist = json.loads((base / 'checkpoints/p0i_pilot_m1_persist_s42/train_log.json').read_text())
h2_mon = json.loads((base / 'checkpoints/ouro_retention_m2_h2v2_final_rerun/monitor_history.json').read_text())
h2_train = json.loads((base / 'checkpoints/ouro_retention_m2_h2v2_final_rerun/train_log.json').read_text())

# --- data extraction ---
m0_acc = h2_eval['M0_teacher']['candidate_accuracy']
m0_rank = h2_eval['M0_teacher']['mean_rank']
m0_margin = h2_eval['M0_teacher']['mean_margin']
m0_ent = h2_eval['M0_teacher']['entropy']
h2v2 = h2_eval['H2_v2']

# n150 rows
rows = {}
for r in n150['rows']:
    rows[r['tag']] = r

m1_acc = rows['M1_migrated']['cand_acc'][0]
m1_cos = rows['M1_migrated']['logit_cos_mean']
m1_dnll = rows['M1_migrated']['nll_diff_mean']
n600_acc = rows['M2_d2_norm600_gamma_frozen']['cand_acc'][0]
n600_cos = rows['M2_d2_norm600_gamma_frozen']['logit_cos_mean']
n600_dnll = rows['M2_d2_norm600_gamma_frozen']['nll_diff_mean']
n1800_acc = rows['M2_d3_norm1800total_gamma_frozen']['cand_acc'][0]
n1800_cos = rows['M2_d3_norm1800total_gamma_frozen']['logit_cos_mean']
n1800_dnll = rows['M2_d3_norm1800total_gamma_frozen']['nll_diff_mean']

# H2-v2 monitor steps
mon_steps = [m['step'] for m in h2_mon]
mon_acc = [m['cand_acc'] for m in h2_mon]
mon_rank = [m['mean_rank'] for m in h2_mon]
mon_margin = [m['mean_margin'] for m in h2_mon]
mon_cos = [m['logit_cosine'] for m in h2_mon]
mon_dnll = [m['delta_nll'] for m in h2_mon]
mon_hcos = [m['hidden_cos'] for m in h2_mon]

# Training losses (smoothed)
def smooth(losses, w=20):
    out = []
    for i in range(len(losses)):
        lo = max(0, i - w)
        out.append(statistics.mean(losses[lo:i+1]))
    return out

m1r_losses = m1_reset['losses']
m1r_smooth = smooth(m1r_losses)
m1p_losses = m1_persist['losses']
m1p_smooth = smooth(m1p_losses)
h2_losses = h2_train['losses']
h2_smooth = smooth(h2_losses)

# --- HTML ---
html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>P0I Pilot — Ouro-1.4B Retention Substrate Experiment Report</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;line-height:1.6}
.container{max-width:1400px;margin:0 auto;padding:24px}
h1{font-size:2em;color:#58a6ff;border-bottom:2px solid #30363d;padding-bottom:12px;margin-bottom:24px}
h2{font-size:1.5em;color:#79c0ff;margin-top:36px;margin-bottom:16px;border-left:4px solid #58a6ff;padding-left:12px}
h3{font-size:1.15em;color:#d2a8ff;margin-top:24px;margin-bottom:12px}
p{margin-bottom:12px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:0.9em}
th,td{border:1px solid #30363d;padding:8px 12px;text-align:center}
th{background:#161b22;color:#58a6ff;font-weight:600}
tr:nth-child(even){background:#161b22}
.good{color:#3fb950;font-weight:bold}
.bad{color:#f85149;font-weight:bold}
.warn{color:#d29922;font-weight:bold}
.neutral{color:#8b949e}
.chart-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:16px 0}
.note{background:#1c2128;border-left:4px solid #f85149;padding:12px 16px;margin:16px 0;border-radius:4px}
.insight{background:#1c2128;border-left:4px solid #3fb950;padding:12px 16px;margin:16px 0;border-radius:4px}
.code{background:#1c2128;padding:12px;border-radius:4px;font-family:monospace;font-size:0.85em;overflow-x:auto;border:1px solid #30363d}
.footer{margin-top:48px;padding-top:16px;border-top:1px solid #30363d;color:#8b949e;font-size:0.85em}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.8em;font-weight:bold}
.tag-pass{background:#1a3a1a;color:#3fb950}
.tag-fail{background:#3a1a1a;color:#f85149}
.tag-warn{background:#3a3a1a;color:#d29922}
.tag-na{background:#2a2a2a;color:#8b949e}
</style>
</head>
<body>
<div class="container">

<h1>P0I Pilot Report — Ouro-1.4B Retention Substrate</h1>
<p style="color:#8b949e;">Generated: 2026-08-26 | Canonical root: <code>D:\\training</code> | Seed: 42</p>

<div class="note">
<strong>核心问题：</strong>
<em>Does internal cross-turn recurrent persistence causally improve task behavior after recurrent conversion?</em>
</div>

<h2>1. 实验全景总览</h2>
<p>本报告整合 Phase 0H (distillation) 与 Phase 0I pilot (task training) 的全部已有数据。
0H 阶段证明 teacher imitation 与 candidate ranking 系统性错位（teacher fidelity 越高，candidate ranking 反而越差）。
0I pilot 改用<strong>真实 CLUTRR task objective</strong>（candidate cross-entropy），而非 teacher imitation，
在 M1 和 norm1800 两个初始化上测试 reset vs persist 的因果效应。</p>

<h2>2. P0H5 Canonical N=150 Candidate-Scoring 统一比较</h2>
<table>
<tr>
<th>模型</th><th>Acc</th><th>Mean Rank</th><th>Margin</th><th>Entropy</th><th>Logit Cos</th><th>ΔNLL</th><th>Spearman vs M0</th><th>判定</th>
</tr>
<tr><td><strong>M0 teacher</strong></td><td class="warn">11/150</td><td>4.47</td><td>-2.05</td><td>3.81</td><td>1.000</td><td>0.00</td><td>1.000</td><td><span class="tag tag-warn">baseline</span></td></tr>
<tr><td>M1 migration</td><td class="warn">15/150</td><td>—</td><td>—</td><td>—</td><td>0.749</td><td>+14.52</td><td>—</td><td><span class="tag tag-warn">migration</span></td></tr>
<tr><td>norm600</td><td class="warn">7/150</td><td>—</td><td>—</td><td>—</td><td>0.847</td><td>+6.77</td><td>—</td><td><span class="tag tag-warn">partial</span></td></tr>
<tr><td>norm1800</td><td class="bad">0/150</td><td class="bad">9.85</td><td class="bad">-4.92</td><td>—</td><td>0.886</td><td>+5.12</td><td>—</td><td><span class="tag tag-fail">ranking fail</span></td></tr>
<tr><td>H2-v2 (1500 steps)</td><td class="bad">0/150</td><td class="bad">10.10</td><td class="bad">-4.28</td><td class="bad">5.53</td><td class="good">0.995</td><td class="good">+3.53</td><td>0.762</td><td><span class="tag tag-fail">ranking fail</span></td></tr>
<tr><td>M1-reset (0I pilot)</td><td colspan="7" class="neutral">pending eval</td><td><span class="tag tag-na">TBD</span></td></tr>
<tr><td>M1-persist (0I pilot)</td><td colspan="7" class="neutral">pending eval</td><td><span class="tag tag-na">TBD</span></td></tr>
<tr><td>N1800-reset (0I pilot)</td><td colspan="7" class="neutral">training in progress</td><td><span class="tag tag-na">TBD</span></td></tr>
<tr><td>N1800-persist (0I pilot)</td><td colspan="7" class="neutral">queued</td><td><span class="tag tag-na">TBD</span></td></tr>
</table>

<div class="insight">
<strong>关键发现：</strong>从 M0 → M1 → norm600 → norm1800 → H2-v2，logit cosine 单调上升（0.749 → 0.995），
但 candidate accuracy 从 15 降到 0，mean rank 从 ~4 恶化到 ~10。
这构成<strong>teacher imitation - candidate ranking 错位</strong>的完整证据链。
</div>

<h2>3. Teacher Imitation vs Candidate Ranking 错位 (H2-v2 1500-step Monitor)</h2>
<p>以下展示 H2-v2 训练 1500 步中，teacher fidelity 指标与 candidate ranking 指标的<strong>反向演化</strong>。</p>

<div class="chart-box">
<svg viewBox="0 0 800 400" style="width:100%;height:auto">
'''

# Chart 1: H2-v2 monitor dual axis
chart1_steps = mon_steps
# Normalize acc to 0-1 scale, rank to 0-14 scale
# Plot: acc (left, 0-20), rank (right, 0-14), margin (right2, -8 to 2)
steps_str = ','.join([str(s) for s in chart1_steps])
max_step = max(chart1_steps)

def svg_line(data, color, max_y, x_offset=60, y_offset=40, w=700, h=280, n_points=None):
    if n_points is None:
        n_points = len(data)
    pts = []
    for i, v in enumerate(data):
        x = x_offset + (i / max(n_points - 1, 1)) * w
        y = y_offset + h - (max(v / max_y, 0) if max_y > 0 else 0) * h
        y = max(y_offset, min(y_offset + h, y))
        pts.append('%.1f,%.1f' % (x, y))
    return '<polyline fill="none" stroke="%s" stroke-width="2.5" points="%s"/>' % (color, ' '.join(pts))

html += '<text x="400" y="25" text-anchor="middle" fill="#c9d1d9" font-size="14">H2-v2 Monitor: Candidate Rank ↑ while Logit Cosine ↑ (anti-correlation)</text>\n'
html += '<line x1="60" y1="40" x2="60" y2="320" stroke="#30363d"/>\n'
html += '<line x1="760" y1="40" x2="760" y2="320" stroke="#30363d"/>\n'
html += '<line x1="60" y1="320" x2="760" y2="320" stroke="#30363d"/>\n'

# rank (scale 0-14)
html += svg_line(mon_rank, '#f85149', 14.0)
# margin (scale -8 to 2 -> shift)
margin_scaled = [(m + 8) / 10.0 for m in mon_margin]
html += svg_line(margin_scaled, '#d29922', 1.0)
# logit cosine (scale 0.85-1.0 -> normalize)
cos_scaled = [(c - 0.85) / 0.15 for c in mon_cos]
html += svg_line(cos_scaled, '#3fb950', 1.0)
# delta NLL (scale 0-8)
dnll_scaled = [d / 8.0 for d in mon_dnll]
html += svg_line(dnll_scaled, '#58a6ff', 1.0)
# hidden cos (scale 0-1)
html += svg_line(mon_hcos, '#bc8cff', 1.0)

# legend
html += '<rect x="80" y="335" width="12" height="3" fill="#f85149"/><text x="98" y="338" fill="#c9d1d9" font-size="11">Mean Rank</text>\n'
html += '<rect x="180" y="335" width="12" height="3" fill="#d29922"/><text x="198" y="338" fill="#c9d1d9" font-size="11">Margin</text>\n'
html += '<rect x="260" y="335" width="12" height="3" fill="#3fb950"/><text x="278" y="338" fill="#c9d1d9" font-size="11">Logit Cos</text>\n'
html += '<rect x="360" y="335" width="12" height="3" fill="#58a6ff"/><text x="378" y="338" fill="#c9d1d9" font-size="11">ΔNLL</text>\n'
html += '<rect x="430" y="335" width="12" height="3" fill="#bc8cff"/><text x="448" y="338" fill="#c9d1d9" font-size="11">Hidden Cos</text>\n'

# x-axis labels
for i, s in enumerate(chart1_steps):
    x = 60 + (i / max(len(chart1_steps) - 1, 1)) * 700
    html += '<text x="%.0f" y="315" text-anchor="middle" fill="#8b949e" font-size="10">%d</text>\n' % (x, s)

html += '</svg>\n</div>\n'

# Chart 2: Training loss comparison
html += '<h2>4. 0I Pilot Training Loss: M1-reset vs M1-persist vs H2-v2</h2>\n'
html += '<div class="chart-box">\n<svg viewBox="0 0 800 400" style="width:100%;height:auto">\n'
html += '<text x="400" y="25" text-anchor="middle" fill="#c9d1d9" font-size="14">Smoothed Training Loss (window=20)</text>\n'
html += '<line x1="60" y1="40" x2="60" y2="320" stroke="#30363d"/>\n'
html += '<line x1="60" y1="320" x2="760" y2="320" stroke="#30363d"/>\n'

# Combine: m1r_smooth (600), m1p_smooth (600), h2_smooth (1500)
# Normalize x to 0-700, y to 0-12 (max loss ~12)
max_loss = max(max(m1r_smooth), max(m1p_smooth), max(h2_smooth))

def svg_line_norm(data, color, max_val, x_offset=60, y_off=40, w=700, h=280):
    pts = []
    for i, v in enumerate(data):
        x = x_offset + (i / max(len(data) - 1, 1)) * w
        y = y_off + h - (v / max_val) * h
        y = max(y_off, min(y_off + h, y))
        pts.append('%.1f,%.1f' % (x, y))
    return '<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (color, ' '.join(pts))

html += svg_line_norm(m1r_smooth, '#58a6ff', max_loss)
html += svg_line_norm(m1p_smooth, '#3fb950', max_loss)
html += svg_line_norm(h2_smooth, '#f85149', max_loss)

html += '<rect x="80" y="335" width="12" height="3" fill="#58a6ff"/><text x="98" y="338" fill="#c9d1d9" font-size="11">M1-reset (task CE)</text>\n'
html += '<rect x="220" y="335" width="12" height="3" fill="#3fb950"/><text x="238" y="338" fill="#c9d1d9" font-size="11">M1-persist (task CE)</text>\n'
html += '<rect x="370" y="335" width="12" height="3" fill="#f85149"/><text x="388" y="338" fill="#c9d1d9" font-size="11">H2-v2 (KL+block+rank)</text>\n'

html += '<text x="40" y="45" fill="#8b949e" font-size="10">%.1f</text>\n' % max_loss
html += '<text x="40" y="324" fill="#8b949e" font-size="10">0</text>\n'
html += '<text x="765" y="315" fill="#8b949e" font-size="10">step</text>\n'
html += '</svg>\n</div>\n'

# Chart 3: Bar chart of candidate accuracy across models
html += '<h2>5. Candidate Accuracy Cross-Model Comparison</h2>\n'
html += '<div class="chart-box">\n<svg viewBox="0 0 800 350" style="width:100%;height:auto">\n'
html += '<text x="400" y="25" text-anchor="middle" fill="#c9d1d9" font-size="14">Candidate Accuracy on P0H5 N=150</text>\n'

models = [
    ('M0\\nteacher', 11, '#d29922'),
    ('M1\\nmigration', 15, '#58a6ff'),
    ('norm600', 7, '#79c0ff'),
    ('norm1800', 0, '#f85149'),
    ('H2-v2', 0, '#f85149'),
]
n_bars = len(models)
bar_w = 100
gap = 30
start_x = 80
max_acc = 20

for i, (name, acc, color) in enumerate(models):
    x = start_x + i * (bar_w + gap)
    h = (acc / max_acc) * 250
    html += '<rect x="%d" y="%d" width="%d" height="%d" fill="%s" rx="4"/>\n' % (
        x, 300 - h, bar_w, h, color)
    html += '<text x="%d" y="%d" text-anchor="middle" fill="%s" font-size="14" font-weight="bold">%d</text>\n' % (
        x + bar_w // 2, 295 - h, color, acc)
    html += '<text x="%d" y="320" text-anchor="middle" fill="#c9d1d9" font-size="11">%s</text>\n' % (
        x + bar_w // 2, name.replace('\\n', ' '))

html += '<line x1="60" y1="300" x2="760" y2="300" stroke="#30363d"/>\n'
html += '<line x1="60" y1="50" x2="60" y2="300" stroke="#30363d"/>\n'
html += '<text x="65" y="55" fill="#8b949e" font-size="10">%d</text>\n' % max_acc
html += '<text x="65" y="305" fill="#8b949e" font-size="10">0</text>\n'
html += '</svg>\n</div>\n'

# Chart 4: Radar-style logit cosine vs accuracy scatter
html += '<h2>6. Logit Fidelity vs Candidate Accuracy: The Inversion</h2>\n'
html += '<div class="chart-box">\n<svg viewBox="0 0 800 400" style="width:100%;height:auto">\n'
html += '<text x="400" y="25" text-anchor="middle" fill="#c9d1d9" font-size="14">Logit Cosine (→) vs Candidate Accuracy (↑): Systematic Inversion</text>\n'

scatter = [
    ('M0 teacher', 1.0, 11, '#d29922', 12),
    ('M1', 0.749, 15, '#58a6ff', 12),
    ('norm600', 0.847, 7, '#79c0ff', 12),
    ('norm1800', 0.886, 0, '#f85149', 12),
    ('H2-v2', 0.995, 0, '#f85149', 14),
]

# axes
html += '<line x1="80" y1="350" x2="760" y2="350" stroke="#30363d"/>\n'  # x-axis (cosine 0.7-1.0)
html += '<line x1="80" y1="50" x2="80" y2="350" stroke="#30363d"/>\n'    # y-axis (acc 0-20)
html += '<text x="420" y="380" text-anchor="middle" fill="#8b949e" font-size="12">Logit Cosine →</text>\n'
html += '<text x="30" y="200" text-anchor="middle" fill="#8b949e" font-size="12" transform="rotate(-90 30 200)">Candidate Acc ↑</text>\n'

for name, cos, acc, color, r in scatter:
    x = 80 + ((cos - 0.7) / 0.3) * 680
    y = 350 - (acc / 20.0) * 300
    html += '<circle cx="%.1f" cy="%.1f" r="%d" fill="%s" opacity="0.8"/>\n' % (x, y, r, color)
    html += '<text x="%.1f" y="%.1f" fill="%s" font-size="11" font-weight="bold">%s</text>\n' % (x + 15, y, color, name)

# trend arrow
html += '<line x1="200" y1="120" x2="600" y2="280" stroke="#f85149" stroke-width="2" stroke-dasharray="5,5" opacity="0.6"/>\n'
html += '<text x="400" y="210" fill="#f85149" font-size="12" font-weight="bold" opacity="0.8">↘ Inverse Trend</text>\n'

html += '</svg>\n</div>\n'

# H2-v2 monitor data table
html += '<h2>7. H2-v2 Monitor History (1500 steps)</h2>\n'
html += '<table>\n<tr><th>Step</th><th>Cand Acc</th><th>Mean Rank</th><th>Margin</th><th>Logit Cos</th><th>ΔNLL</th><th>Hidden Cos</th></tr>\n'
for m in h2_mon:
    html += '<tr><td>%d</td><td class="%s">%d/150</td><td>%.2f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>\n' % (
        m['step'],
        'bad' if m['cand_acc'] == 0 else 'warn',
        m['cand_acc'],
        m['mean_rank'], m['mean_margin'],
        m['logit_cosine'], m['delta_nll'], m['hidden_cos'])
html += '</table>\n'

# Analysis section
html += '''
<h2>8. 推测与结论</h2>

<h3>8.1 Teacher Imitation - Candidate Ranking 错位</h3>
<p>数据清楚地显示了一个<strong>反直觉的系统性模式</strong>：蒸馏过程不断提高 teacher logit fidelity（cosine 从 0.749 升至 0.995），
但 candidate ranking 系统性恶化（accuracy 从 15/150 降至 0/150，mean rank 从 ~4 恶化到 ~10）。</p>

<p>可能的解释：</p>
<ul style="margin-left:24px;margin-bottom:16px;">
<li><strong>Retention/Attention 功能差异本质：</strong>Retention 的衰减递归在数学上不等价于 softmax attention。即使 logit 分布相似，<em>隐含的 information routing 路径</em>不同。当 teacher imitation 压迫 retention 的隐式状态表示向 attention 的 KV cache 靠拢时，它可能破坏 retention 原始（但 differentiable）的跨 token 信息整合方式。</li>
<li><strong>Post-norm 过拟合 teacher：</strong>norm1800 → H2-v2 的路径显示，post-norm 逐渐学习补偿 retention 与 attention 之间的表示差异，使<em>表面</em> logit 趋同，但内部表示的 geometric structure 可能已经偏离了能正确 rank candidate 的方向。</li>
<li><strong>Candidate scoring 是 non-linear selection：</strong>argmax/mean-log-prob 对 logit 空间的微小差异极度敏感。KL divergence 对<em>整体分布</em>施加压力，但 candidate ranking 取决于<em>正确候选 vs 最佳错误候选</em>的 margin——这是一个局部、非线性的指标，KL 无法直接优化。</li>
</ul>

<h3>8.2 0I Pilot 初步分析（M1 两组已完成）</h3>
<p>M1-reset 和 M1-persist 均使用<strong>真实 CLUTRR task objective</strong>（candidate CE），不依赖 teacher cache。
训练 loss 均从 ~10 下降到 ~2.5，说明 task objective 可以有效优化。</p>

<table>
<tr><th>条件</th><th>Steps</th><th>Final Loss</th><th>Init</th><th>可训练参数</th></tr>
<tr><td>M1-reset</td><td>600</td><td>2.459</td><td>M1 migration</td><td>24576 (post-norm)</td></tr>
<tr><td>M1-persist</td><td>600</td><td>2.616</td><td>M1 migration</td><td>24576 (post-norm)</td></tr>
</table>

<p>persist 模式 loss 略高于 reset（2.616 vs 2.459），但这不代表 persist 更差——persist 模式需要跨 turn 积累状态，在 600 步内可能尚未充分学习如何利用 persistent state。
最终判定需等待 canonical P0H5 evaluation。</p>

<h3>8.3 核心结论</h3>
<div class="note">
<ol style="margin-left:20px;">
<li><strong>H2-v2 判定为 negative result，正式关闭。</strong>rank_loss 与 candidate-scoring scoring 函数完全一致，排除了 implementation mismatch。1500 步训练后 teacher fidelity 极高（cos=0.995）但 candidate accuracy 仍为 0。</li>
<li><strong>Teacher imitation objective 是导致 candidate ranking 失败的直接原因。</strong>这不是 capacity 不足（393K LoRA params），而是 objective 方向错误：KL 压迫 retention 的内部表示向 attention 靠拢，破坏了 candidate ranking 所需的局部 margin 结构。</li>
<li><strong>0I pilot 改用 task objective 是正确的方向转折。</strong>不依赖 teacher cache、不用 KL、不用 block loss——直接优化 candidate CE。M1 两组 600 步训练 loss 正常收敛。</li>
<li><strong>0I pilot 的 N1800 两组仍在进行中</strong>，暂时无法给出 persist vs reset 的最终比较。</li>
</ol>
</div>

<h2>9. 思路方向建议</h2>

<h3>9.1 短期（0I pilot 完成后）</h3>
<ul style="margin-left:24px;margin-bottom:16px;">
<li><strong>完成 4 组 0I pilot 评估</strong>：N1800-reset 和 N1800-persist 完成后，在同一 P0H5 N=150 上统一评估，比较 Acc(persist) - Acc(reset)。</li>
<li><strong>如果 persist > reset</strong>：说明 cross-turn retention state persistence 有 causal task effect。此时扩多 seed 验证，并进入 0J 做更细致的 state lifetime 消融。</li>
<li><strong>如果 persist ≈ reset</strong>：需要检查 state 是否真持久、真变化、真被读取。可能需要更长的训练（600 步可能不够让模型学会利用 persistent state）。</li>
<li><strong>如果 persist < reset</strong>：persistent state 可能引入 interference / stale state 问题。保留结果，分析 state norm / delta / cosine。</li>
</ul>

<h3>9.2 中期方向</h3>
<ul style="margin-left:24px;margin-bottom:16px;">
<li><strong>放弃 teacher imitation 路线</strong>：0H/H2 的全部数据证明这条路在 candidate ranking 上是 dead end。task objective（candidate CE / hinge）是唯一能直接优化 candidate margin 的方式。</li>
<li><strong>增加 training scale</strong>：600 步可能不足以让 post-norm 适配 retention 的 differentiability。考虑 1500-3000 步 task training。</li>
<li><strong>考虑 unfreeze gamma</strong>：当前 gamma 冻结。如果 retention 的 decay rate 可以适配 task，cross-turn state 的 information lifetime 可能更合理。但 0H 证明 gamma-only 几乎无修复能力，所以只在 task objective 下尝试。</li>
<li><strong>探索 partial unfreeze</strong>：除了 post-norm，考虑 unfreeze Q/K/V/O 的 LoRA（低 rank），但用 task objective 而非 KL。这比 H2-v2 的关键区别是 objective：task CE vs teacher KL。</li>
</ul>

<h3>9.3 长期方向</h3>
<ul style="margin-left:24px;margin-bottom:16px;">
<li><strong>Retention-specific task architecture</strong>：如果 persist 有 positive effect，考虑设计 retention-native 的 multi-turn reasoning 架构，而非在 attention model 上做 surgical replacement。</li>
<li><strong>State routing / gating</strong>：让模型学习<em>何时</em>使用 persistent state vs 何时 reset。可以加一个 gate（类似 LSTM 的 forget gate）在 turn 边界。</li>
<li><strong>Multi-seed statistical validation</strong>：seed 42 只是 pilot。如果效应存在，需要 5+ seeds 做 bootstrap CI。</li>
</ul>

<h2>10. 实验基础设施修复记录</h2>
<table>
<tr><th>问题</th><th>根因</th><th>修复</th></tr>
<tr><td>H2-v1 随机性不可复现</td><td>未设 random.seed(42)</td><td>统一 seed protocol (random/numpy/torch/CUDA)</td></tr>
<tr><td>H2-v1 数据耗尽</td><td>cache 仅 348 step 数据</td><td>teacher_cache_v2 扩大到 600 traj / 1254 turns</td></tr>
<tr><td>H2-v2 stdout 崩溃 (OSError 22)</td><td>Windows 后台 stdout pipe 失效</td><td>SafeStdout 包装 + 文件镜像</td></tr>
<tr><td>0I persist backward 崩溃</td><td>跨 turn state 携带 autograd graph</td><td>turn 边界 .detach()</td></tr>
<tr><td>0I persist OOM</td><td>14-candidate batch 在 8GB GPU 超限</td><td>chunk_size=2 分批 + empty_cache</td></tr>
<tr><td>0I persist stdout 崩溃</td><td>同 OSError 22</td><td>同样 SafeStdout 修复</td></tr>
</table>

<div class="footer">
<p>Generated by Hermes Agent | Canonical root: D:\\training | Model: Ouro-1.4B (4-bit NF4) | GPU: RTX 4060 8GB</p>
<p>Phase 0H: distillation (teacher imitation) — closed as negative result<br>
Phase 0I: task-objective pilot (candidate CE) — M1 groups complete, N1800 groups in progress</p>
</div>

</div>
</body>
</html>
'''

out = base / 'P0I_PILOT_RESULTS.html'
out.write_text(html, encoding='utf-8')
print('wrote %s (%d bytes)' % (out, out.stat().st_size))
