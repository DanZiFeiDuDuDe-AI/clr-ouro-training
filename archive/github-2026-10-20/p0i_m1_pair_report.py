# -*- coding: utf-8 -*-
"""P0I M1 pair final report: reset vs persist (post-fix).

Reads:
  checkpoints/p0i_pilot_m1_reset_s42/train_log.json
  checkpoints/p0i_pilot_m1_persist_s42/train_log.json
  P0I_PILOT_EVAL.json   (from p0i_pilot_eval.py)

Writes: D:/training/output/P0I_M1_PAIR_RESULTS.md
"""
import os, json, statistics, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "D:/training/checkpoints"
OUT = "D:/training/output/P0I_M1_PAIR_RESULTS.md"

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def sma(xs, w=50):
    if not xs:
        return []
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out

def main():
    rl = load(os.path.join(BASE, "p0i_pilot_m1_reset_s42", "train_log.json"))
    pl = load(os.path.join(BASE, "p0i_pilot_m1_persist_s42", "train_log.json"))
    ev = load("D:/training/P0I_PILOT_EVAL.json")

    def tail_avg(xs, k=50):
        return statistics.mean(xs[-k:]) if len(xs) >= k else statistics.mean(xs)

    lines = []
    A = lines.append
    A("# P0I M1 Pair Results (reset vs persist)")
    A("")
    A("生成时间: 2026-08-26")
    A("")
    A("## 实验设置")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 初始化 | M1 (ouro_retention_m1, migration-only) |")
    A("| 训练目标 | 候选 CE: L = CE(candidate_scores, correct_index) |")
    A("| 优化器 / LR | Adam / 3e-4 |")
    A("| 步数 | 600 (P0I_STEPS=600) |")
    A("| Seed | 42 (torch+random+np, generate_dataset 协议) |")
    A("| Dataset | 同一 generate_dataset(600) 序列, 同一顺序 |")
    A("| State lifetime | reset: turn 边界清零; persist: turn 边界注入上一 turn end state (detached) |")
    A("| Position | reset: 每 turn offset=0; persist: 连续 offset (累计历史长度) |")
    A("| 可训练参数 | post-norm 权重 24576 参数 |")
    A("")
    A("## 1. 训练 loss 曲线")
    A("")
    A("| 条件 | steps | 首 loss | 末 loss (mean-50) | 最低 loss |")
    A("|---|---|---|---|---|")
    A("| M1-reset | %d | %.4f | %.4f | %.4f |" % (
        rl["steps"], rl["losses"][0], tail_avg(rl["losses"]), min(rl["losses"])))
    A("| M1-persist | %d | %.4f | %.4f | %.4f |" % (
        pl["steps"], pl["losses"][0], tail_avg(pl["losses"]), min(pl["losses"])))
    A("")
    A("重置条件收敛: loss 从 ~%.2f 波动下降至末 50 步均值 %.3f%s。" % (
        rl["losses"][0], tail_avg(rl["losses"]),
        "（有效训练）" if tail_avg(rl["losses"]) < rl["losses"][0] * 0.7 else "（收敛存疑）"))
    A("")
    A("## 2. Held-out 评估对比 (N=150, seed42 协议)")
    A("")
    cond = ev["conditions"]
    c_r = cond.get("m1-reset") or {}
    c_p = cond.get("m1-persist") or {}
    if c_r.get("missing") or c_p.get("missing"):
        A("> 评估缺失: reset=%s persist=%s" % (c_r.get("missing"), c_p.get("missing")))
    else:
        def fmt(c, k, nd=4):
            v = c.get(k)
            return "-" if v is None else ("%.4f" % v if isinstance(v, float) else str(v))
        A("| 指标 | M1-reset | M1-persist | Δ (persist − reset) |")
        A("|---|---|---|---|")
        keys = [("candidate_accuracy", "candidate accuracy (hit@1)"),
                ("mean_rank", "correct mean rank"),
                ("mean_margin", "correct-vs-best-wrong margin"),
                ("entropy", "predictive entropy")]
        for k, label in keys:
            vr = c_r.get(k); vp = c_p.get(k)
            d = "-"
            if vr is not None and vp is not None:
                d = ("%+.4f" % (vp - vr)) if isinstance(vr, float) else ("%+d" % (vp - vr))
            A("| %s | %s | %s | %s |" % (label, fmt(c_r, k), fmt(c_p, k), d))
        A("")
        A("### Δ_internal = Perf(M1-persist) − Perf(M1-reset)")
        A("")
        A("| 维度 | Δ | 方向 |")
        A("|---|---|---|")
        d_acc = c_p.get("candidate_accuracy", 0) - c_r.get("candidate_accuracy", 0)
        d_rank = c_p.get("mean_rank", 0) - c_r.get("mean_rank", 0)
        d_margin = c_p.get("mean_margin", 0) - c_r.get("mean_margin", 0)
        A("| candidate accuracy | %+d / %d | %s |" % (
            d_acc, c_p.get("n", 150), "persist 优" if d_acc > 0 else ("reset 优" if d_acc < 0 else "持平")))
        A("| mean rank | %+.4f | %s |" % (
            d_rank, "persist 优" if d_rank < 0 else ("reset 优" if d_rank > 0 else "持平")))
        A("| margin | %+.4f | %s |" % (
            d_margin, "persist 优" if d_margin > 0 else ("reset 优" if d_margin < 0 else "持平")))
        A("")
        A("## 3. State 统计 (persist, 跨 turn)")
        A("")
        A("| 层 | state norm | state delta | cross-turn cosine |")
        A("|---|---|---|---|")
        sn = c_p.get("state_norm_mean", {}) or {}
        sd = c_p.get("state_delta_mean", {}) or {}
        cs = c_p.get("cross_turn_cosine_mean", {}) or {}
        for li in ["3", "7", "11", "15", "19", "23"]:
            A("| %s | %.4f | %.4f | %.4f |" % (
                li, sn.get(li, 0), sd.get(li, 0), cs.get(li, 0)))
        A("")
    A("## 4. 结论")
    A("")
    # summary text
    if c_r.get("missing") or c_p.get("missing"):
        A("评估未完整，结论待定。")
    else:
        d_acc = c_p.get("candidate_accuracy", 0) - c_r.get("candidate_accuracy", 0)
        d_rank = c_p.get("mean_rank", 0) - c_r.get("mean_rank", 0)
        d_margin = c_p.get("mean_margin", 0) - c_r.get("mean_margin", 0)
        verdict = []
        if d_acc > 0: verdict.append("candidate acc 提升 %+d" % d_acc)
        if d_rank < 0: verdict.append("mean rank 下降 %+.2f" % d_rank)
        if d_margin > 0: verdict.append("margin 提升 %+.4f" % d_margin)
        if not verdict:
            verdict.append("四项指标未见一致优势，persist 未带来可测提升")
        A("M1-persist 修复后完成 600 步训练；相对 M1-reset：" + "，".join(verdict) + "。")
        A("")
        A("> 修复内容: turn-boundary 递归 detach state (无 retain_graph)；candidate position_ids 连续 offset 修复 (reset=0 / persist=累计长度)；eval 端 position_ids (n,L) 对齐与真实 cross-turn cosine。")
    os.makedirs("D:/training/output", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("\n".join(lines[:60]))

if __name__ == "__main__":
    main()
