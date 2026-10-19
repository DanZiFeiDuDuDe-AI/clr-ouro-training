# -*- coding: utf-8 -*-
"""
P0H5 candidate diagnostics (网友 spec):
  Same 150 trajectories, add M0 teacher candidate score (missing in screenshot).
  For M0 / M1 / norm600 / norm1800, per-question output:
    - correct candidate rank            (1..14)
    - correct-vs-best-wrong margin      (score[correct] - max score[wrong])
    - candidate-set entropy             (softmax over 14 candidate scores)
    - Spearman correlation vs M0        (student candidate scores vs teacher)
  Goal: classify WHY norm1800 (d3) is 0/150:
    (a) correct answer consistently ranks 2nd/3rd
    (b) candidate distribution collapsed (entropy explosion)
    (c) systematic scorer / calibration shift

Output: P0H5_CANDIDATE_DIAGNOSTICS.md ; stop and report (do NOT auto-enter 0I).
"""
import os, sys, json, time, math, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset, RELATIONS

N_TRAJ = int(os.environ.get("H_N_TRAJ", "150"))
MAX_TOKENS = 128
REPLACED = [3, 7, 11, 15, 19, 23]

CKPTS = [
    ("M0_teacher", None, None),
    ("M1", "D:/training/checkpoints/ouro_retention_m1/best_model.pt", "sigmoid_logit"),
    ("norm600", "D:/training/checkpoints/ouro_retention_m2_d2/best_model.pt", "sigmoid_logit"),
    ("norm1800", "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt", "sigmoid_logit"),
]


def load_model(tok, ckpt_path, gamma_mode):
    model, _ = load_ouro()
    if ckpt_path is None:
        return model, []
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode=gamma_mode)
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    adapters = []
    for ret, idx in converted:
        layer = model.model.layers[idx]
        src_attn = layer.self_attn.src_attn
        from retention_ouro import _RetentionAttentionAdapter
        adapter = _RetentionAttentionAdapter(ret, src_attn)
        layer.self_attn = adapter
        adapters.append(adapter)
        ret.gamma_logit.data.copy_(ck["retention_gamma"][str(idx)])
    pnw = ck.get("post_norm_weights")
    restored = 0
    if pnw:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
                restored += 1
    return model, adapters


@torch.no_grad()
def candidate_scores(model, tok, prompt_ids, cand_ids, pad_id):
    """Return per-candidate scores (mean logprob over answer tokens)."""
    n = len(cand_ids)
    P = len(prompt_ids)
    L = P + max(len(c) for c in cand_ids)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, c in enumerate(cand_ids):
        seq = prompt_ids + c
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
    input_ids, attn = input_ids.cuda(), attn.cuda()
    base_out, _, _ = model.model(input_ids=input_ids, attention_mask=attn,
                                 use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(base_out.last_hidden_state).float(), dim=-1)
    scores = []
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        scores.append(lp)
    return scores


def spearman(a, b):
    """Spearman rank correlation, hand-rolled (no scipy dependency)."""
    def ranks(x):
        idx = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(idx):
            r[i] = pos + 1
        return r
    n = len(a)
    ra, rb = ranks(a), ranks(b)
    ma, mb = (n + 1) / 2.0, (n + 1) / 2.0
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def set_entropy(scores):
    """Entropy of softmax over 14 candidate scores (nats)."""
    mx = max(scores)
    z = math.fsum(math.exp(s - mx) for s in scores)
    p = [math.exp(s - mx) / z for s in scores]
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def main():
    print("=== P0H5 candidate diagnostics (N=%d) ===" % N_TRAJ, flush=True)
    torch.manual_seed(42)
    random.seed(42)
    _train = generate_dataset(600)
    eval_data = generate_dataset(N_TRAJ)[:N_TRAJ]
    print("eval trajectories: %d" % len(eval_data), flush=True)

    _, tok = load_ouro()
    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id

    teacher_scores_cache = {}  # prompt -> teacher candidate scores
    rows = {tag: [] for tag, _, _ in CKPTS}

    t_all = time.time()
    for mi, (tag, ckpt, gm) in enumerate(CKPTS):
        print("\n--- %s ---" % tag, flush=True)
        t0 = time.time()
        model, _ = load_model(tok, ckpt, gm)
        model.eval()
        print("loaded %.0fs" % (time.time() - t0), flush=True)
        tc = 0
        for ti, traj in enumerate(eval_data):
            turn = traj["turns"][0]
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            scores = candidate_scores(model, tok, prompt_ids, cand_ids, pad_id)
            if tag == "M0_teacher":
                teacher_scores_cache[ti] = scores
                ref_scores = scores
            else:
                ref_scores = teacher_scores_cache[ti]
            correct_i = RELATIONS.index(turn["answer"])
            # rank of correct candidate (1=best)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            rank = order.index(correct_i) + 1
            wrong_scores = [s for i, s in enumerate(scores) if i != correct_i]
            margin = scores[correct_i] - max(wrong_scores)
            ent = set_entropy(scores)
            sp = spearman(scores, ref_scores)
            if rank == 1:
                tc += 1
            rows[tag].append({"traj": ti, "rank": rank, "margin": margin,
                              "entropy": ent, "spearman": sp,
                              "correct": RELATIONS[correct_i],
                              "pred": RELATIONS[order[0]]})
            if (ti + 1) % 50 == 0:
                print("  %d/%d" % (ti + 1, N_TRAJ), flush=True)
        print("acc=%d/%d" % (tc, N_TRAJ), flush=True)
        del model
        torch.cuda.empty_cache()

    # summary
    def summarize(tag):
        rs = rows[tag]
        ranks = [r["rank"] for r in rs]
        margins = [r["margin"] for r in rs]
        ents = [r["entropy"] for r in rs]
        sps = [r["spearman"] for r in rs]
        rank_dist = {k: ranks.count(k) for k in (1, 2, 3)}
        rank_dist["gt3"] = sum(1 for x in ranks if x > 3)
        return {
            "n": len(rs), "acc": sum(1 for x in ranks if x == 1),
            "rank_dist": rank_dist,
            "mean_rank": sum(ranks) / len(ranks),
            "mean_margin": sum(margins) / len(margins),
            "mean_entropy": sum(ents) / len(ents),
            "mean_spearman": (sum(sps) / len(sps)) if sps else None,
        }

    print("\n=== SUMMARY ===", flush=True)
    for tag, _, _ in CKPTS:
        s = summarize(tag)
        print(tag, s, flush=True)

    # interpretation
    d3 = summarize("norm1800")
    d3_ranks = [r["rank"] for r in rows["norm1800"]]
    n2 = sum(1 for x in d3_ranks if x == 2)
    n3 = sum(1 for x in d3_ranks if x == 3)
    ngt = sum(1 for x in d3_ranks if x > 3)
    m0_ent = summarize("M0_teacher")["mean_entropy"]
    d3_ent = d3["mean_entropy"]
    verdict = []
    if n2 + n3 >= 60:
        verdict.append("主要死法: 正确候选普遍掉到第2/3名 (rank 2+3 = %d/150)，接近但不够 — '差一点点'型" % (n2 + n3))
    if d3_ent > m0_ent * 1.25:
        verdict.append("候选分布明显更混乱: d3 entropy %.3f vs M0 %.3f — 存在 '整体崩' 成分" % (d3_ent, m0_ent))
    if d3["mean_spearman"] is not None and d3["mean_spearman"] < 0.5:
        verdict.append("候选打分顺序与 M0 相关性低: Spearman %.3f — 存在 'scorer 系统偏移' 成分" % d3["mean_spearman"])
    if not verdict:
        verdict.append("未落入明显单类死法，需结合 rank 分布细看")

    with open("D:/training/P0H5_CANDIDATE_DIAGNOSTICS.md", "w", encoding="utf-8") as f:
        f.write("# P0H5_CANDIDATE_DIAGNOSTICS.md\n\n")
        f.write("Date: 2026-08-25\nN=%d (same draw: seed 42, after 600 replay)\n\n" % N_TRAJ)
        f.write("## Per-model summary\n\n")
        f.write("| Model | acc | rank=1 | rank=2 | rank=3 | rank>3 | mean rank | mean margin | mean entropy | mean Spearman(vs M0) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for tag, _, _ in CKPTS:
            s = summarize(tag)
            f.write("| %s | %d/%d | %d | %d | %d | %d | %.2f | %+.4f | %.3f | %s |\n" % (
                tag, s["acc"], s["n"], s["rank_dist"].get(1, 0), s["rank_dist"].get(2, 0),
                s["rank_dist"].get(3, 0), s["rank_dist"].get("gt3", 0), s["mean_rank"],
                s["mean_margin"], s["mean_entropy"],
                ("%.3f" % s["mean_spearman"]) if s["mean_spearman"] is not None else "-"))
        f.write("\n## Per-question detail (first 20 per model)\n\n")
        f.write("| Model | traj | correct | pred | rank | margin | entropy | spearman |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for tag, _, _ in CKPTS:
            for r in rows[tag][:20]:
                f.write("| %s | %d | %s | %s | %d | %+.4f | %.3f | %.3f |\n" % (
                    tag, r["traj"], r["correct"], r["pred"], r["rank"],
                    r["margin"], r["entropy"], r["spearman"]))
        f.write("\n## Interpretation\n\n")
        f.write("norm1800 (d3): rank2+3=%d, rank>3=%d, entropy=%.3f (M0=%.3f), Spearman=%.3f\n\n" % (
            n2 + n3, ngt, d3_ent, m0_ent,
            d3["mean_spearman"] if d3["mean_spearman"] is not None else -1))
        for v in verdict:
            f.write("- %s\n" % v)
        f.write("\n## Verdict gate\n\n")
        if n2 + n3 >= 60 or d3_ent > m0_ent * 1.25 or (d3["mean_spearman"] is not None and d3["mean_spearman"] < 0.5):
            f.write("> Distillation deepens => candidate ranking degrades (rank>3=%d, mean rank=%.2f). "
                    "DO NOT freeze norm1800 as M2_CANONICAL for 0I. "
                    "Proceed to H2 rank-4 LoRA on Q/K/V/O of the 6 converted layers.\n" % (ngt, d3["mean_rank"]))
        else:
            f.write("> No systematic candidate-ranking collapse detected. norm1800 may be frozen as "
                    "M2_CANONICAL for 0I. (Confirm with 网友 before proceeding.)\n")
        f.write("\n## Raw rows (all %d)\n\n" % N_TRAJ)
        f.write("```json\n")
        f.write(json.dumps({tag: rows[tag] for tag, _, _ in CKPTS}, indent=1))
        f.write("\n```\n")

    print("\nwrote D:/training/P0H5_CANDIDATE_DIAGNOSTICS.md (%.0fs)" % (time.time() - t_all), flush=True)
    print("P0H5 DIAGNOSTICS OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
