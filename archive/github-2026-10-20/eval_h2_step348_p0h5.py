# -*- coding: utf-8 -*-
"""
Evaluate H2 best_model.pt (step ~348) on the EXACT P0H5 held-out set.
Protocol (identical to p0h5_candidate_diagnostics.py):
    torch.manual_seed(42); random.seed(42)
    _train = generate_dataset(600)          # replay RNG stream (parity)
    eval_data = generate_dataset(150)[:150] # held-out N=150
Metrics:
    candidate: acc / mean_rank / mean_margin / rank_dist / entropy / Spearman(vs M0)
    teacher fidelity (H2 monitor口径, LOGIT_N=20): logit_cosine / delta_nll
Also verifies:
    - held-out item IDs (md5 of prompt|answer) vs P0H5 raw rows (correct list)
    - evaluator hash (md5 of p0h5 script + this script)
    - side-by-side: seed42 600->150 vs seed42 600->300->150 (H2-monitor-style draw)
Output: D:/training/checkpoints/ouro_retention_m2_h2/eval_step348_p0h5_aligned.json
"""
import os, sys, json, time, math, random, hashlib, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset, RELATIONS

N_TRAJ = 150
MAX_TOKENS = 128
REPLACED = [3, 7, 11, 15, 19, 23]
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
H2_CKPT = "D:/training/checkpoints/ouro_retention_m2_h2/best_model.pt"
BASE_CKPT = "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt"
P0H5_MD = "D:/training/P0H5_CANDIDATE_DIAGNOSTICS.md"
P0H5_PY = "D:/training/p0h5_candidate_diagnostics.py"
OUT_JSON = "D:/training/checkpoints/ouro_retention_m2_h2/eval_step348_p0h5_aligned.json"
LOGIT_N = 20


class LoRALinear(torch.nn.Module):
    def __init__(self, orig, r, alpha):
        super().__init__()
        self.orig = orig
        self.r, self.alpha = r, alpha
        self.A = torch.nn.Parameter(torch.empty(orig.in_features, r, dtype=torch.bfloat16))
        self.B = torch.nn.Parameter(torch.zeros(r, orig.out_features, dtype=torch.bfloat16))
        self.orig.requires_grad_(False)

    def forward(self, x):
        return self.orig(x) + (x @ self.A @ self.B) * (self.alpha / self.r)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def load_h2(model, tok):
    """Load H2 LoRA ckpt on top of norm1800 base (mirrors p0h2_lora.py load path)."""
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="sigmoid_logit")
    for ret, idx in converted:
        layer = model.model.layers[idx]
        src_attn = layer.self_attn.src_attn
        from retention_ouro import _RetentionAttentionAdapter
        layer.self_attn = _RetentionAttentionAdapter(ret, src_attn)
    freeze_all(model)
    ck = torch.load(H2_CKPT, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        ret.gamma_logit.data.copy_(ck["retention_gamma"][str(idx)])
    pnw = ck.get("post_norm_weights")
    if pnw:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
    for ret, idx in converted:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            orig = getattr(ret, proj)
            lora = LoRALinear(orig, ck["lora"][str(idx)][proj]["r"],
                              ck["lora"][str(idx)][proj]["alpha"]).to(orig.weight.device)
            lora.A.data.copy_(ck["lora"][str(idx)][proj]["A"].to(lora.A.device).to(torch.bfloat16))
            lora.B.data.copy_(ck["lora"][str(idx)][proj]["B"].to(lora.B.device).to(torch.bfloat16))
            setattr(ret, proj, lora)
    return model, converted


@torch.no_grad()
def candidate_scores(model, tok, prompt_ids, cand_ids, pad_id):
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
    bo, _, _ = model.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(bo.last_hidden_state).float(), dim=-1)
    scores = []
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        scores.append(lp)
    return scores


def spearman(a, b):
    def ranks(x):
        idx = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(idx):
            r[i] = pos + 1
        return r
    n = len(a)
    ra, rb = ranks(a), ranks(b)
    ma = mb = (n + 1) / 2.0
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def set_entropy(scores):
    mx = max(scores)
    z = math.fsum(math.exp(s - mx) for s in scores)
    p = [math.exp(s - mx) / z for s in scores]
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


@torch.no_grad()
def fidelity(model, teacher_model, tok, eval_data, cand_ids, pad_id):
    cos_acc, nll_delta = 0.0, 0.0
    nn = min(LOGIT_N, len(eval_data))
    for ti, traj in enumerate(eval_data[:nn]):
        turn = traj["turns"][0]
        prompt_ids = tok(turn["prompt"], truncation=True, max_length=MAX_TOKENS)["input_ids"]
        ans_ids = tok(turn["answer"], add_special_tokens=False)["input_ids"]
        P = len(prompt_ids)
        inp = torch.tensor([prompt_ids]).cuda()
        attn = torch.ones_like(inp)
        bo_s, _, _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
        bo_t, _, _ = teacher_model.model(input_ids=inp, attention_mask=attn, use_cache=False)
        ls = model.lm_head(bo_s.last_hidden_state[:, P - 1]).float()[0]
        lt = teacher_model.lm_head(bo_t.last_hidden_state[:, P - 1]).float()[0]
        cos_acc += F.cosine_similarity(ls.unsqueeze(0), lt.unsqueeze(0)).item()
        nll_s = -F.log_softmax(ls, dim=-1)[ans_ids[0]].item()
        nll_t = -F.log_softmax(lt, dim=-1)[ans_ids[0]].item()
        nll_delta += (nll_s - nll_t)
    return cos_acc / nn, nll_delta / nn, nn


def md5file(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def item_id(turn):
    return hashlib.md5((turn["prompt"] + "|" + turn["answer"]).encode("utf-8")).hexdigest()


def main():
    print("=== eval H2 step348 on P0H5-aligned N=150 ===", flush=True)
    # ---- held-out draw, EXACT P0H5 protocol ----
    torch.manual_seed(42)
    random.seed(42)
    _train = generate_dataset(600)
    eval_data = generate_dataset(N_TRAJ)[:N_TRAJ]
    ids_p0h5 = [item_id(t["turns"][0]) for t in eval_data]
    corrects = [t["turns"][0]["answer"] for t in eval_data]
    print("eval trajectories: %d" % len(eval_data), flush=True)

    # ---- side-by-side: H2-monitor-style draw (600 -> 300 -> 150) ----
    torch.manual_seed(42)
    random.seed(42)
    _ = generate_dataset(600)
    _ = generate_dataset(300)[:300]
    eval_h2style = generate_dataset(N_TRAJ)[:N_TRAJ]
    ids_h2style = [item_id(t["turns"][0]) for t in eval_h2style]
    id_overlap = len(set(ids_p0h5) & set(ids_h2style))
    print("ID overlap P0H5-draw vs H2-style-draw: %d/150" % id_overlap, flush=True)

    # ---- verify vs P0H5 raw rows (correct list) ----
    match = None
    try:
        txt = open(P0H5_MD, encoding="utf-8").read()
        m = re.search(r"```json\n(\{.*?\})\n```", txt, re.S)
        raw = json.loads(m.group(1))
        m0_correct = [r["correct"] for r in raw["M0_teacher"]]
        match = sum(1 for a, b in zip(corrects, m0_correct) if a == b)
        print("correct-answer match vs P0H5 raw rows: %d/%d" % (match, len(m0_correct)), flush=True)
    except Exception as e:
        print("P0H5 md parse failed: %r" % e, flush=True)

    # ---- load models (independent instances; do NOT reuse teacher as base) ----
    print("loading teacher M0...", flush=True)
    teacher_model, tok = load_ouro()
    teacher_model.eval()
    print("loading H2 best_model.pt (separate instance)...", flush=True)
    model, _ = load_ouro()
    model, converted = load_h2(model, tok)
    model.eval()
    # sanity: must NOT be the same object
    assert model is not teacher_model, "model and teacher must be distinct instances"
    print("distinct instances OK", flush=True)

    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id

    # ---- candidate behavior on all 150 ----
    ranks, margins, ents, sps = [], [], [], []
    teacher_scores = {}
    t0 = time.time()
    for ti, traj in enumerate(eval_data):
        turn = traj["turns"][0]
        prompt_ids = tok(turn["prompt"], truncation=True, max_length=MAX_TOKENS)["input_ids"]
        s = candidate_scores(model, tok, prompt_ids, cand_ids, pad_id)
        ts = candidate_scores(teacher_model, tok, prompt_ids, cand_ids, pad_id)
        teacher_scores[ti] = ts
        correct_i = RELATIONS.index(turn["answer"])
        order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
        rank = order.index(correct_i) + 1
        wrong = [x for i, x in enumerate(s) if i != correct_i]
        ranks.append(rank)
        margins.append(s[correct_i] - max(wrong))
        ents.append(set_entropy(s))
        sps.append(spearman(s, ts))
        if (ti + 1) % 50 == 0:
            print("  cand %d/%d" % (ti + 1, N_TRAJ), flush=True)
    rank_dist = {k: ranks.count(k) for k in (1, 2, 3)}
    rank_dist["gt3"] = sum(1 for x in ranks if x > 3)
    cand = {
        "acc": sum(1 for x in ranks if x == 1), "n": N_TRAJ,
        "rank_dist": rank_dist,
        "mean_rank": sum(ranks) / len(ranks),
        "mean_margin": sum(margins) / len(margins),
        "mean_entropy": sum(ents) / len(ents),
        "mean_spearman": sum(sps) / len(sps),
    }
    print("candidate: %s" % json.dumps(cand), flush=True)

    # ---- teacher fidelity (H2 monitor口径) ----
    lc, dn, nn = fidelity(model, teacher_model, tok, eval_data, cand_ids, pad_id)
    fid = {"logit_cosine": lc, "delta_nll": dn, "logit_n": nn}
    print("fidelity: %s" % json.dumps(fid), flush=True)

    result = {
        "model": "H2 best_model.pt (step ~348, base norm1800 + rank4 LoRA)",
        "eval_protocol": "P0H5-aligned: torch.manual_seed(42)+random.seed(42) -> generate_dataset(600) -> generate_dataset(150)[:150]",
        "n": N_TRAJ,
        "item_ids": ids_p0h5,
        "item_ids_agg_md5": hashlib.md5("".join(ids_p0h5).encode()).hexdigest(),
        "draw_compare": {
            "p0h5_draw_vs_h2style_draw_id_overlap": id_overlap,
            "note": "H2 training script did NOT call random.seed(42); its monitor eval draw is not reproducible and differs from P0H5 set.",
        },
        "p0h5_correct_match": match,
        "candidate": cand,
        "teacher_fidelity": fid,
        "evaluator_hash": {
            "p0h5_candidate_diagnostics.py_md5": md5file(P0H5_PY),
            "this_script_md5": md5file(os.path.abspath(__file__)),
        },
        "p0h5_norm1800_reference": {
            "acc": 0, "mean_rank": 9.85, "mean_margin": -4.9246,
            "mean_entropy": 1.632, "mean_spearman": 0.743,
        },
        "h2_monitor_reference": {
            "step0": {"cand_acc": 0, "mean_rank": 9.5933, "mean_margin": -4.8658, "logit_cosine": 0.887356, "delta_nll": 4.5056},
            "step200": {"cand_acc": 0, "mean_rank": 9.3467, "mean_margin": -4.3067, "logit_cosine": 0.887312, "delta_nll": 1.8546},
            "note": "monitor eval draw was NOT P0H5-aligned (no random.seed); numbers are NOT directly comparable to P0H5.",
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nwrote %s (%.0fs)" % (OUT_JSON, time.time() - t0), flush=True)
    print("EVAL H2 STEP348 OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
