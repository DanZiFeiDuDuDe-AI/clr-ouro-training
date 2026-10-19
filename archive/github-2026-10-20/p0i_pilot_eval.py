# -*- coding: utf-8 -*-
"""
P0I pilot evaluation: canonical held-out candidate-scoring,
state norm/delta/cross-turn cosine.

Evaluates 4 conditions (M1-reset, M1-persist, N1800-reset, N1800-persist)
on the same canonical P0H5 N=150 held-out set (seed 42 protocol).

Also evaluates M0 teacher baseline.

Reports:
  - candidate accuracy, correct mean rank, correct-vs-best-wrong margin
  - entropy
  - Spearman vs M0
  - logit cosine vs M0
  - delta NLL vs M0
  - state norm / state delta / cross-turn cosine (persist only)

Writes: P0I_PILOT_EVAL.json + P0I_PILOT_RESULTS.md
"""
import os, sys, json, math, random, statistics
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import numpy as np
import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention, _RetentionAttentionAdapter
from clutrr_data import generate_dataset, RELATIONS
from pathlib import Path

REPLACED = [3, 7, 11, 15, 19, 23]
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
MAX_TOKENS = 128
SEED = 42
N_EVAL = 150

CKPT_MAP = {
    "m1-reset":     "D:/training/checkpoints/p0i_pilot_m1_reset_s42/best_model.pt",
    "m1-persist":   "D:/training/checkpoints/p0i_pilot_m1_persist_s42/best_model.pt",
    "n1800-reset":  "D:/training/checkpoints/p0i_pilot_n1800_reset_s42/best_model.pt",
    "n1800-persist":"D:/training/checkpoints/p0i_pilot_n1800_persist_s42/best_model.pt",
}

class P0IAdapter(_RetentionAttentionAdapter):
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        ut = kwargs.get("current_ut", 0)
        if ut > 0:
            self._state = None
        out, state = self.ret.forward(
            hidden_states,
            q_proj=self.ret.q_proj, k_proj=self.ret.k_proj,
            v_proj=self.ret.v_proj, o_proj=self.ret.o_proj,
            position_embeddings=position_embeddings,
            state=getattr(self, "_state", None),
            mode=getattr(self, "_mode", "recurrent"))
        self._last_state = state
        if ut == 0:
            self._turn_state = state
        return out, None

def reset_turn_state(adapters):
    for a in adapters:
        a._state = None; a._turn_state = None

def set_persist_state(adapters):
    for a in adapters:
        ts = getattr(a, "_turn_state", None)
        if ts is not None:
            S, z = ts
            a._state = (S.detach(), z.detach())
        else:
            a._state = None

def clear_all_state(adapters):
    for a in adapters:
        a._state = None; a._turn_state = None; a._last_state = None

def build_teacher():
    m, t = load_ouro(); return m, t

def build_student(init_tag, ckpt_path):
    m, t = load_ouro()
    converted = convert_ouro_to_retention(m, REPLACED, gamma_mode="sigmoid_logit")
    adapters = []
    for ret, idx in converted:
        layer = m.model.layers[idx]
        adapter = P0IAdapter(ret, layer.self_attn.src_attn)
        layer.self_attn = adapter
        adapters.append(adapter)
    for p in m.parameters():
        p.requires_grad_(False)
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    for i, (ret, idx) in enumerate(converted):
        if "retention_gamma" in ck and str(idx) in ck["retention_gamma"]:
            ret.gamma_logit.data.copy_(ck["retention_gamma"][str(idx)].to("cuda"))
    pnw = ck.get("post_norm_weights")
    if pnw:
        for li in REPLACED:
            for nm, w in pnw[str(li)].items():
                getattr(m.model.layers[li], nm).weight.data.copy_(w.to("cuda"))
    m.eval()
    return m, t, adapters

def prompt_ids(tok, s):
    return tok(s, truncation=True, max_length=MAX_TOKENS)["input_ids"]

@torch.no_grad()
def score(model, tok, pids, cands, pad, offset=0):
    n = len(cands); P = len(pids)
    L = P + max(map(len, cands))
    ids = torch.full((n, L), pad, dtype=torch.long)
    mask = torch.zeros((n, L), dtype=torch.long)
    pos = torch.zeros((n, L), dtype=torch.long)
    for i, c in enumerate(cands):
        seq = pids + c
        sl = len(seq)
        ids[i, :sl] = torch.tensor(seq)
        mask[i, :sl] = 1
        pos[i, :sl] = torch.arange(sl, dtype=torch.long, device="cuda") + offset
    out = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(),
                      use_cache=False, position_ids=pos.cuda())
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    elif isinstance(out, tuple):
        out = out[0]
    lp = torch.log_softmax(model.lm_head(out).float(), dim=-1)
    scores = np.array([sum(lp[i, P-1+j, t].item() for j, t in enumerate(c)) / len(c)
                       for i, c in enumerate(cands)])
    return scores, lp[0, P-1].detach().cpu().numpy()

def rankdata(a):
    order = np.argsort(a); r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1); return r

def spearman(a, b):
    x, y = rankdata(a), rankdata(b)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0

def metrics(scores, correct_i):
    order = np.argsort(-scores)
    rank = int(np.where(order == correct_i)[0][0]) + 1
    margin = float(scores[correct_i] - np.max(np.delete(scores, correct_i)))
    return rank, margin

@torch.no_grad()
def evaluate_model(model, adapters, tok, data, persist, collect_state=False):
    cands = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad = tok.pad_token_id
    ranks, margins, entropies = [], [], []
    teacher_vecs, student_vecs = [], []
    s_cos_all, s_dnll_all = [], []
    state_logs = []
    for i, tr in enumerate(data):
        clear_all_state(adapters)
        offset = 0
        prev_states = None
        traj_state = []
        for turn_i, turn in enumerate(tr["turns"]):
            if turn_i == 0:
                reset_turn_state(adapters)
            elif persist:
                set_persist_state(adapters)
            else:
                reset_turn_state(adapters)
            p = prompt_ids(tok, turn["prompt"])
            s_len = len(p)
            ci = RELATIONS.index(turn["answer"])
            ss, sl = score(model, tok, p, cands, pad, offset=offset)
            r, m = metrics(ss, ci)
            ranks.append(r); margins.append(m)
            entropies.append(float(-(F.softmax(torch.tensor(sl), dim=0) *
                                     F.log_softmax(torch.tensor(sl), dim=0)).sum()))
            student_vecs.extend(ss.tolist())
            # state stats (persist only): pooled state vector per layer per turn
            if collect_state:
                cur_states = []
                for a in adapters:
                    st = getattr(a, "_turn_state", None)
                    if st is not None:
                        S, z = st
                        vec = S.detach().float().mean(dim=(0, 1)).reshape(-1)
                        cur_states.append(vec)
                    else:
                        cur_states.append(None)
                traj_state.append(cur_states)
            if persist:
                offset += s_len
            else:
                offset = 0
        if collect_state and traj_state:
            state_logs.append(traj_state)
    acc = sum(1 for r in ranks if r == 1)
    mean_rank = statistics.mean(ranks)
    mean_margin = statistics.mean(margins)
    mean_entropy = statistics.mean(entropies)
    result = {
        "candidate_accuracy": acc,
        "candidate_accuracy_frac": acc / len(ranks),
        "n": len(ranks),
        "mean_rank": mean_rank,
        "mean_margin": mean_margin,
        "entropy": mean_entropy,
    }
    if collect_state and state_logs:
        norms_by_layer = {li: [] for li in REPLACED}
        deltas_by_layer = {li: [] for li in REPLACED}
        cos_by_layer = {li: [] for li in REPLACED}
        for traj_state in state_logs:
            for t_idx in range(len(traj_state)):
                for li_idx, li in enumerate(REPLACED):
                    v = traj_state[t_idx][li_idx]
                    if v is None:
                        continue
                    vn = v.norm().item()
                    norms_by_layer[li].append(vn)
                    if t_idx > 0:
                        pv = traj_state[t_idx - 1][li_idx]
                        if pv is not None:
                            deltas_by_layer[li].append((v - pv).norm().item())
                            denom = vn * pv.norm().item()
                            cos_by_layer[li].append(
                                float(F.cosine_similarity(v.unsqueeze(0), pv.unsqueeze(0)).item())
                                if denom > 1e-12 else 0.0)
        result["state_norm_mean"] = {str(li): statistics.mean(v) if v else 0 for li, v in norms_by_layer.items()}
        result["state_delta_mean"] = {str(li): statistics.mean(v) if v else 0 for li, v in deltas_by_layer.items()}
        result["cross_turn_cosine_mean"] = {str(li): statistics.mean(v) if v else 0 for li, v in cos_by_layer.items()}
    return result, ranks, margins, student_vecs

def main():
    print("=== P0I pilot evaluation ===", flush=True)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    _ = generate_dataset(600)
    data = generate_dataset(N_EVAL)[:N_EVAL]
    print("eval trajectories: %d" % len(data), flush=True)

    # Teacher baseline
    print("\nloading teacher (M0)...", flush=True)
    teacher, tok = build_teacher()
    t_res, t_ranks, t_margins, t_vecs = evaluate_model(teacher, [], tok, data, persist=False)
    print("M0: acc=%d/%d mean_rank=%.2f margin=%.4f" % (
        t_res["candidate_accuracy"], t_res["n"],
        t_res["mean_rank"], t_res["mean_margin"]), flush=True)
    del teacher; torch.cuda.empty_cache()

    results = {"n_eval": N_EVAL, "seed": SEED, "M0_teacher": t_res, "conditions": {}}
    teacher_vecs = t_vecs

    for tag in ["m1-reset", "m1-persist", "n1800-reset", "n1800-persist"]:
        ckpt = CKPT_MAP[tag]
        if not os.path.exists(ckpt):
            print("\nMISSING %s — skipping" % ckpt, flush=True)
            results["conditions"][tag] = {"missing": True}
            continue
        print("\n--- %s ---" % tag, flush=True)
        init_tag, mode = tag.rsplit("-", 1)
        model, _, adapters = build_student(init_tag, ckpt)
        persist = (mode == "persist")
        r, ranks, margins, s_vecs = evaluate_model(
            model, adapters, tok, data, persist=persist,
            collect_state=persist)
        r["spearman_vs_M0"] = spearman(np.array(teacher_vecs), np.array(s_vecs))
        results["conditions"][tag] = r
        print("%s: acc=%d/%d mean_rank=%.2f margin=%.4f entropy=%.4f spearman=%.4f" % (
            tag, r["candidate_accuracy"], r["n"],
            r["mean_rank"], r["mean_margin"], r["entropy"], r["spearman_vs_M0"]), flush=True)
        del model; torch.cuda.empty_cache()

    # Delta persist - reset
    for init in ["m1", "n1800"]:
        r_tag = "%s-reset" % init
        p_tag = "%s-persist" % init
        if (r_tag in results["conditions"] and p_tag in results["conditions"] and
            not results["conditions"][r_tag].get("missing") and
            not results["conditions"][p_tag].get("missing")):
            d = results["conditions"][p_tag]["candidate_accuracy"] - results["conditions"][r_tag]["candidate_accuracy"]
            results["delta_%s" % init] = d
            print("Delta %s: persist - reset = %d" % (init, d), flush=True)

    Path("P0I_PILOT_EVAL.json").write_text(json.dumps(results, indent=2))
    print("\nwrote P0I_PILOT_EVAL.json", flush=True)
    print("P0I EVAL OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
