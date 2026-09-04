#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate-scoring audit v2 (no-space candidates, multiple checkpoints).

Fixes vs v1: candidates tokenized WITHOUT leading space, matching the trained
target format (fast_eval trained/eval'd against tok(answer)[0], no space).
Runs the audit on both the ut4 and ut1 checkpoints to test whether the
loop-depth effect (fast_eval: 32% -> 52%) survives a format-free readout.
"""
import torch
import os, sys, json, random, math

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset, bayes_ceiling, RELATIONS
from clr_ouro import load_ouro, inject_film, set_state, turn_forward_v0, PlaaCore

CKPTS = {
    "ut4": ("D:/LoopLM/checkpoints/clr_ouro_v0_s42/best_model.pt", 4),
    "ut1": ("D:/LoopLM/checkpoints/clr_ouro_v0_ut1_s42/best_model.pt", 1),
}
OUT = "D:/LoopLM/checkpoints/candidate_eval_v2.json"
N_TRAJ = 150  # full match to fast_eval's "final" 150-traj measurement (was 100:
              # a strict prefix of the same seed=42 draw, not a different sample)
MAX_TOKENS = 128

Z = 1.959963984540054
def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))

@torch.no_grad()
def score_candidates(model, film_layers, prompt_ids, s_vec, cand_ids, pad_id):
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
    set_state(film_layers, s_vec)
    base_out, _, _ = model.model(input_ids=input_ids, attention_mask=attn,
                                 use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(base_out.last_hidden_state).float(),
                                 dim=-1)
    best, best_i = -1e30, 0
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        if lp > best:
            best, best_i = lp, i
    return best_i

@torch.no_grad()
def eval_ckpt(model, tok, trajs, film_layers, plaa_core, cand_ids, ablate_s):
    pad_id = tok.pad_token_id
    c = [0, 0]; t = [0, 0]
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            s_val = torch.zeros_like(S) if ablate_s else S
            pred = score_candidates(model, film_layers, prompt_ids, s_val,
                                    cand_ids, pad_id)
            ok = int(RELATIONS[pred] == turn["answer"])
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            _, _, h_last = turn_forward_v0(model, inp, film_layers, s_val)
            S = plaa_core(S, h_last.unsqueeze(0))
            idx = 0 if ti == 0 else 1
            t[idx] += 1; c[idx] += ok
            del inp
        torch.cuda.empty_cache()   # 4GB card: fragmentation killed the first run
    return {"t1": round(c[0]/max(t[0],1), 3),
            "t2p": round(c[1]/max(t[1],1), 4),
            "t2p_counts": f"{c[1]}/{t[1]}",
            "t2p_wilson95": wilson(c[1], t[1])}

def main():
    print("Candidate audit v2 (no-space candidates)", flush=True)
    torch.manual_seed(42); random.seed(42)
    _train = generate_dataset(600)
    eval_data = generate_dataset(150)[:N_TRAJ]
    print(f"eval={len(eval_data)} ceiling={bayes_ceiling(eval_data):.4f}", flush=True)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    plaa_core = PlaaCore()
    # no-space candidates: matches the trained target tok(answer)[0] format
    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]

    results = {"n_traj": len(eval_data), "candidates": "no_leading_space",
               "runs": {}}
    for tag, (path, ut) in CKPTS.items():
        try:
            ckpt = torch.load(path, map_location="cuda", weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location="cuda", weights_only=False)
        plaa_core.load_state_dict(ckpt["plaa_core"])
        for l, sd in zip(film_layers, ckpt["scale_proj"]):
            l.scale_proj.load_state_dict(sd)
        model.model.total_ut_steps = ut
        model.config.total_ut_steps = ut
        norms = [round(float(l.scale_proj.weight.float().norm()), 2)
                 for l in film_layers]
        assert all(x > 0.1 for x in norms)
        print(f"[{tag}] ckpt step {ckpt.get('step')} ut={ut} norms={norms}",
              flush=True)
        for cond, ab in [("without_S", True), ("with_S", False)]:
            r = eval_ckpt(model, tok, eval_data, film_layers, plaa_core,
                          cand_ids, ab)
            results["runs"][f"{tag}_{cond}"] = r
            print(f"  {tag}_{cond:10s} t1={r['t1']} t2p={r['t2p_counts']} = "
                  f"{r['t2p']} W95{r['t2p_wilson95']}", flush=True)

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nDONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
