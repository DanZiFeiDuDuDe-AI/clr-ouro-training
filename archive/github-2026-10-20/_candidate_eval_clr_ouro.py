#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format-free candidate-scoring evaluation (v0 checkpoint).

Closes the biggest open critique of the fast_eval protocol: under next-token
argmax, without_S = 0 could be mere format non-compliance (the unmodulated
model never emits the answer-format token) rather than absent memory. Here
every turn is scored by ranking all 14 relation words as continuations
(mean token log-prob), so format compliance is irrelevant by construction.

Expectations (mirroring the 0.8B line, where candidate scoring put the zeroed
floor at ~0.08 instead of 0):
  - with_S:    should stay near the fast_eval number (~0.5)
  - without_S: should rise from exactly-0 to a small but nonzero floor
    (copy-the-atomic-relation behavior gets some t2p items right by chance)
The delta between protocols isolates the format component of the effect.
"""
import torch
import os, sys, json, random, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset, bayes_ceiling, RELATIONS
from clr_ouro import load_ouro, inject_film, set_state, turn_forward_v0, PlaaCore

CKPT = "D:/LoopLM/checkpoints/clr_ouro_v0_s42/best_model.pt"
OUT = "D:/LoopLM/checkpoints/clr_ouro_v0_s42/candidate_eval.json"
N_TRAJ = 150
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
    """Rank candidates by mean token log-prob as continuations of the prompt.
    One batched (n_cand) forward through the full UT-loop stack."""
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
def eval_candidates(model, tok, trajs, film_layers, plaa_core, cand_ids,
                    ablate_s=False):
    pad_id = tok.pad_token_id
    correct = [0, 0]; total = [0, 0]
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            s_val = torch.zeros_like(S) if ablate_s else S
            pred = score_candidates(model, film_layers, prompt_ids, s_val,
                                    cand_ids, pad_id)
            ok = int(RELATIONS[pred] == turn["answer"])
            # v0 protocol: update S from a prompt-only forward (pre-update
            # state used for the prediction, matching fast_eval)
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            _, _, h_last = turn_forward_v0(model, inp, film_layers, s_val)
            S = plaa_core(S, h_last.unsqueeze(0))
            if ti == 0: total[0] += 1; correct[0] += ok
            else:       total[1] += 1; correct[1] += ok
        torch.cuda.empty_cache()
    return {"t1": round(correct[0]/max(total[0],1), 4),
            "t2p": round(correct[1]/max(total[1],1), 4),
            "t2p_counts": f"{correct[1]}/{total[1]}",
            "t2p_wilson95": wilson(correct[1], total[1])}

def main():
    print("="*64, flush=True)
    print("Format-free candidate-scoring eval (v0 ckpt)", flush=True)
    print("="*64, flush=True)
    torch.manual_seed(42); random.seed(42)
    _train = generate_dataset(600)
    eval_data = generate_dataset(N_TRAJ)
    print(f"eval={len(eval_data)} ceiling={bayes_ceiling(eval_data):.4f}", flush=True)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    plaa_core = PlaaCore()
    try:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    except Exception:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    plaa_core.load_state_dict(ckpt["plaa_core"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    norms = [round(float(l.scale_proj.weight.float().norm()), 3) for l in film_layers]
    print(f"ckpt step {ckpt.get('step')} | norms {norms}", flush=True)
    assert all(x > 0.1 for x in norms)

    # candidates as natural continuations (leading space), same format for all
    cand_ids = [tok(" " + r, add_special_tokens=False)["input_ids"]
                for r in RELATIONS]

    results = {"ckpt": CKPT, "n_traj": len(eval_data),
               "ceiling": bayes_ceiling(eval_data), "conditions": {}}
    for name, ab in [("with_S", False), ("without_S", True)]:
        r = eval_candidates(model, tok, eval_data, film_layers, plaa_core,
                            cand_ids, ablate_s=ab)
        results["conditions"][name] = r
        print(f"  {name:10s} t1={r['t1']:.3f} t2p={r['t2p_counts']} = {r['t2p']:.4f} "
              f"W95{r['t2p_wilson95']}", flush=True)

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nDONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
