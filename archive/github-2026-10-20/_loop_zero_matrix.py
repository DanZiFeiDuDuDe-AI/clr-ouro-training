#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ignition/Bootstrap hypothesis — the critical falsification test.

Distinguishes:
  M2's v0 "zeroed" (0%): FiLM sees S=0 at ALL 4 loops for the whole turn
    (v0 has no per-loop rebuild mechanism at all).
  Swap test's v1 "D_content_erase" (100%, unaffected): FiLM sees S=0 only
    at loop 1; loops 2-4 rebuild a nonzero S from real within-turn hidden
    states via the GRU, since only the INCOMING value was zeroed, not the
    per-loop override.

Prediction table (per user):
  E: every_loop_zero   FiLM=0 at loops 1,2,3,4 (replicate v0-zeroed within
                        the v1 per-loop protocol) -> predict COLLAPSE (~0%)
  G: loop1_real_rest_zero  loop1 sees the TRUE evolved incoming S; loops
                        2,3,4 forced to zero -> tests whether a single early
                        "kick" survives without sustained non-zero input
                        -> if ignition needs late-loop reinforcement: COLLAPSE
                        -> if one early kick is enough to carry through: RECOVER

(Already known, not rerun: "loop1 zero only, rest real" == swap test's D
condition == 100%/no effect, both seeds.)
"""
import torch
import os, sys, json, random, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import load_ouro, inject_film, turn_forward_v1, PlaaCore

MATRIX_SEED = os.environ.get("CLR_MATRIX_SEED", "42")
CKPT = f"D:/LoopLM/checkpoints/clr_ouro_v1_s{MATRIX_SEED}/best_model.pt"
OUT = f"D:/LoopLM/checkpoints/clr_ouro_v1_s{MATRIX_SEED}/loop_zero_matrix.json"
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
def eval_condition(model, tok, trajs, film_layers, plaa_core, mode):
    """mode: A_normal | E_every_loop_zero | G_loop1_real_rest_zero |
    H_loop4_only_real (loops1-3 forced to zero; loop4 explicitly given the
    TRUE untouched incoming S, captured before any manipulation this turn --
    NOT the same as merely "not overriding" loop4, since S's internal GRU
    tracking still drifts through loops1-3's zero-modulated hidden states
    even when FiLM itself is overridden there)."""
    correct = [0, 0]; total = [0, 0]
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            override = None
            zero = torch.zeros_like(S)
            true_incoming = S.clone()
            if mode == "E_every_loop_zero":
                override = {0: zero, 1: zero, 2: zero, 3: zero}
            elif mode == "G_loop1_real_rest_zero":
                override = {1: zero, 2: zero, 3: zero}  # loop0(=loop1) untouched: sees real S
            elif mode == "H_loop4_only_real":
                override = {0: zero, 1: zero, 2: zero, 3: true_incoming}
            elif mode == "I_loops12_zero_natural_rebuild":
                # loops1-2 forced zero; loops3-4 NOT overridden -> natural
                # GRU rebuild from real within-turn hidden-state feedback,
                # same style as D (which zeroed only loop1) but with a
                # 2-loop-zero starting deficit instead of 1-loop.
                override = {0: zero, 1: zero}
            logits, S_new, _ = turn_forward_v1(model, inp, film_layers, plaa_core,
                                               S, inject_override=override)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            idx = 0 if ti == 0 else 1
            total[idx] += 1; correct[idx] += ok
        torch.cuda.empty_cache()
    k, n = correct[1], total[1]
    return {"t2p": round(k/max(n,1), 4), "t2p_counts": f"{k}/{n}",
            "t2p_wilson95": wilson(k, n)}

def main():
    print("="*64, flush=True)
    print(f"Loop-zero matrix (Ignition/Bootstrap falsification) | seed={MATRIX_SEED}",
          flush=True)
    print("="*64, flush=True)
    torch.manual_seed(42); random.seed(42)   # same shared eval set as swap test
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

    if os.path.exists(OUT):
        results = json.load(open(OUT))
        print(f"loaded existing results, prior conditions: "
              f"{list(results['conditions'].keys())}", flush=True)
    else:
        results = {"ckpt": CKPT, "n_traj": len(eval_data), "conditions": {}}
    for mode in ["I_loops12_zero_natural_rebuild"]:
        r = eval_condition(model, tok, eval_data, film_layers, plaa_core, mode)
        results["conditions"][mode] = r
        print(f"  {mode:26s} {r['t2p_counts']} = {r['t2p']:.4f} "
              f"W95{r['t2p_wilson95']}", flush=True)

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nDONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
