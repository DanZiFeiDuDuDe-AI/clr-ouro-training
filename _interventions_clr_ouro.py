#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2 eval-time intervention suite on the trained CLR×Ouro checkpoint.

Conditions (all on the SAME 150-trajectory eval set as the M1 final eval,
reproduced by replaying the training script's RNG stream):
  A  full        evolved S (positive control; must reproduce 78/150)
  B  zeroed      S = 0                     (negative control; must collapse)
  C  random      S ~ N(0, I) each turn     (non-specific perturbation control)
  D  frozen      S = S0, no evolution      (eval-time static control)
  E  dose        alpha * evolved S, alpha in {0.25, 0.5, 0.75, 1.0}

Checkpoint: checkpoints/clr_ouro_v0_s42/best_model.pt (step 100).
Protocol: identical fast_eval clone as training (pre-update S, next-token argmax).
"""
import torch
import os, sys, json, random, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import (load_ouro, inject_film, turn_forward_v0, PlaaCore,
                      DS, FILM_LAYERS)

CKPT = "D:/LoopLM/checkpoints/clr_ouro_v0_s42/best_model.pt"
OUT = "D:/LoopLM/checkpoints/clr_ouro_v0_s42/interventions.json"
SEED = 42
EVAL_TRAJ = 150
MAX_TOKENS = 128

Z = 1.959963984540054
def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))

@torch.no_grad()
def eval_condition(model, tok, trajs, film_layers, plaa_core, mode, alpha=1.0):
    """mode: full | zeroed | random | frozen | dose"""
    correct = [0,0,0,0]; total = [0,0,0,0]
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            if mode == "zeroed":
                s_val = torch.zeros_like(S)
            elif mode == "random":
                s_val = torch.randn_like(S)
            elif mode == "dose":
                s_val = alpha * S
            else:  # full / frozen
                s_val = S
            _, logits, h_last = turn_forward_v0(model, inp, film_layers, s_val)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            if mode != "frozen":                       # frozen: no evolution
                S = plaa_core(S, h_last.unsqueeze(0))
            total[3] += 1; correct[3] += ok
            if ti == 0: total[0] += 1; correct[0] += ok
            else:       total[1] += 1; correct[1] += ok
            if ti == len(traj["turns"])-1: total[2] += 1; correct[2] += ok
        torch.cuda.empty_cache()
    k, n = correct[1], total[1]
    return {"t2p_correct": k, "t2p_total": n, "t2p": round(k/max(n,1), 4),
            "t2p_wilson95": wilson(k, n), "t1": round(correct[0]/max(total[0],1), 4)}

def main():
    print("="*64, flush=True)
    print("CLR x Ouro — eval-time intervention suite (M2)", flush=True)
    print("="*64, flush=True)
    torch.manual_seed(SEED); random.seed(SEED)
    _train = generate_dataset(600)                 # replay RNG stream
    eval_data = generate_dataset(EVAL_TRAJ)        # same eval set as M1 final
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
    print(f"ckpt step {ckpt.get('step')} | scale_proj norms {norms}", flush=True)
    assert all(x > 0.1 for x in norms), "zero injection weights"

    results = {"ckpt_step": int(ckpt.get("step", -1)),
               "ceiling": bayes_ceiling(eval_data), "conditions": {}}
    plan = [("A_full", "full", 1.0), ("B_zeroed", "zeroed", 1.0),
            ("C_random", "random", 1.0), ("D_frozen_S0", "frozen", 1.0),
            ("E_dose_0.25", "dose", 0.25), ("E_dose_0.50", "dose", 0.50),
            ("E_dose_0.75", "dose", 0.75), ("E_dose_1.00", "dose", 1.00)]
    for name, mode, a in plan:
        r = eval_condition(model, tok, eval_data, film_layers, plaa_core, mode, a)
        results["conditions"][name] = r
        print(f"  {name:14s} t2p={r['t2p']:.4f} ({r['t2p_correct']}/{r['t2p_total']}) "
              f"W95{r['t2p_wilson95']} t1={r['t1']:.3f}", flush=True)

    # positive-control check: A must match the M1 final eval (78/150)
    a = results["conditions"]["A_full"]
    if a["t2p_correct"] != 78:
        print(f"  [warn] A_full={a['t2p_correct']}/150, M1 final was 78/150 "
              f"(init-noise jitter tolerance: ±2)", flush=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nDONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
