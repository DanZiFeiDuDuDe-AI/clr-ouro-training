#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4/S2 swap causal test (v1 checkpoint) — the observation->mechanism gate.

The s_dynamics analysis showed: mid-loop states are a constant saturated
carrier (cross-trajectory cos = 1.0) while the turn-final S4 is the only
input-specific vector (cos = 0.307). If that picture is causal:

  B (content swap)  — replacing each turn's INCOMING state with another
                      trajectory's same-position S4 should CHANGE behavior.
  C (carrier swap)  — overriding what FiLM sees at loops 2-4 with another
                      trajectory's mid-loop states should do NOTHING
                      (they are identical anyway).
  D (content erase) — zeroing the incoming state at every turn should kill
                      cross-turn memory (t2p -> ~0).

Tension to resolve: on the 0.8B (non-looped) backbone, cross-trajectory state
swaps had NO effect (phase 5B v4). If B hurts here, the looped backbone has
upgraded S from a mode signal to a genuine content channel.

Method: two passes. Pass 1 (= condition A, normal) records per-(traj, turn)
state banks {s_in, s1..s4}. Pass 2 runs B/C/D against those recorded banks
(standard activation-patching methodology; donor = trajectory (i+7) mod N at
the same turn index, falling back to any trajectory that has that turn).
"""
import torch
import os, sys, json, random, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import load_ouro, inject_film, turn_forward_v1, PlaaCore

SWAP_SEED = os.environ.get("CLR_SWAP_SEED", "42")
CKPT = f"D:/LoopLM/checkpoints/clr_ouro_v1_s{SWAP_SEED}/best_model.pt"
OUT = f"D:/LoopLM/checkpoints/clr_ouro_v1_s{SWAP_SEED}/swap_test.json"
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
def run_pass(model, tok, trajs, film_layers, plaa_core, mode, banks=None):
    """mode: A_record | B_content_swap | C_carrier_swap | D_content_erase"""
    correct = [0, 0]; total = [0, 0]     # t1, t2p
    rec = {} if mode == "A_record" else None
    n_traj = len(trajs)

    def donor_entry(ti_idx, turn_idx):
        for off in range(7, 7 + n_traj):
            cand = (ti_idx + off) % n_traj
            if cand != ti_idx and turn_idx in banks[cand]:
                return banks[cand][turn_idx]
        return None

    for tj, traj in enumerate(trajs):
        S = plaa_core.init_state(1, "cuda")
        if rec is not None:
            rec[tj] = {}
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            override = None
            if mode == "B_content_swap":
                d = donor_entry(tj, ti)
                if d is not None:
                    S = d["s_in"].to(device="cuda",
                                     dtype=torch.bfloat16).unsqueeze(0).clone()
            elif mode == "D_content_erase":
                S = torch.zeros_like(S)
            elif mode == "C_carrier_swap":
                d = donor_entry(tj, ti)
                if d is not None:
                    override = {1: d["loops"][0].unsqueeze(0),
                                2: d["loops"][1].unsqueeze(0),
                                3: d["loops"][2].unsqueeze(0)}
            slog = [] if rec is not None else None
            s_in = S.detach().float().cpu().squeeze(0).clone() if rec is not None else None
            logits, S_new, _ = turn_forward_v1(model, inp, film_layers, plaa_core,
                                               S, state_log=slog,
                                               inject_override=override)
            if rec is not None:
                rec[tj][ti] = {"s_in": s_in, "loops": slog}
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            if ti == 0: total[0] += 1; correct[0] += ok
            else:       total[1] += 1; correct[1] += ok
        torch.cuda.empty_cache()
    k, n = correct[1], total[1]
    return {"t2p": round(k/max(n,1), 4), "t2p_counts": f"{k}/{n}",
            "t2p_wilson95": wilson(k, n)}, rec

def main():
    print("="*64, flush=True)
    print("S4/S2 swap causal test (v1, per-loop protocol)", flush=True)
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

    results = {"ckpt": CKPT, "n_traj": len(eval_data), "conditions": {}}
    a, banks = run_pass(model, tok, eval_data, film_layers, plaa_core, "A_record")
    results["conditions"]["A_normal"] = a
    print(f"  A_normal        {a['t2p_counts']} = {a['t2p']:.4f} W95{a['t2p_wilson95']}",
          flush=True)
    for mode in ["B_content_swap", "C_carrier_swap", "D_content_erase"]:
        r, _ = run_pass(model, tok, eval_data, film_layers, plaa_core, mode, banks)
        results["conditions"][mode] = r
        print(f"  {mode:15s} {r['t2p_counts']} = {r['t2p']:.4f} W95{r['t2p_wilson95']}",
              flush=True)

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nDONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
