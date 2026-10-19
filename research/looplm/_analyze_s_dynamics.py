#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification of the observed intra-cycle expansion-compression pattern (v1).

Question (per review discussion): the norm trajectory [~7.1, ~7.9, ~7.5, ~0.4]
alone cannot distinguish
  (a) genuine compression: norm drops AND direction changes substantially
  (b) amplitude rescaling: norm drops but cos(S3, S4) stays high
  (c) gate closure / near-constant output: S4 nearly identical across inputs.

Metrics on the trained v1 checkpoint (60 eval trajectories, evolved S):
  1. per-transition cosine: cos(S1,S2), cos(S2,S3), cos(S3,S4) within each turn
  2. per-transition ||dS||
  3. cross-trajectory similarity of the compressed state: cos(S4^i, S4^j)
     over random pairs of different trajectories (same turn index) --
     high => S4 is a near-constant mode signal; spread => carries content
  4. same for the expanded state S2 (contrast)
"""
import torch
import os, sys, json, random

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset
from clr_ouro import load_ouro, inject_film, turn_forward_v1, PlaaCore

DYN_SEED = os.environ.get("CLR_DYN_SEED", "42")
CKPT = f"D:/LoopLM/checkpoints/clr_ouro_v1_s{DYN_SEED}/best_model.pt"
# 2026-07-13: write to a new file first (not overwriting the existing
# verified s_dynamics.json) so the rerun's S2/S4 numbers can be checked
# against the already-published ones before trusting the new S1/S3 numbers.
OUT = f"D:/LoopLM/checkpoints/clr_ouro_v1_s{DYN_SEED}/s_dynamics_full.json"
N_TRAJ = 60
MAX_TOKENS = 128

def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
    return {"mean": round(m, 4), "sd": round(sd, 4), "n": n}

def main():
    torch.manual_seed(42); random.seed(42)
    _train = generate_dataset(600)          # replay stream -> same eval set
    eval_data = generate_dataset(150)[:N_TRAJ]

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
    print(f"ckpt step {ckpt.get('step')} variant {ckpt.get('variant')}", flush=True)

    trans_cos = {k: [] for k in ["c12", "c23", "c34"]}
    trans_dnorm = {k: [] for k in ["d12", "d23", "d34"]}
    norms = {k: [] for k in ["n1", "n2", "n3", "n4"]}
    s4_by_turnidx = {}      # turn index -> list of S4 vectors (per trajectory)
    s2_by_turnidx = {}
    s1_by_turnidx = {}      # added 2026-07-13: complete the S1..S4 matrix
    s3_by_turnidx = {}

    with torch.no_grad():
        for traj in eval_data:
            S = plaa_core.init_state(1, "cuda")
            for ti, turn in enumerate(traj["turns"]):
                inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                          max_length=MAX_TOKENS).to("cuda")
                slog = []
                _, S_new, _ = turn_forward_v1(model, inp, film_layers, plaa_core,
                                              S, state_log=slog)
                S = S_new
                s1, s2, s3, s4 = slog          # states after loops 1..4
                trans_cos["c12"].append(cos(s1, s2))
                trans_cos["c23"].append(cos(s2, s3))
                trans_cos["c34"].append(cos(s3, s4))
                trans_dnorm["d12"].append(float((s2 - s1).norm()))
                trans_dnorm["d23"].append(float((s3 - s2).norm()))
                trans_dnorm["d34"].append(float((s4 - s3).norm()))
                for i, s in enumerate([s1, s2, s3, s4], 1):
                    norms[f"n{i}"].append(float(s.norm()))
                s4_by_turnidx.setdefault(ti, []).append(s4)
                s2_by_turnidx.setdefault(ti, []).append(s2)
                s1_by_turnidx.setdefault(ti, []).append(s1)
                s3_by_turnidx.setdefault(ti, []).append(s3)
            torch.cuda.empty_cache()

    # cross-trajectory similarity at fixed turn index (500 random pairs each)
    rng = random.Random(20260710)
    def cross_sim(bank):
        sims = []
        for ti, vecs in bank.items():
            if len(vecs) < 2:
                continue
            for _ in range(min(500, len(vecs) * (len(vecs) - 1) // 2)):
                i, j = rng.sample(range(len(vecs)), 2)
                sims.append(cos(vecs[i], vecs[j]))
        return stats(sims)

    results = {
        "ckpt": CKPT, "n_traj": len(eval_data),
        "norms": {k: stats(v) for k, v in norms.items()},
        "consecutive_cos": {k: stats(v) for k, v in trans_cos.items()},
        "consecutive_dnorm": {k: stats(v) for k, v in trans_dnorm.items()},
        "cross_traj_cos_S4_compressed": cross_sim(s4_by_turnidx),
        "cross_traj_cos_S2_expanded": cross_sim(s2_by_turnidx),
        "cross_traj_cos_S1": cross_sim(s1_by_turnidx),
        "cross_traj_cos_S3": cross_sim(s3_by_turnidx),
    }
    print(json.dumps(results, indent=2), flush=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
