#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_seqstate_s0 / dynamics.py — per-slot latent dynamics & collapse diagnostics.

For a trained checkpoint, saves the raw per-(trajectory, turn, loop) latent
Z logs (post-update state after each UT loop, the same convention as the
existing s_dynamics analysis) and computes:

  per_slot_norm          mean/std of ||slot_k|| per loop
  slot_cos_matrix        mean pairwise slot-slot cosine (K x K) per loop
  slot_variance          mean coordinate variance per slot per loop
  cross_traj_cos_slot    cross-trajectory cosine per slot per loop
                         (pooled over turn indices, 500 random pairs each)
                         -> the "first loops invariant / last loop
                            trajectory-dependent" probe, per slot
  cross_iter_cos_slot    cos(Z_i[slot], Z_{i+1}[slot]) within a turn
  pooled_cross_traj_cos  cross-trajectory cosine of the pooled modulation
                         m = to_modulation(Z) per loop (mirror of the
                         existing S4/S2 analysis)
  effective_rank         participation ratio + top singular values of the
                         flattened 64-dim states per loop (collapse check)

Vector configs run through the same code with K=1, d=64 (per-slot metrics
become per-vector metrics), giving the reference numbers.

Output: runs/<dir>/dynamics_s0.json + runs/<dir>/z_logs.pt
"""
import argparse
import json
import os
import random
import sys

import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = "D:/LoopLM"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clutrr_data import generate_dataset          # noqa: E402
from clr_ouro import (load_ouro, inject_film,     # noqa: E402
                      turn_forward_v1, DS)
from evaluate import (load_checkpoint,            # noqa: E402
                      build_state_module, replay_eval_data)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.reshape(1, -1), b.reshape(1, -1)))


def stats(xs):
    n = len(xs)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0}
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
    return {"mean": round(m, 4), "sd": round(sd, 4), "n": n}


def participation_ratio(s):
    """Effective rank: (sum s)^2 / sum s^2, in [1, len(s)]."""
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    return float(s.sum() ** 2 / (s ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-traj", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=500)
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    cfg = ckpt["config"]
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out_path = args.out or f"{run_dir}/dynamics_s0.json"
    K = cfg["num_slots"]
    d = cfg["slot_dim"]
    max_tokens = cfg["max_tokens"]

    print("=" * 72, flush=True)
    print(f"[S0 dynamics] ckpt={args.ckpt}", flush=True)
    print(f"  state type : {cfg['state_type']}  K={K}  d={d}  "
          f"seed={cfg['seed']}  n_traj={args.n_traj}", flush=True)
    print("=" * 72, flush=True)

    eval_data = replay_eval_data(cfg)[:args.n_traj]

    model, tok = load_ouro()
    film_layers = inject_film(model)
    state_module = build_state_module(cfg)
    state_module.load_state_dict(ckpt["state_module"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    print(f"ckpt step {ckpt.get('step')}", flush=True)

    # z_logs[loop][traj_turn] = [K, d] float32 cpu (post-update state)
    z_logs = {i: [] for i in range(4)}
    m_logs = {i: [] for i in range(4)}   # pooled modulation per loop
    with torch.no_grad():
        for traj in eval_data:
            S = state_module.init_state(1, "cuda")
            for ti, turn in enumerate(traj["turns"]):
                inp = tok(turn["prompt"], return_tensors="pt",
                          truncation=True, max_length=max_tokens).to("cuda")
                slog = []
                _, S_new, _ = turn_forward_v1(
                    model, inp, film_layers, None, S,
                    state_module=state_module, state_log=slog)
                S = S_new
                for i, z in enumerate(slog):
                    z_logs[i].append(z.reshape(K, d))
                    m_logs[i].append(
                        state_module.to_modulation(
                            z.reshape(1, K, d).to("cuda")).float().cpu()
                        .squeeze(0))
                del inp
                torch.cuda.empty_cache()

    results = {"ckpt": args.ckpt, "n_traj": len(eval_data),
               "K": K, "d": d, "state_type": cfg["state_type"]}

    # ---- per-slot norm (per loop) ----
    results["per_slot_norm"] = {
        f"loop{i+1}": {f"slot{j+1}": stats(
            [float(z[j].float().norm()) for z in z_logs[i]])
            for j in range(K)}
        for i in range(4)}

    # ---- slot-slot cosine matrix (per loop) ----
    results["slot_cos_matrix"] = {}
    for i in range(4):
        mat = []
        for j1 in range(K):
            row = []
            for j2 in range(K):
                row.append(stats([cos(z[j1], z[j2]) for z in z_logs[i]]))
            mat.append(row)
        offdiag = [mat[j1][j2]["mean"] for j1 in range(K)
                   for j2 in range(K) if j1 != j2]
        results["slot_cos_matrix"][f"loop{i+1}"] = {
            "matrix": [[round(r["mean"], 4) for r in row] for row in mat],
            "mean_pairwise": round(sum(offdiag) / max(len(offdiag), 1), 4)}

    # ---- slot variance (per loop, mean coordinate variance) ----
    results["slot_variance"] = {}
    for i in range(4):
        results["slot_variance"][f"loop{i+1}"] = {
            f"slot{j+1}": round(float(
                torch.stack([z[j] for z in z_logs[i]]).var(dim=0).mean()), 6)
            for j in range(K)}

    # ---- cross-trajectory cosine per slot (per loop, pooled over turns) ----
    rng = random.Random(20260816)
    idx = 0
    turn_bank = {i: {} for i in range(4)}
    for traj in eval_data:
        for ti in range(len(traj["turns"])):
            for i in range(4):
                turn_bank[i].setdefault(ti, []).append(z_logs[i][idx])
            idx += 1

    def cross_sim(bank, j):
        sims = []
        for ti, vecs in bank.items():
            if len(vecs) < 2:
                continue
            for _ in range(min(args.pairs, len(vecs) * (len(vecs) - 1) // 2)):
                a, b = rng.sample(range(len(vecs)), 2)
                sims.append(cos(vecs[a][j], vecs[b][j]))
        return stats(sims)

    results["cross_traj_cos_slot"] = {}
    for i in range(4):
        results["cross_traj_cos_slot"][f"loop{i+1}"] = {
            f"slot{j+1}": cross_sim(turn_bank[i], j) for j in range(K)}

    # ---- cross-iteration cosine per slot (within a turn) ----
    results["cross_iter_cos_slot"] = {}
    for i in range(3):
        results["cross_iter_cos_slot"][f"c{i+1}{i+2}"] = {
            f"slot{j+1}": stats([cos(z1[j], z2[j])
                                for z1, z2 in zip(z_logs[i], z_logs[i + 1])])
            for j in range(K)}

    # ---- pooled-state cross-trajectory cosine (per loop) ----
    pooled_bank = {i: {} for i in range(4)}
    idx = 0
    for traj in eval_data:
        for ti in range(len(traj["turns"])):
            for i in range(4):
                pooled_bank[i].setdefault(ti, []).append(m_logs[i][idx])
            idx += 1

    def pooled_sim(bank):
        sims = []
        for ti, vecs in bank.items():
            if len(vecs) < 2:
                continue
            for _ in range(min(args.pairs, len(vecs) * (len(vecs) - 1) // 2)):
                a, b = rng.sample(range(len(vecs)), 2)
                sims.append(cos(vecs[a], vecs[b]))
        return stats(sims)

    results["pooled_cross_traj_cos"] = {
        f"loop{i+1}": pooled_sim(pooled_bank[i]) for i in range(4)}

    # ---- effective rank / singular values of flattened Z (per loop) ----
    results["effective_rank"] = {}
    for i in range(4):
        mat = torch.stack([z.reshape(-1).float() for z in z_logs[i]])
        mat = mat - mat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(mat)
        cum = torch.cumsum(s, dim=0) / s.sum()
        results["effective_rank"][f"loop{i+1}"] = {
            "participation_ratio": round(participation_ratio(s), 2),
            "top5_svals": [round(float(x), 3) for x in s[:5]],
            "cum_energy_top5": [round(float(x), 3) for x in cum[:5]],
            "n_samples": mat.shape[0]}

    # ---- raw logs ----
    torch.save({"z_logs": {str(k): torch.stack(v) for k, v in z_logs.items()},
                "config": cfg}, f"{run_dir}/z_logs.pt")

    json.dump(results, open(out_path, "w"), indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
