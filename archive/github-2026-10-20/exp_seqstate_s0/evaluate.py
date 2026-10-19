#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_seqstate_s0 / evaluate.py — post-training evaluation suite.

For a trained checkpoint (runs/<tag>_<state>_<K>x<d>_s<seed>/best_model.pt):

  1. k-curve        eval-time loop-depth scan k = 1..4 (max_ut), with_S and
                    without_S (literal-zero modulation). This is the primary
                    table: accuracy vs recurrent depth under the SAME trained
                    weights (the depth-compute Pareto probe).
  2. early readout  one extra pass per config: full 4-loop state evolution
                    but reading logits after every loop (isolates readout
                    depth from state-update count).
  3. interventions  (sequence configs only) on the same 150-trajectory eval
                    set, k=4: normal, zero_slots, random_slots,
                    static_trained, slot_permutation (3 random perms),
                    single_slot (per slot), slot_dropout (p in {0.25,0.5,0.75},
                    3 mask samples each). All are injection-only: the true
                    latent chain still evolves, only the FiLM input changes.
                    static_trained freezes the latent (no evolution at all).

Output: runs/<dir>/eval_s0.json + printed tables.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = "D:/LoopLM"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clutrr_data import generate_dataset, bayes_ceiling            # noqa: E402
from clr_ouro import (load_ouro, inject_film, turn_forward_v1,     # noqa: E402
                      FILM_LAYERS)
from state_modules import VectorStateModule, SequenceStateModule   # noqa: E402

Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = (Z / d) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def build_state_module(cfg):
    if cfg["state_type"] == "vector":
        return VectorStateModule()
    return SequenceStateModule(n_slots=cfg["num_slots"],
                               d_slot=cfg["slot_dim"],
                               n_layers=cfg.get("updater_layers", 1),
                               pool=cfg.get("pool_type", "flatten"))


def load_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cuda", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cuda", weights_only=False)
    return ckpt


def replay_eval_data(cfg):
    """Reproduce the exact eval split used by train.py (same RNG stream)."""
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    generate_dataset(cfg["train_samples"])
    return generate_dataset(cfg["eval_traj_final"])


@torch.no_grad()
def eval_cond(model, tok, trajs, film_layers, state_module, max_tokens,
              ablate_s=False, no_update=False, intervene=None, max_ut=4,
              label="eval", progress=50):
    """v1 per-loop protocol eval; returns t1/t2p/last/all bucket dict."""
    import time as _t
    t0 = _t.time()
    correct = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    for ti_, traj in enumerate(trajs):
        if progress and ti_ % progress == 0 and ti_ > 0:
            print(f"    [{label}] {ti_}/{len(trajs)} traj, "
                  f"{_t.time()-t0:.0f}s", flush=True)
        S = state_module.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=max_tokens).to("cuda")
            logits, S_new, _ = turn_forward_v1(
                model, inp, film_layers, None, S, ablate_s=ablate_s,
                state_module=state_module, max_ut=max_ut,
                intervene=intervene, no_update=no_update)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            total[3] += 1
            correct[3] += ok
            if ti == 0:
                total[0] += 1
                correct[0] += ok
            else:
                total[1] += 1
                correct[1] += ok
            if ti == len(traj["turns"]) - 1:
                total[2] += 1
                correct[2] += ok
        torch.cuda.empty_cache()
    return {k: {"correct": correct[i], "total": total[i],
                "acc": correct[i] / max(total[i], 1)}
            for i, k in enumerate(["t1", "t2p", "last", "all"])}


@torch.no_grad()
def eval_readout_curve(model, tok, trajs, film_layers, state_module,
                       max_tokens):
    """Full 4-loop evolution; score the logits after every loop."""
    n_ut = 4
    correct = [[0, 0, 0, 0] for _ in range(n_ut)]
    total = [[0, 0, 0, 0] for _ in range(n_ut)]
    for traj in trajs:
        S = state_module.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=max_tokens).to("cuda")
            _, S_new, _, pll = turn_forward_v1(
                model, inp, film_layers, None, S,
                state_module=state_module, per_loop_logits=True)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            for i, lg in enumerate(pll):
                ok = int(lg.argmax().item() == tgt)
                total[i][3] += 1
                correct[i][3] += ok
                if ti == 0:
                    total[i][0] += 1
                    correct[i][0] += ok
                else:
                    total[i][1] += 1
                    correct[i][1] += ok
                if ti == len(traj["turns"]) - 1:
                    total[i][2] += 1
                    correct[i][2] += ok
            S = S_new
        torch.cuda.empty_cache()
    keys = ["t1", "t2p", "last", "all"]
    return {f"loop{i+1}": {k: {"correct": correct[i][j],
                               "total": total[i][j],
                               "acc": correct[i][j] / max(total[i][j], 1)}
                           for j, k in enumerate(keys)}
            for i in range(n_ut)}


def make_intervene(mode, K, d, perm=None, slot=None, p=None, gen=None):
    """Return (intervene_fn_or_None, no_update) for an eval condition."""
    if mode == "normal":
        return None, False
    if mode == "zero_slots":
        return (lambda Z: torch.zeros_like(Z)), False
    if mode == "random_slots":
        def rnd(Z):
            # randn_like accepts no generator on CUDA tensors; sample CPU-side
            # with the seeded generator and move over (deterministic).
            g = torch.randn(Z.shape, generator=gen, dtype=torch.float32)
            return g.to(device=Z.device, dtype=Z.dtype)
        return rnd, False
    if mode == "permute":
        perm_t = torch.tensor(perm, device="cuda")
        return (lambda Z: Z[:, perm_t]), False
    if mode == "single_slot":
        mask = torch.zeros(K, d, device="cuda")
        mask[slot] = 1.0
        return (lambda Z: Z * mask), False
    if mode == "dropout":
        def drop(Z):
            keep = (torch.rand(Z.shape, generator=gen) > p)
            return Z * keep.to(device=Z.device, dtype=Z.dtype)
        return drop, False
    if mode == "static_trained":
        return None, True
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-eval", type=int, default=0)   # 0 -> config value
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--interventions-only", action="store_true",
                    help="skip k-curve + early readout; only run interventions "
                         "(resume after a crashed interventions stage)")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    cfg = ckpt["config"]
    n_eval = args.n_eval or cfg["eval_traj_final"]
    max_tokens = cfg["max_tokens"]
    seed = cfg["seed"]
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out_path = args.out or f"{run_dir}/eval_s0.json"
    ckpt_hash = hashlib.sha256(open(args.ckpt, "rb").read()).hexdigest()[:16]

    # preamble (required before evaluation)
    print("=" * 72, flush=True)
    print(f"[S0 eval] ckpt={args.ckpt}", flush=True)
    print(f"  state type      : {cfg['state_type']}", flush=True)
    print(f"  num slots       : {cfg['num_slots']}  slot dim: {cfg['slot_dim']}",
          flush=True)
    print(f"  trainable params: {cfg['n_trainable']:,}", flush=True)
    print(f"  backbone frozen : {cfg['n_backbone_frozen']:,}", flush=True)
    print(f"  dataset split   : train={cfg['train_samples']} traj, "
          f"eval={n_eval} traj (replayed RNG stream, seed {seed})", flush=True)
    print(f"  seed            : {seed}", flush=True)
    print(f"  eval protocol   : {cfg['protocol']}", flush=True)
    print(f"  checkpoint hash : {ckpt_hash} (recorded: {cfg.get('ckpt_sha256')})",
          flush=True)
    print("=" * 72, flush=True)

    eval_data = replay_eval_data(cfg)
    eval_data = eval_data[:n_eval]
    ceiling = bayes_ceiling(eval_data)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    state_module = build_state_module(cfg)
    state_module.load_state_dict(ckpt["state_module"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    print(f"ckpt step {ckpt.get('step')} | scale_proj norms "
          f"{[round(float(l.scale_proj.weight.float().norm()), 3) for l in film_layers]}",
          flush=True)

    results = {"ckpt": args.ckpt, "ckpt_sha256": ckpt_hash,
               "ckpt_step": int(ckpt.get("step", -1)), "ceiling": ceiling,
               "n_eval": len(eval_data), "seed": seed,
               "protocol": cfg["protocol"],
               "trainable_params": cfg["n_trainable"],
               "backbone_frozen_params": cfg["n_backbone_frozen"]}

    def save():
        # incremental checkpoint: long GPU job, must survive interrupts
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"), indent=2)
        os.replace(tmp, out_path)

    if args.interventions_only:
        # resume mode: keep k_curve / early_readout from a prior partial run
        if os.path.exists(out_path):
            prev = json.load(open(out_path))
            results.update({k: v for k, v in prev.items()
                            if k in ("k_curve", "early_readout")})
            print(f"  [interventions-only] kept k_curve/early_readout from "
                  f"{out_path}", flush=True)

    # ---- 1. k-curve ----
    if not args.interventions_only:
        kcurve = {}
        for k in range(1, 5):
            w = eval_cond(model, tok, eval_data, film_layers, state_module,
                          max_tokens, max_ut=k, label=f"k{k}_with")
            o = eval_cond(model, tok, eval_data, film_layers, state_module,
                          max_tokens, ablate_s=True, max_ut=k,
                          label=f"k{k}_without")
            kw, nw = w["t2p"]["correct"], w["t2p"]["total"]
            ko, no = o["t2p"]["correct"], o["t2p"]["total"]
            kcurve[f"k{k}"] = {
                "with_S": w, "without_S": o,
                "t2p_acc": w["t2p"]["acc"],
                "t2p_counts": f"{kw}/{nw}", "t2p_wilson95": wilson(kw, nw),
                "without_t2p_counts": f"{ko}/{no}",
                "delta_t2p": round(w["t2p"]["acc"] - o["t2p"]["acc"], 4)}
            print(f"  k={k}  with={w['t2p']['acc']:.4f} ({kw}/{nw})  "
                  f"without={o['t2p']['acc']:.4f} ({ko}/{no})  "
                  f"delta={kcurve[f'k{k}']['delta_t2p']:+.4f}", flush=True)
            save()
        results["k_curve"] = kcurve
        save()

        # ---- 2. early readout (full evolution) ----
        ro = eval_readout_curve(model, tok, eval_data, film_layers,
                                state_module, max_tokens)
        results["early_readout"] = {
            k: {"t2p_acc": v["t2p"]["acc"], "t2p_counts":
                f"{v['t2p']['correct']}/{v['t2p']['total']}"}
            for k, v in ro.items()}
        print("  early-readout t2p: " +
              ", ".join(f"{k}={v['t2p']['acc']:.4f} ({v['t2p']['correct']}/"
                        f"{v['t2p']['total']})" for k, v in ro.items()), flush=True)
        save()

    # ---- 3. interventions (sequence only) ----
    if cfg["state_type"] == "vector":
        print("  (vector config: slot interventions not applicable; "
              "existing vector interventions live in _interventions_*.py)",
              flush=True)
        results["interventions"] = {"note": "vector: not applicable"}
    elif not args.skip_ablations:
        K, d = cfg["num_slots"], cfg["slot_dim"]
        interv = {}
        results["interventions"] = interv   # attach early: cond() -> save()
                                            # persists every condition as it lands
        base_rng = random.Random(20260816)

        def cond(name, mode, **kw):
            # CPU generator: deterministic rand/randn sampling below (CUDA
            # generators are not accepted by torch.randn on CPU tensors).
            gen = torch.Generator().manual_seed(
                base_rng.randrange(1 << 30))
            fn, no_upd = make_intervene(mode, K, d, gen=gen, **kw)
            r = eval_cond(model, tok, eval_data, film_layers, state_module,
                          max_tokens, intervene=fn, no_update=no_upd,
                          label=name, progress=50)
            k, n = r["t2p"]["correct"], r["t2p"]["total"]
            interv[name] = {"t2p_acc": r["t2p"]["acc"],
                            "t2p_counts": f"{k}/{n}",
                            "t2p_wilson95": wilson(k, n), "all": r["all"]}
            print(f"  {name:16s} t2p={r['t2p']['acc']:.4f} ({k}/{n})  "
                  f"W95{interv[name]['t2p_wilson95']}", flush=True)
            save()

        cond("normal", "normal")
        cond("zero_slots", "zero_slots")
        cond("random_slots", "random_slots")
        cond("static_trained", "static_trained")
        perms = []
        for pi in range(3):
            perm = list(range(K))
            base_rng.shuffle(perm)
            perms.append(perm)
            cond(f"permute_{pi+1}", "permute", perm=perm)
        for j in range(K):
            cond(f"single_slot_{j+1}", "single_slot", slot=j)
        for p in (0.25, 0.5, 0.75):
            for si in range(3):
                cond(f"dropout_p{p}_s{si+1}", "dropout", p=p)
        interv["_perms"] = perms
        save()

    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nDONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
