#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_seqstate_s0 / verify_baseline.py — safety gate for the clr_ouro.py edits.

Check 1 (repro): load checkpoints/clr_ouro_v1_s42/best_model.pt, run the v1
eval protocol (per-loop update, argmax readout, 150 trajectories) through the
GENERIC forward path (VectorStateModule). Must reproduce the published
with_S t2p = 76/150 within the ±2 init-noise jitter tolerance, and
without_S must stay at 0.

Check 2 (path equivalence): on a 40-trajectory subset, with the torch RNG
re-seeded identically, the legacy path (plaa_core, state_module=None) and
the generic path (VectorStateModule) must produce IDENTICAL bucket counts.
This proves the clr_ouro.py generalization is behavior-preserving for all
existing callers.

Exit code 0 only if both checks pass.
"""
import os
import random
import sys

import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = os.environ.get("OURO_ROOT", "D:/LoopLM")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clutrr_data import generate_dataset          # noqa: E402
from clr_ouro import (load_ouro, inject_film,     # noqa: E402
                      turn_forward_v1, PlaaCore)
from state_modules import VectorStateModule       # noqa: E402

CKPT = os.environ.get("OURO_CKPT",
        "D:/LoopLM/checkpoints/clr_ouro_v1_s42/best_model.pt")
EXPECTED = 76          # published v1_s42 with_S t2p correct/150
TOL = 2                # init-noise jitter tolerance (same as interventions)
N_REPRO = 150
N_EQUIV = 40
MAX_TOKENS = 128


@torch.no_grad()
def eval_path(model, tok, trajs, film_layers, sm, legacy, ablate_s=False,
              label="eval", progress=25):
    """v1 eval; legacy=True -> state_module=None + PlaaCore."""
    import time as _t
    t0 = _t.time()
    correct = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    for ti_, traj in enumerate(trajs):
        if progress and ti_ % progress == 0 and ti_ > 0:
            print(f"    [{label}] {ti_}/{len(trajs)} traj, "
                  f"{_t.time()-t0:.0f}s elapsed", flush=True)
        S = sm.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            if legacy:
                logits, S_new, _ = turn_forward_v1(model, inp, film_layers,
                                                   sm, S, ablate_s=ablate_s)
            else:
                logits, S_new, _ = turn_forward_v1(
                    model, inp, film_layers, None, S, state_module=sm,
                    ablate_s=ablate_s)
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
    return {"t1": correct[0], "t2p": correct[1], "last": correct[2],
            "all": correct[3],
            "totals": (total[0], total[1], total[2], total[3])}


def main():
    torch.manual_seed(42)
    random.seed(42)
    _train = generate_dataset(600)          # replay RNG stream
    eval_data = generate_dataset(150)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    sm = VectorStateModule()
    try:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    except Exception:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    sm.load_state_dict(ckpt["plaa_core"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    print(f"ckpt step {ckpt.get('step')} variant {ckpt.get('variant')}",
          flush=True)

    # ---- check 1: repro on 150 via the generic path ----
    torch.manual_seed(4242)
    w = eval_path(model, tok, eval_data, film_layers, sm, legacy=False,
                  label="check1_withS_generic")
    torch.manual_seed(4242)
    w2 = eval_path(model, tok, eval_data, film_layers, sm, legacy=True,
                   label="check1_withS_legacy")
    torch.manual_seed(4243)
    o = eval_path(model, tok, eval_data, film_layers, sm, legacy=False,
                  ablate_s=True, label="check1_withoutS")
    print(f"repro generic: with_S t2p={w['t2p']}/{w['totals'][1]}  "
          f"without_S={o['t2p']}/{o['totals'][1]}", flush=True)
    print(f"repro legacy : with_S t2p={w2['t2p']}/{w2['totals'][1]}",
          flush=True)
    assert abs(w["t2p"] - EXPECTED) <= TOL, \
        f"repro FAILED: generic with_S t2p={w['t2p']} vs expected {EXPECTED}"
    assert o["t2p"] == 0, f"repro FAILED: without_S t2p={o['t2p']} != 0"
    assert w["t2p"] == w2["t2p"], "generic vs legacy disagree on t2p (150)"
    print(f"check 1 OK: with_S {w['t2p']}/{w['totals'][1]} reproduces "
          f"{EXPECTED}±{TOL}, without_S 0", flush=True)

    # ---- check 2: exact path equivalence on a subset, same init draws ----
    sub = eval_data[:N_EQUIV]
    torch.manual_seed(777)
    a = eval_path(model, tok, sub, film_layers, sm, legacy=False)
    torch.manual_seed(777)
    b = eval_path(model, tok, sub, film_layers, sm, legacy=True)
    print(f"path equiv: generic {a} vs legacy {b}", flush=True)
    assert a == b, f"path equivalence FAILED: {a} vs {b}"
    print("check 2 OK: generic path == legacy path bucket-for-bucket",
          flush=True)

    print("\nBASELINE VERIFICATION PASSED", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "verify_baseline.passed"), "w") as f:
        f.write(f"generic with_S={w['t2p']}/{w['totals'][1]} "
                f"without_S={o['t2p']}/{o['totals'][1]}\n")
    print("marker written: verify_baseline.passed", flush=True)


if __name__ == "__main__":
    main()
