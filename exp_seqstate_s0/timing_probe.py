#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick timing probe: real per-trajectory cost of turn_forward_v1 (eval +
one training-style step) on this machine, so the S0 pipeline can be sized."""
import os, random, sys, time
import torch
import torch.nn.functional as F

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = "D:/LoopLM"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clutrr_data import generate_dataset
from clr_ouro import load_ouro, inject_film, turn_forward_v1
from state_modules import VectorStateModule, SequenceStateModule

MAX_TOKENS = 128

def main():
    torch.manual_seed(42); random.seed(42)
    trajs = generate_dataset(6)
    model, tok = load_ouro()
    film_layers = inject_film(model)
    for name, sm in [("vector", VectorStateModule()),
                     ("seq_4x16", SequenceStateModule(4, 16))]:
        t0 = time.time()
        with torch.no_grad():
            for traj in trajs:
                S = sm.init_state(1, "cuda")
                for turn in traj["turns"]:
                    inp = tok(turn["prompt"], return_tensors="pt",
                              truncation=True, max_length=MAX_TOKENS).to("cuda")
                    t1 = time.time()
                    _, S, _ = turn_forward_v1(model, inp, film_layers, None,
                                              S, state_module=sm)
                    torch.cuda.synchronize()
                    print(f"  [{name}] turn fwd {time.time()-t1:.1f}s", flush=True)
                torch.cuda.empty_cache()
        print(f"[{name}] 6-trajectory eval pass: {time.time()-t0:.1f}s total",
              flush=True)

    # one training-style step (vector) with backward
    sm = VectorStateModule()
    opt = torch.optim.AdamW([p for p in sm.parameters()], lr=3e-4)
    traj = trajs[0]
    S = sm.init_state(1, "cuda")
    t0 = time.time()
    for turn in traj["turns"]:
        inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                  max_length=MAX_TOKENS).to("cuda")
        logits, S_new, _ = turn_forward_v1(model, inp, film_layers, None, S,
                                           state_module=sm)
        ans = tok(turn["answer"], add_special_tokens=False)
        tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
        loss = F.cross_entropy(logits.unsqueeze(0),
                               torch.tensor([tgt], device="cuda"))
        loss.backward()
        S = S_new.detach()
    opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    print(f"[vector] 1 training step ({len(traj['turns'])} turns, fwd+bwd): "
          f"{time.time()-t0:.1f}s", flush=True)
    print("PROBE DONE", flush=True)

if __name__ == "__main__":
    main()
