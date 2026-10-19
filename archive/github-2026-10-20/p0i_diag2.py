# -*- coding: utf-8 -*-
"""Reproduce EXACT training protocol data and time the FIRST trajectory turn-by-turn."""
import os, sys, json, random, time
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")
import numpy as np
import torch
import torch.nn.functional as F
from clutrr_data import generate_dataset, RELATIONS
import p0i_pilot_train as T

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def main():
    model, tok, adapters, train_params = T.build_model("m1")
    opt = torch.optim.Adam(train_params, lr=3e-4)
    cands = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad = tok.pad_token_id
    # EXACT training protocol
    _ = generate_dataset(600)
    _ = generate_dataset(150)[:150]
    corpus = generate_dataset(600)[:600]
    print("corpus:", len(corpus), flush=True)
    lens = [len(tok(t["prompt"], truncation=True, max_length=128)["input_ids"])
            for tr in corpus for t in tr["turns"]]
    print("prompt lens: min=%d max=%d mean=%.1f" % (min(lens), max(lens), np.mean(lens)), flush=True)
    for ti, traj in enumerate(corpus[:5]):
        print("traj %d turns=%d lens=%s" % (ti, len(traj["turns"]),
              [len(tok(t["prompt"], truncation=True, max_length=128)["input_ids"]) for t in traj["turns"]]), flush=True)
    # time first trajectory persist
    traj = corpus[0]
    T.clear_all_state(adapters)
    offset = 0
    print("=== persist timing on traj 0 ===", flush=True)
    for turn_i, turn in enumerate(traj["turns"]):
        t0 = time.time()
        if turn_i == 0:
            T.reset_turn_state(adapters)
        else:
            T.set_persist_state(adapters)
        p = tok(turn["prompt"], truncation=True, max_length=128)["input_ids"]
        ci = RELATIONS.index(turn["answer"])
        sc = T.candidate_scores(model, tok, p, cands, pad, offset=offset)
        loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([ci], device=sc.device))
        opt.zero_grad(); loss.backward(); opt.step()
        dt = time.time() - t0
        offset += len(p)
        print("  t%d len=%d loss=%.3f %.1fs" % (turn_i, len(p), loss.item(), dt), flush=True)
        if dt > 120:
            print("  >>> SLOW/HANG at turn %d" % turn_i, flush=True)
            break

if __name__ == "__main__":
    main()
