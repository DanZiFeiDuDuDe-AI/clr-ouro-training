# -*- coding: utf-8 -*-
"""Diagnose persist hang on REAL CLUTRR data (first trajectories)."""
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
    _ = generate_dataset(600)
    data = generate_dataset(10)[:10]
    print("diag trajectories:", len(data), flush=True)

    for mode in ["reset", "persist"]:
        print("===== MODE=%s =====" % mode, flush=True)
        for ti, traj in enumerate(data):
            T.clear_all_state(adapters)
            offset = 0
            print("traj %d turns=%d" % (ti, len(traj["turns"])), flush=True)
            for turn_i, turn in enumerate(traj["turns"]):
                t0 = time.time()
                if turn_i == 0:
                    T.reset_turn_state(adapters)
                elif mode == "reset":
                    T.reset_turn_state(adapters)
                else:
                    T.set_persist_state(adapters)
                p = tok(turn["prompt"], truncation=True, max_length=128)["input_ids"]
                ci = RELATIONS.index(turn["answer"])
                sc = T.candidate_scores(model, tok, p, cands, pad, offset=offset)
                loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([ci], device=sc.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                dt = time.time() - t0
                offset = offset + len(p) if mode == "persist" else 0
                print("  t%d lens=%d loss=%.3f %.1fs" % (turn_i, len(p), loss.item(), dt), flush=True)
            if ti >= 3:
                break

if __name__ == "__main__":
    main()
