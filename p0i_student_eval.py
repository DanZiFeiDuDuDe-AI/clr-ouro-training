# -*- coding: utf-8 -*-
"""Evaluate ONE m1-reset seed checkpoint using cached M0 teacher logits.
Usage: python p0i_student_eval.py 43
Same eval protocol as p0i_pilot_eval.py. Single model in memory at a time.
Writes: p0i_m1_run_s{seed}.json
"""
import os, sys, json, random
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import numpy as np
import torch
from clutrr_data import generate_dataset
from p0i_pilot_eval import build_student, evaluate_model, spearman, SEED, N_EVAL

seed = int(sys.argv[1])
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
_ = generate_dataset(600)
data = generate_dataset(N_EVAL)[:N_EVAL]

ckpt = "D:/training/checkpoints/p0i_pilot_m1_reset_s%d/best_model.pt" % seed
t_vecs = np.load("D:/training/checkpoints/p0i_m1_teacher_vecs.npy")

model, tok, adapters = build_student("m1", ckpt)
r, ranks, margins, s_vecs = evaluate_model(
    model, adapters, tok, data, persist=False)
r["spearman_vs_M0"] = spearman(t_vecs, np.array(s_vecs))
tlog = "D:/training/checkpoints/p0i_pilot_m1_reset_s%d/train_log.json" % seed
r["final_loss"] = json.load(open(tlog))["final_loss"] if os.path.exists(tlog) else None
out = "D:/training/checkpoints/p0i_m1_run_s%d.json" % seed
json.dump({"seed": seed, **r}, open(out, "w"), indent=2)
print("s%d: acc=%d/%d mean_rank=%.2f margin=%.4f entropy=%.4f "
      "spearman=%.4f final_loss=%s" % (
          seed, r["candidate_accuracy"], r["n"], r["mean_rank"],
          r["mean_margin"], r["entropy"], r["spearman_vs_M0"],
          "%.4f" % r["final_loss"] if r["final_loss"] is not None else "n/a"),
      flush=True)
print("STUDENT EVAL OK seed=%d" % seed, flush=True)
