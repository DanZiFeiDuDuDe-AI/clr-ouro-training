# -*- coding: utf-8 -*-
"""Cache M0 teacher baseline (result + per-turn logits) for multi-seed eval.
Same protocol as p0i_pilot_eval.py (N=150 held-out, seed 42).
Writes: p0i_m1_teacher_vecs.npy + p0i_m1_teacher_result.json
"""
import os, sys, json, random
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import numpy as np
import torch
from clutrr_data import generate_dataset
from p0i_pilot_eval import build_teacher, evaluate_model, SEED, N_EVAL

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
_ = generate_dataset(600)
data = generate_dataset(N_EVAL)[:N_EVAL]

teacher, tok = build_teacher()
t_res, t_ranks, t_margins, t_vecs = evaluate_model(
    teacher, [], tok, data, persist=False)
np.save("D:/training/checkpoints/p0i_m1_teacher_vecs.npy", np.array(t_vecs))
json.dump(t_res, open("D:/training/checkpoints/p0i_m1_teacher_result.json", "w"), indent=2)
print("M0: acc=%d/%d mean_rank=%.2f margin=%.4f vecs=%d" % (
    t_res["candidate_accuracy"], t_res["n"], t_res["mean_rank"],
    t_res["mean_margin"], len(t_vecs)), flush=True)
print("TEACHER CACHE OK", flush=True)
