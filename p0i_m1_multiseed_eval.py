# -*- coding: utf-8 -*-
"""Multi-seed M1-reset evaluation.

Same canonical eval protocol as p0i_pilot_eval.py (N=150 held-out, seed 42,
same M0 teacher baseline, same metric computation), but evaluates m1-reset
checkpoints from seeds 43/44/45 and aggregates mean/std/min/max.

Writes: D:/training/checkpoints/P0I_M1_MULTISEED_EVAL.json
"""
import os, sys, json, random, statistics
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import numpy as np
import torch
from clutrr_data import generate_dataset
from p0i_pilot_eval import (build_teacher, build_student, evaluate_model,
                            spearman, SEED, N_EVAL)

SEEDS = [42, 43, 44]
BASE = "D:/training/checkpoints/p0i_pilot_m1_reset_s%d/best_model.pt"


def main():
    print("=== P0I M1 multi-seed eval ===", flush=True)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    _ = generate_dataset(600)
    data = generate_dataset(N_EVAL)[:N_EVAL]
    print("eval trajectories: %d" % len(data), flush=True)

    # Teacher baseline (same protocol as p0i_pilot_eval.py)
    teacher, tok = build_teacher()
    t_res, t_ranks, t_margins, t_vecs = evaluate_model(
        teacher, [], tok, data, persist=False)
    print("M0: acc=%d/%d mean_rank=%.2f margin=%.4f" % (
        t_res["candidate_accuracy"], t_res["n"],
        t_res["mean_rank"], t_res["mean_margin"]), flush=True)
    del teacher; torch.cuda.empty_cache()

    results = {"n_eval": N_EVAL, "protocol_seed": SEED,
               "M0_teacher": t_res, "runs": {}}
    rows = []
    for seed in SEEDS:
        ckpt = BASE % seed
        if not os.path.exists(ckpt):
            print("MISSING %s — skip" % ckpt, flush=True)
            results["runs"][str(seed)] = {"missing": True}
            continue
        tlog = os.path.join(os.path.dirname(ckpt), "train_log.json")
        final_loss = None
        if os.path.exists(tlog):
            final_loss = json.load(open(tlog))["final_loss"]
        model, _, adapters = build_student("m1", ckpt)
        r, ranks, margins, s_vecs = evaluate_model(
            model, adapters, tok, data, persist=False)
        r["spearman_vs_M0"] = spearman(np.array(t_vecs), np.array(s_vecs))
        r["final_loss"] = final_loss
        results["runs"][str(seed)] = r
        print("s%d: acc=%d/%d mean_rank=%.2f margin=%.4f entropy=%.4f "
              "spearman=%.4f final_loss=%s" % (
                  seed, r["candidate_accuracy"], r["n"], r["mean_rank"],
                  r["mean_margin"], r["entropy"], r["spearman_vs_M0"],
                  "%.4f" % final_loss if final_loss is not None else "n/a"),
              flush=True)
        rows.append(r)
        del model; torch.cuda.empty_cache()

    if rows:
        def agg(key):
            vals = [r[key] for r in rows if key in r]
            if not vals:
                return None
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals))}
        summary = {
            "candidate_accuracy_frac": agg("candidate_accuracy_frac"),
            "mean_rank": agg("mean_rank"),
            "mean_margin": agg("mean_margin"),
            "entropy": agg("entropy"),
            "spearman_vs_M0": agg("spearman_vs_M0"),
            "final_loss": agg("final_loss"),
        }
        results["summary"] = summary
        print("\n=== SUMMARY (seeds 43/44/45) ===", flush=True)
        for k, v in summary.items():
            print("%s: %s" % (k, json.dumps(v)), flush=True)

    import pathlib
    out_json = "D:/training/checkpoints/P0I_M1_MULTISEED_EVAL.json"
    pathlib.Path(out_json).write_text(json.dumps(results, indent=2))
    print("wrote %s" % out_json, flush=True)
    print("P0I M1 MULTISEED EVAL OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
