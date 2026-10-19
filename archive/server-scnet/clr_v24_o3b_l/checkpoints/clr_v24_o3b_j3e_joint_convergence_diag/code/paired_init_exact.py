from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

import j3e_runner_v2 as j3e


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag")


def main() -> int:
    xs, ys, means, stds, xtr, xval, ytr, yval, nclass = j3e.init_cuda_data()
    maps = j3e.make_transform(means, stds)
    seeds = j3e.SEEDS
    W, B = j3e.init_params(len(seeds) * 2, nclass, 256, seeds)
    initial = []
    with torch.no_grad():
        for si, seed in enumerate(seeds):
            w1 = W[si*2].detach().cpu().numpy().astype(np.float64)
            b1 = B[si*2].detach().cpu().numpy().astype(np.float64)
            w2, b2 = j3e.transfer(w1, b1, "R1", "R2", maps)
            W[si*2+1].copy_(torch.from_numpy(w2.astype(np.float32)).to(j3e.DEVICE))
            B[si*2+1].copy_(torch.from_numpy(b2.astype(np.float32)).to(j3e.DEVICE))
            for split, mat, labels in (("train", xs[("R1", "train")], ys["train"]),
                                       ("val", xs[("R1", "val")], ys["val"])):
                mat2 = xs[("R2", split)]
                l1 = mat.astype(np.float64) @ w1.T + b1
                l2 = mat2.astype(np.float64) @ w2.T + b2
                known = (labels >= 0) & (labels < nclass)
                mismatch = float(np.mean(l1[known].argmax(1) != l2[known].argmax(1)))
                initial.append({"seed": seed, "split": split,
                                "initial_max_logit_error": float(np.max(np.abs(l1[known] - l2[known]))),
                                "initial_mean_logit_error": float(np.mean(np.abs(l1[known] - l2[known]))),
                                "initial_prediction_mismatch_rate": mismatch,
                                "initial_R1_accuracy": float(np.mean(l1[known].argmax(1) == labels[known])),
                                "initial_R2_transferred_accuracy": float(np.mean(l2[known].argmax(1) == labels[known])),
                                "known_rows": int(known.sum())})
    candidates = json.loads((OUT / "configs" / "phase_b_candidates.json").read_text(encoding="utf-8"))["phase_b_candidates"]
    cfg = next((r for r in candidates if r["optimizer"] == "AdamW" and r["weight_decay"] == 0.0), candidates[0])
    sm, curves, _ = j3e.train_config({k: cfg[k] for k in ("optimizer", "lr", "weight_decay")},
                                     10000, 200, seeds, 1024, "paired_init_detailed", paired_init=True)
    init_map = {(r["seed"], r["representation"]): r for r in []}
    # Every seed's R2 initialization is the affine transfer of its R1 initialization.
    for row in sm:
        seed, rep = int(row["seed"]), row["representation"]
        row["initialization_relation"] = "paired affine transfer; equal initial logits by construction"
        row["initial_prediction_mismatch_rate_train"] = next(r["initial_prediction_mismatch_rate"] for r in initial if r["seed"] == seed and r["split"] == "train")
        row["initial_prediction_mismatch_rate_val"] = next(r["initial_prediction_mismatch_rate"] for r in initial if r["seed"] == seed and r["split"] == "val")
        row["initial_max_logit_error_train"] = next(r["initial_max_logit_error"] for r in initial if r["seed"] == seed and r["split"] == "train")
        row["initial_max_logit_error_val"] = next(r["initial_max_logit_error"] for r in initial if r["seed"] == seed and r["split"] == "val")
        row["representation"] = rep
    j3e.write_csv(OUT / "paired_initialization_control.csv", sm)
    j3e.write_csv(OUT / "paired_initialization_initial_equivalence.csv", initial)
    j3e.write_csv(OUT / "paired_initialization_curves.csv", curves)
    print(json.dumps({"paired_init_control_complete": True, "runs": len(sm),
                      "initial_max_logit_error": max(r["initial_max_logit_error"] for r in initial),
                      "initial_prediction_mismatch_max": max(r["initial_prediction_mismatch_rate"] for r in initial),
                      "plateau_count": sum(bool(r["plateau_confirmed"]) for r in sm)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
