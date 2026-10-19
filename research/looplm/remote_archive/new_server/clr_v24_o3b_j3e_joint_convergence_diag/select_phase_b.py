from __future__ import annotations

import csv
import json
from pathlib import Path


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag")


def main() -> int:
    with (OUT / "optimizer_search.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["diverged"].lower() == "false"]
    for r in rows:
        for k in ("lr", "weight_decay", "validation_ce_last2_mean_R1_R2", "validation_accuracy_last2_mean_R1_R2"):
            r[k] = float(r[k])
    best_a = min((r for r in rows if r["optimizer"] == "AdamW"), key=lambda r: r["validation_ce_last2_mean_R1_R2"])
    best_s = min((r for r in rows if r["optimizer"] == "SGD_momentum_0.9"), key=lambda r: r["validation_ce_last2_mean_R1_R2"])
    contrast = [r for r in rows if r["optimizer"] == "AdamW" and r["lr"] == best_a["lr"] and r["weight_decay"] != best_a["weight_decay"]]
    if best_a["weight_decay"] == 0.0:
        contrast_row = max(contrast, key=lambda r: r["weight_decay"])
    else:
        contrast_row = next((r for r in contrast if r["weight_decay"] == 0.0), min(contrast, key=lambda r: r["weight_decay"]))
    candidates = []
    for row in (best_a, best_s, contrast_row):
        candidates.append({k: row[k] for k in ("optimizer", "lr", "weight_decay",
                                                "validation_ce_last2_mean_R1_R2",
                                                "validation_accuracy_last2_mean_R1_R2")})
    unique = {(r["optimizer"], r["lr"], r["weight_decay"]): r for r in candidates}
    result = {"selection_is_train_validation_only": True, "test_metrics_read": False,
              "screening_steps": 250,
              "selection_rule": "retain best AdamW and best SGD by R1/R2 mean validation CE; add a same-LR AdamW weight-decay contrast, using the strongest screened nonzero decay if the best AdamW is wd=0",
              "phase_b_candidates": list(unique.values())}
    (OUT / "configs" / "phase_b_candidates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
