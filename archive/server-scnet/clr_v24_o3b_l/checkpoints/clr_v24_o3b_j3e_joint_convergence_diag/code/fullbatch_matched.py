from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ["LD_LIBRARY_PATH"] = "/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import j3e_runner_v2 as j3e


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag")


def main() -> int:
    frozen = json.loads((OUT / "j3e_probe_protocol_frozen.json").read_text(encoding="utf-8"))
    cfg = {k: frozen[k] for k in ("optimizer", "lr", "weight_decay")}
    summary, curves, _ = j3e.train_config(cfg, 5000, 100, [11], 1024,
                                           "full_batch_matched_protocol", full_batch=True)
    p = OUT / "full_batch_control.csv"
    existing = []
    if p.exists():
        with p.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    j3e.write_csv(p, existing + summary)
    j3e.write_csv(OUT / "full_batch_control_curves.csv", curves)
    print(json.dumps({"matched_full_batch_complete": True, "protocol": cfg,
                      "runs": len(summary), "plateaus": sum(bool(r["plateau_confirmed"]) for r in summary)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
