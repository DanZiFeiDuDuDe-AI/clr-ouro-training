from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration")


def main() -> int:
    marker = {
        "status": "INVALIDATED_NOT_FOR_INFERENCE",
        "experiment": "O3b-J3D initial harness run",
        "reason": "Grouped probe training failed to freeze each model at its own early-stop checkpoint; some best_step values exceeded stop_step.",
        "action": "Process 290 was terminated before held-out TestA/TestB performance evaluation. Partial outputs are preserved for audit only.",
        "optimizer_screen_completed": True,
        "main_train_validation_fits_started": True,
        "heldout_performance_metrics_read": False,
        "test_identity_metadata_was_checked_for_stage0_integrity": True,
        "model_or_checkpoint_modified": False,
        "invalidated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "early_stop_violation": "tables/r0_r1_converged_results.csv contains examples with best_step later than stop_step",
            "run_log": "logs/j3d_pipeline_run.log",
            "pid": 290,
        },
        "replacement_run": "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration_v2",
    }
    path = OUT / "INVALIDATED_RUN.json"
    path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"marker": str(path), "status": marker["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
