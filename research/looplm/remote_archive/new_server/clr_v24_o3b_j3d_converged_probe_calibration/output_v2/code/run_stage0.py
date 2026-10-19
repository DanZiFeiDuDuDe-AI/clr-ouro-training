from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration_v2")
CODE = OUT / "code"


def run(script: Path, log_path: Path) -> int:
    with log_path.open("wb") as log:
        return subprocess.run([sys.executable, str(script)], cwd=str(OUT),
                              stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def main() -> int:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    integrity_code = run(CODE / "stage0_integrity.py", OUT / "logs" / "stage0_integrity.log")
    integrity_file = OUT / "cache_integrity_recheck.json"
    if integrity_code != 0 or not integrity_file.is_file() or not json.loads(integrity_file.read_text()).get("pass"):
        print(json.dumps({"stage0": "FAIL", "integrity_exit_code": integrity_code}), flush=True)
        return 2
    transform_code = run(CODE / "stage0_transform_audit.py", OUT / "logs" / "stage0_transform_audit.log")
    transform_file = OUT / "sanity" / "stage0_transform_audit.json"
    passed = transform_file.is_file() and json.loads(transform_file.read_text()).get("pass")
    print(json.dumps({"stage0": "PASS" if transform_code == 0 and passed else "FAIL",
                      "integrity_exit_code": integrity_code, "transform_exit_code": transform_code,
                      "transform_pass": bool(passed)}), flush=True)
    return 0 if transform_code == 0 and passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
