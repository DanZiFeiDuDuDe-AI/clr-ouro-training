from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration")
PIPELINE = OUT / "code" / "j3d_pipeline_calibrated_v2.py"
LOG = OUT / "logs" / "j3d_pipeline_run.log"
PIDFILE = OUT / "logs" / "j3d_pipeline.pid"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("logs", "sanity", "tables", "plots", "report", "configs"):
        (OUT / folder).mkdir(exist_ok=True)
    for path in (PIPELINE, OUT / "cache_integrity_recheck.json", OUT / "sanity" / "stage0_transform_audit.json",
                 OUT / "configs" / "protocol_selection_preregistered.json",
                 OUT / "configs" / "j_cliff_interpretation_preregistered.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not json.loads((OUT / "cache_integrity_recheck.json").read_text()).get("pass"):
        raise RuntimeError("certified cache integrity gate is not PASS")
    if not json.loads((OUT / "sanity" / "stage0_transform_audit.json").read_text()).get("pass"):
        raise RuntimeError("Stage 0 transform audit gate is not PASS")
    if (OUT / "FINAL_VERDICT.json").exists() or (OUT / "probe_protocol_frozen.json").exists():
        raise RuntimeError("output already contains a pipeline run; refusing to overwrite existing experiment state")
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())
            os.kill(old_pid, 0)
        except (ValueError, ProcessLookupError, FileNotFoundError):
            PIDFILE.unlink()
        else:
            raise RuntimeError(f"existing J3D process appears active: pid {old_pid}")

    env = os.environ.copy()
    library_dirs = ["/opt/dtk-26.04/hip/lib", "/opt/dtk-26.04/lib", "/opt/dtk-26.04/.hyhal/rocm_smi/lib"]
    env["LD_LIBRARY_PATH"] = ":".join(library_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    started = datetime.now(timezone.utc).isoformat()
    with LOG.open("ab", buffering=0) as log_file:
        proc = subprocess.Popen([sys.executable, str(PIPELINE)], cwd=str(OUT), env=env,
                                stdout=log_file, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    PIDFILE.write_text(f"{proc.pid}\n")
    manifest = {"pid": proc.pid, "started_utc": started, "python": sys.executable,
                "python_version": sys.version, "pipeline": str(PIPELINE),
                "pipeline_sha256": sha(PIPELINE), "stdout_stderr_log": str(LOG),
                "cache_integrity_stage0_pass": True, "row_transform_stage0_pass": True,
                "ld_library_path": env["LD_LIBRARY_PATH"], "checkpoint_modified": False}
    (OUT / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"launched": True, "pid": proc.pid, "started_utc": started,
                      "pipeline_sha256": manifest["pipeline_sha256"], "log": str(LOG)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
