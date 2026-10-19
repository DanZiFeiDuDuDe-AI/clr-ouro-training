from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: launch_j3e_v2.py <search|main|controls|final>")
    stage = sys.argv[1]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    log = (OUT / "logs" / f"{stage}.log").open("ab", buffering=0)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [sys.executable, "-u", str(OUT / "code" / "j3e_runner_v2.py"), stage]
    p = subprocess.Popen(cmd, cwd=OUT, env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True)
    (OUT / "logs" / f"{stage}.pid").write_text(f"{p.pid}\n", encoding="utf-8")
    print(f"launched stage={stage} pid={p.pid} log={OUT / 'logs' / f'{stage}.log'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
