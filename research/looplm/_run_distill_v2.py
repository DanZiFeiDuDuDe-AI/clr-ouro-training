"""Upload distill_v2.py to cheems and run it, streaming output.

Usage:
  python _run_distill_v2.py smoke          # D_SMOKE=1 D_STEPS=10 foreground
  python _run_distill_v2.py smoke_env ...  # full custom env? use env vars
Set env vars D_STEPS, D_LR, etc. on the LOCAL side before running this script
and they will be passed to the remote command (same names).
"""
import os, sys, warnings, time
import paramiko

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOCAL = "D:/LoopLM/distill_v2.py"
REMOTE = "D:/training/distill_v2.py"

mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("cheems", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=20, banner_timeout=30, auth_timeout=30)
try:
    sftp = c.open_sftp()
    sftp.put(LOCAL, REMOTE)
    sftp.close()
    print("uploaded %s -> %s" % (LOCAL, REMOTE), flush=True)

    env = {}
    if mode == "smoke":
        env["D_SMOKE"] = "1"
        env["D_STEPS"] = "10"
        env["D_OUT_DIR"] = "D:/training/checkpoints/ouro_retention_m2_d2_smoke"
    else:
        env["D_SMOKE"] = "0"
    # inherit optional overrides from local env
    for k in ["D_STEPS", "D_LR", "D_LAM_BLOCK", "D_LAM_LOGIT", "D_OUT_DIR", "D_UNFREEZE_POSTNORM"]:
        if k in os.environ:
            env[k] = os.environ[k]
    # build remote command with cmd set (avoids quoting/backslash issues)
    set_parts = []
    for k, v in env.items():
        set_parts.append("%s=%s" % (k, v))
    cmd = "cmd /c \"" + " && ".join(set_parts + ["python -u D:/training/distill_v2.py"]) + "\""
    print("running: %s" % cmd, flush=True)
    t0 = time.time()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=3600)
    while True:
        line = stdout.readline()
        if not line:
            break
        s = line.rstrip("\n")
        print(s, flush=True)
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if err:
        print("[stderr]\n" + err, flush=True)
    print("[rc=%d elapsed=%.0fs]" % (rc, time.time() - t0), flush=True)
finally:
    c.close()
