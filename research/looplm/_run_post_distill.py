"""Upload post_distill_eval.py to cheems and run it, streaming output."""
import sys, warnings, time
import paramiko

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOCAL = "D:/LoopLM/post_distill_eval.py"
REMOTE = "D:/training/post_distill_eval.py"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("cheems", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=20, banner_timeout=30, auth_timeout=30)
try:
    sftp = c.open_sftp()
    sftp.put(LOCAL, REMOTE)
    sftp.close()
    print("uploaded %s -> %s" % (LOCAL, REMOTE), flush=True)

    cmd = "python -u D:/training/post_distill_eval.py"
    print("running: %s" % cmd, flush=True)
    t0 = time.time()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=1800)
    # read line by line to keep local output timely
    out_lines = []
    while True:
        line = stdout.readline()
        if not line:
            break
        s = line.rstrip("\n")
        print(s, flush=True)
        out_lines.append(s)
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if err:
        print("[stderr]\n" + err, flush=True)
    print("[rc=%d elapsed=%.0fs]" % (rc, time.time() - t0), flush=True)
finally:
    c.close()
