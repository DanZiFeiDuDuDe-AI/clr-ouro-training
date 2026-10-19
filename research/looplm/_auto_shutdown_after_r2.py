# -*- coding: utf-8 -*-
"""
Watch for auto_gammafix_r2 completion, fetch results to local, then shut down cheems.

Trigger: local log contains "MATCHED EVAL (rc=0)" and the launcher has exited.
Actions:
  1. fetch r2 result files from cheems to local
  2. wait until no remote python is running
  3. issue Windows shutdown on cheems (shutdown /s /t 30)

Log: D:/LoopLM/tmp/auto_shutdown.log
"""
import os, sys, time, socket
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "cheems"
HOST_IP = "100.104.57.18"
USER = "ai training"
PASSWORD = __import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")
WATCH_LOG = "D:/LoopLM/tmp/auto_gammafix_r2.log"
LOCAL_DIR = "D:/LoopLM/tmp/ckpt_inspect_r2"
SHUTDOWN_LOG = "D:/LoopLM/tmp/auto_shutdown.log"
SHUTDOWN_FLAG = "D:/LoopLM/tmp/auto_shutdown.done"

REMOTE_FILES = {
    "matched_eval_results_r2.json": "D:/training/matched_eval_results_r2.json",
    "p0h_post_distill_eval_gammafix_r2_gamma.json": "D:/training/p0h_post_distill_eval_gammafix_r2_gamma.json",
    "p0h_post_distill_eval_gammafix_r2_norm.json": "D:/training/p0h_post_distill_eval_gammafix_r2_norm.json",
    "gammafix_norm_r2_diag.json": "D:/training/checkpoints/ouro_retention_m2_gammafix_norm_r2/gamma_diagnostics.json",
    "gammafix_r2_gamma_best.pt": "D:/training/checkpoints/ouro_retention_m2_gammafix_r2/best_model.pt",
    "gammafix_norm_r2_best.pt": "D:/training/checkpoints/ouro_retention_m2_gammafix_norm_r2/best_model.pt",
}

log_fh = None

def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if log_fh is not None:
        log_fh.write(line + "\n")
        log_fh.flush()


def connect_ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last_err = None
    for h in (HOST, HOST_IP):
        try:
            c.connect(h, username=USER, password=PASSWORD, timeout=30,
                      banner_timeout=30, auth_timeout=30)
            return c
        except Exception as e:
            last_err = e
            continue
    raise last_err


def host_reachable(host, timeout=8):
    try:
        s = socket.create_connection((host, 22), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def resolve_host():
    try:
        socket.getaddrinfo(HOST, 22)
        return HOST
    except Exception:
        return HOST_IP


def remote_python_running(c):
    try:
        stdin, stdout, stderr = c.exec_command(
            'tasklist /FI "IMAGENAME eq python.exe" /FO CSV', timeout=30)
        out = stdout.read().decode("utf-8", "replace").strip()
        stdout.channel.recv_exit_status()
        return "python.exe" in out
    except Exception as e:
        log("python_running check failed: %s" % str(e))
        return True


def main():
    global log_fh
    os.makedirs(LOCAL_DIR, exist_ok=True)
    log_fh = open(SHUTDOWN_LOG, "a", encoding="utf-8")
    log("=== auto_shutdown watcher started ===")

    # 1. wait for r2 launcher success
    watch_start = time.time()
    while True:
        if os.path.exists(SHUTDOWN_FLAG):
            log("already shutdown/done flag exists; exiting")
            break
        try:
            with open(WATCH_LOG, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            content = ""
        if "MATCHED EVAL (rc=0)" in content and "=== auto_gammafix_r2 exiting ===" in content:
            log("r2 queue completed successfully")
            break
        if time.time() - watch_start > 6 * 3600:
            log("watch timeout; not shutting down")
            return
        time.sleep(20)

    # 2. fetch results
    h = resolve_host()
    while not host_reachable(h):
        log("cheems not reachable for result fetch; retrying in 20s")
        time.sleep(20)
    try:
        c = connect_ssh()
        sftp = c.open_sftp()
        for name, rp in REMOTE_FILES.items():
            lp = os.path.join(LOCAL_DIR, name)
            try:
                sftp.get(rp, lp)
                log("fetched %s" % name)
            except Exception as e:
                log("fetch failed %s: %s" % (name, str(e)))
        sftp.close()
        c.close()
    except Exception as e:
        log("result fetch connection error: %s" % str(e))

    # 3. wait until no remote python running
    while True:
        h = resolve_host()
        if not host_reachable(h):
            log("cheems not reachable while waiting idle; retrying in 20s")
            time.sleep(20)
            continue
        try:
            c = connect_ssh()
            running = remote_python_running(c)
            c.close()
        except Exception as e:
            log("idle check failed: %s" % str(e))
            running = True
        if not running:
            break
        log("remote python still running; waiting 20s")
        time.sleep(20)

    # 4. issue shutdown on cheems
    log("issuing shutdown /s /t 30 on cheems")
    try:
        c = connect_ssh()
        stdin, stdout, stderr = c.exec_command("shutdown /s /t 30 /c \"LoopLM r2 experiments complete\"", timeout=60)
        out = stdout.read().decode("utf-8", "replace").strip()
        err = stderr.read().decode("utf-8", "replace").strip()
        rc = stdout.channel.recv_exit_status()
        c.close()
        log("shutdown command rc=%d out=%r err=%r" % (rc, out, err))
        if rc == 0:
            with open(SHUTDOWN_FLAG, "w") as f:
                f.write("shutdown issued at %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        log("shutdown command failed: %s" % str(e))

    log("=== auto_shutdown watcher exiting ===")
    if log_fh is not None:
        log_fh.close()


if __name__ == "__main__":
    main()
