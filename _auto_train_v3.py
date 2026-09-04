# -*- coding: utf-8 -*-
"""
Auto-launcher for Phase 0H step2c distillation v3 (continue from M2-v2).

Runs LOCALLY as a detached process:
  1. wait until cheems is reachable on TCP/22
  2. upload distill_v3.py and post_distill_eval_v2.py
  3. 10-step smoke test (v3)
  4. full D_STEPS training (default 1200) from M2-v2
  5. post-distill eval v2 on the new checkpoint (m2_d3)

Stop file: D:/LoopLM/tmp/auto_train_v3.stop
Log file:  D:/LoopLM/tmp/auto_train_v3.log
"""
import os, sys, json, time, socket
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "cheems"
USER = "ai training"
PASSWORD = "REPLACE_ME"
LOCAL_DISTILL = "D:/LoopLM/distill_v3.py"
LOCAL_EVAL = "D:/LoopLM/post_distill_eval_v2.py"
REMOTE_DISTILL = "D:/training/distill_v3.py"
REMOTE_EVAL = "D:/training/post_distill_eval_v2.py"
SMOKE_OUT = "D:/training/checkpoints/ouro_retention_m2_d3_smoke"
FULL_OUT = "D:/training/checkpoints/ouro_retention_m2_d3"
FULL_CKPT = FULL_OUT + "/best_model.pt"
STOP_FILE = "D:/LoopLM/tmp/auto_train_v3.stop"
LOG_PATH = "D:/LoopLM/tmp/auto_train_v3.log"
FULL_STEPS = int(os.environ.get("AUTO_D_STEPS", "1200"))
FULL_LR = os.environ.get("AUTO_D_LR", "3e-4")
WAIT_START = time.time()
MAX_WAIT_SECONDS = 6 * 3600
POLL_SECONDS = 20

log_fh = None

def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if log_fh is not None:
        log_fh.write(line + "\n")
        log_fh.flush()


def stop_requested():
    return os.path.exists(STOP_FILE)


def host_reachable(timeout=8):
    try:
        s = socket.create_connection((HOST, 22), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def connect_ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30,
              banner_timeout=30, auth_timeout=30)
    return c


def upload_file(c, local, remote):
    sftp = c.open_sftp()
    try:
        sftp.put(local, remote)
        log("uploaded %s -> %s" % (local, remote))
    finally:
        sftp.close()


def run_remote(c, cmd, timeout=7200):
    log("running remote: %s" % cmd)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    stdout.channel.settimeout(120)
    tail = []
    t0 = time.time()
    rc = None
    while True:
        if stop_requested():
            log("STOP file detected; aborting remote command")
            return -9, tail
        try:
            line = stdout.readline()
        except socket.timeout:
            if stdout.channel.closed:
                break
            log("(no output for 120s, still running, elapsed %.0fs)" % (time.time() - t0))
            continue
        except Exception as e:
            log("stdout read error: %s" % str(e))
            break
        if not line:
            break
        s = line.rstrip("\n")
        log("REMOTE| " + s)
        tail.append(s)
        if len(tail) > 300:
            tail = tail[-300:]
    try:
        rc = stdout.channel.recv_exit_status()
    except Exception as e:
        log("recv_exit_status error: %s" % str(e))
    err = ""
    try:
        err = stderr.read().decode("utf-8", "replace")
    except Exception:
        pass
    if err:
        log("REMOTE_STDERR| " + err[:4000])
    log("remote command finished rc=%s elapsed=%.0fs" % (rc, time.time() - t0))
    return rc, tail


def remote_status_checks(c):
    cmds = [
        'tasklist /FI "IMAGENAME eq python.exe" /FO CSV',
        'if exist D:\\training\\checkpoints\\ouro_retention_m2_d3\\best_model.pt (echo FULL_CKPT_EXISTS) else (echo NO_FULL_CKPT)',
        'if exist D:\\training\\checkpoints\\ouro_retention_m2_d3\\distill_log.json (echo FULL_LOG_EXISTS) else (echo NO_FULL_LOG)',
        'if exist D:\\training\\checkpoints\\ouro_retention_m2_d3_smoke\\best_model.pt (echo SMOKE_CKPT_EXISTS) else (echo NO_SMOKE_CKPT)',
    ]
    res = {"python_running": False, "full_complete": False, "smoke_ok": False}
    for cmd in cmds:
        try:
            stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
            out = stdout.read().decode("utf-8", "replace").strip()
            stdout.channel.recv_exit_status()
            if "python.exe" in out:
                res["python_running"] = True
            if "FULL_CKPT_EXISTS" in out or "FULL_LOG_EXISTS" in out:
                res["full_complete"] = True
            if "SMOKE_CKPT_EXISTS" in out:
                res["smoke_ok"] = True
        except Exception as e:
            log("status cmd failed: %s (%s)" % (cmd, str(e)))
    return res


def wait_for_cheems():
    while True:
        if stop_requested():
            return False
        if time.time() - WAIT_START > MAX_WAIT_SECONDS:
            log("max wait time reached, giving up")
            return False
        if host_reachable():
            return True
        log("cheems not reachable (TCP/22). waiting %ds..." % POLL_SECONDS)
        time.sleep(POLL_SECONDS)


def smoke_test():
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_DISTILL, REMOTE_DISTILL)
        cmd = ('cmd /c "set D_SMOKE=1&& set D_STEPS=10&& '
               'set D_OUT_DIR=D:/training/checkpoints/ouro_retention_m2_d3_smoke&& '
               'python -u D:/training/distill_v3.py"')
        rc, tail = run_remote(c, cmd, timeout=1800)
        ok = (rc == 0) and any("PHASE 0H step2c OK" in s for s in tail)
        log("SMOKE %s (rc=%s)" % ("PASSED" if ok else "FAILED", rc))
        return ok
    except Exception as e:
        log("smoke_test connection error: %s" % str(e))
        return False
    finally:
        c.close()


def full_training():
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_DISTILL, REMOTE_DISTILL)
        cmd = ('cmd /c "set D_SMOKE=0&& set D_STEPS=%d&& set D_LR=%s&& '
               'set D_OUT_DIR=D:/training/checkpoints/ouro_retention_m2_d3&& '
               'python -u D:/training/distill_v3.py"' % (FULL_STEPS, FULL_LR))
        rc, tail = run_remote(c, cmd, timeout=7200)
        ok = (rc == 0) and any("PHASE 0H step2c OK" in s for s in tail)
        log("FULL TRAINING %s (rc=%s)" % ("PASSED" if ok else "FAILED", rc))
        return ok
    except Exception as e:
        log("full_training connection error: %s" % str(e))
        return False
    finally:
        c.close()


def run_eval():
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_EVAL, REMOTE_EVAL)
        cmd = ('cmd /c "set H_M2_CKPT=D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt&& '
               'set H_OUT_JSON=D:/training/p0h_post_distill_eval_v3.json&& '
               'python -u D:/training/post_distill_eval_v2.py"')
        rc, tail = run_remote(c, cmd, timeout=1800)
        ok = (rc == 0) and any("PHASE 0H POST-DISTILL EVAL V2 OK" in s for s in tail)
        log("POST-DISTILL EVAL %s (rc=%s)" % ("PASSED" if ok else "FAILED", rc))
        return ok
    except Exception as e:
        log("run_eval connection error: %s" % str(e))
        return False
    finally:
        c.close()


def main():
    global log_fh
    log_fh = open(LOG_PATH, "a", encoding="utf-8")
    log("=== auto_train_v3 started, D_STEPS=%d, D_LR=%s ===" % (FULL_STEPS, FULL_LR))

    while True:
        if stop_requested():
            log("STOP file detected before wait")
            break
        if time.time() - WAIT_START > MAX_WAIT_SECONDS:
            log("max wait time reached")
            break
        if not wait_for_cheems():
            break
        log("cheems reachable")
        try:
            c = connect_ssh()
            st = remote_status_checks(c)
            c.close()
        except Exception as e:
            log("status check failed: %s" % str(e))
            st = None
        if st is not None:
            if st["full_complete"]:
                log("full training already complete on remote; running eval")
                run_eval()
                break
            if st["python_running"]:
                log("remote python.exe is already running; waiting and rechecking")
                time.sleep(POLL_SECONDS)
                continue
        if not smoke_test():
            log("smoke failed; will wait and retry")
            time.sleep(POLL_SECONDS)
            continue
        if full_training():
            log("full training completed; running post-distill eval")
            run_eval()
            break
        else:
            log("full training did not complete; will retry after wait")
            time.sleep(POLL_SECONDS)

    log("=== auto_train_v3 exiting ===")
    if log_fh is not None:
        log_fh.close()


if __name__ == "__main__":
    main()
