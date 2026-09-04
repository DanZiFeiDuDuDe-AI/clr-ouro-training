# -*- coding: utf-8 -*-
"""
Auto-launcher for Phase 0H step2b distillation v2.

Run LOCALLY (on LoopLM) as a detached process. It:
  1. waits until cheems is reachable on TCP/22
  2. uploads distill_v2.py
  3. runs a 10-step smoke test foreground
  4. if smoke passes, runs full D_STEPS=600 training foreground over SSH,
     streaming all output to auto_train_v2.log
  5. if the SSH link drops mid-run, waits for reconnect and (after checking
     that no remote python is running and M2-v2 is not already complete)
     relaunches the full training.

Stop file: D:/LoopLM/tmp/auto_train_v2.stop — create it to abort the loop.
"""
import os, sys, json, time, socket, traceback
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "cheems"
USER = "ai training"
PASSWORD = "REPLACE_ME"
LOCAL_DISTILL = "D:/LoopLM/distill_v2.py"
REMOTE_DISTILL = "D:/training/distill_v2.py"
SMOKE_OUT = "D:/training/checkpoints/ouro_retention_m2_d2_smoke"
FULL_OUT = "D:/training/checkpoints/ouro_retention_m2_d2"
FULL_CKPT = FULL_OUT + "/best_model.pt"
FULL_LOG = FULL_OUT + "/distill_log.json"
STOP_FILE = "D:/LoopLM/tmp/auto_train_v2.stop"
LOG_DIR = "D:/LoopLM/tmp"
FULL_STEPS = int(os.environ.get("AUTO_D_STEPS", "600"))
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
    c.connect(HOST, username=USER, password=PASSWORD, timeout=20,
              banner_timeout=30, auth_timeout=30)
    return c


def upload(sftp_client=None):
    """sftp_client may be an SSHClient (open_sftp used) or an SFTPClient."""
    c = None
    own_sftp = False
    try:
        if sftp_client is None:
            c = connect_ssh()
            sftp = c.open_sftp()
            own_sftp = True
        elif hasattr(sftp_client, "open_sftp"):
            sftp = sftp_client.open_sftp()
            own_sftp = True
        else:
            sftp = sftp_client
        sftp.put(LOCAL_DISTILL, REMOTE_DISTILL)
        log("uploaded %s -> %s" % (LOCAL_DISTILL, REMOTE_DISTILL))
        return True
    except Exception as e:
        log("upload failed: %s" % str(e))
        return False
    finally:
        if own_sftp:
            try:
                sftp.close()
            except Exception:
                pass
        if c is not None:
            c.close()


def run_remote(c, cmd, timeout=7200):
    """Run cmd on an open SSH client, stream output to log, return rc, out_tail."""
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
            # no output for 120s; if channel closed, recv_exit_status will tell
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
        if len(tail) > 200:
            tail = tail[-200:]
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
    """Return dict with python_running, full_complete, smoke_ok."""
    cmds = [
        'tasklist /FI "IMAGENAME eq python.exe" /FO CSV',
        'if exist D:\\training\\checkpoints\\ouro_retention_m2_d2\\best_model.pt (echo FULL_CKPT_EXISTS) else (echo NO_FULL_CKPT)',
        'if exist D:\\training\\checkpoints\\ouro_retention_m2_d2\\distill_log.json (echo FULL_LOG_EXISTS) else (echo NO_FULL_LOG)',
        'if exist D:\\training\\checkpoints\\ouro_retention_m2_d2_smoke\\best_model.pt (echo SMOKE_CKPT_EXISTS) else (echo NO_SMOKE_CKPT)',
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
        if not upload(c):
            return False
        cmd = ('cmd /c "set D_SMOKE=1&& set D_STEPS=10&& '
               'set D_OUT_DIR=D:/training/checkpoints/ouro_retention_m2_d2_smoke&& '
               'python -u D:/training/distill_v2.py"')
        rc, tail = run_remote(c, cmd, timeout=1800)
        ok = (rc == 0) and any("PHASE 0H step2b OK" in s for s in tail)
        if ok:
            log("SMOKE PASSED")
            return True
        log("SMOKE FAILED (rc=%s)" % rc)
        return False
    except Exception as e:
        log("smoke_test connection error: %s" % str(e))
        return False
    finally:
        c.close()


def full_training():
    c = connect_ssh()
    try:
        if not upload(c):
            return False
        cmd = ('cmd /c "set D_SMOKE=0&& set D_STEPS=%d&& '
               'set D_OUT_DIR=D:/training/checkpoints/ouro_retention_m2_d2&& '
               'python -u D:/training/distill_v2.py"' % FULL_STEPS)
        rc, tail = run_remote(c, cmd, timeout=7200)
        ok = (rc == 0) and any("PHASE 0H step2b OK" in s for s in tail)
        log("FULL TRAINING %s (rc=%s)" % ("PASSED" if ok else "FAILED", rc))
        return ok
    except Exception as e:
        log("full_training connection error: %s" % str(e))
        return False
    finally:
        c.close()


def main():
    global log_fh
    os.makedirs(LOG_DIR, exist_ok=True)
    log_fh = open(os.path.join(LOG_DIR, "auto_train_v2.log"), "a", encoding="utf-8")
    log("=== auto_train_v2 started, D_STEPS=%d ===" % FULL_STEPS)

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
        # check remote state before launching
        try:
            c = connect_ssh()
            st = remote_status_checks(c)
            c.close()
        except Exception as e:
            log("status check failed: %s" % str(e))
            st = None
        if st is not None:
            if st["full_complete"]:
                log("full training already complete on remote, nothing to launch")
                break
            if st["python_running"]:
                log("remote python.exe is already running; waiting and rechecking")
                time.sleep(POLL_SECONDS)
                continue
        # smoke then full
        if not smoke_test():
            log("smoke failed; will wait and retry")
            time.sleep(POLL_SECONDS)
            continue
        if full_training():
            log("full training completed successfully")
            break
        else:
            log("full training did not complete; will retry after wait")
            time.sleep(POLL_SECONDS)

    log("=== auto_train_v2 exiting ===")
    if log_fh is not None:
        log_fh.close()


if __name__ == "__main__":
    main()
