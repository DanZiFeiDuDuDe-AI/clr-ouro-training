# -*- coding: utf-8 -*-
"""
Auto-launcher for fixed-gamma distillation runs (after v3 finishes).

Runs LOCALLY as a detached process:
  1. wait until cheems reachable AND no remote python is running (so it
     never collides with the v3 run / evals)
  2. upload updated retention_ouro.py (clamped_direct support),
     distill_gammafix.py, post_distill_eval_gammafix.py
  3. smoke 10-step gamma mode, then:
       a) full gamma-only 600-step training -> eval
       b) full gamma+norm 600-step training -> eval

Stop file: D:/LoopLM/tmp/auto_gammafix.stop
Log file:  D:/LoopLM/tmp/auto_gammafix.log
"""
import os, sys, json, time, socket
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "cheems"
USER = "ai training"
PASSWORD = "REPLACE_ME"
LOCAL_RET = "D:/LoopLM/retention_ouro.py"
LOCAL_TRAIN = "D:/LoopLM/distill_gammafix.py"
LOCAL_EVAL = "D:/LoopLM/post_distill_eval_gammafix.py"
REMOTE_RET = "D:/training/retention_ouro.py"
REMOTE_TRAIN = "D:/training/distill_gammafix.py"
REMOTE_EVAL = "D:/training/post_distill_eval_gammafix.py"
GAMMA_OUT = "D:/training/checkpoints/ouro_retention_m2_gammafix"
GAMMANORM_OUT = "D:/training/checkpoints/ouro_retention_m2_gammafix_norm"
STOP_FILE = "D:/LoopLM/tmp/auto_gammafix.stop"
LOG_PATH = "D:/LoopLM/tmp/auto_gammafix.log"
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


def remote_python_running(c):
    try:
        stdin, stdout, stderr = c.exec_command(
            'tasklist /FI "IMAGENAME eq python.exe" /FO CSV', timeout=30)
        out = stdout.read().decode("utf-8", "replace").strip()
        stdout.channel.recv_exit_status()
        return "python.exe" in out
    except Exception as e:
        log("python_running check failed: %s" % str(e))
        return True  # be conservative: assume running


def wait_until_idle():
    while True:
        if stop_requested():
            return False
        if time.time() - WAIT_START > MAX_WAIT_SECONDS:
            log("max wait time reached")
            return False
        if not host_reachable():
            log("cheems not reachable (TCP/22). waiting %ds..." % POLL_SECONDS)
            time.sleep(POLL_SECONDS)
            continue
        try:
            c = connect_ssh()
            running = remote_python_running(c)
            c.close()
        except Exception as e:
            log("idle check failed: %s" % str(e))
            running = True
        if not running:
            return True
        log("remote python still running; waiting %ds..." % POLL_SECONDS)
        time.sleep(POLL_SECONDS)


def run_train(mode, steps=600, smoke=False):
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_RET, REMOTE_RET)
        upload_file(c, LOCAL_TRAIN, REMOTE_TRAIN)
        out = GAMMA_OUT if mode == "gamma" else GAMMANORM_OUT
        if smoke:
            out = out + "_smoke"
            steps = 10
        cmd = "python -u D:/training/distill_gammafix.py %s %d %s %d" % (
            mode, steps, out, 1 if smoke else 0)
        rc, tail = run_remote(c, cmd, timeout=(1800 if smoke else 7200))
        ok = (rc == 0) and any("PHASE 0H step2d OK" in s for s in tail)
        log("TRAIN %s %s (rc=%s)" % (mode, "SMOKE" if smoke else "FULL", rc))
        return ok
    except Exception as e:
        log("run_train %s error: %s" % (mode, str(e)))
        return False
    finally:
        c.close()


def run_eval(mode):
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_RET, REMOTE_RET)
        upload_file(c, LOCAL_EVAL, REMOTE_EVAL)
        out = GAMMA_OUT if mode == "gamma" else GAMMANORM_OUT
        ckpt = out + "/best_model.pt"
        out_json = ("D:/training/p0h_post_distill_eval_gammafix_gamma.json" if mode == "gamma"
                    else "D:/training/p0h_post_distill_eval_gammafix_norm.json")
        cmd = "python -u D:/training/post_distill_eval_gammafix.py %s %s" % (ckpt, out_json)
        rc, tail = run_remote(c, cmd, timeout=1800)
        ok = (rc == 0) and any("PHASE 0H POST-DISTILL EVAL GAMMAFIX OK" in s for s in tail)
        log("EVAL %s (rc=%s)" % (mode, rc))
        return ok
    except Exception as e:
        log("run_eval %s error: %s" % (mode, str(e)))
        return False
    finally:
        c.close()


def main():
    global log_fh
    log_fh = open(LOG_PATH, "a", encoding="utf-8")
    log("=== auto_gammafix started ===")
    if not wait_until_idle():
        log("never became idle; exiting")
        return
    log("cheems idle; starting fixed-gamma runs")

    # smoke gamma mode first
    if not run_train("gamma", smoke=True):
        log("gamma smoke failed; exiting")
        return
    # full gamma-only
    if run_train("gamma", steps=600, smoke=False):
        run_eval("gamma")
    else:
        log("gamma-only full training failed; continuing? no, exiting")
        return
    # full gamma+norm
    if run_train("gammanorm", steps=600, smoke=False):
        run_eval("gammanorm")
    else:
        log("gammanorm full training failed")

    log("=== auto_gammafix exiting ===")
    if log_fh is not None:
        log_fh.close()


if __name__ == "__main__":
    main()
