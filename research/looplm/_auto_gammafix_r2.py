# -*- coding: utf-8 -*-
"""
Auto-launcher round 2 for fixed-gamma experiments with fp32 gamma params.

Fixes vs round 1:
  - clamped_direct gamma is kept fp32 (not bf16) so Adam can move it
  - clamp bounds widened to 1e-6..1-1e-6
  - matched_eval now seeds random.seed(42) (canonical draw)

Runs:
  1. gamma-only fixed 600 -> post-distill eval
  2. gamma+norm fixed 600 -> post-distill eval
  3. matched_eval unified table

Stop file: D:/LoopLM/tmp/auto_gammafix_r2.stop
Log file:  D:/LoopLM/tmp/auto_gammafix_r2.log
"""
import os, sys, json, time, socket
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "cheems"
HOST_IP = "100.104.57.18"
USER = "ai training"
PASSWORD = __import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")
LOCAL_RET = "D:/LoopLM/retention_ouro.py"
LOCAL_TRAIN = "D:/LoopLM/distill_gammafix.py"
LOCAL_EVAL = "D:/LoopLM/post_distill_eval_gammafix.py"
LOCAL_MATCHED = "D:/LoopLM/matched_eval.py"
REMOTE_RET = "D:/training/retention_ouro.py"
REMOTE_TRAIN = "D:/training/distill_gammafix.py"
REMOTE_EVAL = "D:/training/post_distill_eval_gammafix.py"
REMOTE_MATCHED = "D:/training/matched_eval.py"
GAMMA_OUT = "D:/training/checkpoints/ouro_retention_m2_gammafix_r2"
GAMMANORM_OUT = "D:/training/checkpoints/ouro_retention_m2_gammafix_norm_r2"
STOP_FILE = "D:/LoopLM/tmp/auto_gammafix_r2.stop"
LOG_PATH = "D:/LoopLM/tmp/auto_gammafix_r2.log"
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
        return True


def wait_until_idle():
    while True:
        if stop_requested():
            return False
        if time.time() - WAIT_START > MAX_WAIT_SECONDS:
            log("max wait time reached")
            return False
        h = resolve_host()
        if not host_reachable(h):
            log("cheems not reachable (TCP/22 via %s). waiting %ds..." % (h, POLL_SECONDS))
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


def run_train(mode, out):
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_RET, REMOTE_RET)
        upload_file(c, LOCAL_TRAIN, REMOTE_TRAIN)
        cmd = "python -u D:/training/distill_gammafix.py %s 600 %s 0" % (mode, out)
        rc, tail = run_remote(c, cmd, timeout=7200)
        ok = (rc == 0) and any("PHASE 0H step2d OK" in s for s in tail)
        log("TRAIN %s FULL (rc=%s)" % (mode, rc))
        return ok
    except Exception as e:
        log("run_train %s error: %s" % (mode, str(e)))
        return False
    finally:
        c.close()


def run_eval(mode, out):
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_RET, REMOTE_RET)
        upload_file(c, LOCAL_EVAL, REMOTE_EVAL)
        ckpt = out + "/best_model.pt"
        out_json = ("D:/training/p0h_post_distill_eval_gammafix_r2_gamma.json" if mode == "gamma"
                    else "D:/training/p0h_post_distill_eval_gammafix_r2_norm.json")
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


def run_matched_eval():
    c = connect_ssh()
    try:
        upload_file(c, LOCAL_RET, REMOTE_RET)
        upload_file(c, LOCAL_MATCHED, REMOTE_MATCHED)
        cmd = "python -u D:/training/matched_eval.py"
        rc, tail = run_remote(c, cmd, timeout=3600)
        ok = (rc == 0) and any("MATCHED EVAL OK" in s for s in tail)
        log("MATCHED EVAL (rc=%s)" % rc)
        return ok
    except Exception as e:
        log("run_matched_eval error: %s" % str(e))
        return False
    finally:
        c.close()


def main():
    global log_fh
    log_fh = open(LOG_PATH, "a", encoding="utf-8")
    log("=== auto_gammafix_r2 started ===")
    if not wait_until_idle():
        log("never became idle; exiting")
        return
    log("cheems idle; starting r2 fixed-gamma runs")

    if run_train("gamma", GAMMA_OUT):
        run_eval("gamma", GAMMA_OUT)
    else:
        log("gamma r2 training failed; exiting")
        return

    if run_train("gammanorm", GAMMANORM_OUT):
        run_eval("gammanorm", GAMMANORM_OUT)
    else:
        log("gammanorm r2 training failed")

    run_matched_eval()
    log("=== auto_gammafix_r2 exiting ===")
    if log_fh is not None:
        log_fh.close()


if __name__ == "__main__":
    main()
