# -*- coding: utf-8 -*-
"""Retry cheems connectivity, upload O3b-L, then run it once."""
import socket
import sys
import time
import warnings
from pathlib import Path

import paramiko

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOSTS = ("cheems", "100.104.57.18")
USER = "ai training"
PASSWORD = __import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")
LOCAL = Path(__file__).with_name("a1_v24_o3b_l.py")
REMOTE = "D:/training/a1_v24_o3b_l.py"
MAX_WAIT = int(__import__("os").environ.get("O3BL_MAX_WAIT", "7200"))
POLL = int(__import__("os").environ.get("O3BL_POLL", "20"))


def log(msg):
    print("[O3b-L launcher] " + msg, flush=True)


def connect():
    last = None
    for host in HOSTS:
        try:
            sock = socket.create_connection((host, 22), timeout=8)
            sock.close()
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, username=USER, password=PASSWORD, timeout=20, banner_timeout=30, auth_timeout=30)
            return c, host
        except Exception as ex:
            last = ex
    raise last


def main():
    if not LOCAL.exists(): raise FileNotFoundError(LOCAL)
    deadline = time.time() + MAX_WAIT
    c = None
    while time.time() < deadline:
        try:
            c, host = connect()
            log("connected to " + host)
            sftp = c.open_sftp()
            try: sftp.put(str(LOCAL), REMOTE)
            finally: sftp.close()
            log("uploaded " + str(LOCAL) + " -> " + REMOTE)
            _, stdout, stderr = c.exec_command("python -u D:/training/a1_v24_o3b_l.py", timeout=7200)
            stdout.channel.settimeout(120)
            while True:
                line = stdout.readline()
                if not line: break
                log("REMOTE| " + line.rstrip())
            err = stderr.read().decode("utf-8", "replace")
            rc = stdout.channel.recv_exit_status()
            log("remote rc=" + str(rc))
            if err: log("REMOTE STDERR| " + err[-4000:])
            return rc
        except Exception as ex:
            log("cheems unavailable or command failed: " + repr(ex))
            if c is not None:
                try: c.close()
                except Exception: pass
                c = None
            time.sleep(POLL)
    log("timeout waiting for cheems")
    return 2


if __name__ == "__main__": raise SystemExit(main())
