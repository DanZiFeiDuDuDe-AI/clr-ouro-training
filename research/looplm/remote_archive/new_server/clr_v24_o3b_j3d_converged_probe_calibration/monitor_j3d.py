from __future__ import annotations

import getpass
import json
import sys
import time

import paramiko


OUT = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration_v2"
LOG = OUT + "/logs/j3d_pipeline_run.log"
PIDFILE = OUT + "/logs/j3d_pipeline.pid"
FINAL = OUT + "/FINAL_VERDICT.json"


def read_remote(sftp: paramiko.SFTPClient, path: str) -> bytes:
    with sftp.open(path, "rb") as f:
        return f.read()


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("ssh.zzai.scnet.cn", port=11810, username="root",
                   password=getpass.getpass("SSH password: "), timeout=40,
                   auth_timeout=40, banner_timeout=40)
    client.get_transport().set_keepalive(30)
    sftp = client.open_sftp()
    try:
        pid = int(read_remote(sftp, PIDFILE).decode().strip())
        print(json.dumps({"monitoring_pid": pid, "log": LOG}), flush=True)
        offset = 0
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                stat = sftp.stat(LOG)
                if stat.st_size > offset:
                    with sftp.open(LOG, "rb") as f:
                        f.seek(offset)
                        chunk = f.read(stat.st_size - offset)
                    offset += len(chunk)
                    sys.stdout.write(chunk.decode("utf-8", "replace"))
                    sys.stdout.flush()
            except OSError as exc:
                print(f"LOG_READ_RETRY {exc!r}", flush=True)
            try:
                verdict = json.loads(read_remote(sftp, FINAL).decode("utf-8"))
                print("FINAL_VERDICT " + json.dumps(verdict, ensure_ascii=False), flush=True)
                return 0
            except OSError:
                pass
            try:
                sftp.stat(f"/proc/{pid}")
            except OSError:
                print("PROCESS_EXITED_WITHOUT_FINAL_VERDICT", flush=True)
                return 2
            time.sleep(10)
        print("MONITOR_TIMEOUT", flush=True)
        return 3
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
