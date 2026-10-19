from __future__ import annotations

import getpass
import sys
import time

import paramiko


REMOTE = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag"


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "main"
    log_path = f"{REMOTE}/logs/{stage}.log"
    pid_path = f"{REMOTE}/logs/{stage}.pid"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("ssh.zzai.scnet.cn", port=11810, username="root",
                   password=getpass.getpass("SSH password: "), timeout=40,
                   auth_timeout=40, banner_timeout=40)
    client.get_transport().set_keepalive(30)
    sftp = client.open_sftp()
    try:
        with sftp.open(pid_path, "r") as f:
            pid = int(f.read().decode().strip())
        offset = 0
        deadline = time.time() + 6 * 60 * 60
        print(f"monitor stage={stage} pid={pid}", flush=True)
        while time.time() < deadline:
            try:
                size = sftp.stat(log_path).st_size
                if size > offset:
                    with sftp.open(log_path, "rb") as f:
                        f.seek(offset)
                        data = f.read(size - offset)
                    offset += len(data)
                    sys.stdout.write(data.decode("utf-8", "replace"))
                    sys.stdout.flush()
            except OSError as exc:
                print(f"log read retry: {exc!r}", flush=True)
            try:
                sftp.stat(f"/proc/{pid}")
            except OSError:
                print("PROCESS_EXITED", flush=True)
                return 0
            time.sleep(10)
        print("MONITOR_TIMEOUT", flush=True)
        return 3
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
