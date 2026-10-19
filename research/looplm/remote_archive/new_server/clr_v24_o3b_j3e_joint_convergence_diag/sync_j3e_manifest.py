from __future__ import annotations

import getpass
import hashlib
from pathlib import Path

import paramiko


REMOTE = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag/artifact_hash_manifest.json"
LOCAL = Path(__file__).resolve().parent / "output" / "artifact_hash_manifest.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("ssh.zzai.scnet.cn", port=11810, username="root",
                   password=getpass.getpass("SSH password: "), timeout=40,
                   auth_timeout=40, banner_timeout=40)
    sftp = client.open_sftp()
    try:
        LOCAL.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(REMOTE, str(LOCAL))
        data = LOCAL.read_bytes()
        print(f"MANIFEST_SYNCED bytes={len(data)} sha256={digest(data)}")
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
