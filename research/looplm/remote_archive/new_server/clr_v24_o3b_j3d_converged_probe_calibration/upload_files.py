from __future__ import annotations

import getpass
import hashlib
import sys
from pathlib import Path

import paramiko


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remote_sha(sftp: paramiko.SFTPClient, path: str) -> str:
    h = hashlib.sha256()
    with sftp.open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ensure_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    current = ""
    for part in path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def main() -> int:
    if len(sys.argv) < 3 or (len(sys.argv) - 1) % 2:
        raise SystemExit("usage: upload_files.py LOCAL_PATH REMOTE_PATH [LOCAL_PATH REMOTE_PATH ...]")
    pairs = [(Path(sys.argv[i]), sys.argv[i + 1]) for i in range(1, len(sys.argv), 2)]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("ssh.zzai.scnet.cn", port=11810, username="root",
                   password=getpass.getpass("SSH password: "), timeout=30,
                   auth_timeout=30, banner_timeout=30)
    sftp = client.open_sftp()
    try:
        for local, remote in pairs:
            if not local.is_file():
                raise FileNotFoundError(local)
            ensure_dir(sftp, remote.rpartition("/")[0])
            expected = sha(local)
            try:
                sftp.stat(remote)
                got = remote_sha(sftp, remote)
                if got != expected:
                    raise FileExistsError(f"refusing to overwrite different remote file: {remote}")
            except FileNotFoundError:
                sftp.put(str(local), remote)
                got = remote_sha(sftp, remote)
            if got != expected:
                raise IOError(f"SHA-256 mismatch for {remote}: local={expected} remote={got}")
            print(f"UPLOAD_VERIFIED {remote} {got}", flush=True)
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
