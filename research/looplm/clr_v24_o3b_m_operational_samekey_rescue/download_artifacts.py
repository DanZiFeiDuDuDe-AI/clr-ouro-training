from __future__ import annotations

import getpass
import hashlib
import stat
from pathlib import Path, PurePosixPath

import paramiko


REMOTE = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m_operational_samekey_rescue"
LOCAL = Path(__file__).resolve().parent
REPLACE_ALLOWED = {"artifact_sha256_manifest.json"}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_remote(sftp: paramiko.SFTPClient, path: str) -> str:
    h = hashlib.sha256()
    with sftp.open(path, "rb") as f:
        while True:
            block = f.read(4 * 1024 * 1024)
            if not block:
                return h.hexdigest()
            h.update(block)


def walk(sftp: paramiko.SFTPClient, remote_dir: str, rel: PurePosixPath = PurePosixPath(".")) -> int:
    count = 0
    for item in sftp.listdir_attr(remote_dir):
        remote_path = remote_dir.rstrip("/") + "/" + item.filename
        local_rel = PurePosixPath(item.filename) if rel == PurePosixPath(".") else rel / item.filename
        if stat.S_ISDIR(item.st_mode):
            (LOCAL / Path(*local_rel.parts)).mkdir(parents=True, exist_ok=True)
            count += walk(sftp, remote_path, local_rel)
            continue
        local_path = LOCAL / Path(*local_rel.parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_hash = digest_remote(sftp, remote_path)
        if local_path.exists():
            if digest_file(local_path) != remote_hash:
                if local_rel.as_posix() not in REPLACE_ALLOWED:
                    raise FileExistsError(f"refusing to overwrite different local file: {local_path}")
                sftp.get(remote_path, str(local_path))
        else:
            sftp.get(remote_path, str(local_path))
        if digest_file(local_path) != remote_hash:
            raise IOError(f"SHA-256 mismatch after download: {remote_path}")
        count += 1
    return count


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("ssh.zzai.scnet.cn", port=11810, username="root", password=getpass.getpass("SSH password: "), timeout=40, auth_timeout=40, banner_timeout=40)
    sftp = client.open_sftp()
    try:
        count = walk(sftp, REMOTE)
        print(f"DOWNLOADED_VERIFIED_FILES={count}")
        print(f"LOCAL_OUTPUT={LOCAL}")
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
