from __future__ import annotations

import getpass
import hashlib
import stat
from pathlib import Path, PurePosixPath

import paramiko


REMOTE = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag"
LOCAL = Path(__file__).resolve().parent / "output"
UPDATED_OUTPUTS = {"final_report.md", "FINAL_VERDICT.txt", "final_verdict.json", "verdict.json",
                   "artifact_hash_manifest.json", "j3e_probe_protocol_frozen.json"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remote_digest(sftp, path: str) -> str:
    h = hashlib.sha256()
    with sftp.open(path, "rb") as f:
        while True:
            block = f.read(4 * 1024 * 1024)
            if not block:
                return h.hexdigest()
            h.update(block)


def walk(sftp, remote_dir: str, rel: PurePosixPath = PurePosixPath(".")) -> int:
    count = 0
    for entry in sftp.listdir_attr(remote_dir):
        rp = remote_dir.rstrip("/") + "/" + entry.filename
        sub = PurePosixPath(entry.filename) if rel == PurePosixPath(".") else rel / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            (LOCAL / Path(*sub.parts)).mkdir(parents=True, exist_ok=True)
            count += walk(sftp, rp, sub)
            continue
        lp = LOCAL / Path(*sub.parts)
        lp.parent.mkdir(parents=True, exist_ok=True)
        rh = remote_digest(sftp, rp)
        relname = sub.as_posix()
        if lp.exists() and digest(lp) != rh:
            if relname not in UPDATED_OUTPUTS:
                raise FileExistsError(f"local file differs; refusing overwrite: {lp}")
            sftp.get(rp, str(lp))
        elif not lp.exists():
            sftp.get(rp, str(lp))
        if digest(lp) != rh:
            raise IOError(f"SHA-256 mismatch after download: {rp}")
        count += 1
    return count


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("ssh.zzai.scnet.cn", port=11810, username="root",
                   password=getpass.getpass("SSH password: "), timeout=40,
                   auth_timeout=40, banner_timeout=40)
    sftp = client.open_sftp()
    try:
        LOCAL.mkdir(parents=True, exist_ok=True)
        count = walk(sftp, REMOTE)
        print(f"DOWNLOADED_VERIFIED_FILES={count}")
        print(f"LOCAL_OUTPUT={LOCAL}")
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
