from __future__ import annotations

import getpass
import hashlib
import stat
from pathlib import Path, PurePosixPath

import paramiko


REMOTE = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration_v2"
LOCAL = Path(__file__).resolve().parent / "output_v2"
EXCLUDED_DIRS = {"features"}
REFINED_OUTPUTS = {
    "final_report.md", "report/10_final_report.md", "final_evidence_summary.json",
    "FINAL_VERDICT.json", "verdict.json", "artifact_hash_manifest.json",
    "code/finalize_j3d_artifacts.py",
}


def sha_local(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_remote(sftp: paramiko.SFTPClient, path: str) -> str:
    h = hashlib.sha256()
    with sftp.open(path, "rb") as f:
        while True:
            block = f.read(4 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_tree(sftp: paramiko.SFTPClient, remote_dir: str, relative: PurePosixPath = PurePosixPath(".")) -> int:
    count = 0
    for item in sftp.listdir_attr(remote_dir):
        remote_path = remote_dir.rstrip("/") + "/" + item.filename
        rel = PurePosixPath(item.filename) if relative == PurePosixPath(".") else relative / item.filename
        if stat.S_ISDIR(item.st_mode):
            if item.filename in EXCLUDED_DIRS:
                continue
            (LOCAL / Path(*rel.parts)).mkdir(parents=True, exist_ok=True)
            count += download_tree(sftp, remote_path, rel)
            continue
        local_path = LOCAL / Path(*rel.parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_hash = sha_remote(sftp, remote_path)
        if local_path.exists():
            if sha_local(local_path) != remote_hash:
                if rel.as_posix() not in REFINED_OUTPUTS:
                    raise FileExistsError(f"local artifact differs; refusing overwrite: {local_path}")
                sftp.get(remote_path, str(local_path))
        else:
            sftp.get(remote_path, str(local_path))
        if sha_local(local_path) != remote_hash:
            raise IOError(f"download hash mismatch: {remote_path}")
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
        count = download_tree(sftp, REMOTE)
        print(f"DOWNLOADED_VERIFIED_FILES={count}")
        print(f"LOCAL_OUTPUT={LOCAL}")
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
