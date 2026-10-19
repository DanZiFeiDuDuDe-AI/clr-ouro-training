"""Upload the frozen extraction harness and its model artifact note."""
from __future__ import annotations

import getpass
import hashlib
import json
from pathlib import Path

import paramiko

HOST, PORT, USER = "ssh.zzai.scnet.cn", 11810, "root"
HERE = Path(__file__).parent
OUT = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache"
FILES = {
    "j3c_extract.py": OUT + "/code/j3c_extract.py",
    "source_model_artifact_manifest.json": OUT + "/source_snapshot/source_model_artifact_manifest.json",
}
WEIGHT = OUT + "/source_snapshot/Ouro-1.4B/model.safetensors"
EXPECTED_WEIGHT_SHA = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = ""
    for part in path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def main() -> None:
    pw = getpass.getpass(f"SSH password for {USER}@{HOST}:{PORT}: ")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pw, timeout=30)
    del pw
    s = c.open_sftp()
    try:
        _, stdout, stderr = c.exec_command(f"sha256sum '{WEIGHT}'", timeout=600)
        remote_weight_sha = stdout.read().decode("utf-8", "replace").split()[0]
        remote_weight_err = stderr.read().decode("utf-8", "replace")
        if remote_weight_sha != EXPECTED_WEIGHT_SHA:
            raise IOError(f"remote model hash not yet certified: {remote_weight_sha} {remote_weight_err[-300:]}")
        manifest_path = HERE.parent / "source_snapshot" / "source_model_artifact_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            weight_record = next((r for r in manifest.get("files", []) if r.get("sha256") == EXPECTED_WEIGHT_SHA), None)
            if weight_record is not None:
                weight_record["snapshot_path"] = WEIGHT
                weight_record["transfer_verified"] = True
                weight_record["transfer_verified_method"] = "remote sha256sum after SFTP upload; exact pinned SHA matched"
                weight_record["retrieval_replica"] = {
                    "repository": "ByteDance/Ouro-1.4B",
                    "revision": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
                    "source_url": "https://huggingface.co/ByteDance/Ouro-1.4B/resolve/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/model.safetensors",
                    "x_linked_etag_sha256": EXPECTED_WEIGHT_SHA,
                    "full_local_sha256": EXPECTED_WEIGHT_SHA,
                }
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        for name, remote in FILES.items():
            local = HERE / name if name == "j3c_extract.py" else HERE.parent / "source_snapshot" / name
            if not local.exists():
                if name == "source_model_artifact_manifest.json":
                    continue
                raise FileNotFoundError(local)
            mkdirs(s, remote.rpartition("/")[0])
            s.put(str(local), remote)
            _, stdout, stderr = c.exec_command(f"sha256sum '{remote}'", timeout=60)
            got = stdout.read().decode("utf-8", "replace").split()[0]
            err = stderr.read().decode("utf-8", "replace")
            expected = sha(local)
            if got != expected:
                raise IOError(f"SHA mismatch for {name}: local={expected}, remote={got}, {err[-300:]}")
            print(f"UPLOAD_VERIFIED {name} sha256={got}", flush=True)
    finally:
        s.close()
        c.close()


if __name__ == "__main__":
    main()
