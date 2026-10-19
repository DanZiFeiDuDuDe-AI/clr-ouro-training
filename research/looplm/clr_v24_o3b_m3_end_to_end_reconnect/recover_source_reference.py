from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import paramiko


OUT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/private_data/clr_v24_o3b_l"
CHECKPOINTS = REMOTE_ROOT + "/checkpoints"
SOURCES = {
    "a1_v24_o21a.py": REMOTE_ROOT + "/code/a1_v24_o21a.py",
    "a1_v24_o21a_mr.py": REMOTE_ROOT + "/code/a1_v24_o21a_mr.py",
    "a1_v24_vgen.py": REMOTE_ROOT + "/code/a1_v24_vgen.py",
    "a1_v24_o3a.py": REMOTE_ROOT + "/code/a1_v24_o3a.py",
    "a1_v24_o3b.py": REMOTE_ROOT + "/code/a1_v24_o3b.py",
    "a1_v24_o2_common.py": REMOTE_ROOT + "/code/a1_v24_o2_common.py",
}
CONFIGS = {
    "o3b_l_diagnostic_config.json": REMOTE_ROOT + "/output/config.json",
    "o3b_k_diagnostic_config.json": REMOTE_ROOT + "/context/o3b_k/config.json",
}
EXPECTED_DEPENDENCIES = {
    "o21a_mr_checkpoint": CHECKPOINTS + "/clr_v24_o21a_mr/best_model.pt",
    "vgen2_checkpoint": CHECKPOINTS + "/clr_v24_vgen2/best.pt",
    "o3a_allocator_checkpoint": CHECKPOINTS + "/clr_v24_o3a_allocation/allocator_kaware_best.pt",
    "baseline_router_checkpoint": CHECKPOINTS + "/clr_v24_o3b_update/router_best.pt",
    "baseline_o3b_config": CHECKPOINTS + "/clr_v24_o3b_update/config.json",
    "o21b_checkpoint": CHECKPOINTS + "/clr_v24_o21b/b1_best.pt",
    "candidate_b_checkpoint": CHECKPOINTS + "/clr_v24_o3b_m_operational_samekey_rescue/models/candidate_b_seed_43.pt",
}


def remote_digest(sftp: paramiko.SFTPClient, path: str) -> str:
    digest = hashlib.sha256()
    with sftp.open(path, "rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "ssh.zzai.scnet.cn",
        port=11810,
        username="root",
        password=getpass.getpass("SSH password: "),
        timeout=40,
        auth_timeout=40,
        banner_timeout=40,
        allow_agent=False,
        look_for_keys=False,
    )
    sftp = client.open_sftp()
    evidence = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshots": {},
        "diagnostic_configs_not_baseline": {},
        "expected_dependency_paths": {},
    }
    try:
        sources_dir = OUT / "source_reference"
        sources_dir.mkdir(parents=True, exist_ok=True)
        for name, remote_path in SOURCES.items():
            target = sources_dir / name
            sftp.get(remote_path, str(target))
            attr = sftp.stat(remote_path)
            remote_sha = remote_digest(sftp, remote_path)
            local_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            if local_sha != remote_sha:
                raise IOError(f"source snapshot SHA mismatch: {remote_path}")
            evidence["source_snapshots"][name] = {
                "remote_path": remote_path,
                "local_snapshot": str(target),
                "bytes": attr.st_size,
                "sha256": remote_sha,
                "provenance": "read-only snapshot of the active remote CLR v2.4 code tree",
            }
        for name, remote_path in CONFIGS.items():
            target = sources_dir / name
            sftp.get(remote_path, str(target))
            attr = sftp.stat(remote_path)
            remote_sha = remote_digest(sftp, remote_path)
            local_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            if local_sha != remote_sha:
                raise IOError(f"config snapshot SHA mismatch: {remote_path}")
            evidence["diagnostic_configs_not_baseline"][name] = {
                "remote_path": remote_path,
                "local_snapshot": str(target),
                "bytes": attr.st_size,
                "sha256": remote_sha,
                "classification": "O3b-L or O3b-K diagnostic config; not the original O3b integration config",
            }
        for name, remote_path in EXPECTED_DEPENDENCIES.items():
            try:
                attr = sftp.stat(remote_path)
                evidence["expected_dependency_paths"][name] = {
                    "path": remote_path,
                    "exists": True,
                    "bytes": attr.st_size,
                    "sha256": remote_digest(sftp, remote_path) if attr.st_size < 50_000_000 else None,
                }
            except OSError as exc:
                evidence["expected_dependency_paths"][name] = {
                    "path": remote_path,
                    "exists": False,
                    "error_type": type(exc).__name__,
                }
    finally:
        sftp.close()
        client.close()
    path = OUT / "remote_source_and_checkpoint_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("EVIDENCE=" + str(path))
    print(json.dumps(evidence["expected_dependency_paths"], indent=2, ensure_ascii=False))
    print(json.dumps(evidence["source_snapshots"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
