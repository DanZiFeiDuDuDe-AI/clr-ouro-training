"""Read-only snapshot of the pinned Ouro config/tokenizer/modeling sources.

Connects to the existing Cheems artifact host using its saved local helper;
does not load a model or perform inference.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import _pull_o3b_cheems as cheems  # noqa: E402


REMOTE_DIR = "D:/training/Ouro-1.4B"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "source_snapshot" / "Ouro-1.4B"
FILES = (
    "config.json",
    "configuration_ouro.py",
    "modeling_ouro.py",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
KNOWN_MODEL_SHA256 = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    client = cheems.connect()
    sftp = client.open_sftp()
    rows = []
    try:
        for name in FILES:
            remote = REMOTE_DIR + "/" + name
            local = LOCAL_DIR / name
            st = sftp.stat(remote)
            sftp.get(remote, str(local))
            actual = digest(local)
            rows.append({
                "source_host": "cheems",
                "source_path": remote,
                "snapshot_path": str(local),
                "size_bytes": st.st_size,
                "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
                "sha256": actual,
                "transfer_verified": local.stat().st_size == st.st_size,
            })
        model_path = REMOTE_DIR + "/model.safetensors"
        mst = sftp.stat(model_path)
        rows.append({
            "source_host": "cheems",
            "source_path": model_path,
            "snapshot_path": "REMOTE COPY PENDING; not yet transferred to J3C target",
            "size_bytes": mst.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(mst.st_mtime, dt.timezone.utc).isoformat(),
            "sha256": KNOWN_MODEL_SHA256,
            "sha256_method": "remote Windows certutil -hashfile SHA256; verified against project pinned SHA",
            "sha256_verified": True,
            "transfer_verified": False,
        })
    finally:
        sftp.close()
        client.close()
    payload = {
        "model_id": "ByteDance/Ouro-1.4B",
        "pinned_model_sha256": KNOWN_MODEL_SHA256,
        "source_dir": REMOTE_DIR,
        "files": rows,
        "note": "Metadata/tokenizer/modeling files were copied read-only from the pinned source. The large model tensor is hashed on source but is not staged to the J3C target until extraction policy has been written.",
    }
    out = LOCAL_DIR.parent / "source_model_artifact_manifest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out), "files": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
