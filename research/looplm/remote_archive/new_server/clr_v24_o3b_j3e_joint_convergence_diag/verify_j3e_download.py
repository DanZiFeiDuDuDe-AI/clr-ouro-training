from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "output"
MANIFEST = ROOT / "artifact_hash_manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for row in manifest["files"]:
        p = ROOT / Path(*row["path"].split("/"))
        if not p.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            failures.append(f"hash/size mismatch: {row['path']}")
    print(json.dumps({"manifest_file_count": manifest["file_count"],
                      "checked": len(manifest["files"]), "failures": failures,
                      "pass": not failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
