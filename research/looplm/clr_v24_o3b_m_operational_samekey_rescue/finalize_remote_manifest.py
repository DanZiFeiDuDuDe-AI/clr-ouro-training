from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256_manifest.json":
            files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"root": str(root), "manifest_updated_after_final_report_sync": True, "files": files}
    (root / "artifact_sha256_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MANIFEST_REFRESHED files={len(files)} path={root / 'artifact_sha256_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
