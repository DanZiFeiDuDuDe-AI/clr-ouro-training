"""Read-only SHA-256 inventory for O3b-J3P provenance recovery."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(label: str, root: Path) -> list[dict]:
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    rows = []
    for path in paths:
        st = path.stat()
        rows.append({
            "label": label,
            "path": str(path),
            "size_bytes": st.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
            "sha256": digest(path),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", action="append", default=[], help="label=path; repeatable")
    ap.add_argument("--existing", type=Path, help="merge a previously generated manifest")
    args = ap.parse_args()

    all_rows = []
    if args.existing:
        old = json.loads(args.existing.read_text(encoding="utf-8"))
        all_rows.extend(old.get("artifacts", []))
    for spec in args.root:
        if "=" not in spec:
            raise SystemExit(f"Invalid --root (expected label=path): {spec}")
        label, raw_path = spec.split("=", 1)
        root = Path(raw_path)
        if not root.exists():
            all_rows.append({"label": label, "path": raw_path, "status": "MISSING"})
            continue
        all_rows.extend(inventory(label, root))

    rows_by_path = {}
    for row in all_rows:
        rows_by_path[(row["label"], row["path"])] = row
    rows = sorted(rows_by_path.values(), key=lambda r: (r["label"], r["path"]))
    payload = {
        "experiment": "O3b-J3P ROW PROVENANCE RECOVERY",
        "snapshot_created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "artifacts": rows,
        "summary": {
            "hashed_file_count": sum("sha256" in row for row in rows),
            "missing_roots": [row["path"] for row in rows if row.get("status") == "MISSING"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
