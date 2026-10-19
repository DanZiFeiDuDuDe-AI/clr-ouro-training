from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact_sha256_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    json_files = sorted(ROOT.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    csv_files = sorted(ROOT.rglob("*.csv"))
    for path in csv_files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            list(csv.DictReader(f))

    o21a_gate = json.loads((ROOT / "components/o21a/o21a_rebase_gate.json").read_text(encoding="utf-8"))
    vgen_gate = json.loads((ROOT / "components/vgen2/vgen2_rebase_gate.json").read_text(encoding="utf-8"))
    candidate = json.loads((ROOT / "candidate_b_integrity.json").read_text(encoding="utf-8"))
    if not o21a_gate.get("gate_pass") or vgen_gate.get("gate_pass"):
        raise SystemExit("gate state changed; refusing to finalize an inconsistent report")
    if vgen_gate.get("test_evaluation") != "NOT_RUN_VAL_GATE_FAILED":
        raise SystemExit("unexpected VGen2 test exposure status")
    if candidate.get("threshold") != 0.8504285216331482 or candidate.get("modified") is not False:
        raise SystemExit("frozen Candidate B / Rule A integrity mismatch")
    if sha256(ROOT.parent / "clr_v24_o3b_m_operational_samekey_rescue/models/candidate_b_seed_43.pt") != candidate["expected_sha256"]:
        raise SystemExit("local frozen Candidate B SHA mismatch")

    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "experiment": "CLR-O3-REBASE-1",
        "hash_algorithm": "SHA-256",
        "manifest_excludes_itself": True,
        "json_files_validated": len(json_files),
        "csv_files_validated": len(csv_files),
        "o21a_gate": "PASS",
        "vgen2_gate": "FAIL; test not run",
        "candidate_b_frozen_sha_verified": True,
        "files": files,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "files_hashed": len(files), "json_valid": len(json_files), "csv_valid": len(csv_files), "o21a": "PASS", "vgen2": "FAIL_STOP"}, indent=2))


if __name__ == "__main__":
    main()
