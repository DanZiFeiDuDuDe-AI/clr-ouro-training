from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent / "output_v2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fits(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if not r.get("split")]


def main() -> int:
    manifest = json.loads((OUT / "artifact_hash_manifest.json").read_text(encoding="utf-8"))
    bad = []
    for item in manifest["files"]:
        path = OUT / Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            bad.append(item["path"])
    summaries = {}
    for name in ("r0_r1_converged_results.csv", "r1_r2_converged_results.csv"):
        rows = fits(OUT / name)
        invalid = [r for r in rows if int(r["best_step"]) > int(r["stop_step"])]
        summaries[name] = {"fit_rows": len(rows), "best_after_stop_rows": len(invalid),
                           "converged_rows": sum(r["convergence_flag"] == "CONVERGED" for r in rows)}
    verdict = json.loads((OUT / "FINAL_VERDICT.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "sanity" / "measurement_gate.json").read_text(encoding="utf-8"))
    stagewise = (OUT / "tables" / "stagewise_linear_results.csv").read_text(encoding="utf-8-sig")
    stagewise_not_run = "NOT_RUN" in stagewise[:150]
    result = {"artifact_hashes_pass": not bad, "hashed_files": len(manifest["files"]),
              "hash_mismatches": bad, "fit_table_checks": summaries,
              "measurement_gate": gate, "stagewise_not_run_marker": stagewise_not_run,
              "checkpoint_modified": verdict.get("checkpoint_modified"),
              "data_leakage_detected": verdict.get("data_leakage_detected"),
              "primary": verdict.get("primary")}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if bad or any(v["best_after_stop_rows"] for v in summaries.values()) or not stagewise_not_run:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
