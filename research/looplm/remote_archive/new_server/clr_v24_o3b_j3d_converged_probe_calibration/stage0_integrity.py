from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


CACHE = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache")
OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration_v2")
SPLITS = {"train": 11665, "val": 2332, "testA": 2332, "testB": 2332}
EXPECTED_DIMS = {
    "H_mention_fresh": 2048, "H_fresh": 2048, "Z0": 2048,
    "Z1": 256, "Z2": 256, "Z3": 64, "Z4": 64,
    "R0": 256, "R1": 256, "R2": 256,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite_mmap(path: Path, chunk_rows: int = 2048) -> tuple[list[int], str]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"not a 2D feature matrix: {path.name} {arr.shape}")
    if arr.dtype != np.float32:
        raise ValueError(f"dtype mismatch: {path.name} {arr.dtype}")
    for start in range(0, len(arr), chunk_rows):
        if not np.isfinite(arr[start:start + chunk_rows]).all():
            raise ValueError(f"non-finite feature value: {path.name} rows {start}+")
    return list(arr.shape), str(arr.dtype)


def main() -> int:
    start_time = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("configs", "features", "probe_runs", "sanity", "j_revalidation", "geometry", "plots", "tables", "logs", "report"):
        (OUT / folder).mkdir(exist_ok=True)

    manifest_path = CACHE / "fresh_cache_hash_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {r["filename"]: r for r in manifest["files"]}
    selected = [f"{stage}_{split}.npy" for split in SPLITS for stage in EXPECTED_DIMS]
    selected += [f"labels_{split}.csv" for split in SPLITS]
    selected += [f"H_fresh_{split}_sidecar.csv" for split in SPLITS]
    selected += ["feature_row_plan.csv", "canonical_source_manifest.jsonl", "mention_resolution_manifest.jsonl",
                 "fresh_row_join_audit.csv", "fresh_row_join_summary.json", "stage_definition.json",
                 "w1_svd_matrices.npz", "w1_svd_metrics.json", "extraction_policy_frozen.json",
                 "artifact_hash_manifest.json", "checkpoint_sha256_before_after.json"]
    checks: list[dict] = []
    errors: list[str] = []
    for name in selected:
        path = CACHE / name
        expected = by_name.get(name)
        rec = {"filename": name, "exists": path.is_file(), "manifest_entry": expected is not None}
        if not path.is_file() or expected is None:
            errors.append(f"missing file or certified hash entry: {name}")
            checks.append(rec)
            continue
        rec["bytes"] = path.stat().st_size
        rec["expected_bytes"] = expected.get("bytes")
        rec["sha256"] = sha256(path)
        rec["expected_sha256"] = expected.get("sha256")
        rec["hash_pass"] = rec["bytes"] == rec["expected_bytes"] and rec["sha256"] == rec["expected_sha256"]
        if not rec["hash_pass"]:
            errors.append(f"hash/size mismatch: {name}")
        checks.append(rec)

    expected_stage_defs = json.loads((CACHE / "stage_definition.json").read_text(encoding="utf-8"))
    row_checks: dict[str, dict] = {}
    all_feature_ids: set[str] = set()
    all_row_indices: set[str] = set()
    all_canonical_ids: set[str] = set()
    plan = read_csv(CACHE / "feature_row_plan.csv")
    plan_by_split: dict[str, list[dict[str, str]]] = {s: [] for s in SPLITS}
    for r in plan:
        if r.get("split") in plan_by_split:
            plan_by_split[r["split"]].append(r)
    for split, expected_n in SPLITS.items():
        labels = read_csv(CACHE / f"labels_{split}.csv")
        sidecar = read_csv(CACHE / f"H_fresh_{split}_sidecar.csv")
        split_errors: list[str] = []
        if len(labels) != expected_n or len(sidecar) != expected_n or len(plan_by_split[split]) != expected_n:
            split_errors.append(f"row count mismatch labels={len(labels)} sidecar={len(sidecar)} plan={len(plan_by_split[split])} expected={expected_n}")
        ids = [r.get("feature_row_id", "") for r in sidecar]
        row_indices = [r.get("row_index", "") for r in sidecar]
        canonical_ids = [r.get("canonical_fact_id", "") for r in sidecar]
        if len(set(ids)) != expected_n or any(not x for x in ids):
            split_errors.append("duplicate/empty feature_row_id in sidecar")
        duplicate_global = all_feature_ids.intersection(ids)
        if duplicate_global:
            split_errors.append(f"cross-split duplicate feature IDs: {len(duplicate_global)}")
        all_feature_ids.update(ids)
        if len(set(row_indices)) != expected_n or all_row_indices.intersection(row_indices):
            split_errors.append("duplicate row_index within/across splits")
        all_row_indices.update(row_indices)
        if len(set(canonical_ids)) != expected_n or all_canonical_ids.intersection(canonical_ids):
            split_errors.append("duplicate canonical_fact_id within/across splits")
        all_canonical_ids.update(canonical_ids)
        if [r.get("feature_row_id") for r in labels] != ids:
            split_errors.append("label/sidecar feature_row_id order mismatch")
        if [r.get("feature_row_id") for r in plan_by_split[split]] != ids:
            split_errors.append("plan/sidecar feature_row_id order mismatch")
        if [r.get("canonical_fact_id") for r in labels] != [r.get("canonical_fact_id") for r in sidecar]:
            split_errors.append("label/sidecar canonical_fact_id mismatch")
        if [r.get("canonical_fact_id") for r in plan_by_split[split]] != [r.get("canonical_fact_id") for r in sidecar]:
            split_errors.append("plan/sidecar canonical_fact_id mismatch")
        for i, (lab, sc, pl) in enumerate(zip(labels, sidecar, plan_by_split[split])):
            if int(lab["row_index"]) != int(sc["row_index"]) or int(lab["row_index"]) != int(pl["row_index"]) or int(lab["split_row_index"]) != i or int(sc["split_row_index"]) != i or int(pl["split_row_index"]) != i:
                split_errors.append(f"non-canonical row indices at row {i}")
                break
            pairs = (("entity_class_id", "entity_label"), ("relation_class_id", "relation_label"), ("joint_class_id", "joint_label"))
            for label_field, side_field in pairs:
                if str(lab[label_field]) != str(sc[side_field]):
                    split_errors.append(f"label mismatch {label_field}/{side_field} at row {i}")
                    break
            if str(lab.get("relation_id")) != str(sc.get("relation_id")):
                split_errors.append(f"relation ID mismatch at row {i}")
                break
            if str(lab.get("entity_semantic_value")) != str(sc.get("entity_raw")):
                split_errors.append(f"entity semantic mismatch at row {i}")
                break
            if str(lab.get("relation_semantic_value")) != str(sc.get("relation_raw")):
                split_errors.append(f"relation semantic mismatch at row {i}")
                break
            if str(lab.get("joint_semantic_pair")) != str(sc.get("joint_semantic_pair")):
                split_errors.append(f"joint semantic pair mismatch at row {i}")
                break
        feature_info = {}
        for stage, dim in EXPECTED_DIMS.items():
            path = CACHE / f"{stage}_{split}.npy"
            shape, dtype = finite_mmap(path)
            feature_info[stage] = {"shape": shape, "dtype": dtype, "finite": True,
                                   "expected_dim": dim, "dim_pass": shape == [expected_n, dim]}
            if not feature_info[stage]["dim_pass"]:
                split_errors.append(f"shape mismatch {stage}: {shape}, expected {[expected_n, dim]}")
            entry = by_name.get(path.name, {})
            if entry.get("row_count") != expected_n or entry.get("shape") != shape:
                split_errors.append(f"certified row metadata mismatch {stage}")
            if entry.get("first_feature_row_id") != (ids[0] if ids else None) or entry.get("last_feature_row_id") != (ids[-1] if ids else None):
                split_errors.append(f"certified first/last row identity mismatch {stage}")
        row_checks[split] = {
            "expected_rows": expected_n, "label_rows": len(labels), "sidecar_rows": len(sidecar),
            "plan_rows": len(plan_by_split[split]), "unique_feature_ids": len(set(ids)),
            "feature_ids_first_last": [ids[0], ids[-1]] if ids else [], "feature_arrays": feature_info,
            "errors": split_errors, "pass": not split_errors,
        }
        errors.extend(f"{split}: {e}" for e in split_errors)

    join_summary = json.loads((CACHE / "fresh_row_join_summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((CACHE / "checkpoint_sha256_before_after.json").read_text(encoding="utf-8"))
    result = {
        "experiment": "O3b-J3D Stage 0 certified cache integrity recheck",
        "cache_path": str(CACHE), "output_path": str(OUT),
        "source_certification_verdict": manifest.get("certification_verdict"),
        "source_manifest_file_count": manifest.get("file_count"),
        "selected_file_count": len(selected), "selected_file_checks": checks,
        "splits": row_checks, "all_feature_ids_unique_across_splits": len(all_feature_ids) == sum(SPLITS.values()),
        "all_row_indices_unique_across_splits": len(all_row_indices) == sum(SPLITS.values()),
        "all_canonical_fact_ids_unique_across_splits": len(all_canonical_ids) == sum(SPLITS.values()),
        "total_unique_feature_ids": len(all_feature_ids), "stage_definition": expected_stage_defs,
        "source_row_join_summary": join_summary, "checkpoint_before_after": checkpoint,
        "test_splits_used_for_metrics_or_selection": False,
        "errors": errors, "pass": not errors, "elapsed_seconds": time.time() - start_time,
    }
    (OUT / "cache_integrity_recheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "sanity" / "stage0_audit.md").write_text(
        "# O3b-J3D Stage 0 — Certified cache integrity recheck\n\n"
        f"Verdict: **{'PASS' if not errors else 'FAIL'}**. Selected-file SHA-256/byte checks: {sum(bool(x.get('hash_pass')) for x in checks)}/{len(checks)}.\n\n"
        "All four splits were checked for the certified row counts, row-order identity across feature-row IDs / canonical IDs / labels / sidecars / plan, duplicate IDs, label agreement, feature shapes, dtype, and finite values. TestA/TestB were read only for integrity and identity checks; no accuracy/loss was computed and no test result informed selection.\n\n"
        + "\n".join(f"- {s}: {row_checks[s]['label_rows']} rows; {row_checks[s]['unique_feature_ids']} unique IDs; {'PASS' if row_checks[s]['pass'] else 'FAIL'}" for s in SPLITS)
        + ("\n\nErrors:\n" + "\n".join(f"- {e}" for e in errors) if errors else "\n\nNo integrity discrepancies were detected. Probe fitting has not started.\n"),
        encoding="utf-8",
    )
    print(json.dumps({"stage": "stage0_complete", "pass": not errors, "selected_files": len(checks),
                      "unique_ids": len(all_feature_ids), "errors": errors, "elapsed_s": result["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
