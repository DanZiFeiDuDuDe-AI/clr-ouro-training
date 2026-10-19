from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CACHE = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache")
OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag")
SPLITS = {"train": 11665, "val": 2332, "testA": 2332, "testB": 2332}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "logs", "tables", "probe_runs", "report", "sanity", "plots"):
        (OUT / name).mkdir(exist_ok=True)
    manifest = json.loads((CACHE / "fresh_cache_hash_manifest.json").read_text(encoding="utf-8"))
    by_name = {r["filename"]: r for r in manifest["files"]}
    selected = [f"{rep}_{sp}.npy" for sp in SPLITS for rep in ("R1", "R2")]
    selected += [f"labels_{sp}.csv" for sp in SPLITS]
    selected += [f"H_fresh_{sp}_sidecar.csv" for sp in SPLITS]
    selected += ["feature_row_plan.csv", "fresh_row_join_summary.json", "stage_definition.json",
                 "w1_svd_matrices.npz", "checkpoint_sha256_before_after.json"]
    checks, errors = [], []
    for name in selected:
        p = CACHE / name
        entry = by_name.get(name)
        rec = {"filename": name, "exists": p.is_file(), "in_source_manifest": entry is not None}
        if not p.is_file() or entry is None:
            errors.append(f"missing file or source-manifest record: {name}")
        else:
            got = sha(p)
            rec.update({"bytes": p.stat().st_size, "expected_bytes": entry.get("bytes"),
                        "sha256": got, "expected_sha256": entry.get("sha256"),
                        "sha_pass": got == entry.get("sha256") and p.stat().st_size == entry.get("bytes")})
            if not rec["sha_pass"]:
                errors.append(f"SHA/size mismatch: {name}")
        checks.append(rec)

    plan = rows(CACHE / "feature_row_plan.csv")
    plan_by_split = {sp: [r for r in plan if r.get("split") == sp] for sp in SPLITS}
    all_ids: set[str] = set()
    split_info = {}
    labels_by_split = {}
    for sp, expected_n in SPLITS.items():
        labels = rows(CACHE / f"labels_{sp}.csv")
        side = rows(CACHE / f"H_fresh_{sp}_sidecar.csv")
        ps = plan_by_split[sp]
        ids = [r.get("feature_row_id", "") for r in side]
        split_errors = []
        if not (len(labels) == len(side) == len(ps) == expected_n):
            split_errors.append(f"row counts labels/sidecar/plan={len(labels)}/{len(side)}/{len(ps)}, expected {expected_n}")
        if any(not x for x in ids) or len(set(ids)) != len(ids):
            split_errors.append("empty or duplicate feature_row_id")
        if all_ids.intersection(ids):
            split_errors.append("feature_row_id overlaps another split")
        all_ids.update(ids)
        for name, rr in (("labels", labels), ("plan", ps)):
            if [r.get("feature_row_id") for r in rr] != ids:
                split_errors.append(f"{name}/sidecar row-ID order mismatch")
        if [r.get("canonical_fact_id") for r in labels] != [r.get("canonical_fact_id") for r in side]:
            split_errors.append("labels/sidecar canonical ID mismatch")
        if [r.get("canonical_fact_id") for r in ps] != [r.get("canonical_fact_id") for r in side]:
            split_errors.append("plan/sidecar canonical ID mismatch")
        for i, (lab, sc, pl) in enumerate(zip(labels, side, ps)):
            if any(int(r.get("split_row_index", -1)) != i for r in (lab, sc, pl)):
                split_errors.append(f"noncanonical split_row_index at row {i}")
                break
        joint = np.asarray([int(r["joint_class_id"]) for r in labels], dtype=np.int64)
        side_joint = np.asarray([int(r["joint_label"]) for r in side], dtype=np.int64)
        if not np.array_equal(joint, side_joint):
            split_errors.append("Joint label mismatch between labels and sidecar")
        if any(int(r.get("split_row_index", -1)) != i for i, r in enumerate(labels)):
            split_errors.append("noncanonical label split_row_index")
        feature_meta = {}
        for rep in ("R1", "R2"):
            p = CACHE / f"{rep}_{sp}.npy"
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            ok = arr.shape == (expected_n, 256) and arr.dtype == np.float32
            feature_meta[rep] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                 "source_row_count": by_name.get(p.name, {}).get("row_count"),
                                 "metadata_pass": ok}
            if not ok:
                split_errors.append(f"{rep} shape/dtype mismatch")
        known = joint >= 0
        labels_by_split[sp] = joint
        split_info[sp] = {"rows": len(labels), "unique_feature_row_ids": len(set(ids)),
                          "joint_known": int(known.sum()), "joint_unknown": int((~known).sum()),
                          "joint_class_count": int(len(np.unique(joint[known]))) if known.any() else 0,
                          "R1_R2_metadata": feature_meta, "errors": split_errors,
                          "pass": not split_errors}
        errors.extend(f"{sp}: {e}" for e in split_errors)

    with np.load(CACHE / "w1_svd_matrices.npz", allow_pickle=False) as z:
        U = np.asarray(z["U"], dtype=np.float64)
        S = np.asarray(z["S"], dtype=np.float64).reshape(-1)
        Vh = np.asarray(z["Vh"], dtype=np.float64)
    orth_err = float(np.max(np.abs(U.T @ U - np.eye(256))))
    mu1 = np.asarray(np.load(CACHE / "R1_train.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64).mean(0)
    mu2 = np.asarray(np.load(CACHE / "R2_train.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64).mean(0)
    sd1 = np.asarray(np.load(CACHE / "R1_train.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64).std(0)
    sd2 = np.asarray(np.load(CACHE / "R2_train.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64).std(0)
    sd1[sd1 < 1e-8] = 1.0
    sd2[sd2 < 1e-8] = 1.0
    # For row-vector coordinates, raw R2 = R1 @ U.T.
    B = (sd1[:, None] * U.T) / sd2[None, :]
    offset = (mu1 @ U.T - mu2) / sd2
    transform_errors = {}
    x1v = x2v = None
    for sp in ("train", "val"):
        x1s = (np.asarray(np.load(CACHE / f"R1_{sp}.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64) - mu1) / sd1
        x2s = (np.asarray(np.load(CACHE / f"R2_{sp}.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64) - mu2) / sd2
        err = x2s - (x1s @ B + offset)
        transform_errors[sp] = {"max_abs": float(np.max(np.abs(err))), "mean_abs": float(np.mean(np.abs(err)))}
        if sp == "val":
            x1v, x2v = x1s, x2s
    rng = np.random.default_rng(314159)
    W = rng.normal(size=(int(labels_by_split["train"].max()) + 1, 256))
    b = rng.normal(size=W.shape[0])
    W2 = np.linalg.solve(B, W.T).T
    b2 = b - offset @ W2.T
    logit_err = float(np.max(np.abs((x1v @ W.T + b) - (x2v @ W2.T + b2))))
    train_y = labels_by_split["train"]
    val_y = labels_by_split["val"]
    class_ids = np.unique(train_y[train_y >= 0])
    joint_info = {"train_known": int((train_y >= 0).sum()),
                  "train_classes": int(len(class_ids)),
                  "class_ids_contiguous": bool(np.array_equal(class_ids, np.arange(len(class_ids)))),
                  "val_known_in_train_classes": int(np.isin(val_y, class_ids).sum()),
                  "testA_known_in_train_classes": int(np.isin(labels_by_split["testA"], class_ids).sum()),
                  "testB_known_in_train_classes": int(np.isin(labels_by_split["testB"], class_ids).sum())}
    if not joint_info["class_ids_contiguous"] or joint_info["val_known_in_train_classes"] == 0:
        errors.append("Joint labels do not satisfy frozen contiguous class mapping / validation availability")
    if U.shape != (256, 256) or S.shape != (256,) or Vh.shape != (256, 2048) or orth_err > 1e-5:
        errors.append("SVD dimensions or U orthogonality mismatch")
    if max(v["max_abs"] for v in transform_errors.values()) > 5e-4 or logit_err > 1e-4:
        errors.append("standardized induced transform / classifier-transfer numeric check failed")

    result = {"experiment": "O3b-J3E Stage 0 input integrity and standardized transform audit",
              "cache_path": str(CACHE), "output_path": str(OUT),
              "source_certification_verdict": manifest.get("certification_verdict"),
              "selected_file_count": len(selected), "selected_files": checks,
              "splits": split_info, "joint_label_integrity": joint_info,
              "svd": {"U_shape": list(U.shape), "S_shape": list(S.shape), "Vh_shape": list(Vh.shape),
                      "U_orthogonality_max_abs_error": orth_err,
                      "raw_definition": "R2 = R1 @ U.T",
                      "standardized_definition": "R2_std = R1_std @ B + offset; B=diag(sd_R1) U.T diag(1/sd_R2)",
                      "B_condition_number": float(np.linalg.cond(B)),
                      "train_standardized_transform_error": transform_errors["train"],
                      "validation_standardized_transform_error": transform_errors["val"],
                      "synthetic_bidirectional_affine_classifier_transfer_max_logit_error": logit_err},
              "test_splits_used_for_model_selection_or_metrics": False,
              "test_features_used_for_standardization": False, "checkpoint_modified": False,
              "errors": errors, "pass": not errors}
    (OUT / "input_integrity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "stage0_audit.md").write_text(
        "# O3b-J3E Stage 0 input integrity\n\n"
        f"Verdict: **{'PASS' if not errors else 'FAIL'}**. Joint-only label integrity, certified R1/R2 files, row order, and standardized affine equivalence were audited; no probe was trained.\n\n"
        f"R1/R2 standardization used train-only means/stds. The induced standardized transform has validation max absolute reconstruction error {transform_errors['val']['max_abs']:.4g}; synthetic affine classifier-transfer max logit error {logit_err:.4g}.\n\n"
        f"Joint classes: {joint_info['train_classes']}; validation known {joint_info['val_known_in_train_classes']}; TestA known {joint_info['testA_known_in_train_classes']}; TestB known {joint_info['testB_known_in_train_classes']}.\n\n"
        + ("Errors:\n" + "\n".join(f"- {e}" for e in errors) if errors else "No integrity discrepancies detected. Test splits were used only for hashes, row identity, and known-label counts; no Test metric was computed or used for selection.") + "\n",
        encoding="utf-8")
    print(json.dumps({"stage": "stage0", "pass": not errors, "checked_files": len(checks),
                      "errors": errors, "joint_info": joint_info,
                      "transform_max_abs": transform_errors["val"]["max_abs"], "transfer_logit_error": logit_err}, ensure_ascii=False), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
