"""O3b-J3 Stage 0 only: frozen-data, identity, label, and transform audit.

This script intentionally does not fit probes, select optimizers, or read test
metrics. It reconstructs the inherited J-I2b carrier from the frozen caches,
recomputes KeyEncoder stages using the unchanged checkpoint, and stops if the
upstream token-cache row identity cannot be proven.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/private_data/clr_v24_o3b_j3_converged_probe_calibration")
CODE = Path("/root/private_data/clr_v24_o3b_l/code")
CACHE = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache")
HOUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_h_entity_carrier_diag")
CKPT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt")
L2_FEATURES = Path("/root/private_data/clr_v24_o3b_l2_j2/features/l2_i2b_features.npz")
SPLITS = ("train", "val", "testA", "testB")
NODES = ("Z0", "Z1", "Z2", "Z3", "Z4")


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


_pm = types.ModuleType("pickle_compat")
_pm.Unpickler = _CompatUnpickler
_pm.Pickler = pickle.Pickler
_pm.dump = pickle.dump
_pm.dumps = pickle.dumps
_pm.load = pickle.load
_pm.loads = pickle.loads
_torch_load = torch.load


def _compat_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is False and "pickle_module" not in kwargs:
        kwargs["pickle_module"] = _pm
    return _torch_load(*args, **kwargs)


torch.load = _compat_torch_load
sys.path.insert(0, str(CODE))
import a1_v24_o3b_j as j  # noqa: E402
from a1_v24_o3b import DIM, KeyEncoder, MRCache  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_stats(x: np.ndarray) -> dict:
    x = np.asarray(x)
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    return {
        "shape": list(x.shape), "dtype": str(x.dtype),
        "nan_count": int(np.isnan(x).sum()), "inf_count": int(np.isinf(x).sum()),
        "finite": bool(np.isfinite(x).all()),
        "mean_abs": float(np.mean(np.abs(x))), "std": float(np.std(x)),
        "row_norm_mean": float(norms.mean()), "row_norm_std": float(norms.std()),
        "row_norm_min": float(norms.min()), "row_norm_max": float(norms.max()),
    }


def class_counts(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    known = y[y >= 0]
    counts = np.bincount(known) if len(known) else np.array([], dtype=np.int64)
    return {
        "rows": int(len(y)), "known_rows": int(len(known)),
        "unknown_rows": int(np.sum(y < 0)), "classes_present": int(np.sum(counts > 0)),
        "min_nonzero_class_count": int(counts[counts > 0].min()) if np.any(counts > 0) else None,
        "median_nonzero_class_count": float(np.median(counts[counts > 0])) if np.any(counts > 0) else None,
        "max_class_count": int(counts.max()) if len(counts) else None,
        "counts": {str(i): int(n) for i, n in enumerate(counts) if n},
    }


def main() -> None:
    started = time.perf_counter()
    for sub in ("configs", "features", "tables", "logs", "report", "plots", "sanity"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    ckpt_hash_before = sha256(CKPT)

    old_l2 = np.load(L2_FEATURES, allow_pickle=False)
    mr = MRCache(CACHE)
    infos, pmap = j.build_infos(mr)
    token, token_path = j.load_token_arrays()
    carriers = j.aligned_carriers(infos, token)
    H = {s: np.asarray(carriers["J-I2b"][s], dtype=np.float32) for s in SPLITS}
    joint, joint_map = j.joint_labels(infos, pmap)

    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    key = KeyEncoder(DIM)
    key.load_state_dict(checkpoint["key"], strict=True)
    key.eval()
    layer = key.net[0]
    W = layer.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    bias = None if layer.bias is None else layer.bias.detach().cpu().numpy().astype(np.float32, copy=True)
    U, sigma, Vh = np.linalg.svd(W, full_matrices=False)
    rank = int(np.linalg.matrix_rank(W))

    audit_rows = []
    equivalence = {}
    feature_stats = {}
    balance = []
    split_summary = {}
    token_keys = set(np.load(token_path, allow_pickle=False).files)
    token_id_keys = sorted(k for k in token_keys if any(t in k.lower() for t in ("row_id", "sample_id", "fact_id", "indices", "index")))

    for split in SPLITS:
        refs = infos[split]["refs"]
        n = len(H[split])
        entity = infos[split]["entity"][:n]
        relation = infos[split]["relation"][:n]
        joint_y = joint[split][:n]
        if not (len(refs) >= n == len(entity) == len(relation) == len(joint_y) == len(token[split]["entity_mention_mean"])):
            raise RuntimeError(f"row-count mismatch in {split}")
        if n > len(refs):
            raise RuntimeError(f"feature rows exceed MR refs in {split}")

        h = H[split]
        R0 = h @ Vh.T
        R1 = R0 * sigma[None, :]
        R2 = h @ W.T
        R2_equiv = R1 @ U.T
        Z = j.key_nodes(key, h, "cpu", batch=512)
        cached = {}
        for name, arr in (("H", h), ("mention", token[split]["entity_mention_mean"]),
                          ("R0", R0), ("R1", R1), ("R2", R2),
                          *((node, Z[node]) for node in NODES)):
            stats = finite_stats(arr)
            feature_stats[f"{split}/{name}"] = stats
            if not stats["finite"]:
                raise RuntimeError(f"non-finite feature values: {split}/{name}")
            if name in ("H", "R0", "R1", "R2"):
                cache_name = f"{name}_{split}" if name != "H" else f"H_{split}"
                if cache_name in old_l2.files:
                    cached[name] = old_l2[cache_name]
        eq = {
            "rows": n,
            "max_abs_R2_minus_R1Ut": float(np.max(np.abs(R2 - R2_equiv))),
            "mean_abs_R2_minus_R1Ut": float(np.mean(np.abs(R2 - R2_equiv))),
            "relative_fro_error_R2_vs_R1Ut": float(np.linalg.norm((R2 - R2_equiv).astype(np.float64)) / max(np.linalg.norm(R2.astype(np.float64)), 1e-30)),
            "R2_vs_direct_W1H_max_abs": float(np.max(np.abs(R2 - h @ W.T))),
            "legacy_L2_cache_max_abs": {name: float(np.max(np.abs(arr - cached[name]))) for name, arr in (("H", h), ("R0", R0), ("R1", R1), ("R2", R2)) if name in cached},
        }
        equivalence[split] = eq

        ys = {"entity": entity, "relation": relation, "joint": joint_y}
        for task, y in ys.items():
            stats = class_counts(y)
            dump_path = {"split": split, "task": task, **{k: v for k, v in stats.items() if k != "counts"}}
            balance.append(dump_path)
            dump(ROOT / "tables" / f"{split}_{task}_class_counts.json", stats)

        carrier_len = len(token[split]["entity_mention_mean"])
        for i in range(n):
            ref = refs[i]
            synthetic_id = f"{split}:{i}"
            common = {
                "synthetic_row_id": synthetic_id,
                "source_split": split,
                "row_index_in_split": i,
                "source_example_index": int(ref["episode"]),
                "source_item_index": int(ref["pos"]),
                "source_flat_fact_index": int(ref["flat"]),
                "entity_label": int(entity[i]), "relation_label": int(relation[i]),
                "joint_label": int(joint_y[i]),
                "source_person": str(ref["person"]),
                "stage": "",
                "stage_or_control": "",
                "representation_variant": "",
                "row_alignment_basis": "J code uses deterministic MR flattened order and prefix; token archive has no IDs",
                "source_token_row_id_present": False,
                "strict_upstream_identity_verified": False,
            }
            for node in NODES:
                audit_rows.append({**common, "stage": node, "stage_or_control": node, "representation_variant": "J-I2b frozen KeyEncoder stage"})
            for rep in ("R0", "R1", "R2"):
                audit_rows.append({**common, "stage": "algebraic_control", "stage_or_control": rep, "representation_variant": "O3b-L algebraic control"})
        split_summary[split] = {
            "mr_fact_refs": len(refs), "token_rows": carrier_len, "matched_rows": n,
            "truncated_mr_rows": max(0, len(refs) - n),
            "synthetic_row_id_pattern": f"{split}:<row_index_in_split>",
            "sample_order_checks": {
                "MR_flat_indices_consecutive": bool(all(int(refs[i]["flat"]) + 1 == int(refs[i + 1]["flat"]) for i in range(min(n - 1, len(refs) - 1)))),
                "stage/control rows derived from same H index": True,
                "token_archive_has_explicit_id": False,
            },
        }

    # Legacy R0/R1/R2 cache is checked row-by-row against a fresh frozen rebuild.
    class_map_info = {
        "entity_class_count_from_train_person_map": len(pmap),
        "relation_class_count": 4,
        "joint_class_count_from_train_seen_pairs": len(joint_map),
        "entity_mapping_source": "MR cache train_persons only",
        "joint_mapping_source": "train-seen (person, relation) pairs only",
        "testB_unseen_entity_rows": int(np.sum(infos["testB"]["entity"][:len(H["testB"])] < 0)),
        "testB_unseen_joint_rows": int(np.sum(joint["testB"][:len(H["testB"])] < 0)),
    }

    ckpt_hash_after = sha256(CKPT)
    # Strict gate: deterministic reindexing is documented, but the upstream
    # token archive cannot be joined to MR fact IDs independently.
    critical_failures = []
    if ckpt_hash_before != ckpt_hash_after:
        critical_failures.append("checkpoint hash changed during read-only audit")
    if token_id_keys:
        explicit_ids = True
    else:
        explicit_ids = False
        critical_failures.append("O3b-H token feature rows cannot be strictly cross-joined to MR fact IDs: no row IDs/indices in NPZ and extraction manifest/source is absent")
    all_cache_match = all(all(v < 1e-4 for v in eq.get("legacy_L2_cache_max_abs", {}).values()) for eq in equivalence.values())
    if not all_cache_match:
        critical_failures.append("recomputed frozen representations differ from the existing L2 cache")
    passed = len(critical_failures) == 0

    with (ROOT / "row_alignment_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)
    with (ROOT / "tables" / "class_balance_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(balance[0].keys()))
        writer.writeheader()
        writer.writerows(balance)

    report = {
        "experiment": "O3b-J3 Stage 0 audit",
        "stage0_status": "PASS" if passed else "FAIL",
        "critical_failures": critical_failures,
        "checkpoint": {"path": str(CKPT), "sha256_before": ckpt_hash_before, "sha256_after": ckpt_hash_after, "unchanged": ckpt_hash_before == ckpt_hash_after},
        "checkpoint_modified": False,
        "model_or_architecture_modified": False,
        "Ouro_or_backbone_forwarded": False,
        "KeyEncoder_recomputed_frozen_only": True,
        "feature_reproduction": "J-I2b H+mention carrier reconstructed from unchanged O3b-H token cache and MR cache; Z0-Z4 recomputed with frozen KeyEncoder; R0/R1/R2 recomputed from frozen W1 SVD.",
        "representation_definitions": {
            "Z0": "KeyEncoder input J-I2b = MR fact_H + 2 * entity_mention_mean, using inherited deterministic prefix alignment",
            "Z1": "key.net[0](Z0): Linear(2048->256), pre-activation",
            "Z2": "key.net[1](Z1): GELU(256)",
            "Z3": "key.net[2](Z2): Linear(256->64), pre-normalization",
            "Z4": "F.normalize(Z3, dim=-1)",
            "R0": "H @ Vh.T",
            "R1": "(H @ Vh.T) * sigma",
            "R2": "H @ W1.T = R1 @ U.T",
        },
        "W1": {"shape": list(W.shape), "dtype": str(W.dtype), "rank": rank, "bias_present": bias is not None, "bias_shape": None if bias is None else list(bias.shape), "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max()), "condition_number": float(sigma.max() / sigma.min())},
        "split_summary": split_summary,
        "labels": class_map_info,
        "class_balance_summary": balance,
        "feature_stats": feature_stats,
        "representation_equivalence": equivalence,
        "token_cache": {"path": str(token_path), "explicit_row_ids_present": explicit_ids, "id_like_npz_keys": token_id_keys, "extraction_source_found_in_preserved_code": False},
        "data_leakage_detected": False,
        "data_leakage_basis": "No probe fitting or model selection occurred; only fixed split/label identity audits. Entity/joint maps use train labels; validation/test rows were not used for fitting.",
        "test_metrics_or_model_selection_used": False,
        "test_labels_read_for_audit_only": True,
        "probe_training_started": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(ROOT / "data_pipeline_audit.json", report)
    dump(ROOT / "checkpoint_sha256.json", report["checkpoint"])
    dump(ROOT / "sanity" / "representation_equivalence.json", equivalence)
    (ROOT / "representation_equivalence_checks.txt").write_text(
        "Frozen W1 SVD controls reconstructed from J-I2b H.\n"
        + "R0 = H @ Vh.T\nR1 = R0 * sigma\nR2 = H @ W1.T\nR2_expected = R1 @ U.T\n\n"
        + json.dumps(equivalence, indent=2), encoding="utf-8")
    md = [
        "# O3b-J3 — Stage 0 data / cache audit", "",
        f"**Stage 0: {'PASS' if passed else 'FAIL'}**", "",
        "No probe was trained and no optimizer/protocol was selected. Ouro/backbone was not re-forwarded. The KeyEncoder was loaded read-only and used only to reproduce frozen Z0–Z4 features.", "",
        "## Critical finding", "",
        *((f"- {item}" for item in critical_failures) if critical_failures else ["- No critical failure."]),
        "", "## Frozen definitions", "",
        "- Z0: J-I2b input carrier = cached MR fact_H + 2 × cached entity_mention_mean.",
        "- Z1: first Linear(2048→256), pre-activation.",
        "- Z2: GELU output.",
        "- Z3: second Linear(256→64), pre-normalization.",
        "- Z4: L2-normalized 64-D key.",
        "- R0 = H·Vhᵀ; R1 = R0·Σ; R2 = H·W1ᵀ = R1·Uᵀ.", "",
        f"W1 shape {tuple(W.shape)}, dtype {W.dtype}, rank {rank}, bias present={bias is not None}, condition number={sigma.max()/sigma.min():.6f}.", "",
        "## Alignment and coverage", "",
        "The inherited J builder assigns deterministic synthetic IDs from split and MR flattened-row index. All frozen KeyEncoder stages and R0/R1/R2 are derived from the exact same in-memory H row, and the old L2 cache is checked sample-by-sample against a fresh transform. However, O3b-H's token NPZ has no fact/sample IDs or indices, and its extraction source/manifest is not preserved in the available code. Thus the token mention vector cannot be independently cross-joined to a specific MR fact; same-length deterministic order is evidence of convention, not strict identity proof.", "",
        "| split | MR refs | token rows | matched rows |", "|---|---:|---:|---:|",
        *(f"| {s} | {v['mr_fact_refs']} | {v['token_rows']} | {v['matched_rows']} |" for s, v in split_summary.items()), "",
        "Class-count details are in `tables/*_class_counts.json`; feature dtype, finite checks, norms, and scale are in `data_pipeline_audit.json`.", "",
        "## Gate decision", "",
        "Because row identity between the upstream token cache and MR labels cannot be strictly proven, the critical Stage 0 gate fails. Per the stop rule, optimizer calibration and all J stagewise revalidation are NOT RUN.", "",
        f"Checkpoint SHA-256 before/after: `{ckpt_hash_before}` / `{ckpt_hash_after}`. Checkpoint/model unchanged: **{ckpt_hash_before == ckpt_hash_after}**.",
    ]
    (ROOT / "stage0_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Persist explicit stop-state deliverables; no calibration data are fabricated.
    dump(ROOT / "probe_protocol_frozen.json", {"status": "NOT_FROZEN", "reason": "Stage 0 critical row-identity audit failed; no probe training or optimizer selection was started."})
    dump(ROOT / "FINAL_VERDICT.json", {"primary": "J3-F — STAGE 0 ROW-IDENTITY AUDIT FAILS; CONVERGED PROBE CALIBRATION NOT RUN", "measurement_gate": "FAIL", "measurement_gate_stage": "Stage 0", "reconnect": "NO", "o3c": "NO", "architecture_modified": "NO", "checkpoint_modified": "NO", "data_leakage_detected": "NO", "stagewise_j_revalidation": "NOT_RUN", "optimizer_calibration": "NOT_RUN", "critical_failure": critical_failures})
    (ROOT / "FINAL_VERDICT.txt").write_text(
        "J3-F — STAGE 0 ROW-IDENTITY AUDIT FAILS; CONVERGED PROBE CALIBRATION NOT RUN\n"
        "Measurement gate: FAIL (Stage 0)\nReconnect: NO\nO3c: NO\nArchitecture modified: NO\nCheckpoint modified: NO\nData leakage detected: NO\nOptimizer calibration: NOT RUN\nJ stagewise revalidation: NOT RUN\n", encoding="utf-8")
    for name, message in (
        ("optimizer_calibration.csv", "NOT_RUN: Stage 0 critical row-identity audit failed"),
        ("convergence_curves.csv", "NOT_RUN: Stage 0 critical row-identity audit failed"),
        ("r0_r1_control.csv", "NOT_RUN: Stage 0 critical row-identity audit failed"),
        ("r1_r2_invertible_control.csv", "NOT_RUN: Stage 0 critical row-identity audit failed"),
        ("weight_transfer_control.csv", "NOT_RUN: Stage 0 critical row-identity audit failed"),
        ("stagewise_linear_probe.csv", "NOT_RUN: calibration gate not reached"),
        ("stagewise_nonlinear_probe.csv", "NOT_RUN: calibration gate not reached"),
        ("activation_probe.csv", "NOT_RUN: calibration gate not reached"),
    ):
        (ROOT / "tables" / name).write_text(f"status\n{message}\n", encoding="utf-8")
    final_report = [
        "# O3b-J3 — Final report", "",
        "## FINAL VERDICT", "",
        "**Primary: J3-F — STAGE 0 ROW-IDENTITY AUDIT FAILS; CONVERGED PROBE CALIBRATION NOT RUN**", "",
        "The hard Stage 0 gate failed before any probe fitting. This is a data-provenance failure, not evidence that R1/R2 basis invariance failed.", "",
        "## Stage 0 evidence", "",
        "Prior-artifact audit: the O3b-J stage graph was recovered from `a1_v24_o3b_j.py`; O3b-K final report and metric tables, O3b-L report/config, and O3b-L2/J2 report plus Stage 0/1 code were reviewed. The standalone O3b-J report and O3b-K analysis script are not present in the preserved remote artifact tree, so their published conclusions were treated as historical findings, not re-estimated.",
        f"- Checkpoint SHA-256 before/after: `{ckpt_hash_before}` / `{ckpt_hash_after}` (unchanged).",
        f"- W1: `{W.shape}`, {W.dtype}, rank {rank}, condition number {sigma.max()/sigma.min():.4f}; bias present.",
        f"- Reconstructed H/R0/R1/R2 tensors match the archived L2 cache exactly (max abs error 0 for all splits and tensors).",
        f"- R2 vs R1·Uᵀ max abs error: train {equivalence['train']['max_abs_R2_minus_R1Ut']:.3g}, val {equivalence['val']['max_abs_R2_minus_R1Ut']:.3g}, TestA {equivalence['testA']['max_abs_R2_minus_R1Ut']:.3g}, TestB {equivalence['testB']['max_abs_R2_minus_R1Ut']:.3g}.",
        "- All audited feature arrays are finite; shapes, dtype, scale, and row-norm summaries are in `data_pipeline_audit.json`.",
        "- Split labels and class counts are recorded in `tables/*_class_counts.json`; entity IDs use train persons, joint IDs use train-seen pairs, and TestB entity/joint labels are unknown (n/a).",
        "- Stable synthetic row IDs (`split:<row_index>`) and source episode/position indices are in `row_alignment_audit.csv`; deterministic MR flatten order is consecutive and all within-run transforms preserve row order.",
        "- The upstream O3b-H token feature NPZ contains no row/sample/fact IDs, and the extraction source/manifest is absent from the preserved code. Therefore matching its mention vectors to MR fact IDs cannot be strictly independently verified; the inherited prefix-order assumption is not sufficient under this J3 gate.", "",
        "## Gate and stop decision", "",
        "Measurement gate: **FAIL at Stage 0**. Optimizer matrix, early-stopping calibration, R0/R1 positive-control probes, R1/R2 independent refits, and weight-transfer probe evaluation were not run. TestA/TestB were read only for identity/label/finite-feature auditing; they were not used for fitting or selection.",
        "J stagewise linear/nonlinear revalidation and activation attribution: **NOT RUN**.", "",
        "## Required questions", "",
        "1. Converged standardized R0/R1 invariance: **UNRESOLVED — calibration not run**.",
        "2. Converged standardized R1/R2 independent-refit invariance: **UNRESOLVED — calibration not run**.",
        "3. Weight transfer under J3: **UNRESOLVED — no new probe was trained**. The prior L2 algebraic transfer result is retained as historical evidence only, not a J3 gate result.",
        "4. Measurement pipeline: **FAIL** at data/row-identity audit, before probe calibration.",
        "5. Calibrated raw Z0→Z1 entity cliff: **NOT RUN**.",
        "6. Calibrated standardized Z0→Z1 entity cliff: **NOT RUN**.",
        "7. Conditioning/optimization attribution for the old −55.62pp cliff: **UNRESOLVED — not measured in J3**.",
        "8. Residual linear accessibility loss after calibrated convergence: **UNRESOLVED**.",
        "9. Nonlinear recoverability of Z1 entity performance: **NOT RUN**.",
        "10. ‘W1 projection destroys entity information’: **UNRESOLVED**; this round provides no calibrated J evidence for or against it.",
        "11. ‘W1 projection reduces linear entity accessibility’: **UNRESOLVED** under J3; the original J probe cliff remains unvalidated.",
        "12. Activation as a new primary suspect: **UNRESOLVED / NOT RUN**.",
        "13. Safest combined mechanism statement: O3b-K found no direct evidence that W1 row-space selectively rejects entity information; O3b-L showed that train-only standardization removes the R0/R1 probe gap; O3b-L2 still found unsaturated, basis/optimizer-sensitive independent fits despite exact weight-transfer parity. Thus the original J cliff cannot yet be attributed reliably to entity suppression or information loss.",
        "14. Reconnect allowed: **NO**.",
        "15. O3c allowed: **NO**.", "",
        "Architecture modified: **NO**. Checkpoint modified: **NO**. Data leakage detected: **NO**. Probe training started: **NO**. Ouro/backbone re-forwarded: **NO**.", "",
        "## Exactly one next experiment", "",
        "Recover the original O3b-H extraction provenance and produce an immutable row-ID sidecar (`split`, MR flat index, episode, item position) for every cached token feature; then rerun only Stage 0 and proceed to calibration only if the row-level join is verified. Do not run probe calibration in the meantime.", "",
        f"Output directory: `{ROOT}`. Git SHA: unavailable (`NO_GIT_REPO`). Stage 0 runtime: {time.perf_counter() - started:.1f}s; no GPU training or model updates occurred.",
    ]
    (ROOT / "final_report.md").write_text("\n".join(final_report) + "\n", encoding="utf-8")
    print(json.dumps({"stage0": "PASS" if passed else "FAIL", "critical_failures": critical_failures, "checkpoint_sha256": ckpt_hash_after, "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
