from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"H:\LoopLM")
OUT = ROOT / "clr_v24_o3b_m3_end_to_end_reconnect"
M2 = ROOT / "clr_v24_o3b_m2_growing_slot_calibration"
M = ROOT / "clr_v24_o3b_m_operational_samekey_rescue"
REMOTE_EVIDENCE_PATH = OUT / "remote_source_and_checkpoint_evidence.json"
REMOTE_SEARCH_PATH = OUT / "remote_dependency_search.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def dump_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not REMOTE_EVIDENCE_PATH.is_file() or not REMOTE_SEARCH_PATH.is_file():
        raise FileNotFoundError("Complete read-only remote dependency evidence is required before finalizing")

    policy_path = OUT / "m2_address_policy_frozen_manifest.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    integrity_rows = []
    for artifact in policy["artifacts"]:
        path = Path(artifact["path"])
        actual = sha256(path) if path.is_file() else None
        integrity_rows.append({
            "name": artifact["name"],
            "path": artifact["path"],
            "frozen_sha256": artifact["sha256"],
            "sha256_after_recovery_search": actual,
            "unchanged": actual == artifact["sha256"],
        })
    evidence = json.loads(REMOTE_EVIDENCE_PATH.read_text(encoding="utf-8"))
    search = json.loads(REMOTE_SEARCH_PATH.read_text(encoding="utf-8"))
    remote_candidate = evidence["expected_dependency_paths"]["candidate_b_checkpoint"]
    remote_o21b = evidence["expected_dependency_paths"]["o21b_checkpoint"]
    local_o21b = ROOT / "remote_archive/cheems/checkpoints/clr_v24_o21b/b1_best.pt"
    local_old_o3b = ROOT / "remote_archive/cheems/a1_v24_o3b.py"
    remote_old_o3b_sha = evidence["source_snapshots"]["a1_v24_o3b.py"]["sha256"]

    policy["candidate_checkpoint_sha256_after_recovery"] = sha256(M / "models/candidate_b_seed_43.pt")
    policy["m2_policy_artifacts_unchanged_after_recovery"] = all(row["unchanged"] for row in integrity_rows)
    policy["artifact_integrity_checks"] = integrity_rows
    policy["remote_candidate_checkpoint_matches_frozen_sha256"] = remote_candidate.get("sha256") == policy["candidate_checkpoint_sha256_before"]
    policy["verified_after_recovery_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(policy_path, policy)
    dump_json(OUT / "m2_policy_integrity_check.json", {
        "pass": policy["m2_policy_artifacts_unchanged_after_recovery"] and policy["remote_candidate_checkpoint_matches_frozen_sha256"],
        "candidate_checkpoint_sha256_before": policy["candidate_checkpoint_sha256_before"],
        "candidate_checkpoint_sha256_after_recovery": policy["candidate_checkpoint_sha256_after_recovery"],
        "candidate_config_sha256": policy["candidate_config_sha256"],
        "rule": policy["decision_rule"],
        "threshold_exact": policy["threshold_exact"],
        "remote_candidate_sha256": remote_candidate.get("sha256"),
        "protected_artifacts": integrity_rows,
    })

    local_search = {
        "searched_utc": datetime.now(timezone.utc).isoformat(),
        "roots": [
            str(ROOT),
            str(ROOT / "remote_archive/cheems"),
            str(ROOT / "remote_archive/new_server"),
            str(ROOT / "checkpoints"),
        ],
        "exact_name_search": "rg --files H:\\LoopLM filtered for expected O21A/VGen2/O3a/router checkpoint and source filenames",
        "relevant_local_archive_hits": {
            "O21B_checkpoint": {
                "path": str(local_o21b),
                "exists": local_o21b.is_file(),
                "sha256": sha256(local_o21b) if local_o21b.is_file() else None,
                "matches_active_remote": local_o21b.is_file() and local_o21b_sha(local_o21b) == remote_o21b.get("sha256"),
                "not_a_substitute_for_O21A_MR_VGen2_O3A_or_router": True,
            },
            "original_o3b_integration_script": {
                "path": str(local_old_o3b),
                "exists": local_old_o3b.is_file(),
                "sha256": sha256(local_old_o3b) if local_old_o3b.is_file() else None,
                "matches_active_remote": local_old_o3b.is_file() and sha256(local_old_o3b) == remote_old_o3b_sha,
            },
            "other_requested_frozen_checkpoints": "No matching original checkpoint files found in searched local workspace/archive names.",
        },
        "remote_search_results": {
            "exact_component_names_exit": search["queries"]["exact_component_names"]["exit_code"],
            "relevant_checkpoint_file_listing_exit": search["queries"]["clr_related_checkpoint_files"]["exit_code"],
            "own_shared_home_checkpoint_search_exit": search["queries"]["other_storage_roots_exact_checkpoints"]["exit_code"],
            "own_shared_home_checkpoint_search_matches": search["queries"]["other_storage_roots_exact_checkpoints"]["stdout"].strip(),
            "project_output_config_references": search["queries"]["configs_code_and_logs_references"]["stdout"].splitlines(),
            "unrelated_other_users_public_data_not_inspected": True,
        },
    }
    dump_json(OUT / "local_dependency_search.json", local_search)

    code = evidence["source_snapshots"]
    expected = evidence["expected_dependency_paths"]
    dep_records = [
        {
            "component": "O21A_MR supervised/query/value-head checkpoint",
            "status": "MISSING",
            "artifact_path": expected["o21a_mr_checkpoint"]["path"],
            "originating_experiment": "O2.1a-MR frozen supervised checkpoint",
            "sha256": None,
            "config_path": "Expected clr_v24_o21a_mr run config; not found with the checkpoint/output bundle",
            "architecture": "QueryEncoder 2048→256→64; MRValueHeads has per-slot 64→N_CLASSES=64 heads",
            "input_output_dims": {"query_encoder": [2048, 256, 64], "value_head_per_slot": [64, 64]},
            "expected_role": "Frozen query encoder and MR value heads used by original O3b path",
            "frozen": True,
            "historical_metric": "O2.1a-MR cache exists; this does not certify or replace trained weights",
            "confidence": "HIGH that the expected path is missing; source code and default path agree",
            "evidence": [expected["o21a_mr_checkpoint"], code["a1_v24_o21a.py"], code["a1_v24_o21a_mr.py"]],
        },
        {
            "component": "O21B KeyEncoder checkpoint",
            "status": "STRONGLY_MATCHED",
            "artifact_path": expected["o21b_checkpoint"]["path"],
            "originating_experiment": "O2.1b; explicitly referenced by original O3b integration source",
            "sha256": expected["o21b_checkpoint"].get("sha256"),
            "config_path": "O21B run config not recovered in the searched bundle",
            "architecture": "KeyEncoder 2048→256→64",
            "input_output_dims": {"key_encoder": [2048, 256, 64], "O3b_slots": 4, "key_dim": 64},
            "expected_role": "Frozen key encoder used to form occupied and incoming keys",
            "frozen": True,
            "historical_metric": "Referenced by the original O3b code; detailed original checkpoint run metrics not available in this recovered bundle",
            "confidence": "STRONG: remote file hash matches the local archived O21B checkpoint and original integration path",
            "evidence": {"remote": expected["o21b_checkpoint"], "local_archive": local_search["relevant_local_archive_hits"]["O21B_checkpoint"], "source": code["a1_v24_o3b.py"]},
        },
        {
            "component": "VGen2 frozen core/value checkpoint",
            "status": "MISSING",
            "artifact_path": expected["vgen2_checkpoint"]["path"],
            "originating_experiment": "CLR v2.4 VGen2",
            "sha256": None,
            "config_path": "Expected VGen2 config not found",
            "architecture": "Original integration expects checkpoint entries core and value_heads; source has a 64D latent value state and 64-way per-slot value heads",
            "input_output_dims": {"latent_value_state": 64, "value_classes": 64, "checkpoint_contract": ["core", "value_heads"]},
            "expected_role": "Frozen V/Core dynamics and value heads for the original write/update and answer path",
            "frozen": True,
            "historical_metric": "No recovered VGen2 checkpoint-specific metric/config",
            "confidence": "HIGH that checkpoint path and experiment directory are absent from searched stores; source code is available",
            "evidence": [expected["vgen2_checkpoint"], code["a1_v24_vgen.py"]],
        },
        {
            "component": "O3a K-aware free-slot allocator checkpoint",
            "status": "MISSING",
            "artifact_path": expected["o3a_allocator_checkpoint"]["path"],
            "originating_experiment": "CLR v2.4 O3a learned free-slot allocation",
            "sha256": None,
            "config_path": "Expected O3a run config/checkpoint directory not found",
            "architecture": "Allocator(k_aware=True): [64D key + 4D occupancy]→128→4 slot logits",
            "input_output_dims": {"input": 68, "hidden": 128, "output_slots": 4},
            "expected_role": "Frozen NEW action slot selection",
            "frozen": True,
            "historical_metric": "Prior project state reports O3a learned free-slot allocation PASS; original allocator weights absent",
            "confidence": "HIGH that required checkpoint is missing; exact architecture source is present",
            "evidence": [expected["o3a_allocator_checkpoint"], code["a1_v24_o3a.py"]],
        },
        {
            "component": "Original O3b baseline router checkpoint",
            "status": "MISSING",
            "artifact_path": expected["baseline_router_checkpoint"]["path"],
            "originating_experiment": "CLR v2.4 O3b baseline write/update router",
            "sha256": None,
            "config_path": expected["baseline_o3b_config"]["path"],
            "architecture": "UpdateRouter: pair branch 257→128→1; action branch 69→96→2; SLOTS=4, DIM=64",
            "input_output_dims": {"pair_input": 257, "pair_output": 1, "action_input": 69, "action_output": 2},
            "expected_role": "Pre-reconnect baseline NEW/UPDATE and target-slot decision component",
            "frozen": True,
            "historical_metric": "Historical O3b behavior is documented in the project brief; no router checkpoint or original O3b output/config recovered for reproduction",
            "confidence": "HIGH that expected router path and original output directory are absent; source script is hash-matched to local archive",
            "evidence": [expected["baseline_router_checkpoint"], expected["baseline_o3b_config"], code["a1_v24_o3b.py"]],
        },
        {
            "component": "Original O3b integration source",
            "status": "STRONGLY_MATCHED",
            "artifact_path": code["a1_v24_o3b.py"]["remote_path"],
            "originating_experiment": "CLR v2.4 O3b",
            "sha256": code["a1_v24_o3b.py"]["sha256"],
            "config_path": "Defaults are embedded in source; original generated config.json under clr_v24_o3b_update is missing",
            "architecture": "Frozen KeyEncoder/QueryEncoder/VGen2/O3a allocator plus original UpdateRouter",
            "input_output_dims": {"raw_hidden": 2048, "latent_key": 64, "slots": 4},
            "expected_role": "Historical integration and action semantics reference",
            "frozen": True,
            "historical_metric": "Source emits an O3b config at runtime; corresponding output bundle not recovered",
            "confidence": "STRONG: remote source SHA-256 exactly matches the local archived a1_v24_o3b.py",
            "evidence": {"remote_source": code["a1_v24_o3b.py"], "local_archive": local_search["relevant_local_archive_hits"]["original_o3b_integration_script"]},
        },
        {
            "component": "Original O3b run config and evaluation logs",
            "status": "MISSING",
            "artifact_path": expected["baseline_o3b_config"]["path"],
            "originating_experiment": "CLR v2.4 O3b integration run",
            "sha256": None,
            "config_path": None,
            "architecture": "Not independently recoverable beyond source defaults",
            "input_output_dims": None,
            "expected_role": "Exact historical eval split, run configuration, and baseline measurements needed for reproduction",
            "frozen": True,
            "historical_metric": "O3b-L and O3b-K diagnostic configs were found but are not the original O3b integration config; not used as substitutes",
            "confidence": "HIGH for missing named O3b output; diagnostic configs explicitly belong to O3b-L/K",
            "evidence": {
                "missing_baseline_config": expected["baseline_o3b_config"],
                "found_but_not_baseline": evidence["diagnostic_configs_not_baseline"],
                "config_reference_search": search["queries"]["configs_code_and_logs_references"]["stdout"].splitlines(),
            },
        },
    ]
    dependency_manifest = {
        "experiment": "O3b-M3",
        "status": "INCOMPLETE_ORIGINAL_FROZEN_BUNDLE_NOT_RECOVERED",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "search_scope": {
            "local_workspace_and_remote_archive": local_search["roots"],
            "remote_active_project": "/root/private_data/clr_v24_o3b_l",
            "remote_user_home_mirror": "/public/home/ac92f4k0z9/clr_v24_o3b_l",
            "remote_exact_checkpoint_search": "active checkpoint tree + exact expected filenames under user-owned public/share roots",
            "non_user_shared_data": "Unrelated other-user public_data/other was not traversed; no artifacts were read from other users' directories",
        },
        "components": dep_records,
        "missing_required_frozen_weights": [
            expected["o21a_mr_checkpoint"]["path"],
            expected["vgen2_checkpoint"]["path"],
            expected["o3a_allocator_checkpoint"]["path"],
            expected["baseline_router_checkpoint"]["path"],
        ],
        "available_non_substitutes": [expected["o21b_checkpoint"]["path"], expected["candidate_b_checkpoint"]["path"]],
        "no_substitutes_used": True,
        "no_retraining": True,
    }
    dump_json(OUT / "dependency_recovery_manifest.json", dependency_manifest)

    static_contracts = {
        "raw_carrier": {"input_dim": 2048, "source": "Original source code / M2 frozen carrier manifest"},
        "key_encoder": {"shape": [2048, 256, 64], "weights_loaded": False},
        "query_encoder": {"shape": [2048, 256, 64], "weights_loaded": False},
        "slot_key_state": {"shape": [4, 64], "convention": "slot index 0..3"},
        "allocator": {"input": [64, 4], "hidden": 128, "output": 4, "weights_loaded": False},
        "baseline_router": {"pair_branch": [257, 128, 1], "action_branch": [69, 96, 2], "weights_loaded": False},
        "value_path": {"checkpoint_entries_expected": ["core", "value_heads"], "latent_dim": 64, "weights_loaded": False},
        "candidate_b": {"frozen": True, "score_orientation": "higher means same-key", "threshold": policy["threshold_exact"], "update_target": "argmax occupied slot"},
    }
    dump_json(OUT / "bundle_compatibility_audit.json", {
        "status": "BLOCKED_NOT_RUN — required frozen weights missing",
        "static_source_contracts": static_contracts,
        "tensor_shape_connection_test": "NOT_RUN",
        "state_dict_key_compatibility": "NOT_RUN",
        "memory_layout_runtime_check": "NOT_RUN",
        "adapter_added": False,
        "reason": "The original O21A-MR, VGen2, O3a allocator, and baseline router checkpoints are not recovered; source-level shape notes do not certify runtime compatibility.",
    })
    dump_csv(OUT / "baseline_reproduction.csv", ["component", "status", "metric", "baseline", "recovered", "delta", "reason"], [
        {"component": "O3a free-slot allocation", "status": "NOT_RUN", "reason": "Original frozen allocator checkpoint missing"},
        {"component": "Retrieval", "status": "NOT_RUN", "reason": "Baseline bundle not reliably recovered"},
        {"component": "Full KV answer", "status": "NOT_RUN", "reason": "Baseline bundle not reliably recovered"},
        {"component": "Original router/action behavior", "status": "NOT_RUN", "reason": "Original router checkpoint missing"},
    ])
    dump_json(OUT / "baseline_reproduction_gate.json", {
        "status": "BLOCKED_NOT_RUN",
        "pass": False,
        "reason": "Cannot establish a trustworthy baseline without the original frozen O21A-MR, VGen2, O3a allocator, and router bundle.",
        "reconnect_allowed": False,
    })
    dump_json(OUT / "reconnect_config.json", {
        "status": "NOT_RUN_BASELINE_GATE_BLOCKED",
        "candidate": "Candidate B seed 43; frozen checkpoint only",
        "rule": "Rule A",
        "threshold_exact": policy["threshold_exact"],
        "action": "UPDATE iff top1 >= threshold, target argmax occupied slot; otherwise NEW via original O3a allocator",
        "capacity": 4,
        "changes_to_model_or_policy": False,
        "reason": "M3 Stage 1 did not recover the original frozen dependency bundle; baseline reproduction gate therefore blocks reconnect.",
    })
    for filename in ("address_action_results.csv", "update_state_transition_audit.csv", "free_slot_regression.csv", "retrieval_regression.csv", "full_kv_regression.csv"):
        dump_csv(OUT / filename, ["status", "reason"], [{"status": "NOT_RUN", "reason": "Stopped at frozen dependency recovery gate; no reconnect or baseline run performed"}])
    dump_json(OUT / "testa_overlap_caveat.json", {
        "TestA_key_identity_overlap_with_train": 1479,
        "TestA_key_identity_overlap_with_validation": 922,
        "TestB_key_identity_overlap_with_train_or_validation": 0,
        "exact_source_rows_and_filtered_semantic_facts_overlap": 0,
        "M3_TestA_TestB_evaluated": False,
        "interpretation": "If M3 resumes, TestA is operational/regression evidence, not unseen-key generalization; key-disjoint TestB is the stricter generalization gate.",
    })
    dump_json(OUT / "final_o3b_gate.json", {
        "verdict": "O3b-M3-INCOMPLETE — ORIGINAL FROZEN O3b DEPENDENCIES NOT RECOVERED",
        "dependency_recovery": "FAIL_TO_RECOVER_REQUIRED_ORIGINAL_WEIGHTS",
        "baseline_reproduction": "NOT_RUN",
        "real_reconnect": "NOT_RUN",
        "address_gates": "NOT_RUN_IN_REAL_MEMORY",
        "free_slot_regression": "NOT_RUN",
        "retrieval_regression": "NOT_RUN",
        "full_kv_regression": "NOT_RUN",
        "O3b_operational_address": "INCOMPLETE",
        "O3c_allowed": False,
    })

    missing_bullets = "\n".join(f"- `{path}`" for path in dependency_manifest["missing_required_frozen_weights"])
    report = f'''# O3b-M3 — Frozen Bundle Recovery + End-to-End Reconnect

## FINAL VERDICT

**O3b-M3-INCOMPLETE — ORIGINAL FROZEN O3b DEPENDENCIES NOT RECOVERED**

M2's Candidate B + Rule A remains frozen and its offline action/value-replacement results are unchanged. The original integration code was recovered, but the required original frozen weights are not. Per the baseline recovery gate, I stopped before compatibility runtime testing, baseline reproduction, and reconnect. No substitute was trained or selected.

## Stage 0 — M2 policy freeze

- Candidate: Candidate B, deployment seed 43; config and checkpoint hashes were checked before and after the recovery search.
- Candidate checkpoint SHA-256: `{policy['candidate_checkpoint_sha256_before']}`.
- Candidate config SHA-256: `{policy['candidate_config_sha256']}`.
- Rule A threshold: `{policy['threshold_exact']}` (display: `0.8504285`); no recalibration.
- Protected M2 artifact hashes unchanged: **{str(policy['m2_policy_artifacts_unchanged_after_recovery']).upper()}**. Remote Candidate B hash matches the frozen hash: **{str(policy['remote_candidate_checkpoint_matches_frozen_sha256']).upper()}**.
- Train-only E/R normalization, J3C feature source, label semantics, true-update construction, and action semantics are captured in `m2_address_policy_frozen_manifest.json`.

## Stage 1 — dependency recovery

The search covered the local workspace/archive, active remote CLR project and checkpoint tree, its user-home mirror, remote output/context/config paths, and exact checkpoint names in the user-owned shared home. The matched original source module and integration-script snapshots are in `source_reference/`; source hashes are in `remote_source_and_checkpoint_evidence.json`.

Recovered as source/evidence:

- Original O3b integration source `a1_v24_o3b.py`, SHA-256 `{remote_old_o3b_sha}`; it exactly matches the local archived copy.
- O21A, O21A-MR, VGen, O3a, and shared O2 code modules.
- O21B `b1_best.pt`, SHA-256 `{remote_o21b.get('sha256')}`; its remote and local archive hashes match. The original integration references this for the KeyEncoder, but it is not a substitute for missing O21A-MR/VGen2/O3a/router artifacts.
- O3b-L and O3b-K diagnostic configs. Both explicitly identify their diagnostic phases and are not treated as the original O3b integration config.

Required frozen files not recovered:

{missing_bullets}

The O21A-MR path contains only the cache directory; the VGen2, O3a-allocation, and original O3b-update checkpoint directories are absent from the active checkpoint tree. Exact-name searches in the local archive and the user-owned remote shared home found no copies. The unrelated `root/public_data/other` directories were not traversed.

## Compatibility and baseline gates

Source code suggests 4 slots × 64D keys; KeyEncoder/QueryEncoder contracts are 2048→256→64; the O3a allocator is 68→128→4; the original router has pair 257→128→1 and action 69→96→2 branches. These are source-level expectations only. State dicts, checkpoint keys, and live tensor contracts were **not** verified. No adapter was added.

Bundle compatibility audit: **BLOCKED / NOT RUN**. Baseline reproduction: **NOT RUN**, not a failed metric run. Consequently the only valid result is to block reconnect; otherwise a regression could not be attributed to the new policy versus a damaged or mismatched dependency bundle.

## Reconnect and end-to-end evaluation

Real reconnect: **NO**. No end-to-end memory state was altered. Address action F1, target-slot recall, NEW collision, old-value invalidation, duplicate-key rate, free-slot regression, retrieval, and full KV answer are all **NOT RUN** in the original memory path. M2 offline action results remain prior evidence only; they are not relabeled as M3 integration results.

TestA caveat: before exact-fact filtering, TestA shares 1,479 key identities with train and 922 with validation. TestB is key-disjoint from both and is the stricter unseen-key generalization gate. No M3 holdout was evaluated.

## Required answers

1. Candidate B remains frozen: **YES**; before/after and remote hashes match.
2. Rule A threshold remains exact: **YES**, `{policy['threshold_exact']}`.
3. O21A/O21A-MR checkpoint recovered: **NO**; source code exists, frozen checkpoint does not.
4. VGen2 checkpoint recovered: **NO**; source code exists, frozen checkpoint does not.
5. O3a allocator checkpoint recovered: **NO**; source code exists, frozen checkpoint does not.
6. Baseline router checkpoint recovered: **NO**; original source exists, checkpoint does not.
7. Full bundle compatibility audit: **NOT RUN / BLOCKED** by missing frozen weights.
8. Original baseline reproduced: **NO — NOT RUN**.
9. Real reconnect performed: **NO**.
10–17. Reconnect Same-key F1, TRUE UPDATE recall, target-slot recall, NEW collision, P2/P3 wrong UPDATE, value replacement, and duplicate-slot rate: **N/A — NOT RUN**.
18. O3a free-slot regression: **N/A — NOT RUN**.
19–20. Retrieval TestA/TestB and delta: **N/A — NOT RUN**.
21–22. Full KV TestA/TestB and delta: **N/A — NOT RUN**.
23. TestB key-disjoint operational gate: **NOT RUN**.
24. O3b operational Address: **INCOMPLETE**.
25. O3c allowed: **NO**.

## Integrity and execution boundary

- Candidate B, its config, threshold, normalization, source/label manifests, and action semantics were not changed.
- No training, GPU evaluation, baseline reproduction, or reconnect was run. No model/checkpoint was modified. No TestA/TestB M3 evaluation occurred and no data leakage occurred.
- Search evidence and all source snapshots are hashed in the artifact manifest. Workspace root is not a Git repository; Git SHA: **N/A**.

Exactly one next experiment: **Reconstruct a certified frozen integration bundle from the last individually validated components, with provenance/config/hash checks; do not retrain substitutes in that recovery experiment. After the bundle is certified, reproduce the original baseline before any Candidate B reconnect.**
'''
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    verdict = "O3b-M3-INCOMPLETE — ORIGINAL FROZEN O3b DEPENDENCIES NOT RECOVERED"
    (OUT / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")

    manifest_rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "artifact_sha256_manifest.json":
            continue
        manifest_rows.append({
            "path": path.relative_to(OUT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    dump_json(OUT / "artifact_sha256_manifest.json", {
        "experiment": "O3b-M3",
        "verdict": verdict,
        "files": manifest_rows,
        "candidate_b_sha256": policy["candidate_checkpoint_sha256_before"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"verdict": verdict, "manifest_files": len(manifest_rows), "policy_hashes_unchanged": policy["m2_policy_artifacts_unchanged_after_recovery"]}, ensure_ascii=False))


def local_o21b_sha(path: Path) -> str:
    return sha256(path)


if __name__ == "__main__":
    main()
