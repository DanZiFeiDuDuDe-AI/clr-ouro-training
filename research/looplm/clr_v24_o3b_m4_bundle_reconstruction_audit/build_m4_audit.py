from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"H:\LoopLM")
OUT = ROOT / "clr_v24_o3b_m4_bundle_reconstruction_audit"
M2 = ROOT / "clr_v24_o3b_m2_growing_slot_calibration"
M3 = ROOT / "clr_v24_o3b_m3_end_to_end_reconnect"
M3SRC = M3 / "source_reference"
M4SRC = OUT / "source_reference"
J3C = ROOT / "remote_archive" / "new_server" / "clr_v24_o3b_j3c_fresh_certified_feature_cache"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path, role: str) -> dict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha(path) if path.is_file() else None,
        "role": role,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    M4SRC.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    m2_candidate = json.loads((M2 / "candidate_b_frozen_manifest.json").read_text(encoding="utf-8"))
    m3_deps = json.loads((M3 / "dependency_recovery_manifest.json").read_text(encoding="utf-8"))
    m3_remote = json.loads((M3 / "remote_dependency_search.json").read_text(encoding="utf-8"))
    m3_local = json.loads((M3 / "local_dependency_search.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((J3C / "source_snapshot" / "source_model_artifact_manifest.json").read_text(encoding="utf-8"))

    source_specs = {
        "a1_v24_o3b.py": (M3SRC / "a1_v24_o3b.py", "687e0c3024cbb04100c4fb06bdbec5af471a7fdb9c4cc9ccf3852aab6ca3ddfd"),
        "a1_v24_o21a.py": (M3SRC / "a1_v24_o21a.py", "f91ff8f05c6f9427bf724f4040b3616f3da409bc600358d4a80f0f895955d0fb"),
        "a1_v24_o21a_mr.py": (M3SRC / "a1_v24_o21a_mr.py", "fe18d2425357c1152d98c216e8e793a4f6f1db611ab2bd12082b7863776169df"),
        "a1_v24_vgen.py": (M3SRC / "a1_v24_vgen.py", "9702072f84695750b4977c47e7c0aaeccb6ad66ac620548247e95ba9b77d2d13"),
        "a1_v24_o3a.py": (M3SRC / "a1_v24_o3a.py", "f7142e82aa35233175ca6b14aaeb383ffe5bdee37c48371b9af314cbfa5f7b92"),
        "a1_v24_o2_common.py": (M3SRC / "a1_v24_o2_common.py", "947ce501081fa63e338b3c99c76705792050df0c2cce9e76725807079da5446d"),
        "a1_v23_r2_common.py": (M4SRC / "a1_v23_r2_common.py", "2d9affa2dd0638c800a3e0d0e8a0f286e8ea18972390de889882b323dabd811f"),
        "clr_v23_oracle_r2.py": (M4SRC / "clr_v23_oracle_r2.py", "0f3bb6477ce716b3f40f802492057ee34fc565539ef46af28835e929e60a7dc0"),
    }
    source_hashes = {}
    for name, (path, expected) in source_specs.items():
        actual = sha(path) if path.is_file() else None
        source_hashes[name] = {
            "path": str(path),
            "sha256": actual,
            "expected_or_remote_sha256": expected,
            "matches_expected": actual == expected,
            "bytes": path.stat().st_size if path.is_file() else None,
            "provenance": "M3 remote snapshot" if path.parent == M3SRC else "M4 read-only SFTP copy; remote sha256sum matched",
        }

    rule_path = M2 / "decision_rule_frozen.json"
    m2_report = M2 / "final_report.md"
    m3_dep_path = M3 / "dependency_recovery_manifest.json"
    m3_policy_path = M3 / "m2_address_policy_frozen_manifest.json"
    m3_source_evidence = M3 / "remote_source_and_checkpoint_evidence.json"
    m2_rule = json.loads(rule_path.read_text(encoding="utf-8"))
    rule_threshold = m2_rule["all_rule_validation_summary"]["A"]["params"]["threshold"]

    expected_remote = [
        ("O21A-MR", "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/best_model.pt", "MISSING"),
        ("VGen2", "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_vgen2/best.pt", "MISSING"),
        ("O3a K-aware allocator", "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3a_allocation/allocator_kaware_best.pt", "MISSING"),
        ("O3b baseline router", "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_update/router_best.pt", "MISSING"),
        ("O2.3 R2 core source checkpoint", "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v23_oracle_r2/best_model.pt", "MISSING; upstream VGen2 training/loader dependency"),
        ("O21B KeyEncoder", "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt", "EXACTLY RECOVERABLE"),
        ("Candidate B M2 address checkpoint", m2_candidate["checkpoint_source"], "EXACTLY RECOVERABLE; not a baseline substitute"),
        ("O3b baseline config", "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_update/config.json", "MISSING"),
        ("O3b-L diagnostic config", "/root/private_data/clr_v24_o3b_l/output/config.json", "PRESENT; diagnosis-only, not baseline config"),
        ("O3b-K diagnostic config", "/root/private_data/clr_v24_o3b_l/context/o3b_k/config.json", "PRESENT; diagnosis-only, not baseline config"),
    ]
    remote_model_files = [
        ("O21B", 4356237, "6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570"),
        ("Candidate B seeds 17/29/43/61/79", 5 * 1166389, "only seed 43 frozen SHA individually certified by M2/M3"),
        ("Candidate A/C seeds", 10 * 3029, "M2 candidate files; not baseline component state"),
    ]

    starting = {
        "experiment": "O3b-M4 CERTIFIED BASELINE INTEGRATION BUNDLE RECONSTRUCTION AUDIT",
        "created_utc": now,
        "workspace": str(ROOT),
        "freeze_status": "M2 Candidate B and Rule A untouched; M4 performs read-only forensic analysis except copying original source files into this new audit folder",
        "frozen_m2_artifacts": {
            "candidate": m2_candidate["candidate"],
            "candidate_checkpoint_sha256": m2_candidate["checkpoint_sha256_after"],
            "candidate_checkpoint_remote_path": m2_candidate["checkpoint_source"],
            "candidate_config_sha256": m2_candidate["candidate_config_sha256"],
            "candidate_state_dict_sha256": m2_candidate["state_dict_sha256_after"],
            "rule": "A",
            "rule_a_threshold": rule_threshold,
            "rule_a_frozen_config_path": str(rule_path),
            "rule_a_frozen_config_sha256": sha(rule_path),
            "m2_final_report_path": str(m2_report),
            "m2_final_report_sha256": sha(m2_report),
            "m2_frozen_manifest_sha256": sha(M2 / "candidate_b_frozen_manifest.json"),
        },
        "m3_artifacts": {
            "dependency_recovery_manifest_path": str(m3_dep_path),
            "dependency_recovery_manifest_sha256": sha(m3_dep_path),
            "policy_manifest_path": str(m3_policy_path),
            "policy_manifest_sha256": sha(m3_policy_path),
            "remote_source_and_checkpoint_evidence_path": str(m3_source_evidence),
            "remote_dependency_search_sha256": sha(M3 / "remote_dependency_search.json"),
            "local_dependency_search_sha256": sha(M3 / "local_dependency_search.json"),
        },
        "source_hashes": source_hashes,
        "source_identity_notes": {
            "O3b_router_source": "Exact remote/local archived source match; SHA verified.",
            "VGen2_source": "Only a1_v24_vgen.py was found; its default output is clr_v24_vgen, whereas O3b expects clr_v24_vgen2/best.pt. No vgen2-named source/config/log was found, so source identity for that exact run is not fully certified.",
            "R2_class_source": "M4 copied clr_v23_oracle_r2.py and a1_v23_r2_common.py; both local hashes match remote sha256sum.",
        },
        "discovered_remote_paths": {
            "checkpoint_config": [{"component": n, "path": p, "status": s} for n, p, s in expected_remote],
            "user_owned_remote_model_candidates": remote_model_files,
            "o21a_mr_cache_files": [
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/manifest.json",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/relations.json",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/train_data.npz",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/val_data.npz",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/testA_data.npz",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/testB_data.npz",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/train_meta.jsonl",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/val_meta.jsonl",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/testA_meta.jsonl",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/testB_meta.jsonl",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/train_persons.json",
            ],
            "diagnostic_configs_not_baseline": [
                "/root/private_data/clr_v24_o3b_l/output/config.json",
                "/root/private_data/clr_v24_o3b_l/context/o3b_k/config.json",
            ],
            "diagnostic_and_m2_logs": [
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m2_growing_slot_calibration/logs/calibrate_v4.log",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m2_growing_slot_calibration/logs/holdout_v4.log",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m2_growing_slot_calibration/logs/finalize_v6.log",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m_operational_samekey_rescue/logs/runner_v3.stdout.log",
                "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m_operational_samekey_rescue/logs/runner_v3.stderr.log",
                "/root/private_data/clr_v24_o3b_l/output/run.log",
            ],
            "remote_tmp_files": ["/tmp/tmp1i7o9mxn/_remote_module_non_scriptable.py", "/tmp/tmp1i7o9mxn/__pycache__/_remote_module_non_scriptable.cpython-311.pyc"],
            "additional_source_files_fetched_read_only": [
                str(M4SRC / "a1_v23_r2_common.py"),
                str(M4SRC / "clr_v23_oracle_r2.py"),
            ],
        },
        "remote_search_scope": {
            "active_project": "/root/private_data/clr_v24_o3b_l",
            "user_owned_home_mirror": "/public/home/ac92f4k0z9",
            "user_owned_shared_storage": "/public/share/ac92f4k0z9",
            "tmp_scope": "/tmp/tmp1i7o9mxn plus CLR/O3b-named top-level temporary files",
            "other_users_data": "Not inspected",
            "remote_current_hash_recheck": {
                "candidate_b": m2_candidate["checkpoint_sha256_after"],
                "o21b": "6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570",
            },
        },
    }
    dump_json(OUT / "m4_starting_artifact_manifest.json", starting)

    contract = {
        "experiment": "O3b-M4",
        "source_of_truth": "hash-matched a1_v24_o3b.py plus dependency sources; contract describes source, not recovered weights or measured behavior",
        "constants": {"raw_hidden_dim": 2048, "slots": 4, "key_dim": 64, "value_dim": 64, "value_classes": 64, "relation_families": ["current_city", "birth_city", "favorite_city", "work_city"]},
        "execution_order": [
            "Frozen extractor/cache supplies fact_H (2048D) and fact_query_H/query hidden input (2048D).",
            "O21B KeyEncoder maps fact_H to normalized K in R^64; O21A-MR QueryEncoder maps query hidden to normalized Q in R^64.",
            "Initialize K memory zeros [B,4,64], V memory via OracleLocalCore.zero_state zeros [B,4,64], and occupancy zeros [B,4].",
            "For initial facts, frozen O3a K-aware Allocator receives [K_new(64), occupancy(4)] and emits 4 slot logits; occupied slots are masked for safety.",
            "For an incoming fact, frozen key encoder forms K_new. UpdateRouter emits action logits [NEW, UPDATE] and four pair logits. UPDATE if softmax(action)[UPDATE] >= the threshold saved in router_best.pt.",
            "UPDATE selects argmax pair logit over occupied slots, retains the existing K, and writes the new value representation into that V slot. NEW uses the frozen O3a allocator, writes K and V, and marks occupancy.",
            "Read uses normalized Q-to-K dot product divided by temperature 0.1, argmax slot selection, then the selected 64D V state is decoded by the corresponding frozen 64->64 local value head.",
        ],
        "components": [
            {"name": "Ouro frozen feature extractor", "role": "produces 2048D hidden vectors used by the caches; O3b integration consumes cache vectors rather than loading Ouro in load_all", "input": "tokenized fact/query", "output": "fact_H/query_H: 2048D", "state_dict_keys": "Pinned ByteDance/Ouro-1.4B model.safetensors; not part of O3b router state", "learned_parameters": 1400000000, "status": "EXACTLY RECOVERABLE as pinned upstream model artifact; M4 did not rehash the 2.87GB file"},
            {"name": "O21B KeyEncoder", "role": "encode stored/incoming fact keys", "input": "H: [B,2048]", "output": "L2-normalized K: [B,64]", "state_dict_keys": ["key.net.0.weight (256,2048)", "key.net.0.bias (256)", "key.net.2.weight (64,256)", "key.net.2.bias (64)"], "parameters": 540992, "normalization": "F.normalize(..., dim=-1)", "load_contract": "torch.load(O21B)['key']; strict=True", "checkpoint": "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt", "status": "EXACTLY RECOVERABLE"},
            {"name": "O21A-MR QueryEncoder", "role": "encode read-side query", "input": "query hidden: [B,2048]", "output": "L2-normalized Q: [B,64]", "state_dict_keys": ["query.net.0.weight (256,2048)", "query.net.0.bias (256)", "query.net.2.weight (64,256)", "query.net.2.bias (64)"], "parameters": 540992, "load_contract": "torch.load(O21A_MR)['query']; strict=True", "status": "LEARNED COMPONENT — CHECKPOINT MISSING"},
            {"name": "R2 OracleLocalCore / VGen2 value path", "role": "content read plus recurrent per-slot V update; VGen2 supplies final trained core state and value heads", "input": "fact_H: [B,2048], prior selected V slot: [B,64]", "output": "V memory [B,4,64]", "core_state_dict_keys": ["core.content_read.weight (64,2048)", "core.content_read.bias (64)", "core.shared_gru.weight_ih (192,64)", "core.shared_gru.weight_hh (192,64)", "core.shared_gru.bias_ih (192)", "core.shared_gru.bias_hh (192)"], "core_parameters": 156096, "vgen2_load_contract": "torch.load(VGEN2)['core']; strict=True", "initializer_checkpoint": "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v23_oracle_r2/best_model.pt", "initializer_status": "NOT FOUND; final O3b core state would be supplied by VGen2 if that full checkpoint existed; original source loader nevertheless calls load_r2_checkpoint first", "status": "VGen2 learned core is missing; R2 initializer is an additional upstream training/loader dependency"},
            {"name": "VGen2 MRValueHeads", "role": "per-physical-slot value decoder", "input": "selected V slot: [B,64]", "output": "64 class logits", "state_dict_keys": ["value_heads.heads.{0..3}.weight (64,64)", "value_heads.heads.{0..3}.bias (64)"], "parameters": 16640, "load_contract": "torch.load(VGEN2)['value_heads']; strict=True", "status": "LEARNED COMPONENT — CHECKPOINT MISSING"},
            {"name": "O3a K-aware FreeAllocator", "role": "choose a legal free slot for NEW writes", "input": "K_new [B,64] concatenated with occupancy [B,4]", "output": "4 slot logits [B,4]", "state_dict_keys": ["allocator.net.0.weight (128,68)", "allocator.net.0.bias (128)", "allocator.net.2.weight (4,128)", "allocator.net.2.bias (4)"], "parameters": 9348, "load_contract": "torch.load(O3A)['allocator']; strict=True", "status": "LEARNED COMPONENT — CHECKPOINT MISSING"},
            {"name": "O3b baseline UpdateRouter", "role": "NEW/UPDATE decision and UPDATE target slot", "input": {"K_new": "[B,64]", "existing_K": "[B,4,64]", "occupancy": "[B,4]"}, "pair_branch": "[K_new, K_slot, K_new*K_slot, abs(K_new-K_slot), occupancy_bit] = 257D -> 128 -> 1, per slot", "action_branch": "[K_new, max occupied pair logit, occupancy(4)] = 69D -> 96 -> 2", "output": {"action_logits": "[B,2], class 0=NEW, 1=UPDATE", "pair_logits": "[B,4]"}, "state_dict_keys": ["router.pair.0.{weight,bias}", "router.pair.2.{weight,bias}", "router.action.0.{weight,bias}", "router.action.2.{weight,bias}"], "parameters": 40067, "checkpoint_payload": ["router", "threshold", "step", "score", "params"], "status": "LEARNED COMPONENT — CHECKPOINT MISSING"},
            {"name": "Retrieval matcher", "role": "select slot from Q and K", "formula": "(Q[:,None,:] * K).sum(-1) / 0.1; argmax", "learned_parameters": 0, "deterministic": True, "state_dependency": "requires recovered O21B KeyEncoder and O21A-MR QueryEncoder"},
            {"name": "Memory initializer and occupancy", "role": "initial K/V and slot metadata", "formula": "K=zeros([B,4,64]); V=OracleLocalCore.zero_state=zeros([B,4,64]); occupancy=zeros([B,4])", "learned_parameters": 0, "buffers": "none; the V initializer is deterministic zero state, distinct from the learned write-cell weights", "status": "DETERMINISTICALLY RECONSTRUCTABLE"},
            {"name": "Value decoder", "role": "turn selected V into a 64-class output", "formula": "selected MRValueHeads Linear(64,64) for physical slot", "learned_parameters": 16640, "state_dependency": "VGen2 value_heads; missing"},
        ],
        "router_action_semantics": {"0": "NEW", "1": "UPDATE", "decision": "UPDATE when softmax(action_logits)[1] >= checkpoint threshold", "update_target": "argmax pair logit after masking empty slots", "new_target": "argmax O3a allocator logits after masking occupied slots", "update_key_rule": "retain old K; overwrite/update only selected V slot", "new_key_rule": "write incoming K and V to selected free slot"},
        "memory_state_layout": {"K": "float [4,64]", "V": "float [4,64]", "occupancy": "binary/float [4]", "slot_ids": "physical indices 0..3", "values": "64-class label space"},
        "data_and_mapping": {"cache": "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache", "labels": "cache metadata; 4 relation names and 64 value classes", "cache_found": True, "cache_is_model_state": False},
        "source_limits": ["O3b-L/K diagnostic configs are not original O3b config.", "O3b source defaults are visible but original environment overrides, exact router threshold, selected steps, and original run logs are missing.", "Only a1_v24_vgen.py was found; its default output folder is clr_v24_vgen, not the O3b VGEN2 path. The VGen2 run identity/config is not fully certified."],
    }
    dump_json(OUT / "historical_bundle_contract.json", contract)

    contract_md = """# O3b-M4 — Historical bundle contract (source-derived)\n\nThis is a static contract recovered from hash-matched `a1_v24_o3b.py` and its source dependencies. It does not certify missing learned weights or reproduce historical behavior.\n\n## Bundle and execution order\n\n1. Frozen feature cache supplies fact/query hidden vectors at 2048D. The pinned Ouro-1.4B source model is separately archived; the O3b `load_all` path itself consumes cache vectors.\n2. O21B `KeyEncoder` maps fact H to L2-normalized 64D keys. O21A-MR `QueryEncoder` maps query H to L2-normalized 64D queries.\n3. Initialize K `[B,4,64]`, V `[B,4,64]`, and occupancy `[B,4]` to zero. Initial NEW writes use the frozen learned O3a allocator.\n4. The incoming fact goes through the key encoder and the baseline `UpdateRouter`. Action class `0=NEW`, `1=UPDATE`; UPDATE is selected when its softmax probability meets the threshold stored in the router checkpoint.\n5. UPDATE chooses the highest occupied pair logit, preserves the existing K, and updates that slot's V through `content_read` + `shared_gru`. NEW chooses an unoccupied slot through O3a and writes both K and V.\n6. Query read uses `dot(Q,K)/0.1`, argmax slot selection, then the selected physical-slot MRValueHeads decoder outputs 64 class logits.\n\n## Learned state contract\n\n| Component | Interface | State loaded from | Parameters | Current state |\n|---|---|---|---:|---|\n| O21B KeyEncoder | 2048 → 256 → GELU → 64 → L2 normalize | O21B `key` | 540,992 | exact checkpoint present |\n| O21A-MR QueryEncoder | 2048 → 256 → GELU → 64 → L2 normalize | O21A-MR `query` | 540,992 | checkpoint missing |\n| VGen2 OracleLocalCore | 2048D H + selected prior 64D V → `Linear(2048,64)` + `GRUCell(64,64)` | VGen2 `core` | 156,096 | checkpoint missing |\n| VGen2 MRValueHeads | four independent `Linear(64,64)` heads | VGen2 `value_heads` | 16,640 | checkpoint missing |\n| O3a K-aware allocator | concat(K64, occupancy4) → 128 → GELU → 4 | O3a `allocator` | 9,348 | checkpoint missing |\n| O3b UpdateRouter | pair 257→128→1; action 69→96→2 | router `router` + threshold | 40,067 | checkpoint missing |\n\nLinear/GELU/GRU modules have no running-stat buffers. L2 normalization, dot-product matching, zero memory initialization, occupancy masking, action/slot branching, and write control flow are deterministic. The learned write-cell parameters must not be confused with deterministic zero V initialization.\n\n## File and run-identity caveats\n\n- O3b source strictly loads `SUPERVISED['query']`, `VGEN2['core']`, `VGEN2['value_heads']`, `O3A['allocator']`, and `router_best['router']` plus its threshold.\n- The available O21B checkpoint contains a same-shaped `query` field, but it is O21B provenance, not the frozen O21A-MR query; it is not a certified substitute.\n- O3b source loads R2 once before replacing the core with VGen2's full `core`. The R2 checkpoint is absent; this is a loader/upstream training dependency. The exact VGen2 final core state would subsume it at serving time if that state were recovered.\n- Only `a1_v24_vgen.py` source was found. It defaults to `clr_v24_vgen`; O3b expects `clr_v24_vgen2/best.pt`. No VGen2-specific source/config/log/checkpoint was found.\n- O3b-L/K configs are diagnostic-only. The original O3b config, router threshold, best step, and baseline logs were not recovered.\n"""
    (OUT / "historical_bundle_contract.md").write_text(contract_md, encoding="utf-8")

    matrix = [
        {"component_name": "Ouro-1.4B frozen hidden extractor", "learned": "YES (pretrained, frozen)", "parameters": "~1.4B", "checkpoint_required": "for regenerating hidden cache; not loaded by O3b load_all", "exact_checkpoint_found": "YES; pinned model.safetensors in J3C snapshot", "partial_state_found": "NO", "deterministic_reconstruction_possible": "NO; exact model file is present", "evidence": "J3C source_model_artifact_manifest.json; pinned SHA 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af; transfer_verified=true", "status_classification": "EXACTLY RECOVERABLE"},
        {"component_name": "O21B KeyEncoder", "learned": "YES", "parameters": "540992", "checkpoint_required": "YES", "exact_checkpoint_found": "YES", "partial_state_found": "FULL", "deterministic_reconstruction_possible": "NO; learned state loaded exactly", "evidence": "Remote/local b1_best.pt SHA 6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570; strict load from ['key']", "status_classification": "EXACTLY RECOVERABLE"},
        {"component_name": "O21A-MR QueryEncoder (O3b read query path)", "learned": "YES", "parameters": "540992", "checkpoint_required": "YES", "exact_checkpoint_found": "NO", "partial_state_found": "NO target-provenance state; O21B has a non-substitute same-shape query", "deterministic_reconstruction_possible": "NO", "evidence": "Expected best_model.pt absent; only O21A-MR cache directory exists; O3b strict-loads ['query']", "status_classification": "LEARNED COMPONENT — CHECKPOINT MISSING"},
        {"component_name": "VGen2 OracleLocalCore (content_read + shared_gru)", "learned": "YES", "parameters": "156096", "checkpoint_required": "YES for final V-state behavior", "exact_checkpoint_found": "NO", "partial_state_found": "NO", "deterministic_reconstruction_possible": "NO", "evidence": "VGen2 best.pt and directory absent; source strict-loads ['core']; no matching state/signature artifact", "status_classification": "LEARNED COMPONENT — CHECKPOINT MISSING"},
        {"component_name": "VGen2 MRValueHeads", "learned": "YES", "parameters": "16640", "checkpoint_required": "YES", "exact_checkpoint_found": "NO", "partial_state_found": "NO certified VGen2 state", "deterministic_reconstruction_possible": "NO", "evidence": "VGen2 best.pt absent; O21A-MR/R2 local heads are different provenance and not substitutes", "status_classification": "LEARNED COMPONENT — CHECKPOINT MISSING"},
        {"component_name": "O3a K-aware free-slot allocator", "learned": "YES", "parameters": "9348", "checkpoint_required": "YES", "exact_checkpoint_found": "NO", "partial_state_found": "NO", "deterministic_reconstruction_possible": "NO", "evidence": "Source is MLP 68→128→4; strict-loads ['allocator']; required allocator_kaware_best.pt absent", "status_classification": "LEARNED COMPONENT — CHECKPOINT MISSING"},
        {"component_name": "O3b baseline UpdateRouter", "learned": "YES", "parameters": "40067", "checkpoint_required": "YES", "exact_checkpoint_found": "NO", "partial_state_found": "NO", "deterministic_reconstruction_possible": "NO", "evidence": "Original source is hash-matched; expected router_best.pt, stored threshold and original config absent", "status_classification": "LEARNED COMPONENT — CHECKPOINT MISSING"},
        {"component_name": "R2 OracleLocalCore initializer checkpoint", "learned": "YES", "parameters": "core 156096 + four local heads 16640", "checkpoint_required": "Original loader calls it; final core is overwritten by VGen2 state; also upstream VGen2/O21A-MR training dependency", "exact_checkpoint_found": "NO", "partial_state_found": "Source architecture only", "deterministic_reconstruction_possible": "NO", "evidence": "best_model.pt absent from active project/user home/share artifact-extension inventory; source serializes ['model','heads']", "status_classification": "LEARNED COMPONENT — CHECKPOINT MISSING"},
        {"component_name": "Query-key retrieval matcher", "learned": "NO (formula); depends on learned encoders", "parameters": "0", "checkpoint_required": "NO separate matcher checkpoint", "exact_checkpoint_found": "N/A", "partial_state_found": "N/A", "deterministic_reconstruction_possible": "YES", "evidence": "Source formula dot(Q,K)/0.1 + argmax", "status_classification": "DETERMINISTICALLY RECONSTRUCTABLE"},
        {"component_name": "L2 normalization", "learned": "NO", "parameters": "0", "checkpoint_required": "NO", "exact_checkpoint_found": "N/A", "partial_state_found": "N/A", "deterministic_reconstruction_possible": "YES", "evidence": "KeyEncoder/QueryEncoder F.normalize in exact source", "status_classification": "DETERMINISTICALLY RECONSTRUCTABLE"},
        {"component_name": "Memory zero-state and occupancy initializer", "learned": "NO", "parameters": "0", "checkpoint_required": "NO", "exact_checkpoint_found": "N/A", "partial_state_found": "N/A", "deterministic_reconstruction_possible": "YES", "evidence": "K zeros [4,64]; Core.zero_state zeros [4,64]; occupancy zeros [4]", "status_classification": "DETERMINISTICALLY RECONSTRUCTABLE"},
        {"component_name": "Slot/action/index semantics", "learned": "NO", "parameters": "0", "checkpoint_required": "NO", "exact_checkpoint_found": "Source + MR cache metadata present", "partial_state_found": "N/A", "deterministic_reconstruction_possible": "YES", "evidence": "4 physical slots; action 0 NEW / 1 UPDATE; cache relations and 64-class value index metadata", "status_classification": "DETERMINISTICALLY RECONSTRUCTABLE"},
        {"component_name": "Original O3b run config and router threshold", "learned": "NO (metadata)", "parameters": "0", "checkpoint_required": "Needed for exact baseline reproduction", "exact_checkpoint_found": "NO", "partial_state_found": "Source defaults only; O3b-K/L configs are diagnostic", "deterministic_reconstruction_possible": "NO exact run state", "evidence": "Expected O3b update config and router checkpoint absent", "status_classification": "AMBIGUOUS / INSUFFICIENT EVIDENCE"},
        {"component_name": "M2 Candidate B + Rule A (protected, non-baseline)", "learned": "YES (Candidate B); Rule A deterministic threshold policy", "parameters": "frozen Candidate B checkpoint", "checkpoint_required": "For M2 address policy only", "exact_checkpoint_found": "YES; SHA verified", "partial_state_found": "FULL for its own task", "deterministic_reconstruction_possible": "N/A", "evidence": f"Candidate B SHA {m2_candidate['checkpoint_sha256_after']}; Rule A threshold {rule_threshold}; not a baseline component or substitute", "status_classification": "EXACTLY RECOVERABLE (SEPARATE FROZEN ARTIFACT)"},
    ]
    write_csv(OUT / "component_recoverability_matrix.csv", list(matrix[0]), matrix)

    recovery_rows = [
        {"target_component": "O21A-MR query", "candidate_artifact": "O21B b1_best.pt top-level query.net.*", "candidate_state_keys_or_signature": "net.0.weight (256,2048), net.0.bias (256), net.2.weight (64,256), net.2.bias (64)", "target_match": "SHAPE ONLY; source checkpoint is O21B and O3b expressly requests O21A-MR query", "full_key_shape_value_coverage": "No target values to compare; no O21A-MR checkpoint", "recoverable": "NO", "decision": "Do not substitute; not certified target state"},
        {"target_component": "VGen2 core", "candidate_artifact": "R2 source code / O21B / M2 Candidate B / Ouro model snapshot", "candidate_state_keys_or_signature": "architecture or unrelated learned states; no VGen2['core'] tensor state", "target_match": "NO", "full_key_shape_value_coverage": "0% certified VGen2 values", "recoverable": "NO", "decision": "No nested core/EMA/optimizer/script/export state found"},
        {"target_component": "VGen2 value_heads", "candidate_artifact": "O21A-MR cache; R2 source; O21B", "candidate_state_keys_or_signature": "MR cache arrays contain examples/labels; R2 local heads are not VGen2 multi-relation decoder state", "target_match": "NO", "full_key_shape_value_coverage": "0% certified VGen2 value-head values", "recoverable": "NO", "decision": "Data cache cannot mathematically determine learned head weights"},
        {"target_component": "O3a allocator", "candidate_artifact": "O3a source/config references", "candidate_state_keys_or_signature": "source defines MLP but no weights; no allocator state fragment", "target_match": "NO", "full_key_shape_value_coverage": "0%", "recoverable": "NO", "decision": "Need learned state; occupancy rule does not determine k-aware allocator weights"},
        {"target_component": "O3b baseline router", "candidate_artifact": "M2 Candidate B checkpoint", "candidate_state_keys_or_signature": "M2 pair head uses factorized entity/relation 256→64→1; baseline router requires pair 257→128→1 AND action 69→96→2", "target_match": "NO; different architecture, features, objective and provenance", "full_key_shape_value_coverage": "0% baseline router values", "recoverable": "NO", "decision": "Not a diagnostic-router substitute"},
        {"target_component": "R2 initializer state", "candidate_artifact": "Ouro-1.4B safetensors / source-only OracleLocalCore", "candidate_state_keys_or_signature": "Ouro weights are a different model; source defines core shape but no trained values", "target_match": "NO", "full_key_shape_value_coverage": "0% R2 state values", "recoverable": "NO", "decision": "R2 best_model.pt and serialized model/heads absent"},
        {"target_component": "Optimizer/EMA/trainer snapshots", "candidate_artifact": "remote/local checkpoint extension scan (.pt/.pth/.ckpt/.bin/.pkl/.pickle/.onnx) and logs", "candidate_state_keys_or_signature": "Only O21B and M2 Candidate A/B/C model checkpoints in CLR-owned stores; no target-named optimizer/EMA bundles", "target_match": "NO", "full_key_shape_value_coverage": "None", "recoverable": "NO", "decision": "No surviving shadow copy identified"},
        {"target_component": "Serialized exports / cached exact tensors", "candidate_artifact": "MR cache NPZ/JSONL, J3C NPY/NPZ features, pinned base-model safetensors", "candidate_state_keys_or_signature": "Input features and labels, not learned O21A-MR/VGen2/O3a/router state", "target_match": "NO", "full_key_shape_value_coverage": "None", "recoverable": "NO", "decision": "No ONNX/ScriptModule/export sufficient to reconstruct target modules found"},
    ]
    write_csv(OUT / "partial_state_recovery_audit.csv", list(recovery_rows[0]), recovery_rows)

    source_rows = [
        {"component": "O3b integration/control flow", "source_or_config": source_hashes["a1_v24_o3b.py"]["path"], "source_complete": "YES", "learned_state_required": "YES for router and all strict-loaded encoders/allocator/core", "state_analytically_derivable": "NO", "source_only_result": "Interfaces, NEW/UPDATE semantics, checkpoint keys, defaults are recoverable; original weights/config/threshold are not"},
        {"component": "KeyEncoder / QueryEncoder", "source_or_config": source_hashes["a1_v24_o21a.py"]["path"], "source_complete": "YES", "learned_state_required": "YES", "state_analytically_derivable": "NO", "source_only_result": "Architecture 2048→256→64 and L2 normalization recovered; O21B key exact, O21A-MR query absent"},
        {"component": "VGen2 Core", "source_or_config": source_hashes["clr_v23_oracle_r2.py"]["path"], "source_complete": "YES for module definition; training source partly recovered", "learned_state_required": "YES", "state_analytically_derivable": "NO", "source_only_result": "Core state keys and exact architecture recovered; trained core/value head state absent; VGen2 run identity not certified"},
        {"component": "O3a allocator", "source_or_config": source_hashes["a1_v24_o3a.py"]["path"], "source_complete": "YES", "learned_state_required": "YES", "state_analytically_derivable": "NO", "source_only_result": "K-aware MLP and soft empty-mass loss recovered; weights absent"},
        {"component": "Baseline UpdateRouter", "source_or_config": source_hashes["a1_v24_o3b.py"]["path"], "source_complete": "YES", "learned_state_required": "YES", "state_analytically_derivable": "NO", "source_only_result": "Architecture/objective/defaults and threshold-selection logic recovered; checkpoint, selected threshold and actual run config absent"},
        {"component": "L2, dot-product read, zero-state, occupancy masks, slot/action control flow", "source_or_config": "hash-matched source modules", "source_complete": "YES", "learned_state_required": "NO", "state_analytically_derivable": "YES", "source_only_result": "Deterministically recoverable without model weights"},
        {"component": "Label/index maps and MR cache", "source_or_config": "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache", "source_complete": "YES for cache artifacts/metadata presence", "learned_state_required": "NO", "state_analytically_derivable": "YES", "source_only_result": "Dataset/cache survives; this does not encode model parameters"},
        {"component": "Original baseline experiment config/report", "source_or_config": "expected clr_v24_o3b_update/config.json and output report", "source_complete": "NO", "learned_state_required": "N/A", "state_analytically_derivable": "NO exact reconstruction", "source_only_result": "Only defaults and diagnostic O3b-K/L configs survive; exact seed overrides/best step/threshold/report not recoverable"},
    ]
    write_csv(OUT / "source_only_reconstruction_audit.csv", list(source_rows[0]), source_rows)

    inventory = []
    def inv(path, kind, result, size=None, digest=None, scope=""):
        inventory.append({"scope": scope, "artifact_path": path, "artifact_type": kind, "bytes": size, "sha256": digest, "forensic_result": result})

    for name, (path, expected) in source_specs.items():
        if path.is_file(): inv(str(path), "source", "present; local SHA matches recorded/remote SHA" if sha(path) == expected else "present; SHA MISMATCH", path.stat().st_size, sha(path), "local audit source")
    for p, role in [(M2 / "decision_rule_frozen.json", "Rule A frozen config"), (M2 / "final_report.md", "M2 final report"), (M2 / "candidate_b_frozen_manifest.json", "M2 Candidate B freeze manifest"), (M3 / "dependency_recovery_manifest.json", "M3 recovery manifest"), (M3 / "remote_dependency_search.json", "M3 remote search evidence"), (M3 / "local_dependency_search.json", "M3 local search evidence"), (M3 / "final_report.md", "M3 report")]:
        inv(str(p), role, "present; hashed", p.stat().st_size, sha(p), "local prior-stage artifact")
    for p in sorted(M2.rglob("*.log")):
        inv(str(p), "M2 log", "present; Candidate B calibration/holdout logs, not historical O3b router training", p.stat().st_size, sha(p), "local M2")
    for p in sorted(M3SRC.glob("*.json")):
        inv(str(p), "diagnostic config snapshot", "present; O3b-K/L diagnosis only; not original baseline config", p.stat().st_size, sha(p), "local M3 source_reference")
    for name, path, status in expected_remote:
        inv(path, "checkpoint/config", status, None, None, "remote active project + user-owned home/share exact-path recheck")
    inv("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt", "checkpoint", "present; exact O21B state; includes its own query encoder but that query is not an O21A-MR substitute", 4356237, "6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570", "remote active project and home mirror")
    inv(m2_candidate["checkpoint_source"], "checkpoint", "present; frozen Candidate B, not historical baseline router", None, m2_candidate["checkpoint_sha256_after"], "remote active project")
    inv("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/*", "dataset/cache", "cache directory contains manifest, relation map, train/val/TestA/TestB NPZ and metadata; no model state", None, None, "remote active project")
    inv("/tmp/tmp1i7o9mxn/*", "temporary files", "only remote module helper and pycache; no model/checkpoint", None, None, "remote user task temp")
    inv("/public/home/ac92f4k0z9 and /public/share/ac92f4k0z9", "checkpoint extension scan", "only O21B and M2 Candidate A/B/C .pt files; no additional target state; one unrelated GStreamer registry .bin", None, None, "user-owned remote storage")
    inv("H:\\LoopLM\\remote_archive\\new_server\\clr_v24_o3b_j3c_fresh_certified_feature_cache\\source_snapshot\\Ouro-1.4B\\model.safetensors", "pretrained base model", "pinned upstream model; SHA in certified source manifest; M4 does not recompute large-file hash", 2869336434, model_manifest["pinned_model_sha256"], "local archived J3C source snapshot")
    inv("/root/private_data/clr_v24_o3b_l/code/a1_v23_r2_common.py and clr_v23_oracle_r2.py", "additional source", "read-only copied to M4; local hashes matched remote sha256sum", None, None, "remote original code and M4 source_reference")
    write_csv(OUT / "artifact_forensics_inventory.csv", ["scope", "artifact_path", "artifact_type", "bytes", "sha256", "forensic_result"], inventory)

    reconstructed = {
        "experiment": "O3b-M4",
        "verdict": "PARTIALLY RECONSTRUCTABLE — RETRAINING REQUIRED",
        "learned_checkpoints_created": [],
        "learned_checkpoints_modified": [],
        "deterministic_contract_elements": [
            {"component": "K/V/occupancy shapes and zero initialization", "source": "a1_v24_o3b.py, a1_v23_r2_common.py, clr_v23_oracle_r2.py", "status": "Recovered in historical_bundle_contract.json; no new model artifact materialized"},
            {"component": "L2 normalization and Q-K dot/argmax matcher", "source": "a1_v24_o21a.py and a1_v24_o3a.py", "status": "Recovered from exact source"},
            {"component": "NEW/UPDATE action and slot semantics", "source": "a1_v24_o3b.py", "status": "Recovered from exact source; not behaviorally executed"},
            {"component": "cache relation/value index metadata", "source": "O21A-MR cache manifest/relations/person metadata", "status": "Present remotely; data, not learned state"},
        ],
        "not_reconstructed_reason": "Every missing target module listed is learned and has no complete serialized state. Source architecture alone or same-shaped foreign weights cannot certify historical behavior.",
        "source_snapshots_copied": [file_record(M4SRC / "a1_v23_r2_common.py", "source"), file_record(M4SRC / "clr_v23_oracle_r2.py", "source")],
    }
    dump_json(OUT / "reconstructed_components_manifest.json", reconstructed)

    dryrun = {
        "status": "STATIC INTERFACE DRY RUN PASS; behavioral/runtime evaluation NOT RUN",
        "kind": "shape-only contract trace; no modules with missing learned weights were instantiated or scored",
        "uses_fabricated_weights": False,
        "mock_or_stub_outputs": "Only symbolic shape contracts; no generated tensor weights or predictions",
        "checks": {
            "input_hidden": {"dtype": "float32 as source cache cast", "shape": "[B,2048]", "status": "PASS"},
            "key_encoder_output": {"shape": "[B,64]", "normalization": "L2", "status": "PASS by source"},
            "query_encoder_output": {"shape": "[B,64]", "normalization": "L2", "status": "PASS by source; learned weights absent"},
            "memory_state": {"K": "[B,4,64]", "V": "[B,4,64]", "occupancy": "[B,4]", "status": "PASS by source"},
            "allocator": {"input": "[B,68]", "output": "[B,4]", "status": "PASS by source; weights absent"},
            "router": {"pair_input": "[B,4,257]", "pair_logits": "[B,4]", "action_input": "[B,69]", "action_logits": "[B,2]", "status": "PASS by source; weights/threshold absent"},
            "update_write": {"key": "preserve target K", "value": "write selected slot through Core", "status": "PASS by source semantics"},
            "new_write": {"key": "write incoming key to free slot", "value": "write selected slot through Core", "status": "PASS by source semantics"},
            "retrieval": {"scores": "[B,4]", "selected_slot": "[B]", "answer_logits": "[B,64]", "status": "PASS by source semantics"},
        },
        "not_run": ["torch module execution with learned weights", "baseline metric reproduction", "retrieval/full-KV evaluation", "free-slot regression", "Candidate B reconnect"],
    }
    dump_json(OUT / "bundle_interface_dryrun.json", dryrun)

    anchors = {
        "experiment": "O3b-M4 historical anchor recovery",
        "source_quality": "User-provided project history + extant source/report evidence; anchors below are labeled by provenance and are not conflated with the missing baseline router's own test metrics.",
        "O21A_MR": {"last_validation_pass": "NOT RECOVERED as an output artifact", "metrics": None, "available": "Exact training source and MR cache survive; source report template says to report retrieval/answer gates, but metrics.json/train.log/best_model.pt were not found."},
        "VGen2": {"last_validation_pass": "NOT RECOVERED", "metrics": None, "available": "a1_v24_vgen.py source contains staged validation selection; no VGen2-specific run config, metrics, train log, or checkpoint; expected output path differs from source default."},
        "O3a": {"last_validation_pass": "QUALITATIVE PASS from supplied project state: 'PASS — learned free-slot allocation'", "metrics": None, "available": "Exact source objective/hparams survive; original report, config, allocator checkpoint and exact numeric validation metrics absent."},
        "O3b_baseline_router": {"last_validation_pass": "NOT RECOVERED for router_best.pt", "metrics": None, "available": "Original source and defaults survive; original output config/report/log/router checkpoint absent."},
        "historical_O3b_operational_context_from_user_brief": {
            "raw_cosine_same_key_f1_percent": {"TestA": 88.44, "TestB": 86.56},
            "best_pair_mlp_same_key_f1_percent": {"TestA": 95.09, "TestB": 94.07},
            "update_recall_percent": {"TestA": 96.20, "TestB": 94.07},
            "same_entity_different_relation_P2_accuracy_percent": {"TestA": 82.4, "TestB": 83.6},
            "K_to_relation_probe_percent_approx": "59–61",
            "caveat": "These are prior O3b operational/diagnostic anchors as supplied, not certified metrics of the missing original UpdateRouter checkpoint."},
        "O21B": {"checkpoint_sha256": "6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570", "source_run_metric": "Detailed O21B run metrics/config not recovered in this bundle."},
    }
    dump_json(OUT / "historical_metric_anchors.json", anchors)

    requirements = """# O3b-M4 — Retraining / recovery requirements (not executed)\n\n**RETRAINING REQUIRED.** The historical integration bundle cannot be certified from source/config alone. M4 did not train or initialize substitute learned modules.\n\n## Required target states and order\n\n1. **Upstream R2 OracleLocalCore training state/config (recovery dependency).** The source-faithful O3b loader calls `load_r2_checkpoint` before loading VGen2. The final VGen2 checkpoint would contain the full core state and supersede the R2 values at serving time, but the original VGen2 training path uses R2 and the checkpoint is absent. The module source is known (`content_read: Linear(2048,64)`, `shared_gru: GRUCell(64,64)`); the original R2 training objective/run config was not recovered from the searched source tree. Recover the original R2 checkpoint/config if available; do not silently replace it with random initialization.\n2. **O21A-MR QueryEncoder state.** O3b consumes `best_model.pt['query']`; architecture is `2048→256→GELU→64→L2`, 540,992 parameters. The exact O21A-MR checkpoint is missing. Source recipe: 3,000 default steps, AdamW LR 1e-3, weight decay 0.01, batch up to 128, gradient clip 1.0, retrieval CE + 0.5 alignment CE + 0.1 entity/query-entity CE; validation macro retrieval every 100 steps. The actual warm-start flag, best step, exact config overrides, and metrics are missing. O21B's shape-matched query must not be substituted.\n3. **VGen2 Core + MRValueHeads.** Required payload keys are `core` (156,096 params) and `value_heads` (16,640 params). Available `a1_v24_vgen.py` documents T1-A0/T1-A1/T2/T3 stages, AdamW LR 3e-4, WD 0.01, 3,000 default steps, T2/T3 50% replay, and validation selection. But the O3b expected artifact is under `clr_v24_vgen2/best.pt`; only `a1_v24_vgen.py` with default output `clr_v24_vgen` was found. No VGen2-specific config, exact selected stage, metrics, or log survives. Its exact historical objective/config is therefore only partially recoverable.\n4. **O3a K-aware allocator.** `Allocator(68→128→GELU→4)`, 9,348 params. Source recipe is recoverable: AdamW 3e-4, WD 0.01, clip 1.0, 3,000 default steps; loss is negative log probability mass over all empty slots, with 50% random legal occupancy initialization; validation macro valid-allocation rate chooses best. The supplied history says O3a PASS, but exact numeric anchor, run config/selected step, and checkpoint are absent.\n5. **Original O3b UpdateRouter.** `pair:257→128→GELU→1` and `action:69→96→GELU→2`, 40,067 params. Source recipe is recoverable: AdamW 3e-4, WD 0.01, 3,000 default steps, 6,000 train / 1,000 validation examples, balanced NEW/UPDATE, action CE + target-slot CE + 0.5 occupied-pair BCE, clip 1.0; the UPDATE threshold is selected on validation and saved with the router. The exact historical router threshold, best step, actual environment overrides and router-specific metrics are missing. User-provided earlier O3b metrics are not accepted as this router's metric anchor.\n\n## Sufficiency judgment\n\n- **Can rerun a source-defined training recipe?** Mostly yes for O21A-MR, O3a, and the baseline router; only partly for VGen2.\n- **Can reproduce the exact historical learned states/configuration without additional artifacts?** No. R2 initializer state/objective, O21A-MR warm-start/actual run record, VGen2 run identity/config, and original O3b/O3a selected state metadata are absent.\n- **Historical metric anchors:** Only O3a has a qualitative PASS in the supplied project history; exact component-specific numeric reports for O21A-MR/VGen2/O3a/router were not found. Do not calibrate to unrelated O3b/J/K/L numbers.\n\n## Gate\n\nDo not run baseline reproduction or Candidate B reconnect until either (a) exact checkpoints are recovered and hash-certified, or (b) a separately authorized new baseline is trained under frozen, explicit configs with new acceptance anchors. This document is a requirement list only; no training was run.\n"""
    (OUT / "retraining_requirements.md").write_text(requirements, encoding="utf-8")

    verdict = "O3b-M4-R — BASELINE BUNDLE RECONSTRUCTION REQUIRES RETRAINING"
    final = f"""# O3b-M4 — Certified Baseline Integration Bundle Reconstruction Audit\n\n## FINAL VERDICT\n\n**{verdict}**\n\nThe original integration contract is recoverable from exact source, but the bundle is not fully reconstructable without retraining: four required learned targets (O21A-MR query, VGen2 core/value heads, O3a allocator, and O3b baseline router) have no certified state. The upstream R2 initialization checkpoint is also absent. No shape-compatible foreign state was substituted.\n\n## Frozen artifacts\n\n- Candidate B checkpoint SHA-256: `{m2_candidate['checkpoint_sha256_after']}`; remote recheck matched.\n- Candidate B config SHA-256: `{m2_candidate['candidate_config_sha256']}`.\n- Rule A threshold: `{rule_threshold}`; frozen config SHA-256 `{sha(rule_path)}`.\n- M2 final report SHA-256: `{sha(m2_report)}`.\n- M3 dependency manifest SHA-256: `{sha(m3_dep_path)}`.\n- Original `a1_v24_o3b.py` SHA-256: `{source_hashes['a1_v24_o3b.py']['sha256']}`. Candidate B and Rule A were not changed.\n\n## Component disposition\n\n| Component | Kind | State found | No-retrain status |\n|---|---|---|---|\n| O21A-MR QueryEncoder | learned | missing | **RETRAINING REQUIRED** |\n| VGen2 Core + MRValueHeads | learned | missing | **RETRAINING REQUIRED** |\n| O3a K-aware allocator | learned | missing | **RETRAINING REQUIRED** |\n| O3b baseline UpdateRouter + threshold | learned | missing | **RETRAINING REQUIRED** |\n| O21B KeyEncoder | learned | exact SHA-matched checkpoint | recoverable |\n| Ouro-1.4B source model | pretrained frozen | pinned artifact in J3C snapshot | recoverable; not loaded by O3b `load_all` when consuming the cached hidden vectors |\n| L2, dot matcher, zero state, occupancy/action control | deterministic | exact source | deterministically reconstructable |\n\n### Q1–Q21\n\n1. **O21A-MR learned or deterministic?** Learned. Its QueryEncoder is used by O3b; O3b freezes it.\n2. **Complete O21A-MR learned state found?** No.\n3. **Can O21A-MR be restored without retraining?** No. O21B's same-shaped query is different provenance and prohibited as a substitute.\n4. **VGen2 learned or deterministic?** Learned: trained Core state and learned per-slot value heads.\n5. **Complete VGen2 state found?** No.\n6. **Can VGen2 be restored without retraining?** No.\n7. **O3a allocator learned or deterministic?** Learned K-aware MLP, 68→128→4.\n8. **Does it actually require a checkpoint?** Yes; original O3b strict-loads the `allocator` state.\n9. **Can O3a be restored without retraining?** No.\n10. **Baseline router learned or deterministic?** Learned pair/action MLP.\n11. **Complete router state found?** No.\n12. **Can the baseline router be restored without retraining?** No.\n13. **Was required state found nested in another bundle?** No certified target state. O21B contains its own QueryEncoder; Candidate B is a different head; neither is target state.\n14. **Can we build the complete historical bundle contract?** Yes, at source/interface level. Exact VGen2 run identity/config and original baseline config/threshold remain unknown.\n15. **Can we do an interface-only dry run?** Yes, static shape/control-flow contract PASS; no missing learned model was run or scored.\n16. **Whole bundle status?** `PARTIALLY RECONSTRUCTABLE — RETRAINING REQUIRED`.\n17. **Which components require retraining?** O21A-MR QueryEncoder; VGen2 Core and MRValueHeads; O3a K-aware allocator; original O3b UpdateRouter. The R2 initializer checkpoint is also missing and is an upstream source-loader/training dependency; see the qualification below.\n18. **Are historical objectives/configs sufficient?** Partially. Source defaults/objectives are largely available for O21A-MR/O3a/router; exact run overrides/best steps/metrics are absent. VGen2-specific identity/config is incomplete because only `a1_v24_vgen.py` was found and its default path differs. R2's source architecture survives but its original training recipe/checkpoint does not. Therefore exact historical state reproduction is not supported.\n19. **Baseline reproduction allowed?** No.\n20. **Candidate B reconnect allowed?** No.\n21. **O3c allowed?** No.\n\n## Forensic scope and disposition\n\nRead-only search covered local workspace/archive, local M2/M3 records, the active remote CLR project, the user's own remote home mirror/share, component code, checkpoint-like file extensions, configs/logs, MR cache files, and the identified remote temporary directory. The remote checkpoint inventory showed O21B and M2 Candidate A/B/C checkpoints only; the expected O21A-MR/VGen2/O3a/router directories or files were absent. The O21A-MR directory contains the train/validation/TestA/TestB cache and metadata, not model parameters. Other users' data was not inspected.\n\nThe O3b contract's deterministic interfaces are recoverable. The candidate M2 policy remains separately frozen; it is not part of the original baseline bundle and is not used as a substitute. No baseline score, retrieval/full-KV test, baseline reproduction, or reconnect was run.\n\n## Exactly one next experiment\n\n**Reconstruct only the missing learned baseline components using their recovered historical objectives/configs and validate each against its historical metric anchor before assembling a new certified bundle.** Start at the most upstream missing dependency: recover the R2 Core checkpoint/recipe gate, then O21A-MR → VGen2 → O3a allocator → baseline router. Because numeric historical anchors for several components are missing, do not start training until those anchors are recovered or a new baseline acceptance protocol is explicitly authorized.\n\n## Output and integrity\n\n- Output directory: `{OUT}`\n- Key files: `historical_bundle_contract.md`, `component_recoverability_matrix.csv`, `artifact_forensics_inventory.csv`, `partial_state_recovery_audit.csv`, `source_only_reconstruction_audit.csv`, `retraining_requirements.md`, `bundle_interface_dryrun.json`.\n- Git SHA: N/A (workspace is not a Git repository).\n- Model/checkpoint modifications: none. Only original source files were copied read-only into this new audit folder.\n- Data leakage: none; no evaluation was run.\n- GPU/training: none.\n"""
    final = final.replace(
        "## Exactly one next experiment",
        "## Forensic limitation\n\nRemote tensor deserialization/signature inspection was unavailable because importing PyTorch on the remote host failed on missing `libgalaxyhip.so.5`. The scoped filename/extension inventory and exact expected-path checks found no target checkpoints; local O21B state and source-defined strict-load signatures were inspected. Therefore the missing-state conclusion is based on bounded file/path inventory, not successful remote tensor loading.\n\n## Exactly one next experiment",
    )
    (OUT / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    (OUT / "final_report.md").write_text(final, encoding="utf-8")

    log_lines = [
        "O3b-M4 read-only forensic audit log",
        f"created_utc={now}",
        "Local roots: H:\\LoopLM, remote_archive\\cheems, remote_archive\\new_server, checkpoints, M2, M3.",
        "Remote roots: /root/private_data/clr_v24_o3b_l; /public/home/ac92f4k0z9; /public/share/ac92f4k0z9; /tmp/tmp1i7o9mxn.",
        "Remote model-file extension scan found O21B plus M2 Candidate A/B/C state files only; expected O21A-MR/VGen2/O3a/router/R2 state absent.",
        "Remote O21A-MR directory contains cache manifest, relation map, train/val/TestA/TestB NPZ and metadata; no checkpoint.",
        "Remote temp directory contains _remote_module_non_scriptable.py and its pycache only.",
        "Remote Candidate B SHA recheck matched frozen M2 SHA; O21B SHA recheck matched the archive SHA.",
        "O21B checkpoint was inspected locally with trusted weights-only-off loading: top keys include key/query; query tensor shapes match but provenance is not O21A-MR and it was rejected as a substitute.",
        "Remote PyTorch tensor deserialization/signature inspection was unavailable because import failed on missing libgalaxyhip.so.5; no-target findings are based on scoped filename/extension inventory and exact path checks, not remote tensor loading.",
        "Additional source files a1_v23_r2_common.py and clr_v23_oracle_r2.py were copied read-only from the remote user's project code directory; local SHA-256 matched remote sha256sum.",
        "No torch model inference, training, baseline reproduction, retrieval/full-KV evaluation, or reconnect was performed.",
        "No other users' storage was inspected; no model/checkpoint was modified.",
    ]
    (OUT / "audit_search_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # Hash all deliverables and read-only source snapshots, excluding the manifest itself.
    manifest_path = OUT / "artifact_sha256_manifest.json"
    records = []
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x != manifest_path):
        records.append({"path": str(p.relative_to(OUT)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha(p)})
    dump_json(manifest_path, {"experiment": "O3b-M4", "created_utc": now, "hash_algorithm": "SHA-256", "manifest_excludes_itself": True, "files": records})


if __name__ == "__main__":
    main()
