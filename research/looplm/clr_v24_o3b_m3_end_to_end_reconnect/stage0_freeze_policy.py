from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"H:\LoopLM")
OUT = ROOT / "clr_v24_o3b_m3_end_to_end_reconnect"
M2 = ROOT / "clr_v24_o3b_m2_growing_slot_calibration"
M = ROOT / "clr_v24_o3b_m_operational_samekey_rescue"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((M2 / "candidate_b_frozen_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((M2 / "decision_rule_frozen.json").read_text(encoding="utf-8"))
    rule = decision["rule"]
    threshold = float(decision["params"]["threshold"])
    if frozen["candidate"] != "B" or not frozen["weights_frozen"]:
        raise RuntimeError("Candidate B frozen manifest did not certify a frozen Candidate B")
    if rule != "A" or threshold != 0.8504285216331482:
        raise RuntimeError("Unexpected frozen decision rule or threshold")

    artifact_paths = {
        "candidate_b_checkpoint": M / "models/candidate_b_seed_43.pt",
        "candidate_b_config": M / "winner_config.json",
        "candidate_b_training_config": M / "experiment_config.json",
        "feature_source_manifest": M / "feature_source_manifest.json",
        "train_only_normalization_statistics": M / "feature_standardization.json",
        "label_and_pair_construction_manifest": M / "pair_dataset_manifest.json",
        "growing_slot_episode_manifest": M2 / "episode_dataset_manifest.json",
        "true_update_manifest": M2 / "true_update_dataset_manifest.json",
        "decision_rule_a_config": M2 / "decision_rule_frozen.json",
        "candidate_frozen_manifest": M2 / "candidate_b_frozen_manifest.json",
        "m2_input_integrity": M2 / "input_integrity.json",
        "m2_action_gate": M2 / "action_gate.json",
        "m2_value_replacement_gate": M2 / "value_replacement_gate.json",
        "candidate_b_training_code": M / "o3b_m_runner.py",
        "m2_action_simulation_code": M2 / "o3b_m2_runner.py",
    }
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing policy artifacts: " + "; ".join(missing))
    config_sha = sha256(artifact_paths["candidate_b_config"])
    checkpoint_sha = sha256(artifact_paths["candidate_b_checkpoint"])
    if config_sha != frozen["candidate_config_sha256"]:
        raise RuntimeError("Candidate B configuration no longer matches its frozen manifest")
    if checkpoint_sha != frozen["checkpoint_sha256_before"] or checkpoint_sha != frozen["checkpoint_sha256_after"]:
        raise RuntimeError("Candidate B checkpoint no longer matches its frozen before/after hashes")

    records = []
    for name, path in artifact_paths.items():
        records.append({
            "name": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "frozen": True,
        })
    manifest = {
        "experiment": "O3b-M3",
        "status": "POLICY_FROZEN_BEFORE_DEPENDENCY_RECOVERY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "B — factorized key + symmetric pair MLP",
        "candidate_seed": 43,
        "candidate_checkpoint_sha256_before": checkpoint_sha,
        "candidate_checkpoint_sha256_after_stage0": checkpoint_sha,
        "candidate_state_dict_sha256_before": frozen["state_dict_sha256_before"],
        "candidate_state_dict_sha256_after_stage0": frozen["state_dict_sha256_after"],
        "candidate_config_sha256": config_sha,
        "decision_rule": rule,
        "threshold_exact": threshold,
        "threshold_display": "0.8504285",
        "normalization": "frozen train-only per-dimension E/R statistics; no refit",
        "feature_source": {
            "entity": "J3C H_mention_fresh, 2048D frozen mention-mean carrier",
            "relation": "J3C H_base_I0, 2048D frozen carrier, row-aligned with MR fact_H",
            "candidate_key_parts": "separate 64D entity and relation projections",
        },
        "label_semantics": {
            "P1": "same entity AND same relation => same-key / UPDATE; M3 true updates must also use a different object_value",
            "P2": "same entity AND different relation => different key / NEW",
            "P3": "different entity AND same relation => different key / NEW",
            "P4": "different entity AND different relation => different key / NEW",
        },
        "action_semantics": {
            "score_all_occupied_slots": True,
            "update_if": "top1_score >= frozen threshold",
            "update_target": "argmax occupied-slot score",
            "otherwise": "NEW via original frozen O3a free-slot allocator",
            "capacity": 4,
            "no_margin_or_slot_count_threshold": True,
        },
        "artifacts": records,
        "immutable_for_entire_m3": True,
    }
    out_path = OUT / "m2_address_policy_frozen_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(out_path),
        "candidate_checkpoint_sha256": checkpoint_sha,
        "candidate_config_sha256": config_sha,
        "rule": rule,
        "threshold": threshold,
        "frozen_artifact_count": len(records),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
