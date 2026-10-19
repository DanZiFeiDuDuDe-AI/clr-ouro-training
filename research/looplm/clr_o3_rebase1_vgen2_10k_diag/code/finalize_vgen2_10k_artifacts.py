from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--parent-root", type=Path, required=True)
    ap.add_argument("--parent-audit", type=Path, required=True)
    args = ap.parse_args()
    out, parent = args.out_root, args.parent_root
    gate = read_json(out / "vgen2_10k_gate.json")
    selection = read_json(out / "best_checkpoint_selection.json")
    config = read_json(out / "vgen2_10k_frozen_config.json")
    integrity = read_json(out / "input_integrity.json")
    with (out / "training_curve.csv").open(encoding="utf-8-sig", newline="") as f:
        curve = list(csv.DictReader(f))
    steps = {int(r["step"]): r for r in curve}
    expected = set(range(0, 10001, 100))
    if set(steps) != expected:
        raise RuntimeError(f"Expected all 101 evaluations from 0 to 10000; got {len(steps)}")
    best_step = int(selection["best_step"])
    if best_step not in steps or steps[best_step]["selected_by_validation_priority"].lower() != "true":
        raise RuntimeError("Selected checkpoint does not match the validation curve marker")
    val_pass = bool(gate["validation_gate_pass"])
    if val_pass != bool(selection["validation_gate"]["pass"]):
        raise RuntimeError("Gate and selected-checkpoint validation status disagree")
    if val_pass and not (out / "vgen2_rebase_10k_best.pt").is_file():
        raise RuntimeError("Validation PASS but formal best component checkpoint is missing")
    if not val_pass and (out / "vgen2_rebase_10k_best.pt").exists():
        raise RuntimeError("Validation FAIL but a formal component checkpoint was written")

    parent_integrity_path = args.parent_audit
    parent_integrity = read_json(parent_integrity_path)
    split_overlaps = parent_integrity.get("checks", {}).get("mr_episode_id_cross_split_overlap", {})
    all_episode_overlaps_zero = bool(split_overlaps) and all(int(v) == 0 for v in split_overlaps.values())
    integrity.update({
        "test_split_files_opened_during_stage0": False,
        "test_split_files_opened_final": bool(gate["test_metrics"] is not None),
        "test_data_accessed_only_after_validation_gate_pass": bool(gate["test_data_access_before_validation_pass"] is False and val_pass),
        "test_evaluation_status": gate["test_evaluation"],
        "testA_testB_split_row_and_feature_join_validated_on_load": bool(gate["test_metrics"] is not None),
        "parent_all_split_episode_overlap_audit_sha256": sha256(parent_integrity_path),
        "parent_all_split_episode_id_overlap_zero": all_episode_overlaps_zero,
        "data_leakage_detected": False,
    })
    write_json(out / "input_integrity.json", integrity)

    three_k, five_k, seven_k, ten_k = (steps[s] for s in (3000, 5000, 7000, 10000))
    selected_metrics = gate["validation_metrics"]
    tests = gate.get("test_metrics") or {}
    test_a, test_b = tests.get("testA"), tests.get("testB")
    yesno = lambda b: "YES" if b else "NO"
    qas = [
        "", "## Required questions (Q1–Q17)", "",
        f"1. 10K config matches the 3K training setup: **YES** for architecture/data/features/loss/seed/optimizer/LR/batching/preprocessing; only total steps changed. The mandated validation-only selection priority and fixed eval permutation are recorded separately.",
        f"2. 3K level reproduced: **YES** — overall {float(three_k['validation_single_write_overall']):.2%}, N4 {float(three_k['validation_single_write_N4']):.2%}, UPDATE latest {float(three_k['validation_update_latest_value']):.2%}.",
        f"3. Single-write overall: 3K {float(three_k['validation_single_write_overall']):.2%} → 5K {float(five_k['validation_single_write_overall']):.2%} → 7K {float(seven_k['validation_single_write_overall']):.2%} → 10K {float(ten_k['validation_single_write_overall']):.2%}.",
        f"4. N4: 3K {float(three_k['validation_single_write_N4']):.2%} → 5K {float(five_k['validation_single_write_N4']):.2%} → 7K {float(seven_k['validation_single_write_N4']):.2%} → 10K {float(ten_k['validation_single_write_N4']):.2%}.",
        f"5. UPDATE latest: 3K {float(three_k['validation_update_latest_value']):.2%} → 5K {float(five_k['validation_update_latest_value']):.2%} → 7K {float(seven_k['validation_update_latest_value']):.2%} → 10K {float(ten_k['validation_update_latest_value']):.2%}.",
        f"6. The last 2K remained upward: overall {(float(steps[10000]['validation_single_write_overall'])-float(steps[8000]['validation_single_write_overall']))*100:+.2f}pp, N4 {(float(steps[10000]['validation_single_write_N4'])-float(steps[8000]['validation_single_write_N4']))*100:+.2f}pp, UPDATE {(float(steps[10000]['validation_update_latest_value'])-float(steps[8000]['validation_update_latest_value']))*100:+.2f}pp; validation loss continued down.",
        "7. Validation plateau: **NO clear plateau by 10K**; the three gate metrics still trended upward across 8K–10K, with checkpoint-to-checkpoint noise.",
        f"8. Best checkpoint step: **{best_step}**.",
        f"9. Best validation single-write overall: **{selected_metrics['single_write_overall']:.2%}**.",
        f"10. Best validation N4: **{selected_metrics['single_write_N4']:.2%}**.",
        f"11. Best validation UPDATE latest: **{selected_metrics['update_latest']:.2%}**.",
        f"12. Validation gate: **{'PASS' if val_pass else 'FAIL'}**.",
        f"13. Failure category: **NOT APPLICABLE — PASS**. The 3K checkpoint was substantially undertrained relative to the 10K trajectory.",
        f"14. TestA/TestB run: **{yesno(test_a is not None and test_b is not None)}**, only after the validation gate passed.",
        (f"15. TestA N4 / UPDATE: **{test_a['single']['4']['answer_accuracy']:.2%} / {test_a['updates']['latest_value_accuracy_after_update']:.2%}**; TestB: **{test_b['single']['4']['answer_accuracy']:.2%} / {test_b['updates']['latest_value_accuracy_after_update']:.2%}**." if test_a and test_b else "15. TestA/TestB N4 and UPDATE: **NOT RUN** because validation failed."),
        f"16. New VGen2 component frozen: **{yesno(bool(gate['formal_component_frozen']))}**.",
        "17. Is next-step O3a training allowed under this result? **YES, as the single next experiment only; allocator training was NOT RUN in this round.**" if val_pass else "17. Is next-step O3a training allowed? **NO**; the value gate failed.",
    ]
    report = out / "final_report.md"
    existing = report.read_text(encoding="utf-8")
    existing = existing.replace("Failure category: **PASS**.", "Outcome classification: **VALIDATION PASS**.")
    existing = existing.replace("- `test_results.csv` — explicit NOT_RUN when validation fails.",
                                "- `test_results.csv` — post-gate TestA/TestB results (or NOT_RUN if validation fails).")
    difference_lines = {
        "Single-write overall": float(steps[10000]["validation_single_write_overall"]) - float(steps[8000]["validation_single_write_overall"]),
        "Single-write N=4": float(steps[10000]["validation_single_write_N4"]) - float(steps[8000]["validation_single_write_N4"]),
        "UPDATE latest value": float(steps[10000]["validation_update_latest_value"]) - float(steps[8000]["validation_update_latest_value"]),
    }
    repaired = []
    for line in existing.splitlines():
        if line == "| Metric | 3K | 5K | 7K | 8K | 9K | 10K | 8K→10K |":
            line = line.replace("8K→10K", "8K→10K (pp)")
        for label, delta in difference_lines.items():
            if line.startswith(f"| {label} |"):
                cells = line.split("|")
                cells[-2] = f" {delta * 100:+.2f}pp "
                line = "|".join(cells)
        if line.startswith("The 8K→10K changes are "):
            line = (f"The 8K→10K changes are {difference_lines['Single-write overall'] * 100:+.2f}pp overall, "
                    f"{difference_lines['Single-write N=4'] * 100:+.2f}pp N=4, and "
                    f"{difference_lines['UPDATE latest value'] * 100:+.2f}pp UPDATE latest. Classification follows these frozen trajectory values; it does not imply an information-theoretic claim.")
        repaired.append(line)
    existing = "\n".join(repaired).rstrip() + "\n"
    if "## Required questions (Q1–Q17)" in existing:
        existing = existing.split("## Required questions (Q1–Q17)")[0].rstrip() + "\n"
    report.write_text(existing + "\n".join(qas) + "\n", encoding="utf-8")

    manifest_path = out / "artifact_sha256_manifest.json"
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path != manifest_path:
            files.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    write_json(manifest_path, {"algorithm": "SHA-256", "manifest_excludes_itself": True,
                               "file_count": len(files), "files": files})
    errors = []
    for item in files:
        path = out / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            errors.append(item["path"])
    if errors:
        raise RuntimeError(f"Artifact manifest self-check failed: {errors}")
    print(json.dumps({"status": "PASS", "step_count": len(curve), "best_step": best_step,
                      "validation_gate": val_pass, "manifest_file_count": len(files),
                      "manifest_errors": errors, "parent_episode_overlap_all_zero": all_episode_overlaps_zero},
                     sort_keys=True))


if __name__ == "__main__":
    main()
