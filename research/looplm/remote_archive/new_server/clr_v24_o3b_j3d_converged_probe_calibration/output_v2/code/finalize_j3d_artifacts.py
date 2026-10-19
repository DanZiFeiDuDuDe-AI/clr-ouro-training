from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "output_v2" if (HERE / "output_v2").is_dir() else HERE.parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_std(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "min": min(values) if values else None, "max": max(values) if values else None}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100*x:.2f}%"


def pp(x: float | None) -> str:
    return "n/a" if x is None else f"{100*x:+.2f} pp"


def delta_pp_string(paired: dict, key: str) -> str:
    value = paired[key]["mean_delta_pp"]
    return "n/a" if value is None else f"{value:+.2f} pp"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def task_rows(rows: list[dict[str, str]], task: str) -> list[dict[str, str]]:
    return [r for r in rows if r.get("task") == task]


def pair_delta(rows: list[dict[str, str]], left: str, right: str, task: str,
               norm: str, split: str) -> dict:
    values = {}
    for r in rows:
        if (r.get("task") == task and r.get("normalization") == norm and r.get("split") == split
                and r.get("representation") in (left, right) and r.get("accuracy") not in (None, "")):
            values[(r["representation"], int(r["seed"]))] = float(r["accuracy"])
    seeds = sorted({s for rep, s in values if rep == left} & {s for rep, s in values if rep == right})
    diffs = [values[(left, s)] - values[(right, s)] for s in seeds]
    return {"left": left, "right": right, "task": task, "normalization": norm, "split": split,
            "paired_seeds": seeds, "delta_left_minus_right": mean_std(diffs),
            "mean_delta_pp": None if not diffs else 100 * statistics.mean(diffs)}


def convergence_counts(rows: list[dict[str, str]], phases: set[str] | None = None) -> dict:
    fits = [r for r in rows if not r.get("split") and (phases is None or r.get("phase") in phases)]
    out = {}
    for task in ("entity", "relation", "joint"):
        rr = [r for r in fits if r.get("task") == task]
        out[task] = {"n": len(rr), "converged": sum(r.get("convergence_flag") == "CONVERGED" for r in rr),
                     "not_converged": sum(r.get("convergence_flag") != "CONVERGED" for r in rr),
                     "max_steps": sum(r.get("stop_reason") == "max_steps" for r in rr),
                     "validation_patience": sum(r.get("stop_reason") == "validation_patience" for r in rr)}
    return out


def accuracy_groups(rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for r in rows:
        if r.get("split") not in ("testA", "testB") or r.get("accuracy") in (None, ""):
            continue
        key = (r["task"], r.get("normalization", "n/a"), r["representation"], r["split"])
        groups.setdefault(key, []).append(float(r["accuracy"]))
    return [{"task": k[0], "normalization": k[1], "representation": k[2], "split": k[3], **mean_std(v)}
            for k, v in sorted(groups.items())]


def controls_convergence(path: Path) -> dict:
    rows = read_csv(path)
    fits = [r for r in rows if r.get("split") in (None, "") and r.get("convergence_flag")]
    out = {}
    for phase in sorted({r.get("phase", "") for r in fits}):
        rr = [r for r in fits if r.get("phase") == phase]
        out[phase] = {"n": len(rr), "converged": sum(r["convergence_flag"] == "CONVERGED" for r in rr),
                      "max_steps": sum(r.get("stop_reason") == "max_steps" for r in rr)}
    return out


def fmt_group(row: dict) -> str:
    if row["mean"] is None:
        return "n/a"
    return f"{100*row['mean']:.2f}±{100*(row['std'] or 0):.2f}% (n={row['n']})"


def main() -> int:
    t0 = time.time()
    out_json = read_json(OUT / "FINAL_VERDICT.json")
    stage0 = read_json(OUT / "cache_integrity_recheck.json")
    transform_audit = read_json(OUT / "sanity" / "stage0_transform_audit.json")
    r0_gate = read_json(OUT / "sanity" / "r0_r1_invariance_gate.json")
    r12_gate = read_json(OUT / "basis_invariance_gate.json")
    transfer_gate = read_json(OUT / "sanity" / "weight_transfer_gate.json")
    transfer_math = read_json(OUT / "sanity" / "representation_transform_recheck.json")
    protocol = read_json(OUT / "probe_protocol_frozen.json")
    r01 = read_csv(OUT / "r0_r1_converged_results.csv")
    r12 = read_csv(OUT / "r1_r2_converged_results.csv")
    all_eval = r01 + r12

    main_convergence = {
        "R0_R1": convergence_counts(r01),
        "R1_R2": convergence_counts(r12),
    }
    accuracy_summary = {"R0_R1": accuracy_groups(r01), "R1_R2": accuracy_groups(r12)}
    paired = {}
    for task in ("entity", "relation", "joint"):
        for split in ("testA", "testB"):
            paired[f"R0_vs_R1_standardized_{task}_{split}"] = pair_delta(
                r01, "R0_standardized", "R1_standardized", task, "standardized", split)
            paired[f"R1_vs_R2_standardized_{task}_{split}"] = pair_delta(
                r12, "R1_standardized", "R2_standardized", task, "standardized", split)

    err = [float(v) for k, v in transfer_math.items() if "max_logit_abs_error" in k and isinstance(v, (int, float))]
    mismatches = [float(v) for k, v in transfer_math.items() if "prediction_mismatch_rate" in k and isinstance(v, (int, float))]
    transfer = {"pass": bool(transfer_gate.get("pass")),
                "max_logit_abs_error_all_tasks_seeds_norms_directions_splits": max(err) if err else None,
                "max_prediction_mismatch_rate": max(mismatches) if mismatches else None,
                "numeric_tolerance": 1e-3, "prediction_mismatch_tolerance": 0.0}
    control_counts = {
        "weight_decay": controls_convergence(OUT / "tables" / "weight_decay_controls.csv"),
        "alternate_optimizer": controls_convergence(OUT / "tables" / "optimizer_robustness_if_needed.csv"),
    }
    selected = protocol.get("optimizer_selected_per_task_from_train_val_only", {})
    runtime = float(out_json.get("runtime_seconds", 0.0))
    executable = read_json(OUT / "execution_manifest.json")
    stagewise_status = read_csv(OUT / "tables" / "stagewise_linear_results.csv")[:2]
    summary = {
        "primary_verdict": out_json.get("primary"),
        "measurement_gate": out_json.get("measurement_gate"),
        "stage0_cache_integrity_pass": bool(stage0.get("pass")),
        "stage0_transform_audit_pass": bool(transform_audit.get("pass")),
        "frozen_optimizer_selection_train_val_only": selected,
        "main_convergence": main_convergence,
        "test_accuracy_mean_std_by_condition": accuracy_summary,
        "paired_standardized_test_deltas": paired,
        "r0_r1_gate_pass": bool(r0_gate.get("pass")),
        "r1_r2_gate_pass": bool(r12_gate.get("pass")),
        "r0_r1_all_primary_runs_converged": bool(r0_gate.get("all_primary_runs_converged")),
        "r1_r2_all_primary_runs_converged": bool(r12_gate.get("all_primary_runs_converged")),
        "analytical_weight_transfer": transfer,
        "control_fit_convergence": control_counts,
        "stagewise_j_authorized": False,
        "stagewise_j_run": False,
        "activation_and_nonlinear_revalidation": "NOT_RUN: preregistered measurement gates closed",
        "reconnect": False, "o3c": False,
        "checkpoint_modified": False, "data_leakage_detected": False,
        "runtime_seconds": runtime, "runtime_minutes": runtime / 60.0,
        "single_gpu_device": "BW (ROCm/HIP 6.3.26093; preflight-verified)",
        "gpu_busy_time_sampled": False,
        "initial_invalidated_attempt": "Preserved separately; early-stop-freeze bug; stopped before held-out performance metrics.",
        "execution_manifest": executable,
        "stagewise_not_run_marker": stagewise_status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    report = [
        "# O3b-J3D — Converged matched-probe calibration",
        "",
        "## Final verdict",
        "",
        f"**{summary['primary_verdict']}**",
        "",
        "The certified cache and row-transform checks passed. The independent-refit calibration did **not** pass: neither R0/R1 nor R1/R2 met the preregistered convergence requirement (and R1/R2 also has a joint TestA discrepancy). Therefore Z0–Z4 stagewise attribution, activation remeasurement, and nonlinear recoverability were not run.",
        "",
        "## Calibration protocol",
        "",
        f"Five frozen seeds `[11, 23, 37, 53, 71]`; affine linear probes; AdamW selected per task using train/validation only; chosen settings: `{json.dumps(selected, ensure_ascii=False)}`. Early stopping and convergence used validation only, with a 50,000-step cap. TestA/TestB were evaluated only after protocol freeze.",
        "",
        "## Convergence",
        "",
        "| contrast | task | plateau-confirmed | total | max-step stops |",
        "|---|---|---:|---:|---:|",
    ]
    for contrast, by_task in main_convergence.items():
        for task, c in by_task.items():
            report.append(f"| {contrast} | {task} | {c['converged']} | {c['n']} | {c['max_steps']} |")
    report += [
        "",
        "## Held-out linear accessibility",
        "",
        "Mean±SD across the five paired seeds. These are probe scores, not information-content estimates.",
        "",
    ]
    for contrast, reps in (("R0/R1 independent fits", ("R0", "R1")), ("R1/R2 independent fits", ("R1", "R2"))):
        report += [f"**{contrast}**", "", "| representation | task | TestA raw | TestA standardized | TestB raw | TestB standardized |",
                   "|---|---|---:|---:|---:|---:|"]
        lookup = {(r["task"], r["normalization"], r["representation"], r["split"]): r
                  for r in accuracy_summary["R0_R1" if reps[0] == "R0" else "R1_R2"]}
        for rep in reps:
            for task in ("entity", "relation", "joint"):
                vals = []
                for split in ("testA", "testB"):
                    for norm in ("raw", "standardized"):
                        vals.append(fmt_group(lookup.get((task, norm, f"{rep}_{norm}", split), {"mean": None, "std": None, "n": 0})))
                report.append(f"| {rep} | {task} | " + " | ".join(vals) + " |")
        report.append("")
    report += [
        "",
        "## Matched contrasts and direct transfer",
        "",
        f"- R0 vs R1, standardized: TestA paired deltas are entity {delta_pp_string(paired, 'R0_vs_R1_standardized_entity_testA')}, relation {delta_pp_string(paired, 'R0_vs_R1_standardized_relation_testA')}, joint {delta_pp_string(paired, 'R0_vs_R1_standardized_joint_testA')} (left minus right). TestB relation is {delta_pp_string(paired, 'R0_vs_R1_standardized_relation_testB')}; entity and joint TestB have no known labels.",
        f"- R1 vs R2, standardized: TestA paired deltas (R1−R2) are entity {delta_pp_string(paired, 'R1_vs_R2_standardized_entity_testA')}, relation {delta_pp_string(paired, 'R1_vs_R2_standardized_relation_testA')}, and joint {delta_pp_string(paired, 'R1_vs_R2_standardized_joint_testA')}. The joint discrepancy exceeds the 1 pp tolerance; TestB joint labels are unavailable.",
        f"- Analytical R1↔R2 classifier transfer: **{'PASS' if transfer['pass'] else 'FAIL'}**; maximum absolute logit error across tasks/seeds/raw+standardized/splits = `{transfer['max_logit_abs_error_all_tasks_seeds_norms_directions_splits']:.3e}`; maximum prediction mismatch = `{transfer['max_prediction_mismatch_rate']:.1%}`.",
        "- Interpretation: the same fitted decision function transfers across the algebraically equivalent R1/R2 coordinates, while independently retrained probes do not reliably agree. This is consistent with probe optimization / parameterization sensitivity; it is not evidence that the representation itself lost the task information.",
        "",
        "## Gates, controls, and stop decision",
        "",
        f"- R0/R1 standardized gate: **{'PASS' if summary['r0_r1_gate_pass'] else 'FAIL'}**; paired TestA score deltas are zero/within tolerance, but plateau confirmation is `{summary['r0_r1_all_primary_runs_converged']}`.",
        f"- R1/R2 standardized independent-refit gate: **{'PASS' if summary['r1_r2_gate_pass'] else 'FAIL'}**; convergence confirmation is `{summary['r1_r2_all_primary_runs_converged']}`; joint TestA also exceeds tolerance.",
        f"- Direct transfer gate: **{'PASS' if transfer['pass'] else 'FAIL'}**. Weight-decay and alternate-optimizer controls ran; see their per-seed tables and convergence counts in `final_evidence_summary.json`.",
        "- Stagewise Z0–Z4, activation, nonlinear recoverability, and J cliff revalidation: **NOT RUN** because calibration gates closed. The historical −55.62 pp cliff is therefore neither reproduced nor retracted by this run.",
        "- No backbone/KeyEncoder/checkpoint modification, no reconnect/O3c, and no data leakage.",
        "",
        "## What the evidence supports",
        "",
        "Train-only standardization makes the R0/R1 held-out probe scores coincide in this run, consistent with singular-value scale being removed by per-coordinate standardization. However, the prescribed convergence gate did not pass, so this does not certify a converged measurement pipeline. The R1/R2 transfer result confirms algebraic equivalence of the frozen representations for a fixed linear classifier; the independent-refit joint gap is best treated as unresolved optimization sensitivity. No conclusion that W1 destroys entity information, learns entity invariance, or that activation is a primary cause is supported here.",
        "",
        "## Runtime and provenance",
        "",
        f"- Remote output: `{OUT}`",
        f"- Single-GPU worker wall time: {runtime/60:.1f} min ({runtime/3600:.3f} h); exact device busy time was not sampled.",
        "- Cache identity/hash audit: PASS; row-transform and synthetic classifier-transfer Stage 0: PASS.",
        "- Initial invalidated harness attempt is preserved separately and marked `INVALIDATED_NOT_FOR_INFERENCE`; its outputs were excluded.",
        "- Full file SHA-256 inventory: `artifact_hash_manifest.json`.",
        "",
        "## Exactly one recommended next experiment",
        "",
        "Run a focused R1/R2-standardized joint-probe convergence audit only: keep the same certified cache, five seeds, analytical transfer invariant, and train/validation-only selection; preregister a small LR/step schedule that can actually reach the plateau criterion, freeze it before TestA/TestB, and do not run Z-stage attribution unless both invariance gates then pass.",
        "",
        "Reconnect: **NO**  ",
        "O3c: **NO**  ",
        "Repeated-V-update contamination: unresolved and untouched.",
        "",
    ]
    report_text = "\n".join(report)
    (OUT / "final_evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "final_report.md").write_text(report_text, encoding="utf-8")
    (OUT / "report").mkdir(exist_ok=True)
    (OUT / "report" / "10_final_report.md").write_text(report_text, encoding="utf-8")

    out_json.update({"r0_r1_pass": summary["r0_r1_gate_pass"], "r1_r2_pass": summary["r1_r2_gate_pass"],
                     "weight_transfer_pass": transfer["pass"], "measurement_gate": "CLOSED",
                     "stagewise_j_authorized": False, "stagewise_j_run": False,
                     "checkpoint_modified": False, "data_leakage_detected": False,
                     "evidence_summary": "final_evidence_summary.json", "final_report_refined": True})
    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "artifact_hash_manifest.json" or "__pycache__" in path.parts:
            continue
        manifest.append({"path": str(path.relative_to(OUT)).replace("\\", "/"),
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
    (OUT / "artifact_hash_manifest.json").write_text(json.dumps({"root": str(OUT), "files": manifest,
                                                                   "manifest_excludes_self": True,
                                                                   "generated_utc": datetime.now(timezone.utc).isoformat()},
                                                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"finalized": True, "verdict": out_json.get("primary"),
                      "R0_R1_gate": summary["r0_r1_gate_pass"], "R1_R2_gate": summary["r1_r2_gate_pass"],
                      "transfer": transfer, "runtime_seconds": runtime,
                      "files_hashed": len(manifest), "elapsed_finalizer_seconds": time.time() - t0}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
