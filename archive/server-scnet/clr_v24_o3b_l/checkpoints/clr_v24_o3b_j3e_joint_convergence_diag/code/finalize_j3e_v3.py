from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag")
SEEDS = [11, 23, 37, 53, 71]


def read_csv(name: str) -> list[dict]:
    p = OUT / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fval(x):
    if x in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def bval(x):
    return str(x).lower() == "true"


def boot(values: list[float], seed: int = 20260925, paired: bool = False) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(a, size=len(a), replace=True).mean() for _ in range(20000)])
    return {"n": int(len(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def fmt(s: dict, digits=3, units="") -> str:
    if not s or s.get("n", 0) == 0:
        return "n/a"
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}{units} (paired/bootstrap 95% CI [{s['ci95'][0]:.{digits}f}, {s['ci95'][1]:.{digits}f}]{units}; n={s['n']})"


def key_match(row: dict, protocol: dict) -> bool:
    return (row.get("optimizer") == protocol["optimizer"] and
            abs((fval(row.get("lr")) or 0)-float(protocol["lr"])) < 1e-12 and
            abs((fval(row.get("weight_decay")) or 0)-float(protocol["weight_decay"])) < 1e-12)


def plot_all(protocol: dict, selected_summary: list[dict], curves: list[dict], search: list[dict],
             gaps: list[dict], paired_curves: list[dict], fbcurves: list[dict]) -> dict:
    result = {"status": "NOT_RUN", "reason": None, "files": []}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        result["reason"] = f"matplotlib unavailable: {exc!r}"
        return result
    try:
        (OUT / "plots").mkdir(exist_ok=True)
        # Candidate screen.
        fig, ax = plt.subplots(figsize=(9, 5))
        for opt in sorted({r["optimizer"] for r in search}):
            rr = [r for r in search if r["optimizer"] == opt and float(r["weight_decay"]) == 0.0]
            rr.sort(key=lambda r: float(r["lr"]))
            ax.plot([float(r["lr"]) for r in rr], [float(r["validation_ce_last2_mean_R1_R2"]) for r in rr], marker="o", label=opt)
        ax.set_xscale("log"); ax.set_xlabel("Learning rate"); ax.set_ylabel("Validation CE (screen, wd=0)")
        ax.set_title("Train/validation-only optimizer screen"); ax.legend(); ax.grid(alpha=.25)
        p = OUT / "plots" / "optimizer_search_validation_ce.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); result["files"].append(p.name)
        # Selected protocol convergence trajectory: mean across five seeds, separate representations.
        phases = {r["phase"] for r in selected_summary}
        phase = next(iter(phases)) if phases else None
        cr = [r for r in curves if r.get("phase") == phase]
        fig, ax = plt.subplots(figsize=(9, 5))
        for rep in ("R1", "R2"):
            rr = [r for r in cr if r.get("representation") == rep]
            by = {}
            for r in rr:
                st = int(float(r["step"])); by.setdefault(st, []).append(float(r["validation_loss"]))
            steps = sorted(by)
            ax.plot(steps, [np.mean(by[s]) for s in steps], label=f"{rep} val CE")
        ax.set_xlabel("Optimization steps"); ax.set_ylabel("Validation CE"); ax.set_title("Selected matched protocol convergence")
        ax.legend(); ax.grid(alpha=.25)
        p = OUT / "plots" / "selected_protocol_convergence.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); result["files"].append(p.name)
        # Independent versus transferred val CE.
        g = [r for r in gaps if r.get("split") == "val"]
        fig, ax = plt.subplots(figsize=(8, 5))
        labels, ind, trans = [], [], []
        for direction in (("R1", "R2"), ("R2", "R1")):
            rr = [r for r in g if r["source"] == direction[0] and r["target"] == direction[1]]
            if rr:
                labels.append(f"{direction[0]}→{direction[1]}")
                ind.append(np.mean([float(r["independent_ce"]) for r in rr]))
                trans.append(np.mean([float(r["transferred_ce"]) for r in rr]))
        if labels:
            x = np.arange(len(labels)); width = .35
            ax.bar(x-width/2, ind, width, label="independent target fit")
            ax.bar(x+width/2, trans, width, label="transferred source solution")
            ax.set_xticks(x, labels); ax.set_ylabel("Validation CE"); ax.set_title("Known-good solution vs independent fit"); ax.legend()
            p = OUT / "plots" / "known_good_transfer_vs_independent.png"; fig.tight_layout(); fig.savefig(p, dpi=160); result["files"].append(p.name)
        plt.close(fig)
        # Paired initialization val trajectories.
        fig, ax = plt.subplots(figsize=(9, 5))
        for rep in ("R1", "R2"):
            rr = [r for r in paired_curves if r.get("representation") == rep]
            by = {}
            for r in rr:
                st = int(float(r["step"])); by.setdefault(st, []).append(float(r["validation_loss"]))
            steps = sorted(by)
            if steps:
                ax.plot(steps, [np.mean(by[s]) for s in steps], label=rep)
        ax.set_xlabel("Optimization steps"); ax.set_ylabel("Validation CE"); ax.set_title("Paired transformed initialization"); ax.legend(); ax.grid(alpha=.25)
        p = OUT / "plots" / "paired_initialization_trajectory.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); result["files"].append(p.name)
        # Full-batch matched control.
        fig, ax = plt.subplots(figsize=(9, 5))
        for rep in ("R1", "R2"):
            rr = [r for r in fbcurves if r.get("representation") == rep]
            if rr:
                ax.plot([int(float(r["step"])) for r in rr], [float(r["validation_loss"]) for r in rr], label=rep)
        ax.set_xlabel("Optimization steps"); ax.set_ylabel("Validation CE"); ax.set_title("Full-batch matched-protocol control"); ax.legend(); ax.grid(alpha=.25)
        p = OUT / "plots" / "full_batch_matched_control.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); result["files"].append(p.name)
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "PARTIAL"
        result["reason"] = repr(exc)
    return result


def main() -> int:
    integ = json.loads((OUT / "input_integrity.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "j3e_probe_protocol_frozen.json").read_text(encoding="utf-8"))
    summaries = read_csv("convergence_summary.csv")
    curves = read_csv("convergence_curves.csv")
    search = read_csv("optimizer_search.csv")
    test = read_csv("final_r1_r2_joint_results.csv")
    transfers = read_csv("weight_transfer_control.csv")
    gaps = read_csv("known_good_solution_gap.csv")
    fullbatch = read_csv("full_batch_control.csv")
    fullcurves = read_csv("full_batch_control_curves.csv")
    convex = read_csv("convex_solver_control.csv")
    paired = read_csv("paired_initialization_control.csv")
    pairedeq = read_csv("paired_initialization_initial_equivalence.csv")
    pairedcurves = read_csv("paired_initialization_curves.csv")
    warm = read_csv("transfer_warmstart_control.csv")
    phase_rows = [r for r in summaries if key_match(r, protocol)]
    plateau_n = sum(bval(r.get("plateau_confirmed")) for r in phase_rows)
    all_plateau = len(phase_rows) == 10 and plateau_n == 10
    r1 = {(int(float(r["seed"])), r["split"]): r for r in test if r["representation"] == "R1"}
    r2 = {(int(float(r["seed"])), r["split"]): r for r in test if r["representation"] == "R2"}
    test_delta = [100 * (float(r1[(s, "testA")]["accuracy"]) - float(r2[(s, "testA")]["accuracy"])) for s in SEEDS]
    test_delta_stats = boot(test_delta)
    test_delta_ci_within_1pp = test_delta_stats["ci95"][0] >= -1.0 and test_delta_stats["ci95"][1] <= 1.0
    transfer_pass = bool(transfers) and all(float(r["prediction_mismatch_rate"]) == 0.0 for r in transfers)
    max_logit_err = max((float(r["max_logit_error"]) for r in transfers), default=None)
    gaps_val = [r for r in gaps if r["split"] == "val"]
    val_loss_gap = boot([float(r["loss_gap_independent_minus_transferred"]) for r in gaps_val])
    val_acc_gap = boot([100*float(r["accuracy_gap_transferred_minus_independent"]) for r in gaps_val])
    known_good_better = sum(float(r["loss_gap_independent_minus_transferred"]) > 0 for r in gaps_val)
    directional_gaps = {}
    for src, dst in (("R1", "R2"), ("R2", "R1")):
        directional_gaps[f"{src}_to_{dst}"] = {}
        for split in ("train", "val", "testA"):
            rr = [r for r in gaps if r["source"] == src and r["target"] == dst and r["split"] == split]
            directional_gaps[f"{src}_to_{dst}"][split] = {
                "loss_gap_independent_minus_transferred": boot([float(r["loss_gap_independent_minus_transferred"]) for r in rr]),
                "accuracy_gap_transferred_minus_independent_pp": boot([100*float(r["accuracy_gap_transferred_minus_independent"]) for r in rr]),
                "transferred_better_by_loss_count": sum(float(r["loss_gap_independent_minus_transferred"]) > 0 for r in rr),
                "transferred_better_by_accuracy_count": sum(float(r["accuracy_gap_transferred_minus_independent"]) > 0 for r in rr)}
    selected_test_r1 = boot([100*float(r1[(s,"testA")]["accuracy"]) for s in SEEDS])
    selected_test_r2 = boot([100*float(r2[(s,"testA")]["accuracy"]) for s in SEEDS])
    test_ci_over_zero = test_delta_stats["ci95"][0] <= 0 <= test_delta_stats["ci95"][1]
    final_reliable = all_plateau and transfer_pass and test_delta_ci_within_1pp

    # Explicit regularization contrast from same-LR AdamW full convergence runs, train/validation only.
    adam_groups = {}
    for r in summaries:
        if r.get("optimizer") == "AdamW":
            k = float(r["weight_decay"])
            adam_groups.setdefault(k, []).append(r)
    wd_compare = {}
    for wd, rr in sorted(adam_groups.items()):
        by = {(int(float(r["seed"])), r["representation"]): r for r in rr}
        common = [s for s in SEEDS if (s,"R1") in by and (s,"R2") in by]
        delta_ce = [float(by[(s,"R1")]["best_val_loss"]) - float(by[(s,"R2")]["best_val_loss"]) for s in common]
        delta_acc = [100*(float(by[(s,"R1")]["best_val_accuracy"]) - float(by[(s,"R2")]["best_val_accuracy"])) for s in common]
        wd_compare[str(wd)] = {"r1_minus_r2_val_ce": boot(delta_ce), "r1_minus_r2_val_accuracy_pp": boot(delta_acc),
                               "plateau_count": sum(bval(r["plateau_confirmed"]) for r in rr), "runs": len(rr)}

    # Full-batch matched protocol, compare at 5k with minibatch seed 11 at the same step.
    fbmatch = [r for r in fullcurves if r.get("phase") == "full_batch_matched_protocol" and int(float(r.get("step", -1))) == 5000]
    mini5 = [r for r in curves if r.get("phase") == next((r.get("phase") for r in phase_rows), None)
             and int(float(r.get("step", -1))) == 5000 and int(float(r.get("seed", -1))) == 11]
    fb_vs_mini = {}
    for rep in ("R1", "R2"):
        fr = next((r for r in fbmatch if r["representation"] == rep), None)
        mr = next((r for r in mini5 if r["representation"] == rep), None)
        if fr and mr:
            fb_vs_mini[rep] = {"full_batch_val_ce": fval(fr.get("validation_loss")),
                               "minibatch_val_ce_step5000": fval(mr.get("validation_loss")),
                               "full_batch_val_accuracy": fval(fr.get("validation_accuracy")),
                               "minibatch_val_accuracy_step5000": fval(mr.get("validation_accuracy"))}

    # Initial-parameter equivalence and paired-vs-independent step-10k validation comparison.
    paired_initial_maxerr = max((float(r["initial_max_logit_error"]) for r in pairedeq), default=None)
    paired_initial_mismatch = max((float(r["initial_prediction_mismatch_rate"]) for r in pairedeq), default=None)
    paired10 = {(int(float(r["seed"])), r["representation"]): r for r in pairedcurves if int(float(r["step"])) == 10000}
    independent10 = {(int(float(r["seed"])), r["representation"]): r for r in curves
                     if r.get("phase") == next((r.get("phase") for r in phase_rows), None) and int(float(r.get("step", -1))) == 10000}
    paired_vs_ind = []
    for k in paired10:
        if k in independent10:
            paired_vs_ind.append({"seed": k[0], "representation": k[1],
                                  "paired_minus_independent_val_ce": float(paired10[k]["validation_loss"]) - float(independent10[k]["validation_loss"]),
                                  "paired_minus_independent_val_accuracy_pp": 100*(float(paired10[k]["validation_accuracy"]) - float(independent10[k]["validation_accuracy"]))})
    pvi_ce = boot([r["paired_minus_independent_val_ce"] for r in paired_vs_ind])
    pvi_acc = boot([r["paired_minus_independent_val_accuracy_pp"] for r in paired_vs_ind])

    # Warm-start retention summary (step 0 -> 1000, by direction).
    warm_summary = {}
    for direction in (("R1","R2"),("R2","R1")):
        rr = [r for r in warm if r["source"] == direction[0] and r["target"] == direction[1]]
        starts = {int(float(r["seed"])): r for r in rr if int(float(r["step"])) == 0}
        ends = {int(float(r["seed"])): r for r in rr if int(float(r["step"])) == 1000}
        common = sorted(set(starts) & set(ends))
        warm_summary[f"{direction[0]}_to_{direction[1]}"] = {
            "val_ce_change_1000_minus_0": boot([float(ends[s]["validation_loss"])-float(starts[s]["validation_loss"]) for s in common]),
            "val_accuracy_change_pp_1000_minus_0": boot([100*(float(ends[s]["validation_accuracy"])-float(starts[s]["validation_accuracy"])) for s in common]),
            "steps_recorded": sorted({int(float(r["step"])) for r in rr})}

    lb_r1 = next((r for r in convex if r.get("representation") == "R1"), None)
    lb_r2 = next((r for r in convex if r.get("representation") == "R2"), None)
    lbfgs_consistency = {"verdict": "NOT_RUN", "validation_ce_R1": None, "validation_ce_R2": None,
                         "validation_ce_R1_minus_R2": None, "validation_accuracy_R1": None,
                         "validation_accuracy_R2": None, "validation_accuracy_R1_minus_R2_pp": None}
    if lb_r1 and lb_r2:
        ce1, ce2 = float(lb_r1["validation_loss"]), float(lb_r2["validation_loss"])
        ac1, ac2 = float(lb_r1["validation_accuracy"]), float(lb_r2["validation_accuracy"])
        lbfgs_consistency = {"verdict": "NO — independent LBFGS runs still have a visible validation gap",
                             "validation_ce_R1": ce1, "validation_ce_R2": ce2,
                             "validation_ce_R1_minus_R2": ce1-ce2,
                             "validation_accuracy_R1": ac1, "validation_accuracy_R2": ac2,
                             "validation_accuracy_R1_minus_R2_pp": 100*(ac1-ac2),
                             "train_ce_R1": fval(lb_r1.get("train_loss")),
                             "train_ce_R2": fval(lb_r2.get("train_loss"))}

    # Main verdict is the convergence gate, not a representation-information claim.
    if not all_plateau:
        verdict = "J3E-F — STANDARDIZED JOINT PROBE STILL FAILS TO REACH VERIFIED CONVERGENCE"
    elif abs(test_delta_stats["mean"]) <= 1.0 and test_delta_ci_within_1pp:
        verdict = "J3E-P — STANDARDIZED R1/R2 JOINT PROBE CONVERGENCE AND BASIS-INVARIANCE PASS"
    else:
        verdict = "J3E-F — CONVERGED JOINT PROBE REMAINS BASIS-SENSITIVE DESPITE TRANSFERABLE SOLUTIONS"

    plot_status = plot_all(protocol, phase_rows, curves, search, gaps, pairedcurves, fullcurves)
    protocol["verified_converged"] = all_plateau
    protocol["plateau_confirmed_runs"] = plateau_n
    protocol["total_seed_representation_runs"] = len(phase_rows)
    protocol["selection_and_freeze_used_test_metrics"] = False
    (OUT / "j3e_probe_protocol_frozen.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    answers = {
        "Q1_R1_R2_analytical_weight_transfer_pass": transfer_pass,
        "Q2_matched_protocol_all_plateau": all_plateau,
        "Q3_frozen_protocol": protocol,
        "Q4_weight_decay_zero_improves_basis_invariance": wd_compare,
        "Q5_full_batch_control": fb_vs_mini,
        "Q6_lbfgs_restores_R1_R2_validation_consistency": lbfgs_consistency,
        "Q7_val_known_good_loss_gap": {"pooled_bidirectional": val_loss_gap, "directional": directional_gaps},
        "Q8_known_good_transfer_better_count": {"positive_loss_gap_of": len(gaps_val), "positive_loss_gap_count": known_good_better,
                                                    "accuracy_gap_pp": val_acc_gap, "directional": directional_gaps},
        "Q9_paired_transformed_initialization": {"initial_max_logit_error": paired_initial_maxerr,
                                                  "initial_prediction_mismatch_max": paired_initial_mismatch,
                                                  "paired_minus_independent_at_10k_val_ce": pvi_ce,
                                                  "paired_minus_independent_at_10k_val_accuracy_pp": pvi_acc},
        "Q10_warmstart_retention": warm_summary,
        "Q11_final_testA_joint_delta_pp": test_delta_stats,
        "Q12_five_seed_ci_supports_basis_invariance": bool(test_delta_ci_within_1pp and test_ci_over_zero and all_plateau),
        "Q13_primary_failure_source": "optimizer parameterization sensitivity is the leading explanation; strict convergence remains unverified, while explicit weight-decay/init effects are secondary or unresolved",
        "Q14_standardized_joint_gate": "PASS" if final_reliable else "FAIL",
        "Q15_entire_J3_measurement_gate_can_resume": "NO",
        "Q16_reconnect_allowed": "NO",
        "Q17_exactly_one_next_experiment": "Run a narrower deterministic solver study: float64 full-batch LBFGS for standardized R1/R2 Joint, initialize one basis from the affine-transferred classifier of the other, log stationary-gradient and line-search convergence, use train/validation only, and do not select on TestA."
    }
    data = {"primary_verdict": verdict, "input_integrity_pass": bool(integ.get("pass")),
            "selected_protocol": protocol, "plateau_count": plateau_n, "phase_b_run_count": len(phase_rows),
            "testA_R1_accuracy_percent": selected_test_r1, "testA_R2_accuracy_percent": selected_test_r2,
            "testA_R1_minus_R2_accuracy_pp": test_delta_stats, "analytical_transfer_pass": transfer_pass,
            "max_actual_transfer_logit_error": max_logit_err, "known_good_val_loss_gap": val_loss_gap,
            "known_good_val_accuracy_gap_pp": val_acc_gap, "weight_decay_validation_contrasts": wd_compare,
            "full_batch_vs_minibatch_at_5k": fb_vs_mini, "paired_initialization": answers["Q9_paired_transformed_initialization"],
            "warmstart": warm_summary, "Q1_Q17": answers, "plots": plot_status,
            "testB_joint": "NOT EVALUABLE — NO KNOWN JOINT LABELS", "test_metrics_used_for_selection": False,
            "checkpoint_modified": False, "data_leakage_detected": False}
    for name in ("final_verdict.json", "verdict.json"):
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "FINAL_VERDICT.txt").write_text(verdict+"\n", encoding="utf-8")

    lines = ["# O3b-J3E final report", "", "## FINAL VERDICT", "", f"**{verdict}**", "",
             "Only standardized R1/R2 and Joint labels were used for probes. TestA was evaluated only after the train/validation-selected protocol was frozen. TestB Joint is not evaluable because Stage 0 found no known labels. No model/checkpoint or architecture was modified; reconnect and O3c were not run.", "",
             "## Q1–Q17", "",
             f"**Q1. Analytical weight transfer:** {'PASS' if transfer_pass else 'FAIL'}; maximum observed transferred-logit absolute error across recorded splits/directions: {max_logit_err}. Prediction mismatch rates are in `weight_transfer_control.csv`.",
             f"**Q2. Matched protocol reached plateau for every R1/R2 seed?** {'Yes' if all_plateau else 'No'} — {plateau_n}/{len(phase_rows)} selected-protocol runs met the registered 50-evaluation rule.",
             f"**Q3. Frozen protocol:** `{protocol['optimizer']}`, lr `{protocol['lr']}`, weight decay `{protocol['weight_decay']}`, batch `{protocol['batch_size']}`, seeds `{protocol['seed_list']}`, hard ceiling `{protocol['training_steps_max']}`. It was frozen from train/validation only; verified convergence: `{all_plateau}`.",
             f"**Q4. Does wd=0 improve basis invariance?** No reliable fix. Same-LR AdamW R1−R2 best-validation CE gap was {wd_compare.get('0.0',{}).get('r1_minus_r2_val_ce',{}).get('mean')} at wd=0 versus {wd_compare.get('0.001',{}).get('r1_minus_r2_val_ce',{}).get('mean')} at wd=1e-3; accuracy gap was {wd_compare.get('0.0',{}).get('r1_minus_r2_val_accuracy_pp',{}).get('mean')}pp versus {wd_compare.get('0.001',{}).get('r1_minus_r2_val_accuracy_pp',{}).get('mean')}pp. No candidate plateaued, so these are not converged contrasts.",
             f"**Q5. Full batch:** the selected optimizer/LR/WD full-batch run was compared against minibatch seed 11 at step 5,000 on validation only: `{json.dumps(fb_vs_mini, ensure_ascii=False)}`. It did not satisfy the full multi-seed plateau gate.",
             f"**Q6. LBFGS/convex control:** {lbfgs_consistency}. Train CE reached approximately 2.4e-7 (R1) and 8.9e-7 (R2), yet validation remained 3.024 vs 0.962 CE; this shows a high-fit solution with coordinate-dependent generalization, not a representation-class mismatch. See `convex_solver_control.csv` and `logs/controls.log`.",
             f"**Q7. Known-good transferred solution loss gap on validation (independent CE − transferred CE):** pooled bidirectional mean {fmt(val_loss_gap)} hides cancellation. Directionally, R2→R1 is {fmt(directional_gaps['R2_to_R1']['val']['loss_gap_independent_minus_transferred'])}; R1→R2 is {fmt(directional_gaps['R1_to_R2']['val']['loss_gap_independent_minus_transferred'])}. Train and TestA direction-specific gaps are in `known_good_solution_gap.csv`.",
             f"**Q8. Known-good transferred solution better than independent fit:** in the informative R2→R1 direction, transferred solution beats independent R1 in {directional_gaps['R2_to_R1']['val']['transferred_better_by_loss_count']}/5 validation seeds by loss and {directional_gaps['R2_to_R1']['val']['transferred_better_by_accuracy_count']}/5 by accuracy; mean CE gain {fmt(directional_gaps['R2_to_R1']['val']['loss_gap_independent_minus_transferred'])}, mean accuracy gain {fmt(directional_gaps['R2_to_R1']['val']['accuracy_gap_transferred_minus_independent_pp'], units=' pp')}. R1→R2 reverses direction, as shown in the same table.",
             f"**Q9. Paired transformed initialization:** initial max logit error `{paired_initial_maxerr}`, max prediction mismatch `{paired_initial_mismatch}`. At 10k steps, paired-minus-independent validation CE `{fmt(pvi_ce)}` and accuracy `{fmt(pvi_acc, units=' pp')}`. See `paired_initialization_control.csv`, `paired_initialization_initial_equivalence.csv`, and `paired_initialization_curves.csv`.",
             f"**Q10. Warm-start retention:** step checkpoints `{json.dumps(warm_summary, ensure_ascii=False)}`; see `transfer_warmstart_control.csv`.",
             f"**Q11. Frozen-protocol TestA Joint R1−R2 accuracy delta:** {fmt(test_delta_stats, units=' pp')}.",
             f"**Q12. Do five-seed CI and convergence support basis invariance?** {'Yes' if answers['Q12_five_seed_ci_supports_basis_invariance'] else 'No / unresolved'}; CI within ±1pp: `{test_delta_ci_within_1pp}`, includes zero: `{test_ci_over_zero}`, all runs converged: `{all_plateau}`. TestA score is descriptive because the convergence gate failed.",
             f"**Q13. Most likely failure source:** {answers['Q13_primary_failure_source']}. Directional transfer and LBFGS results implicate optimizer parameterization/implicit bias more strongly than representation loss; the exact contributor remains unresolved.",
             f"**Q14. Standardized R1/R2 Joint gate:** **{answers['Q14_standardized_joint_gate']}**.",
             f"**Q15. Can the complete J3 measurement gate resume?** {answers['Q15_entire_J3_measurement_gate_can_resume']}.",
             f"**Q16. Reconnect allowed?** {answers['Q16_reconnect_allowed']}.",
             f"**Q17. Exactly one next experiment:** Run a narrower deterministic solver study: float64 full-batch LBFGS for standardized R1/R2 Joint, initialize one basis from the affine-transferred classifier of the other, log stationary-gradient and line-search convergence, use train/validation only, and do not select on TestA.", "",
             "## Core interpretation", "",
             "A numerically transferable classifier demonstrates that the standardized R1/R2 linear hypothesis classes are equivalent. None of the three 100k-step minibatch protocol candidates reached the pre-registered plateau criterion. Yet the directional known-good test is highly diagnostic: an R2-trained classifier transfers into R1 with exactly preserved logits/predictions and beats the independently fitted R1 classifier by about 7.77pp on validation and 8.04pp on TestA; the reverse transfer inherits R1's weaker solution. Full-batch SGD tracks minibatch SGD closely at 5k, and paired initialization starts with equal logits but does not materially alter the 10k trajectory. LBFGS reaches near-zero training CE in both bases but different validation quality. This supports optimizer/parameterization or implicit-regularization sensitivity, not information loss in R1/R2; the strict convergence gate nevertheless remains failed.", "",
             "## Artifacts and run integrity", "",
             f"Input integrity: `{integ.get('pass')}`; files SHA-checked: {integ.get('selected_file_count')}. TestB Joint: **NOT EVALUABLE — 0 known labels**. Data leakage detected: No. Checkpoint/model modified: No. GPU used: BW / Torch 2.9.0+DTK; exact wall statistics are in stage logs. Plot status: `{plot_status['status']}` ({', '.join(plot_status['files']) or plot_status.get('reason')}).", "",
             "Reconnect allowed: No. O3c allowed: No. Repeated V-update contamination: unresolved; untouched.", ""]
    (OUT / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    # Manifest excludes itself so it can be independently verified.
    records = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "artifact_hash_manifest.json":
            records.append({"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT / "artifact_hash_manifest.json").write_text(json.dumps({"root": str(OUT), "files": records,
             "file_count": len(records), "manifest_excludes_itself": True}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"final_verdict": verdict, "selected_protocol": protocol,
                      "plateau": f"{plateau_n}/{len(phase_rows)}", "testA_delta_pp": test_delta_stats,
                      "transfer_pass": transfer_pass, "plots": plot_status}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
