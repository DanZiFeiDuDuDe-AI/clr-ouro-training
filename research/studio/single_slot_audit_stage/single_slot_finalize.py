import json
from pathlib import Path

ROOT = Path(r"D:\training")
OUT = ROOT / "checkpoints" / "single_slot_diff_audit"


def main():
    audit = json.loads((OUT / "audit_results.json").read_text(encoding="utf-8"))
    canonical = json.loads((OUT / "canonical_rerun.json").read_text(encoding="utf-8"))
    audit["canonical_rerun"] = canonical
    audit["final_status"] = {
        "positive_control": "RESTORED",
        "oracle_allocation_hypothesis": "READY TO RETEST",
        "v23_o0_result": "INVALID_FOR_ORACLE_HYPOTHESIS because full trained model was not SS64-equivalent",
        "qualified_for_v23_o1": False,
        "reason_not_qualified": "This audit stops before O1/N>1; a future retest must first use the canonical SS64-equivalent O0 protocol.",
    }
    (OUT / "audit_results.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    fwd = audit["forward_equivalence"]
    cache = audit["cache_diff"]["hidden_equality"]
    ev = audit["eval_crosscheck"]
    fresh = audit["fresh_probe_results"]
    runs = canonical["runs"]
    lines = [
        "# SS-64 vs CLR-v2.3-OA Differential Audit",
        "",
        "No O1, N=2/N=4, query retrieval, A2, architecture experiment, or v2.3 main retraining was run.",
        "",
        "## Final verdict",
        "",
        "- Core arithmetic forward equivalence: **YES**.",
        "- Full trained-model equivalence: **NO**.",
        "- Cache/data equivalence: **YES**.",
        "- Evaluator/checkpoint roundtrip bug: **NOT FOUND**.",
        "- Canonical SS-64 positive control: **RESTORED**.",
        "- Oracle-allocation hypothesis: **READY TO RETEST**, not tested by the invalid v2.3-O0 comparison.",
        "",
        "The prior v2.3-OA O0 FAIL is retained as an artifact, but it is not a valid test of oracle allocation because its trained model was not SS-64-equivalent.",
        "",
        "## Architecture comparison",
        "",
        "| Component | SS-64 | v2.3-O0 | Same? |",
        "|---|---|---|---|",
    ]
    for k, v in audit["architecture_diff"].items():
        if isinstance(v, dict) and "SS64" in v:
            lines.append(f"| {k} | {v.get('SS64')} | {v.get('v23_O0')} | {v.get('same')} |")
    lines += [
        "",
        "## Forward equivalence",
        "",
        f"Shared-parameter core: {fwd['forward_equivalent_core']}; full trained model: {fwd['forward_equivalent_full_trained_model']}.",
        "The first arithmetic path does not diverge: ContentRead, GRUCell candidate, and slot0 match within fp tolerance.",
        "The trained models diverge at initial state/readout semantics: SS-64 uses fresh zero hidden plus Linear(64,64); v2.3 uses learned init_s[4,64] plus Linear(256,64).",
        "",
        "## Cache and split audit",
        "",
        "The Closure and v2.3 runs use the same LS2 cache and cap-1 rows. Hidden arrays match exactly:",
    ]
    for s, v in cache.items():
        lines.append(f"- {s}: max_abs={v['max_abs_diff']:.3g}, cosine={v['mean_cosine']:.9f}, allclose={v['allclose_1e-6']}")
    lines += [
        "- Cache: 10,000/2,000/2,000/2,000 episodes; cap-1 closure subsets are 2,000/400/400/400.",
        "- TestA query persons are seen in train; TestB query persons are unseen under the metadata audit.",
        "",
        "## Training protocol differences",
        "",
        "SS-64: AdamW lr=1e-3, weight_decay=0, batch=128, max 100 epochs, patience 10, optimization seed 42, no clipping.",
        "v2.3-O0: AdamW lr=3e-4, weight_decay=0.01, batch=64, 2,000 steps, no early stopping, seed 2300, grad clip 1.0.",
        "These are material differences in addition to the architecture mismatch.",
        "",
        "## Unified evaluation",
        "",
        f"- SS-64 checkpoint: train {ev['ss64']['train']['acc']:.3f}, val {ev['ss64']['val']['acc']:.3f}, TestA {ev['ss64']['testA']['acc']:.3f}, TestB {ev['ss64']['testB']['acc']:.3f}.",
        f"- v2.3-O0 on cache: train {ev['v23_o0']['train']['acc']:.3f}, val {ev['v23_o0']['val']['acc']:.3f}, TestA {ev['v23_o0']['testA']['acc']:.3f}, TestB {ev['v23_o0']['testB']['acc']:.3f}.",
        f"- v2.3-O0 on Closure H arrays gives the same values, ruling out evaluator/cache mismatch.",
        "- Checkpoint roundtrip logits match; best_o0 and best_model are identical in model/head weights.",
        "",
        "## Fresh slot0 readout control",
        "",
        f"- SS-64 slot0: TestA {fresh['SS64_slot0']['testA_acc']:.3f}, TestB {fresh['SS64_slot0']['testB_acc']:.3f}.",
        f"- v2.3 slot0: TestA {fresh['v23_slot0']['testA_acc']:.3f}, TestB {fresh['v23_slot0']['testB_acc']:.3f}.",
        "The gap remains after fitting a fresh Linear(64,64), so it is not only jointly trained head state; the actual v2.3 representation also differs.",
        "",
        "## Canonical single-slot rerun",
        "",
        "Protocol: same cap-1 cache, same 2,000/400/400/400 split, H→Linear(2048,64)→GRUCell(zero hidden)→Linear(64,64), AdamW lr=1e-3, weight_decay=0, batch=128, patience=10, optimization seed 42.",
    ]
    for r in runs:
        lines.append(f"- init seed {r['init_seed']}: train {r['train_acc']:.3f}, val {r['val_acc']:.3f}, TestA {r['testA_acc']:.3f}, TestB {r['testB_acc']:.3f}, epochs {r['epochs']}.")
    lines += [
        "",
        "Seed 42 exactly reproduces the historical SS-64 84.75%/83.75% result; seeds 43/44 remain above 80% on both tests.",
        "",
        "## Answers",
        "",
        "Q1: Core forward arithmetic is equivalent; the full trained models are not.",
        "Q2: Yes. Same cache, cap-1 rows, hidden values, labels, and split protocol.",
        "Q3: No. Optimizer, learning rate, weight decay, batch size, step/epoch schedule, clipping, seed handling, and early stopping differ.",
        "Q4: No checkpoint serialization or best-model load bug found.",
        "Q5: No evaluator discrepancy found.",
        "Q6: The dominant issue is the v2.3-O0 model/protocol mismatch: learned 4x64 init state and 256-input head versus SS-64 zero hidden and 64-input head; training protocol differences are secondary contributors. Fresh probes show the representation gap persists.",
        "Q7: Yes. The old SS-64 84% result is independently reproduced.",
        f"Q8: Canonical single-slot TestA/TestB at seed 42: {runs[0]['testA_acc']:.3f}/{runs[0]['testB_acc']:.3f}; three-seed range: TestA {min(r['testA_acc'] for r in runs):.3f}-{max(r['testA_acc'] for r in runs):.3f}, TestB {min(r['testB_acc'] for r in runs):.3f}-{max(r['testB_acc'] for r in runs):.3f}.",
        "Q9: READY TO RETEST. Oracle allocation itself remains untested; the prior v2.3-O0 comparison is invalid as a positive-control comparison.",
        "Q10: No qualification to enter v2.3-O1/N=2/N=4 in this task. A future run must first use the canonical SS-64-equivalent O0 protocol and separately re-register its gate.",
        "",
        "## Stop",
        "",
        "Completed the differential audit and canonical control. No further architecture experiment was started.",
    ]
    (OUT / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"positive_control": "RESTORED", "oracle_hypothesis": "READY TO RETEST", "qualified_for_o1": False}, indent=2))


if __name__ == "__main__":
    main()
