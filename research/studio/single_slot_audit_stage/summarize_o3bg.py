import json
from pathlib import Path

p = Path("o3bg_final")
vr = json.loads((p / "variant_results.json").read_text())
base = json.loads((p / "baseline_reproduction.json").read_text())
probes = json.loads((p / "probe_results.json").read_text())
retr = json.loads((p / "retrieval_error_decomposition.json").read_text())
margin = json.loads((p / "margin_analysis.json").read_text())
geometry = json.loads((p / "binding_geometry.json").read_text())
gate = json.loads((p / "gate_evaluation.json").read_text())
cfg = json.loads((p / "config.json").read_text())


def pc(m, n):
    return m.get("per_category_accuracy", {}).get(n, {}).get("accuracy")


def fmt(x):
    return "NA" if x is None else f"{100*x:.3f}%"


print("BASELINE")
for s in ("val", "testA", "testB"):
    m = base[s]
    print(s, "F1", fmt(m["f1"]), "recall", fmt(m["recall"]), "P1", fmt(pc(m, "P1_same_key")), "P2", fmt(pc(m, "P2_same_entity_diff_relation")), "AUROC", fmt(m["auroc"]), "PRAUC", fmt(m["pr_auc"]))

print("VARIANTS")
for n, v in vr.items():
    print(n, "params", v["params"], "best_val", fmt(v["best_val"]["f1"]), "A", fmt(v["testA"]["f1"]), fmt(v["testA"]["recall"]), fmt(pc(v["testA"], "P1_same_key")), fmt(pc(v["testA"], "P2_same_entity_diff_relation")), fmt(v["testA"]["auroc"]), fmt(v["testA"]["pr_auc"]), "B", fmt(v["testB"]["f1"]), fmt(v["testB"]["recall"]), fmt(pc(v["testB"], "P1_same_key")), fmt(pc(v["testB"], "P2_same_entity_diff_relation")), fmt(v["testB"]["auroc"]), fmt(v["testB"]["pr_auc"]), "same", fmt(v["same_entity_relation"]["testB"]["accuracy"]), "fixed", fmt(v["entity_fixed_relation"]["testB"]["accuracy"]), "wall", f"{v['wall_clock_s']:.2f}s")

for name in ("G3_projected_tensor", "G2_residual_l2", "G1_hadamard", "G4_learned_concat"):
    print("PROBES", name)
    x = probes[name]["representations"]
    for rep in ("E", "R", "K"):
        for target in ("entity", "relation", "joint"):
            q = x[rep][target]
            print(rep, target, "linA", fmt(q["linear"]["testA"]["accuracy"]), "mlpA", fmt(q["mlp"]["testA"]["accuracy"]), "linB", fmt(q["linear"]["testB"]["accuracy"]), "mlpB", fmt(q["mlp"]["testB"]["accuracy"]))
    print("factorized", probes[name]["factorized_reconstruction"])

print("RETRIEVAL")
for name in vr:
    for s in ("testA", "testB"):
        r = retr[name][s]
        print(name, s, "accuracy", fmt(r["accuracy"]), "n", r["n"], "errors", r["error_counts"], "fractions", r["error_fractions"])

print("MARGINS")
for name in ("G3_projected_tensor", "G2_residual_l2", "G1_hadamard", "G4_learned_concat"):
    print(name, margin[name]["testB"])

print("GEOMETRY_OVERLAP")
for name in vr:
    print(name, "A", geometry[name]["testA"].get("overlap_P1_P2"), "B", geometry[name]["testB"].get("overlap_P1_P2"))

print("GATE", gate)
print("CONFIG", cfg)
