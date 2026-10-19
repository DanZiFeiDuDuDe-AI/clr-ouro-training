import json
from pathlib import Path

import numpy as np


ROOT = Path("o3bg_final")
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


vr = load("variant_results.json")
base = load("baseline_reproduction.json")
probes = load("probe_results.json")
retr = load("retrieval_error_decomposition.json")
margin = load("margin_analysis.json")
geometry = load("binding_geometry.json")
explicit = load("explicit_binding_results.json")
cross = load("cross_interface_retrieval.json")
gate = load("gate_evaluation.json")
cfg = load("config.json")
pair_stats = load("pair_dataset_stats.json")


def pct(x):
    return "N/A" if x is None else f"{100*x:.2f}%"


def pc(m, name):
    return m.get("per_category_accuracy", {}).get(name, {}).get("accuracy")


def metric_line(m):
    return f"F1 {pct(m['f1'])}; recall {pct(m['recall'])}; P1 {pct(pc(m, 'P1_same_key'))}; P2 {pct(pc(m, 'P2_same_entity_diff_relation'))}; AUROC {pct(m['auroc'])}; PR-AUC {pct(m['pr_auc'])}"


variants = list(vr)
best_name = gate["best_variant"]
best_test_name = gate["best_testB_variant_diagnostic"]
best = vr[best_name]
best_test = vr[best_test_name]


rows = [
    "# CLR v2.4 — O3b-G: Learned Binding-Aware Key",
    "",
    f"**{gate['verdict']}**",
    "",
    "## Executive Summary",
    "",
    f"G0 reproduced the canonical O3b-F K–K baseline exactly within the expected tolerance: TestA/TestB F1 = {pct(base['testA']['f1'])}/{pct(base['testB']['f1'])}. The validation-selected best learned binding variant was **{best_name}**, with TestB F1 {pct(best['testB']['f1'])}; the highest TestB F1 was **{best_test_name}**, {pct(best_test['testB']['f1'])}. No learned variant reached the preregistered addressing gate. No oracle labels were used at inference, no real write/read reconnect was run, and O3c was not entered.",
    "",
    "The result is a failure of the proposed learned binding intervention, not evidence that explicit binding is impossible: the reused explicit concat/Hadamard/Kronecker controls remain essentially perfect, while learned variants remain below the K–K gate.",
    "",
    "## Research Question and Boundaries",
    "",
    "This run asked whether a small learned composition head can turn frozen hidden-derived keys into a stable `(entity, relation)` address. It stayed entirely at offline representation/equality/retrieval level. Backbone/Ouro, V/ValueEncoder, GRU, allocator, memory write/update, slot logic, downstream answer head, and the original O2.1b KeyEncoder were frozen.",
    "",
    "Auxiliary entity/relation labels were used only during training of the diagnostic binding head and its train-only classifiers. The inference path receives only the frozen hidden-derived key; it does not receive entity ID, relation ID, oracle routing, or answer supervision.",
    "",
    "## Protocol Reproduction",
    "",
    f"Protocol: exact O3b-D/E/F K–K pair protocol, `q=K_a`, `k=K_b`; validation-only threshold selection. Baseline TestA: {metric_line(base['testA'])}. Baseline TestB: {metric_line(base['testB'])}.",
    "",
    f"Baseline reproduction status: **{base.get('testA', {}).get('f1') is not None and gate['baseline_reproduced']}**. No protocol-impacting bug was found in the G0 reproduction.",
    "",
    "## Dataset and Training",
    "",
    f"Pair counts: train exact {pair_stats['train_exact']['n']}, train hard-negative resample {pair_stats['train_hard']['n']}, validation/TestA/TestB each {pair_stats['val_test']['val']['n']}/{pair_stats['val_test']['testA']['n']}/{pair_stats['val_test']['testB']['n']}. The training hard-negative mix was P2=25% and P3=25% of the negative pool; P1 positives remained balanced. Optimizer was AdamW, lr=3e-4, weight decay=0.01, 3000 steps per variant, single seed 42.",
    "",
    "Loss: `L_pair + 0.25 L_entity + 0.25 L_relation + 0.5 L_hard`. The hard term emphasized same-entity/different-relation and same-relation/different-entity negatives. Cache was reused from `D:\\training\\checkpoints\\clr_v24_o21a_mr\\cache`; no Ouro recomputation was done in training.",
    "",
    "## Binding Variants and Main Results",
    "",
    "| Variant | Added params | Val F1 | TestA F1 | TestB F1 | TestA recall | TestB recall | TestA P2 | TestB P2 | TestB AUROC | TestB PR-AUC | same-E relation B | entity-fixed B |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for name in variants:
    r = vr[name]
    rows.append(
        f"| {name} | {r['params']:,} | {pct(r['best_val']['f1'])} | {pct(r['testA']['f1'])} | {pct(r['testB']['f1'])} | {pct(r['testA']['recall'])} | {pct(r['testB']['recall'])} | {pct(pc(r['testA'], 'P2_same_entity_diff_relation'))} | {pct(pc(r['testB'], 'P2_same_entity_diff_relation'))} | {pct(r['testB']['auroc'])} | {pct(r['testB']['pr_auc'])} | {pct(r['same_entity_relation']['testB']['accuracy'])} | {pct(r['entity_fixed_relation']['testB']['accuracy'])} |"
    )
rows += [
    "",
    f"Best by validation: **{best_name}** ({best['params']:,} added parameters; {best['wall_clock_s']:.2f}s variant wall-clock). Best TestB diagnostic: **{best_test_name}** ({best_test['params']:,} added parameters; TestB F1 {pct(best_test['testB']['f1'])}).",
    "",
    "## Probes",
    "",
]
for name in (best_name, best_test_name):
    p = probes[name]["representations"]
    rows.append(f"### {name}")
    rows.append("")
    rows.append("| Representation | Entity linear A | Relation linear A/B | Joint linear A | Joint MLP A | Entity MLP A | Relation MLP A/B |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|")
    for rep in ("E", "R", "K"):
        rows.append(f"| {rep} | {pct(p[rep]['entity']['linear']['testA']['accuracy'])} | {pct(p[rep]['relation']['linear']['testA']['accuracy'])}/{pct(p[rep]['relation']['linear']['testB']['accuracy'])} | {pct(p[rep]['joint']['linear']['testA']['accuracy'])} | {pct(p[rep]['joint']['mlp']['testA']['accuracy'])} | {pct(p[rep]['entity']['mlp']['testA']['accuracy'])} | {pct(p[rep]['relation']['mlp']['testA']['accuracy'])}/{pct(p[rep]['relation']['mlp']['testB']['accuracy'])} |")
    f = probes[name]["factorized_reconstruction"]
    rows.append(f"Factorized reconstruction TestA: joint {pct(f['testA']['accuracy'])}, entity {pct(f['testA']['entity_accuracy'])}, relation {pct(f['testA']['relation_accuracy'])}.")
    rows.append("")
rows += [
    "Interpretation: the relation branch became decodable (especially in G2/G3), but the joint address probe did not improve over O3b-F. In the validation-selected G3, K→joint linear was 7.20% and shallow MLP 3.56% on TestA, versus the O3b-F reference of approximately 9.39% and 5.62%. E→entity and R→entity remained near chance, so the intended entity branch did not acquire a reliable entity code.",
    "",
    "## Same-Entity Relation Retrieval",
    "",
    f"For the validation-selected {best_name}, TestB same-entity relation retrieval was {pct(best['same_entity_relation']['testB']['accuracy'])} and entity-fixed relation retrieval was {pct(best['entity_fixed_relation']['testB']['accuracy'])}; both are far below the 90% diagnostic requirement. The best TestB variant {best_test_name} reached only {pct(best_test['same_entity_relation']['testB']['accuracy'])}/{pct(best_test['entity_fixed_relation']['testB']['accuracy'])} on those two measures.",
    "",
    "## Geometry and Margin Diagnostics",
    "",
    "The pairwise geometry is the relevant failure signal. For G3 TestB, P1/P2 histogram overlap was 21.53% (O3b-F cosine reference was about 44.5%); P2 mean score was -0.021 versus P1 mean 0.611. G2 residual λ=2 had TestB overlap 23.33%, while G4 learned concat was worse at 32.60%.",
    "",
    f"The stored K–K episode retrieval audit reported 100% top-1 accuracy for every learned variant and therefore produced no E+/R− error cases. This audit is degenerate for this K–K candidate grouping and is not used as evidence for the gate; the canonical pair-level P2, F1, recall, and same-entity relation diagnostics remain the decisive measures. The stored margin distributions are retained in `margin_analysis.json`; for G3 TestB the all-negative margin median was {margin[best_name]['testB']['all_negatives']['p50']:.3f}, and the same-entity/wrong-relation margin median was {margin[best_name]['testB']['same_entity_wrong_relation']['p50']:.3f} under that degenerate episode audit.",
    "",
    "### Supplementary established Q→K audit",
    "",
    "To make the requested error categories non-degenerate, a separate offline audit reused the established O3b-F query-to-key protocol with the frozen QueryEncoder and each learned binding head. This is supplementary only and is not used for O3b-G gate selection. TestB results:",
    "",
    "| Variant | Q→K top-1 accuracy | E+/R− | E−/R+ | E−/R− | Error sample n |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in variants:
    c = cross["variants"][name]["testB"]
    ef = c["error_fractions"]
    err_n = sum(c["error_counts"].values())
    rows.append(f"| {name} | {pct(c['accuracy'])} | {pct(ef.get('E+_R-', 0.0))} | {pct(ef.get('E-_R+', 0.0))} | {pct(ef.get('E-_R-', 0.0))} | {err_n} |")
rows += [
    "",
    f"For validation-selected {best_name}, Q→K TestB accuracy was {pct(cross['variants'][best_name]['testB']['accuracy'])}; its error cases were classified as E+/R−={pct(cross['variants'][best_name]['testB']['error_fractions'].get('E+_R-', 0.0))}, E−/R+={pct(cross['variants'][best_name]['testB']['error_fractions'].get('E-_R+', 0.0))}, E−/R−={pct(cross['variants'][best_name]['testB']['error_fractions'].get('E-_R-', 0.0))}. Only variants with at least 100 errors can satisfy the requested 100-case sampling condition; the best-by-validation variant had fewer than 100 errors, so all available errors were retained instead.",
    "",
    "## Explicit Binding Controls",
    "",
    "| Representation | TestA F1 | TestB F1 | TestB P2 |",
    "|---|---:|---:|---:|",
]
for key in ("concat", "hadamard", "kronecker"):
    e = explicit[key]
    rows.append(f"| Explicit {key} | {pct(e['testA']['f1'])} | {pct(e['testB']['f1'])} | {pct(pc(e['testB'], 'P2_same_entity_diff_relation'))} |")
rows += [
    f"| Frozen learned K baseline | {pct(base['testA']['f1'])} | {pct(base['testB']['f1'])} | {pct(pc(base['testB'], 'P2_same_entity_diff_relation'))} |",
    f"| Learned {best_name} | {pct(best['testA']['f1'])} | {pct(best['testB']['f1'])} | {pct(pc(best['testB'], 'P2_same_entity_diff_relation'))} |",
    "",
    "Explicit binding is therefore substantially stronger than every learned factorized/compositional head tested here. It is an upper-bound control using structured identity codes, not a reconnect candidate.",
    "",
    "## Error Decomposition",
    "",
    "Under the stored K–K episode retrieval audit all variants had no top-1 errors, so E+/R−, E−/R+, and E−/R− proportions are empty/0 by construction rather than evidence of solved addressing. The pair-level hard-negative results are the valid error decomposition for this run: same-entity/different-relation P2 remains only 82–90% on TestB across variants, while no variant reached the required 95%.",
    "",
    "## Gate Evaluation",
    "",
    f"{json.dumps(gate, ensure_ascii=False)}",
    "",
    f"Gate status: **FAIL**. TestB F1 maximum was {pct(max(v['testB']['f1'] for v in vr.values()))}, below 95%; TestB UPDATE recall maximum was {pct(max(v['testB']['recall'] for v in vr.values()))}, below 95%; TestB P2 maximum was {pct(max(pc(v['testB'], 'P2_same_entity_diff_relation') for v in vr.values()))}, below 95%; same-entity and entity-fixed relation retrieval remained below 90% for the best variants.",
    "",
    "## Compute and Reproducibility",
    "",
    f"Git SHA: `{cfg['git_sha']}` (workspace is not a Git repository). Seed: {cfg['seed']}. Device: {cfg['device']}. Cache reused: yes. Pair sizes: train/val/TestA/TestB = {cfg['pair_sizes']['train']}/{cfg['pair_sizes']['val']}/{cfg['pair_sizes']['testA']}/{cfg['pair_sizes']['testB']}. Total wall-clock: {cfg['wall_clock_s']:.2f}s. Peak VRAM: allocated {cfg['peak_vram']['allocated_gb']:.3f} GB, reserved {cfg['peak_vram']['reserved_gb']:.3f} GB. Limitation: one seed.",
    "",
    "## Final Verdict",
    "",
    "**O3b-G FAIL: LEARNED BINDING INDUCTIVE BIAS INSUFFICIENT**",
    "",
    "Primary diagnosis: the small learned composition heads did not produce a recoverable, stable joint `(entity, relation)` geometry. Relation information was present in R/E probes, but entity identity was not reliably established in the intended E branch and joint probes remained weak; P2 and same-entity retrieval stayed below gate. This is consistent with persistent binding failure, with the previously observed metric/geometry incompatibility still present as a secondary factor. Explicit binding controls remain near-perfect, so the failure is not a fundamental limit of the offline task.",
    "",
    "Answer to the central question: adding the tested learned binding inductive biases did **not** make KeyEncoder produce a stable, retrievable, TestB-generalizing associative address. Among learned variants, G3 projected tensor was best by validation and G2 residual λ=2 had the best TestB F1, but neither was close enough to permit reconnect. Learned Hadamard improved P2 separation, while projected tensor gave the best validation F1; learned concat was not competitive. No learned structure recovered the explicit-binding advantage.",
    "",
    "Reconnect allowed? **No.** O3c allowed? **No.** Do not automatically design O3b-H.",
    "",
    "Repeated V update contamination remains an independent unresolved issue and was not modified or tested here.",
]
(ROOT / "v24_o3b_g_binding_key_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


# Lightweight SVG plots (standard library only; no plotting package required).
labels = [x.replace("G2_residual_", "G2 ").replace("G3_projected_tensor", "G3 tensor").replace("G4_learned_concat", "G4 concat").replace("G1_hadamard", "G1 Hadamard") for x in variants]


def bar_svg(path, title, series, threshold=None):
    width, height = 1100, 560
    left, top, chart_w, chart_h = 70, 70, 980, 370
    n = len(labels); group = chart_w / n; maxv = 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="35" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>']
    parts.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="black"/>')
    if threshold is not None:
        y = top + chart_h * (1 - threshold / maxv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#d62728" stroke-dasharray="8,5"/><text x="{left+chart_w-5}" y="{y-6:.1f}" text-anchor="end" font-size="13" fill="#d62728">gate {threshold:.0%}</text>')
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for si, (name, vals) in enumerate(series):
        for i, val in enumerate(vals):
            bw = group * .28; x0 = left + i * group + group * .18 + si * group * .30
            y0 = top + chart_h * (1 - float(val) / maxv); h = top + chart_h - y0
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[si]}"/><text x="{x0+bw/2:.1f}" y="{y0-4:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{float(val):.2f}</text>')
        lx = left + si * 150; parts.append(f'<rect x="{lx}" y="{height-50}" width="14" height="14" fill="{colors[si]}"/><text x="{lx+20}" y="{height-38}" font-size="13" font-family="Arial">{name}</text>')
    for i, label in enumerate(labels):
        x0 = left + i * group + group / 2
        parts.append(f'<text x="{x0:.1f}" y="{top+chart_h+25}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(25 {x0:.1f},{top+chart_h+25})">{label}</text>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


bar_svg(PLOTS / "variant_f1_p2.svg", "O3b-G K–K F1", [("TestA", [vr[n]["testA"]["f1"] for n in variants]), ("TestB", [vr[n]["testB"]["f1"] for n in variants])], .95)
bar_svg(PLOTS / "p1_p2_overlap_testB.svg", "TestB P1/P2 overlap", [("overlap", [geometry[n]["testB"].get("overlap_P1_P2", np.nan) for n in variants])], .445)

cm = load("collision_matrices.json").get(best_name, {}).get("testB", {})
if cm.get("available"):
    mat = np.asarray(cm["matrix"], dtype=float); cell = 70; pad = 80; w = pad + cell * len(mat) + 20; h = pad + cell * len(mat) + 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="{w/2}" y="22" text-anchor="middle" font-size="18" font-family="Arial">{best_name} TestB collision block</text>']
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            q = max(0, min(1, (float(val)+1)/2)); color = f"rgb({int(255*q)},80,{int(255*(1-q))})"; parts.append(f'<rect x="{pad+j*cell}" y="{pad+i*cell}" width="{cell}" height="{cell}" fill="{color}" stroke="white"/><text x="{pad+j*cell+cell/2}" y="{pad+i*cell+cell/2+5}" text-anchor="middle" font-size="13" font-family="Arial">{float(val):.2f}</text>')
    for i, rel in enumerate(cm["relations"]):
        parts.append(f'<text x="{pad+i*cell+cell/2}" y="{pad+cell*len(mat)+18}" text-anchor="middle" font-size="11">r{rel}</text><text x="{pad-8}" y="{pad+i*cell+cell/2+4}" text-anchor="end" font-size="11">r{rel}</text>')
    parts.append("</svg>"); (PLOTS / "collision_matrix_best_testB.svg").write_text("".join(parts), encoding="utf-8")

print(json.dumps({"report": str(ROOT / 'v24_o3b_g_binding_key_report.md'), "plots": [str(x) for x in PLOTS.iterdir()]}, ensure_ascii=False, indent=2))
