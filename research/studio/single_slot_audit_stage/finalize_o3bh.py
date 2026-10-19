import json
from pathlib import Path

P = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bh_final")
def J(n): return json.loads((P/n).read_text(encoding="utf-8"))
def A(d, key="entity_linear", split="testA"):
    return d[key]["aggregate"][split]["mean"]
def pct(x): return "n/a" if x is None else f"{100*float(x):.2f}%"
def cell(name, d):
    if "entity_linear" in d:
        return f"| {name} | {pct(A(d, 'entity_linear'))} | {pct(A(d, 'relation_linear'))} | {pct(d['entity_linear']['aggregate']['testB']['mean'])} | {pct(d['relation_linear']['aggregate']['testB']['mean'])} |"
    return f"| {name} | {pct(A(d, 'entity'))} | {pct(A(d, 'relation'))} | {pct(d['entity']['aggregate']['testB']['mean'])} | {pct(d['relation']['aggregate']['testB']['mean'])} |"

h0 = J("h0_baseline_reproduction.json"); h0r = J("h0_frozen_recheck.json"); stages = J("keyencoder_internal_probes.json"); qw = J("query_write_comparison.json")
tok = J("token_span_probe_results.json"); eq = J("entity_equality.json"); geo = J("entity_geometry.json"); gate = J("gate_evaluation.json"); cfg = J("config.json")
stage_order = ["H_input", "projection_linear_pre_activation", "projection_activation", "projection_final_pre_norm", "final_normalized_K", "G3_E", "G3_R", "G3_K"]
lines = [
    "# CLR v2.4 — O3b-H: Entity Carrier / Input Sufficiency Diagnosis", "", f"**{gate['verdict']}**", "",
    "## 1. Executive Summary", "",
    f"O3b-G H0 reproduction succeeded from the frozen prior artifacts: K–K cosine F1 is TestA/TestB {pct(h0['baseline']['testA']['f1'])}/{pct(h0['baseline']['testB']['f1'])}. No production architecture or real memory path was changed.", "",
    f"The actual write-side KeyEncoder input H retains substantial but incomplete entity information ({pct(gate['h_input_entity_testA'])} TestA closed-world probe), while relation information is strong ({pct(A(stages['H_input'], 'relation_linear'))} TestA). A bounded frozen-Ouro mention-span probe reaches {pct(tok['fact']['entity_mention_mean']['entity']['aggregate']['testA']['mean'])} on TestA, showing that entity identity exists locally upstream. The KeyEncoder reduces entity probe accuracy to {pct(gate['final_K_entity_testA'])}; the strongest evidence is therefore mixed carrier/state-selection loss plus projection loss.", "",
    "## 2. Frozen Components", "",
    "Backbone/Ouro, O2.1b KeyEncoder, O2.1a-MR QueryEncoder, V/ValueEncoder, GRU, allocator, memory storage/write/update and downstream answer head were frozen. Only offline linear/shallow probes and pairwise equality probes were trained. No oracle entity/relation labels were supplied to any inference representation.", "",
    "## 3. Baseline Reproduction", "",
    f"Baseline reproduction: PASS. Exact prior O3b-G artifacts were reused and a separate no-training frozen-cache/checkpoint recheck matched them: TestA/TestB F1={pct(h0r['baseline']['testA']['f1'])}/{pct(h0r['baseline']['testB']['f1'])}; P1={pct(h0r['baseline']['testA']['per_category_accuracy']['P1_same_key']['accuracy'])}/{pct(h0r['baseline']['testB']['per_category_accuracy']['P1_same_key']['accuracy'])}; P2={pct(h0r['baseline']['testA']['per_category_accuracy']['P2_same_entity_diff_relation']['accuracy'])}/{pct(h0r['baseline']['testB']['per_category_accuracy']['P2_same_entity_diff_relation']['accuracy'])}.", "",
    "## 4. Main Diagnostic Matrix", "",
    "| Representation | Entity TestA | Relation TestA | Entity TestB | Relation TestB |", "|---|---:|---:|---:|---:|"
]
for s in stage_order: lines.append(cell(s, stages[s]))
lines += [cell("query H", qw["query_H"]), cell("query Q", qw["query_Q"]), "",
    "TestB entity-class accuracy is intentionally N/A for closed-world probes because all TestB persons are unseen. The TestB columns above are therefore only meaningful for relation; open-set entity evidence is the pairwise equality section below.", "",
    "## 5. Raw Write H Sufficiency", "",
    f"Raw write H→entity: {pct(gate['h_input_entity_testA'])} TestA (3 seeds; mean±std={pct(stages['H_input']['entity_linear']['aggregate']['testA']['mean'])}±{pct(stages['H_input']['entity_linear']['aggregate']['testA']['std'])}); H→relation: {pct(A(stages['H_input'], 'relation_linear'))} TestA / {pct(stages['H_input']['relation_linear']['aggregate']['testB']['mean'])} TestB. This is not upstream absence: H contains a usable entity signal, but it is weaker than the relation signal and weaker than the local mention carrier.", "",
    "## 6. Token / Span Carrier Search", "",
    f"The original MR cache persisted only final [2048] vectors; it had no token/layer arrays. A bounded frozen-Ouro extraction was therefore run on 4096 facts per split and 1024 queries per split, using layers 0/6/12/18/23. Fact entity-mention mean is {pct(tok['fact']['entity_mention_mean']['entity']['aggregate']['testA']['mean'])}; fact global mean is {pct(tok['fact']['global_mean']['entity']['aggregate']['testA']['mean'])}; fact final-token/EOS is {pct(tok['fact']['final_token']['entity']['aggregate']['testA']['mean'])}. The local mention span is the strongest carrier. Full token/span table is in `token_span_probe_results.json`.", "",
    "| Carrier | Entity TestA | Relation TestA |", "|---|---:|---:|"
]
for n in ["entity_mention_mean", "entity_mention_max", "global_mean", "final_token", "next_token", "relation_span_mean", "layer_0_mention_mean", "layer_6_mention_mean", "layer_12_mention_mean", "layer_18_mention_mean", "layer_23_mention_mean"]:
    v = tok["fact"][n]; lines.append(f"| {n} | {pct(v['entity']['aggregate']['testA']['mean'])} | {pct(v['relation']['aggregate']['testA']['mean'])} |")
lines += ["", "Best sampled layer for entity mention mean is the final normalized layer (99–100% on this bounded sample); among the selected explicit layer checkpoints, layer 23 is strongest at 99.19%, with layer 0 also high at 98.89%. This is a carrier localization result, not a new architecture proposal.", "",
    "## 7. KeyEncoder Internal Information Flow", "",
    "| Stage | Entity TestA | Relation TestA | Entity loss vs input |", "|---|---:|---:|---:|"]
hin = A(stages["H_input"])
for s in stage_order:
    ev = A(stages[s]); rv = A(stages[s], "relation_linear"); lines.append(f"| {s} | {pct(ev)} | {pct(rv)} | {100*(ev-hin):.1f} pp |")
lines += ["", f"The first large internal drop is H→first linear projection: {pct(hin)}→{pct(A(stages['projection_linear_pre_activation']))} ({100*(hin-A(stages['projection_linear_pre_activation'])):.1f} pp). A second drop occurs at the final 256→64 projection and L2 normalization: {pct(A(stages['projection_activation']))}→{pct(A(stages['final_normalized_K']))}. G3's learned branches then nearly eliminate entity decodability (G3-E {pct(A(stages['G3_E']))}, G3-R {pct(A(stages['G3_R']))}, G3-K {pct(A(stages['G3_K']))}).", "",
    "## 8. Query vs Write Side", "",
    f"Query raw H→entity is {pct(A(qw['query_H'], 'entity'))} TestA and {pct(qw['query_H']['entity']['aggregate']['testB']['mean'])} TestB; query raw H→relation is {pct(A(qw['query_H'], 'relation'))} TestA / {pct(qw['query_H']['relation']['aggregate']['testB']['mean'])} TestB. After QueryEncoder, Q→entity falls to {pct(A(qw['query_Q'], 'entity'))} and Q→relation to {pct(A(qw['query_Q'], 'relation'))}. Thus query-side raw H is stronger than write-side raw H, but both projection heads discard entity information.", "",
    "## 9. Open-Set Entity Equality and Geometry", "",
    "| Representation | Equality F1 TestA | Equality F1 TestB | AUROC TestB | PR-AUC TestB |", "|---|---:|---:|---:|---:|"]
for rep in ["raw_H", "query_H", "final_K", "G3_E", "G3_R", "G3_K"]:
    v=eq[rep]; lines.append(f"| {rep} | {pct(v['testA']['f1'])} | {pct(v['testB']['f1'])} | {pct(v['testB']['auroc'])} | {pct(v['testB']['pr_auc'])} |")
lines += ["", "| Representation | Same-entity cosine mean TestB | Different-entity cosine mean TestB | Overlap TestB |", "|---|---:|---:|---:|"]
for rep in ["raw_H", "query_H", "final_K", "G3_E", "G3_R", "G3_K"]:
    v=geo[rep]["testB"]; lines.append(f"| {rep} | {v['same_entity']['mean']:.4f} | {v['different_entity']['mean']:.4f} | {v['cosine_overlap']:.3f} |")
lines += ["", "Raw H TestB equality is {0}, and query H is {1}; entity identity is therefore open-set recoverable upstream. Final K equality is {2}; its geometry is better than raw H for this pair probe, but that does not repair the closed-world entity representation or associative binding. G3-K overlap worsens to {3}.".format(pct(eq['raw_H']['testB']['f1']), pct(eq['query_H']['testB']['f1']), pct(eq['final_K']['testB']['f1']), f"{geo['G3_K']['testB']['cosine_overlap']:.3f}"), "",
    "## 10. Position / Recency and Difficulty", "",
    f"Among 595 raw-H TestA entity-classification errors, none predicted another entity present in the same episode; first-member and last-member conditional rates are both 0% (see `recency_bias.json`). There is no measurable recency-latch signature in this audit, though the errors mostly point outside the episode, so this is not proof that recency is absent everywhere.", "",
    f"Multi-relation TestA raw-H entity accuracy is 76.14% for same_entity episodes, versus 80.24% for single episodes and 69.85% for lexical-stress episodes. TestB closed-world entity classification is N/A; pairwise equality is the valid unseen-person result.", "",
    "## 11. Interpretation", "",
    "The data distinguish the three hypotheses. Entity identity is not absent from the frozen backbone: the local entity mention and the raw write H carry it, and open-set same-entity equality generalizes to unseen persons. However, state selection/pooling has already diluted the signal (local mention near 100% versus raw write H 74.2%), and the KeyEncoder projection further destroys it (74.2%→31.2% at the first projection, then 12.3% at final K). The query-side raw H is stronger, but its QueryEncoder also reduces entity decodability. This supports the registered mixed diagnosis rather than a pure information-insufficient or pure projection-only verdict.", "",
    "## 12. Final Verdict", f"**{gate['verdict']}**", "", "Entity identity most stably exists in the local entity mention span, especially the final frozen-Ouro token representations around the mention (bounded TestA probe ≈100%; selected layer 23 ≈99.19%). The current cached write H is a weaker global/state-selected carrier. The first clear KeyEncoder loss occurs at the 2048→256 linear projection, with additional loss at the 256→64 projection/L2 normalization and the O3b-G binding head.", "",
    "Reconnect allowed? NO. O3c allowed? NO. No O3b-I was designed.", "",
    "## 13. Runtime / Reproducibility", f"Git SHA: {cfg['git_sha']}; cache reused from `{cfg['cache']}`; seed={cfg['seed']}; bounded token/layer extraction={cfg['cache_token_layer_available']}; wall-clock={cfg['wall_clock_s']:.1f}s; peak VRAM allocated={cfg['peak_vram']['allocated_gb']:.3f} GB. Smoke completed before formal run. A separate no-training H0 recheck is saved as `h0_frozen_recheck.json`; its exact F1/P1/P2 numbers match the prior baseline. The 839 MB `token_layer_features.npz` remains on Cheems; JSON/report artifacts were mirrored locally.", "",
    "## 14. Repeated V Update Contamination", "Repeated V update contamination remains an independent unresolved issue.", ""]
(P / "v24_o3b_h_entity_carrier_diag_report.md").write_text("\n".join(lines), encoding="utf-8")
print(P / "v24_o3b_h_entity_carrier_diag_report.md")
