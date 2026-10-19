# CLR v2.4 — O3b-G: Learned Binding-Aware Key

**O3b-G FAIL: LEARNED BINDING INDUCTIVE BIAS INSUFFICIENT**

## Executive Summary

G0 reproduced the canonical O3b-F K–K baseline exactly within the expected tolerance: TestA/TestB F1 = 88.44%/86.56%. The validation-selected best learned binding variant was **G3_projected_tensor**, with TestB F1 88.00%; the highest TestB F1 was **G2_residual_l2**, 88.26%. No learned variant reached the preregistered addressing gate. No oracle labels were used at inference, no real write/read reconnect was run, and O3c was not entered.

The result is a failure of the proposed learned binding intervention, not evidence that explicit binding is impossible: the reused explicit concat/Hadamard/Kronecker controls remain essentially perfect, while learned variants remain below the K–K gate.

## Research Question and Boundaries

This run asked whether a small learned composition head can turn frozen hidden-derived keys into a stable `(entity, relation)` address. It stayed entirely at offline representation/equality/retrieval level. Backbone/Ouro, V/ValueEncoder, GRU, allocator, memory write/update, slot logic, downstream answer head, and the original O2.1b KeyEncoder were frozen.

Auxiliary entity/relation labels were used only during training of the diagnostic binding head and its train-only classifiers. The inference path receives only the frozen hidden-derived key; it does not receive entity ID, relation ID, oracle routing, or answer supervision.

## Protocol Reproduction

Protocol: exact O3b-D/E/F K–K pair protocol, `q=K_a`, `k=K_b`; validation-only threshold selection. Baseline TestA: F1 88.44%; recall 88.73%; P1 88.73%; P2 54.80%; AUROC 95.69%; PR-AUC 95.78%. Baseline TestB: F1 86.56%; recall 85.87%; P1 85.87%; P2 53.60%; AUROC 94.94%; PR-AUC 94.94%.

Baseline reproduction status: **True**. No protocol-impacting bug was found in the G0 reproduction.

## Dataset and Training

Pair counts: train exact 6000, train hard-negative resample 6000, validation/TestA/TestB each 3000/3000/3000. The training hard-negative mix was P2=25% and P3=25% of the negative pool; P1 positives remained balanced. Optimizer was AdamW, lr=3e-4, weight decay=0.01, 3000 steps per variant, single seed 42.

Loss: `L_pair + 0.25 L_entity + 0.25 L_relation + 0.5 L_hard`. The hard term emphasized same-entity/different-relation and same-relation/different-entity negatives. Cache was reused from `D:\training\checkpoints\clr_v24_o21a_mr\cache`; no Ouro recomputation was done in training.

## Binding Variants and Main Results

| Variant | Added params | Val F1 | TestA F1 | TestB F1 | TestA recall | TestB recall | TestA P2 | TestB P2 | TestB AUROC | TestB PR-AUC | same-E relation B | entity-fixed B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1_hadamard | 26,204 | 89.46% | 89.32% | 87.04% | 90.07% | 86.87% | 88.00% | 90.00% | 93.51% | 94.38% | 62.71% | 62.60% |
| G2_residual_l0p5 | 30,428 | 87.57% | 86.52% | 83.99% | 86.47% | 83.27% | 50.80% | 52.00% | 92.67% | 92.85% | 54.12% | 52.06% |
| G2_residual_l1 | 30,428 | 89.30% | 88.45% | 85.31% | 88.87% | 87.87% | 89.60% | 84.00% | 93.00% | 93.52% | 63.79% | 62.60% |
| G2_residual_l2 | 30,428 | 89.42% | 88.25% | 88.26% | 90.60% | 91.20% | 84.40% | 82.00% | 94.65% | 95.08% | 63.22% | 62.65% |
| G3_projected_tensor | 28,796 | 91.62% | 90.76% | 88.00% | 90.33% | 86.27% | 90.80% | 90.00% | 94.90% | 95.23% | 64.04% | 63.43% |
| G4_learned_concat | 28,252 | 88.07% | 87.39% | 81.92% | 88.27% | 83.53% | 80.80% | 80.80% | 89.42% | 90.25% | 60.65% | 59.62% |

Best by validation: **G3_projected_tensor** (28,796 added parameters; 871.76s variant wall-clock). Best TestB diagnostic: **G2_residual_l2** (30,428 added parameters; TestB F1 88.26%).

## Probes

### G3_projected_tensor

| Representation | Entity linear A | Relation linear A/B | Joint linear A | Joint MLP A | Entity MLP A | Relation MLP A/B |
|---|---:|---:|---:|---:|---:|---:|
| E | 1.97% | 73.71%/72.64% | 0.56% | 2.53% | 6.09% | 87.74%/88.25% |
| R | 0.94% | 91.04%/91.34% | 0.56% | 1.46% | 3.17% | 95.41%/94.77% |
| K | 4.93% | 42.58%/46.14% | 7.20% | 3.56% | 1.67% | 49.36%/50.69% |
Factorized reconstruction TestA: joint 1.84%, entity 1.97%, relation 91.04%.

### G2_residual_l2

| Representation | Entity linear A | Relation linear A/B | Joint linear A | Joint MLP A | Entity MLP A | Relation MLP A/B |
|---|---:|---:|---:|---:|---:|---:|
| E | 6.05% | 74.10%/73.80% | 2.02% | 2.19% | 6.43% | 88.38%/88.16% |
| R | 2.19% | 94.17%/93.57% | 1.20% | 0.73% | 1.54% | 95.50%/94.81% |
| K | 3.47% | 47.21%/49.91% | 5.66% | 2.49% | 1.80% | 60.51%/61.62% |
Factorized reconstruction TestA: joint 5.83%, entity 6.05%, relation 94.17%.

Interpretation: the relation branch became decodable (especially in G2/G3), but the joint address probe did not improve over O3b-F. In the validation-selected G3, K→joint linear was 7.20% and shallow MLP 3.56% on TestA, versus the O3b-F reference of approximately 9.39% and 5.62%. E→entity and R→entity remained near chance, so the intended entity branch did not acquire a reliable entity code.

## Same-Entity Relation Retrieval

For the validation-selected G3_projected_tensor, TestB same-entity relation retrieval was 64.04% and entity-fixed relation retrieval was 63.43%; both are far below the 90% diagnostic requirement. The best TestB variant G2_residual_l2 reached only 63.22%/62.65% on those two measures.

## Geometry and Margin Diagnostics

The pairwise geometry is the relevant failure signal. For G3 TestB, P1/P2 histogram overlap was 21.53% (O3b-F cosine reference was about 44.5%); P2 mean score was -0.021 versus P1 mean 0.611. G2 residual λ=2 had TestB overlap 23.33%, while G4 learned concat was worse at 32.60%.

The stored K–K episode retrieval audit reported 100% top-1 accuracy for every learned variant and therefore produced no E+/R− error cases. This audit is degenerate for this K–K candidate grouping and is not used as evidence for the gate; the canonical pair-level P2, F1, recall, and same-entity relation diagnostics remain the decisive measures. The stored margin distributions are retained in `margin_analysis.json`; for G3 TestB the all-negative margin median was 0.811, and the same-entity/wrong-relation margin median was 0.786 under that degenerate episode audit.

### Supplementary established Q→K audit

To make the requested error categories non-degenerate, a separate offline audit reused the established O3b-F query-to-key protocol with the frozen QueryEncoder and each learned binding head. This is supplementary only and is not used for O3b-G gate selection. TestB results:

| Variant | Q→K top-1 accuracy | E+/R− | E−/R+ | E−/R− | Error sample n |
|---|---:|---:|---:|---:|---:|
| G1_hadamard | 91.37% | 44.35% | 20.00% | 35.65% | 115 |
| G2_residual_l0p5 | 97.97% | 66.67% | 14.81% | 18.52% | 27 |
| G2_residual_l1 | 90.92% | 43.80% | 19.01% | 37.19% | 121 |
| G2_residual_l2 | 89.64% | 47.83% | 18.84% | 33.33% | 138 |
| G3_projected_tensor | 95.65% | 51.72% | 25.86% | 22.41% | 58 |
| G4_learned_concat | 81.46% | 51.82% | 18.22% | 29.96% | 247 |

For validation-selected G3_projected_tensor, Q→K TestB accuracy was 95.65%; its error cases were classified as E+/R−=51.72%, E−/R+=25.86%, E−/R−=22.41%. Only variants with at least 100 errors can satisfy the requested 100-case sampling condition; the best-by-validation variant had fewer than 100 errors, so all available errors were retained instead.

## Explicit Binding Controls

| Representation | TestA F1 | TestB F1 | TestB P2 |
|---|---:|---:|---:|
| Explicit concat | 100.00% | 100.00% | 100.00% |
| Explicit hadamard | 100.00% | 99.87% | 100.00% |
| Explicit kronecker | 99.93% | 100.00% | 100.00% |
| Frozen learned K baseline | 88.44% | 86.56% | 53.60% |
| Learned G3_projected_tensor | 90.76% | 88.00% | 90.00% |

Explicit binding is therefore substantially stronger than every learned factorized/compositional head tested here. It is an upper-bound control using structured identity codes, not a reconnect candidate.

## Error Decomposition

Under the stored K–K episode retrieval audit all variants had no top-1 errors, so E+/R−, E−/R+, and E−/R− proportions are empty/0 by construction rather than evidence of solved addressing. The pair-level hard-negative results are the valid error decomposition for this run: same-entity/different-relation P2 remains only 82–90% on TestB across variants, while no variant reached the required 95%.

## Gate Evaluation

{"baseline_reproduced": true, "selection": "best variant selected by validation F1; TestB not used for selection", "best_variant": "G3_projected_tensor", "best_testB_variant_diagnostic": "G2_residual_l2", "testB_f1_ge_95": false, "testB_recall_ge_95": false, "testB_P2_ge_95": false, "same_entity_relation_ge_90": false, "entity_fixed_relation_ge_90": false, "joint_linear_improved": false, "joint_mlp_improved": false, "eplus_rminus_error_fraction": 0.0, "verdict": "O3b-G FAIL: LEARNED BINDING INDUCTIVE BIAS INSUFFICIENT", "reconnect": false, "o3c": false, "repeated_v_update_contamination": "independent unresolved issue", "partial_binding_recovery": false, "secondary_diagnosis": null}

Gate status: **FAIL**. TestB F1 maximum was 88.26%, below 95%; TestB UPDATE recall maximum was 91.20%, below 95%; TestB P2 maximum was 90.00%, below 95%; same-entity and entity-fixed relation retrieval remained below 90% for the best variants.

## Compute and Reproducibility

Git SHA: `NO_GIT_REPO` (workspace is not a Git repository). Seed: 42. Device: cuda. Cache reused: yes. Pair sizes: train/val/TestA/TestB = 6000/3000/3000/3000. Total wall-clock: 6164.92s. Peak VRAM: allocated 0.146 GB, reserved 0.195 GB. Limitation: one seed.

## Final Verdict

**O3b-G FAIL: LEARNED BINDING INDUCTIVE BIAS INSUFFICIENT**

Primary diagnosis: the small learned composition heads did not produce a recoverable, stable joint `(entity, relation)` geometry. Relation information was present in R/E probes, but entity identity was not reliably established in the intended E branch and joint probes remained weak; P2 and same-entity retrieval stayed below gate. This is consistent with persistent binding failure, with the previously observed metric/geometry incompatibility still present as a secondary factor. Explicit binding controls remain near-perfect, so the failure is not a fundamental limit of the offline task.

Answer to the central question: adding the tested learned binding inductive biases did **not** make KeyEncoder produce a stable, retrievable, TestB-generalizing associative address. Among learned variants, G3 projected tensor was best by validation and G2 residual λ=2 had the best TestB F1, but neither was close enough to permit reconnect. Learned Hadamard improved P2 separation, while projected tensor gave the best validation F1; learned concat was not competitive. No learned structure recovered the explicit-binding advantage.

Reconnect allowed? **No.** O3c allowed? **No.** Do not automatically design O3b-H.

Repeated V update contamination remains an independent unresolved issue and was not modified or tested here.
