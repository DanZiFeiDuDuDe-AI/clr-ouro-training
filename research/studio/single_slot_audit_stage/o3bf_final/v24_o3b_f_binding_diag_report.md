# CLR v2.4 — O3b-F: Binding Geometry Diagnosis

## 1. Executive Summary

**Final verdict: O3b-F: MIXED BINDING + METRIC FAILURE**

The frozen learned key contains substantial relation information, but it does not form a reliable `(entity, relation)` address. The exact O3b-D/E K–K metric addendum confirms that native cosine remains weak (TestB F1 86.56%), a learned bilinear metric does not rescue it (best bilinear TestB F1 87.42%, P2 60.0%), and a small interaction matcher improves the diagnostic upper bound only to TestB F1 90.75%, P2 68.0%. Explicit concatenation, Hadamard, and Kronecker binding controls are essentially perfect offline controls. This is evidence for a mixed binding/geometry limitation, not for information absence alone.

No real memory write/read reconnect was performed. O3c was not entered.

## 2. Prior Evidence and Research Question

O3b-D showed raw K cosine F1 88.44% / 86.56% on TestA/TestB, with strong Pair-MLP but weak P2. O3b-E improved relation probes and P1/P2 overlap, but failed its addressing gate. O3b-F asks whether the remaining failure is information insufficiency, binding failure, metric incompatibility, or a mixture.

## 3. Frozen Components and Reuse

Frozen throughout: O2.1b KeyEncoder, O2.1a QueryEncoder, backbone/Ouro, V/ValueEncoder/GRU, allocator, memory storage, slots, downstream answer head, and all real write/update logic. Only offline linear/MLP probes, bilinear matchers, and explicit symbolic controls were evaluated.

Reused code and artifacts:

- `a1_v24_o3b.py`: cache/pool and frozen encoder loading.
- `a1_v24_o3b_diag.py`: O3b-D pair sampling, K bank, metrics and threshold selection.
- `a1_v24_o3b_e.py`: O3b-E factor head, factorized pair protocol and scoring helpers.
- Existing O3b-E `best_model.pt`, O2.1b KeyEncoder checkpoint, and O2.1a-MR cache.
- New implementation: `a1_v24_o3b_f.py`.
- Protocol correction/addendum: `a1_v24_o3b_f_kk_addendum.py`.

## 4. Baseline Reproduction

F0 reproduced the O3b-E best `original / E1_a1_b1` checkpoint exactly within rounding:

| Metric | TestA | TestB |
|---|---:|---:|
| F1 | 90.73% | 88.97% |
| UPDATE recall | 93.00% | 89.00% |
| P1 | 93.00% | 89.00% |
| P2 | 85.20% | 90.80% |
| AUROC | 96.99% | 95.90% |
| PR-AUC | 96.97% | 95.78% |

Baseline reproduction passed. Dataset split, cache, representation checkpoint, pair counts and seeds were preserved.

## 5. Protocol Audit

The first F3 implementation contained a new O3b-F-only issue: `add_query_arrays` created `Q_a/Q_b`, but the matcher consumed `Q_a` with `K_a` rather than `K_b`. Therefore its initial Q–K pair-matcher table is superseded and is not used for the verdict. This did not affect O3b-D, O3b-E, F0, the explicit controls, or the exact K–K addendum.

The canonical F3 metric result is `kk_metric_addendum.json`, which uses the exact O3b-D/E protocol: `q=K_a`, `k=K_b`, identical cache, split, pair seeds and frozen KeyEncoder.

## 6. Joint Identity Probes

The probes use train-person closed-world labels; TestB is unseen-person and is reported as N/A for entity/joint classification rather than being trained or threshold-tuned on TestB.

| Representation/probe | TestA | TestB |
|---|---:|---:|
| K → joint, linear accuracy | 9.39% | N/A |
| K → joint, shallow MLP accuracy | 5.62% | N/A |
| K → entity, linear accuracy | 21.10% | N/A |
| K → relation, linear accuracy | 88.51% | 87.18% |
| K → relation, shallow MLP accuracy | 95.84% | 94.81% |
| Factorized reconstruction, exact joint | 18.65% | N/A |
| Factorized reconstruction, entity component | 21.05% | N/A |
| Factorized reconstruction, relation component | 89.11% | N/A |

The direct joint decoder is much weaker than the relation decoder. Relation is present, but a stable joint address is not directly recoverable at low complexity.

The O3b-E factor-head probes remain consistent with role collapse: `K_e → relation` 59.18% / 59.13%, `K_r → relation` 82.68% / 81.90%, and `K_full → relation` 92.75% / 92.41%; `K_e → entity` was only 2.36% on TestA and `K_full → entity` 4.55%. Thus factorization improved relation readout without creating a reliable entity branch.

## 7. Conditional Relation Probes

| Diagnostic | TestA | TestB |
|---|---:|---:|
| Same-entity leave-one-out relation accuracy | 50.60% | 50.57% |
| Per-entity macro accuracy | 40.48% | 39.31% |
| Same-entity P1/P2 binary F1 | 92.22% | 92.36% |
| Same-entity P1 accuracy | 98.33% | 98.67% |
| Same-entity P2 accuracy | 10.40% | 10.00% |
| Relation retrieval with entity fixed | 56.47% | 56.53% |

The P2 result is the key conditional failure: when the entity is held fixed, the learned geometry still usually cannot distinguish relation equality from relation mismatch.

## 8. Frozen-K Metric Diagnosis — Canonical K–K Protocol

Thresholds were selected on validation only and then fixed. The exact raw K–K baseline is the O3b-D baseline.

| Matcher | TestA F1 | TestB F1 | TestA recall | TestB recall | TestA P2 | TestB P2 | TestB AUROC | TestB PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosine | 88.44% | 86.56% | 88.73% | 85.87% | 54.80% | 53.60% | 94.94% | 94.94% |
| Euclidean | 88.44% | 86.56% | 88.73% | 85.87% | 54.80% | 53.60% | 94.94% | 94.94% |
| Full bilinear | 87.96% | 87.38% | 92.53% | 91.40% | 55.60% | 59.60% | 94.51% | 93.86% |
| Low-rank 16 | 85.12% | 82.23% | 94.40% | 93.33% | 52.00% | 54.00% | 90.37% | 89.47% |
| Low-rank 32 | 86.50% | 83.52% | 93.13% | 91.40% | 56.80% | 61.60% | 91.68% | 90.55% |
| Low-rank 64 | 86.44% | 83.77% | 93.73% | 92.73% | 57.20% | 58.40% | 92.12% | 90.98% |
| Symmetric bilinear | 87.14% | 87.42% | 88.73% | 89.40% | 62.00% | 60.00% | 93.95% | 93.76% |
| Interaction MLP diagnostic upper bound | 92.57% | 90.75% | 93.47% | 90.53% | 68.00% | 68.00% | 97.03% | 97.01% |

The best bilinear metric is below every addressing gate. The interaction MLP provides a 4.18pp TestB F1 gain over cosine and a 14.40pp P2 gain, but remains below 95% F1/recall and 90% P2. This is diagnostic evidence of some recoverable interaction, not a solved key representation.

For completeness, K–K top-1 retrieval inside the fixed four-fact episode was 100% / 100% for raw cosine and Euclidean. This does not contradict the pair F1 failure: top-1 episode retrieval is an easier closed-set task than the balanced P1–P7 equality protocol, and its perfect score is not used as an addressing gate.

Threshold selection remained validation-only. The raw K–K threshold was 0.493089, identical to O3b-D; the TestB raw F1 remained 85.43%–86.54% over representative 0.45–0.55 threshold choices, so the failure is not a TestB threshold artifact.

## 9. Explicit Binding Controls

Stable hash-derived entity/relation codes were created without fitting on TestA/TestB labels. These are symbolic offline controls, not reconnect candidates.

| Control | TestA F1 | TestB F1 | TestA recall | TestB recall |
|---|---:|---:|---:|---:|
| Explicit concatenation | 100.00% | 100.00% | 100.00% | 100.00% |
| Hadamard binding | 100.00% | 99.87% | 100.00% | 99.73% |
| Kronecker binding | 99.93% | 100.00% | 99.87% | 100.00% |

The explicit controls are dramatically above the learned K and demonstrate that a conjunction-preserving code is sufficient for this offline protocol. They do not show that the current backbone can produce such a code.

## 10. Geometry and Collision Diagnostics

Raw learned K P1/P2 histogram overlap was 39.67% on TestA and 43.47% on TestB. The same-entity relation collision mean was 0.7446 / 0.7161 for P1 and 0.4662 / 0.4635 for P2. The explicit controls had zero histogram overlap.

For the selected same-entity four-relation block, learned-K off-diagonal cosine averaged 0.6032 on TestA and 0.3339 on TestB. The O3b-E relation-factor block was more separated, but this did not translate into P2 equality: the factor relation head had low cross-relation similarity while the entity role was not separately represented.

## 11. Retrieval Error Decomposition and Margins

The valid Q–K top-1 episode retrieval diagnostic achieved 99.47% / 99.40% on TestA/TestB (1325/1332 and 1324/1332). Its small error sets were dominated by correct-entity/wrong-relation cases: 6/7 errors on TestA and 7/8 on TestB. K–K raw top-1 retrieval was 100% on both sets, so it has no error cases to decompose.

For the valid Q–K episode retrieval margins, cosine mean/p50/negative-rate were 0.4858/0.4625/0.53% on TestA and 0.4621/0.4428/0.60% on TestB. Same-entity wrong-relation margin mean/negative-rate were 0.2639/1.13% and 0.2684/1.23%. These margins show that the query interface can often select among the four fixed candidates, while the balanced K–K equality protocol exposes the unstable relation binding.

## 12. K_e Role-Collapse Analysis

The O3b-E factor head did not form a clean entity/relation decomposition. `K_e` carried only weak relation information and almost no entity information under the prior probe; `K_r` carried most of the relation signal but also did not provide a reliable entity-conditioned address. Full K retained relation information, but direct joint decoding was weak. The factor-head output norms were normalized and its flattened K_e/K_r correlation was approximately −0.027 on TestA and −0.028 on TestB, so the failure is not explained by simple numerical collinearity; it is a binding/role-allocation problem.

## 13. Information vs Geometry Diagnosis

| Test | Result | Interpretation |
|---|---|---|
| Relation probe | MLP 95.84% / 94.81% | Relation information is present. |
| Joint linear probe | 9.39% TestA | Joint identity is not linearly organized. |
| Joint shallow MLP | 5.62% TestA | A tiny nonlinear decoder does not recover the conjunction. |
| Conditional relation | ≈50.6% | Relation is not reliably usable under entity-fixed competition. |
| Native cosine/euclidean | TestB F1 86.56% | Native K geometry fails the equality gate. |
| Learned bilinear | TestB F1 87.42% | Simple metric change does not rescue addressing. |
| Interaction MLP | TestB F1 90.75%, P2 68% | Some recoverable interaction exists, but not enough for key geometry. |
| Explicit binding | ≈100% | Binding inductive bias is strongly superior offline. |

The evidence is therefore not “information is absent.” It is: relation information is decodable, but entity and relation are not combined into a stable usable address, and flexible metric interaction can only partially recover the missing geometry.

## 14. Gate Evaluation

The canonical gate is not passed:

- TestB equality F1 ≥95%: **false** for best bilinear and interaction matcher.
- TestB UPDATE recall ≥95%: **false** for canonical K–K best bilinear (89.40%) and interaction matcher (90.53%).
- TestB P2 ≥90%: **false** for all learned K matchers; best is 68.00%.
- Explicit controls are high, but they are symbolic controls and cannot convert into a learned-key PASS.

Because individual relation information is strong, direct joint recovery is weak, metric interaction gives a partial gain, and explicit conjunction binding is much better, the final mechanism verdict is:

**O3b-F: MIXED BINDING + METRIC FAILURE**

## 15. Reconnect Readiness and O3 Boundary

Reconnect: **NO**.

O3c: **NO**.

The current offline representation is not ready for a causal memory-pipeline reconnect under the preregistered addressing gate. No O3b-G is designed or started.

## 16. Repeated V Update Contamination

**Repeated V update contamination remains an independent unresolved issue.** O3b-F did not modify V, ValueEncoder, GRU, allocator, memory storage, slot update logic, or answer supervision.

## 17. Run Metadata

- Git SHA: `NO_GIT_REPO`.
- Device: CUDA.
- Cache reused: yes; 11 files, 181,337,971 bytes; cache generation time 0s.
- Pair sets: train 6000, val 3000, TestA 3000, TestB 3000; balanced P1/P2–P7 protocol.
- Formal O3b-F main run: 1512.96s.
- Formal K–K addendum: 1915.01s.
- Peak VRAM main run: 0.145 GB allocated / 0.182 GB reserved.
- Smoke: passed interface validation; no scientific verdict taken from smoke.

## 18. Final Answer

O3b-D’s failure is not best explained by total absence of relation information. O3b-E/F show that relation is partially and even highly decodable, but `(entity, relation)` binding is not organized into a stable address. A flexible interaction matcher partially rescues the frozen representation, yet fails the addressing gate; explicit binding inductive biases are essentially perfect. Therefore the clean diagnosis is **mixed binding plus metric failure**, with binding failure as the dominant limitation.

This does not authorize reconnect or O3c. The result does not address repeated V contamination, real multi-slot writing, or downstream memory behavior.
