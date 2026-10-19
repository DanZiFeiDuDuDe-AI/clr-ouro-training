# O3b-M — Operational Same-Key Addressing Rescue

## Verdict

**O3b-M-F — SAME-KEY CLASSIFICATION PASSES BUT ACTION-LEVEL ADDRESSING FAILS**

Candidate B passes the balanced offline pair gate by a wide margin, but the same frozen threshold produces excessive false UPDATEs when each query is compared against a growing slot set. The false slot actions were P3 (different entity, same relation), not P2. The reconnect gate therefore remains closed.

## Scope, provenance, and leakage audit

J/K/L/J3 is archived as **“ARCHIVED — ATTRIBUTION UNRESOLVED, NOT REQUIRED FOR CURRENT OPERATIONAL ADDRESS RESCUE.”** No attribution analysis was resumed. The J3C provenance-certified cache was reused; no backbone forward, model/checkpoint edit, read/value path change, allocator change, or O3c run occurred.

Frozen input carriers:

- Entity side: `H_mention_fresh_{split}.npy`, exact entity-mention mean hidden state, 2048D, float32, frozen, not normalized in cache.
- Relation side: `H_base_I0_{split}.npy`, canonical-row-joined frozen MR `fact_H`, 2048D, float32, not normalized in cache. This is the selected relation-side carrier, not a claim that it is relation-exclusive.
- Candidate B applies train-only per-dimension standardization and learns separate 64D entity/relation key parts. Its pair head receives symmetric absolute-difference and Hadamard features.

Pairs were generated only within their fixed source split, balanced P1/P2/P3/P4. The train/val/TestA/TestB intersection was zero for canonical fact IDs, feature-row IDs, and stable source-record hashes. Test labels were not used for candidate, checkpoint, or threshold selection. TestB closed-world joint class IDs were not used; operational equality used the preserved entity/relation semantic identities.

One important coverage limitation: the selected source cache had **zero P1 pairs with different `object_value`** in every split. Thus this experiment tests same-key routing on repeated-key facts, but does not test replacement of a changed value. That further argues against claiming full UPDATE/write closure.

## Candidate and offline results

Candidate selection was frozen from validation means across five seeds; Candidate B won (operational score `0.99967 ± 0.00023`). The deployed checkpoint is the pre-registered median-validation-score seed 43; no ensemble was used. A and C validation scores were 0.71245 and 0.70562.

| Candidate B, seed 43 | TestA | TestB |
|---|---:|---:|
| Same-key F1 | 99.871% | 99.777% |
| Precision | 99.743% | 99.733% |
| UPDATE recall / P1 accuracy | 100.000% | 99.822% |
| P2 accuracy (same entity, different relation) | 99.828% | 99.911% |
| P3 accuracy (different entity, same relation) | 99.914% | 99.822% |
| P4 accuracy | 100.000% | 100.000% |
| Offline NEW false-UPDATE rate | 0.086% | 0.089% |

Across all five Candidate B seeds, same-key F1 was `99.931% ± 0.058%` on TestA and `99.795% ± 0.093%` on TestB. Offline gate: **PASS**. Candidate A and C failed the offline gate (their five-seed mean F1s were about 62.1%/62.3% and 58.4%/58.5% on TestA/TestB, respectively).

The frozen validation-selected threshold was 0.57. It was selected on pairwise validation examples, not TestA/TestB.

## Action-level slot simulation

The selected Candidate B checkpoint and threshold were then evaluated in source order with an initially empty, unbounded slot list. Address vectors remain attached to their original slot on UPDATE, matching the existing “keep K, replace V” behavior.

| Metric | TestA | TestB | Gate |
|---|---:|---:|---:|
| Action accuracy | 92.45% | 90.39% | diagnostic |
| UPDATE target-slot recall | 100.00% | 99.87% | ≥95% PASS |
| NEW collision / false UPDATE | 11.41% | 14.31% | ≤5% FAIL |
| Duplicate same-key slot creation | 0.00% | 0.13% | ≤5% PASS |
| P2 wrong-UPDATE rate | 0.00% | 0.00% | ≤5% PASS |
| P3 wrong-UPDATE rate | 11.44% | 14.35% | diagnostic; dominant error |

Action gate: **FAIL**. Pairwise accuracy did not transfer to max-over-growing-memory decisions: a small per-pair false-positive tail accumulated over many candidate slots. The observed false UPDATEs targeted P3 distractors (same relation, different entity). This is an operational entity-confusion / threshold-calibration-under-slot-competition failure, not evidence for an attribution mechanism.

## Reconnect and regression

- Real O3b reconnect: **NOT RUN**, correctly locked by the failed action gate.
- Reconnect retrieval and full KV answer: **N/A — not evaluated**.
- Regression vs pre-reconnect baseline: **N/A — no reconnect occurred**.
- Existing model/checkpoint: **not modified**.
- Data leakage: **none detected** across source splits.
- O3c: **NO**.

## Required answers

1. **Best minimal head:** Candidate B, factorized key plus symmetric pair MLP.
2. **Same-key F1:** selected seed 43, TestA 99.871%, TestB 99.777% (five-seed means 99.931% and 99.795%).
3. **UPDATE recall:** TestA 100.000%, TestB 99.822% offline; action target recall 100.000% and 99.871%.
4. **P2 accuracy:** TestA 99.828%, TestB 99.911% offline; zero P2 wrong UPDATEs in the simulator.
5. **NEW false UPDATE/collision:** offline pair FPR 0.086% / 0.089%; slot simulation collision 11.41% / 14.31% (fails).
6. **P3:** pair accuracy 99.914% / 99.822%, but P3 wrong UPDATE rate under slot competition 11.44% / 14.35%.
7. **Main error:** P3 entity confusion amplified by threshold calibration under many-slot competition; not target-slot selection.
8. **Offline same-key gate:** PASS.
9. **Action simulation gate:** FAIL.
10. **Real reconnect:** No.
11. **Post-reconnect retrieval:** N/A.
12. **Post-reconnect full KV answer:** N/A.
13. **Regression >1pp:** not tested; no reconnect.
14. **O3b operational Address:** Not PASS.
15. **Reconnect retained:** YES — no reconnect was made.
16. **Ready for next CLR mainline phase:** No; the action gate failed and changed-value UPDATE coverage is absent.
17. **O3c allowed:** NO.

## Exactly one recommended next experiment

**O3b-M2: validation-calibrated Candidate B slot-competition diagnostic.** Keep Candidate B’s architecture frozen; calibrate one threshold using only validation episodes with the same growing-slot/max-over-slots action rule, include repeated keys with conflicting values, then evaluate once on a fresh untouched source-separated holdout. Do not reconnect unless the unchanged action gate passes.

## Runtime and artifacts

Remote output: `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m_operational_samekey_rescue/`  
Local verified mirror: `H:\LoopLM\clr_v24_o3b_m_operational_samekey_rescue\`

Runtime was 114.75 seconds on remote CUDA. Peak VRAM was not recorded. The original experiment stdout/stderr logs, per-seed learning curves, checkpoints, pair CSVs, configuration, input integrity, gate JSONs, and SHA-256 manifest are preserved alongside this report.
