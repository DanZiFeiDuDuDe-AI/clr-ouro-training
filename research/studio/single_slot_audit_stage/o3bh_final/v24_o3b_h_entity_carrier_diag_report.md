# CLR v2.4 — O3b-H: Entity Carrier / Input Sufficiency Diagnosis

**O3b-H: MIXED ENTITY CARRIER + PROJECTION FAILURE**

## 1. Executive Summary

O3b-G H0 reproduction succeeded from the frozen prior artifacts: K–K cosine F1 is TestA/TestB 88.44%/86.56%. No production architecture or real memory path was changed.

The actual write-side KeyEncoder input H retains substantial but incomplete entity information (74.20% TestA closed-world probe), while relation information is strong (97.34% TestA). A bounded frozen-Ouro mention-span probe reaches 100.00% on TestA, showing that entity identity exists locally upstream. The KeyEncoder reduces entity probe accuracy to 12.29%; the strongest evidence is therefore mixed carrier/state-selection loss plus projection loss.

## 2. Frozen Components

Backbone/Ouro, O2.1b KeyEncoder, O2.1a-MR QueryEncoder, V/ValueEncoder, GRU, allocator, memory storage/write/update and downstream answer head were frozen. Only offline linear/shallow probes and pairwise equality probes were trained. No oracle entity/relation labels were supplied to any inference representation.

## 3. Baseline Reproduction

Baseline reproduction: PASS. Exact prior O3b-G artifacts were reused and a separate no-training frozen-cache/checkpoint recheck matched them: TestA/TestB F1=88.44%/86.56%; P1=88.73%/85.87%; P2=54.80%/53.60%.

## 4. Main Diagnostic Matrix

| Representation | Entity TestA | Relation TestA | Entity TestB | Relation TestB |
|---|---:|---:|---:|---:|
| H_input | 74.20% | 97.34% | n/a | 97.04% |
| projection_linear_pre_activation | 31.17% | 95.50% | n/a | 94.21% |
| projection_activation | 31.00% | 95.93% | n/a | 94.60% |
| projection_final_pre_norm | 17.97% | 87.86% | n/a | 86.58% |
| final_normalized_K | 12.29% | 84.61% | n/a | 83.53% |
| G3_E | 1.13% | 57.89% | n/a | 55.53% |
| G3_R | 0.81% | 85.63% | n/a | 84.95% |
| G3_K | 2.73% | 41.85% | n/a | 42.88% |
| query H | 90.05% | 99.06% | n/a | 98.84% |
| query Q | 13.77% | 86.58% | n/a | 85.33% |

TestB entity-class accuracy is intentionally N/A for closed-world probes because all TestB persons are unseen. The TestB columns above are therefore only meaningful for relation; open-set entity evidence is the pairwise equality section below.

## 5. Raw Write H Sufficiency

Raw write H→entity: 74.20% TestA (3 seeds; mean±std=74.20%±0.26%); H→relation: 97.34% TestA / 97.04% TestB. This is not upstream absence: H contains a usable entity signal, but it is weaker than the relation signal and weaker than the local mention carrier.

## 6. Token / Span Carrier Search

The original MR cache persisted only final [2048] vectors; it had no token/layer arrays. A bounded frozen-Ouro extraction was therefore run on 4096 facts per split and 1024 queries per split, using layers 0/6/12/18/23. Fact entity-mention mean is 100.00%; fact global mean is 74.61%; fact final-token/EOS is 58.23%. The local mention span is the strongest carrier. Full token/span table is in `token_span_probe_results.json`.

| Carrier | Entity TestA | Relation TestA |
|---|---:|---:|
| entity_mention_mean | 100.00% | 25.51% |
| entity_mention_max | 100.00% | 25.09% |
| global_mean | 74.61% | 100.00% |
| final_token | 58.23% | 97.30% |
| next_token | 57.38% | 79.63% |
| relation_span_mean | 27.23% | 100.00% |
| layer_0_mention_mean | 98.89% | 25.13% |
| layer_6_mention_mean | 90.87% | 25.39% |
| layer_12_mention_mean | 90.74% | 26.07% |
| layer_18_mention_mean | 97.30% | 25.30% |
| layer_23_mention_mean | 99.19% | 25.43% |

Best sampled layer for entity mention mean is the final normalized layer (99–100% on this bounded sample); among the selected explicit layer checkpoints, layer 23 is strongest at 99.19%, with layer 0 also high at 98.89%. This is a carrier localization result, not a new architecture proposal.

## 7. KeyEncoder Internal Information Flow

| Stage | Entity TestA | Relation TestA | Entity loss vs input |
|---|---:|---:|---:|
| H_input | 74.20% | 97.34% | 0.0 pp |
| projection_linear_pre_activation | 31.17% | 95.50% | -43.0 pp |
| projection_activation | 31.00% | 95.93% | -43.2 pp |
| projection_final_pre_norm | 17.97% | 87.86% | -56.2 pp |
| final_normalized_K | 12.29% | 84.61% | -61.9 pp |
| G3_E | 1.13% | 57.89% | -73.1 pp |
| G3_R | 0.81% | 85.63% | -73.4 pp |
| G3_K | 2.73% | 41.85% | -71.5 pp |

The first large internal drop is H→first linear projection: 74.20%→31.17% (43.0 pp). A second drop occurs at the final 256→64 projection and L2 normalization: 31.00%→12.29%. G3's learned branches then nearly eliminate entity decodability (G3-E 1.13%, G3-R 0.81%, G3-K 2.73%).

## 8. Query vs Write Side

Query raw H→entity is 90.05% TestA and n/a TestB; query raw H→relation is 99.06% TestA / 98.84% TestB. After QueryEncoder, Q→entity falls to 13.77% and Q→relation to 86.58%. Thus query-side raw H is stronger than write-side raw H, but both projection heads discard entity information.

## 9. Open-Set Entity Equality and Geometry

| Representation | Equality F1 TestA | Equality F1 TestB | AUROC TestB | PR-AUC TestB |
|---|---:|---:|---:|---:|
| raw_H | 81.58% | 82.79% | 91.25% | 91.15% |
| query_H | 90.11% | 90.44% | 96.69% | 96.25% |
| final_K | 85.44% | 85.96% | 93.48% | 93.10% |
| G3_E | 74.55% | 73.17% | 80.16% | 80.72% |
| G3_R | 67.64% | 67.61% | 62.96% | 61.40% |
| G3_K | 66.68% | 66.67% | 58.92% | 65.51% |

| Representation | Same-entity cosine mean TestB | Different-entity cosine mean TestB | Overlap TestB |
|---|---:|---:|---:|
| raw_H | 0.9267 | 0.7874 | 0.566 |
| query_H | 0.9524 | 0.8520 | 0.471 |
| final_K | 0.5335 | 0.0478 | 0.259 |
| G3_E | 0.5132 | 0.0938 | 0.462 |
| G3_R | 0.2891 | 0.0456 | 0.712 |
| G3_K | 0.1238 | -0.0011 | 0.708 |

Raw H TestB equality is 82.79%, and query H is 90.44%; entity identity is therefore open-set recoverable upstream. Final K equality is 85.96%; its geometry is better than raw H for this pair probe, but that does not repair the closed-world entity representation or associative binding. G3-K overlap worsens to 0.708.

## 10. Position / Recency and Difficulty

Among 595 raw-H TestA entity-classification errors, none predicted another entity present in the same episode; first-member and last-member conditional rates are both 0% (see `recency_bias.json`). There is no measurable recency-latch signature in this audit, though the errors mostly point outside the episode, so this is not proof that recency is absent everywhere.

Multi-relation TestA raw-H entity accuracy is 76.14% for same_entity episodes, versus 80.24% for single episodes and 69.85% for lexical-stress episodes. TestB closed-world entity classification is N/A; pairwise equality is the valid unseen-person result.

## 11. Interpretation

The data distinguish the three hypotheses. Entity identity is not absent from the frozen backbone: the local entity mention and the raw write H carry it, and open-set same-entity equality generalizes to unseen persons. However, state selection/pooling has already diluted the signal (local mention near 100% versus raw write H 74.2%), and the KeyEncoder projection further destroys it (74.2%→31.2% at the first projection, then 12.3% at final K). The query-side raw H is stronger, but its QueryEncoder also reduces entity decodability. This supports the registered mixed diagnosis rather than a pure information-insufficient or pure projection-only verdict.

## 12. Final Verdict
**O3b-H: MIXED ENTITY CARRIER + PROJECTION FAILURE**

Entity identity most stably exists in the local entity mention span, especially the final frozen-Ouro token representations around the mention (bounded TestA probe ≈100%; selected layer 23 ≈99.19%). The current cached write H is a weaker global/state-selected carrier. The first clear KeyEncoder loss occurs at the 2048→256 linear projection, with additional loss at the 256→64 projection/L2 normalization and the O3b-G binding head.

Reconnect allowed? NO. O3c allowed? NO. No O3b-I was designed.

## 13. Runtime / Reproducibility
Git SHA: NO_GIT_REPO; cache reused from `D:\training\checkpoints\clr_v24_o21a_mr\cache`; seed=42; bounded token/layer extraction=True; wall-clock=880.5s; peak VRAM allocated=1.217 GB. Smoke completed before formal run. A separate no-training H0 recheck is saved as `h0_frozen_recheck.json`; its exact F1/P1/P2 numbers match the prior baseline. The 839 MB `token_layer_features.npz` remains on Cheems; JSON/report artifacts were mirrored locally.

## 14. Repeated V Update Contamination
Repeated V update contamination remains an independent unresolved issue.
