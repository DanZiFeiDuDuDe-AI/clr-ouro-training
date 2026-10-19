# CLR v2.4 — O3b-I: Mention-Aware Write State

**O3b-I: WRITE-STATE SELECTION IS A MAJOR ADDRESSING BOTTLENECK**

## 1. Executive Summary

This run changed only the write-side state supplied to the frozen original KeyEncoder. O3b-H frozen-Ouro mention/layer features were reused; no backbone/Ouro rerun, KeyEncoder training, QueryEncoder change, binding change, real memory reconnect or O3c was performed.
Validation-selected best variant: `I1_pure_mention_mean`. TestB diagnostic-best variant: `I1_pure_mention_mean`. Input entity probe changed from 74.20% to 100.00%, while final-K entity probe changed from 12.29% to 6.05%. TestB K-K F1 changed from 86.56% to 92.38%.

## 2. Frozen / Changed

Frozen: backbone/Ouro, original O2.1b KeyEncoder including 2048→256, GELU, 256→64 and L2 normalization, QueryEncoder, V/GRU, allocator, memory/write path and downstream head. Changed: write-side input state selection only. I3 is a parameter-free norm-adaptive gate; no oracle label is used at inference.

## 3. I0 Baseline Reproduction

Exact no-training K-K baseline reproduced: TestA/TestB F1=88.44%/86.56%; UPDATE recall=88.73%/85.87%; P1=88.73%/85.87%; P2=54.80%/53.60%; AUROC=95.69%/94.94%; PR-AUC=95.78%/94.94%.

## 4. Mention Carrier Variants

| Variant | TestA F1 | TestB F1 | TestA recall | TestB recall | TestA P2 | TestB P2 | same-E rel B | entity-fixed B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I0_raw_write_H | 88.44% | 86.56% | 88.73% | 85.87% | 54.80% | 53.60% | 56.79% | 54.22% |
| I1_pure_mention_mean | 92.42% | 92.38% | 100.00% | 99.80% | 1.60% | 2.40% | 21.66% | 22.33% |
| I2_residual_a0p5 | 90.09% | 88.97% | 94.87% | 93.60% | 32.40% | 31.60% | 56.84% | 54.22% |
| I2_residual_a1 | 90.89% | 90.54% | 97.07% | 96.93% | 22.00% | 21.20% | 56.74% | 53.96% |
| I2_residual_a2 | 91.47% | 91.20% | 99.67% | 99.47% | 4.40% | 6.00% | 56.94% | 54.37% |
| I3_norm_gate | 89.93% | 89.14% | 92.53% | 91.93% | 40.40% | 38.00% | 56.69% | 54.17% |
| I4_layer23_mention | 92.42% | 92.38% | 100.00% | 99.80% | 1.60% | 2.40% | 21.60% | 22.89% |

## 5. Information Flow

| Variant | Input entity | Input relation | First projection entity | Pre-norm K entity | Final K entity | Final K relation |
|---|---:|---:|---:|---:|---:|---:|
| I0_raw_write_H | 74.20% | 97.34% | 31.62% | 18.32% | 12.29% | 84.61% |
| I1_pure_mention_mean | 100.00% | 25.51% | 54.57% | 9.69% | 6.05% | 25.73% |
| I2_residual_a0p5 | 93.41% | 96.14% | 34.29% | 17.92% | 10.75% | 76.80% |
| I2_residual_a1 | 99.24% | 95.28% | 51.14% | 19.05% | 9.91% | 68.65% |
| I2_residual_a2 | 100.00% | 92.62% | 78.23% | 25.14% | 8.96% | 56.65% |
| I3_norm_gate | 95.08% | 96.01% | 36.44% | 18.08% | 10.59% | 75.73% |
| I4_layer23_mention | 99.17% | 25.43% | 27.27% | 3.26% | 1.69% | 26.50% |

Input probe TestB entity classification is N/A because TestB persons are unseen; TestB same-entity equality is used below.

## 6. Entity Equality / Geometry

| Variant | Equality F1 TestA | Equality F1 TestB | AUROC TestB | PR-AUC TestB | Overlap TestB |
|---|---:|---:|---:|---:|---:|
| I0_raw_write_H | 83.13% | 84.17% | 92.15% | 91.81% | 0.566 |
| I1_pure_mention_mean | 100.00% | 100.00% | 100.00% | 100.00% | 0.000 |
| I2_residual_a0p5 | 92.03% | 92.66% | 97.89% | 98.13% | 0.407 |
| I2_residual_a1 | 97.89% | 97.80% | 99.87% | 99.88% | 0.189 |
| I2_residual_a2 | 99.71% | 99.68% | 100.00% | 100.00% | 0.029 |
| I3_norm_gate | 93.78% | 93.92% | 98.47% | 98.66% | 0.374 |
| I4_layer23_mention | 100.00% | 100.00% | 100.00% | 100.00% | 0.000 |

## 7. Collision / Margin / Error Audit

Best variant `I1_pure_mention_mean` TestB margin and error decomposition are saved in `margin_analysis.json` and `error_audit.json`. E+/R−, E−/R+, E−/R− categories are reported from the exact within-episode retrieval audit; no Q→K non-gate metric is substituted.
Best variant TestB margin: {"all_negatives": {"n": 1332, "mean": 0.15710550609293641, "std": 0.1538268532088462, "p10": 0.0, "p25": 0.0, "p50": 0.17196083068847656, "p75": 0.27541178464889526, "p90": 0.3587459266185764, "negative_rate": 0.0}, "same_entity_wrong_relation": {"n": 568, "mean": 3.4739643755093427e-06, "std": 3.5293883967342415e-05, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "negative_rate": 0.0}, "wrong_entity_same_relation": {"n": 252, "mean": 0.286662014703902, "std": 0.09938520758034551, "p10": 0.17348432540893555, "p25": 0.22072643041610718, "p50": 0.27167025208473206, "p75": 0.34162233769893646, "p90": 0.4090774357318878, "negative_rate": 0.0}}
Best variant TestB error fractions: {"E+_R-": 1.0}

## 8. State Selection Contribution
Mention-aware input entity gain=25.80 pp. Final K entity gain=-6.25 pp. The residual projection loss remains 93.95 pp for the best variant, versus 61.91 pp for I0.

## 9. Final Verdict
**O3b-I: WRITE-STATE SELECTION IS A MAJOR ADDRESSING BOTTLENECK**

The result separates the two mechanisms: write-state selection is evaluated by the input probe gain; projection remains fixed and is evaluated by the input-to-final-K drop. This run is diagnostic only and does not establish a reconnect-ready key.

Reconnect allowed: NO. O3c allowed: NO. O3b-J was not designed.

## 10. Runtime / Repeated V Contamination
Git SHA=NO_GIT_REPO; cache reused; O3b-H token/layer features reused without Ouro rerun; wall-clock=451.4s; peak VRAM={"allocated_gb": 0.270918144, "reserved_gb": 0.291504128}.
Repeated V update contamination remains an independent unresolved issue.
