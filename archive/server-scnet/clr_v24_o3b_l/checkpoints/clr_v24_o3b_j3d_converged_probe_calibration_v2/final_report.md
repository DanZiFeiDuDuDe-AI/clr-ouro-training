# O3b-J3D — Converged matched-probe calibration

## Final verdict

**J3D-F — CONVERGED MATCHED-PROBE CALIBRATION GATE FAILED**

The certified cache and row-transform checks passed. The independent-refit calibration did **not** pass: neither R0/R1 nor R1/R2 met the preregistered convergence requirement (and R1/R2 also has a joint TestA discrepancy). Therefore Z0–Z4 stagewise attribution, activation remeasurement, and nonlinear recoverability were not run.

## Calibration protocol

Five frozen seeds `[11, 23, 37, 53, 71]`; affine linear probes; AdamW selected per task using train/validation only; chosen settings: `{"entity": {"optimizer": "AdamW", "lr": 0.003, "weight_decay": 0.0, "candidate_index": 2}, "relation": {"optimizer": "AdamW", "lr": 0.003, "weight_decay": 0.0, "candidate_index": 2}, "joint": {"optimizer": "AdamW", "lr": 0.003, "weight_decay": 0.0, "candidate_index": 2}}`. Early stopping and convergence used validation only, with a 50,000-step cap. TestA/TestB were evaluated only after protocol freeze.

## Convergence

| contrast | task | plateau-confirmed | total | max-step stops |
|---|---|---:|---:|---:|
| R0_R1 | entity | 0 | 20 | 10 |
| R0_R1 | relation | 0 | 20 | 0 |
| R0_R1 | joint | 3 | 20 | 0 |
| R1_R2 | entity | 0 | 20 | 8 |
| R1_R2 | relation | 0 | 20 | 0 |
| R1_R2 | joint | 3 | 20 | 0 |

## Held-out linear accessibility

Mean±SD across the five paired seeds. These are probe scores, not information-content estimates.

**R0/R1 independent fits**

| representation | task | TestA raw | TestA standardized | TestB raw | TestB standardized |
|---|---|---:|---:|---:|---:|
| R0 | entity | 100.00±0.00% (n=5) | 100.00±0.00% (n=5) | n/a | n/a |
| R0 | relation | 97.45±0.07% (n=5) | 97.18±0.02% (n=5) | 90.70±0.09% (n=5) | 90.53±0.06% (n=5) |
| R0 | joint | 84.15±0.17% (n=5) | 83.97±0.07% (n=5) | n/a | n/a |
| R1 | entity | 100.00±0.00% (n=5) | 100.00±0.00% (n=5) | n/a | n/a |
| R1 | relation | 97.18±0.04% (n=5) | 97.18±0.02% (n=5) | 90.79±0.05% (n=5) | 90.53±0.06% (n=5) |
| R1 | joint | 87.11±0.35% (n=5) | 83.97±0.07% (n=5) | n/a | n/a |

**R1/R2 independent fits**

| representation | task | TestA raw | TestA standardized | TestB raw | TestB standardized |
|---|---|---:|---:|---:|---:|
| R1 | entity | 100.00±0.00% (n=5) | 100.00±0.00% (n=5) | n/a | n/a |
| R1 | relation | 97.18±0.04% (n=5) | 97.18±0.02% (n=5) | 90.79±0.05% (n=5) | 90.53±0.06% (n=5) |
| R1 | joint | 87.11±0.35% (n=5) | 83.97±0.07% (n=5) | n/a | n/a |
| R2 | entity | 100.00±0.00% (n=5) | 100.00±0.00% (n=5) | n/a | n/a |
| R2 | relation | 96.95±0.08% (n=5) | 97.36±0.08% (n=5) | 90.50±0.31% (n=5) | 90.09±0.06% (n=5) |
| R2 | joint | 92.28±0.42% (n=5) | 92.73±0.14% (n=5) | n/a | n/a |


## Matched contrasts and direct transfer

- R0 vs R1, standardized: TestA paired deltas are entity +0.00 pp, relation +0.00 pp, joint +0.00 pp (left minus right). TestB relation is +0.00 pp; entity and joint TestB have no known labels.
- R1 vs R2, standardized: TestA paired deltas (R1−R2) are entity +0.00 pp, relation -0.18 pp, and joint -8.76 pp. The joint discrepancy exceeds the 1 pp tolerance; TestB joint labels are unavailable.
- Analytical R1↔R2 classifier transfer: **PASS**; maximum absolute logit error across tasks/seeds/raw+standardized/splits = `1.449e-05`; maximum prediction mismatch = `0.0%`.
- Interpretation: the same fitted decision function transfers across the algebraically equivalent R1/R2 coordinates, while independently retrained probes do not reliably agree. This is consistent with probe optimization / parameterization sensitivity; it is not evidence that the representation itself lost the task information.

## Gates, controls, and stop decision

- R0/R1 standardized gate: **FAIL**; paired TestA score deltas are zero/within tolerance, but plateau confirmation is `False`.
- R1/R2 standardized independent-refit gate: **FAIL**; convergence confirmation is `False`; joint TestA also exceeds tolerance.
- Direct transfer gate: **PASS**. Weight-decay and alternate-optimizer controls ran; see their per-seed tables and convergence counts in `final_evidence_summary.json`.
- Stagewise Z0–Z4, activation, nonlinear recoverability, and J cliff revalidation: **NOT RUN** because calibration gates closed. The historical −55.62 pp cliff is therefore neither reproduced nor retracted by this run.
- No backbone/KeyEncoder/checkpoint modification, no reconnect/O3c, and no data leakage.

## What the evidence supports

Train-only standardization makes the R0/R1 held-out probe scores coincide in this run, consistent with singular-value scale being removed by per-coordinate standardization. However, the prescribed convergence gate did not pass, so this does not certify a converged measurement pipeline. The R1/R2 transfer result confirms algebraic equivalence of the frozen representations for a fixed linear classifier; the independent-refit joint gap is best treated as unresolved optimization sensitivity. No conclusion that W1 destroys entity information, learns entity invariance, or that activation is a primary cause is supported here.

## Runtime and provenance

- Remote output: `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration_v2`
- Single-GPU worker wall time: 29.9 min (0.499 h); exact device busy time was not sampled.
- Cache identity/hash audit: PASS; row-transform and synthetic classifier-transfer Stage 0: PASS.
- Initial invalidated harness attempt is preserved separately and marked `INVALIDATED_NOT_FOR_INFERENCE`; its outputs were excluded.
- Full file SHA-256 inventory: `artifact_hash_manifest.json`.

## Exactly one recommended next experiment

Run a focused R1/R2-standardized joint-probe convergence audit only: keep the same certified cache, five seeds, analytical transfer invariant, and train/validation-only selection; preregister a small LR/step schedule that can actually reach the plateau criterion, freeze it before TestA/TestB, and do not run Z-stage attribution unless both invariance gates then pass.

Reconnect: **NO**  
O3c: **NO**  
Repeated-V-update contamination: unresolved and untouched.
