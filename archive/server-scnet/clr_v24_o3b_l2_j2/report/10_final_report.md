# O3b-L2 / J2 calibration block — final report

## Executive verdict

Primary: `J2-F — MEASUREMENT PIPELINE STILL NOT RELIABLE` for projection-cliff mechanism attribution.
Secondary: standardized R0/R1 are feature-level invariant, and transferred probes confirm R1/R2 linear-algebraic equivalence; independently refitted probes continue to show optimization/basis sensitivity through 1000 steps.

## What was tested

The O3b-L exact 5 long-run seeds and frozen L-I2b carrier were reused. R0/R1/R2 probes used same splits, class mappings, float32 linear architecture, AdamW, initialization, task-specific minibatches, 500-step primary endpoint and 1000-step continuation. Standardization was fit on train only. No checkpoint/model weights were changed and no J revalidation, O3c or reconnect was run.

## Stage 1 measurement sanity

At 500 steps, standardized R0 and R1 features were numerically the same after per-coordinate train standardization (max error <1.2e-6), as expected for positive diagonal singular rescaling. But the validation curves are not at a plateau: at 1000 steps several task/representation scores still improve materially, especially joint address. R1/R2 independently trained probes remain separated for some tasks/splits despite their algebraic equivalence.

At 1000 standardized TestA, R1→R2 paired changes were approximately: entity −0.67pp, relation −1.23pp, joint +7.55pp; the joint delta 95% CI excludes zero. R1→R2 relation moved in the opposite direction on TestB (+1.24pp), so it does not replicate across held-out split. The most important diagnostic is the systematic R2 joint advantage (also in raw probes), despite equivalence; this is a probe optimization/conditioning discrepancy, not a semantic conclusion.


Standardized TestA probe scores (mean±SD) continue to rise from 500 to 1000 steps:

| task | R0 500→1000 | R1 500→1000 | R2 500→1000 |
|---|---:|---:|---:|
| entity | 100.00±0.00% → 100.00±0.00% | 99.99±0.02% → 100.00±0.00% | 98.38±0.15% → 99.33±0.08% |
| relation | 86.98±0.66% → 91.29±0.22% | 86.79±0.50% → 91.19±0.12% | 87.03±0.34% → 89.97±0.21% |
| joint | 50.71±0.12% → 56.69±0.13% | 50.42±0.27% → 56.55±0.38% | 56.81±0.18% → 64.10±0.22% |

At 1× learning rate, paired standardized R0/R1 TestA differences were:

- Entity: 500 -0.01pp (paired t95 -0.03 to +0.02); 1000 +0.00pp (paired t95 +0.00 to +0.00).
- Relation: 500 -0.19pp (paired t95 -1.09 to +0.71); 1000 -0.09pp (paired t95 -0.22 to +0.03).
- Joint: 500 -0.29pp (paired t95 -0.77 to +0.19); 1000 -0.14pp (paired t95 -0.63 to +0.35).

At 1000 steps, paired R2−R1 standardized differences were:

- Entity TestA: -0.67pp (paired t95 -0.77 to -0.57).
- Relation TestA / TestB: -1.23pp (paired t95 -1.45 to -1.00) / +1.24pp (paired t95 +0.60 to +1.89).
- Joint TestA: +7.55pp (paired t95 +6.87 to +8.24).

The 0.5×/1×/2× matrix also changes R1/R2 gaps and, for raw relation probes, reverses the TestA/TestB ranking sign at 1000 steps. Exact per-seed paired intervals are in `optimizer_robustness_paired.csv`; no test result was used to choose an LR.

## Weight-transfer invariance

A classifier trained in R1 coordinates was analytically transformed into R2 coordinates, with a separate affine formula for independently standardized coordinates. All split/task predictions matched in accuracy; max raw logit discrepancy is on the order of 1e-5 and standardized on the order of float32 numerical error. Thus, a probe's learned decision function is transferable; differing independently refitted scores are attributable to optimization/regularization path, not absent linear accessibility.

## Conditioning and branch B

R1 raw covariance is more ill-conditioned than R0 (see `conditioning_raw_standardized.json`). Train-only standardization makes R0/R1 nearly identical, but does not make 1000-step performance plateau. The optimizer matrix varies learning rate by 0.5×/1×/2× while holding all other protocol components fixed; results and paired CIs are in `optimizer_robustness.csv` and `optimizer_robustness_paired.csv`. Repeated-run determinism is in `deterministic_repeat_control.csv`.

## Pipeline audit and provenance

Label/reference row alignment, class balance, seeds, fresh optimizer state, init, fixed minibatch order, precision, weight decay and no hidden normalization layers are documented in `measurement_pipeline_audit.json`. No TestA/TestB leakage into standardization or early-stop/hyperparameter choice occurred. One residual limitation: the token feature NPZ does not carry explicit sample IDs; alignment follows the inherited deterministic flattened order and exact O3b-L builder, not a keyed join.

## J cliff revalidation

Not run: Stage 1 failed the measurement gate. Therefore old J −55.62pp is not recalibrated here. The correct current statement is unresolved under a not-yet-saturated probe pipeline; neither support nor rejection of broader representational degradation is warranted from these scores.

## What survives / what is weakened

- Survives: O3b-J localized a large *probe-score cliff* at Z0→Z1 under its protocol; O3b-K found no direct evidence that entity differences were primarily placed in W1 nullspace.
- Survives: O3b-L showed singular scaling is conditioning-sensitive under finite-step probes.
- Weaken: any interpretation of the original −55.62pp as entity information destruction or learned entity invariance. L2 shows substantial optimization dependence and exact algebraic transfer.
- Not concluded: W1 preserves all information; classifier transfer proves equivalence only within the tested R1/R2 linear-coordinate relation, not the Z0→Z1 dimension reduction.

## Remaining uncertainty

The optimizer matrix tests finite-step sensitivity; exact token-to-cache sample IDs are absent in the upstream NPZ; no calibrated J stage-wise remeasurement was run because the measurement gate failed. Nonlinear recoverability and activation are consequently unresolved in this run.

## Exactly one recommended next experiment

A single extended matched linear-probe convergence study on frozen O3b-J stage features and controls, with optimizer settings chosen only from the train/validation calibration split and a stopping rule fixed before TestA/TestB evaluation; do not reconnect memory or change architecture.

## Operational status

Reconnect: No. O3c: No. Repeated-V update contamination: unresolved; not touched. Checkpoint/model modified: No. Data leakage: none detected. Runtime/GPU: see `measurement_runtime.json`.
