# O3b-J3 — Final report

## FINAL VERDICT

**Primary: J3-F — STAGE 0 ROW-IDENTITY AUDIT FAILS; CONVERGED PROBE CALIBRATION NOT RUN**

The hard Stage 0 gate failed before any probe fitting. This is a data-provenance failure, not evidence that R1/R2 basis invariance failed.

## Stage 0 evidence

Prior-artifact audit: the O3b-J stage graph was recovered from `a1_v24_o3b_j.py`; O3b-K final report and metric tables, O3b-L report/config, and O3b-L2/J2 report plus Stage 0/1 code were reviewed. The standalone O3b-J report and O3b-K analysis script are not present in the preserved remote artifact tree, so their published conclusions were treated as historical findings, not re-estimated.
- Checkpoint SHA-256 before/after: `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570` / `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570` (unchanged).
- W1: `(256, 2048)`, float32, rank 256, condition number 6.9585; bias present.
- Reconstructed H/R0/R1/R2 tensors match the archived L2 cache exactly (max abs error 0 for all splits and tensors).
- R2 vs R1·Uᵀ max abs error: train 3.24e-05, val 3.43e-05, TestA 3.05e-05, TestB 4.01e-05.
- All audited feature arrays are finite; shapes, dtype, scale, and row-norm summaries are in `data_pipeline_audit.json`.
- Split labels and class counts are recorded in `tables/*_class_counts.json`; entity IDs use train persons, joint IDs use train-seen pairs, and TestB entity/joint labels are unknown (n/a).
- Stable synthetic row IDs (`split:<row_index>`) and source episode/position indices are in `row_alignment_audit.csv`; deterministic MR flatten order is consecutive and all within-run transforms preserve row order.
- The upstream O3b-H token feature NPZ contains no row/sample/fact IDs, and the extraction source/manifest is absent from the preserved code. Therefore matching its mention vectors to MR fact IDs cannot be strictly independently verified; the inherited prefix-order assumption is not sufficient under this J3 gate.

## Gate and stop decision

Measurement gate: **FAIL at Stage 0**. Optimizer matrix, early-stopping calibration, R0/R1 positive-control probes, R1/R2 independent refits, and weight-transfer probe evaluation were not run. TestA/TestB were read only for identity/label/finite-feature auditing; they were not used for fitting or selection.
J stagewise linear/nonlinear revalidation and activation attribution: **NOT RUN**.

## Required questions

1. Converged standardized R0/R1 invariance: **UNRESOLVED — calibration not run**.
2. Converged standardized R1/R2 independent-refit invariance: **UNRESOLVED — calibration not run**.
3. Weight transfer under J3: **UNRESOLVED — no new probe was trained**. The prior L2 algebraic transfer result is retained as historical evidence only, not a J3 gate result.
4. Measurement pipeline: **FAIL** at data/row-identity audit, before probe calibration.
5. Calibrated raw Z0→Z1 entity cliff: **NOT RUN**.
6. Calibrated standardized Z0→Z1 entity cliff: **NOT RUN**.
7. Conditioning/optimization attribution for the old −55.62pp cliff: **UNRESOLVED — not measured in J3**.
8. Residual linear accessibility loss after calibrated convergence: **UNRESOLVED**.
9. Nonlinear recoverability of Z1 entity performance: **NOT RUN**.
10. ‘W1 projection destroys entity information’: **UNRESOLVED**; this round provides no calibrated J evidence for or against it.
11. ‘W1 projection reduces linear entity accessibility’: **UNRESOLVED** under J3; the original J probe cliff remains unvalidated.
12. Activation as a new primary suspect: **UNRESOLVED / NOT RUN**.
13. Safest combined mechanism statement: O3b-K found no direct evidence that W1 row-space selectively rejects entity information; O3b-L showed that train-only standardization removes the R0/R1 probe gap; O3b-L2 still found unsaturated, basis/optimizer-sensitive independent fits despite exact weight-transfer parity. Thus the original J cliff cannot yet be attributed reliably to entity suppression or information loss.
14. Reconnect allowed: **NO**.
15. O3c allowed: **NO**.

Architecture modified: **NO**. Checkpoint modified: **NO**. Data leakage detected: **NO**. Probe training started: **NO**. Ouro/backbone re-forwarded: **NO**.

## Exactly one next experiment

Recover the original O3b-H extraction provenance and produce an immutable row-ID sidecar (`split`, MR flat index, episode, item position) for every cached token feature; then rerun only Stage 0 and proceed to calibration only if the row-level join is verified. Do not run probe calibration in the meantime.

Output directory: `/root/private_data/clr_v24_o3b_j3_converged_probe_calibration`. Git SHA: unavailable (`NO_GIT_REPO`). Stage 0 runtime: 23.1s; no GPU training or model updates occurred.
