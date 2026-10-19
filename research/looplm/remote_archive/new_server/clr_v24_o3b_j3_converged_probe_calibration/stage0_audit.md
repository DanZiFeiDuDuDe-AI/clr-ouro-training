# O3b-J3 — Stage 0 data / cache audit

**Stage 0: FAIL**

No probe was trained and no optimizer/protocol was selected. Ouro/backbone was not re-forwarded. The KeyEncoder was loaded read-only and used only to reproduce frozen Z0–Z4 features.

## Critical finding

- O3b-H token feature rows cannot be strictly cross-joined to MR fact IDs: no row IDs/indices in NPZ and extraction manifest/source is absent

## Frozen definitions

- Z0: J-I2b input carrier = cached MR fact_H + 2 × cached entity_mention_mean.
- Z1: first Linear(2048→256), pre-activation.
- Z2: GELU output.
- Z3: second Linear(256→64), pre-normalization.
- Z4: L2-normalized 64-D key.
- R0 = H·Vhᵀ; R1 = R0·Σ; R2 = H·W1ᵀ = R1·Uᵀ.

W1 shape (256, 2048), dtype float32, rank 256, bias present=True, condition number=6.958538.

## Alignment and coverage

The inherited J builder assigns deterministic synthetic IDs from split and MR flattened-row index. All frozen KeyEncoder stages and R0/R1/R2 are derived from the exact same in-memory H row, and the old L2 cache is checked sample-by-sample against a fresh transform. However, O3b-H's token NPZ has no fact/sample IDs or indices, and its extraction source/manifest is not preserved in the available code. Thus the token mention vector cannot be independently cross-joined to a specific MR fact; same-length deterministic order is evidence of convention, not strict identity proof.

| split | MR refs | token rows | matched rows |
|---|---:|---:|---:|
| train | 11665 | 4096 | 4096 |
| val | 2332 | 2332 | 2332 |
| testA | 2332 | 2332 | 2332 |
| testB | 2332 | 2332 | 2332 |

Class-count details are in `tables/*_class_counts.json`; feature dtype, finite checks, norms, and scale are in `data_pipeline_audit.json`.

## Gate decision

Because row identity between the upstream token cache and MR labels cannot be strictly proven, the critical Stage 0 gate fails. Per the stop rule, optimizer calibration and all J stagewise revalidation are NOT RUN.

Checkpoint SHA-256 before/after: `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570` / `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570`. Checkpoint/model unchanged: **True**.
