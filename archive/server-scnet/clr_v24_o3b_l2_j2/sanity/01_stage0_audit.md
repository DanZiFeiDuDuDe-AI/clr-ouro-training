# Stage 0 — reproduction audit

O3b-L config reused: `/root/private_data/clr_v24_o3b_l/output/config.json`.
Primary L2 carrier: `L-I2b` (J-I2b aligned rows); fixed seeds: `[11, 23, 37, 53, 71]`.

Frozen checkpoint: `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt`; SHA-256: `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570`. It was loaded read-only; no model/checkpoint writes.
W1 actual shape/dtype: `(256, 2048)` / `float32`; bias present: `True`; numerical rank: `256`.

Definitions match O3b-L: `R0=H@Vh.T`, `R1=R0*Sigma`, `R2=H@W1.T`; therefore `R2=R1@U.T` up to floating-point error. The cache is loaded once and all transformations preserve the existing split/sample order.

Exact inherited probe settings: 5 long-run seeds from config; linear `nn.Linear`; AdamW `lr=3e-4`, `weight_decay=0.01`; batch size `min(1024,n)`; seed+task initialization offsets and task-specific fixed NumPy minibatch schedule as in O3b-L; primary endpoint 500 steps, continuation to 1000 only for convergence diagnosis.

Sample/label maps are rebuilt using the O3b-J source functions from the same immutable MR cache; entity class ids come from train persons, relation ids are unchanged, joint ids are train-seen pairs only. Test-only unseen entity/joint metrics remain `n/a`.

## Algebraic equivalence checks

| split | max abs R2−R1Uᵀ | mean abs | relative Fro error | mean cosine | pairwise-distance corr |
|---|---:|---:|---:|---:|---:|
| train | 3.242e-05 | 1.685e-06 | 4.262e-07 | 1.000000000 | 1.000000000 |
| val | 3.433e-05 | 1.680e-06 | 4.268e-07 | 1.000000000 | 1.000000000 |
| testA | 3.052e-05 | 1.681e-06 | 4.257e-07 | 1.000000000 | 1.000000000 |
| testB | 4.005e-05 | 1.681e-06 | 4.263e-07 | 1.000000000 | 1.000000000 |

R2 is compared both to `R1 @ U.T` and the direct frozen `H @ W1.T`. Any large discrepancy would halt downstream mechanism interpretation. Pairwise checks use same-row random pairs and do not fit on evaluation data.

## Sample alignment and label coverage

O3b-H caches mention carriers for only the first 4096 train facts; MR cache contains 11665 train facts. As in O3b-L, the carrier and every label are aligned to the same deterministic prefix via `[:len(carrier)]` (validation/test are fully covered). Full label count and aligned carrier-prefix count are both recorded in `stage0_metadata.json`; entity/relation/joint label arrays have identical source ordering. No test split contributes to standardization, hyperparameter selection, or stopping.
