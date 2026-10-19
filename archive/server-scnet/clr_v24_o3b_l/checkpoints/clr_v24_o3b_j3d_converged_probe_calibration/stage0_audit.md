# O3b-J3D Stage 0 — Transform and equivalence audit

Verdict: **PASS**. No probes were trained; no held-out metrics or model weights were read/changed.

SVD shapes: U=(256, 256), Sigma=(256,), Vh=[256, 2048]; U orthogonality max error=1.998e-15.

For train and validation, cached R1 was checked against R0·Sigma and cached R2 against R1·Uᵀ, including maximum/mean error, relative Frobenius error, per-row cosine, and sampled pair-distance correlation. A synthetic 7-class affine classifier transfer was checked in both directions on raw and train-standardized coordinates.

- train: PASS; R0→R1 max abs 1.907e-06; R1→R2 max abs 1.783e-06
- val: PASS; R0→R1 max abs 2.607e-06; R1→R2 max abs 1.591e-06

Classifier transfer synthetic checks:

- R1_to_R2: raw max logit error=1.190e-13, standardized=2.842e-13
- R2_to_R1: raw max logit error=9.948e-14, standardized=2.296e-13
