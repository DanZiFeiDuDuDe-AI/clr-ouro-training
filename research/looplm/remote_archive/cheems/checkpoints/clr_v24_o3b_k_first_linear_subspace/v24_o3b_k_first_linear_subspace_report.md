# CLR v2.4 — O3b-K First-Linear Subspace Attribution

O3b-K Verdict

"O3b-K: K-E NO DIRECT EVIDENCE FOR LEARNED ENTITY INVARIANCE"

Secondary:
"singular-value weighting and downstream geometry remain contributory; PCA-256 retains more entity signal than learned row-space"

Row-space attribution

Entity in row-space:
"86.36%"
Entity in nullspace:
"100.00%"
Relation in row-space:
"81.65%"
Relation in nullspace:
"81.13%"
Joint:
"row=29.20%; null=42.39%"

Discriminative-subspace alignment

Entity alignment:
"classifier top-64 alignment=0.1754; Fisher top-64 alignment=0.1853"
Relation alignment:
"classifier top-64 alignment=0.1947; Fisher top-64 alignment=0.3960"

Learned vs controls

Learned-256:
"entity=63.25%; relation=63.51%; joint=18.34% (K-I2b, 180-step probe)"
PCA-256:
"entity=86.75%; relation=45.03%; joint=24.78% (K-I2b, 180-step probe)"
Random-256 mean ± std:
"entity=56.84% ± 1.61% (N=10, K-I2b, 180-step probe)"
Learned entity percentile among random:
"100.0th percentile"

Singular weighting

Whitened row-space:
"entity=61.32%; relation=54.93%; joint=19.20%"
Real W1 output:
"entity=48.97%; relation=69.34%; joint=15.31%"
Interpretation:
"Whitened row coordinates retain more entity signal than real W1 output by 12.35pp; singular-value weighting contributes, but the row-space is not entity-empty."

Direct invariance test

Entity contraction:
"C_e=0.6149"
Relation contraction:
"C_r=0.7480"
Interpretation:
"C_e/C_r=0.8221; entity differences contract more than relation differences, but row/null attribution and controls do not establish learned entity invariance."

Core answer

On K-I2b, the input carrier begins with entity=100.00% and relation=85.55%; W1 row-space retains entity=86.36% and relation=81.65%, while the nullspace probe retains entity=100.00% and relation=81.13%. Entity information is therefore present in both row-space and nullspace, not selectively expelled into nullspace; classifier alignment is only relation 0.1947 vs entity 0.1754, with Fisher control showing a larger but non-exclusive relation preference. The orthonormal row coordinates retain entity=61.32% versus real W1H=48.97%, so anisotropic singular weighting contributes to the loss. The direct contraction ratio C_e/C_r=0.8221 shows stronger entity-difference contraction, but together with the row/null and random/PCA controls it does not upgrade O3b-J to direct learned entity-invariance evidence.

Reconnect allowed

No

O3c allowed

No

Repeated V update contamination

Unresolved; not touched.

Recommended next experiment

Run one frozen K-I2b comparison of real W1H against orthonormal row-space coordinates with matched 180-step probes, targeting singular-value weighting only; do not execute it in O3b-K.
