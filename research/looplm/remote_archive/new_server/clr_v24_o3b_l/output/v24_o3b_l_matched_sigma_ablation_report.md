# CLR v2.4 — O3b-L Matched Singular-Weighting Ablation

O3b-L Verdict

"O3b-L: L-B SINGULAR WEIGHTING MAINLY CHANGES PROBE CONDITIONING"

Secondary:
"Raw-vs-standardized comparison and Sigma permutations are the primary controls; no architecture repair was performed."

Matched probe results

R0 orthonormal row:
"TestA 61.97%/58.37%/19.10%; TestB n/a/57.11%/n/a"
R1 singular-weighted:
"TestA 39.75%/53.31%/10.66%; TestB n/a/53.37%/n/a"
R2 full W1 no bias:
"TestA 47.44%/69.90%/15.02%; TestB n/a/70.21%/n/a"
R3 W1 + bias:
"TestA 47.45%/66.83%/15.14%; TestB n/a/66.80%/n/a"

Standardized probe results

R0:
"TestA 96.41%/75.12%/33.05%; TestB n/a/71.61%/n/a"
R1:
"TestA 96.37%/75.07%/32.74%; TestB n/a/72.29%/n/a"
R2:
"TestA 85.32%/81.46%/32.52%; TestB n/a/80.08%/n/a"
R3:
"TestA 85.46%/81.29%/32.32%; TestB n/a/79.73%/n/a"

Singular weighting effect

Entity:
"raw testA Δ=-22.22pp (95% CI -23.64,-20.90); testB Δ=n/a (no train-seen labels); standardized testA Δ=-0.04pp (95% CI -0.35,+0.29); testB Δ=n/a (no train-seen labels)"
Relation:
"raw testA Δ=-5.06pp (95% CI -8.46,-1.35); testB Δ=-3.73pp (95% CI -7.64,+0.30); standardized testA Δ=-0.05pp (95% CI -0.82,+0.79); testB Δ=+0.68pp (95% CI -0.34,+1.87)"
Joint:
"raw testA Δ=-8.43pp (95% CI -9.83,-7.24); testB Δ=n/a (no train-seen labels); standardized testA Δ=-0.31pp (95% CI -0.72,+0.12); testB Δ=n/a (no train-seen labels)"
Available-split summary:
"Standardized E/R/J deltas are -0.04/+0.32/-0.31pp over non-empty split(s) only; see per-split paired results in matched_effects.csv."

Probe-conditioning check

180-step:
"Raw entity Δ=-22.22pp; standardized entity Δ=-0.04pp; all representations used identical 180-step linear-probe protocol."
Longer-step sanity:
"500-step R1-R0 entity Δ=-3.95pp using 5 seeds; see longer_step_sanity.csv."
Interpretation:
"Feature covariance and raw-vs-standardized results distinguish scale/conditioning from a stable information change; R1/R2 equivalence checks are in representation_equivalence_checks.txt."

Sigma semantic structure
corr(sigma, entity-score):
"Pearson=-0.2332; Spearman=-0.5524"
corr(sigma, relation-score):
"Pearson=0.3477; Spearman=0.4191"
Real Sigma entity percentile among permutations:
"mean available-split percentile=0.0th (lower-tail statistic); detailed rows in sigma_permutation_results.csv"
Real Sigma relation percentile among permutations:
"mean available-split percentile=15.0th (higher is more relation-favorable); detailed rows in sigma_permutation_results.csv"

Geometry
P0/P1:
"TestA cosine AUROC R0=0.9825, R1=0.9876; euclidean and Mahalanobis are in geometry_results.csv; TestB cosine AUROC R0=0.9832, R1=0.9839; euclidean and Mahalanobis are in geometry_results.csv"
P0/P2:
"TestA cosine AUROC R0=0.7430, R1=0.7833; euclidean and Mahalanobis are in geometry_results.csv; TestB cosine AUROC R0=0.6980, R1=0.7521; euclidean and Mahalanobis are in geometry_results.csv"
Cosine vs Euclidean interpretation:
"Cosine is scale-insensitive; Euclidean is scale-sensitive; Mahalanobis uses train covariance. Compare all three in geometry_results.csv."

Attribution
Original O3b-J entity cliff:
"55.62pp"
Sigma-attributed entity change:
"standardized ΔΣ(E)=-0.04pp; raw ΔΣ(E)=-22.22pp"
Approximate attribution ratio:
"standardized=0.07%; raw=39.94%"

Core answer
Under the fully matched 180-step linear protocol, R0 versus R1 changes entity by -0.04pp, relation by +0.32pp, and joint by -0.31pp after train-only standardization. The raw difference is -22.22pp entity and -4.40pp relation, indicating strong scale/conditioning sensitivity. However, the 180-vs-500-step gap and R1/R2 probe disagreement show that optimization is not saturated; no singular-weighting semantic-mechanism conclusion is warranted. R1 and R2 are algebraically equivalent up to the orthogonal U rotation, with reconstruction verified numerically; the probe-score difference is an optimization/protocol warning. The measured Sigma effect is attributed diagnostically against the 55.62pp O3b-J cliff, not treated as a strict causal decomposition.

Reconnect allowed

No

O3c allowed

No

Repeated V update contamination

Unresolved; not touched.

Recommended next experiment

Only one follow-up is proposed: extend the existing 500-step R0/R1/R2 sanity check to train-standardized features using the same five seeds, to test convergence and orthogonal-rotation invariance. It was not executed in O3b-L.

TestB label coverage

Entity known: 0/2332.
Relation known: 2332/2332.
Joint train-seen (entity, relation) labels: 0/2332. Unknown labels remain masked as -1 under the fixed closed-set protocol; metrics are n/a, not imputed.

Probe optimization saturation

PROBE OPTIMIZATION NOT SATURATED: raw 180-step R1-R0 entity Δ=-22.22pp versus 500-step Δ=-3.95pp; score-gap change=+18.27pp.
Raw 180-step R2-R1 probe deltas despite algebraic equivalence: entity=+7.68pp, relation=+16.60pp, joint=+4.36pp. Maximum float32 reconstruction error was 4.01e-05.
Under the prespecified stop rule, these optimization-sensitive probe results do not support a strong semantic-mechanism conclusion about Sigma.
