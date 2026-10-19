# CLR-v2.4-O3b-D — Same-Key Equivalence Diagnosis and Repair

**O3b-D FAIL: RELATION IDENTITY UNDER-REPRESENTED**

## Scope
Only `(K_new, K_old) -> SAME/DIFFERENT` was studied. KeyEncoder, value path, query path, O3a allocator, Ouro, and physical slots were frozen. No V write, allocation, eviction, or K retraining was performed unless the offline gate passed.

## Baseline ladder
Best judge: B3_pair_mlp; parameters=16513; val threshold=0.422842; best step=3000.
B0_raw_cosine: TestA F1=88.44%, recall=88.73%; TestB F1=86.56%, recall=85.87%.
B1_negative_l2: TestA F1=88.44%, recall=88.73%; TestB F1=86.56%, recall=85.87%.
B2_margin_linear: TestA F1=89.26%, recall=90.87%, P2=51.60%; TestB F1=86.78%, recall=87.07%, P2=50.40%.
B3_pair_mlp: TestA F1=95.09%, recall=96.20%, P2=82.40%; TestB F1=94.07%, recall=94.07%, P2=83.60%.
B4_symmetric_pair_judge: TestA F1=94.55%, recall=95.93%, P2=76.80%; TestB F1=92.96%, recall=93.33%, P2=76.40%.

## Offline gate
TestA F1/recall/P2: 95.09% / 96.20% / 82.40%.
TestB F1/recall/P2: 94.07% / 94.07% / 83.60%.
Reconnect: SKIPPED.

## Diagnostics
Geometry, threshold overlap, margin/relative-score analysis, pair probes, calibration, symmetry, and hard-negative results are in the JSON outputs. The P2 category is same entity/different relation and is the primary failure diagnostic.

## Boundary
Repeated V-update contamination remains a separate unresolved issue and was not modified. This run does not claim eviction, replacement, full-memory new-key policy, or latent reasoning.

## Stop
O3b-D complete; do not automatically enter O3c.
