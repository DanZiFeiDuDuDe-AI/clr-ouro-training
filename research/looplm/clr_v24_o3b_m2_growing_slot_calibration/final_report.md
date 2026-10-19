# O3b-M2 — Growing-Slot Competition Calibration + True Value Replacement

## FINAL VERDICT

**O3b-M2-INCOMPLETE — OFFLINE ACTION GATES PASS; REAL RECONNECT BLOCKED BY MISSING FROZEN DEPENDENCIES**

Offline Candidate B action and true-value replacement gates pass at the existing O3b capacity of 4. A real O3b reconnect and retrieval/KV regression test were **not run**: required frozen O3b checkpoints are absent from the active remote checkpoint tree. Therefore this is not an end-to-end O3b PASS.

## Frozen policy and holdout

- Candidate B: frozen deployment seed 43; checkpoint SHA-256 before/after: `43b9b6dc4c0baf3d85c37c6debe2ae7eaaf0baeb0636a50367a001024c0f94fc` / `43b9b6dc4c0baf3d85c37c6debe2ae7eaaf0baeb0636a50367a001024c0f94fc`; state-dict SHA-256 before/after: `c7a087c7e99fa276e49c4a7497516aa6179496566e92c138034116785ee10808` / `c7a087c7e99fa276e49c4a7497516aa6179496566e92c138034116785ee10808`. No retraining or weight modification occurred.
- Validation selected Rule A, one global threshold `0.8504285216331482`; no N-dependent threshold and no top1-top2 margin. A/B/C all passed the operational validation gate; the simpler Rule A won by the preregistered priority.
- TestA/TestB were evaluated only after policy freeze. Both were filtered to remove exact `(entity, relation, value)` semantic facts overlapping train/validation; remaining cross-split semantic-triple overlap is zero. Source facts/source episodes are split-disjoint. Before exact-triple filtering, TestA shares 1,479 `(entity, relation)` key identities with train and 922 with validation; TestB shares zero with either. This key-identity overlap is disclosed, while source-row/feature-row/record overlap and post-filter exact-triple overlap are zero.
- Each holdout has 240 generated episodes per N × event-type cell; source facts can be reused between Monte Carlo episodes within a split. The episode draws are not 6,720 independent source facts.

## Operational-capacity results (N=1,2,4)

| Test split | NEW collision | P2 wrong UPDATE | P3 wrong UPDATE | TRUE UPDATE recall | Target-slot recall | Value replacement | Duplicate same-key slot | Overall action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TestA | 0.000% | 0.000% | 0.000% | 99.861% | 99.861% | 99.861% | 0.139% | 99.965% |
| TestB | 0.000% | 0.000% | 0.000% | 99.444% | 99.444% | 99.444% | 0.556% | 99.861% |

At N=4, NEW collision is 0.000% (TestA) and 0.000% (TestB). N=4 NEW events are decision-only in the simulator because the real 4-slot allocator is already full; no capacity/overflow behavior is claimed. N=64 is out-of-capacity stress only. Under the frozen rule its NEW collision is 0.000% / 0.000%, and its update target recall is 100.000% / 98.333% (TestA/TestB).

The direct true-update set contains 3360 holdout episodes with verified `old_value != new_value`. Correct target selection, replacing the stored value, removing the prior current value, no duplicate key, and unchanged slot count are simulated together. Value-replacement gate: **PASS**. Combined offline action gate: **PASS**.

## Slot competition and rule choice

- Validation rule status: A: PASS; B: PASS; C: PASS. Selected Rule A threshold `0.850429`; Rule B/C add complexity without changing the operational-capacity outcome.
- With the historical pairwise threshold 0.57, the NEW top-1 negative p99 rises from TestA `0.070` to `0.512`, and TestB `0.058` to `0.597` between N=1 and N=64. Historical-threshold collision also rises from 0.000% to 0.556% (TestA) and 0.000% to 1.667% (TestB). This confirms extreme-value accumulation through N=64, while the actual O3b capacity gate remains N≤4.
- On the frozen policy, every per-N action/value gate through N=64 passes in these sampled episodes; N>4 remains stress-only and does not extend the real memory capacity claim.

## Reconnect status

`NOT_RUN: frozen O3b dependencies missing`. The active remote has O21B and Candidate B, but lacks:

- `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/best_model.pt (frozen query encoder/value heads)`
- `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_vgen2/best.pt (frozen V/Core and value heads)`
- `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3a_allocation/allocator_kaware_best.pt (frozen free-slot allocator)`
- `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_update/router_best.pt (pre-reconnect baseline router for regression comparison)`

Consequently post-reconnect retrieval, full KV answer, regression vs baseline, and free-slot regression are **N/A / NOT RUN**. No architecture, model weights, allocator, Read path, or Value path was changed.

## Required answers

1. Candidate B fully frozen: **YES**; before/after hashes match.
2. NEW negative top-1 rises with N: **YES**; p99 increases substantially through N=64.
3. Multi-slot extreme-value accumulation: **CONFIRMED** under the historical threshold; frozen calibrated policy controls it in the measured range.
4. Validation choice: A/B/C all pass operational N; **Rule A** selected by simplicity priority.
5. Frozen rule: **Rule A**, `UPDATE iff top1 >= 0.8504285216331482`; otherwise NEW.
6. Threshold depends on N: **NO**.
7. Top1-top2 margin used: **NO**.
8. Holdout aggregate NEW collision, TestA/TestB: **0.000% / 0.000%** at N≤4.
9. Largest operational N=4 NEW collision: **0.000% / 0.000%**. N=64 stress-only frozen-rule result: **0.000% / 0.000%**.
10. P2 wrong UPDATE at N≤4: **0.000% / 0.000%**.
11. P3 wrong UPDATE at N≤4: **0.000% / 0.000%**.
12. TRUE UPDATE recall at N≤4: **99.861% / 99.444%**.
13. Correct target-slot recall at N≤4: **99.861% / 99.444%**.
14. Value replacement accuracy at N≤4: **99.861% / 99.444%**.
15. Duplicate same-key slot rate at N≤4: **0.139% / 0.556%**.
16. Offline action gate: **PASS**.
17. Real reconnect: **NO**; blocked by missing frozen O3b dependencies.
18. Post-reconnect retrieval: **NOT RUN**.
19. Post-reconnect full KV answer: **NOT RUN**.
20. Regression >1pp: **NOT EVALUABLE**; no reconnect.
21. Full O3b operational Address: **FAIL / NOT VERIFIED** until real reconnect and regression gate pass. Offline action/value behavior itself passes.
22. O3c allowed: **NO**.

Exactly one next experiment: **Restore the original frozen O3b checkpoint bundle on the active remote, then reconnect only the frozen Candidate B + Rule A decision to the unchanged write path and run retrieval/full-KV/free-slot regression on the same source-separated holdout. Do not retrain Candidate B or change any other module.**

## Reproducibility and artifacts

- Remote output: `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m2_growing_slot_calibration`
- Candidate B checkpoint was unchanged; no model/checkpoint was written or modified.
- Data leakage: no canonical source-row, feature-row, source-record, or exact semantic-triple overlap between the used validation and holdout episodes. TestA key-identity overlap with calibration is disclosed above; TestB is key-disjoint.
- Runtime: calibration 46.53s; holdout 10.85s, on CUDA. Peak VRAM not recorded.
- Git SHA: no Git repository at the workspace root.
- `artifact_sha256_manifest.json` records every delivered experiment artifact.
