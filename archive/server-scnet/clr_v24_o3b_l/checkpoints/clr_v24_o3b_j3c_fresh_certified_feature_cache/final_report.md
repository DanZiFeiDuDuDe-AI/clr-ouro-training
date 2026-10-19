# CLR v2.4 / O3b-J3C — Fresh ID-Carrying Feature Cache Certification

## Verdict

`J3C-P — FRESH ID-CARRYING FEATURE CACHE PROVENANCE CERTIFIED`

This is a newly constructed cache with source-based IDs. It does not retroactively certify historical O3b-H/J/L/L2 rows; those remain numerically reproducible but not row-level provenance-certified. No probe was trained.

## Frozen measurement protocol

- Policy SHA-256: `b53182cce202549cf3f65f2de27e58a8c7dc43ada7965749832337eb60ed3e67`; frozen before the first Ouro forward (freeze log records zero forwards).
- Extraction code SHA-256: `46c71a7d4ad502caa45b00ddebc1eee961495f753cf9f0aa0a03f3b089951cb5`.
- Extraction wall time: `205.5` seconds on `BW`.
- Ouro checkpoint SHA-256 before/after: `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` / `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
- KeyEncoder checkpoint SHA-256 before/after: `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570` / `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570`.
- H_mention is mean-pooled from the final UT pass (index 3 of 4), after RMSNorm, over the uniquely matched entity mention tokens. This is a frozen backbone hidden state, not a memory write state.
- Primary J input is fresh I2b = source fact_H joined by source fact_offsets + 2 × fresh mention state. I0, I2a, and both components are also retained with identical IDs.
- Mention matching is exact/case-sensitive with no normalization, all occurrences counted; non-unique or missing mentions are explicitly excluded. No truncation is performed; >512-token inputs are excluded.
- Train, validation, TestA and TestB all follow the same pre-frozen rules. Eval splits did not determine any extraction rule.

## Per-split source and mention resolution

- train: source=11665, accepted=11665, no_match=0, multiple_matches=0, too_long=0, token_span_unmapped=0, token_span_noncontiguous=0
- val: source=2332, accepted=2332, no_match=0, multiple_matches=0, too_long=0, token_span_unmapped=0, token_span_noncontiguous=0
- testA: source=2332, accepted=2332, no_match=0, multiple_matches=0, too_long=0, token_span_unmapped=0, token_span_noncontiguous=0
- testB: source=2332, accepted=2332, no_match=0, multiple_matches=0, too_long=0, token_span_unmapped=0, token_span_noncontiguous=0

## Required provenance questions

1. Policy frozen before any model forward: **YES**.
2. Canonical IDs derive from split/episode/item/fact semantics, never tensor row index: **YES**.
3. Every accepted row has exactly one deterministic character/token mention: **YES (18661 rows)**.
4. Missing/multiple/too-long/unmappable rows are explicitly counted above and excluded, never silently selected or truncated.
5. Fresh train source facts / accepted mention rows: **11665 / 11665**.
6. Accepted val/TestA/TestB rows: **2332 / 2332 / 2332**.
7. H row ↔ source fact: **PASS**, one accepted mention per fact via feature_row_id.
8. All J stages preserve row identity/order: **PASS**.
9. R0/R1/R2 preserve row identity/order: **PASS**.
10. Labels join through feature_row_id: **PASS** (bidirectional join audit, not position-only assumption).
11. Duplicate/missing/orphan/mismatched feature rows: **0 failed rows; duplicate IDs=0**.
12. Deterministic re-extraction and re-tokenization: **PASS**, tokenization identity checks=256, sampled feature rows=256; H_fresh max abs=0.0, max mean abs=0.0, min cosine=0.9999999403953552; predeclared bf16 tolerance is in `numerical_reextraction_summary.json`.
13. Checkpoint hashes unchanged: **YES**.
14. Fresh cache row-level provenance certified: **CERTIFIED**.
15. Measurement gate may reopen: **YES, for the fresh cache only**.
16. J3 converged matched linear-probe calibration may start: **YES, beginning only with R0/R1 and invertible R1/R2 calibration gates**.
17. Reconnect allowed: **NO**.
18. O3c allowed: **NO**.

## Artifacts and validation

`fresh_row_join_audit.csv`, `fresh_row_join_summary.json`, `numerical_reextraction_audit.csv`, `numerical_reextraction_summary.json`, `checkpoint_sha256_before_after.json`, and `fresh_cache_hash_manifest.json` carry row-level, numeric, checkpoint, and storage evidence. All saved NumPy arrays are reloaded with `allow_pickle=False` and checked for shape, float32 dtype, and finite values before certification.
No model/checkpoint was modified. No probe training, reconnect, O3c, or historical-cache mutation occurred.

## Exactly one next experiment

Resume J3 converged matched linear-probe calibration on the fresh certified cache, beginning with R0/R1 and invertible R1/R2 controls. Do not run stagewise J attribution until the calibration gate passes.
