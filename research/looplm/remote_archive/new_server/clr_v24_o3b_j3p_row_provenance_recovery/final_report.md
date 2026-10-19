# O3b-J3P — Row Provenance Recovery and Join Certification

## FINAL VERDICT

**J3P-F — HISTORICAL ROW PROVENANCE CANNOT BE CERTIFIED**

This is a provenance gate failure—not a probe/basis-invariance failure and not a confirmed row misalignment. No probe was trained; no J revalidation, model load/forward, or feature re-extraction was run.

## Findings

- The exact J label traversal was recovered from `a1_v24_o3b_j.py::build_infos` and `joint_labels`, then checked against every MR flat entity/relation/class label. Source capacities, offsets and semantic metadata agree for train 11,665; val 2,332; TestA 2,332; TestB 2,332, with zero detected source/label mismatches.
- Stable source-based canonical IDs and SHA-256 record hashes were constructed for all 18,661 MR facts. The 64-row stratified semantic audit passed; entity/relation/class array checks were evaluated over all source rows. The J joint-label mapping is reproducible from sorted training `(person, relation_id)` keys.
- The remote O3b-H token cache is 838,539,950 bytes, SHA-256 `81fd14e8a9e2ddae1fedad504536fed5c51505a75cf9f4e050070e46230e3481`. Its feature shapes are train 4096×2048 and 2332×2048 for each evaluation split. The NPZ has no row/sample/fact ID or index-like key.
- J's preserved consumer pairs `raw[:n]` with `mention[:n]`. That records a downstream prefix convention only; it does not recover the historical H extraction loop or prove feature row *i* represents source fact *i*. The full train source/label count is 11,665 while the feature cache contains 4,096 rows.
- No O3b-H extraction source, tokenizer revision, mention-selection rule, per-row token/character span, or historical feature-order sidecar was found. Feature↔source duplicates, missing rows and orphans therefore cannot be tested; no actual row mismatch is confirmed.
- The local `remote_archive/cheems` file with the same NPZ name is a different 7,702,416-byte artifact (SHA-256 `f5286b7d609b48c2a8aca11532c5891acf851771d396bc5334ad6511e30251c5`) and fails ZIP/NPZ parsing (`BadZipFile`). It was not used; the remote cache above is the valid canonical cache. This is a local archive-copy integrity issue, not proof of row misalignment in the remote experiment.
- The source and label traversal are auditable, but they are not shown to have shared one original iterator: the H feature traversal is missing. The expected historical feature-order CSV and feature re-extraction audit are explicitly marked `NOT_RUN` rather than fabricated from the prefix assumption.

## Required questions

| # | Answer |
|---|---|
| Q1. O3b-H original extraction provenance found? | **No.** Cache/config/report survive; extractor source, row manifest and token-span provenance do not. |
| Q2. Historical feature traversal recovered exactly? | **No.** Only the J consumer's array-load/prefix behavior is preserved. |
| Q3. Historical label traversal recovered exactly? | **Yes.** J `build_infos` and `joint_labels`; verified against all MR source arrays. |
| Q4. Same source iterator/order? | **Unproven.** The H extraction iterator is unavailable. |
| Q5. Source-based canonical IDs built? | **Yes**, for all 18,661 facts; IDs use split/episode/item plus source-record hash, never feature-row index. Mention index is null. |
| Q6. Feature rows and MR facts bidirectionally 1:1 joined? | **Not certified.** Candidate prefix counts match, but there are no shared historical IDs. |
| Q7. Labels and MR facts joined? | **Yes for reconstructed J label order and source arrays.** Historical probe inputs have no saved per-row identity sidecar. |
| Q8. Per-split `Nsource=Nfeature=Nlabel`? | **No for full train** (11,665 vs 4,096 cached feature rows); the candidate scored prefix has equal counts of 4,096. Val/TestA/TestB each have 2,332, but equal counts do not prove identity. |
| Q9. Duplicate/missing/orphan rows? | **No source-ID duplicates or source/label omissions detected. Feature-side status is undeterminable; no misalignment is confirmed.** |
| Q10. Mention/token spans replayable? | **No.** Historical tokenizer and mention-selection semantics are missing. |
| Q11. Re-extracted H matches old cache? | **NOT RUN** because the strict join gate failed and extraction semantics are unknown. |
| Q12. Label semantics match source? | **Yes for source/reconstructed labels:** entity/relation/class arrays checked over all rows; 64 sampled texts passed; joint labels follow the recovered J mapping. Feature-row semantics remain unverified. |
| Q13. Historical J/L/L2 row-level provenance? | **NOT CERTIFIED** (not misalignment-confirmed). |
| Q14. Measurement gate reopened? | **No.** |
| Q15. J3 converged probe calibration allowed? | **No.** |
| Q16. Reconnect allowed? | **No.** |
| Q17. O3c allowed? | **No.** |

## Frozen-artifact checks

- Checkpoint `b1_best.pt`, before and after: `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570` (identical).
- O3b-L2 feature cache: SHA-256 `46676516632aa6120bc435713723521659f9fb102cc244755f09b797f95c3b4d`.
- Local archived H-cache copy: malformed NPZ, SHA-256 `f5286b7d609b48c2a8aca11532c5891acf851771d396bc5334ad6511e30251c5`; retained unchanged and excluded from analysis.
- The starting inventory records SHA-256, size and modification time for 409 local/remote artifact entries; a 220-entry remote-only snapshot is retained separately.
- No checkpoint/model/architecture changes, no model forward, no probe training, and no newly introduced data leakage.

## Interpretation and one next experiment

The source-label side is recoverable and internally consistent; the unresolved link is specifically old mention/token feature rows to semantic source rows. Historical probe results remain numerically reproducible but lack certifiable row-level semantic provenance; this does not show that the results are wrong.

**Exactly one next experiment:** Build a fresh ID-carrying feature cache from original MR sources and the frozen checkpoint, then certify that cache before any probe training.

Full generated row manifests, sidecars, joins, and machine-readable certification are on the remote under `/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3p_row_provenance_recovery/`.
