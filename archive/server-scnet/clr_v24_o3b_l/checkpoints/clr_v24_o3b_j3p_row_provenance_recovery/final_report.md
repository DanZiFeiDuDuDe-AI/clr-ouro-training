# O3b-J3P — Row Provenance Recovery and Join Certification

## FINAL VERDICT

**J3P-F — HISTORICAL ROW PROVENANCE CANNOT BE CERTIFIED**

This is a provenance gate failure, not evidence of a probe/basis failure and not a confirmed row misalignment. No probe was trained; no J revalidation, model load/forward, or feature re-extraction was run.

## Result summary

- The exact J label traversal was recovered and independently checked across all source facts: train 11,665; val 2,332; testA 2,332; testB 2,332. Source capacities, `fact_offsets`, semantic metadata and flat entity/relation/class labels agree with zero structural/label mismatches.
- A deterministic source-based ID and SHA-256 source-record hash were produced for each of the 18,661 source facts. A 64-row split-stratified semantic audit passed all checks; the full source-to-flat-array entity/relation/class checks also passed.
- The remote O3b-H feature archive is `81fd14e8a9e2ddae1fedad504536fed5c51505a75cf9f4e050070e46230e3481` (838,539,950 bytes), with shapes train 4096×2048 and each evaluation split 2332×2048. It contains no ID/index-like keys. The train feature cap is 4,096 while the full train source/label sequence is 11,665.
- O3b-J's consumer preserves the NPZ order and pairs arrays using `[:n]`; this only defines a candidate prefix join. It does not recover the historical H extraction traversal or prove that feature row *i* came from source fact *i*.
- No mention span/token indices, tokenizer name/revision, mention match/selection rule, or H extraction source/row manifest was recovered. Therefore strict bidirectional feature↔source identity, orphan/duplicate status on the feature side, and numerical re-extraction certification remain untestable. No actual mismatch was observed.
- The local `remote_archive/cheems` file with the same NPZ name is a different 7,702,416-byte artifact (SHA-256 `f5286b7d609b48c2a8aca11532c5891acf851771d396bc5334ad6511e30251c5`) and fails ZIP/NPZ parsing (`BadZipFile`). It was not used; the remote cache above is the valid canonical cache. This is a local archive-copy integrity issue, not proof of row misalignment in the remote experiment.

## Q1–Q17

| Question | Answer |
|---|---|
| Q1. O3b-H original extraction provenance found? | **No.** The cache/config/report exist; extractor source, row manifest, tokenizer revision and per-row mention spans do not. |
| Q2. Historical feature traversal recovered exactly? | **No.** Only the J downstream array-load/prefix operation is preserved. |
| Q3. Historical label traversal recovered exactly? | **Yes.** `a1_v24_o3b_j.py::build_infos` and `joint_labels`; verified against all MR flat label arrays. |
| Q4. Same historical source iterator/order? | **Unproven.** Feature extractor loop is missing; label loop is recoverable. |
| Q5. Source-based canonical IDs built? | **Yes**, 18,661 source records; IDs use split/episode/item plus a stable source-record hash, never feature row number. Mention index remains null because not recoverable. |
| Q6. Feature↔MR bidirectional 1:1 join? | **Not certified.** Prefix counts match for the cached rows, but no shared IDs. |
| Q7. Labels↔MR facts 1:1 join? | **Yes for reconstructed J label order and source arrays**; the separate historical probe-label input used no saved ID sidecar. |
| Q8. Per-split Nsource=Nfeature=Nlabel? | **No for full train** (11,665 vs 4,096 cached features); candidate scored-prefix counts are 4,096 each. Val/TestA/TestB counts are 2,332 each, but count equality is not identity proof. |
| Q9. Duplicate/missing/orphan rows? | **No source-ID duplicates or source/label omissions detected. Feature-side duplicates/orphans are undeterminable without IDs; no row misalignment is confirmed.** |
| Q10. Mention/token spans replayable? | **No.** Historical tokenizer and mention-selection semantics are absent. |
| Q11. Re-extracted H matches old cache? | **NOT RUN** because the strict join gate failed and extraction semantics are unknown. |
| Q12. Entity/relation/joint semantics match source? | **Source and reconstructed labels pass.** Entity/relation/class arrays were checked over every row; 64 sampled source texts and the J joint-label mapping passed. There is no old feature-row identity to extend this to feature↔label semantics. |
| Q13. Historical J/L/L2 semantic provenance? | **NOT CERTIFIED** (not misalignment-confirmed). |
| Q14. Measurement gate reopened? | **No.** |
| Q15. J3 converged probe calibration allowed now? | **No.** |
| Q16. Reconnect allowed? | **No.** |
| Q17. O3c allowed? | **No.** |

## Frozen artifact checks

- Checkpoint before: `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570`
- Checkpoint after: `6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570` (identical; no parameter/checkpoint modification)
- Remote H token cache: SHA-256 `81fd14e8a9e2ddae1fedad504536fed5c51505a75cf9f4e050070e46230e3481`
- Remote L2 feature cache: SHA-256 `46676516632aa6120bc435713723521659f9fb102cc244755f09b797f95c3b4d`
- Local archived H-cache copy: malformed NPZ, SHA-256 `f5286b7d609b48c2a8aca11532c5891acf851771d396bc5334ad6511e30251c5`; retained unchanged and excluded from analysis.
- Full start-of-run inventory (SHA-256, size, mtime) is `artifact_hash_manifest.json`; the remote-only starting snapshot is retained as `artifact_hash_manifest_remote.json`.
- No data leakage: **none introduced**. Train/test mapping was not tuned; mapping rules were recovered from source code before audit. No probe scores were run or consulted.

## Interpretation and next step

The source-label side is recoverable and internally consistent; the unresolved break is specifically between the old mention/token feature array and semantic source rows. Thus previous probe numbers remain numerically reproducible but lack certifiable row-level semantic provenance; they are not thereby shown to be wrong. Keep the measurement gate closed.

**Exactly one next experiment:** Build a fresh ID-carrying feature cache from original MR sources and the frozen checkpoint, then certify that cache before any probe training.
