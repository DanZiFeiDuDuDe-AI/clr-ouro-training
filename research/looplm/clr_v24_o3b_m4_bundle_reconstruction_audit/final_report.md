# O3b-M4 — Certified Baseline Integration Bundle Reconstruction Audit

## FINAL VERDICT

**O3b-M4-R — BASELINE BUNDLE RECONSTRUCTION REQUIRES RETRAINING**

The original integration contract is recoverable from exact source, but the bundle is not fully reconstructable without retraining: four required learned targets (O21A-MR query, VGen2 core/value heads, O3a allocator, and O3b baseline router) have no certified state. The upstream R2 initialization checkpoint is also absent. No shape-compatible foreign state was substituted.

## Frozen artifacts

- Candidate B checkpoint SHA-256: `43b9b6dc4c0baf3d85c37c6debe2ae7eaaf0baeb0636a50367a001024c0f94fc`; remote recheck matched.
- Candidate B config SHA-256: `17bf2445b14cfd0a701f28c3a671da2844d37dab4f4aadc91b240b14c4068114`.
- Rule A threshold: `0.8504285216331482`; frozen config SHA-256 `82c236fc6f098508aeed4d2bc7488b5edae8a287128f57f0a042ae16352d0f5c`.
- M2 final report SHA-256: `d45368dd6fb9d41fbf51dae2c0aba1655b46d30d17dc85eee1cc688421428abb`.
- M3 dependency manifest SHA-256: `35da2b54982ce01e98e4919af9daf4c1d9ae9252b53f3b15e8e2c02530573f56`.
- Original `a1_v24_o3b.py` SHA-256: `687e0c3024cbb04100c4fb06bdbec5af471a7fdb9c4cc9ccf3852aab6ca3ddfd`. Candidate B and Rule A were not changed.

## Component disposition

| Component | Kind | State found | No-retrain status |
|---|---|---|---|
| O21A-MR QueryEncoder | learned | missing | **RETRAINING REQUIRED** |
| VGen2 Core + MRValueHeads | learned | missing | **RETRAINING REQUIRED** |
| O3a K-aware allocator | learned | missing | **RETRAINING REQUIRED** |
| O3b baseline UpdateRouter + threshold | learned | missing | **RETRAINING REQUIRED** |
| O21B KeyEncoder | learned | exact SHA-matched checkpoint | recoverable |
| Ouro-1.4B source model | pretrained frozen | pinned artifact in J3C snapshot | recoverable; not loaded by O3b `load_all` when consuming the cached hidden vectors |
| L2, dot matcher, zero state, occupancy/action control | deterministic | exact source | deterministically reconstructable |

### Q1–Q21

1. **O21A-MR learned or deterministic?** Learned. Its QueryEncoder is used by O3b; O3b freezes it.
2. **Complete O21A-MR learned state found?** No.
3. **Can O21A-MR be restored without retraining?** No. O21B's same-shaped query is different provenance and prohibited as a substitute.
4. **VGen2 learned or deterministic?** Learned: trained Core state and learned per-slot value heads.
5. **Complete VGen2 state found?** No.
6. **Can VGen2 be restored without retraining?** No.
7. **O3a allocator learned or deterministic?** Learned K-aware MLP, 68→128→4.
8. **Does it actually require a checkpoint?** Yes; original O3b strict-loads the `allocator` state.
9. **Can O3a be restored without retraining?** No.
10. **Baseline router learned or deterministic?** Learned pair/action MLP.
11. **Complete router state found?** No.
12. **Can the baseline router be restored without retraining?** No.
13. **Was required state found nested in another bundle?** No certified target state. O21B contains its own QueryEncoder; Candidate B is a different head; neither is target state.
14. **Can we build the complete historical bundle contract?** Yes, at source/interface level. Exact VGen2 run identity/config and original baseline config/threshold remain unknown.
15. **Can we do an interface-only dry run?** Yes, static shape/control-flow contract PASS; no missing learned model was run or scored.
16. **Whole bundle status?** `PARTIALLY RECONSTRUCTABLE — RETRAINING REQUIRED`.
17. **Which components require retraining?** O21A-MR QueryEncoder; VGen2 Core and MRValueHeads; O3a K-aware allocator; original O3b UpdateRouter. The R2 initializer checkpoint is also missing and is an upstream source-loader/training dependency; see the qualification below.
18. **Are historical objectives/configs sufficient?** Partially. Source defaults/objectives are largely available for O21A-MR/O3a/router; exact run overrides/best steps/metrics are absent. VGen2-specific identity/config is incomplete because only `a1_v24_vgen.py` was found and its default path differs. R2's source architecture survives but its original training recipe/checkpoint does not. Therefore exact historical state reproduction is not supported.
19. **Baseline reproduction allowed?** No.
20. **Candidate B reconnect allowed?** No.
21. **O3c allowed?** No.

## Forensic scope and disposition

Read-only search covered local workspace/archive, local M2/M3 records, the active remote CLR project, the user's own remote home mirror/share, component code, checkpoint-like file extensions, configs/logs, MR cache files, and the identified remote temporary directory. The remote checkpoint inventory showed O21B and M2 Candidate A/B/C checkpoints only; the expected O21A-MR/VGen2/O3a/router directories or files were absent. The O21A-MR directory contains the train/validation/TestA/TestB cache and metadata, not model parameters. Other users' data was not inspected.

The O3b contract's deterministic interfaces are recoverable. The candidate M2 policy remains separately frozen; it is not part of the original baseline bundle and is not used as a substitute. No baseline score, retrieval/full-KV test, baseline reproduction, or reconnect was run.

## Forensic limitation

Remote tensor deserialization/signature inspection was unavailable because importing PyTorch on the remote host failed on missing `libgalaxyhip.so.5`. The scoped filename/extension inventory and exact expected-path checks found no target checkpoints; local O21B state and source-defined strict-load signatures were inspected. Therefore the missing-state conclusion is based on bounded file/path inventory, not successful remote tensor loading.

## Exactly one next experiment

**Reconstruct only the missing learned baseline components using their recovered historical objectives/configs and validate each against its historical metric anchor before assembling a new certified bundle.** Start at the most upstream missing dependency: recover the R2 Core checkpoint/recipe gate, then O21A-MR → VGen2 → O3a allocator → baseline router. Because numeric historical anchors for several components are missing, do not start training until those anchors are recovered or a new baseline acceptance protocol is explicitly authorized.

## Output and integrity

- Output directory: `H:\LoopLM\clr_v24_o3b_m4_bundle_reconstruction_audit`
- Key files: `historical_bundle_contract.md`, `component_recoverability_matrix.csv`, `artifact_forensics_inventory.csv`, `partial_state_recovery_audit.csv`, `source_only_reconstruction_audit.csv`, `retraining_requirements.md`, `bundle_interface_dryrun.json`.
- Git SHA: N/A (workspace is not a Git repository).
- Model/checkpoint modifications: none. Only original source files were copied read-only into this new audit folder.
- Data leakage: none; no evaluation was run.
- GPU/training: none.
