# CLR-O3-REBASE-1 — Fresh Operational Latent-Slot Baseline

## Final verdict

**O3-REBASE1-F — VGEN2 REBUILD FAILS OPERATIONAL GATE**

The new baseline contract and dependency DAG were frozen before training. O21A K/Q rebase passed its standalone retrieval gate. The fresh VGen2-equivalent value path did not pass its pre-registered validation gate after the fixed 3,000-step run, so the stop rule was applied: no TestA/TestB value evaluation, O3a training, router integration, memory integration, or full episode evaluation was performed.

Prior J/K/L/J3 attribution artifacts and old baseline-recovery records were left untouched and were not used to block or tune this operational rebase.

## Frozen baseline contract

- Capacity: four latent slots; no scale-up.
- Retrieval address: O21A-rebase normalized K/Q, 64D, dot product divided by 0.1 over occupied slots.
- Update address: frozen Candidate B factorized entity/relation encoders and symmetric pair scorer; factors are 64D each.
- Rule A threshold: `0.8504285216331482`, unchanged.
- Value state: 64D; source-shaped `Linear(2048,64) + GRUCell(64,64)` and four slot-specific `Linear(64,64)` value heads.
- NEW/UPDATE/READ routing is specified as deterministic logic; no learned router.
- Exact interfaces and dependency order: [o3_rebase_contract.json](H:/LoopLM/clr_o3_rebase1_fresh_operational_baseline/o3_rebase_contract.json) and [component_dependency_dag.json](H:/LoopLM/clr_o3_rebase1_fresh_operational_baseline/component_dependency_dag.json).

## Component results

### O21A rebase — PASS

Trained from fresh initialization on the fixed MR cache with the recovered O21A-MR objective: retrieval CE + 0.5 batch-positive alignment CE + 0.1 fact/query entity CE; AdamW 1e-3, weight decay 0.01, 3,000 steps, batch 128, seed 42. Checkpoint selection used validation only; best step was 1,100.

- TestA retrieval: N=1/2/4 = 100% / 100% / 100%; same-entity/different-relation N=4 = 100%; different-entity/same-relation N=4 = 100%.
- TestB retrieval: N=1/2/4 = 100% / 99.70% / 100%; both N=4 stress subsets = 100%.
- Rebuilt weights: [o21a_rebase.pt](H:/LoopLM/clr_o3_rebase1_fresh_operational_baseline/components/o21a/o21a_rebase.pt).

This validates the standalone query-to-key retrieval component, not the integrated memory READ path.

### VGen2 rebase — FAIL; stop here

The historical R2/VGen weights were not substituted. A fresh source-shaped Core and value heads were trained for 3,000 fixed steps (AdamW 3e-4, weight decay 0.01, clip 1.0, seed 42). Training alternated source-separated multi-slot episodes and same-key/different-value two-write episodes. Model selection used only validation.

At the selected checkpoint (step 3,000):

- Validation single-write answer: overall 83.53%; N=4 84.38% (required: 95%).
- Validation after same-slot changed-value update: latest-value answer 87.03% (required: 95%).
- Earlier value after its initial write: 85.51%.
- Non-target slot mutation: none.
- TestA/TestB: **NOT RUN**, because validation failed.

The checkpoint and full curve are retained for diagnosis: [vgen2_rebase.pt](H:/LoopLM/clr_o3_rebase1_fresh_operational_baseline/components/vgen2/vgen2_rebase.pt), [learning curve](H:/LoopLM/clr_o3_rebase1_fresh_operational_baseline/components/vgen2/vgen2_rebase_learning_curve.csv), [gate](H:/LoopLM/clr_o3_rebase1_fresh_operational_baseline/components/vgen2/vgen2_rebase_gate.json).

## Remaining components and integration

- Candidate B remains frozen; local and remote checkpoint SHA-256 both match `43b9b6dc4c0baf3d85c37c6debe2ae7eaaf0baeb0636a50367a001024c0f94fc`.
- Rule A threshold remains exactly `0.8504285216331482`; no recalibration occurred.
- O3a allocator training: **NOT RUN**.
- Deterministic router: contract/config only; not integrated.
- Basic WRITE, actual TRUE UPDATE, P2/P3/P4 action tests, queried READ in slot memory, latest-value READ, full KV answer, multi-step episodes: **NOT RUN**.
- New frozen bundle: **NOT CREATED**.

## Input integrity and leakage

MR `fact_H` and J3C `H_base_I0` were bitwise equal in all four splits, with matching row counts (train 11,665; val/TestA/TestB 2,332 each). Episode IDs had zero cross-split overlap. The frozen Candidate B artifact and its certified source-row mapping remain unchanged; its embedded train-only statistics were not refit. Candidate B was not run in an integrated episode because the value-path stop gate failed. O21A TestA/TestB were evaluated only after checkpoint selection and the validation gate. VGen2 training and selection used train/validation only; TestA/TestB were not opened. No data leakage was detected.

## Required questions

1. **New contract?** Four slots; retrieval K/Q=64D; Candidate B entity/relation factors=64D each; V=64D; deterministic NEW/UPDATE/READ router. Contract JSON is frozen.
2. **Components requiring new learned weights?** O21A K/Q and VGen2 Core/value heads; O3a allocator would also require training, but was not started. Candidate B is frozen.
3. **O21A rebase?** PASS.
4. **VGen2 rebase?** FAIL standalone validation gate.
5. **O3a allocator?** NOT RUN due stop rule.
6. **Candidate B frozen?** YES, SHA verified.
7. **Rule A threshold unchanged?** YES, `0.8504285216331482`.
8. **Basic WRITE?** NOT RUN.
9. **Integrated TRUE UPDATE?** NOT RUN; standalone value update validation was 87.03% and failed.
10. **P2/P3?** NOT RUN in the new integrated baseline. The old Candidate B action result is not substituted for rebase integration evidence.
11. **Free-slot allocation?** NOT RUN.
12. **Queried READ?** O21A standalone retrieval PASS; integrated queried READ NOT RUN.
13. **Latest-value READ after UPDATE?** Standalone validation 87.03%; integrated read NOT RUN; gate failed.
14. **Full KV answer?** NOT EVALUATED.
15. **Multi-step episode?** NOT RUN.
16. **Address operationally closed?** NO.
17. **New reproducible baseline bundle?** NO; bundle not created/frozen.
18. **Proceed to Read/Write/Update scale/generalization?** NO; first resolve VGen2 standalone value accuracy.

## Exactly one next experiment

Run one pre-registered **10,000-total-step VGen2 rebase** from the same seed and initialization policy, with the same architecture, data, update/episode mixture, optimizer, and learning rate; select only on validation and keep TestA/TestB unopened until both validation gates pass. (The current artifact does not save optimizer state, so rerun from step zero rather than claiming an exact optimizer-state continuation.) The 3,000-step validation curve was still improving, so this isolates under-training without architecture search. Do not train O3a or integrate memory during that experiment.

## Reproducibility and files

- Output: `H:\LoopLM\clr_o3_rebase1_fresh_operational_baseline`
- Contract/DAG, component configs/checkpoints, learning curves, logs, candidate integrity, NOT_RUN artifacts, and SHA-256 manifest are included.
- Git SHA: N/A (workspace is not a Git repository).
- Existing model/checkpoints modified: **No**. New diagnostic O21A and failed VGen2 checkpoints were written only under this fresh output tree.
- GPU: remote BW accelerator, 64 GiB VRAM; total stage wall time is recorded in each component runtime JSON.
