# O3-REBASE2 VGen2 failure localization

**O3-REBASE2-INCOMPLETE — ROOT CAUSE NOT YET IDENTIFIED**

Current 3K shortfall can be localized using paired rebase checkpoints. The exact historical passing VGen2 implementation/checkpoint/dataset/config/commit is not authenticated, so historical regression attribution remains incomplete. Longer training is not a repaired implementation defect; PASS-A is not claimed. Candidate legacy-source drift alone cannot prove FAIL-CONTRACT for an unidentified passing run.

## Baseline correction and measured attribution

The pasted O3-REBASE1-F freeze is superseded by an archived 10K standalone passing run. All 18 archived checkpoints were re-downloaded from the current server and checked against their original SHA-256 manifest. This diagnosis does not authorize downstream work.

The original failing 3K checkpoint and the 10K run's step-3000 snapshot have **all core/head tensors exactly equal=True**, max absolute difference 0. This controls initialization and the first 3000 optimization updates: the later passing checkpoint is obtained by further optimization in the same training trajectory, not an input/label substitution.

| Checkpoint | Step | Single-write overall | N=4 | UPDATE latest |
|---|---:|---:|---:|---:|
| 3k | 3000 | 83.23% | 84.46% | 87.03% |
| 10k | 9600 | 97.08% | 97.15% | 97.94% |

3K checkpoint SHA-256: `f7ef1bf210981de8215fce831cb85ca23ee8f1e32cffd21d68ee6a658e8dd8fd`.
10K selected checkpoint SHA-256: `ddb31622105ce646efb3d0ea4ef1bb2523d93a30b023d6a1b27415b0d775efbb`.
3K config SHA-256: `5be7a10274df9b1ae7689f71cc1bb3726fd03cdf605303988b06fa434b576b8b`; 10K config SHA-256: `8fc1168bc4127d23c4b2498c91de0097499a5f40175cca1e1ef50ac3774a1409`.
Train MR dataset SHA-256: `612f387f326d1c4cb1490fdc3eac9bb62d502315a81d203f7407597d56a65e23`.
VAL MR dataset SHA-256: `967c7b7bf761d8baeeb900a1bc1413013d1825a7facee6555263a5c73fbb6ceb`.

Identical frozen validation source rows and labels are replayed through both checkpoints. Same deterministic physical slot assignments are used for the paired per-example representation capture; the standalone metric table also uses a fixed validation permutation. Small differences from archived 3K metrics can therefore reflect only a changed evaluation permutation, not a changed dataset.

## Lineage / contract audit

A true historical passing VGen2 cannot currently be recovered. O3 source loaders name `clr_v24_vgen2/best.pt`, while recovered `a1_v24_vgen.py` writes staged `clr_v24_vgen/t*_best.pt`. The latter is a candidate related source, not authenticated passing lineage. The missing R2 initialization and VGen2 artifact were already documented in M4; this round checks the active project again.

The recovered legacy source uses pretrained R2 initialization and LS0 single-relation replay. Rebase explicitly uses fresh initialization and episode/UPDATE supervision. Historical queried-answer macro evaluation also differs from current all-written-values and latest-overwrite gates. Thus a numeric historical score must not be treated as a like-for-like 3K rebase target. Exact historical cache identities, label order, run overrides and commit remain unknown.

`contract_diff.json` and `.csv` cover input source, shapes/dtypes, normalization, extraction, carriers, labels/vocabulary, masks, negatives, N, write/update mixture, splits, seed, optimizer/LR/batch/scheduler/steps, initialization, frozen components, and selection. Unknown historical evidence is marked explicitly.

Extraction qualification: the copied J3C policy describes the **fresh entity-mention branch** and composite `H_fresh=H_base_I0+2*H_mention_fresh`. VGen2 consumes **H_base_I0 / MR fact_H**, not that composite. The MR cache manifest only certifies frozen Ouro, float32, 2048 dimensions and cache schema; the original fact_H layer/token selection and builder commit remain unrecorded. Do not silently assign the entity-mention pooling policy to the value source. `lineage/mr_manifest.json` and scoped source hashes retain this distinction.

Current train/VAL canonical fact overlap: 0. Value mapping conflicts: 0. R0/R1 bitwise equality: True. Rowwise MR/certified carrier equality and label/sidecar ID joins pass. Canonical IDs include split prefixes, so zero ID overlap alone does not prove source separation. The supplemental source-text audit finds **7 shared exact fact texts / 2332 VAL fact rows**, and 22 shared semantic triples. These small duplicate-source overlaps are disclosed; the frozen VAL set is not altered for primary metrics.

Excluding exact train-text duplicates from reporting only, raw linear single-write=96.65%, UPDATE=96.42%. Train/VAL query-target offset mismatches and metadata fact-class mismatches are all zero. See `source_audit/source_split_audit.json` for array shapes/dtypes and denominators. Historical contamination cannot be ruled out for missing historical caches. The backbone was not re-extracted.

## Raw-carrier supervised diagnostic probes

| Probe | Value identity incl. UPDATE contexts | N=1 | N=2 | N=4 | Single-write | UPDATE latest |
|---|---:|---:|---:|---:|---:|---:|
| centroid | 7.48% | 8.68% | 6.01% | 8.11% | 7.59% | 7.25% |
| linear | 96.58% | 95.51% | 97.30% | 96.62% | 96.66% | 96.42% |
| mlp | 92.64% | 91.62% | 93.09% | 92.27% | 92.41% | 93.11% |

All fits use train only; VAL is evaluated once at the fixed final step. Linear/MLP use train-only feature means/scales, AdamW LR 0.003, WD 0.01, seed 42, batch 256 (half single-write, half UPDATE contexts), 2000 steps. MLP hidden width is 64. Centroids fit cosine means on train only. UPDATE raw input is the latest incoming fact; raw probes do not emulate a recurrent overwrite cell. A failed finite-capacity probe is not proof of information-theoretic insufficiency.

## R0–R4 information preservation

| Snapshot | Stage | Single-write probe | UPDATE probe | Within-value cosine | Between-value cosine | Pair AUC |
|---|---|---:|---:|---:|---:|---:|
| 10k | R3 | 96.66% | 96.69% | 0.7704 | 0.0246 | 0.9963 |
| 3k | R0 | 92.41% | 93.11% | 0.7943 | 0.7818 | 0.5346 |
| 3k | R1 | 92.41% | 93.11% | 0.7943 | 0.7818 | 0.5346 |
| 3k | R2 | 90.44% | 91.06% | 0.9508 | 0.8818 | 0.9259 |
| 3k | R3 | 87.44% | 87.92% | 0.7326 | 0.0359 | 0.9802 |
| 3k | R4 | 84.22% | 87.21% | 0.8103 | 0.0097 | 0.9691 |

Largest measured MLP readout drop: single-write **R3 -> R4 (3.22pp)**; UPDATE **R2 -> R3 (3.13pp)**. Raw linear exceeds 95%, while the same-budget raw MLP is lower; therefore finite-probe optimization/generalization is a real confound. These stage differences localize degraded readout but do not prove irreversible information loss. The 10K R3 probe recovers the operational region without altering source inputs or architecture.

R0 = certified cached source state; R1 = float32 core input; R2 = content_read linear output; R3 = GRU value state (including same-slot UPDATE history); R4 = selected slot decoder logits. The same examples are saved in `prepare/3k_val.npz` and `10k_val.npz`; train representations remain SHA-manifested on the server. Identical two-layer MLP architecture family (hidden width 64, 64 outputs, same optimizer/budget) is used at every stage; input dimensions differ (2048 vs 64), so parameter totals are reported and are not falsely claimed equal. Geometry uses frozen validation representations and fixed sampled non-self pairs; these cosine statistics are descriptive, not an optimal separability bound.

## Paired replay and failure taxonomy

3K failing vs 10K passing: source input equal=True; labels equal=True; passing-correct/failing-wrong=452; both wrong=80; failing-correct/passing-wrong=13.
Maximum projection difference=5.212073; maximum latent difference=1.989275. Changed learned transformation, not changed input/label, explains these paired differences. This comparison is not substituted for unavailable historical checkpoint replay.

Full entity/relation/value and overwrite-versus-single-write taxonomy is in `prepare/paired_replay.json`; every paired prediction is in `paired_replay.csv`. The largest error classes should be read as frequencies with denominators, not presumed causal defects.

## Optimization sanity

A raw supervised probe clears the required information gate (linear can establish sufficiency even when the MLP generalizes less well), so fixed 2000-step overfit diagnostics run from the failing 3K checkpoint, using 256 or 1024 unique train facts and all four physical heads. This is a single-write optimization diagnostic; UPDATE is evaluated on frozen VAL, not fit in these runs. LR controls at 0.25x / 1x / 4x are diagnostic contrasts, never selected for production.

| Unique train facts | LR multiplier | Train accuracy across slots | VAL overall | VAL UPDATE |
|---:|---:|---:|---:|---:|
| 1024 | 1.0 | 100.00% | 86.54% | 87.48% |
| 256 | 0.25 | 100.00% | 74.70% | 76.03% |
| 256 | 1.0 | 100.00% | 70.45% | 72.99% |
| 256 | 4.0 | 100.00% | 67.50% | 70.84% |

Per-run curves, gradient norms before clipping, class histograms, prediction entropy and final confusion matrices are saved with each checkpoint. No VGen production architecture, Candidate B, Rule A, or upstream state is changed.

## Verdict and next permitted step

The current rebase carrier and failing learned transformation can be diagnosed from the measurements above; the archived longer run already demonstrates that current inputs support the operational gate. The 3K experiment is a fresh-baseline training-budget failure, not evidence that the carrier cannot encode values. However, no implementation defect has been localized and repaired, and the true historical passing contract has not been authenticated. Therefore strict PASS-A/FAIL-REP/FAIL-CONTRACT requirements are not claimed; the overall historical-root-cause task remains INCOMPLETE.

Permitted: recover the exact historical `clr_v24_vgen2/best.pt`, its training source/config and R2 initialization/cache artifacts, then replay the same train/VAL rows under that authenticated contract; or explicitly accept the SHA-pinned 10K fresh rebase as the new standalone baseline in a later instruction. Prohibited: allocator, router/memory reconnect, integrated O3 evaluation, TestA/TestB tuning, changing Candidate B/Rule A, or redesigning the architecture to mask attribution.

## Reproducibility and downloads

Git SHA: **N/A**, the local workspace is not a Git repository and no historical commit was recovered. Do not invent a commit ID. Dataset, config, source and checkpoint content hashes are in `prepare/input_audit.json`, `source_config_hashes.json`, the original checkpoint manifest verification, and each run manifest. Current server: `ssh.zzai.scnet.cn:11810`, root; runtime torch 2.9.0 / BW GPU with the pinned DTK library path. Password is never included in experiment artifacts.

Each training task runs through `tools/train_and_pull.py`: close its SSH connection, reconnect, download checkpoints/config/metrics, validate server SHA-256, atomically promote downloads, and save `download_verification.json` before the next training task. Existing differing local weights are preserved. Project `AGENTS.md` records this policy for future training turns.

Resume existing artifacts: run local `run_diagnostics.py`; completed task markers allow artifact recovery without silent retraining. For a fresh reproduction, use a separate experiment output root in the scripts, preserve source MR/J3C and checkpoint inputs by SHA-256, run `prepare`, diagnostic tasks, and `audit_source_split.py`, then mirror results. Do not overwrite this frozen run. `build_report.py` verifies lineage and regenerates the report.
