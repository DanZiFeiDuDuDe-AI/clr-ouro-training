# O3-REBASE1-V2 — VGen2 Fixed 10K Rebuild Diagnostic

## Final verdict

**O3-REBASE1-V2-P — VGEN2 10K REBUILD PASSES OPERATIONAL VALUE GATE**

Outcome classification: **VALIDATION PASS**.

## Frozen run and integrity

- Training config matches the archived 3K core config: **YES**; fresh initialization at step 0; exactly 10,000 updates.
- Selected checkpoint step: 9600; selection was validation-only with priority UPDATE latest, N4, overall, then lower validation loss.
- Train/validation source overlap: 0 canonical fact IDs; TestA/TestB opened before gate: **NO**.
- O21A frozen checkpoint verified: `dadf66a59c8722ef86604aff67528f415acb18f3e72ba4ccfac415df37e90ced`; O21A/Candidate B/Rule A unchanged.
- O3a allocator: NOT RUN. Memory/router integration: NOT RUN.

## Validation trajectory

| Metric | 3K | 5K | 7K | 8K | 9K | 10K | 8K→10K (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-write overall | 83.53% | 92.80% | 95.71% | 96.36% | 97.04% | 97.38% | +1.03pp |
| Single-write N=4 | 84.38% | 93.24% | 96.17% | 96.47% | 97.75% | 97.60% | +1.13pp |
| UPDATE latest value | 87.03% | 94.10% | 95.62% | 96.42% | 97.50% | 97.58% | +1.16pp |

## Selected validation checkpoint

- Single-write overall: 97.17%
- N=1 / N=2 / N=4: 97.01% / 97.45% / 97.07%
- UPDATE old-value retention: 97.05%
- UPDATE latest-value: 97.94%
- Validation loss (50/50 episode/update): 0.104402
- Validation gate (overall≥95%, N4≥95%, update latest≥95%): PASS.
- Validation TestA/TestB status: testA: overall=0.9704, N4=0.9707, update=0.9825; testB: overall=0.9511, N4=0.9565, update=0.9565.

## Diagnostic interpretation

The 8K→10K changes are +1.03pp overall, +1.13pp N=4, and +1.16pp UPDATE latest. Classification follows these frozen trajectory values; it does not imply an information-theoretic claim.
No architecture, optimizer, learning rate, loss, preprocessing, feature source, backbone, or checkpoint was changed.

## Test policy and next step

TestA/TestB: **testA: overall=0.9704, N4=0.9707, update=0.9825; testB: overall=0.9511, N4=0.9565, update=0.9565**.
One next experiment: Freeze VGen2 and train/rebuild the O3a free-slot allocator under the frozen O3 Rebase contract.

## Artifact index

- `vgen2_10k_frozen_config.json` — frozen rebuild config.
- `training_curve.csv`, `validation_curve.csv`, `checkpoint_metrics.csv` — full per-100-step trajectory.
- `single_write_breakdown.csv`, `update_breakdown.csv`, `error_breakdown.csv` — capacity/update/value-class breakdowns.
- `best_checkpoint_selection.json`, `vgen2_10k_gate.json`, `FINAL_VERDICT.txt` — machine-readable selection and gate.
- Formal best component checkpoint: `vgen2_rebase_10k_best.pt` (written and SHA-manifested).
- `test_results.csv` — post-gate TestA/TestB results (or NOT_RUN if validation fails).

## Required questions (Q1–Q17)

1. 10K config matches the 3K training setup: **YES** for architecture/data/features/loss/seed/optimizer/LR/batching/preprocessing; only total steps changed. The mandated validation-only selection priority and fixed eval permutation are recorded separately.
2. 3K level reproduced: **YES** — overall 83.53%, N4 84.38%, UPDATE latest 87.03%.
3. Single-write overall: 3K 83.53% → 5K 92.80% → 7K 95.71% → 10K 97.38%.
4. N4: 3K 84.38% → 5K 93.24% → 7K 96.17% → 10K 97.60%.
5. UPDATE latest: 3K 87.03% → 5K 94.10% → 7K 95.62% → 10K 97.58%.
6. The last 2K remained upward: overall +1.03pp, N4 +1.13pp, UPDATE +1.16pp; validation loss continued down.
7. Validation plateau: **NO clear plateau by 10K**; the three gate metrics still trended upward across 8K–10K, with checkpoint-to-checkpoint noise.
8. Best checkpoint step: **9600**.
9. Best validation single-write overall: **97.17%**.
10. Best validation N4: **97.07%**.
11. Best validation UPDATE latest: **97.94%**.
12. Validation gate: **PASS**.
13. Failure category: **NOT APPLICABLE — PASS**. The 3K checkpoint was substantially undertrained relative to the 10K trajectory.
14. TestA/TestB run: **YES**, only after the validation gate passed.
15. TestA N4 / UPDATE: **97.07% / 98.25%**; TestB: **95.65% / 95.65%**.
16. New VGen2 component frozen: **YES**.
17. Is next-step O3a training allowed under this result? **YES, as the single next experiment only; allocator training was NOT RUN in this round.**
