# P0H2-v2 Results

## Status

**INCONCLUSIVE / NOT A CANDIDATE.** H2-v2 did not complete the planned 1,500 optimization steps, so no final H2-v2 checkpoint or P0H5 final comparison was produced. Per protocol, it must **not** enter 0I.

## Experimental setup actually exercised

- Canonical source: `D:\training`
- Base checkpoint: `checkpoints/ouro_retention_m2_d3/best_model.pt` (norm1800)
- Converted retention layers: `{3, 7, 11, 15, 19, 23}`
- Trainable parameters: rank-4 Q/K/V/O LoRA, 24 projections, 393,216 parameters
- Frozen: all other parameters, including retention gamma and post-norm weights
- LR: `1e-4`
- Objective: `KL + 0.3 * block + 0.1 * hinge-ranking`, margin `1.0`
- Ranking supervision: dataset ground-truth answer (`turn["answer"]`); teacher candidate ranking was not used
- Seed: `random/numpy/torch/CUDA = 42`
- Cache: `teacher_cache_v2`, 600 trajectories / 1,254 turns; held-out P0H5 was skipped
- Planned schedule: 1,500 steps, up to 3 deterministic epochs
- Independent monitor: 150 trajectories, seed `777`

## Execution result

The run started successfully and reached **step 30/1,500**, with `matched=30`, `skipped=0`, epoch 1. The run then terminated before producing a checkpoint. No final artifact exists under `checkpoints/ouro_retention_m2_h2v2_final`.

Observed training progress at step 30:

| quantity | value |
|---|---:|
| total loss | 0.5666906 |
| block loss | 0.8597106 |
| KL loss | 0.00005455 |
| ranking hinge loss | 3.0872288 |
| elapsed time | 1,366.15 s |

The observed bottleneck was the differentiable candidate-scoring forward: approximately 40–50 seconds per optimization step in this Windows environment. This is an execution blocker, not evidence that the objective succeeds or fails.

## Monitor baseline (independent validation set, seed 777)

The trainer's step-0 monitor was successfully written. It is **not** the canonical P0H5 held-out set and must not be substituted for it.

| step | candidate acc | mean rank | margin | logit cosine | ΔNLL | hidden cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0/150 | 9.6133 | -4.8350 | 0.8856 | +4.7396 | 0.5224 |

## Canonical P0H5 reference comparison

These are the previously established, same-batch P0H5 N=150 results. H2-v2 has no valid row because it did not reach a final checkpoint and was not evaluated on P0H5.

| model | candidate acc | mean rank | margin | logit cosine | ΔNLL |
|---|---:|---:|---:|---:|---:|
| M0 teacher | 11/150 | 4.47 | — | 1.000 | 0.00 |
| M1 migration | 15/150 | — | — | 0.749 | +14.52 |
| norm600 | 7/150 | — | — | 0.847 | +6.77 |
| norm1800 | 0/150 | 9.85 | -4.92 | 0.886 | +5.12 |
| H2-v2 | **not available** | **not available** | **not available** | **not available** | **not available** |

## Conclusion

No claim about ranking-loss efficacy can be made from this run. H2-v2 is retained as an **incomplete execution / negative availability result**, not as M2. The experiment stops here and does not automatically proceed to 0I.
## Completed H2-v2 run and canonical P0H5 evaluation

H2-v2 completed the planned 1,500 steps from norm1800. The final checkpoint is:
`checkpoints/ouro_retention_m2_h2v2_final_rerun/best_model.pt`.

| model | candidate accuracy | correct mean rank | correct-vs-best-wrong margin | entropy | Spearman vs M0 | logit cosine | ΔNLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 teacher | 11/150 | 4.4733 | -2.0516 | 3.8099 | 1.0000 | 1.0000 | 0.0000 |
| H2-v2 | 0/150 | 10.1000 | -4.2827 | 5.5348 | 0.7617 | 0.9950 | +3.5336 |

The independent monitor showed the same direction: at step 1500, candidate accuracy remained 0/150, mean rank 10.02, and margin -4.2207.

## Decision

H2-v2 is **not** an M2 candidate. Although teacher/logit fidelity became very high, candidate behavior did not recover: accuracy stayed at 0, mean rank worsened relative to M0, and the margin remained strongly negative. This confirms that the tested ranking objective/configuration did not solve the ranking failure. Do not enter 0I and do not expand LoRA based on this run.

Raw final evaluation: `P0H2_V2_P0H5_EVAL.json`.
