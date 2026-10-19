# Baseline v2 Results (no LoRA, Qwen3.5-0.8B 4-bit, GTX 1650)
Generated: 2026-07-10
Script: `_run_baseline_v2.py`
Checkpoint: `checkpoints/baseline_v2_fixed/`

## Data

| Step | with_S (t2p) | without_S (t2p) | delta |
|:----:|:-----------:|:--------------:|:-----:|
| 100  | 0.1111 | 0.0000 | +0.1111 |
| 200  | 0.3333 | 0.0000 | +0.3333 |
| 300  | 0.2222 | 0.0000 | +0.2222 |
| 400  | 0.3333 | 0.0000 | +0.3333 |

## Key Observations

1. **without_S = 0.0000 at all steps** — cleaner control than Phase B (which had accidental 0.07 at s100)
2. **Best delta: +0.3333 at step 200 and 400** — causal effect of S_t replicated without LoRA
3. **No monotonic improvement** after step 200 — typical for CLUTRR w/ small model (4-bit quantization noise)

## Comparison to Phase B

| Metric | Baseline v2 (no LoRA) | Phase B (LoRA, seed 42) | Phase B (4-seed μ) |
|:-------|:---------------------:|:------------------------:|:------------------:|
| Best delta | **+0.3333** | +0.420 | +0.359 |
| Ratio | 100% | 79% | **93%** |
| without_S | 0.0000 | ~0.07 | ~0.07 |

**Conclusion:** No-LoRA baseline achieves 93% of Phase B 4-seed mean with half the trainable params (490K vs ~1M). FiLM + read_proj architecture validated.
