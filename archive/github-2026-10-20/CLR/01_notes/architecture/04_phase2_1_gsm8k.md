# CLR Phase 2.1 — GSM8K Results

**Date:** 2026-05-31
**Model:** Qwen3.6-35B-A3B (4-bit)
**Steps:** 500

| Step | ΔPPL | ΔACC | Trend |
|:----:|:----:|:----:|:------|
| 100 | -2.53 | -2.1% | Harmful |
| 200 | -2.05 | -1.9% | Improving |
| 300 | -0.68 | -0.4% | Near zero |
| 400 | +0.53 | +1.8% | **Flip!** |
| 500 | +1.26 | +2.4% | **Positive** |

**Conclusion:** S_t carries cross-turn reasoning state. Ablating S_t at step 500 → PPL+1.26, ACC-2.4%.

**Fixes applied vs original:**
1. Metric: attention_mask index (not -1)
2. Metric: full-answer PPL + token acc (not argmax)
3. Optimizer: FiLM scale_proj included
4. device_map: cuda:0 (not auto)
5. Eval: FiLM _s activated

**Files:** 02_results/phase_2_1_gsm8k/
