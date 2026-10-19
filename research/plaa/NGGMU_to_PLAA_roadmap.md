# CLR — Continuous Latent Reasoning: Project Roadmap

> Formerly PLAA (Persistent Latent Affective Architecture)
> 2026-05-31 · 项目更名

---

## Completed Milestones

### ✅ Milestone 0: Rigid Attractor Discovery (v2)
| Status | Insight |
|:------:|:--------|
| ✅ | Additive injection is structurally weak: transformer residual absorbs perturbations |
| ✅ | ΔPPL=2.01 at noise std=±19, well within noise band |
| ✅ | Gate has real but minimal gradient (avg 0.056) with three-layer attenuation chain |
| ✅ | **Conclusion:** Additive injection architecture bottleneck confirmed |

### ✅ Milestone 1: Gate Writes to Residual (Phase 2.0)
| Status | Insight |
|:------:|:--------|
| ✅ | Gate_only: ΔPPL grows from 0.49 to 2.25 over 500 steps |
| ✅ | Gate_lora: ΔPPL grows from 0.10 to 1.17 (LoRA acts as shock absorber) |
| ✅ | MI proxy: 0.16 → 0.36 (gate_only), causal influence measurable |
| ✅ | **Conclusion:** Trainable gate mechanism verified |

### ✅ Milestone 2: S_t Carries Cross-turn State (Phase 2.1)
| Status | Insight |
|:------:|:--------|
| ✅ | GSM8K: S_t ablated → PPL +1.26, ACC -2.4% at step 500 |
| ✅ | Phase transition: harmful (-2.53) → beneficial (+1.26) |
| ✅ | Qwen3.6-35B-A3B, 500 steps, 4-bit |
| ✅ | **Conclusion:** S_t carries functional cross-turn reasoning state |

### ✅ Milestone 3: FiLM Multiplicative Control (Phase 3C)
| Status | Insight |
|:------:|:--------|
| ✅ | FiLM achieves Δ=11.44 PPL, 5.7× stronger than additive |
| ✅ | Structural differentiation: w_std 0.06→0.40 (6.6×) |
| ⚠️ | But structural ≠ functional: controlled ablation Δ=0.044 |
| ✅ | **Conclusion:** Multiplicative modulation enables control, but static rather than temporal |

## Next Steps (Proposed)

1. **Phase 3C FiLM v5 completion** (~2,500 steps)
2. **Phase 2.1 evaluation on server logs** — check all 5 fixed eval points
3. **Experiment 1: State trajectory analysis** — does S_t form attractors?
4. **Experiment 2: Memory decay** — how long does S_t carry state?
5. **CLR-Bench** — build systematic continuity evaluation

## Story Line for Papers

1. **Paper 1 (Framework):** "Continuous Latent Reasoning" — the CLR architecture, additive injection, embedding projection, and training pipeline
2. **Paper 2 (Multiplicative):** FiLM modulation, structural specialization, the additive→multiplicative transition
3. **GSM8K Verification (Supplementary):** S_t carries cross-turn cognitive state — causal evidence
