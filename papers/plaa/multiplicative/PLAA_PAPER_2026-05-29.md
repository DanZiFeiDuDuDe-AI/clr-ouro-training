# PLAA: Persistent Latent Affective Architecture
## Continuous State Dynamics for Emotion-Conditioned Language Generation

> 论文草稿 · 2026-05-29 · 含实验数据 + PLAA-Bench

---

## Abstract

Emotion in dialogue is not a categorical annotation but a temporally evolving latent trajectory. We present PLAA, a persistent latent affective architecture that models this trajectory within a pretrained LLM (Qwen3-4B). PLAA maintains a learnable latent state S_t ∈ ℝ^256 that evolves via a GRU-style transition function conditioned on the transformer's hidden states. S_t is projected into emotion memory tokens e_mem ∈ ℝ^{16×2560} and injected into the LLM's residual stream via gated cross-attention (layers 16-28). We verify full gradient flow through all cross-attention projections. Trajectory experiments show emotion-specific attractors (PCA 22-30%, vs 11.5% random), curvature convergence (Sad: 0.48→0.08), and cross-seed robustness. With neutral input, different initial S_0 produce divergent trajectories—**state-driven behavioral emergence** independent of input semantics. We introduce PLAA-Bench: a standardized benchmark for evaluating continuous latent state dynamics in language models.

---

## 1. Introduction

### 1.1 The Problem
LLMs lack emotional continuity—each turn is an effective emotional reset.

### 1.2 Contributions
1. **Persistent latent state injection into Qwen3-4B** (layers 16-28, gated cross-attention)
2. **Full gradient flow verification** through all cross-attention projections
3. **Empirical demonstration of continuous state dynamics**: emotion-specific attractors, curvature convergence, state-driven emergence
4. **PLAA-Bench**: standardized benchmark for state-conditioned language behavior

---

## 2. Method

### 2.1 System Architecture

```
Qwen3-4B Base (4-bit, SDPA)
  ├── Adapters: phase1 + phase2 + persona (separate, no merge)
  └── PlaaInjectedLayer [16..28]
        ├── base_layer (original Qwen decoder)
        ├── EmotionCrossAttention (Q=hidden, K,V=e_mem)
        └── Gated Residual: hidden += tanh(α) × attn_out
```

### 2.2 State Pipeline
S₀ → each turn: e_mem = π(S_t) → inject → forward → h_t → S_{t+1} = GRU(S_t, h_t)

### 2.3 Critical Implementation Detail
Injection BEFORE PEFT wrapping. Post-wrapping path: `peft.base_model.model.model.layers[i]`

---

## 3. Experiments

### 3.1 Setup
- Qwen3-4B (4-bit, SDPA), RTX PRO 6000 Blackwell, frozen weights

### 3.2 Trajectory Clustering

| Emotion | Terminal | PCA | Curvature (early→late) | Energy |
|---------|:--------:|:---:|:----------------------:|:------:|
| Happy | (-0.79, +1.41) | 22.3% | 0.64→0.23 ↓ | +12 ↑ |
| Sad | (-3.70, -2.68) | **30.5%** | **0.48→0.08** ↓↓ | +13 ↑ |
| Angry | (-1.62, -1.51) | 17.5% | 0.20→0.34 ↑ | +14 ↑ |
| Calm | (+0.30, +1.90) | **30.4%** | 0.19→0.34 ↑ | +10 ↑ |

### 3.3 Neutral Trigger (Critical)
Constant "嗯" input, different S₀ → **different trajectories**:
- S₀=+0.5 → stable (87→91), terminal (+3.26, +1.88)
- S₀=-0.5 → stable (82→86), terminal (-2.23, -1.75)
- S₀=0 → growing (71→94), terminal (-4.54, +4.56)

### 3.4 Ablation Summary
| Ablation | Finding |
|----------|---------|
| No GRU | State variance=0 — trajectory requires state evolution |
| Random hidden | PCA 11.5% vs 22-30% real — 2-3× signal from real states |
| 5 seeds | Sad converges to x=-1.00 across ALL seeds |
| Alpha sweep | Single decay CANNOT produce stable attractors |

---

## 4. PLAA-Bench: Standardized Evaluation

### 4.1 Core Goal
Evaluate: controllable, separable, stable behavioral trajectories under latent state conditioning.

### 4.2 Tasks
1. **Static Probe**: Fixed prompt + different S₀ → single-turn output separation
2. **Trajectory Drift**: Fixed prompt ("嗯"), 30-50 turns, S₀ ∈ emotion set
3. **Perturbation Sensitivity**: S₀ + ε → smooth or abrupt output change

### 4.3 Metrics
| Metric | Formula | Purpose |
|--------|---------|---------|
| Separation | Var_between / Var_within | Cluster distinctness |
| Drift Index | Σ||h_t − h_{t-1}|| | Dynamic evolution |
| Controllability | Acc(C(output)→S₀) | S₀ predicts output |
| Stability | 1 − Var(outputs) | Consistency |
| Expression Entropy | H(lexicon) + H(topics) | Mode collapse |

### 4.4 Baselines
Vanilla LLM / Prompt Persona / PLAA

---

## 5. Related Work

- Affective Computing in Dialogue: OCC, PAD, EmpatheticDialogues — emotions as labels, not trajectories
- Persistent Persona: Character.AI, SillyTavern, GPT memory — text-level, no latent dynamics
- Latent State Dynamics: Deep Kalman Filters, VRNN, Latent ODE — not dialogue-specific
- Gated Cross-Attention: Flamingo, NGGMU — PLAA repurposes for emotion injection

---

## 6. Conclusion

We demonstrate that a persistent latent state injected into Qwen3-4B's residual stream creates emotion-conditioned token prediction dynamics. Trajectories cluster by emotion, converge under attractor dynamics, and drive behavior independently of input semantics. PLAA-Bench provides a standardized evaluation framework for continuous latent state dynamics in language models.

> Emotion is not a categorical annotation. It is a temporally evolving latent trajectory.

---

## Appendix: Training Pipeline

| Phase | Content | Status |
|-------|---------|:------:|
| 1 | YaRN 256K, r=16 | ✅ |
| 2 | SFT (UltraChat+OASST1+Dolly) | ✅ |
| 2.5 | Persona re-alignment (lr=5e-6) | ✅ |
| 3A | Stability (1K steps, loss 1.2→0.05) | ✅ |
| 3B | Gradient flow verification | ✅ |
| 3C | Full PLAA training | 🔜 |

All adapters separate. No merge. Inference routing per mode.
