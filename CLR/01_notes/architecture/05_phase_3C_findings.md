# CLR Phase 3C Findings

## Three Theoretical Regimes (FiLM)
1. **Violent Exploration:** |S|>5, dS>5 — discovery
2. **Convergent Modulation:** |S|~1-2, dS~1 — selective control
3. **Structural Specialization:** w_std>0.05 — attractors (not achieved)

## Key: Structural ≠ Functional
- w_std grew 6.6× (0.06→0.40) ✓
- But controlled ablation: Δ=0.044 PPL ✗
- Layers differentiate in weight space, not function space

## FiLM Version History
| v | Change | Result |
|:-:|:-------|:-------|
| 1 | AdaLN ±0.1 | Baseline |
| 2 | +tanh bound | No divergence |
| 3 | +w_std diff loss | ✅ Divergence |
| 5 | +grad cosine | w_std=0.033, cos=0.11-0.24 |
