# CLR Architecture

**Base:** PLAA (H:\PLAA\)
**Core:** S_t ∈ ℝ²⁵⁶ via GRU → FiLM modulation of frozen LLM

### Two Injection Mechanisms

**Additive (PLAA v2):** hs = hs + gate·e_mem — absorbed by residual
**FiLM (PLAA Phase 3C):** hs = LN(hs)·(1+0.1·tanh(scale(S))) — 5.7× stronger

### Future Directions

- Hierarchical latent (persona/emotion/relation/memory)
- Read/write separation
- Consistency loss
- Late unfreeze of top layers
