# CLR Training Pipeline

| Phase | Content | Model | Status |
|:------|:--------|:------|:------:|
| 1 | YaRN 256K | Qwen3-4B, LoRA r=16 | ✅ |
| 2 | SFT 134K | Qwen3-4B, LoRA r=32 | ✅ |
| 2.5 | Persona alignment | Qwen3-4B, LR=5e-6 epoch=1 | ✅ |
| 2.1 | GSM8K reasoning | Qwen3.6-35B-A3B 4-bit | ✅ |
| 3A | Stability validation | 1K steps | ✅ |
| 3B | Gradient loop + Probe | — | ✅ |
| 3C | Dynamics (additive/FiLM) | 500-5000 steps | 🔄 |
