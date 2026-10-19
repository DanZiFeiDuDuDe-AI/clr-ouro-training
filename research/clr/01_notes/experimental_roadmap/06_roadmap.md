# CLR Experimental Roadmap

1. **S_t trajectory analysis** — PCA/t-SNE, does S_t form attractors?
2. **Memory decay** — inject event, 50 turns later, check carryover
3. **Personality drift** — positive→depressing→recovery, measure smoothness
4. **State controllability** — S_t += α·v_emotion, steer persona
5. **Attention probe** — which tokens depend on latent?
6. **Latent interpolation** — S_mix = 0.3·S_happy + 0.7·S_sad
7. **Multi-conversation isolation** — same model, different pools

**Engineering:** CLR-Bench (continuity evaluation framework)
