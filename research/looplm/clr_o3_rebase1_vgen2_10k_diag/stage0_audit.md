# Stage 0 — Exact Rebuild Audit

Status: PASS. The original 3K config and final source script hashes match the archived local/remote artifacts.

Training configuration reused exactly:
- Architecture: OracleLocalCore: Linear(2048,64) + GRUCell(64,64), plus four Linear(64,64) slot decoders
- Seed / initialization: 42 / fresh random initialization; no missing R2/VGen state substituted
- Optimizer: AdamW, lr=0.0003, weight_decay=0.01, gradient clip=1.0
- Batch size: 64; same 50/50 episode/update mixture; same losses and raw frozen features.
- Source config SHA-256: `5be7a10274df9b1ae7689f71cc1bb3726fd03cdf605303988b06fa434b576b8b`
- Source training script SHA-256: `179a9473810ad62bc2e890b6620f0e383970e0c0a0ace2809bc0063c27e545be`
- Frozen O21A checkpoint SHA-256 verified: `dadf66a59c8722ef86604aff67528f415acb18f3e72ba4ccfac415df37e90ced`
- Only horizon changed from 3,000 to 10,000 total steps. Selection ordering and fixed validation slot permutations follow this round's explicit requirements; neither changes training updates.
- TestA/TestB caches were not opened during Stage 0.

Input join and source separation:
- Train: 5000 episodes / 11665 facts; update pairs=27955.
- Validation: 1000 episodes / 2332 facts; update pairs=1118.
- Canonical fact overlap: 0; episode overlap: not exposed in NPZ schema.
- MR `fact_H` equals certified J3C `H_base_I0` bitwise per split; sidecar and label IDs match rowwise.
