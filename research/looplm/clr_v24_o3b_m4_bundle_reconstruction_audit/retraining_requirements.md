# O3b-M4 — Retraining / recovery requirements (not executed)

**RETRAINING REQUIRED.** The historical integration bundle cannot be certified from source/config alone. M4 did not train or initialize substitute learned modules.

## Required target states and order

1. **Upstream R2 OracleLocalCore training state/config (recovery dependency).** The source-faithful O3b loader calls `load_r2_checkpoint` before loading VGen2. The final VGen2 checkpoint would contain the full core state and supersede the R2 values at serving time, but the original VGen2 training path uses R2 and the checkpoint is absent. The module source is known (`content_read: Linear(2048,64)`, `shared_gru: GRUCell(64,64)`); the original R2 training objective/run config was not recovered from the searched source tree. Recover the original R2 checkpoint/config if available; do not silently replace it with random initialization.
2. **O21A-MR QueryEncoder state.** O3b consumes `best_model.pt['query']`; architecture is `2048→256→GELU→64→L2`, 540,992 parameters. The exact O21A-MR checkpoint is missing. Source recipe: 3,000 default steps, AdamW LR 1e-3, weight decay 0.01, batch up to 128, gradient clip 1.0, retrieval CE + 0.5 alignment CE + 0.1 entity/query-entity CE; validation macro retrieval every 100 steps. The actual warm-start flag, best step, exact config overrides, and metrics are missing. O21B's shape-matched query must not be substituted.
3. **VGen2 Core + MRValueHeads.** Required payload keys are `core` (156,096 params) and `value_heads` (16,640 params). Available `a1_v24_vgen.py` documents T1-A0/T1-A1/T2/T3 stages, AdamW LR 3e-4, WD 0.01, 3,000 default steps, T2/T3 50% replay, and validation selection. But the O3b expected artifact is under `clr_v24_vgen2/best.pt`; only `a1_v24_vgen.py` with default output `clr_v24_vgen` was found. No VGen2-specific config, exact selected stage, metrics, or log survives. Its exact historical objective/config is therefore only partially recoverable.
4. **O3a K-aware allocator.** `Allocator(68→128→GELU→4)`, 9,348 params. Source recipe is recoverable: AdamW 3e-4, WD 0.01, clip 1.0, 3,000 default steps; loss is negative log probability mass over all empty slots, with 50% random legal occupancy initialization; validation macro valid-allocation rate chooses best. The supplied history says O3a PASS, but exact numeric anchor, run config/selected step, and checkpoint are absent.
5. **Original O3b UpdateRouter.** `pair:257→128→GELU→1` and `action:69→96→GELU→2`, 40,067 params. Source recipe is recoverable: AdamW 3e-4, WD 0.01, 3,000 default steps, 6,000 train / 1,000 validation examples, balanced NEW/UPDATE, action CE + target-slot CE + 0.5 occupied-pair BCE, clip 1.0; the UPDATE threshold is selected on validation and saved with the router. The exact historical router threshold, best step, actual environment overrides and router-specific metrics are missing. User-provided earlier O3b metrics are not accepted as this router's metric anchor.

## Sufficiency judgment

- **Can rerun a source-defined training recipe?** Mostly yes for O21A-MR, O3a, and the baseline router; only partly for VGen2.
- **Can reproduce the exact historical learned states/configuration without additional artifacts?** No. R2 initializer state/objective, O21A-MR warm-start/actual run record, VGen2 run identity/config, and original O3b/O3a selected state metadata are absent.
- **Historical metric anchors:** Only O3a has a qualitative PASS in the supplied project history; exact component-specific numeric reports for O21A-MR/VGen2/O3a/router were not found. Do not calibrate to unrelated O3b/J/K/L numbers.

## Gate

Do not run baseline reproduction or Candidate B reconnect until either (a) exact checkpoints are recovered and hash-certified, or (b) a separately authorized new baseline is trained under frozen, explicit configs with new acceptance anchors. This document is a requirement list only; no training was run.
