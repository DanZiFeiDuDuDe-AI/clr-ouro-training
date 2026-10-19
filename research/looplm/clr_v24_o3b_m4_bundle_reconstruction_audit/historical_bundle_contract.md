# O3b-M4 — Historical bundle contract (source-derived)

This is a static contract recovered from hash-matched `a1_v24_o3b.py` and its source dependencies. It does not certify missing learned weights or reproduce historical behavior.

## Bundle and execution order

1. Frozen feature cache supplies fact/query hidden vectors at 2048D. The pinned Ouro-1.4B source model is separately archived; the O3b `load_all` path itself consumes cache vectors.
2. O21B `KeyEncoder` maps fact H to L2-normalized 64D keys. O21A-MR `QueryEncoder` maps query H to L2-normalized 64D queries.
3. Initialize K `[B,4,64]`, V `[B,4,64]`, and occupancy `[B,4]` to zero. Initial NEW writes use the frozen learned O3a allocator.
4. The incoming fact goes through the key encoder and the baseline `UpdateRouter`. Action class `0=NEW`, `1=UPDATE`; UPDATE is selected when its softmax probability meets the threshold stored in the router checkpoint.
5. UPDATE chooses the highest occupied pair logit, preserves the existing K, and updates that slot's V through `content_read` + `shared_gru`. NEW chooses an unoccupied slot through O3a and writes both K and V.
6. Query read uses `dot(Q,K)/0.1`, argmax slot selection, then the selected physical-slot MRValueHeads decoder outputs 64 class logits.

## Learned state contract

| Component | Interface | State loaded from | Parameters | Current state |
|---|---|---|---:|---|
| O21B KeyEncoder | 2048 → 256 → GELU → 64 → L2 normalize | O21B `key` | 540,992 | exact checkpoint present |
| O21A-MR QueryEncoder | 2048 → 256 → GELU → 64 → L2 normalize | O21A-MR `query` | 540,992 | checkpoint missing |
| VGen2 OracleLocalCore | 2048D H + selected prior 64D V → `Linear(2048,64)` + `GRUCell(64,64)` | VGen2 `core` | 156,096 | checkpoint missing |
| VGen2 MRValueHeads | four independent `Linear(64,64)` heads | VGen2 `value_heads` | 16,640 | checkpoint missing |
| O3a K-aware allocator | concat(K64, occupancy4) → 128 → GELU → 4 | O3a `allocator` | 9,348 | checkpoint missing |
| O3b UpdateRouter | pair 257→128→1; action 69→96→2 | router `router` + threshold | 40,067 | checkpoint missing |

Linear/GELU/GRU modules have no running-stat buffers. L2 normalization, dot-product matching, zero memory initialization, occupancy masking, action/slot branching, and write control flow are deterministic. The learned write-cell parameters must not be confused with deterministic zero V initialization.

## File and run-identity caveats

- O3b source strictly loads `SUPERVISED['query']`, `VGEN2['core']`, `VGEN2['value_heads']`, `O3A['allocator']`, and `router_best['router']` plus its threshold.
- The available O21B checkpoint contains a same-shaped `query` field, but it is O21B provenance, not the frozen O21A-MR query; it is not a certified substitute.
- O3b source loads R2 once before replacing the core with VGen2's full `core`. The R2 checkpoint is absent; this is a loader/upstream training dependency. The exact VGen2 final core state would subsume it at serving time if that state were recovered.
- Only `a1_v24_vgen.py` source was found. It defaults to `clr_v24_vgen`; O3b expects `clr_v24_vgen2/best.pt`. No VGen2-specific source/config/log/checkpoint was found.
- O3b-L/K configs are diagnostic-only. The original O3b config, router threshold, best step, and baseline logs were not recovered.
