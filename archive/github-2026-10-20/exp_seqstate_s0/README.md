# exp_seqstate_s0 — Sequence-State S0 Screening

## Question

> With total latent capacity matched at 64 scalars, does replacing the single
> GRU latent vector (`1×64`) with a fixed-length latent sequence
> (`2×32`, `4×16`, `8×8`) change task performance or dynamical behavior?

This experiment deliberately does **not** implement Paper 2 / JEPA / halting /
variable-length latents. It only swaps the latent representation layout and
its (tiny) updater, keeping every other protocol term fixed.

## Protocol (all pinned to the v1 baseline)

| term | value |
|:--|:--|
| backbone | Ouro-1.4B, frozen, 4-bit (`load_ouro`), 24 shared layers |
| FiLM layers | {3, 7, 11, 15, 19, 23}, `h <- h·(1 + 0.1·tanh(W s))` |
| state update granularity | per UT loop (v1): after each of 4 loops |
| dataset | CLUTRR-style, 600 train / 150 eval trajectories, replayed RNG stream, seed 42 |
| steps / batch | 400 steps, batch-1 trajectories, `random.shuffle` per epoch |
| optimizer | AdamW lr=3e-4 wd=0.01, cosine, 20 warmup |
| loss | `0.95·CE + 0.05·‖Z_new − Z‖` (same mix as v1) |
| eval protocol | per-loop evolution, argmax next-token readout, with_S vs without_S (literal-zero modulation), t1/t2p/last/all |
| seed | 42 (screening, 1 seed) |

## Capacity / parameter accounting

| config | state module params | + 6×scale_proj (798,720) | total trainable |
|:--|:--:|:--:|:--:|
| vector 1×64 | 156,160 | 798,720 | **954,880** |
| sequence 2×32 | 148,192 | 798,720 | **946,912** |
| sequence 4×16 | 138,768 | 798,720 | **937,488** |
| sequence 8×8  | 136,360 | 798,720 | **935,080** |

All configs within ±2.1% of each other; the dominant FiLM readout term is
identical. Differences in later analysis can be attributed to representation
layout, not parameter count.

## Design decisions

1. **StateModule interface** (`state_modules.py`): `init_state / update /
   to_modulation`. `VectorStateModule` is `PlaaCore` verbatim (identity
   modulation), so existing v1 checkpoints load unchanged and vector runs
   reproduce the v1 numbers.
2. **Summary read is 2048→64, reshaped into K context tokens** — NOT
   2048→d. This keeps the upstream summary bandwidth identical across
   `K/d` configs and identical to the vector baseline; only the latent
   layout and the tiny updater differ.
3. **Updater**: 1 single-head pre-norm transformer block (no dropout,
   deterministic) over `[K context tokens + K slots]`, learned position
   embeddings (init N(0, 0.02)) so slots are not an unordered bag.
4. **Pooling**: flatten + learned linear `W_p: 64→64` (bias zero-init) by
   default; optional attention pooling exists but is not the S0 default.
5. **without_S parity**: `ablate_s` injects literal zeros in modulation
   space, bypassing `W_p` — byte-identical downstream to the vector
   protocol. The `zero_slots` ablation (which goes through `W_p`, i.e.
   includes its learned bias) is an additional diagnostic, not the
   protocol floor.
6. **`turn_forward_v1`** gained optional kwargs `state_module / max_ut /
   intervene / no_update / per_loop_logits`, all None-preserving. Existing
   callers (interventions, swap tests, halt) are unchanged; `verify_baseline.py`
   gates this: v1_s42 must reproduce 76/150 and the legacy path must equal
   the generic path bucket-for-bucket.

## Files

```
exp_seqstate_s0/
  state_modules.py     StateModule interface + Vector/Sequence modules
  train.py             S0 CLI trainer (--state-type, --num-slots, --slot-dim)
  evaluate.py          k-curve (k=1..4) + early readout + slot interventions
  dynamics.py          per-slot norms / cos matrices / effective rank
  verify_baseline.py   safety gate: v1_s42 repro + path equivalence
  run_all.sh           sequential driver (idempotent, safe to interrupt)
  runs/<tag>_<state>_<K>x<d>_s42/
    config.json best_model.pt results.json eval_log.json
    eval_s0.json dynamics_s0.json z_logs.pt
```

## Commands

```bash
# full screening (baseline gate -> train -> eval -> dynamics)
bash run_all.sh                 # log: run_all.log

# individual runs
python train.py --state-type vector   --num-slots 1 --slot-dim 64 --seed 42
python train.py --state-type sequence --num-slots 4 --slot-dim 16 --seed 42
python train.py --state-type sequence --num-slots 8 --slot-dim 8  --smoke

python evaluate.py --ckpt runs/s0_sequence_4x16_s42/best_model.pt
python dynamics.py --ckpt runs/s0_sequence_4x16_s42/best_model.pt
python verify_baseline.py
```

## Intervention semantics (evaluate.py, sequence configs)

All injection-only except `static_trained`: the true latent chain still
evolves; only the FiLM input is altered each loop.

| condition | operation |
|:--|:--|
| normal | — (positive control, must match k=4 with_S) |
| zero_slots | inject `W_p(0)` (includes learned readout bias) |
| random_slots | inject `W_p(N(0,1))`, fresh noise per loop |
| static_trained | freeze Z at the trained init (no evolution at all) |
| permute_i | inject `W_p(perm_i Z)` with 3 random permutations |
| single_slot_j | inject `W_p(Z ⊙ mask_j)` |
| dropout_p | inject `W_p(Z ⊙ Bernoulli(1−p))`, fresh mask per loop, 3 samples |

## Results (filled by run_all.sh)

TBD
