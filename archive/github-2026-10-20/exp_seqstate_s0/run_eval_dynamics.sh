#!/usr/bin/env bash
# exp_seqstate_s0 / run_eval_dynamics.sh — phase 2: full evaluation suite.
# Runs sequentially on the single GTX 1650 (no parallelism). Each stage saves
# incrementally (eval_s0.json is rewritten after every condition), so the
# whole chain is safely interruptible per GPU_COORDINATION.md convention.
#
# Order: vector first (fastest, reference), then sequence configs by slot count.
set -u
cd "$(dirname "$0")"
PY="py -3.13"

echo "==== EVAL vector 1x64 ===="
$PY -u evaluate.py --ckpt runs/s0smoke_vector_1x64_s42/best_model.pt 2>&1 \
  | grep -vE "Loading weights|it/s\]|deprecated|FutureWarning|warnings.warn|check_model_inputs|cache_position|layer_type_validation|rope_config"

echo "==== EVAL sequence 2x32 ===="
$PY -u evaluate.py --ckpt runs/s0smoke_sequence_2x32_s42/best_model.pt 2>&1 \
  | grep -vE "Loading weights|it/s\]|deprecated|FutureWarning|warnings.warn|check_model_inputs|cache_position|layer_type_validation|rope_config"

echo "==== EVAL sequence 4x16 ===="
$PY -u evaluate.py --ckpt runs/s0smoke_sequence_4x16_s42/best_model.pt 2>&1 \
  | grep -vE "Loading weights|it/s\]|deprecated|FutureWarning|warnings.warn|check_model_inputs|cache_position|layer_type_validation|rope_config"

echo "==== EVAL sequence 8x8 ===="
$PY -u evaluate.py --ckpt runs/s0smoke_sequence_8x8_s42/best_model.pt 2>&1 \
  | grep -vE "Loading weights|it/s\]|deprecated|FutureWarning|warnings.warn|check_model_inputs|cache_position|layer_type_validation|rope_config"

# ---- dynamics (forward-only, ~2 min each) ----
for d in s0smoke_vector_1x64_s42 s0smoke_sequence_2x32_s42 \
         s0smoke_sequence_4x16_s42 s0smoke_sequence_8x8_s42; do
  echo "==== DYNAMICS $d ===="
  $PY -u dynamics.py --ckpt "runs/$d/best_model.pt" > "runs/$d/dynamics_stdout.log" 2>&1
  echo "  -> runs/$d/dynamics_s0.json"
done

echo "ALL EVAL+DYNAMICS DONE"
