#!/usr/bin/env bash
# exp_seqstate_s0 / run_all.sh — sequential S0 screening driver.
# Safe to interrupt: every phase writes JSON/checkpoints and skips when done.
# Usage: bash run_all.sh [--smoke] [--no-eval] [--no-dynamics]
set -u
cd "$(dirname "$0")"
LOG=run_all.log
exec > >(tee -a "$LOG") 2>&1

SMOKE=0; DO_EVAL=1; DO_DYN=1
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --no-eval) DO_EVAL=0 ;;
    --no-dynamics) DO_DYN=0 ;;
  esac
done

PY="py -3.13 -u"   # experiment env: transformers 5.6.2 + torch 2.6.0 + bnb/accelerate
CONFIGS=("vector 1 64" "sequence 2 32" "sequence 4 16" "sequence 8 8")
if [ "$SMOKE" = "1" ]; then TAG=s0smoke; EXTRA="--smoke"; else TAG=s0; EXTRA=""; fi
phase() { echo; echo "==================== [$1] ===================="; }

phase "0. baseline safety gate (v1_s42 repro + path equivalence)"
if [ -f verify_baseline.passed ]; then
  echo "skip verify (marker present: $(cat verify_baseline.passed))"
else
  $PY verify_baseline.py || { echo "!! BASELINE CHECK FAILED — aborting"; exit 1; }
fi

phase "1. train (4 configs, $TAG)"
for c in "${CONFIGS[@]}"; do
  set -- $c
  dir="runs/${TAG}_${1}_${2}x${3}_s42"
  if [ -f "$dir/results.json" ] && [ -f "$dir/best_model.pt" ]; then
    echo "skip $dir (already done)"
  else
    $PY train.py --state-type "$1" --num-slots "$2" --slot-dim "$3" \
                 --seed 42 --tag "$TAG" $EXTRA
  fi
done

phase "2. evaluate (k-curve + early readout; ablations only for best seq config)"
for c in "${CONFIGS[@]}"; do
  set -- $c
  dir="runs/${TAG}_${1}_${2}x${3}_s42"
  [ "$DO_EVAL" = "1" ] || { echo "skip eval (--no-eval)"; break; }
  if [ -f "$dir/eval_s0.json" ]; then
    echo "skip eval $dir (already done)"
  else
    $PY evaluate.py --ckpt "$dir/best_model.pt"
  fi
done

phase "3. dynamics (per-slot diagnostics for every config)"
for c in "${CONFIGS[@]}"; do
  set -- $c
  dir="runs/${TAG}_${1}_${2}x${3}_s42"
  [ "$DO_DYN" = "1" ] || { echo "skip dynamics (--no-dynamics)"; break; }
  if [ -f "$dir/dynamics_s0.json" ]; then
    echo "skip dynamics $dir (already done)"
  else
    $PY dynamics.py --ckpt "$dir/best_model.pt"
  fi
done

phase "DONE"
echo "results:"
ls -1 runs/*/results.json 2>/dev/null
