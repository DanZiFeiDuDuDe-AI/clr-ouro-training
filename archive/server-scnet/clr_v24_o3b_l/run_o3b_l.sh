#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/.hyhal/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/root/private_data/clr_v24_o3b_l/code
export V24_O3BL_OUT=/root/private_data/clr_v24_o3b_l/output
export V24_O3BL_CACHE=/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache
export V24_O3BL_HOUT=/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_h_entity_carrier_diag
export V24_O3BL_O21B=/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt
export V24_O3BL_DEVICE=cuda
export V24_O3BL_PROBE_STEPS=180
export V24_O3BL_N_SEEDS=10
export V24_O3BL_N_PERM=20
export V24_O3BL_PAIR_N=1800
export V24_O3BL_LONGER_STEPS=500
export V24_O3BL_LONGER_SEEDS=5
export V24_O3BJ_CACHE=/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache
export V24_O3BJ_HOUT=/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_h_entity_carrier_diag
export V24_O3BJ_O21B=/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt
export V24_O3BJ_DEVICE=cuda
cd /root/private_data/clr_v24_o3b_l/code
exec /usr/local/bin/python3 -u /root/private_data/clr_v24_o3b_l/code/run_o3b_l_compat.py
